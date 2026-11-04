# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Full-parameter SFT of Muse Spark Open 816B-A42B on one TPU7x 4x4x4 slice.

Three things make full-parameter SFT valid on 64 chips, where fp32 master
weights and moments (16 bytes per parameter, about 13 TB) do not fit:

1. An optimizer that keeps sub-ulp updates. At learning rate 1e-6 most AdamW
   updates of bf16 weights are smaller than half a bf16 ulp, and
   ``torch.optim.AdamW`` rounds them away. ``LowPrecisionAdamW``
   (``../low_precision_adamw.py``) adds the rounding residual back at the next
   step (Kahan) or rounds stochastically, so the weights follow fp32 AdamW at
   bf16 memory cost.
2. fp32 gradient reduction. A bf16 reduce-scatter over 128 ranks adds
   percent-level gradient error, so the recipes keep
   ``training.mixed_precision_reduce="float32"`` (``SPARK_GRAD_REDUCE``). The
   expert gradients alone can be reduced in bf16 over their FSDP ranks with
   ``SPARK_EXPERT_GRAD_REDUCE=bfloat16``, which saves HBM; that needs the
   model config's ``ep_grad_reduce_dtype``.
3. Expert parallelism 8 with a measured placement. The released weights route
   skewed, so the expert placement is measured on the same SFT rows first
   (``muse_spark_816b_a42b_ctr_sft_measure_placement_tpu7x``).

Registered recipes (``--module torchtitan.experiments.tpu.post_training.full_param_sft``),
with bytes per parameter for weights + gradients + optimizer states:

* ``muse_spark_816b_a42b_ctr_sft_full_tpu7x``: all weights in bf16,
  ``LowPrecisionAdamW`` with a Kahan residual and float8 moments (8 bytes).
* ``..._ctr_sft_full_sr8_tpu7x`` (alias ``..._full_adamw8bit_tpu7x``):
  stochastic rounding and float8 moments (6 bytes).
* ``..._ctr_sft_full_bf16sr_tpu7x``: stochastic rounding and bf16 moments
  (8 bytes).
* ``..._ctr_sft_full_fp32_tpu7x``: fp32 master weights and fp32 AdamW
  (16 bytes), the reference; a 4x4x8 slice or larger.
* ``..._ctr_sft_full_bf16_tpu7x``: ``torch.optim.AdamW`` on bf16 weights
  (round to nearest); not valid at small learning rates, comparisons only.
* ``..._ctr_sft_measure_placement_tpu7x``: LoRA at expert parallelism 1 for 3
  steps; writes the EP-8 placement measured on the SFT data.
* ``..._ctr_sft_lora_ep8_tpu7x``: LoRA on the full-parameter layout (AdamW on
  the adapters), for side-by-side comparisons.

Launch (placeholders in angle brackets)::

    # 1. measure the routing on the SFT data (EP 1, LoRA, 3 steps) and write the placement
    SPARK_EXPERT_PLACEMENT_OUT=<OUT_DIR>/placement_ep8.json torchrun ... \\
      -m torchtitan.experiments.tpu.post_training.sft_train \\
      --module torchtitan.experiments.tpu.post_training.full_param_sft \\
      --config muse_spark_816b_a42b_ctr_sft_measure_placement_tpu7x \\
      --hf_assets_path <MODEL_DIR> --dataloader.data-path <SFT_JSONL> --dump_folder <OUT_DIR>
    # 2. full-parameter SFT with that placement
    SPARK_EXPERT_PLACEMENT_FILE=<OUT_DIR>/placement_ep8.json SPARK_LR=1e-6 torchrun ... \\
      -m torchtitan.experiments.tpu.post_training.sft_train \\
      --module torchtitan.experiments.tpu.post_training.full_param_sft \\
      --config muse_spark_816b_a42b_ctr_sft_full_tpu7x --training.steps=30 \\
      --hf_assets_path <MODEL_DIR> --dataloader.data-path <SFT_JSONL> --dump_folder <OUT_DIR>

Knobs (environment): ``SPARK_OPTIMIZER`` (a key of ``SPARK_FULL_OPTIMIZERS``:
``kahan8``, ``sr8``, ``sr``, ``kahan``, ``fp32``, ``rne``), ``SPARK_LR``
(1e-6), ``SPARK_WARMUP`` (2 steps), ``SPARK_LR_MIN`` (final factor, 1.0 =
constant), ``SPARK_EP`` (8), ``SPARK_EP_CAPACITY`` (receive buffer, 2.0),
``SPARK_EXPERT_PLACEMENT_FILE`` (required for EP > 1), ``SPARK_SEQ`` (row
length, 8,192; 4,096 keeps 1M tokens per step on the 256 ranks of a 4x4x8
slice), ``SPARK_GRAD_REDUCE`` / ``SPARK_EXPERT_GRAD_REDUCE``, and
``SPARK_UPDATE_PROBE=<n>`` (``SPARK_UPDATE_PROBE_PARAMS``: parameter-name
regex), which samples ``n`` elements of each probed tensor and prints per step
the fraction of fp32 AdamW's update the run keeps (``../update_probe.py``).
The flavor comes from ``SPARK_FLAVOR`` as in ``config_registry``. The
full-parameter recipes only load the checkpoint: a full-parameter save writes
the weights plus the optimizer state (several TB for the full model); add
``--checkpoint.load_only=false --checkpoint.interval=N`` to save.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.experiments.tpu.low_precision_adamw import low_precision_adamw_config
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig
from torchtitan.experiments.tpu.update_probe import install_update_probe

from .config_registry import (
    _base_config,
    _spark,
    _spark_parallel,
    apply_model_hooks,
    BF16,
    MODEL_ASSETS,
    Precision,
    resolve_model_spec,
    SFT_DATA,
    SFT_FULL_SHAPE,
    SFT_LORA_SHAPE,
    spark_tpu_settings,
    SPARK_LORA,
    TrainShape,
)
from .lora import apply_lora, LoRAOptions
from .sft_config import as_sft_config, SFTConfig, SFTTrainerConfig
from .sft_data import ChatSFTDataLoader

REDUCE_DTYPES = ("bfloat16", "float32")


@dataclasses.dataclass(frozen=True)
class FullParamOptimizer:
    """Weight dtype and optimizer of a full-parameter run.

    ``param_update`` "" selects ``torch.optim.AdamW``; "kahan" or "stochastic"
    selects ``LowPrecisionAdamW`` with ``state_dtype`` moments.
    """

    dtype: str
    param_update: str = ""
    state_dtype: str = ""


# Optimizer choices (SPARK_OPTIMIZER overrides a recipe's): LowPrecisionAdamW
# with a Kahan residual or stochastic rounding keeps the sub-ulp updates that
# torch.optim.AdamW rounds away on bf16 weights; fp32 is the reference (fp32
# master weights and moments, 16 bytes per parameter: 4x4x8 or larger).
SPARK_FULL_OPTIMIZERS = {
    "kahan8": FullParamOptimizer("bfloat16", "kahan", "float8"),
    "sr8": FullParamOptimizer("bfloat16", "stochastic", "float8"),
    "sr": FullParamOptimizer("bfloat16", "stochastic", "bfloat16"),
    "kahan": FullParamOptimizer("bfloat16", "kahan", "bfloat16"),
    "rne": FullParamOptimizer("bfloat16"),
    "fp32": FullParamOptimizer("float32"),
}
SPARK_FULL_SFT_LR = 1e-6


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name, "")
    return float(value) if value else default


def _reduce_dtype(value: str, what: str) -> str:
    if value not in REDUCE_DTYPES:
        raise ValueError(f"{what} must be one of {REDUCE_DTYPES}, got {value!r}")
    return value


def spark_fast_checkpoint_reader(spec: Any) -> Any:
    """Load the released checkpoint with the step recipes' parallel positional
    reader (one read per contiguous FSDP / expert slice, expert-placement aware:
    about a minute for the full model on one 4x4x4 slice instead of ~30 minutes
    through the stock reader on Cloud Storage FUSE). Importing the step recipes
    also turns on their per-step HBM log line (``[SPMEM]``) and frees the
    checkpointer's cached state dict after the initial load."""
    from torchtitan.experiments.tpu.muse_spark.spark_64chip_recipes import FastSparkAdapter

    spec.state_dict_adapter = FastSparkAdapter
    return spec


def spark_expert_parallel(config: Any, *, degree: int, capacity_factor: float,
                          expert_grad_reduce: str = "") -> Any:
    """Expert parallelism with the all-gather dispatcher (BF16 combine).

    With the released weights the routing is skewed, so EP > 1 needs an expert
    placement measured on the training text (``SPARK_EXPERT_PLACEMENT_FILE``,
    written by ``muse_spark_816b_a42b_ctr_sft_measure_placement_tpu7x``); a
    route that overflows the receive buffer (``capacity_factor`` x the rank's
    own routes) is dropped, and the per-step ``[EP]`` monitor line reports it.
    The gradient norm runs eagerly (dense and expert gradients live on
    different meshes). ``expert_grad_reduce`` overrides the expert gradients'
    reduce-scatter dtype (over the ``efsdp`` ranks only); it needs the model
    config's ``ep_grad_reduce_dtype`` field.
    """
    if degree <= 1:
        return config
    from torchtitan.experiments.tpu.muse_spark import expert_placement

    model = config.model_spec.model
    config.parallelism.expert_parallel_degree = degree
    config.compile.compile_grad_norm = False
    model.ep_dispatcher = "allgather_bf16"
    model.ep_capacity_factor = capacity_factor
    placement = os.environ.get("SPARK_EXPERT_PLACEMENT_FILE", "")
    if placement:
        model.ep_expert_placement = "file:" + placement
    elif os.environ.get("SPARK_EXPERT_PLACEMENT", "") in ("contiguous", "interleaved"):
        model.ep_expert_placement = os.environ["SPARK_EXPERT_PLACEMENT"]
    else:
        raise ValueError(
            "expert parallelism with the released weights needs a measured placement: set "
            "SPARK_EXPERT_PLACEMENT_FILE (muse_spark_816b_a42b_ctr_sft_measure_placement_tpu7x "
            "writes one), or SPARK_EXPERT_PLACEMENT=contiguous for random weights")
    if expert_grad_reduce:
        _reduce_dtype(expert_grad_reduce, "SPARK_EXPERT_GRAD_REDUCE")
        if "ep_grad_reduce_dtype" not in {f.name for f in dataclasses.fields(model)}:
            raise ValueError(
                "SPARK_EXPERT_GRAD_REDUCE needs a model config with ep_grad_reduce_dtype "
                f"({type(model).__qualname__} has none)")
        model.ep_grad_reduce_dtype = expert_grad_reduce
    expert_placement.install_ep_monitor()
    return config


def _spark_sft_base(*, shape: TrainShape, precision: Precision,
                    lora: LoRAOptions | None) -> TPUTrainerConfig:
    """Shared base of the 64-chip Spark SFT recipes below: the SFT view
    (``ChatSFTDataLoader``, assistant-only loss, packed 8,192-token rows, one
    row per rank and step), the fast reader, no checkpoint saves."""
    model = _spark()
    spec = spark_fast_checkpoint_reader(
        resolve_model_spec(model.name, model.flavor, shape.seq_len, **dict(model.kwargs))
    )
    if lora is not None:
        apply_lora(spec, lora)
    config = _base_config(
        model_spec=spec,
        hf_assets_path=MODEL_ASSETS,
        loss=ChunkedLossWrapper.Config(
            num_chunks=shape.num_chunks, loss_fn=CrossEntropyLoss.Config()
        ),
        dataloader=ChatSFTDataLoader.Config(
            data_path=SFT_DATA, reasoning_effort="low", max_documents_per_row=16,
            overlong="drop", microbatches_per_step=1,
        ),
        shape=shape,
        precision=precision,
        parallel=dataclasses.replace(_spark_parallel(), expert_parallel=1),
        num_tokens_per_train_step=-1,
        checkpoint_interval=10**9,
        seed=42,
    )
    apply_model_hooks(model, config)
    config.checkpoint.load_only = True
    # 128 ranks each load the tokenizer and the SFT records before the first collective.
    config.comm.init_timeout_seconds = 1800
    return config


def _spark_sft_full(optimizer: str) -> SFTTrainerConfig:
    """Full-parameter CTR SFT on the full model, one 4x4x4 slice.

    Optimizer ``SPARK_OPTIMIZER`` (default ``optimizer``, a key of
    ``SPARK_FULL_OPTIMIZERS``); fp32 gradient reduction (``SPARK_GRAD_REDUCE``;
    ``SPARK_EXPERT_GRAD_REDUCE`` for the expert gradients alone); expert
    parallelism ``SPARK_EP`` (default 8) with a measured placement and a receive
    buffer of ``SPARK_EP_CAPACITY`` (default 2.0); learning rate ``SPARK_LR``
    (default 1e-6) after ``SPARK_WARMUP`` linear warmup steps (default 2),
    decaying to ``SPARK_LR_MIN`` x (default 1.0 = constant); one packed row of
    ``SPARK_SEQ`` tokens per rank (default 8,192: 1,048,576 tokens per step on
    128 ranks); full activation checkpointing; Pallas fused cross-entropy; the
    fast placement-aware checkpoint reader; load-only checkpointing (see the
    module docstring); ``SPARK_UPDATE_PROBE=<n>`` reports the fraction of fp32
    AdamW's update that the run keeps.
    """
    name = os.environ.get("SPARK_OPTIMIZER", "") or optimizer
    if name not in SPARK_FULL_OPTIMIZERS:
        raise ValueError(f"SPARK_OPTIMIZER={name!r}; expected one of {sorted(SPARK_FULL_OPTIMIZERS)}")
    choice = SPARK_FULL_OPTIMIZERS[name]
    grad_reduce = _reduce_dtype(os.environ.get("SPARK_GRAD_REDUCE", "") or "float32",
                                "SPARK_GRAD_REDUCE")
    shape = dataclasses.replace(
        SFT_FULL_SHAPE,
        seq_len=int(os.environ.get("SPARK_SEQ", "8192")),
        lr=_env_float("SPARK_LR", SPARK_FULL_SFT_LR),
        warmup_steps=int(os.environ.get("SPARK_WARMUP", "2")),
        lr_min_factor=_env_float("SPARK_LR_MIN", 1.0),
    )
    config = as_sft_config(
        _spark_sft_base(shape=shape, precision=Precision(dtype=choice.dtype, optimizer="AdamW"),
                        lora=None),
        SFTConfig(lora_save="full"),
    )
    if choice.param_update:
        config.optimizer = low_precision_adamw_config(
            lr=shape.lr,
            betas=shape.betas,
            weight_decay=shape.weight_decay,
            param_update=choice.param_update,
            state_dtype=choice.state_dtype,
        )
    spark_tpu_settings(config, full_param=True, fused_ce=True, frozen_head=False)
    # spark_tpu_settings may pick a bf16 reduce for full-parameter runs; these
    # recipes set the reduce dtype explicitly (fp32 unless SPARK_GRAD_REDUCE).
    config.training.mixed_precision_reduce = grad_reduce
    spark_expert_parallel(
        config,
        degree=int(os.environ.get("SPARK_EP", "8")),
        capacity_factor=_env_float("SPARK_EP_CAPACITY", 2.0),
        expert_grad_reduce=os.environ.get("SPARK_EXPERT_GRAD_REDUCE", ""),
    )
    install_update_probe(
        int(os.environ.get("SPARK_UPDATE_PROBE", "0") or 0),
        os.environ.get("SPARK_UPDATE_PROBE_PARAMS", "") or None,
    )
    return config


def muse_spark_816b_a42b_ctr_sft_full_tpu7x() -> SFTTrainerConfig:
    """Full-parameter CTR SFT, the default: LowPrecisionAdamW with a Kahan
    residual and float8 moments (8 bytes per parameter), fp32 gradient
    reduction, EP 8 with a measured placement."""
    return _spark_sft_full("kahan8")


def muse_spark_816b_a42b_ctr_sft_full_sr8_tpu7x() -> SFTTrainerConfig:
    """Full-parameter CTR SFT with stochastic rounding and float8 moments
    (6 bytes per parameter: more HBM headroom than the Kahan default)."""
    return _spark_sft_full("sr8")


def muse_spark_816b_a42b_ctr_sft_full_fp32_tpu7x() -> SFTTrainerConfig:
    """Full-parameter CTR SFT reference with fp32 master weights and fp32 AdamW
    (16 bytes per parameter: a 4x4x8 slice or larger)."""
    return _spark_sft_full("fp32")


def muse_spark_816b_a42b_ctr_sft_full_bf16_tpu7x() -> SFTTrainerConfig:
    """Full-parameter CTR SFT with torch.optim.AdamW on bf16 weights (round to
    nearest). NOT a valid fine-tuning recipe at small learning rates (most
    updates fall below half a bf16 ulp and are lost); kept for step-time and
    update-fraction comparisons."""
    return _spark_sft_full("rne")


def muse_spark_816b_a42b_ctr_sft_full_bf16sr_tpu7x() -> SFTTrainerConfig:
    """Full-parameter CTR SFT with stochastic rounding and bf16 moments
    (LowPrecisionAdamW; 8 bytes per parameter)."""
    return _spark_sft_full("sr")


def muse_spark_816b_a42b_ctr_sft_full_adamw8bit_tpu7x() -> SFTTrainerConfig:
    """Alias of ``muse_spark_816b_a42b_ctr_sft_full_sr8_tpu7x`` (8-bit moments with
    stochastically rounded bf16 weights, LowPrecisionAdamW)."""
    return _spark_sft_full("sr8")


def muse_spark_816b_a42b_ctr_sft_measure_placement_tpu7x() -> SFTTrainerConfig:
    """Measures the routing on the SFT data and writes a load-balanced EP-8
    expert placement for the EP recipes: LoRA (cheap) at expert parallelism 1
    on the same packed chat rows, released weights; rank 0 writes
    ``SPARK_EXPERT_PLACEMENT_OUT`` at step ``SPARK_EXPERT_STATS_STEP``
    (default 2) from the router counts summed over all ranks."""
    from torchtitan.experiments.tpu.muse_spark import expert_placement

    if not os.environ.get("SPARK_EXPERT_PLACEMENT_OUT"):
        raise ValueError("set SPARK_EXPERT_PLACEMENT_OUT to the placement file to write")
    shape = dataclasses.replace(SFT_LORA_SHAPE, warmup_steps=1, lr_min_factor=1.0)
    config = _spark_sft_base(shape=shape, precision=BF16, lora=SPARK_LORA)
    config.training.steps = int(os.environ.get("SPARK_EXPERT_STATS_STEP", "2")) + 1
    config = as_sft_config(config, SFTConfig(lora_save="none"))
    spark_tpu_settings(config, full_param=False, fused_ce=True, frozen_head=True)
    return expert_placement.install_placement_writer(config)


def muse_spark_816b_a42b_ctr_sft_lora_ep8_tpu7x() -> SFTTrainerConfig:
    """LoRA r 16 / alpha 32 CTR SFT on the full-parameter recipes' layout (EP 8
    with the measured placement, fast reader, same data and rows), for
    side-by-side comparisons: ``SPARK_LR`` (default 1e-4) after
    ``SPARK_WARMUP`` warmup steps (default 2), then constant."""
    shape = dataclasses.replace(
        SFT_LORA_SHAPE,
        seq_len=int(os.environ.get("SPARK_SEQ", "8192")),
        lr=_env_float("SPARK_LR", SFT_LORA_SHAPE.lr),
        warmup_steps=int(os.environ.get("SPARK_WARMUP", "2")),
        lr_min_factor=_env_float("SPARK_LR_MIN", 1.0),
    )
    config = as_sft_config(
        _spark_sft_base(shape=shape, precision=BF16, lora=SPARK_LORA),
        SFTConfig(lora_save="none"),
    )
    spark_tpu_settings(config, full_param=False, fused_ce=True, frozen_head=True)
    return spark_expert_parallel(
        config,
        degree=int(os.environ.get("SPARK_EP", "8")),
        capacity_factor=_env_float("SPARK_EP_CAPACITY", 2.0),
    )
