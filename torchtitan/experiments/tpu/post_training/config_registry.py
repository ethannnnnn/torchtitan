# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Chat SFT recipes for TorchTitan on TPU, independent of the model.

``build_sft_config`` turns a ``ModelSpec`` and a few recipe knobs into a
complete trainer config: ``ChatSFTDataLoader`` (assistant-only labels, packed
documents), ``ChunkedLossWrapper(CrossEntropyLoss)`` normalized by the global
count of supervised tokens, optional LoRA, and AdamW / AdamW8bit / AdamWBf16.

Models come from ``resolve_model_spec(name, flavor, seq_len)``: ``"muse_spark"``
(the Muse Spark TPU package; any module listed in ``SPARK_MODEL_MODULES`` that
exposes ``model_registry(flavor, seq_len=...)``) or ``"module.path:function"``
for any other registry, e.g.
``"torchtitan.experiments.tpu.qwen3_5:model_registry"``. The registered recipe
functions at the end of this file are what ``--module`` / ``--config`` select;
paths are placeholders to override on the command line.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any

from torchtitan.components.checkpointer.dcp import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer.lr_scheduler import LRSchedulersContainer
from torchtitan.components.validate import Validator
from torchtitan.config.configs import (
    CommConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.experiments.tpu.tpu_job_config import (
    LossKernelConfig,
    TPUConfig,
    TPUTrainerConfig,
    TpuTrainerCompileConfig,
)

from .lora import LoRAOptions, apply_lora
from .optimizers import adamw_config
from .sft_config import SFTConfig, SFTTrainerConfig, as_sft_config
from .sft_data import ChatSFTDataLoader

SPARK_MODEL_MODULES = ("torchtitan.experiments.tpu.muse_spark",)
MODEL_ASSETS = "/mnt/model"
SFT_DATA = "/mnt/data/sft/train.jsonl"


def resolve_model_spec(
    name: str, flavor: str, seq_len: int, **kwargs: Any
) -> Any:
  """A ``ModelSpec`` from a registry name (see the module doc)."""
  if name == "muse_spark":
    errors = []
    for module_name in SPARK_MODEL_MODULES:
      try:
        module = importlib.import_module(module_name)
      except ImportError as exc:
        errors.append(f"{module_name}: {exc}")
        continue
      return module.model_registry(flavor, seq_len=seq_len, **kwargs)
    raise ImportError(
        "no Muse Spark model registry found: " + "; ".join(errors)
    )
  if ":" in name:
    module_name, function = name.split(":", 1)
    return getattr(importlib.import_module(module_name), function)(
        flavor, seq_len=seq_len, **kwargs
    )
  raise ValueError(f"unknown model registry {name!r}")


@dataclass(frozen=True)
class ModelChoice:
  name: str
  flavor: str
  kwargs: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class TrainShape:
  """Per-rank microbatch geometry and run length."""

  seq_len: int = 8192
  rows_per_microbatch: int = 1
  num_chunks: int = 8
  steps: int = 100
  warmup_steps: int = 10
  lr: float = 1e-5
  lr_decay: str = "cosine"
  lr_min_factor: float = 0.1
  """1.0 with warmup_steps=0 keeps the learning rate constant."""
  betas: tuple[float, float] = (0.9, 0.95)
  weight_decay: float = 0.0
  max_norm: float = 1.0


@dataclass(frozen=True)
class Precision:
  """``optimizer``: AdamW, AdamW8bit or AdamWBf16."""

  dtype: str = "float32"
  optimizer: str = "AdamW"
  lowbit_compile_backend: str = ""


@dataclass(frozen=True)
class Parallel:
  dp_shard: int = -1
  expert_parallel: int = 1
  tensor_parallel: int = 1
  compile_model: bool = False
  eager_mode: str = ""
  """TorchTPU eager mode; "" = DEFER_AND_FUSE when compiling, else default."""
  activation_checkpoint: str = "selective"
  """none | selective | full."""
  fsdp_reshard_after_forward: str = "default"
  compile_grad_norm: bool = False


def apply_model_hooks(model: ModelChoice, config: Any) -> Any:
  """Model-specific trainer settings the model owner requires.

  Muse Spark on TPU trains its routed experts with the Pallas grouped-matmul
  kernels (``muse_spark.config_registry._use_tpu_moe_kernels``; a no-op when
  the model was built without the TPU expert converters).
  """
  if model.name == "muse_spark":
    from torchtitan.experiments.tpu.muse_spark.config_registry import (
        _use_tpu_moe_kernels,
    )

    _use_tpu_moe_kernels(config)
  return config


def _model_spec(
    model: ModelChoice, seq_len: int, lora: LoRAOptions | None
) -> Any:
  spec = resolve_model_spec(
      model.name, model.flavor, seq_len, **dict(model.kwargs)
  )
  if lora is not None:
    apply_lora(spec, lora)
  return spec


def base_config(
    *,
    model_spec: Any,
    hf_assets_path: str,
    loss: Any,
    dataloader: Any,
    shape: TrainShape,
    precision: Precision,
    parallel: Parallel,
    num_tokens_per_train_step: int,
    checkpoint_interval: int,
    seed: int,
) -> TPUTrainerConfig:
  """The trainer config every SFT recipe builds on.

  ``shape`` sets the microbatch geometry, run length, learning-rate schedule
  and AdamW hyperparameters, ``precision`` the weight dtype and optimizer, and
  ``parallel`` the parallelism, activation checkpointing and compile policy.
  The checkpointer loads the initial weights from the Hugging Face checkpoint
  in ``hf_assets_path`` and saves every ``checkpoint_interval`` steps;
  gradients are reduced in fp32.
  """
  microbatch_tokens = shape.rows_per_microbatch * shape.seq_len
  if microbatch_tokens % shape.num_chunks:
    raise ValueError(
        f"num_chunks {shape.num_chunks} must divide {microbatch_tokens}"
    )
  return TPUTrainerConfig(
      hf_assets_path=hf_assets_path,
      model_spec=model_spec,
      loss=loss,
      loss_kernel=LossKernelConfig(use_loss_kernel=False),
      metrics=MetricsProcessor.Config(log_freq=1),
      optimizer=adamw_config(
          lr=shape.lr,
          name=precision.optimizer,
          betas=shape.betas,
          weight_decay=shape.weight_decay,
          lowbit_compile_backend=precision.lowbit_compile_backend,
      ),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=shape.warmup_steps,
          decay_type=shape.lr_decay,
          min_lr_factor=shape.lr_min_factor,
      ),
      training=TrainingConfig(
          steps=shape.steps,
          max_context_length=shape.seq_len,
          num_tokens_per_microbatch_per_dp_rank=microbatch_tokens,
          num_tokens_per_train_step=num_tokens_per_train_step,
          dtype=precision.dtype,
          mixed_precision_param="bfloat16",
          mixed_precision_reduce="float32",
          max_norm=shape.max_norm,
      ),
      dataloader=dataloader,
      parallelism=ParallelismConfig(
          data_parallel_shard_degree=parallel.dp_shard,
          data_parallel_replicate_degree=1,
          tensor_parallel_degree=parallel.tensor_parallel,
          context_parallel_degree=1,
          pipeline_parallel_degree=1,
          expert_parallel_degree=parallel.expert_parallel,
          use_simple_fsdp=True,
          fsdp_reshard_after_forward=parallel.fsdp_reshard_after_forward,
      ),
      checkpoint=CheckpointManager.Config(
          enable=True,
          initial_load_in_hf=True,
          initial_load_model_only=True,
          interval=checkpoint_interval,
          last_save_model_only=False,
          last_save_in_hf=False,
          async_mode="disabled",
      ),
      activation_checkpoint={
          "none": None,
          "selective": SelectiveAC.Config(preserve_rng_state=False),
          "full": FullAC.Config(preserve_rng_state=False),
      }[parallel.activation_checkpoint],
      # Per-layer TorchTPU compile ("layer") of the model when compiling;
      # the compiled gradient norm is independent of model compilation.
      compile=(
          TpuTrainerCompileConfig(
              enable=True,
              backend="tpu",
              components=["model"],
              mode="layer",
              compile_grad_norm=parallel.compile_grad_norm,
          )
          if parallel.compile_model
          else TpuTrainerCompileConfig(
              enable=False,
              backend="tpu",
              mode="layer",
              compile_grad_norm=parallel.compile_grad_norm,
          )
      ),
      comm=CommConfig(train_timeout_seconds=1800),
      validator=Validator.Config(enable=False),
      debug=DebugConfig(seed=seed),
      tpu_config=TPUConfig(
          enable_amp=False,
          eager_mode=parallel.eager_mode
          or ("DEFER_AND_FUSE" if parallel.compile_model else ""),
      ),
  )


# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SFTRecipe:
  model: ModelChoice
  hf_assets_path: str = MODEL_ASSETS
  data_path: str = SFT_DATA
  shape: TrainShape = TrainShape(lr=1e-5)
  precision: Precision = Precision()
  parallel: Parallel = Parallel()
  lora: LoRAOptions | None = None
  microbatches_per_step: int = 1
  """Gradient-accumulation microbatches per rank per optimizer step."""
  reasoning_effort: str = ""
  max_documents_per_row: int = 16
  overlong: str = "drop"
  checkpoint_interval: int = 10**9
  lora_save: str = "adapter"
  """LoRA runs: see sft_config.SFTConfig.lora_save."""
  seed: int = 42


def build_sft_config(recipe: SFTRecipe) -> SFTTrainerConfig:
  shape = recipe.shape
  spec = _model_spec(recipe.model, shape.seq_len, recipe.lora)
  config = base_config(
      model_spec=spec,
      hf_assets_path=recipe.hf_assets_path,
      loss=ChunkedLossWrapper.Config(
          num_chunks=shape.num_chunks, loss_fn=CrossEntropyLoss.Config()
      ),
      dataloader=ChatSFTDataLoader.Config(
          data_path=recipe.data_path,
          reasoning_effort=recipe.reasoning_effort,
          max_documents_per_row=recipe.max_documents_per_row,
          overlong=recipe.overlong,
          microbatches_per_step=recipe.microbatches_per_step,
      ),
      shape=shape,
      precision=recipe.precision,
      parallel=recipe.parallel,
      # SFTTPUTrainer takes the accumulation count from the dataloader
      # (microbatches_per_step), independent of the data-parallel degree.
      num_tokens_per_train_step=-1,
      checkpoint_interval=recipe.checkpoint_interval,
      seed=recipe.seed,
  )
  apply_model_hooks(recipe.model, config)
  return as_sft_config(config, SFTConfig(lora_save=recipe.lora_save))


# ---------------------------------------------------------------------------
# Registered recipes
# ---------------------------------------------------------------------------
# Muse Spark Open 816B-A42B on TPU7x. The flavor comes from SPARK_FLAVOR
# (default 816B-A42B; 816B-A42B-2L / -4L / -8L truncate it for small slices),
# SPARK_EP sets the expert-parallel degree and SPARK_COMPILE=1 compiles blocks.
# LoRA SFT trains the adapters with AdamW (lr 1e-4 with warmup and cosine
# decay to 0.1x, betas 0.9 / 0.95, no weight decay) over a frozen bf16 base.
# The recipe was tuned on chat SFT data rendered with reasoning effort "low";
# any chat JSONL works. The full-parameter SFT recipes (LowPrecisionAdamW,
# fp32 gradient reduction, expert parallelism with a measured placement) are
# registered in full_param_sft.py
# (--module torchtitan.experiments.tpu.post_training.full_param_sft).


def spark() -> ModelChoice:
  """Muse Spark in the flavor ``SPARK_FLAVOR`` (default ``816B-A42B``)."""
  return ModelChoice("muse_spark", os.environ.get("SPARK_FLAVOR", "816B-A42B"))


def spark_parallel() -> Parallel:
  """The Muse Spark port's TPU bring-up settings: full activation
  checkpointing, reshard after forward, DEFER_AND_FUSE eager mode, compiled
  gradient norm; per-block compile only with SPARK_COMPILE=1."""
  return Parallel(
      expert_parallel=int(os.environ.get("SPARK_EP", "1")),
      compile_model=os.environ.get("SPARK_COMPILE", "0") == "1",
      eager_mode="DEFER_AND_FUSE",
      activation_checkpoint="full",
      fsdp_reshard_after_forward="always",
      compile_grad_norm=True,
  )


# Earlier names of base_config, spark and spark_parallel, kept for the modules
# that import them.
_base_config = base_config
_spark = spark
_spark_parallel = spark_parallel


def spark_tpu_settings(
    config: Any, *, full_param: bool, fused_ce: bool, frozen_head: bool
) -> Any:
  """TPU7x settings measured on one 4x4x4 slice with the 816B model.

  The shared Spark TPU7x settings (``muse_spark/tpu7x.py``: Tokamax Splash
  attention with the Spark blocks, unpaired Pallas grouped matmuls with
  512 x 1024 x 1024 tiles and kernel-native expert weights, a 64 MiB
  scoped-VMEM limit, power state 7), a fused route-score backward and the TPU
  topology mesh. ``full_param`` reduces gradients in bf16: with expert
  parallel degree 1 every layer's unsharded expert gradients are large and an
  fp32 reduction doubles them (recipes that keep an fp32 reduction set
  ``training.mixed_precision_reduce`` afterwards). ``fused_ce`` swaps the
  soft-capped LM head for ``FusedLossLinearWrapper`` and the loss for the
  Pallas fused linear cross-entropy carrying the head's multiplier and cap;
  ``frozen_head`` keeps that head frozen (LoRA runs).
  """
  from torchtitan.components.lora import _make_frozen_config
  from torchtitan.config import derive
  from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
  from torchtitan.experiments.tpu.muse_spark import tpu7x
  from torchtitan.experiments.tpu.overrides.fused_loss import FusedLossLinearWrapper
  from torchtitan.experiments.tpu.token_dispatcher import TpuAllToAllTokenDispatcher

  model = config.model_spec.model
  # Tokamax Splash attention with the Spark blocks, tiled Pallas grouped matmuls,
  # 64 MiB scoped VMEM and power state 7 (muse_spark/tpu7x.py).
  tpu7x.apply(config)
  for _, dispatcher, _, _ in model.traverse(TpuAllToAllTokenDispatcher.Config):
    dispatcher.fuse_route_score_backward = True
  config.parallelism.enable_tpu_topo_mesh = True
  if full_param:
    config.training.mixed_precision_reduce = "bfloat16"
  if fused_ce:
    head = model.lm_head
    wrapper = derive(
        head,
        FusedLossLinearWrapper.Config,
        soft_cap=head.output_soft_cap_temp,
        output_multiplier=head.output_multiplier,
    )
    model.lm_head = _make_frozen_config(wrapper) if frozen_head else wrapper
    config.loss = PallasCrossEntropyLoss.Config(
        soft_cap=head.output_soft_cap_temp,
        output_multiplier=head.output_multiplier,
    )
  return config


SPARK_LORA = LoRAOptions(rank=16, alpha=32.0)
SFT_LORA_SHAPE = TrainShape(
    lr=1e-4, steps=1000, warmup_steps=100, lr_min_factor=0.1
)
# The base shape of the full-parameter recipes (full_param_sft.py).
SFT_FULL_SHAPE = TrainShape(
    lr=1e-5, steps=1000, warmup_steps=100, lr_min_factor=0.1
)
BF16 = Precision(dtype="bfloat16", optimizer="AdamW")


def muse_spark_816b_a42b_sft_lora_tpu7x() -> SFTTrainerConfig:
  """Chat SFT with LoRA r 16 / alpha 32 on q, k, v, o over a frozen bf16 base;
  AdamW on the adapters; Pallas fused cross-entropy; 8192-token packed rows."""
  config = build_sft_config(
      SFTRecipe(
          model=spark(),
          shape=SFT_LORA_SHAPE,
          precision=BF16,
          parallel=spark_parallel(),
          lora=SPARK_LORA,
          reasoning_effort="low",
      )
  )
  return spark_tpu_settings(
      config, full_param=False, fused_ce=True, frozen_head=True
  )
