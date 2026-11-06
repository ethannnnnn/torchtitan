# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Muse Spark LoRA SFT recipes for the end-to-end stages. ASCII only.

Built on ``post_training.config_registry`` (``base_config``) and the chat
SFT loader; the differences are the model spec (the Muse Spark HF adapter with
the fast checkpoint reader, see ``lora_publish``) and per-stage geometry:

* ``s0_sft_tiny``: CPU, the public tiny Muse Spark checkpoint (``tiny-hf``),
  no TPU kernels, fp32.
* ``s1_sft_trunc2``: TPU, the first two released layers at real dimensions
  (``816B-A42B-2L``), bf16 weights; fits small slices.
* ``s2_sft_full``: TPU, the full 62-layer model on one 64-chip slice.

Every stage trains LoRA adapters (rank 16 and alpha 32 unless overridden) on
the attention's query, key and value projections and on its output
projection, feeds them packed rows of chat records in which only the
assistant tokens carry labels, keeps nothing but the LoRA state in its
checkpoints, and exports a PEFT adapter through the trainer,
``lora_publish.SparkSFTTrainer``.

Knobs that change between launches come from the environment (defaults in
brackets) so that one registry serves local runs and pods:
``SPARK_STEPS`` [stage default], ``SPARK_LR`` [1e-4 on TPU],
``SPARK_LORA_RANK`` [16], ``SPARK_LORA_ALPHA`` [32], ``SPARK_REASONING``
[low] (chat-template reasoning effort; the template's own default is xhigh,
while the dataset's length statistics and evaluation render with low, so
training and evaluation see the same system prompt), ``SPARK_COMPILE`` [0:
eager DEFER_AND_FUSE; 1 compiles blocks], ``SPARK_AC`` [1 on TPU, 0 on CPU:
full activation checkpointing], ``SPARK_WARMUP`` [2], ``SPARK_ADAPTER_INIT``
(PEFT adapter directory to start from), ``SPARK_EXPORT_EVERY`` [0] (also
export the PEFT adapter every N steps), ``SPARK_FAST_READER`` [1: the
positional reader of ``tpu_knobs``; 0: the stock Hugging Face reader] and
``SPARK_HF_READER_THREADS`` [8] (threads of the positional reader),
``SPARK_SFT_SEQ`` [8192 on TPU, 4096 on CPU] (packed row length, a multiple
of 1,024 on TPU), ``SPARK_SFT_ROWS`` [1] (packed rows per rank and
microbatch; two rows amortize the per-step expert all-gathers of expert
parallelism 1), ``SPARK_SFT_MICROBATCHES`` [1] (gradient-accumulation
microbatches per step), ``SPARK_VAL_DATA`` (held-out chat records, e.g.
``/mnt/data/sft/val.jsonl``; empty = no validation), ``SPARK_VAL_EVERY``
[10] and ``SPARK_VAL_BATCHES`` [0 = all held-out records]: the held-out loss
is measured before the first step, every ``SPARK_VAL_EVERY`` steps and after
the last step (``post_training.sft_train``). ``--checkpoint.interval=N``
adds LoRA-only checkpoints (and PEFT exports) every N steps; the last step
always saves both.

Paths are placeholders; override them on the command line
(``--hf_assets_path``, ``--dataloader.data-path``).
"""

from __future__ import annotations

import os
from typing import Any

from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.experiments.tpu.post_training.config_registry import (
    Parallel,
    Precision,
    TrainShape,
    base_config,
)
from torchtitan.experiments.tpu.post_training.lora import LoRAOptions, apply_lora
from torchtitan.experiments.tpu.post_training.sft_config import SFTConfig, SFTTrainerConfig, as_sft_config
from torchtitan.experiments.tpu.post_training.sft_data import ChatSFTDataLoader

from .lora_publish import FastReaderSparkAdapter

# Upstream's -1: tokens per optimizer step = microbatch tokens x data-parallel degree x accumulated microbatches.
DERIVED_STEP_TOKENS = -1


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def spark_model_spec(flavor: str, seq_len: int, *, lora: LoRAOptions | None,
                     tpu_kernels: bool) -> Any:
    from torchtitan.experiments.tpu.muse_spark import model_registry

    spec = model_registry(flavor, seq_len=seq_len, tpu_moe_kernels=tpu_kernels)
    spec.state_dict_adapter = FastReaderSparkAdapter
    if lora is not None:
        apply_lora(spec, lora)
    return spec


def _lora() -> LoRAOptions:
    return LoRAOptions(rank=_env_int("SPARK_LORA_RANK", 16), alpha=_env_float("SPARK_LORA_ALPHA", 32.0))


def sft_config(*, flavor: str, tpu: bool, shape: TrainShape, precision: Precision,
               parallel: Parallel, data_path: str = "/mnt/data/sft/train.jsonl",
               hf_assets_path: str = "/mnt/model", microbatches_per_step: int = 1,
               max_documents_per_row: int = 8, seed: int = 42) -> SFTTrainerConfig:
    spec = spark_model_spec(flavor, shape.seq_len, lora=_lora(), tpu_kernels=tpu)
    base = base_config(
        model_spec=spec,
        hf_assets_path=hf_assets_path,
        loss=ChunkedLossWrapper.Config(num_chunks=shape.num_chunks, loss_fn=CrossEntropyLoss.Config()),
        dataloader=ChatSFTDataLoader.Config(
            data_path=data_path,
            reasoning_effort=os.environ.get("SPARK_REASONING", "low"),
            max_documents_per_row=max_documents_per_row,
            overlong="drop",
            microbatches_per_step=microbatches_per_step,
        ),
        shape=shape,
        precision=precision,
        parallel=parallel,
        num_tokens_per_train_step=DERIVED_STEP_TOKENS,
        checkpoint_interval=10**9,
        seed=seed,
    )
    # lora_save="full": SFTTPUTrainer leaves the checkpointer alone;
    # SparkSFTTrainer installs the adapter-only checkpoints (resumable DCP of
    # the adapters and their training state) and writes the PEFT exports.
    return as_sft_config(base, SFTConfig(
        lora_save="full",
        val_data_path=os.environ.get("SPARK_VAL_DATA", ""),
        val_every=_env_int("SPARK_VAL_EVERY", 10),
        val_max_microbatches=_env_int("SPARK_VAL_BATCHES", 0),
    ))


# ---------------------------------------------------------------------------
# S0: CPU, public tiny checkpoint
# ---------------------------------------------------------------------------

S0_PRECISION = Precision(dtype="float32", optimizer="AdamW")
S0_PARALLEL = Parallel(dp_shard=-1, compile_model=False, activation_checkpoint="none")


def s0_sft_tiny() -> SFTTrainerConfig:
    # SPARK_AC=1 wraps the blocks for full activation checkpointing as on TPU
    # (the wrappers change parameter paths; the resume test covers both).
    parallel = (Parallel(dp_shard=-1, compile_model=False, activation_checkpoint="full")
                if os.environ.get("SPARK_AC", "0") == "1" else S0_PARALLEL)
    return sft_config(
        flavor="tiny-hf", tpu=False,
        shape=TrainShape(seq_len=_env_int("SPARK_SFT_SEQ", 4096), rows_per_microbatch=1, num_chunks=4,
                         steps=_env_int("SPARK_STEPS", 4), warmup_steps=1,
                         lr=_env_float("SPARK_LR", 1e-3)),
        precision=S0_PRECISION, parallel=parallel,
        microbatches_per_step=_env_int("SPARK_SFT_MICROBATCHES", 1),
    )


# ---------------------------------------------------------------------------
# S1 / S2: TPU (bf16 weights, bf16 AdamW for the adapters)
# ---------------------------------------------------------------------------

TPU_PRECISION = Precision(dtype="bfloat16", optimizer="AdamW")


def _tpu_parallel() -> Parallel:
    # Eager DEFER_AND_FUSE (tpu_knobs) as in the 64-chip bring-up; AC is set there.
    return Parallel(
        dp_shard=-1,
        expert_parallel=1,
        compile_model=os.environ.get("SPARK_COMPILE", "0") == "1",
        activation_checkpoint="none",
    )


def _tpu(cfg):
    from .tpu_knobs import apply_tpu_knobs

    apply_tpu_knobs(cfg)
    return cfg


def _tpu_sft(flavor: str, steps: int) -> SFTTrainerConfig:
    return _tpu(sft_config(
        flavor=flavor, tpu=True,
        shape=TrainShape(seq_len=_env_int("SPARK_SFT_SEQ", 8192),
                         rows_per_microbatch=_env_int("SPARK_SFT_ROWS", 1),
                         num_chunks=16, steps=_env_int("SPARK_STEPS", steps),
                         warmup_steps=_env_int("SPARK_WARMUP", 2),
                         lr=_env_float("SPARK_LR", 1e-4)),
        precision=TPU_PRECISION, parallel=_tpu_parallel(),
        microbatches_per_step=_env_int("SPARK_SFT_MICROBATCHES", 1),
    ))


def s1_sft_trunc2() -> SFTTrainerConfig:
    return _tpu_sft("816B-A42B-2L", 5)


def s2_sft_full() -> SFTTrainerConfig:
    return _tpu_sft("816B-A42B", 20)
