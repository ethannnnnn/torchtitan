# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test-only recipes for the CPU smokes: the SFT builder with 512-token rows on
one CPU rank, on two models -- the TPU Qwen3.5 debug model
(``torchtitan.experiments.tpu.qwen3_5``: dim 256, 8 layers of which 2 use full
attention, 248320-entry vocabulary; random weights) and, when the Muse Spark
TPU package is installed, the Muse Spark port at the public tiny checkpoint's
shape (``tiny-hf``: dim 64, 2 layers, 16 experts, sliding window 32;
plain-PyTorch experts on CPU). Paths come from the command line."""

from __future__ import annotations

from torchtitan.experiments.tpu.post_training.config_registry import (
    ModelChoice,
    Parallel,
    Precision,
    SFTRecipe,
    TrainShape,
    build_sft_config,
)
from torchtitan.experiments.tpu.post_training.lora import LoRAOptions

DEBUG = ModelChoice("torchtitan.experiments.tpu.qwen3_5:model_registry", "debugmodel")
SPARK_TINY = ModelChoice("muse_spark", "tiny-hf", (("tpu_moe_kernels", False),))
CPU = Parallel(compile_model=False)
LORA = LoRAOptions(rank=8, alpha=16.0)


def _shape(lr: float) -> TrainShape:
    return TrainShape(seq_len=512, rows_per_microbatch=2, num_chunks=4, steps=8,
                      warmup_steps=1, lr=lr, lr_decay="linear", lr_min_factor=1.0)


def _sft(**overrides):
    kwargs = dict(
        model=DEBUG,
        shape=_shape(3e-3),
        parallel=CPU,
        microbatches_per_step=2,
        reasoning_effort="low",
        max_documents_per_row=4,
    )
    kwargs.update(overrides)
    return build_sft_config(SFTRecipe(**kwargs))


def sft_debug_smoke():
    return _sft()


def sft_debug_lora_smoke():
    """LoRA with TorchTitan's full DCP save (sft.lora_save=full)."""
    return _sft(lora=LORA, lora_save="full", shape=_shape(2e-2))


def sft_debug_lora_adapter_smoke():
    """LoRA with adapter saves (sft.lora_save=adapter, the default)."""
    return _sft(lora=LORA, shape=_shape(2e-2))


def sft_debug_adamw8bit_smoke():
    return _sft(precision=Precision(dtype="float32", optimizer="AdamW8bit"))


def sft_debug_bf16_smoke():
    return _sft(precision=Precision(dtype="bfloat16", optimizer="AdamWBf16"))


def sft_spark_tiny_smoke():
    return _sft(model=SPARK_TINY)


def sft_spark_tiny_lora_smoke():
    """LoRA with adapter saves (sft.lora_save=adapter, the default)."""
    return _sft(model=SPARK_TINY, lora=LORA, shape=_shape(3e-2))
