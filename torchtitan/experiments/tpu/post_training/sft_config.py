# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration types for SFTTPUTrainer."""

from __future__ import annotations

import dataclasses

from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig

LORA_SAVE_MODES = ("adapter", "full", "none")


@dataclasses.dataclass(kw_only=True, slots=True)
class SFTConfig:
    """SFT options the native TPUTrainerConfig has no field for."""

    lora_save: str = "adapter"
    """LoRA runs only. "adapter": every checkpoint save writes just the LoRA
    adapter, in the PEFT layout, to <dump_folder>/lora_adapter/step-N (the
    frozen base is the released checkpoint, so a full save would rewrite it);
    "full": TorchTitan's DCP save of the whole model and training state;
    "none": no saves. Runs without LoRA always use TorchTitan's saves."""


@dataclasses.dataclass(kw_only=True, slots=True)
class SFTTrainerConfig(TPUTrainerConfig):
    """TPUTrainerConfig plus the SFT block (a subclass, so the TPU startup's
    isinstance checks keep working)."""

    sft: SFTConfig = dataclasses.field(default_factory=SFTConfig)


def as_sft_config(base: TPUTrainerConfig, sft: SFTConfig) -> SFTTrainerConfig:
    if sft.lora_save not in LORA_SAVE_MODES:
        raise ValueError(f"sft.lora_save must be one of {LORA_SAVE_MODES}")
    fields = {
        field.name: getattr(base, field.name)
        for field in dataclasses.fields(base)
        if field.name != "sft"
    }
    return SFTTrainerConfig(**fields, sft=sft)
