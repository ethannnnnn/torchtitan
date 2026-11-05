# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration types for SFTTPUTrainer."""

from __future__ import annotations

import dataclasses

from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig

LORA_SAVE_MODES = ("adapter", "peft", "full", "none")


@dataclasses.dataclass(kw_only=True, slots=True)
class SFTConfig:
  """Supervised fine-tuning settings that ``TPUTrainerConfig`` lacks."""

  lora_save: str = "adapter"
  """LoRA runs only. "adapter": every checkpoint save writes the adapter-only
    resume state (LoRA tensors, their optimizer state, LR schedule, dataloader
    position and step; DCP, <dump_folder>/checkpoint/step-N) and the LoRA
    adapter in the PEFT layout (<dump_folder>/lora_adapter/step-N); restarts
    read the frozen base from the released checkpoint and resume from the
    newest complete step-N (a full save would rewrite the 1.6 TB base);
    "peft": only the PEFT adapter (no resume state); "full": TorchTitan's DCP
    save of the whole model and training state; "none": no saves. Runs
    without LoRA always use TorchTitan's saves."""

  val_data_path: str = ""
  """Held-out chat SFT records in the training data's layout (for example
    the dataset's ``sft/val.jsonl``). Empty: no validation. The held-out loss
    is the summed cross-entropy of the assistant tokens of every held-out
    record divided by their count (the training loss's normalization), measured
    before the first step, every ``val_every`` steps and after the last step."""

  val_every: int = 10
  """Validation interval in optimizer steps; 0: only before the first and
    after the last step."""

  val_max_microbatches: int = 0
  """Cap on packed microbatches per rank per validation pass; 0: all of the
    rank's held-out records."""


@dataclasses.dataclass(kw_only=True, slots=True)
class SFTTrainerConfig(TPUTrainerConfig):
  """``TPUTrainerConfig`` extended with the ``sft`` section.

  Subclassing keeps it acceptable to upstream's TPU start-up, which only takes
  ``TPUTrainerConfig`` instances.
  """

  sft: SFTConfig = dataclasses.field(default_factory=SFTConfig)


# Names of the fields TPUTrainerConfig declares, inherited ones included: what
# as_sft_config takes over from its base.
_TPU_TRAINER_FIELD_NAMES = tuple(
    field.name for field in dataclasses.fields(TPUTrainerConfig)
)


def validated_lora_save(mode: str) -> str:
  """``mode`` if it is one of ``LORA_SAVE_MODES``; ``ValueError`` if not."""
  if mode in LORA_SAVE_MODES:
    return mode
  raise ValueError(
      f"unknown sft.lora_save {mode!r}; expected one of {LORA_SAVE_MODES}"
  )


def as_sft_config(base: TPUTrainerConfig, sft: SFTConfig) -> SFTTrainerConfig:
  """A new ``SFTTrainerConfig`` with ``base``'s TPU trainer fields and ``sft``.

  Field values are shared with ``base`` (not copied) and ``base`` is left
  unchanged; an ``SFTTrainerConfig`` base gets its ``sft`` section replaced.
  Raises ``ValueError`` for an invalid ``sft``.
  """
  validated_lora_save(sft.lora_save)
  if sft.val_every < 0 or sft.val_max_microbatches < 0:
    raise ValueError(
        "sft.val_every and sft.val_max_microbatches cannot be negative, got "
        f"{sft.val_every} and {sft.val_max_microbatches}"
    )
  carried = {name: getattr(base, name) for name in _TPU_TRAINER_FIELD_NAMES}
  return SFTTrainerConfig(sft=sft, **carried)
