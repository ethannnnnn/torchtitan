# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Entry point for chat SFT on TPU.

Runs the upstream TPU trainer startup with ``SFTTPUTrainer``:

* ``TPUTrainer.__init__`` replaces any configured loss with plain
  cross-entropy (or the Pallas fused kernel when selected). After
  ``Trainer.__init__`` has already wired a configured ``ChunkedLossWrapper``
  (``model._skip_lm_head = True``), that replacement would feed hidden states
  to a logits loss, so ``SFTTPUTrainer`` restores the configured chunked loss
  and its lm_head binding. The loss is the summed cross-entropy of the
  supervised tokens divided by the step's global supervised-token count.
* The accumulation count per optimizer step comes from the dataloader
  (``ChatSFTDataLoader.Config.microbatches_per_step``).
* Each step logs (and rank 0 writes to ``<dump_folder>/sft-records-rank0.json``)
  the packed documents, tokens, supervised tokens and this rank's loss share;
  the trainable / frozen parameter counts are logged once (LoRA runs).
* LoRA runs save only the adapter (``sft.lora_save="adapter"``, default): at
  every checkpoint step all ranks gather the adapter tensors (a collective over
  the sorted adapter keys, never the frozen base) and rank 0 writes them in the
  PEFT layout, named through the model's state-dict adapter, to
  ``<dump_folder>/lora_adapter/step-N``. ``"full"`` keeps TorchTitan's DCP save
  of the whole model; ``"none"`` saves nothing.

Launch with torchrun, module torchtitan.experiments.tpu.post_training.sft_train,
``--module torchtitan.experiments.tpu.post_training.config_registry`` and an
SFT recipe; see README.md.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import torch

import torch.distributed as dist
from torch.distributed.tensor import DTensor

from torchtitan.components.checkpointer.base import ModelWrapper
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.experiments.tpu import gmain
from torchtitan.experiments.tpu import train as native_train
from torchtitan.experiments.tpu.train import TPUTrainer

from . import lora as lora_mod
from .lora import lora_parameter_report
from .sft_config import LORA_SAVE_MODES

logger = logging.getLogger(__name__)


class SFTTPUTrainer(TPUTrainer):
    """TPU trainer for supervised fine-tuning with a chunked loss."""

    def __init__(self, job_config: Any) -> None:
        super().__init__(job_config)
        config = self.config
        if isinstance(config.loss, ChunkedLossWrapper.Config):
            rebuilt = config.loss.build(compile_config=config.compile)
            assert len(self.model_parts) == 1, "SFT path expects one model part"
            lm_head = getattr(self.model_parts[0], "lm_head", None)
            assert lm_head is not None, "ChunkedLossWrapper needs model.lm_head"
            rebuilt.set_lm_head(lm_head)
            self.model_parts[0]._skip_lm_head = True
            self.loss_fn = rebuilt
            logger.info("[sft] loss restored to ChunkedLossWrapper(%d chunks)", rebuilt.num_chunks)
        self.parameter_report = lora_parameter_report(self.model_parts[0])
        logger.info("[sft] parameters: %s", self.parameter_report)
        self.sft_records: list[dict[str, Any]] = []
        self._step_loss: torch.Tensor | None = None
        self._rank = int(os.environ.get("RANK", "0"))
        self.lora_exports: list[str] = []
        self._install_lora_save(config)

    # -- LoRA adapter saves ----------------------------------------------
    def _lora_modules(self) -> list[tuple[str, torch.nn.Module]]:
        return [
            (name, module) for name, module in self.model_parts[0].named_modules()
            if isinstance(getattr(module, "lora_a", None), torch.nn.Module)
            and isinstance(getattr(module, "lora_b", None), torch.nn.Module)
        ]

    def _install_lora_save(self, config: Any) -> None:
        sft = getattr(config, "sft", None)
        mode = getattr(sft, "lora_save", "full")
        if mode not in LORA_SAVE_MODES:
            raise ValueError(f"sft.lora_save={mode!r}; expected one of {LORA_SAVE_MODES}")
        modules = self._lora_modules()
        if not modules or mode == "full":
            return
        _, first = modules[0]
        rank = int(first.lora_a.weight.shape[0])
        self._lora_options = lora_mod.LoRAOptions(
            rank=rank, alpha=float(first._lora_scaling) * rank
        )
        checkpointer = self.checkpointer

        def _save(curr_step: int, last_step: bool = False) -> bool:
            if mode == "none" or not checkpointer._should_save(curr_step, last_step):
                return False
            self.export_lora_adapter(curr_step)
            return True

        checkpointer.save = _save
        logger.info("[sft] LoRA run: checkpoint saves write the adapter only (sft.lora_save=%s)", mode)

    def export_lora_adapter(self, step: int) -> str | None:
        """Gather the adapter on every rank; rank 0 writes the PEFT files."""
        state = ModelWrapper(self.model_parts).state_dict()
        gathered: dict[str, torch.Tensor] = {}
        for key in sorted(k for k in state if lora_mod.is_lora_key(k)):
            value = state[key]
            full = value.full_tensor() if isinstance(value, DTensor) else value
            if self._rank == 0:
                gathered[key] = full.detach().to("cpu", torch.float32)
            del full
        out = None
        if self._rank == 0:
            tensors = lora_mod.peft_adapter_state_dict(
                gathered, adapter=self.checkpointer.sd_adapter
            )
            target = Path(self.config.dump_folder) / "lora_adapter" / f"step-{step}"
            out = str(lora_mod.save_peft_adapter(
                target, tensors, options=self._lora_options,
                base_model_name_or_path=str(self.config.hf_assets_path),
            ))
            self.lora_exports.append(out)
            logger.info("[sft] step=%d LoRA adapter (%d tensors) -> %s", step, len(tensors), out)
        if dist.is_initialized():
            dist.barrier()
        return out

    def forward_backward_step(self, *, input_dict, global_valid_tokens):
        loss = super().forward_backward_step(
            input_dict=input_dict, global_valid_tokens=global_valid_tokens
        )
        detached = loss.detach()
        self._step_loss = detached if self._step_loss is None else self._step_loss + detached
        return loss

    def train_step(self, data_iterator):
        if hasattr(self.dataloader, "microbatches_for_step"):
            self.gradient_accumulation_steps = self.dataloader.microbatches_for_step()
        before = getattr(self.dataloader, "stats", None)
        snapshot = (
            (before.documents, before.tokens, before.valid_tokens) if before else (0, 0, 0)
        )
        self._step_loss = None
        result = super().train_step(data_iterator)
        stats = getattr(self.dataloader, "stats", None)
        record: dict[str, Any] = {"step": self.step}
        if self._step_loss is not None:
            # This rank's share of the step's global mean loss (the shares of
            # all data-parallel ranks sum to it). The trainer already reads the
            # loss back every log step, so this adds no device sync there.
            record["loss_local"] = float(self._step_loss)
        if stats is not None:
            record.update(
                epoch=stats.epoch,
                documents=stats.documents - snapshot[0],
                tokens=stats.tokens - snapshot[1],
                valid_tokens_local=stats.valid_tokens - snapshot[2],
                dropped_total=stats.dropped,
                truncated_total=stats.truncated,
            )
        self.sft_records.append(record)
        logger.info("[sft] step=%d %s", self.step, record)
        if self._rank == 0 and self.config.dump_folder:
            path = Path(self.config.dump_folder) / "sft-records-rank0.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(
                {"parameters": self.parameter_report, "steps": self.sft_records}, indent=2
            ))
        return result


def start_trainer(train_config: Any) -> None:
    original = native_train.TPUTrainer
    native_train.TPUTrainer = SFTTPUTrainer  # type: ignore[assignment]
    try:
        native_train.start_trainer(train_config)
    finally:
        native_train.TPUTrainer = original  # type: ignore[assignment]


if __name__ == "__main__":
    gmain.handle_main(start_trainer)
