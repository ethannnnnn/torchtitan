# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""LoRA-aware SFT trainer for Muse Spark. ASCII only.

Two gaps between the generic pieces and a LoRA run on an 816B model:

1. Checkpoint reader. ``MuseSparkStateDictAdapter`` maps the Hugging Face
   checkpoint and already leaves the LoRA adapter keys out of its checks
   (``post_training.lora.apply_lora`` wraps it: ``to_hf`` drops adapter keys,
   ``to_hf_for_export`` merges them). ``FastReaderSparkAdapter`` changes only
   the reader: with ``SPARK_FAST_READER=1`` (the default) the base weights
   are read with ``tpu_knobs.FastHFStorageReader``, one positional read per
   contiguous slice; ``SPARK_FAST_READER=0`` keeps the base adapter's reader.
2. Checkpoints. A LoRA checkpoint holds the adapters, their optimizer
   state, the LR schedule, the dataloader position and the step counter,
   not the frozen 1.64 TB base (``checkpointing.lora_checkpoint.
   enable_adapter_only_checkpoints``, installed before the first load).
   Every start reads the base from the Hugging Face checkpoint and then, if
   ``<dump>/checkpoint`` holds a complete ``step-N``, restores the newest
   one, so relaunching a killed run with the same dump folder resumes it.
   The trainer also writes the adapter in the PEFT layout
   (``<dump>/adapter/step-N``, see ``peft_adapter``) at every checkpoint
   step, at the last step and, with ``SPARK_EXPORT_EVERY=N``, every N steps.
   ``SPARK_ADAPTER_INIT=<dir>`` starts from such an adapter (applied only
   when no adapter checkpoint was restored).

All collectives walk keys in sorted order on every rank.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from torchtitan.experiments.tpu.checkpointing.lora_checkpoint import enable_adapter_only_checkpoints
from torchtitan.experiments.tpu.muse_spark import state_dict_adapter as spark_sda
from torchtitan.experiments.tpu.post_training.lora import is_lora_key
from torchtitan.experiments.tpu.post_training.sft_train import SFTTPUTrainer

from .peft_adapter import adapter_spec_from_model, gather_peft_adapter, load_peft_into_model, save_peft_dir

logger = logging.getLogger(__name__)


class FastReaderSparkAdapter(spark_sda.MuseSparkStateDictAdapter):
    """``MuseSparkStateDictAdapter`` that reads the checkpoint with the fast reader.

    ``SPARK_FAST_READER=1`` (default): ``tpu_knobs.FastHFStorageReader``.
    ``SPARK_FAST_READER=0``, or a model whose expert rows use a permuted
    placement: the base adapter's reader (which applies the placement).

    The 64-chip step recipes (``muse_spark/spark_64chip_recipes.py``) have a
    placement-aware variant of this reader, but importing that module patches
    TorchTitan's checkpoint and metrics classes for the whole process, so the
    SFT path keeps its own reader, the one its full-model TPU run used.
    """

    def get_hf_storage_reader(self, path, from_quantized=False):
        if os.environ.get("SPARK_FAST_READER", "1") == "1" and not self.permutes_experts:
            from .tpu_knobs import fast_reader

            reader = fast_reader(path, from_quantized)
        else:
            reader = super().get_hf_storage_reader(path, from_quantized)
        if int(os.environ.get("RANK", "0")) == 0:
            logger.info("[spark] Hugging Face checkpoint reader: %s", type(reader).__name__)
        return reader


def has_lora(model: torch.nn.Module) -> bool:
    return any(is_lora_key(name) for name, _ in model.named_parameters())


def _export_every() -> int:
    """``SPARK_EXPORT_EVERY``: also export the PEFT adapter every N steps (0 = off)."""
    value = int(os.environ.get("SPARK_EXPORT_EVERY", "0") or 0)
    if value < 0:
        raise ValueError(f"SPARK_EXPORT_EVERY must be >= 0, got {value}")
    return value


class _AdapterExportMixin:
    """Adapter-only checkpoints, optional adapter init, PEFT export."""

    def _spark_lora_setup(self) -> None:
        """Called at the end of ``__init__``: before ``train()`` loads checkpoints."""
        self._spark_lora = has_lora(self.model_parts[0])
        self._spark_rank = int(os.environ.get("RANK", "0"))
        self._spark_adapter_exports: list[dict[str, Any]] = []
        self._spark_started = False
        if self._spark_lora:
            self._spark_spec = adapter_spec_from_model(self.model_parts[0])
            if getattr(self.checkpointer, "enable", False):
                # Saves: adapters + optimizer + LR schedule + dataloader + train
                # state. Loads: HF base, then the newest complete step-N.
                enable_adapter_only_checkpoints(self.checkpointer, self.model_parts)
                logger.info("[spark] checkpoints hold the LoRA adapters and their training state only")

    def _spark_first_step(self) -> None:
        """After the checkpoint load, before the first update."""
        if self._spark_started:
            return
        self._spark_started = True
        if not self._spark_lora:
            return
        resumed = getattr(self.checkpointer, "adapter_resume_step", None)
        init = os.environ.get("SPARK_ADAPTER_INIT", "")
        if init and resumed is not None:
            logger.info("[spark] resumed from adapter checkpoint step %d; SPARK_ADAPTER_INIT=%s ignored",
                        resumed, init)
        elif init:
            info = load_peft_into_model(self.model_parts[0], init)
            logger.info("[spark] adapter initialized from %s", info)

    def _spark_export_adapter(self, step: int, *, tag: str = "step") -> dict[str, Any] | None:
        if not self._spark_lora:
            return None
        t0 = time.perf_counter()
        keep = self._spark_rank == 0
        tensors = gather_peft_adapter(self.model_parts, keep=keep)
        info = None
        if keep:
            out = Path(self.config.dump_folder) / "adapter" / f"{tag}-{step}"
            info = {"step": step, "dir": str(out), "tensors": len(tensors),
                    "params": int(sum(t.numel() for t in tensors.values())),
                    "seconds": round(time.perf_counter() - t0, 3)}
            save_peft_dir(out, tensors, self._spark_spec, base=str(self.config.hf_assets_path),
                          extra=info)
            logger.info("[spark] PEFT adapter exported %s", info)
            self._spark_adapter_exports.append(info)
        if dist.is_initialized():
            dist.barrier()
        return info


class SparkSFTTrainer(_AdapterExportMixin, SFTTPUTrainer):
    """SFT trainer: LoRA-only checkpoints, each with a PEFT export
    (``<dump>/adapter/step-N``) at the checkpoint interval and the last step.

    With ``SPARK_EXPORT_EVERY=N`` the adapter is also exported every N steps,
    so intermediate policies can be evaluated while training continues. Every
    rank takes the same decision (the export is a collective).
    """

    def __init__(self, job_config: Any) -> None:
        super().__init__(job_config)
        self._spark_lora_setup()
        self._spark_export_every = _export_every()

    def train_step(self, data_iterator):
        self._spark_first_step()
        result = super().train_step(data_iterator)
        every = self._spark_export_every
        last = self.step == self.config.training.steps
        if last or self.checkpointer._should_save(self.step) or (every and self.step % every == 0):
            self._spark_export_adapter(self.step)
        return result
