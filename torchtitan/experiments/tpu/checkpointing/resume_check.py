# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test hooks for checkpoint / resume checks of TorchTitan trainers.

``install()`` patches ``torchtitan.trainer.Trainer`` only when the environment
asks for it; otherwise it changes nothing:

* ``TT_FAULT_KILL_STEP=N``: the process sends itself SIGKILL when step N draws
  its first microbatch. Every rank does the same, so the job dies after step
  N's batches were drawn and before anything of step N is computed or saved:
  a hard crash for restart tests (relaunch with the same dump folder).
* ``TT_BATCH_RECORDS_DIR=<dir>``: every rank appends one JSON line per step to
  ``<dir>/rank-NNNNN.jsonl`` with the SHA-256 of every microbatch it drew (all
  tensors of the batch, in key order), so two runs can be compared batch by
  batch across a restart.

Rank 0 also logs ``[resume-check] step=N loss=... grad_norm=...`` with full
float precision whenever the trainer logs metrics (``TT_BATCH_RECORDS_DIR``
set), for bitwise loss comparisons.

The hooks live in ``Trainer.batch_generator`` and ``Trainer.train_step``, so a
subclass gets them only through those methods: one that overrides
``batch_generator`` without calling the base method is never killed and
hashes no batches, and one that overrides ``train_step`` that way (e.g.
``TpuGraphTrainer`` with the optimizer in the graph) writes no batch records.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["install", "batch_sha256"]


def batch_sha256(batch: Mapping[str, Any]) -> str:
    """Hex SHA-256 over the tensors of ``batch``, visited in sorted key order.

    Each tensor contributes its key (UTF-8) followed by its elements as raw
    bytes, row-major from a detached CPU copy. Other values are skipped, so
    the digest ignores insertion order and non-tensor entries. Records from
    earlier runs stay comparable only while this byte layout is unchanged.
    """
    import torch

    h = hashlib.sha256()
    for name in sorted(batch):
        value = batch[name]
        if not isinstance(value, torch.Tensor):
            continue
        t = value.detach().cpu().contiguous().view(-1).view(torch.uint8)
        raw_bytes = t.numpy().tobytes()
        h.update(name.encode("utf-8"))
        h.update(raw_bytes)
    return h.hexdigest()


def install() -> bool:
    """Install the hooks the environment asks for; returns whether any was."""
    kill_step = int(os.environ.get("TT_FAULT_KILL_STEP", "0") or 0)
    records_dir = os.environ.get("TT_BATCH_RECORDS_DIR", "")
    if not kill_step and not records_dir:
        return False
    from torchtitan.components.metrics import MetricsProcessor
    from torchtitan.trainer import Trainer

    if getattr(Trainer.batch_generator, "_rc_hook", False):
        return True
    rank = int(os.environ.get("RANK", "0"))
    orig_batches = Trainer.batch_generator
    orig_step = Trainer.train_step

    def batch_generator(self, data_iterable):
        for batch in orig_batches(self, data_iterable):
            if kill_step and self.step == kill_step:
                logger.warning("[resume-check] SIGKILL at step %d (rank %d)", self.step, rank)
                for handler in logging.getLogger().handlers:
                    handler.flush()
                os.kill(os.getpid(), signal.SIGKILL)
            if records_dir:
                self._resume_check_batches = getattr(self, "_resume_check_batches", [])
                self._resume_check_batches.append(batch_sha256(batch))
            yield batch

    def train_step(self, data_iterator: Iterator[dict[str, Any]]):
        self._resume_check_batches = []
        out = orig_step(self, data_iterator)
        if records_dir:
            root = Path(records_dir)
            root.mkdir(parents=True, exist_ok=True)
            line = {"step": self.step, "rank": rank, "batches": self._resume_check_batches}
            with open(root / f"rank-{rank:05d}.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(line, sort_keys=True) + "\n")
        return out

    batch_generator._rc_hook = True
    Trainer.batch_generator = batch_generator
    Trainer.train_step = train_step

    if records_dir and not getattr(MetricsProcessor.log, "_rc_hook", False):
        orig_log = MetricsProcessor.log

        def log(self, step, global_avg_loss, global_max_loss, grad_norm, *args, **kwargs):
            if rank == 0:
                logger.info("[resume-check] step=%d loss=%r max_loss=%r grad_norm=%r", step,
                            float(global_avg_loss), float(global_max_loss), float(grad_norm))
            return orig_log(self, step, global_avg_loss, global_max_loss, grad_norm, *args, **kwargs)

        log._rc_hook = True
        MetricsProcessor.log = log
    if rank == 0:
        logger.warning("[resume-check] installed: kill_step=%s records_dir=%s", kill_step or None,
                       records_dir or None)
    return True
