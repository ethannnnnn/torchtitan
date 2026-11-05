# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Make every rank leave a DCP save together on TorchTPU.

TorchTPU defers device work until a host readback. The last collective of a
DCP save (the coordinator's broadcast of the save result) stays queued on rank
0, which returns from ``save`` at once, while ranks 1..N-1 block on that
broadcast until rank 0's next readback, typically inside the next training
step. Two consequences, both seen on TPU7x with the full Muse Spark model:

* the save time logged on rank 0 under-reports, and the next step absorbs the
  difference;
* if rank 0 dies between the save and its next readback (a crash, a
  preemption, a kill), every other rank hangs in the save until the
  collective timeout, although the checkpoint on storage is complete.

``flush_after_saves`` wraps ``CheckpointManager.save`` so that after every
save that wrote, each rank all-reduces one int32 on the device and reads it
back. That executes the queued broadcast in order and returns only once every
rank has joined. It changes no numerics and no ordering.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

__all__ = ["device_flush", "flush_after_saves"]


def device_flush(device: Any) -> float:
    """One SUM all-reduce of an int32 on ``device``, read back; returns seconds."""
    start = time.perf_counter()
    if dist.is_initialized() and dist.get_world_size() > 1:
        token = torch.ones(1, dtype=torch.int32, device=device)
        dist.all_reduce(token, op=dist.ReduceOp.SUM)
        joined = int(token.cpu().item())
        if joined != dist.get_world_size():
            raise RuntimeError(f"device flush joined {joined} of {dist.get_world_size()} ranks")
    return time.perf_counter() - start


def flush_after_saves(checkpointer: Any, device: Any) -> None:
    """Wrap ``checkpointer.save`` to flush every rank after a save that wrote.

    Every rank takes the same branch: whether a save writes depends only on
    the step, the interval and ``last_step``.
    """
    if not getattr(checkpointer, "enable", False) or getattr(checkpointer.save, "_flushes", False):
        return
    save = checkpointer.save

    def _save(curr_step: int, last_step: bool = False) -> bool:
        saved = save(curr_step, last_step=last_step)
        if saved:
            seconds = device_flush(device)
            logger.info("[ckpt] step %d: every rank left the save (flush %.3f s)", curr_step, seconds)
        return saved

    _save._flushes = True
    checkpointer.save = _save
