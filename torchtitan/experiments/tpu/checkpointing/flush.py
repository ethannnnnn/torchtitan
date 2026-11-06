# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Drain the TorchTPU device queue on every rank after a checkpoint save.

TorchTPU executes device work lazily: operations issued by the host are queued
and only run when the host next has to read a value back from the device.

Distributed checkpoint (DCP) saves are affected by this. A DCP save ends with a
collective through which the coordinating rank, rank 0, hands the outcome of
the save to all other ranks. On TorchTPU, rank 0 can return from ``save``
before that collective has executed on its device, and the other ranks then
wait inside it until something makes rank 0's device run, which in practice
happens somewhere in the next training step. Two things follow:

* the save time reported by rank 0 is too short, and the missing part is
  charged to the following step;
* if rank 0 dies inside that window, the other ranks stay blocked until the
  collective times out, although every checkpoint file was already written.

``device_flush`` closes the window. Each rank adds up a single integer over
all ranks with a collective on its device and copies the total to the host.
The host read executes everything queued before it, in order, and completes
only once every rank has arrived. ``flush_after_saves`` runs it after each
``save`` call that wrote a checkpoint::

    from torchtitan.experiments.tpu.checkpointing.flush import flush_after_saves

    flush_after_saves(checkpointer, device)  # on every rank

The flush only adds a wait at the end of such saves. Trained values,
checkpoint contents and the order of the save's own work are unchanged.

Both entry points end in a collective on the default process group: call
``device_flush`` on all of its ranks at the same point, and install the save
hook on all ranks or on none. Ranks that reach the collective alone block
until the group's timeout.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

__all__ = ["device_flush", "flush_after_saves"]


def _ranks_in_default_group() -> int:
    """Number of ranks a flush waits for: 1 without an initialized process group."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def device_flush(device: torch.device | str) -> float:
    """Execute this rank's queued device work and wait until all ranks did the same.

    With more than one rank in the default process group, a one-element
    ``int32`` tensor holding 1 is created on ``device``, summed over the group
    with ``all_reduce`` and copied to the host; reading it there is what makes
    the device run its queue. A sum other than the group size raises
    ``RuntimeError``. With no initialized group, or a single rank, nothing else
    happens and ``device`` is not used.

    Every rank of the default group has to call this at the same point (it is a
    collective). Errors and timeouts from ``torch.distributed`` propagate as
    they are.

    Returns:
        Seconds spent in this call.
    """
    entered = time.perf_counter()
    ranks = _ranks_in_default_group()
    if ranks > 1:
        tally = torch.full((1,), 1, dtype=torch.int32, device=device)
        dist.all_reduce(tally, op=dist.ReduceOp.SUM)
        (arrived,) = tally.cpu().tolist()
        if arrived != ranks:
            raise RuntimeError(
                f"device flush: the all_reduce over the default process group summed to "
                f"{arrived}, but the group has {ranks} ranks"
            )
    return time.perf_counter() - entered


class _FlushOnWrite:
    """Instance-level ``save`` for one checkpoint manager: delegate, then flush if it wrote."""

    def __init__(self, delegate: Callable[..., Any], device: torch.device | str) -> None:
        self.delegate = delegate
        self.device = device

    def __call__(self, curr_step: int, last_step: bool = False) -> Any:
        outcome = self.delegate(curr_step, last_step=last_step)
        if outcome:
            # ``device_flush`` is looked up in the module namespace on each call,
            # so patching the module attribute intercepts this flush as well.
            seconds = device_flush(self.device)
            logger.info("Checkpoint step %d: device queues drained on all ranks in %.3f s", curr_step, seconds)
        return outcome


def flush_after_saves(checkpointer: Any, device: torch.device | str) -> None:
    """Run ``device_flush(device)`` after each ``checkpointer.save`` call that wrote a checkpoint.

    ``save`` is replaced on this instance only; the class and other instances
    keep theirs. The replacement has the call contract of TorchTitan's
    ``CheckpointManager.save(curr_step, last_step=False)``. It forwards both
    arguments to the ``save`` bound on the instance when this function runs,
    which may itself be an earlier instance-level override, and returns that
    call's result unchanged. When the result is truthy it then flushes and logs
    the step and the flush time. A manager whose ``enable`` is missing or falsy
    is left alone, and installing twice keeps the first wrapper.

    TorchTitan's managers decide whether a call writes from the step arguments
    and the checkpoint configuration (``enable``, ``load_only``, the interval
    and the first-step setting), all of which agree across ranks, so every rank
    flushes after the same calls, provided every rank installed the hook: call
    this on all ranks or on none. Asynchronous modes report a save once it is
    staged, so there the flush follows the staging step.

    An exception from the wrapped ``save`` propagates and skips the flush. An
    exception from the flush propagates after the save itself has finished.
    """
    if not getattr(checkpointer, "enable", False):
        return
    if isinstance(checkpointer.save, _FlushOnWrite):
        return
    checkpointer.save = _FlushOnWrite(checkpointer.save, device)
