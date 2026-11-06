# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Drain each TorchTPU rank's queued device work right after a save that wrote a checkpoint.

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

``device_flush`` closes that window. Each rank feeds the integer 1 into a
group-wide sum that a collective computes on the rank's own device, then
copies the result to the host. That copy makes the device execute its backlog
in the order it was issued, and it cannot complete before the whole group has
added its share. ``flush_after_saves`` hooks this onto a checkpoint manager so
that it runs whenever ``save`` reports a written checkpoint::

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

import dataclasses
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
    """Synchronous barrier issued on ``device``; returns the seconds it took.

    When the default process group has two or more ranks, the call

    1. creates a one-element ``torch.int32`` tensor holding 1 on ``device``,
    2. sums it over the default group with a blocking ``all_reduce``, and
    3. fetches the reduced value to the host with ``Tensor.item()``.

    Step 3 does the work. On TorchTPU, producing that host value means first
    running everything submitted to the device before it, in submission
    order, and the value only exists once the whole group has contributed to
    the sum. TorchTPU's process group uses the same recipe for its own
    synchronous barrier. ``torch.distributed.barrier()`` would not do: its
    backend is picked from the device types the group was created with, and
    when the default group combines a TorchTPU backend with a CPU one, the
    barrier is served by the CPU backend and the device queue stays as it is.
    The fetched value is not inspected.

    Without an initialized process group, or with a single rank, nothing is
    created or communicated and ``device`` is never used.

    It is a collective, so all ranks of the default group must reach it
    together; a rank that arrives alone hangs until the process group times
    out. Errors and timeouts raised by ``torch.distributed`` propagate
    unchanged. The returned wall time is never negative.
    """
    start_ns = time.perf_counter_ns()
    if _ranks_in_default_group() > 1:
        probe = torch.ones(1, dtype=torch.int32, device=device)
        dist.all_reduce(probe)  # the default reduction is SUM
        probe.item()
    return (time.perf_counter_ns() - start_ns) / 1e9


@dataclasses.dataclass(frozen=True)
class _SaveThenFlush:
    """Instance-level ``save`` installed by ``flush_after_saves``."""

    inner_save: Callable[..., Any]
    device: torch.device | str

    def __call__(self, curr_step: int, last_step: bool = False) -> Any:
        result = self.inner_save(curr_step, last_step=last_step)
        if not result:
            return result
        logger.info(
            "Checkpoint of step %s written; device queue drained in %.3f s", curr_step, device_flush(self.device)
        )
        return result


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
    if isinstance(checkpointer.save, _SaveThenFlush):
        return
    checkpointer.save = _SaveThenFlush(checkpointer.save, device)
