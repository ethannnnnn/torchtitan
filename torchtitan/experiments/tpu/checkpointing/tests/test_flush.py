# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for ``flush``: every rank leaves a checkpoint save together."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torchtitan.experiments.tpu.checkpointing import flush


class _FakeCheckpointer:
    """Writes on even steps and on the last step, like a manager with interval 2."""

    def __init__(self, enable: bool = True):
        self.enable = enable
        self.calls = []

    def save(self, curr_step: int, last_step: bool = False) -> bool:
        self.calls.append((curr_step, last_step))
        return last_step or curr_step % 2 == 0


def _flush_worker(rank: int, world: int, store: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
    try:
        assert flush.device_flush(torch.device("cpu")) >= 0.0
        checkpointer = _FakeCheckpointer()
        flush.flush_after_saves(checkpointer, torch.device("cpu"))
        assert [checkpointer.save(step) for step in (1, 2)] == [False, True]
    finally:
        dist.destroy_process_group()


class FlushTest(unittest.TestCase):
    def test_flushes_only_after_saves_that_wrote(self):
        checkpointer = _FakeCheckpointer()
        with mock.patch.object(flush, "device_flush", return_value=0.0) as device_flush:
            flush.flush_after_saves(checkpointer, "cpu")
            saved = [checkpointer.save(step) for step in range(1, 5)] + [checkpointer.save(5, last_step=True)]
        self.assertEqual(saved, [False, True, False, True, True])
        self.assertEqual(device_flush.call_args_list, [mock.call("cpu")] * 3)
        self.assertEqual(checkpointer.calls[-1], (5, True))

    def test_wrapping_is_idempotent_and_skips_disabled_managers(self):
        checkpointer = _FakeCheckpointer()
        flush.flush_after_saves(checkpointer, "cpu")
        wrapped = checkpointer.save
        flush.flush_after_saves(checkpointer, "cpu")
        self.assertIs(checkpointer.save, wrapped)
        disabled = _FakeCheckpointer(enable=False)
        flush.flush_after_saves(disabled, "cpu")
        self.assertEqual(disabled.save.__func__, _FakeCheckpointer.save)

    def test_single_process_flush_is_a_noop(self):
        self.assertGreaterEqual(flush.device_flush(torch.device("cpu")), 0.0)

    def test_every_rank_joins_the_flush(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(_flush_worker, args=(2, os.path.join(tmp, "store")), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
