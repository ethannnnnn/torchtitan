# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the post-save device flush (``device_flush`` and ``flush_after_saves``)."""

from __future__ import annotations

import contextlib
import datetime
import json
import os
import tempfile
import unittest
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torchtitan.experiments.tpu.checkpointing import flush

FLUSH_SECONDS = 0.25  # what the patched device_flush reports


class _Manager:
    """Checkpoint manager double with the ``save(curr_step, last_step=False)`` signature of TorchTitan's managers.

    ``results`` maps a step to the object ``save`` returns for it (``False`` for
    unlisted steps). Every finished call is recorded in ``calls`` and appended to
    ``events``, a list that a test can share with a fake flush to check ordering.
    """

    enable = True

    def __init__(self, results: dict | None = None, events: list | None = None) -> None:
        self.results = dict(results or {})
        self.events = [] if events is None else events
        self.calls: list[tuple] = []

    def save(self, curr_step, last_step=False):
        self.calls.append((curr_step, last_step))
        self.events.append(("save", curr_step, last_step))
        return self.results.get(curr_step, False)


def _fake_flush(events: list):
    """A ``device_flush`` stand-in that records its arguments in ``events``."""

    def fake(*args, **kwargs):
        events.append(("flush", args, kwargs))
        return FLUSH_SECONDS

    return fake


@contextlib.contextmanager
def _simulated_group(world_size: int, all_reduce):
    """Pretend an initialized default process group of ``world_size`` ranks; ``all_reduce`` replaces the collective."""
    with mock.patch.object(dist, "is_available", return_value=True), \
            mock.patch.object(dist, "is_initialized", return_value=True), \
            mock.patch.object(dist, "get_world_size", return_value=world_size), \
            mock.patch.object(dist, "all_reduce", side_effect=all_reduce) as reduce:
        yield reduce


class FlushAfterSavesTest(unittest.TestCase):
    def test_flush_follows_each_writing_save_only(self):
        """Results pass through as the same objects; one flush with the install-time device per writing call."""
        events = []
        written, staged, empty = object(), ["staged"], []
        results = {1: False, 2: written, 3: None, 4: empty, 5: staged, 6: True}
        manager = _Manager(results, events)
        class_save = vars(_Manager)["save"]
        flush.flush_after_saves(manager, "fake-device:3")  # installed before the patch below
        with mock.patch.object(flush, "device_flush", side_effect=_fake_flush(events)) as spy, \
                self.assertLogs(flush.logger, level="INFO") as logs:
            returned = [manager.save(step, last_step=step == 6) for step in range(1, 7)]
        for step, value in zip(range(1, 7), returned):
            self.assertIs(value, results[step])
        self.assertEqual(manager.calls, [(step, step == 6) for step in range(1, 7)])
        self.assertEqual(spy.call_count, 3)
        flushed = ("flush", ("fake-device:3",), {})
        self.assertEqual(events, [("save", 1, False), ("save", 2, False), flushed, ("save", 3, False),
                                  ("save", 4, False), ("save", 5, False), flushed, ("save", 6, True), flushed])
        self.assertEqual([record.levelname for record in logs.records], ["INFO"] * 3)
        for record, step in zip(logs.records, (2, 5, 6)):
            self.assertRegex(record.getMessage(), rf"\b{step}\b")
            self.assertIn(f"{FLUSH_SECONDS:.3f}", record.getMessage())
        self.assertIs(vars(_Manager)["save"], class_save)  # the class keeps its method
        self.assertNotIn("save", vars(_Manager()))  # and other instances are not wrapped

    def test_upstream_call_forms(self):
        """Training loop, seed checkpoint and test call forms reach save with the same step and flag."""
        manager = _Manager({0: True, 3: True, 4: True, 5: True, 7: True})
        flush.flush_after_saves(manager, "cpu")
        with mock.patch.object(flush, "device_flush", return_value=FLUSH_SECONDS) as spy:
            manager.save(3)
            manager.save(4, last_step=True)
            manager.save(curr_step=0, last_step=True)
            manager.save(curr_step=5)
            manager.save(7, True)
            manager.save(8, last_step=False)
        self.assertEqual(manager.calls, [(3, False), (4, True), (0, True), (5, False), (7, True), (8, False)])
        self.assertEqual(spy.call_args_list, [mock.call("cpu")] * 5)

    def test_second_install_keeps_the_first_wrapper(self):
        """Installing twice leaves the same wrapper bound, and a writing save still flushes once."""
        manager = _Manager({2: True})
        with self.assertNoLogs(flush.logger, level="DEBUG"):
            self.assertIsNone(flush.flush_after_saves(manager, "first"))
            installed = manager.save
            self.assertIsNone(flush.flush_after_saves(manager, "second"))
        self.assertIs(manager.save, installed)
        with mock.patch.object(flush, "device_flush", return_value=FLUSH_SECONDS) as spy:
            self.assertIs(manager.save(2), True)
        spy.assert_called_once_with("first")
        self.assertEqual(manager.calls, [(2, False)])

    def test_an_earlier_instance_override_stays_in_the_chain(self):
        """The wrapper delegates to the save bound on the instance, not to the class method."""
        manager = _Manager({9: True})
        seen = []

        def adapter_only_save(curr_step, last_step=False):
            seen.append((curr_step, last_step))
            return "adapter written"

        manager.save = adapter_only_save
        flush.flush_after_saves(manager, "cpu")
        self.assertIsNot(manager.save, adapter_only_save)
        with mock.patch.object(flush, "device_flush", return_value=FLUSH_SECONDS) as spy:
            self.assertEqual(manager.save(9, last_step=True), "adapter written")
        self.assertEqual(seen, [(9, True)])
        self.assertEqual(manager.calls, [])
        spy.assert_called_once_with("cpu")

    def test_disabled_or_unknown_objects_are_left_alone(self):
        """A falsy or missing ``enable``: nothing is added and save still resolves to the class method."""

        class Disabled(_Manager):
            enable = False

        class NoEnable:
            def save(self, curr_step, last_step=False):
                return True

        zero = _Manager()
        zero.enable = 0
        for obj in (Disabled(), NoEnable(), zero):
            with self.subTest(kind=type(obj).__name__, enable=getattr(obj, "enable", "missing")):
                before = dict(vars(obj))
                self.assertIsNone(flush.flush_after_saves(obj, "cpu"))
                self.assertEqual(vars(obj), before)
                self.assertNotIn("save", vars(obj))
                self.assertIs(obj.save.__func__, type(obj).save)

    def test_save_error_propagates_without_a_flush(self):
        manager = _Manager()

        def failing_save(curr_step, last_step=False):
            raise OSError(f"no space left for step {curr_step}")

        manager.save = failing_save
        flush.flush_after_saves(manager, "cpu")
        with mock.patch.object(flush, "device_flush") as spy, self.assertRaisesRegex(OSError, "step 10"):
            manager.save(10)
        spy.assert_not_called()

    def test_flush_error_propagates_after_the_save(self):
        events = []
        manager = _Manager({4: True}, events)
        flush.flush_after_saves(manager, "cpu")

        def peer_lost(device):
            events.append(("flush", device))
            raise RuntimeError("peer lost")

        with mock.patch.object(flush, "device_flush", side_effect=peer_lost), \
                self.assertRaisesRegex(RuntimeError, "peer lost"):
            manager.save(4)
        self.assertEqual(events, [("save", 4, False), ("flush", "cpu")])


class DeviceFlushTest(unittest.TestCase):
    def test_returns_early_without_peers(self):
        """No usable process group, or a single rank: no collective, no logging, and the device is never used."""
        unusable = object()
        with self.assertRaises((TypeError, RuntimeError)):
            torch.ones(1, device=unusable)  # using this "device" would fail
        cases = {
            # No group in a fresh process; a one-rank group if another test module created one earlier.
            "current process": [],
            "distributed unavailable": [mock.patch.object(dist, "is_available", return_value=False)],
            "not initialized": [mock.patch.object(dist, "is_initialized", return_value=False)],
            "one rank": [mock.patch.object(dist, "is_available", return_value=True),
                         mock.patch.object(dist, "is_initialized", return_value=True),
                         mock.patch.object(dist, "get_world_size", return_value=1)],
        }
        for name, patches in cases.items():
            with self.subTest(name), contextlib.ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                reduce = stack.enter_context(mock.patch.object(dist, "all_reduce"))
                stack.enter_context(self.assertNoLogs(flush.logger, level="DEBUG"))
                seconds = flush.device_flush(unusable)
                reduce.assert_not_called()
                self.assertIsInstance(seconds, float)
                self.assertGreaterEqual(seconds, 0.0)

    def test_one_int32_one_is_summed_on_the_device(self):
        """The collective gets a one-element int32 tensor holding 1 on ``device``, summed over the default group."""
        seen = []

        def all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
            seen.append((tensor.dtype, tuple(tensor.shape), tensor.device, tensor.tolist(), op, group, async_op))
            tensor.fill_(16)

        with _simulated_group(16, all_reduce) as reduce:
            seconds = flush.device_flush("cpu")
        self.assertEqual(reduce.call_count, 1)
        self.assertEqual(seen, [(torch.int32, (1,), torch.device("cpu"), [1], dist.ReduceOp.SUM, None, False)])
        self.assertIsInstance(seconds, float)
        self.assertGreaterEqual(seconds, 0.0)

    def test_wrong_count_raises(self):
        """A sum other than the world size raises RuntimeError naming both numbers."""

        def short_by_three(tensor, *args, **kwargs):
            tensor.fill_(13)

        with _simulated_group(16, short_by_three), self.assertRaises(RuntimeError) as caught:
            flush.device_flush("cpu")
        self.assertRegex(str(caught.exception), r"\b13\b")
        self.assertRegex(str(caught.exception), r"\b16\b")


def _gloo_rank(rank: int, workdir: str) -> None:
    """One rank of ``TwoProcessGlooTest`` (module level, so that ``mp.spawn`` can pickle it)."""
    dist.init_process_group("gloo", init_method="file://" + os.path.join(workdir, "store"), rank=rank,
                            world_size=2, timeout=datetime.timedelta(seconds=30))
    try:
        seconds = flush.device_flush(torch.device("cpu"))
        manager = _Manager({2: True})
        flush.flush_after_saves(manager, "cpu")
        with mock.patch.object(flush, "device_flush", wraps=flush.device_flush) as spy:
            returned = [manager.save(1), manager.save(2, last_step=True)]
        report = {"seconds": seconds, "returned": returned, "flushes": spy.call_count, "calls": manager.calls}
        with open(os.path.join(workdir, f"rank-{rank}.json"), "w") as f:
            json.dump(report, f)
    finally:
        dist.destroy_process_group()


class TwoProcessGlooTest(unittest.TestCase):
    def test_real_collective(self):
        """Two gloo ranks run the real collective, directly and through a wrapped manager."""
        with tempfile.TemporaryDirectory() as workdir:
            mp.spawn(_gloo_rank, args=(workdir,), nprocs=2, join=True)  # raises if a rank fails
            reports = []
            for rank in range(2):
                with open(os.path.join(workdir, f"rank-{rank}.json")) as f:
                    reports.append(json.load(f))
        for rank, report in enumerate(reports):
            with self.subTest(rank=rank):
                self.assertIsInstance(report["seconds"], float)
                self.assertGreaterEqual(report["seconds"], 0.0)
                self.assertEqual(report["returned"], [False, True])
                self.assertEqual(report["flushes"], 1)
                self.assertEqual(report["calls"], [[1, False], [2, True]])


if __name__ == "__main__":
    unittest.main()
