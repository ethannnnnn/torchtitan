# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the multi-threaded DCP writer and the parallel DCP readers."""

from __future__ import annotations

import io
import os
import shutil
import tempfile
import unittest
from unittest import mock

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemWriter

from torchtitan.experiments.tpu.checkpointing import dcp_parallel
from torchtitan.experiments.tpu.checkpointing.tests import fixtures


def _state(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "a": torch.randn(64, 32, generator=g),
        "b": torch.randint(0, 1 << 40, (100,), generator=g),
        "c": torch.randn(7, 3, 5, generator=g).to(torch.bfloat16),
        "nested": {"x": torch.randn(10, generator=g), "count": 3 + seed, "name": f"run-{seed}"},
    }


def _zeros_like(state: dict) -> dict:
    return {k: _zeros_like(v) if isinstance(v, dict) else torch.zeros_like(v) if isinstance(v, torch.Tensor)
            else None for k, v in state.items()}


def _assert_same(test: unittest.TestCase, a, b, path: str = "") -> None:
    """Bitwise equality of nested state (dicts, lists, tensors, scalars)."""
    if isinstance(a, dict):
        test.assertEqual(sorted(a), sorted(b), path)
        for key in a:
            _assert_same(test, a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b), path)
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_same(test, x, y, f"{path}[{i}]")
    elif isinstance(a, torch.Tensor):
        test.assertTrue(torch.equal(a, b), path)
    else:
        test.assertEqual(a, b, path)


class ParallelReaderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="dcp-parallel-test-")
        self.path = os.path.join(self.tmp, "ckpt")
        self.state = _state()
        dcp.save(self.state, storage_writer=FileSystemWriter(self.path, thread_count=4))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _check(self, loaded: dict, want: dict) -> None:
        for key, value in want.items():
            if isinstance(value, dict):
                self._check(loaded[key], value)
            elif isinstance(value, torch.Tensor):
                self.assertTrue(torch.equal(loaded[key], value), key)
            else:
                self.assertEqual(loaded[key], value, key)

    def test_writer_threads(self):
        files = [f for f in os.listdir(self.path) if f.endswith(".distcp")]
        self.assertEqual(len(files), 4, files)

    def test_parallel_filesystem_reader_bit_exact(self):
        for threads in (1, 3, 16):
            with self.subTest(threads=threads):
                fresh = _zeros_like(self.state)
                dcp.load(fresh, storage_reader=dcp_parallel.ParallelFileSystemReader(self.path, read_threads=threads))
                self._check(fresh, self.state)

    def test_parallel_fsspec_reader_bit_exact(self):
        fresh = _zeros_like(self.state)
        dcp.load(fresh, storage_reader=dcp_parallel.ParallelFsspecReader("file://" + self.path, read_threads=3))
        self._check(fresh, self.state)

    def test_short_read_raises(self):
        self.assertEqual(dcp_parallel._read_exact(io.BytesIO(b"abcdef"), 2, 3), b"cde")
        with self.assertRaises(EOFError):
            dcp_parallel._read_exact(io.BytesIO(b"abc"), 1, 5)


class TuneCheckpointManagerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.ensure_process_group()

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="dcp-tune-test-")

    def tearDown(self):
        fixtures.restore_checkpointer_dcp()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _manager(self, seed: int):
        from torchtitan.components.checkpointer.dcp import CheckpointManager
        from torchtitan.components.optimizer.lr_scheduler import LRSchedulersContainer
        from torchtitan.components.optimizer.optimizer import OptimizersContainer, ParamGroupConfig

        torch.manual_seed(seed)
        model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Linear(16, 4))
        optimizers = OptimizersContainer.Config(
            param_groups=[ParamGroupConfig(pattern=".*", optimizer_name="AdamW", optimizer_kwargs={"lr": 1e-2})],
            implementation="for-loop",
        ).build(model_parts=[model])
        lr_schedulers = LRSchedulersContainer.Config(warmup_steps=1).build(optimizers=optimizers, training_steps=4)
        train_state = fixtures.Counter()
        checkpointer = CheckpointManager(
            CheckpointManager.Config(enable=True, folder="checkpoint", interval=1, last_save_model_only=False,
                                     keep_latest_k=0),
            dataloader=fixtures.Counter(), model_parts=[model], optimizers=optimizers,
            lr_schedulers=lr_schedulers, states={"train_state": train_state}, sd_adapter=None, base_folder=self.tmp,
        )
        return model, optimizers, lr_schedulers, train_state, checkpointer

    def test_tuned_save_and_load(self):
        model, optimizers, lr_schedulers, train_state, ckpt = self._manager(seed=0)
        dcp_parallel.tune_checkpoint_manager(ckpt, save_threads=3, load_threads=2)
        optimizers.zero_grad()
        model(torch.randn(5, 8)).square().mean().backward()
        optimizers.step()
        lr_schedulers.step()
        train_state.step = 1
        self.assertTrue(ckpt.save(curr_step=1))
        step1 = os.path.join(self.tmp, "checkpoint", "step-1")
        self.assertEqual(len([f for f in os.listdir(step1) if f.endswith(".distcp")]), 3)

        model_b, optimizers_b, _, train_state_b, ckpt_b = self._manager(seed=1)
        dcp_parallel.tune_checkpoint_manager(ckpt_b, save_threads=3, load_threads=2)
        readers = []

        class CountingReader(dcp_parallel.ParallelFileSystemReader):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                readers.append(self.read_threads)

        with mock.patch.object(dcp_parallel, "ParallelFileSystemReader", CountingReader):
            self.assertTrue(ckpt_b.load())
        self.assertEqual(readers, [2])
        self.assertEqual(train_state_b.step, 1)
        for (name, p), p_b in zip(model.named_parameters(), model_b.parameters()):
            self.assertTrue(torch.equal(p, p_b), name)
        _assert_same(self, optimizers.state_dict(), optimizers_b.state_dict())

    def test_hf_and_remote_loads_keep_the_stock_reader(self):
        *_, ckpt = self._manager(seed=0)
        stock = mock.Mock()
        ckpt._load_checkpoint = stock
        dcp_parallel.tune_checkpoint_manager(ckpt)
        ckpt._load_checkpoint({}, "/hf/checkpoint", from_hf=True, from_quantized=False)
        ckpt._load_checkpoint({}, "memory://checkpoint/step-1", from_hf=False, from_quantized=False)
        self.assertEqual(stock.call_count, 2)

    def test_disabled_manager_is_untouched(self):
        import types

        disabled = types.SimpleNamespace(enable=False)
        dcp_parallel.tune_checkpoint_manager(disabled)
        self.assertFalse(hasattr(disabled, "_tuned_dcp_save"))


if __name__ == "__main__":
    unittest.main()
