# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the resume-check hooks (``TT_FAULT_KILL_STEP``, ``TT_BATCH_RECORDS_DIR``)."""

from __future__ import annotations

import json
import os
import signal
import tempfile
import unittest
from collections.abc import Iterator
from typing import Any
from unittest import mock

import torch

from torchtitan.experiments.tpu.checkpointing import resume_check

_ENV = ("TT_FAULT_KILL_STEP", "TT_BATCH_RECORDS_DIR", "RANK")


class BatchDigestTest(unittest.TestCase):
    def test_digest(self):
        batch = {"input": torch.arange(6).view(2, 3), "labels": torch.ones(2, 3, dtype=torch.int64),
                 "weights": torch.rand(2, 3).to(torch.bfloat16), "meta": "ignored"}
        digest = resume_check.batch_sha256(batch)
        self.assertEqual(digest, resume_check.batch_sha256(dict(reversed(list(batch.items())))))
        self.assertEqual(digest, resume_check.batch_sha256({k: v for k, v in batch.items() if k != "meta"}))
        self.assertNotEqual(digest, resume_check.batch_sha256({**batch, "input": torch.arange(1, 7).view(2, 3)}))
        renamed = dict(batch)
        renamed["tokens"] = renamed.pop("input")
        self.assertNotEqual(digest, resume_check.batch_sha256(renamed))

    def test_pinned_digests(self):
        """Fixed batches keep fixed digests: batch records stay comparable with those of earlier runs."""
        digest = resume_check.batch_sha256
        mixed = {
            "input_ids": torch.arange(12, dtype=torch.int64).reshape(3, 4),
            "labels": -torch.arange(12, dtype=torch.int32).reshape(3, 4),
            "weights": torch.linspace(-2, 2, 12, dtype=torch.float32).to(torch.bfloat16).reshape(3, 4),
            "mask": torch.tensor([[True, False, True, True]]),
            "meta": "not hashed",
            "count": 7,
        }
        mixed_sha = "a3af34137756a40c08d4e1698b26dea9f75273c7431d5ef8b3224096af305b91"
        self.assertEqual(digest(mixed), mixed_sha)
        self.assertEqual(digest(dict(reversed(list(mixed.items())))), mixed_sha)
        self.assertEqual(digest({k: v for k, v in mixed.items() if isinstance(v, torch.Tensor)}), mixed_sha)
        mixed["tokens"] = mixed.pop("input_ids")
        self.assertEqual(digest(mixed), "f93a9e8bad938f7fa4f669d32542fbc0750d06a6d2f2a2993dca300083e0d102")
        transposed = {"x": torch.arange(24, dtype=torch.float32).view(4, 6).t()}
        self.assertEqual(digest(transposed), "1999691eb39a4bc979842a83ba5c76cf7e2193032f2d7c13432c22a0962b09d8")
        odd_shapes = {
            "scalar": torch.tensor(3.5, dtype=torch.float64),
            "empty": torch.zeros(0, 5, dtype=torch.float16),
            "f8": torch.linspace(-1, 1, 8).to(torch.float8_e4m3fn),
        }
        self.assertEqual(digest(odd_shapes), "8e18e6747275f1b9bf34ed4b79dbe4c656dee05d8e55dbee01996adce880afc7")
        no_tensors = {"meta": "x", "count": 3}
        self.assertEqual(digest(no_tensors), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
        needs_grad = {"w": torch.ones(2, 3, requires_grad=True) * 0.5}
        self.assertEqual(digest(needs_grad), "9b3f5582c2cee9946e180b63598ed21f206b8a1dfe8695a45b4f7c625ed831ea")


class InstallTest(unittest.TestCase):
    def setUp(self):
        from torchtitan.components.metrics import MetricsProcessor
        from torchtitan.trainer import Trainer

        self.trainer_cls, self.metrics_cls = Trainer, MetricsProcessor
        self.saved = (Trainer.batch_generator, Trainer.train_step, MetricsProcessor.log)
        self.env = {k: os.environ.pop(k) for k in _ENV if k in os.environ}
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        Trainer, MetricsProcessor = self.trainer_cls, self.metrics_cls
        Trainer.batch_generator, Trainer.train_step, MetricsProcessor.log = self.saved
        for key in _ENV:
            os.environ.pop(key, None)
        os.environ.update(self.env)
        self.tmp.cleanup()

    def test_nothing_is_patched_without_the_environment(self):
        self.assertFalse(resume_check.install())
        self.assertEqual(
            (self.trainer_cls.batch_generator, self.trainer_cls.train_step, self.metrics_cls.log), self.saved)

    def test_batch_records_and_fault_kill(self):
        Trainer, MetricsProcessor = self.trainer_cls, self.metrics_cls

        # Stand-ins for the trainer's methods, wrapped by install() like the real ones.
        def batch_generator(self, data_iterable):
            yield from data_iterable

        def train_step(self, data_iterator: Iterator[dict[str, Any]]):
            self.drawn = [next(data_iterator) for _ in range(2)]

        def log(self, step, global_avg_loss, global_max_loss, grad_norm, extra_metrics=None):
            self.logged = (step, extra_metrics)

        Trainer.batch_generator, Trainer.train_step, MetricsProcessor.log = batch_generator, train_step, log
        records = os.path.join(self.tmp.name, "records")
        os.environ.update({"TT_FAULT_KILL_STEP": "3", "TT_BATCH_RECORDS_DIR": records, "RANK": "0"})
        batches = [{"tokens": torch.tensor([i, i + 1])} for i in range(10)]
        with mock.patch.object(resume_check.os, "kill") as kill:
            self.assertTrue(resume_check.install())
            self.assertTrue(resume_check.install())  # a second call changes nothing
            trainer = object.__new__(Trainer)
            data = Trainer.batch_generator(trainer, batches)
            for step in (1, 2):
                trainer.step = step
                Trainer.train_step(trainer, data)
            kill.assert_not_called()
            trainer.step = 3
            Trainer.train_step(trainer, data)  # step 3 draws its first batch (the mock lets it go on)
            self.assertEqual(kill.call_args_list[0], mock.call(os.getpid(), signal.SIGKILL))
        with open(os.path.join(records, "rank-00000.jsonl")) as f:
            lines = [json.loads(line) for line in f]
        self.assertEqual([line["step"] for line in lines], [1, 2, 3])
        for i, line in enumerate(lines):
            want = [resume_check.batch_sha256(b) for b in batches[2 * i: 2 * i + 2]]
            self.assertEqual(line["batches"], want)
            self.assertEqual(line["rank"], 0)
        metrics = object.__new__(MetricsProcessor)
        with self.assertLogs(resume_check.logger.name, level="INFO") as logs:
            MetricsProcessor.log(metrics, 3, 1.5, 2.0, 0.25, extra_metrics={"lr": 1.0})
        self.assertIn("[resume-check] step=3 loss=1.5 max_loss=2.0 grad_norm=0.25", logs.output[0])
        self.assertEqual(metrics.logged, (3, {"lr": 1.0}))


if __name__ == "__main__":
    unittest.main()
