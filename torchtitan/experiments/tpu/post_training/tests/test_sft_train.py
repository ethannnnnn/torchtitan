# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests of ``sft_train``: its entry point and the chunked-loss restore.

Upstream's TPU start-up is swapped for a stand-in that notes which class the
module-level ``TPUTrainer`` name of ``torchtitan.experiments.tpu.train`` holds
while it runs, so no trainer is ever built; the loss restore runs on a plain
object that stands in for the trainer.
"""

from __future__ import annotations

import unittest
from unittest import mock


class StartTrainerTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    from torchtitan.experiments.tpu import train as tpu_train
    from torchtitan.experiments.tpu.post_training import sft_train

    cls.tpu_train = tpu_train
    cls.sft_train = sft_train
    cls.upstream_cls = tpu_train.TPUTrainer

  def _start(self, stand_in, config) -> None:
    with mock.patch.object(self.tpu_train, "start_trainer", stand_in):
      self.sft_train.start_trainer(config)

  def test_sft_trainer_only_during_the_call(self):
    seen = []

    def stand_in(config):
      seen.append((config, self.tpu_train.TPUTrainer))

    config = object()
    self._start(stand_in, config)
    self.assertIs(self.tpu_train.TPUTrainer, self.upstream_cls)
    self._start(stand_in, config)  # a second call in the same process
    self.assertEqual(seen, [(config, self.sft_train.SFTTPUTrainer)] * 2)
    self.assertIs(self.tpu_train.TPUTrainer, self.upstream_cls)

  def test_upstream_class_back_after_an_exception(self):
    def stand_in(config):
      del config
      self.assertIs(self.tpu_train.TPUTrainer, self.sft_train.SFTTPUTrainer)
      raise RuntimeError("stand-in failed")

    with self.assertRaisesRegex(RuntimeError, "stand-in failed"):
      self._start(stand_in, object())
    self.assertIs(self.tpu_train.TPUTrainer, self.upstream_cls)

  def test_upstream_class_back_after_keyboard_interrupt(self):
    def stand_in(config):
      del config
      raise KeyboardInterrupt

    with self.assertRaises(KeyboardInterrupt):
      self._start(stand_in, object())
    self.assertIs(self.tpu_train.TPUTrainer, self.upstream_cls)

  def test_public_helper_takes_any_trainer_class(self):
    class OtherTrainer(self.sft_train.SFTTPUTrainer):
      pass

    with self.sft_train.tpu_trainer_class(OtherTrainer):
      self.assertIs(self.tpu_train.TPUTrainer, OtherTrainer)
    self.assertIs(self.tpu_train.TPUTrainer, self.upstream_cls)
    self.assertTrue(
        issubclass(self.sft_train.SFTTPUTrainer, self.upstream_cls)
    )


class ChunkedLossRestoreTest(unittest.TestCase):
  """``SFTTPUTrainer._reinstall_chunked_loss`` on a stand-in trainer."""

  def _restore(self, loss_config, parts):
    import types

    from torchtitan.config.configs import CompileConfig
    from torchtitan.experiments.tpu.post_training import sft_train

    stand_in = types.SimpleNamespace(model_parts=parts, loss_fn="as built")
    job = types.SimpleNamespace(loss=loss_config, compile=CompileConfig())
    sft_train.SFTTPUTrainer._reinstall_chunked_loss(stand_in, job)
    return stand_in

  def test_other_losses_are_left_alone(self):
    import types

    from torchtitan.components.loss import CrossEntropyLoss

    part = types.SimpleNamespace(lm_head=object())
    restored = self._restore(CrossEntropyLoss.Config(), [part])
    self.assertEqual(restored.loss_fn, "as built")
    self.assertFalse(hasattr(part, "_skip_lm_head"))

  def test_chunked_loss_takes_the_head(self):
    import types

    import torch

    from torchtitan.components.loss import ChunkedLossWrapper

    head = torch.nn.Linear(4, 6)
    part = types.SimpleNamespace(lm_head=head)
    restored = self._restore(ChunkedLossWrapper.Config(num_chunks=3), [part])
    self.assertIsInstance(restored.loss_fn, ChunkedLossWrapper)
    self.assertIs(restored.loss_fn.lm_head, head)
    self.assertEqual(restored.loss_fn.num_chunks, 3)
    self.assertIs(part._skip_lm_head, True)

  def test_chunked_loss_needs_one_part_with_a_head(self):
    import types

    from torchtitan.components.loss import ChunkedLossWrapper

    headed = types.SimpleNamespace(lm_head=object())
    cases = {
        "two parts": [headed, types.SimpleNamespace(lm_head=object())],
        "no lm_head": [types.SimpleNamespace()],
        "lm_head None": [types.SimpleNamespace(lm_head=None)],
        "no parts": [],
    }
    for label, parts in cases.items():
      with self.subTest(label), self.assertRaises(ValueError):
        self._restore(ChunkedLossWrapper.Config(), parts)
    self.assertFalse(hasattr(headed, "_skip_lm_head"))


if __name__ == "__main__":
  unittest.main()
