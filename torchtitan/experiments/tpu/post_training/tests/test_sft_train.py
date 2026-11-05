# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests of the ``sft_train`` entry point; no trainer is built.

Upstream's TPU startup is replaced by a stand-in that records which class
the name ``torchtitan.experiments.tpu.train.TPUTrainer`` holds while it runs.
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


if __name__ == "__main__":
  unittest.main()
