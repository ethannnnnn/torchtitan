# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the measured expert placement (``expert_parallel`` utilities, ``expert_placement``)."""

from __future__ import annotations

import json
import os
import random
import tempfile
import unittest

from torchtitan.experiments.tpu.muse_spark import expert_parallel as ep
from torchtitan.experiments.tpu.muse_spark import expert_placement as pl


class BalancedPlacementTest(unittest.TestCase):

  def test_balanced_is_a_partition_and_beats_contiguous(self):
    rng = random.Random(0)
    for experts, ranks in ((16, 4), (256, 8), (32, 32)):
      counts = [[rng.paretovariate(1.2) for _ in range(experts)] for _ in range(3)]
      maps = ep.balanced_expert_placement(counts, ranks)
      self.assertEqual(sorted(maps), ["0", "1", "2"])
      for slots in maps.values():
        self.assertEqual(sorted(slots), list(range(experts)))
      balanced = ep.busiest_rank_load(counts, maps, ranks)
      contiguous = ep.busiest_rank_load(counts, None, ranks)
      for b, c in zip(balanced, contiguous):
        self.assertLessEqual(b, c + 1e-12)

  def test_skewed_layer(self):
    counts = [[100.0, 90.0, 80.0, 70.0] + [1.0] * 12]
    summary = pl.skew_summary(counts, 4)
    self.assertGreater(summary["contiguous_max"], 3.0)
    self.assertLess(summary["balanced_max"], 1.2)
    self.assertAlmostEqual(summary["hottest_expert_max"], 100.0 / (352.0 / 16))

  def test_bad_split(self):
    with self.assertRaises(ValueError):
      ep.balanced_expert_placement([[1.0] * 10], 4)

  def test_write_and_load_round_trip(self):
    rng = random.Random(1)
    counts = [[rng.random() for _ in range(16)] for _ in range(2)]
    maps = ep.balanced_expert_placement(counts, 4)
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, "placement.json")
      ep.write_expert_placement(path, maps, 4, source="test")
      with open(path) as f:
        spec = json.load(f)
      self.assertEqual((spec["ep"], spec["num_experts"], spec["source"]), (4, 16, "test"))
      self.assertEqual(ep.load_expert_placement("file:" + path, 4), maps)
      with self.assertRaises(ValueError):
        ep.load_expert_placement("file:" + path, 8)

  def test_writer_is_inert_without_the_variable(self):
    os.environ.pop("SPARK_EXPERT_PLACEMENT_OUT", None)
    sentinel = object()
    self.assertIs(pl.install_placement_writer(sentinel), sentinel)


if __name__ == "__main__":
  unittest.main()
