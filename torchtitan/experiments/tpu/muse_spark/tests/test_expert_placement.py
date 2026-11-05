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
import types
import unittest
from unittest import mock

import torch
from torch import nn

from torchtitan.experiments.tpu.muse_spark import expert_parallel as ep
from torchtitan.experiments.tpu.muse_spark import expert_placement as pl


class _RouterCounts(nn.Module):
  """Stands in for an MoE layer: only its routed-token counter."""

  def __init__(self, counts):
    super().__init__()
    self.register_buffer("tokens_per_expert_E", torch.tensor(counts))


def _two_moe_layers():
  # Layer 0 has two hot experts (0 and 1), layer 1 routes evenly.
  return nn.Sequential(
      _RouterCounts([9.0, 8.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
      _RouterCounts([2.0] * 8),
  )


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

  def test_measure_and_write_balances_router_counts_once(self):
    with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"RANK": "0"}):
      path = os.path.join(tmp, "new_dir", "placement.json")
      pl._measure_and_write(_two_moe_layers(), 2, path, 2)
      maps = ep.load_expert_placement("file:" + path, 2)
      self.assertEqual(sorted(maps), ["0", "1"])
      # The two hot experts of layer 0 end up on different EP ranks.
      self.assertEqual(len({0, 1} & set(maps["0"][:4])), 1)
      with open(path) as f:
        written = f.read()
      # An existing placement is kept: later measurements do not replace it.
      pl._measure_and_write(nn.Sequential(_RouterCounts([1.0] * 7 + [50.0])), 4, path, 2)
      with open(path) as f:
        self.assertEqual(f.read(), written)

  def test_writer_hooks_parallelize_and_metrics_logging(self):
    from torchtitan.components.metrics import MetricsProcessor

    original_log = MetricsProcessor.log
    logged = []

    def fake_log(self, step, *args, **kwargs):
      del self, args, kwargs
      logged.append(step)

    MetricsProcessor.log = fake_log
    try:
      with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "placement.json")
        env = {
            "SPARK_EXPERT_PLACEMENT_OUT": path,
            "SPARK_EXPERT_STATS_STEP": "3",
            "SPARK_EXPERT_STATS_EP": "2",
            "RANK": "0",
        }
        with mock.patch.dict(os.environ, env):
          cfg = types.SimpleNamespace(
              model_spec=types.SimpleNamespace(parallelize_fn=lambda model, **kw: None)
          )
          self.assertIs(pl.install_placement_writer(cfg), cfg)
          cfg.model_spec.parallelize_fn(_two_moe_layers(), parallel_dims=None)
          MetricsProcessor.log(None, 2)
          self.assertFalse(os.path.exists(path))
          MetricsProcessor.log(None, 3)
          self.assertEqual(logged, [2, 3])
          self.assertEqual(sorted(ep.load_expert_placement("file:" + path, 2)), ["0", "1"])
    finally:
      MetricsProcessor.log = original_log


if __name__ == "__main__":
  unittest.main()
