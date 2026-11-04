# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks of ``update_probe.UpdateProbe`` (fraction of the fp32 AdamW update a run keeps).

bf16 weights at the scale of a large LLM (rms 5.5e-3), AdamW at lr 1e-6, the
same gradient stream for every optimizer:

* fp32 weights with ``torch.optim.AdamW``: the shadow reproduces the run (kept 1).
* bf16 weights with ``torch.optim.AdamW`` (round to nearest): kept ~0.1.
* ``LowPrecisionAdamW`` with a Kahan residual: kept ~1 (weights + residual) and
  the bf16 weights alone trail by less than half an ulp.
* ``LowPrecisionAdamW`` with stochastic rounding: kept ~1.
* ``install_update_probe`` wraps ``OptimizersContainer.step`` and reports.

Run: python -m pytest -q torchtitan/experiments/tpu/tests/test_update_probe.py
"""

from __future__ import annotations

import contextlib
import io
import re
import unittest

import torch

from torchtitan.experiments.tpu.low_precision_adamw import LowPrecisionAdamW, low_precision_adamw_config
from torchtitan.experiments.tpu.update_probe import (
    DEFAULT_PATTERN,
    UpdateProbe,
    install_update_probe,
    probe_category,
    uninstall_update_probe,
)

N = 1 << 16
STEPS = 20
LR = 1e-6


def _run(make_opt, dtype, *, steps=STEPS, elements=N):
  """Train one flat parameter on a fixed gradient stream; return the last report."""
  torch.manual_seed(0)
  w = torch.nn.Parameter((torch.randn(N) * 5.5e-3).to(dtype))
  opt = make_opt([w])
  probe = UpdateProbe(elements=elements, pattern=r"layers\.0\.", log=lambda msg: None)
  probe.build([("layers.0.attention.wq.weight", w)], [opt])
  gen = torch.Generator().manual_seed(1)
  bias = torch.randn(N, generator=gen)
  report = {}
  for step in range(steps):
    w.grad = (bias + 0.5 * torch.randn(N, generator=gen)).to(dtype)
    probe.before_step()
    opt.step()
    report = probe.after_step(step + 1)
  return report["all"], w, opt


def _adamw(params):
  return torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.95), weight_decay=0.0, foreach=False)


def _lpa(**kwargs):
  return lambda params: LowPrecisionAdamW(params, lr=LR, betas=(0.9, 0.95), weight_decay=0.0,
                                          tpu_kernel="off", **kwargs)


class UpdateProbeTest(unittest.TestCase):

  def test_fp32_adamw_matches_shadow(self):
    report, _, _ = _run(_adamw, torch.float32)
    self.assertAlmostEqual(report["kept"], 1.0, places=4)
    self.assertLess(report["dev"], 1e-3)

  def test_bf16_round_to_nearest_loses_updates(self):
    report, _, _ = _run(_adamw, torch.bfloat16)
    self.assertLess(report["kept"], 0.3)
    self.assertEqual(report["kept"], report["kept_bf16"])

  def test_kahan_keeps_updates(self):
    report, _, _ = _run(_lpa(param_update="kahan", state_dtype="bfloat16"), torch.bfloat16)
    self.assertGreater(report["kept"], 0.97)
    self.assertLess(report["kept"], 1.03)
    self.assertGreater(report["kept_bf16"], 0.8)

  def test_kahan_float8_states_keep_updates(self):
    report, _, _ = _run(_lpa(param_update="kahan", state_dtype="float8"), torch.bfloat16)
    self.assertGreater(report["kept"], 0.95)
    self.assertLess(report["kept"], 1.05)

  def test_stochastic_rounding_keeps_updates(self):
    report, _, _ = _run(_lpa(param_update="stochastic", state_dtype="bfloat16"), torch.bfloat16)
    self.assertGreater(report["kept"], 0.85)
    self.assertLess(report["kept"], 1.15)

  def test_pattern_and_categories(self):
    w = torch.nn.Parameter(torch.zeros(8))
    v = torch.nn.Parameter(torch.zeros(8))
    probe = UpdateProbe(elements=4, log=lambda msg: None)
    probe.build([("layers.0.moe.routed_experts.inner_experts.w1_EFD", w),
                 ("layers.5.attention.wo.weight", v)], [_adamw([w, v])])
    self.assertEqual([e.name for e in probe.probed], ["layers.0.moe.routed_experts.inner_experts.w1_EFD"])
    self.assertEqual(probe.probed[0].index[0].numel(), 4)
    names = ["tok_embeddings.weight", "layers.0.attention.wq.weight", "layers.10.attention.wq.weight",
             "layers.31.attention_norm.weight", "norm.weight", "output.weight", "lm_head.weight"]
    self.assertEqual([n for n in names if re.search(DEFAULT_PATTERN, n)],
                     ["tok_embeddings.weight", "layers.0.attention.wq.weight", "norm.weight", "output.weight",
                      "lm_head.weight"])
    self.assertEqual(probe_category("layers.3.moe.routed_experts.inner_experts.w2_EDF"), "experts")
    self.assertEqual(probe_category("layers.3.moe.router.gate.weight"), "router")
    self.assertEqual(probe_category("layers.3.attention.qkv_linear.wq.weight"), "attention")
    self.assertEqual(probe_category("tok_embeddings.embedding.weight"), "embed_head")
    self.assertEqual(probe_category("norm.weight"), "other")

  def test_zero_learning_rate_steps_are_skipped(self):
    w = torch.nn.Parameter(torch.ones(16))
    opt = torch.optim.AdamW([w], lr=0.0)
    probe = UpdateProbe(elements=16, pattern=".", log=lambda msg: None)
    probe.build([("w", w)], [opt])
    w.grad = torch.ones(16)
    probe.before_step()
    opt.step()
    self.assertEqual(probe.after_step(), {})
    self.assertEqual(probe.probed[0].step, 0)

  def test_container_hook_reports(self):
    model = torch.nn.Sequential()
    model.add_module("norm", torch.nn.Linear(64, 64, bias=False).to(torch.bfloat16))
    cfg = low_precision_adamw_config(lr=1e-3, param_update="kahan", state_dtype="bfloat16",
                                     tpu_kernel="off")
    opts = cfg.build(model_parts=[model])  # built before the probe is installed
    out = io.StringIO()
    try:
      self.assertFalse(install_update_probe(elements=0))
      self.assertTrue(install_update_probe(elements=32, pattern=r"norm\."))
      with contextlib.redirect_stdout(out):
        for _ in range(2):
          model(torch.randn(4, 64, dtype=torch.bfloat16)).float().pow(2).mean().backward()
          opts.step()
          opts.zero_grad()
    finally:
      uninstall_update_probe()
    lines = [l for l in out.getvalue().splitlines() if l.startswith("[UPDATE-PROBE]")]
    self.assertEqual(len(lines), 3, lines)  # "probing ..." + one line per step
    self.assertIn("all:kept=", lines[-1])
    self.assertIn("other:kept=", lines[-1])


if __name__ == "__main__":
  unittest.main()
