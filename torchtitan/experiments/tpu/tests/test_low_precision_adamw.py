# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks of ``LowPrecisionAdamW`` (bf16 parameters without an fp32 master).

1. Rounding kernels: stochastic rounding to bf16 (bit and arithmetic kernels)
   and to float8_e4m3fn returns one of the two neighbouring grid values and is
   unbiased (mean over draws equals the input).
2. Retained update: bf16 weights at the scale of a large LLM (rms 5.5e-3),
   AdamW at lr 1e-6 with a fixed-sign gradient. Round-to-nearest keeps ~10% of
   the fp32-master displacement; stochastic rounding and Kahan keep ~100%.
3. Expectation match: noisy gradient streams, identical for every optimizer;
   the displacement of each low-precision variant regressed on fp32 AdamW's
   (same initial bf16 weights) has slope ~1 and no bias.
4. TorchTitan integration: the container config builds, steps a module, and
   updates DTensor parameters through their local shards; on two gloo ranks,
   the state of sharded parameters goes through a distributed checkpoint
   (``low_precision_adamw_dcp_check``).
5. Fused TPU kernels (needs ``jax``; runs on JAX's CPU backend): in
   deterministic modes the kernels equal the eager PyTorch update bit for bit,
   except the fp32 float8 block scales, which may differ in the last bit; in
   stochastic modes they are unbiased.

Run: python -m pytest -q torchtitan/experiments/tpu/tests/test_low_precision_adamw.py
"""

from __future__ import annotations

import math
import os
import unittest

import torch
import torch.distributed

from torchtitan.experiments.tpu.low_precision_adamw import (
    LowPrecisionAdamW,
    adam_ratio_bound,
    dequantize_float8_blockwise,
    low_precision_adamw_config,
    quantize_float8_blockwise,
    stochastic_round_to_bf16,
    stochastic_round_to_float8,
    _stochastic_round_to_bf16_arith,
)


def _bf16_neighbours(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """The bf16 values just below and just above (toward -inf / +inf) each fp32 value."""
  bits = x.view(torch.int32)
  trunc = torch.bitwise_and(bits, -(1 << 16)).view(torch.float32)  # toward zero
  away = (torch.bitwise_and(bits, -(1 << 16)) + (1 << 16)).view(torch.float32)
  lo = torch.where(x >= 0, trunc, away)
  hi = torch.where(x >= 0, away, trunc)
  exact = trunc == x
  return torch.where(exact, x, lo), torch.where(exact, x, hi)


class RoundingKernelTest(unittest.TestCase):

  def _values(self, n=4096):
    g = torch.Generator().manual_seed(0)
    mag = torch.exp(torch.empty(n).uniform_(math.log(1e-6), math.log(10.0), generator=g))
    sign = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0)
    return (mag * sign).to(torch.float32)

  def test_bf16_bits_neighbours_and_unbiased(self):
    torch.manual_seed(1)
    x = self._values()
    lo, hi = _bf16_neighbours(x)
    draws = 4000
    acc = torch.zeros_like(x, dtype=torch.float64)
    for _ in range(draws):
      r = torch.randint(0, 1 << 16, x.shape, dtype=torch.int32)
      y = stochastic_round_to_bf16(x, r).to(torch.float32)
      self.assertTrue(bool(((y == lo) | (y == hi)).all()))
      acc += y.double()
    mean = acc / draws
    spacing = (hi - lo).double().clamp_min(1e-30)
    # Each draw has std <= spacing / 2; the mean's error is ~ spacing / (2 sqrt(draws)).
    z = ((mean - x.double()) / (spacing / (2 * math.sqrt(draws)))).abs()
    self.assertLess(float(z.max()), 5.0)
    self.assertLess(abs(float(((mean - x.double()) / spacing).mean())), 2e-3)

  def test_bf16_exact_values_unchanged(self):
    x = self._values().to(torch.bfloat16).to(torch.float32)
    r = torch.randint(0, 1 << 16, x.shape, dtype=torch.int32)
    self.assertTrue(torch.equal(stochastic_round_to_bf16(x, r).to(torch.float32), x))
    inf = torch.tensor([float("inf"), -float("inf")])
    out = stochastic_round_to_bf16(inf, torch.full((2,), 65535, dtype=torch.int32))
    self.assertTrue(torch.isinf(out).all())

  def test_bf16_arith_unbiased(self):
    torch.manual_seed(2)
    x = self._values()
    lo, hi = _bf16_neighbours(x)
    draws = 4000
    acc = torch.zeros_like(x, dtype=torch.float64)
    for _ in range(draws):
      acc += _stochastic_round_to_bf16_arith(x, torch.rand(x.shape)).double()
    mean = acc / draws
    spacing = (hi - lo).double().clamp_min(1e-30)
    rel = ((mean - x.double()) / spacing)
    # Exact except within a quarter spacing above a power of two.
    self.assertLess(float(rel.abs().quantile(0.99)), 0.05)
    self.assertLess(abs(float(rel.mean())), 3e-3)

  def test_float8_unbiased(self):
    torch.manual_seed(3)
    g = torch.Generator().manual_seed(4)
    # Normal range and the subnormal range (|x| < 2**-6).
    x = torch.cat([
        torch.empty(2048).uniform_(-448.0, 448.0, generator=g),
        torch.empty(1024).uniform_(-2.0**-6, 2.0**-6, generator=g),
        torch.empty(1024).uniform_(-1.0, 1.0, generator=g),
    ])
    nearest = x.to(torch.float8_e4m3fn).to(torch.float32)
    draws = 3000
    acc = torch.zeros_like(x, dtype=torch.float64)
    for _ in range(draws):
      r = torch.randint(0, 1 << 20, x.shape, dtype=torch.int32)
      acc += stochastic_round_to_float8(x, r).to(torch.float32).double()
    mean = acc / draws
    err_sr = (mean - x.double()).abs()
    err_rn = (nearest.double() - x.double()).abs()
    # Unbiased: the mean error is far below round-to-nearest's error.
    self.assertLess(float(err_sr.sum() / err_rn.sum()), 0.1)
    self.assertLess(abs(float((mean - x.double()).sum() / err_rn.sum())), 0.01)

  def test_float8_blockwise_roundtrip(self):
    g = torch.Generator().manual_seed(5)
    x = torch.randn(64, 256, generator=g) * torch.logspace(-4, 0, 64).unsqueeze(1)
    codes, scales = quantize_float8_blockwise(x, 256, None)
    y = dequantize_float8_blockwise(codes, scales).view_as(x)
    rel = ((y - x).abs() / x.abs().amax(dim=1, keepdim=True)).max()
    self.assertLess(float(rel), 2.0**-4)  # half of the 3-bit mantissa step at the block max


def _llm_like_weights(n: int, seed: int) -> torch.Tensor:
  g = torch.Generator().manual_seed(seed)
  return (torch.randn(n, generator=g) * 5.5e-3).to(torch.bfloat16)


def _run(opt_factory, w0: torch.Tensor, grads: list[torch.Tensor], master: bool = True) -> torch.Tensor:
  """Final weights; with ``master``, Kahan variants return parameter + residual.

  A Kahan parameter is the bf16 rounding of a value tracked to ~16 significant bits
  (like the bf16 copy of an fp32 master weight); the update is exact in
  parameter + residual, and the bf16 parameter trails it by under half an ulp.
  """
  p = torch.nn.Parameter(w0.clone())
  opt = opt_factory([p])
  for g in grads:
    p.grad = g.to(p.dtype)
    opt.step()
  out = p.detach().to(torch.float32)
  residual = opt.state[p].get("residual") if isinstance(opt, LowPrecisionAdamW) else None
  if master and residual is not None:
    out = out + residual.to(torch.float32)
  return out


KW = dict(lr=1e-6, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)


def _variants():
  return {
      "torch_adamw_bf16_nearest": lambda ps: torch.optim.AdamW(ps, foreach=False, **KW),
      "sr_bf16_states": lambda ps: LowPrecisionAdamW(ps, **KW),
      "sr_bf16_states_arith": lambda ps: LowPrecisionAdamW(ps, sr_impl="arith", **KW),
      "sr_fp32_states": lambda ps: LowPrecisionAdamW(ps, state_dtype="float32", **KW),
      "kahan_bf16_states": lambda ps: LowPrecisionAdamW(ps, param_update="kahan", **KW),
      "sr_float8_states": lambda ps: LowPrecisionAdamW(ps, state_dtype="float8", **KW),
      "kahan_float8_states": lambda ps: LowPrecisionAdamW(ps, param_update="kahan", state_dtype="float8", **KW),
      "kahan_nearest_residual": lambda ps: LowPrecisionAdamW(ps, param_update="kahan", state_rounding="nearest", **KW),
      "nearest_bf16_states": lambda ps: LowPrecisionAdamW(ps, param_update="nearest", state_rounding="nearest", **KW),
  }


class RetainedUpdateTest(unittest.TestCase):

  def _reference(self, w0, grads):
    return _run(lambda ps: torch.optim.AdamW(ps, foreach=False, **KW), w0.to(torch.float32), grads)

  def test_fixed_sign_gradient(self):
    torch.manual_seed(10)
    n, steps = 1 << 16, 20
    w0 = _llm_like_weights(n, 11)
    gdir = torch.randn(n)
    grads = [gdir.clone() for _ in range(steps)]
    ref = self._reference(w0, grads)
    d_ref = ref - w0.float()
    out = {}
    for name, make in _variants().items():
      d = _run(make, w0, grads) - w0.float()
      out[name] = float((d * d_ref).sum() / (d_ref * d_ref).sum())
    print("\n[retained update, fixed-sign gradient, lr 1e-6, 20 steps]", {k: round(v, 4) for k, v in out.items()})
    self.assertLess(out["torch_adamw_bf16_nearest"], 0.2)
    self.assertLess(out["nearest_bf16_states"], 0.2)
    for name in ("sr_bf16_states", "sr_bf16_states_arith", "sr_fp32_states", "kahan_bf16_states",
                 "sr_float8_states", "kahan_float8_states"):
      self.assertGreater(out[name], 0.97, name)
      self.assertLess(out[name], 1.03, name)

  def test_noisy_gradients_match_fp32_in_expectation(self):
    torch.manual_seed(20)
    n, steps = 1 << 16, 40
    w0 = _llm_like_weights(n, 21)
    g = torch.Generator().manual_seed(22)
    mu = torch.randn(n, generator=g) * 0.3
    grads = [mu + torch.randn(n, generator=g) for _ in range(steps)]
    ref = self._reference(w0, grads)
    d_ref = ref - w0.float()
    rows = {}
    for name, make in _variants().items():
      d = _run(make, w0, grads) - w0.float()
      slope = float((d * d_ref).sum() / (d_ref * d_ref).sum())
      bias = float((d - d_ref).mean() / d_ref.pow(2).mean().sqrt())
      rows[name] = (round(slope, 4), round(bias, 5))
    print("\n[noisy gradients, lr 1e-6, 40 steps: (slope vs fp32 AdamW, bias / rms)]", rows)
    # Kahan's bf16 weights (what the forward pass sees) equal the bf16 rounding of
    # the fp32-master trajectory almost everywhere.
    # (8-bit moments perturb the update itself, so fewer elements match exactly.)
    for name, floor in (("kahan_bf16_states", 0.97), ("kahan_float8_states", 0.85)):
      p_only = _run(_variants()[name], w0, grads, master=False)
      same = float((p_only == ref.to(torch.bfloat16).to(torch.float32)).float().mean())
      print(f"[{name}] bf16 weights == bf16(fp32-master weights) for {same:.4f} of elements")
      self.assertGreater(same, floor, name)
    self.assertLess(rows["torch_adamw_bf16_nearest"][0], 0.3)
    for name in ("sr_bf16_states", "sr_fp32_states", "kahan_bf16_states", "sr_float8_states", "kahan_float8_states"):
      slope, bias = rows[name]
      self.assertGreater(slope, 0.95, name)
      self.assertLess(slope, 1.05, name)
      self.assertLess(abs(bias), 0.02, name)

  def test_weight_decay_applied_in_expectation(self):
    torch.manual_seed(30)
    n = 1 << 16
    w0 = _llm_like_weights(n, 31)
    kw = dict(KW, lr=1e-4, weight_decay=0.1)
    zero = [torch.zeros(n) for _ in range(10)]
    ref = _run(lambda ps: torch.optim.AdamW(ps, foreach=False, **kw), w0.to(torch.float32), zero)
    sr = _run(lambda ps: LowPrecisionAdamW(ps, **kw), w0, zero)
    rn = _run(lambda ps: torch.optim.AdamW(ps, foreach=False, **kw), w0, zero)
    d_ref = ref - w0.float()

    def slope(w):
      return float(((w - w0.float()) * d_ref).sum() / (d_ref * d_ref).sum())

    self.assertAlmostEqual(slope(sr), 1.0, delta=0.05)
    self.assertLess(abs(slope(rn)), 0.05)

  def test_ratio_bound(self):
    for b1, b2 in ((0.9, 0.95), (0.9, 0.999), (0.8, 0.99)):
      g = torch.Generator().manual_seed(40)
      m = v = torch.zeros(1 << 14, dtype=torch.float64)
      worst = 0.0
      for t in range(1, 60):
        noise = torch.randn(1 << 14, generator=g, dtype=torch.float64)
        grad = noise * torch.exp(torch.randn(1, generator=g, dtype=torch.float64) * 3)
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad * grad
        r = (m / (1 - b1**t)).abs() / (v / (1 - b2**t)).sqrt()
        worst = max(worst, float(r.max() / adam_ratio_bound(b1, b2, t)))
      self.assertLessEqual(worst, 1.0 + 1e-9)
    self.assertAlmostEqual(adam_ratio_bound(0.9, 0.95, 1), 1.0, places=12)


class TorchTitanIntegrationTest(unittest.TestCase):

  def test_container_builds_and_steps(self):
    torch.manual_seed(50)
    model = torch.nn.Sequential(torch.nn.Linear(64, 512), torch.nn.Linear(512, 8)).to(torch.bfloat16)
    for state_dtype in ("bfloat16", "float8"):
      for update in ("stochastic", "kahan"):
        cfg = low_precision_adamw_config(lr=1e-3, state_dtype=state_dtype, param_update=update)
        opts = cfg.build(model_parts=[model])
        before = [p.detach().clone() for p in model.parameters()]
        loss = model(torch.randn(4, 64, dtype=torch.bfloat16)).float().pow(2).mean()
        loss.backward()
        opts.step()
        opts.zero_grad()
        moved = [bool((p.detach() != b).any()) for p, b in zip(model.parameters(), before)]
        self.assertTrue(all(moved), (state_dtype, update, moved))
        inner = opts.optimizers[0]
        self.assertIsInstance(inner, LowPrecisionAdamW)
        fmts = {"float8" if "exp_avg_f8" in s else "bfloat16" for s in inner.state.values()}
        if state_dtype == "float8":
          self.assertIn("float8", fmts)  # 64x512 weight: float8 moments
          self.assertIn("bfloat16", fmts)  # biases (last dim 8 < 256): bf16 moments

  def test_dtensor_parameters(self):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    created = False
    if not dist.is_initialized():
      os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
      os.environ.setdefault("MASTER_PORT", str(29400 + os.getpid() % 1000))
      dist.init_process_group("gloo", rank=0, world_size=1)
      created = True
    try:
      mesh = init_device_mesh("cpu", (1,))
      w0 = _llm_like_weights(1 << 12, 60).view(16, 256)
      grads = [torch.randn(16, 256) for _ in range(5)]
      for state_dtype in ("bfloat16", "float8"):
        torch.manual_seed(61)
        p_plain = torch.nn.Parameter(w0.clone())
        torch.manual_seed(61)
        p_dt = torch.nn.Parameter(distribute_tensor(w0.clone(), mesh, [Shard(0)]))
        o_plain = LowPrecisionAdamW([p_plain], lr=1e-3, state_dtype=state_dtype)
        o_dt = LowPrecisionAdamW([p_dt], lr=1e-3, state_dtype=state_dtype)
        for g in grads:
          p_plain.grad = g.to(torch.bfloat16)
          p_dt.grad = distribute_tensor(g.to(torch.bfloat16), mesh, [Shard(0)])
          torch.manual_seed(62)
          o_plain.step()
          torch.manual_seed(62)
          o_dt.step()
        self.assertTrue(torch.equal(p_plain.detach(), p_dt.detach().full_tensor()), state_dtype)
    finally:
      if created:
        dist.destroy_process_group()

  @unittest.skipUnless(torch.distributed.is_available() and torch.distributed.is_gloo_available(), "needs gloo")
  def test_distributed_checkpoint_two_ranks(self):
    import tempfile

    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as directory:
      mp.spawn(_dcp_check_rank, args=(os.path.join(directory, "rendezvous"),), nprocs=2, join=True)


def _dcp_check_rank(rank: int, store: str) -> None:
  """One rank of ``low_precision_adamw_dcp_check`` (exits non-zero if a check fails)."""
  from torchtitan.experiments.tpu.tests import low_precision_adamw_dcp_check

  low_precision_adamw_dcp_check.main(f"file://{store}", rank, 2)


def _has_jax() -> bool:
  import importlib.util

  return importlib.util.find_spec("jax") is not None


@unittest.skipUnless(_has_jax(), "jax is not installed")
class FusedKernelTest(unittest.TestCase):
  """The JAX kernels behind ``tpu_kernel`` vs the eager update, on JAX's CPU backend."""

  B1, B2, EPS, WD, LR = 0.9, 0.95, 1e-8, 0.1, 1e-6

  @classmethod
  def setUpClass(cls):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jnp
    import numpy as np

    from torchtitan.experiments.tpu.kernels import low_precision_adamw_kernels as kernels

    # The exact comparisons are for XLA's CPU arithmetic (TPU division and square
    # roots may round differently), so run on JAX's CPU backend even when another
    # backend is the default.
    try:
      cpu = jax.devices("cpu")[0]
    except RuntimeError as e:  # JAX_PLATFORMS without "cpu"
      raise unittest.SkipTest(f"needs JAX's CPU backend: {e}") from e
    device = jax.default_device(cpu)
    device.__enter__()
    cls.addClassCleanup(device.__exit__, None, None, None)
    cls.jnp, cls.np, cls.K = jnp, np, kernels
    torch.manual_seed(70)
    cls.n = 1 << 14
    cls.p0 = _llm_like_weights(cls.n, 71)
    cls.grads = [torch.randn(cls.n).to(torch.bfloat16) for _ in range(3)]

  def _j(self, t):
    jnp = self.jnp
    if t.dtype == torch.bfloat16:
      return jnp.asarray(t.view(torch.int16).numpy()).view(jnp.bfloat16)
    if t.dtype == torch.float8_e4m3fn:
      return jnp.asarray(t.view(torch.uint8).numpy()).view(jnp.float8_e4m3fn)
    return jnp.asarray(t.numpy())

  def _t(self, a):
    np, jnp = self.np, self.jnp
    a = np.asarray(a)
    if a.dtype == jnp.bfloat16:
      return torch.from_numpy(a.view(np.int16).copy()).view(torch.bfloat16)
    if a.dtype == jnp.float8_e4m3fn:
      return torch.from_numpy(a.view(np.uint8).copy()).view(torch.float8_e4m3fn)
    return torch.from_numpy(a.copy())

  def _eager(self, steps, **kw):
    p = torch.nn.Parameter(self.p0.clone())
    opt = LowPrecisionAdamW([p], lr=self.LR, betas=(self.B1, self.B2), eps=self.EPS, weight_decay=self.WD,
                            tpu_kernel="off", **kw)
    for g in self.grads[:steps]:
      p.grad = g
      opt.step()
    return p.detach(), opt.state[p]

  def _coef(self, step):
    b1, b2, lr = self.B1, self.B2, self.LR
    return self.jnp.asarray([lr, lr / (1 - b1**step), 1 / math.sqrt(1 - b2**step), lr * self.WD,
                             lr * adam_ratio_bound(b1, b2, step)], self.jnp.float32)

  def test_deterministic_modes_match_eager(self):
    K, seed = self.K, self.jnp.asarray([1, 2, 3, 4], self.jnp.int32)
    args = (self.B1, self.B2, self.EPS)
    p, st = self._eager(2, param_update="nearest", state_rounding="nearest")
    ref, ref_st = self._eager(3, param_update="nearest", state_rounding="nearest")
    out = K._bf16_states_update(self._j(p), self._j(self.grads[2]), self._j(st["exp_avg"]), self._j(st["exp_avg_sq"]),
                                seed, self._coef(3), *args, False, False)
    self.assertTrue(torch.equal(self._t(out[0]), ref))
    self.assertTrue(torch.equal(self._t(out[1]), ref_st["exp_avg"]))
    self.assertTrue(torch.equal(self._t(out[2]), ref_st["exp_avg_sq"]))

    p, st = self._eager(2, param_update="kahan", state_rounding="nearest")
    ref, ref_st = self._eager(3, param_update="kahan", state_rounding="nearest")
    out = K._bf16_states_kahan_update(self._j(p), self._j(self.grads[2]), self._j(st["exp_avg"]),
                                      self._j(st["exp_avg_sq"]), self._j(st["residual"]), seed, self._coef(3),
                                      *args, False)
    self.assertTrue(torch.equal(self._t(out[0]), ref))
    self.assertTrue(torch.equal(self._t(out[3]), ref_st["residual"]))

    p, st = self._eager(2, param_update="nearest", state_dtype="float8", state_rounding="nearest")
    ref, ref_st = self._eager(3, param_update="nearest", state_dtype="float8", state_rounding="nearest")
    out = K._f8_states_update(self._j(p), self._j(self.grads[2]), self._j(st["exp_avg_f8"].view(-1, 256)),
                              self._j(st["exp_avg_scale"].view(-1, 1)),
                              self._j(st["exp_avg_sq_sqrt_f8"].view(-1, 256)),
                              self._j(st["exp_avg_sq_sqrt_scale"].view(-1, 1)), seed, self._coef(3), *args, 256,
                              False, False)
    self.assertTrue(torch.equal(self._t(out[0]), ref))
    self.assertTrue(torch.equal(self._t(out[1]).float(), ref_st["exp_avg_f8"].view(-1, 256).float()))
    self.assertTrue(torch.equal(self._t(out[3]).float(), ref_st["exp_avg_sq_sqrt_f8"].view(-1, 256).float()))
    # The fp32 block scales (absmax / 448) may differ in the last bit (see
    # test_fused_update_path_matches_eager); everything else is bit-exact.
    for i, key in ((2, "exp_avg_scale"), (4, "exp_avg_sq_sqrt_scale")):
      torch.testing.assert_close(self._t(out[i]).view(-1), ref_st[key].view(-1), rtol=2.0**-23, atol=0.0, msg=key)

  def _cpu_kernels(self):
    """The registered ops' interface on top of the raw JAX functions (JAX's CPU
    backend), so that ``LowPrecisionAdamW._update_fused`` runs without TorchTPU."""
    import types

    def op(fn, num_static):
      def call(*args):
        tensors, static = args[:-num_static], args[-num_static:]
        return tuple(self._t(o) for o in fn(*(self._j(a) for a in tensors), *static))

      return call

    K = self.K
    return types.SimpleNamespace(
        bf16_states_update=op(K._bf16_states_update, 5),
        bf16_states_kahan_update=op(K._bf16_states_kahan_update, 4),
        f8_states_update=op(K._f8_states_update, 6),
        f8_states_kahan_update=op(K._f8_states_kahan_update, 5),
    )

  def test_fused_update_path_matches_eager(self):
    # _update_fused (state views, copies back, seed and coefficients) with the
    # deterministic kernels: step 3 equals the eager step 3 bit for bit, except
    # the fp32 float8 block scales (absmax / 448), which may differ in the last
    # bit: XLA computes the division by the constant as a multiplication by its
    # reciprocal, PyTorch rounds the quotient.
    kernels = self._cpu_kernels()
    for kw in (dict(param_update="nearest"), dict(param_update="kahan"),
               dict(param_update="nearest", state_dtype="float8"),
               dict(param_update="kahan", state_dtype="float8")):
      with self.subTest(**kw):
        ref, ref_st = self._eager(3, state_rounding="nearest", **kw)
        p = torch.nn.Parameter(self.p0.clone())
        opt = LowPrecisionAdamW([p], lr=self.LR, betas=(self.B1, self.B2), eps=self.EPS,
                                weight_decay=self.WD, tpu_kernel="off", state_rounding="nearest", **kw)
        for g in self.grads[:2]:
          p.grad = g
          opt.step()
        p.grad = self.grads[2]
        state = opt.state[p]
        state["step"] += 1
        with torch.no_grad():  # as in step()
          opt._update_fused(kernels, p, state, opt.param_groups[0], self._t(self._coef(3)))
        self.assertTrue(torch.equal(p.detach(), ref))
        self.assertEqual(sorted(state), sorted(ref_st))
        for key, value in ref_st.items():
          self.assertEqual(state[key].dtype, value.dtype, key)
          if key.endswith("_scale"):
            torch.testing.assert_close(state[key], value, rtol=2.0**-23, atol=0.0, msg=key)
          else:
            self.assertTrue(torch.equal(state[key].float(), value.float()), key)

  def test_stochastic_kernel_unbiased(self):
    import jax

    K, np = self.K, self.np
    p, st = self._eager(2)
    g = self.grads[2].float()
    m32 = st["exp_avg"].float() * self.B1 + g * (1 - self.B1)
    v32 = st["exp_avg_sq"].float() * self.B2 + g * g * (1 - self.B2)
    x = p.float()
    coef = self._t(self._coef(3))
    x_new = x - x * self.LR * self.WD - m32 / (v32.sqrt() * coef[2] + self.EPS) * coef[1]
    f = jax.jit(K._bf16_states_update, static_argnums=(6, 7, 8, 9, 10))
    acc = np.zeros(self.n)
    draws = 64
    for i in range(draws):
      out = f(self._j(p), self._j(self.grads[2]), self._j(st["exp_avg"]), self._j(st["exp_avg_sq"]),
              self.jnp.asarray([i, 7 * i + 1, 3 * i + 2, 11 * i + 5], self.jnp.int32), self._coef(3), self.B1, self.B2,
              self.EPS, True, True)
      acc += self._t(out[0]).float().numpy()
    d = (x_new - x).numpy()
    retained = float(((acc / draws - x.numpy()) * d).sum() / (d * d).sum())
    self.assertAlmostEqual(retained, 1.0, delta=0.05)


if __name__ == "__main__":
  unittest.main()
