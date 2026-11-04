# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused TPU kernels for ``LowPrecisionAdamW`` (one XLA program per parameter).

The eager ``LowPrecisionAdamW`` update is ~30 PyTorch ops per parameter; on
TorchTPU that is ~30k dispatches per step for a large model. These JAX functions,
registered as PyTorch ops through ``torch_tpu``'s ``jax_op``, compute the same
update (fp32 math, stochastic rounding, optional Kahan residual, bf16 or
blockwise float8 moments) in one fused program that reads the parameter,
gradient and states once and writes the new values once.

Random bits come from XLA's RngBitGenerator (``jax.random`` with the ``rbg``
implementation; threefry was ~4x slower here) keyed by a four-word seed tensor
drawn on the device per parameter and step, so a new seed never retraces. One
32-bit draw per element feeds all roundings of that element: each noise field
(16 bits for bf16, 20 bits for float8) is a fixed bit range of the draw and so
uniform on its own, which is all that unbiased stochastic rounding needs; the
fields overlap, so the roundings of one element are correlated with each other,
which does not bias any of them.

``coef`` is a float32[5] device tensor: (lr, lr / bias_correction1,
1 / sqrt(bias_correction2), lr * weight_decay, lr * ratio_bound); the betas,
eps, rounding modes and block size are static arguments (part of the trace key).

The raw JAX functions (leading underscore) import without TorchTPU, so the CPU
tests check them against the eager PyTorch update on JAX's CPU backend; the
registered ops exist only when ``torch_tpu`` is importable (``AVAILABLE``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

try:
  from torch_tpu._internal import pallas
except ImportError:  # CPU-only environments: raw functions only
  pallas = None

_F8 = jnp.float8_e4m3fn
_F8_MAX = 448.0


def _sr_bf16(x, noise16):
  bits = lax.bitcast_convert_type(x, jnp.uint32)
  bits = (bits + noise16) & jnp.uint32(0xFFFF0000)
  return lax.bitcast_convert_type(bits, jnp.float32).astype(jnp.bfloat16)


def _round_bf16(x, noise16, stochastic: bool):
  return _sr_bf16(x, noise16) if stochastic else x.astype(jnp.bfloat16)


def _sr_f8(x, noise20):
  bits = lax.bitcast_convert_type(x, jnp.uint32)
  trunc = lax.bitcast_convert_type((bits + noise20) & jnp.uint32(0xFFF00000), jnp.float32)
  dither = (noise20.astype(jnp.float32) * (2.0**-20) - 0.5) * (2.0**-9)
  y = jnp.where(jnp.abs(x) < 2.0**-6, x + dither, trunc)
  return jnp.clip(y, -_F8_MAX, _F8_MAX).astype(_F8)


def _quant_f8(x2d, noise20, stochastic: bool):
  absmax = jnp.max(jnp.abs(x2d), axis=1, keepdims=True)
  scale = jnp.where(absmax > 0, absmax / _F8_MAX, jnp.ones_like(absmax))
  scaled = x2d / scale
  if stochastic:
    codes = _sr_f8(scaled, noise20)
  else:
    codes = jnp.clip(scaled, -_F8_MAX, _F8_MAX).astype(_F8)
  return codes, scale


def _bits(seed, shape):
  """One uint32 draw per element and three overlapping fields of it."""
  key = jax.random.wrap_key_data(seed.astype(jnp.uint32), impl="rbg")
  b = jax.random.bits(key, shape, jnp.uint32)
  return b


def _adamw(p32, g32, m32, v32, coef, beta1, beta2, eps, clamp):
  m32 = m32 * beta1 + g32 * (1.0 - beta1)
  v32 = v32 * beta2 + (g32 * g32) * (1.0 - beta2)
  step = m32 / (jnp.sqrt(v32) * coef[2] + eps) * coef[1]
  if clamp:
    step = jnp.clip(step, -coef[4], coef[4])
  return m32, v32, step


def _bf16_states_update(
    p: jax.Array,
    g: jax.Array,
    m: jax.Array,
    v: jax.Array,
    seed: jax.Array,
    coef: jax.Array,
    beta1: float,
    beta2: float,
    eps: float,
    stochastic_param: bool,
    stochastic_states: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """bf16 parameter (stochastic or nearest) with bf16 moments."""
  b = _bits(seed, p.shape)
  m32, v32, step = _adamw(p.astype(jnp.float32), g.astype(jnp.float32), m.astype(jnp.float32),
                          v.astype(jnp.float32), coef, beta1, beta2, eps, False)
  x = p.astype(jnp.float32)
  x = x - x * coef[3] - step
  return (
      _round_bf16(x, b & jnp.uint32(0xFFFF), stochastic_param),
      _round_bf16(m32, b >> 16, stochastic_states),
      _round_bf16(v32, (b >> 8) & jnp.uint32(0xFFFF), stochastic_states),
  )


def _bf16_states_kahan_update(
    p: jax.Array,
    g: jax.Array,
    m: jax.Array,
    v: jax.Array,
    residual: jax.Array,
    seed: jax.Array,
    coef: jax.Array,
    beta1: float,
    beta2: float,
    eps: float,
    stochastic_states: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Kahan-compensated bf16 parameter (parameter + residual) with bf16 moments."""
  b = _bits(seed, p.shape)
  m32, v32, step = _adamw(p.astype(jnp.float32), g.astype(jnp.float32), m.astype(jnp.float32),
                          v.astype(jnp.float32), coef, beta1, beta2, eps, False)
  x = p.astype(jnp.float32) + residual.astype(jnp.float32)
  x = x - x * coef[3] - step
  new_p = x.astype(jnp.bfloat16)
  new_r = _round_bf16(x - new_p.astype(jnp.float32), b >> 16, stochastic_states)
  return (
      new_p,
      _round_bf16(m32, b & jnp.uint32(0xFFFF), stochastic_states),
      _round_bf16(v32, (b >> 8) & jnp.uint32(0xFFFF), stochastic_states),
      new_r,
  )


def _f8_moments(g32, mq, ms, vq, vs, coef, bits, beta1, beta2, eps, block_size, stochastic_states, shape):
  m32 = (mq.astype(jnp.float32) * ms).reshape(shape)
  s32 = vq.astype(jnp.float32) * vs
  v32 = (s32 * s32).reshape(shape)
  m32, v32, step = _adamw(None, g32, m32, v32, coef, beta1, beta2, eps, True)
  nm = (bits.reshape(-1, block_size) >> 12) & jnp.uint32(0xFFFFF)
  nv = (bits.reshape(-1, block_size) >> 6) & jnp.uint32(0xFFFFF)
  new_mq, new_ms = _quant_f8(m32.reshape(-1, block_size), nm, stochastic_states)
  new_vq, new_vs = _quant_f8(jnp.sqrt(v32).reshape(-1, block_size), nv, stochastic_states)
  return step, new_mq, new_ms, new_vq, new_vs


def _f8_states_update(
    p: jax.Array,
    g: jax.Array,
    mq: jax.Array,
    ms: jax.Array,
    vq: jax.Array,
    vs: jax.Array,
    seed: jax.Array,
    coef: jax.Array,
    beta1: float,
    beta2: float,
    eps: float,
    block_size: int,
    stochastic_param: bool,
    stochastic_states: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """bf16 parameter (stochastic or nearest) with blockwise float8 moments."""
  b = _bits(seed, p.shape)
  step, new_mq, new_ms, new_vq, new_vs = _f8_moments(
      g.astype(jnp.float32), mq, ms, vq, vs, coef, b, beta1, beta2, eps, block_size,
      stochastic_states, p.shape)
  x = p.astype(jnp.float32)
  x = x - x * coef[3] - step
  return _round_bf16(x, b & jnp.uint32(0xFFFF), stochastic_param), new_mq, new_ms, new_vq, new_vs


def _f8_states_kahan_update(
    p: jax.Array,
    g: jax.Array,
    mq: jax.Array,
    ms: jax.Array,
    vq: jax.Array,
    vs: jax.Array,
    residual: jax.Array,
    seed: jax.Array,
    coef: jax.Array,
    beta1: float,
    beta2: float,
    eps: float,
    block_size: int,
    stochastic_states: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Kahan-compensated bf16 parameter with blockwise float8 moments."""
  b = _bits(seed, p.shape)
  step, new_mq, new_ms, new_vq, new_vs = _f8_moments(
      g.astype(jnp.float32), mq, ms, vq, vs, coef, b, beta1, beta2, eps, block_size,
      stochastic_states, p.shape)
  x = p.astype(jnp.float32) + residual.astype(jnp.float32)
  x = x - x * coef[3] - step
  new_p = x.astype(jnp.bfloat16)
  new_r = _round_bf16(x - new_p.astype(jnp.float32), b & jnp.uint32(0xFFFF), stochastic_states)
  return new_p, new_mq, new_ms, new_vq, new_vs, new_r


AVAILABLE = pallas is not None
if AVAILABLE:
  bf16_states_update = pallas.jax_op("tt_lpadamw::bf16_states")(_bf16_states_update)
  bf16_states_kahan_update = pallas.jax_op("tt_lpadamw::bf16_states_kahan")(_bf16_states_kahan_update)
  f8_states_update = pallas.jax_op("tt_lpadamw::f8_states")(_f8_states_update)
  f8_states_kahan_update = pallas.jax_op("tt_lpadamw::f8_states_kahan")(_f8_states_kahan_update)
