# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AdamW for bfloat16 parameters without an fp32 master copy.

``torch.optim.AdamW`` on bf16 parameters rounds every updated parameter to the
nearest bf16 value. bf16 keeps 8 significant bits, so a change smaller than half
a unit in the last place (2**-9 of the magnitude) is dropped. Fine-tuning large
models at small learning rates loses most of each update this way: with
``|w| ~ 4e-3`` and ``lr = 1e-6`` about 90% of the intended change disappears.

``LowPrecisionAdamW`` keeps the update without an fp32 copy of the weights.

``param_update`` selects how the fp32 result is written back to bf16:

* ``"stochastic"``: stochastic rounding. The result is rounded to one of its two
  neighbouring bf16 values with probabilities proportional to closeness, so
  ``E[round(x)] = x`` and sub-ulp updates are applied in expectation. No extra
  memory.
* ``"kahan"``: compensated summation. A bf16 buffer keeps each parameter's
  rounding residual and adds it back before the next update (2 extra bytes per
  parameter). Parameter + residual carries ~16 significant bits (fp32 has 24)
  and the bf16 parameter is its rounding, like the bf16 copy of an fp32 master
  weight; until the accumulated change crosses a rounding boundary it waits in
  the residual. The residual itself is rounded with ``state_rounding``.
* ``"nearest"``: round to nearest even, i.e. the stock behaviour (for
  comparisons).

``state_dtype`` selects the storage of the two Adam moments:

* ``"bfloat16"`` (rounded with ``state_rounding``), ``"float32"``, or
* ``"float8"``: 8-bit moments, ``float8_e4m3fn`` values with one fp32 scale per
  block of ``block_size`` elements (the block maximum maps to 448). The second
  moment is stored as its square root, which halves its dynamic range. Blocks
  run along the last dimension, so the codes have the parameter's shape and the
  scales its shape with the last dimension divided by ``block_size``; both are
  DTensors with the parameter's placements and checkpoint like any sharded
  state. Parameters whose last dimension is sharded or not a multiple of
  ``block_size`` keep bf16 moments. With
  8-bit moments the normalized update ``|m_hat| / sqrt(v_hat)`` is clamped to the
  largest value exact Adam can produce for the given betas (a Cauchy-Schwarz
  bound, about 1.16 for betas (0.9, 0.95)); the clamp is inactive in exact
  arithmetic and only guards against an under-represented second moment.

``state_rounding`` is ``"stochastic"`` (default) or ``"nearest"``.

All arithmetic is fp32 on the local shard of each parameter: DTensor parameters
(FSDP, expert parallel) are updated through their local tensors, which needs no
communication because AdamW is elementwise. Random bits come from
``torch.randint`` on the parameter's device and are xor-ed with a per-rank
constant so that ranks holding same-shaped shards do not share a rounding
pattern. Per-step scalars (learning rate, bias corrections) are passed as
device tensors on accelerators so that changing values never changes a traced
graph.

Memory per parameter, Adam-type training (weights + gradient + moments [+
residual]): fp32 everything 16 bytes; bf16 everything 8 bytes (stochastic or
nearest); bf16 + 8-bit moments 6 bytes (stochastic) or 8 bytes (kahan).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch
from torch.optim import Optimizer

from torchtitan.components.optimizer.optimizer import (
    OptimizersContainer,
    ParamGroupConfig,
)

__all__ = [
    "LowPrecisionAdamW",
    "LowPrecisionOptimizersContainer",
    "adam_ratio_bound",
    "low_precision_adamw_config",
    "quantize_float8_blockwise",
    "dequantize_float8_blockwise",
    "stochastic_round_to_bf16",
    "stochastic_round_to_float8",
]

PARAM_UPDATES = ("stochastic", "kahan", "nearest")
STATE_DTYPES = ("bfloat16", "float32", "float8")
STATE_ROUNDINGS = ("stochastic", "nearest")
SR_IMPLS = ("bits", "arith")
TPU_KERNELS = ("auto", "on", "off")

_F8 = torch.float8_e4m3fn
_F8_MAX = 448.0
_F8_MIN_NORMAL = 2.0**-6
_F8_SUBNORMAL_STEP = 2.0**-9
# int32 views of the masks 0xFFFF0000 and 0xFFF00000.
_KEEP_BF16 = -(1 << 16)
_KEEP_E4M3 = -(1 << 20)


def _local(t: torch.Tensor) -> torch.Tensor:
  """The local shard of a DTensor (the tensor itself otherwise)."""
  local = getattr(t, "_local_tensor", None)
  return local if local is not None else t


def _contiguous_stride(shape) -> tuple[int, ...]:
  stride, acc = [], 1
  for d in reversed(tuple(shape)):
    stride.append(acc)
    acc *= d
  return tuple(reversed(stride))


def _param_like(p: torch.Tensor, local: torch.Tensor, global_shape) -> torch.Tensor:
  """``local`` as a DTensor with ``p``'s mesh and placements (``local`` itself if ``p`` is a plain tensor)."""
  if getattr(p, "_local_tensor", None) is None:
    return local
  from torch.distributed.tensor import DTensor

  shape = torch.Size(global_shape)
  return DTensor.from_local(local, p.device_mesh, p.placements, run_check=False, shape=shape,
                            stride=_contiguous_stride(shape))


def _float8_layout_ok(p: torch.Tensor, block_size: int) -> bool:
  """Blocks along the last dimension need it unsharded and divisible by ``block_size``."""
  local = _local(p)
  if local.ndim == 0 or local.numel() == 0 or local.shape[-1] % block_size:
    return False
  for placement in getattr(p, "placements", ()):
    dim = getattr(placement, "dim", None)
    if dim is not None and dim % p.ndim == p.ndim - 1:
      return False
  return True


def _state_format(state: dict[str, Any]) -> str:
  if "exp_avg_f8" in state:
    return "float8"
  return "float32" if state["exp_avg"].dtype == torch.float32 else "bfloat16"


def _rank() -> int:
  if torch.distributed.is_available() and torch.distributed.is_initialized():
    return torch.distributed.get_rank()
  return 0


def _random_bits(like: torch.Tensor, nbits: int, salt: int) -> torch.Tensor:
  """int32 tensor of independent uniform integers in [0, 2**nbits)."""
  bits = torch.randint(0, 1 << nbits, like.shape, dtype=torch.int32, device=like.device)
  salt &= (1 << nbits) - 1
  if salt:
    bits = torch.bitwise_xor(bits, salt)  # a bijection: still uniform
  return bits


def stochastic_round_to_bf16(x: torch.Tensor, rand16: torch.Tensor) -> torch.Tensor:
  """Rounds float32 ``x`` to bfloat16 stochastically.

  ``rand16`` holds uniform integers in [0, 2**16) (int32, shape of ``x``).
  Adding them to the 16 low bits that bf16 drops carries into the kept bits with
  probability (dropped bits) / 2**16, i.e. the magnitude is rounded away from
  zero with probability equal to the discarded fraction. Sign-magnitude makes
  this correct for negative values too; infinities stay infinite.
  """
  bits = x.to(torch.float32).contiguous().view(torch.int32)
  bits = torch.bitwise_and(bits + rand16, _KEEP_BF16)
  return bits.view(torch.float32).to(torch.bfloat16)


def _stochastic_round_to_bf16_arith(x: torch.Tensor, uniform: torch.Tensor) -> torch.Tensor:
  """Stochastic rounding to bf16 without bit casts (fallback).

  Adds uniform noise of one bf16 spacing, ``(u - 0.5) * 2**(floor(log2|x|) - 7)``,
  and rounds to nearest. Exact SR except within a quarter spacing above a power
  of two, where the spacing below is halved.
  """
  x = x.to(torch.float32)
  exponent = torch.floor(torch.log2(x.abs().clamp_min(1e-30)))
  spacing = torch.exp2(exponent - 7.0)
  return (x + (uniform - 0.5) * spacing).to(torch.bfloat16)


def stochastic_round_to_float8(x: torch.Tensor, rand20: torch.Tensor) -> torch.Tensor:
  """Rounds float32 ``x`` (already scaled into [-448, 448]) to float8_e4m3fn stochastically.

  e4m3 keeps 3 of float32's 23 mantissa bits: random bits are added to the 20
  dropped bits as in :func:`stochastic_round_to_bf16`. Below the smallest normal
  e4m3 value (2**-6) the grid has a fixed spacing of 2**-9; there the value is
  dithered by a uniform fraction of that spacing and rounded to nearest, which is
  the same stochastic rounding.
  """
  x = x.to(torch.float32).contiguous()
  bits = torch.bitwise_and(x.view(torch.int32) + rand20, _KEEP_E4M3).view(torch.float32)
  dither = (rand20.to(torch.float32) * (2.0**-20) - 0.5) * _F8_SUBNORMAL_STEP
  y = torch.where(x.abs() < _F8_MIN_NORMAL, x + dither, bits)
  return y.clamp(-_F8_MAX, _F8_MAX).to(_F8)


def quantize_float8_blockwise(
    x: torch.Tensor, block_size: int, rand20: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
  """Blockwise float8_e4m3fn quantization of a float32 tensor.

  Returns ``(codes [n / block_size, block_size] float8_e4m3fn, scales
  [n / block_size, 1] float32)`` with ``x ~= codes * scales``. ``rand20`` (int32
  uniform in [0, 2**20), shape of the codes) selects stochastic rounding; None
  rounds to nearest.
  """
  blocks = x.to(torch.float32).reshape(-1, block_size)
  absmax = blocks.abs().amax(dim=1, keepdim=True)
  scales = torch.where(absmax > 0, absmax / _F8_MAX, torch.ones_like(absmax))
  scaled = blocks / scales
  if rand20 is None:
    codes = scaled.clamp(-_F8_MAX, _F8_MAX).to(_F8)
  else:
    codes = stochastic_round_to_float8(scaled, rand20)
  return codes, scales


def dequantize_float8_blockwise(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
  return codes.to(torch.float32) * scales


def adam_ratio_bound(beta1: float, beta2: float, step: int) -> float:
  """Largest ``|m_hat| / sqrt(v_hat)`` that exact Adam can produce at ``step``.

  With ``m_t = (1 - b1) sum b1^(t-k) g_k`` and ``v_t = (1 - b2) sum b2^(t-k) g_k^2``,
  Cauchy-Schwarz gives ``|m_t| <= (1 - b1) / sqrt(1 - b2) * sqrt(sum_j r^j) *
  sqrt(v_t)`` with ``r = b1^2 / b2`` and j < t; the bias corrections scale the
  ratio by ``sqrt(1 - b2^t) / (1 - b1^t)``.
  """
  r = beta1 * beta1 / beta2
  series = float(step) if abs(1.0 - r) < 1e-12 else (1.0 - r**step) / (1.0 - r)
  bound = (1.0 - beta1) / math.sqrt(1.0 - beta2) * math.sqrt(series)
  return bound * math.sqrt(1.0 - beta2**step) / (1.0 - beta1**step)


class LowPrecisionAdamW(Optimizer):
  """AdamW for bf16 parameters with stochastic-rounded or compensated updates.

  See the module docstring for ``param_update``, ``state_dtype`` and
  ``state_rounding``. ``sr_impl`` selects the stochastic-rounding kernel:
  ``"bits"`` (integer add on the float32 bit pattern, exact) or ``"arith"``
  (additive noise of one bf16 spacing, no bit casts). fp32 parameters get a
  plain fp32 AdamW update. ``tpu_kernel`` (``"auto"``, ``"on"``, ``"off"``)
  runs the update of bf16 parameters with bf16 or float8 moments on TPU as one
  fused program per parameter (``kernels/low_precision_adamw_kernels.py``;
  random bits from ``jax.random`` instead of ``torch.randint``, same
  distribution); ``"auto"`` uses it when the parameter lives on a TPU and the
  kernel module imports. ``foreach`` / ``fused`` arguments (passed by
  TorchTitan's optimizer container) are accepted and ignored.
  """

  def __init__(
      self,
      params: Iterable[Any],
      lr: float = 1e-3,
      betas: tuple[float, float] = (0.9, 0.999),
      eps: float = 1e-8,
      weight_decay: float = 0.0,
      *,
      param_update: str = "stochastic",
      state_dtype: str = "bfloat16",
      state_rounding: str = "stochastic",
      block_size: int = 256,
      sr_impl: str = "bits",
      tpu_kernel: str = "auto",
      foreach: bool | None = None,
      fused: bool | None = None,
  ) -> None:
    del foreach, fused
    if param_update not in PARAM_UPDATES:
      raise ValueError(f"param_update must be one of {PARAM_UPDATES}, got {param_update!r}")
    if state_dtype not in STATE_DTYPES:
      raise ValueError(f"state_dtype must be one of {STATE_DTYPES}, got {state_dtype!r}")
    if state_rounding not in STATE_ROUNDINGS:
      raise ValueError(f"state_rounding must be one of {STATE_ROUNDINGS}, got {state_rounding!r}")
    if sr_impl not in SR_IMPLS:
      raise ValueError(f"sr_impl must be one of {SR_IMPLS}, got {sr_impl!r}")
    if tpu_kernel not in TPU_KERNELS:
      raise ValueError(f"tpu_kernel must be one of {TPU_KERNELS}, got {tpu_kernel!r}")
    if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
      raise ValueError(f"invalid betas {betas}")
    if block_size <= 0:
      raise ValueError("block_size must be positive")
    defaults = dict(
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        param_update=param_update,
        state_dtype=state_dtype,
        state_rounding=state_rounding,
        block_size=block_size,
        sr_impl=sr_impl,
        tpu_kernel=tpu_kernel,
    )
    super().__init__(params, defaults)
    rank = _rank()
    # Odd multipliers spread consecutive ranks over the 16- and 20-bit fields.
    self._salt16 = (rank * 0x9E37 + 0x79B9) & 0xFFFF
    self._salt20 = (rank * 0x9E3779 + 0x7F4A7) & 0xFFFFF
    self._salt31 = (rank * 0x9E3779B1 + 0x632BE5AB) & 0x7FFFFFFF
    self._kernels: Any = None

  def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    """``Optimizer.load_state_dict`` without casting the states to the parameter dtype.

    PyTorch casts every floating-point state tensor to its parameter's dtype on
    load, which would turn float8 codes and float32 scales or moments of a bf16
    parameter into bf16. The loaded tensors are put back as stored.
    """
    saved_ids = [i for g in state_dict["param_groups"] for i in g["params"]]
    params = [p for g in self.param_groups for p in g["params"]]
    super().load_state_dict(state_dict)
    id_map = dict(zip(saved_ids, params))
    for pid, saved in state_dict["state"].items():
      p = id_map.get(pid)
      if p is None:
        continue
      for key, value in saved.items():
        if key != "step" and isinstance(value, torch.Tensor) and self.state[p][key].dtype != value.dtype:
          self.state[p][key] = value.to(device=_local(p).device)

  # ---------------------------------------------------------------- helpers

  def _round_bf16(self, x: torch.Tensor, stochastic: bool, sr_impl: str) -> torch.Tensor:
    if not stochastic:
      return x.to(torch.bfloat16)
    if sr_impl == "arith":
      return _stochastic_round_to_bf16_arith(x, torch.rand(x.shape, device=x.device))
    return stochastic_round_to_bf16(x, _random_bits(x, 16, self._salt16))

  def _init_state(self, p: torch.Tensor, group: dict[str, Any]) -> dict[str, Any]:
    state = self.state[p]
    local = _local(p)
    state["step"] = torch.tensor(0.0)
    fmt = group["state_dtype"]
    bs = group["block_size"]
    if fmt == "float8" and not _float8_layout_ok(p, bs):
      fmt = "bfloat16"
    if fmt == "float8":
      scale_local = tuple(local.shape[:-1]) + (local.shape[-1] // bs,)
      scale_global = tuple(p.shape[:-1]) + (p.shape[-1] // bs,)
      for name in ("exp_avg", "exp_avg_sq_sqrt"):
        codes = torch.zeros(local.shape, dtype=_F8, device=local.device)
        scales = torch.ones(scale_local, dtype=torch.float32, device=local.device)
        state[f"{name}_f8"] = _param_like(p, codes, p.shape)
        state[f"{name}_scale"] = _param_like(p, scales, scale_global)
    else:
      dtype = torch.float32 if fmt == "float32" else torch.bfloat16
      # zeros_like keeps the DTensor placement, so these states checkpoint like
      # torch.optim.AdamW's.
      state["exp_avg"] = torch.zeros_like(p, dtype=dtype, memory_format=torch.preserve_format)
      state["exp_avg_sq"] = torch.zeros_like(p, dtype=dtype, memory_format=torch.preserve_format)
    if group["param_update"] == "kahan" and p.dtype == torch.bfloat16:
      state["residual"] = torch.zeros_like(p, memory_format=torch.preserve_format)
    return state

  @staticmethod
  def _lr(group: dict[str, Any]) -> float:
    lr = group["lr"]
    return float(lr.item()) if isinstance(lr, torch.Tensor) else float(lr)

  def _coefficients(
      self, group: dict[str, Any], step: int, device: torch.device, cache: dict
  ) -> tuple[Any, ...]:
    """(lr, lr / bias_correction1, 1 / sqrt(bias_correction2), lr * weight_decay, lr * ratio bound)."""
    key = (step, str(device))
    if key in cache:
      return cache[key]
    lr = self._lr(group)
    beta1, beta2 = group["betas"]
    bc1 = 1.0 - beta1**step
    bc2 = 1.0 - beta2**step
    values = (
        lr,
        lr / bc1,
        1.0 / math.sqrt(bc2),
        lr * group["weight_decay"],
        lr * adam_ratio_bound(beta1, beta2, step),
    )
    packed = None
    if device.type != "cpu":
      packed = torch.tensor(values, dtype=torch.float32, device=device)
      values = tuple(packed[i] for i in range(len(values)))
    cache[key] = (values, packed)
    return cache[key]

  # ------------------------------------------------------------------- step

  @torch.no_grad()
  def step(self, closure: Callable[[], float] | None = None) -> float | None:
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()
    for group in self.param_groups:
      cache: dict = {}
      for p in group["params"]:
        if p.grad is None:
          continue
        if p.grad.is_sparse:
          raise RuntimeError("LowPrecisionAdamW does not support sparse gradients")
        state = self.state[p]
        if len(state) == 0:
          state = self._init_state(p, group)
        if self._lr(group) == 0.0:
          # Zero learning rate (e.g. TorchTitan's init_optim_state before a
          # checkpoint load): materialize the state only, change nothing.
          continue
        state["step"] += 1
        step = int(state["step"].item())
        coef, packed = self._coefficients(group, step, _local(p).device, cache)
        kernels = self._tpu_kernels(group, p, state) if packed is not None else None
        if kernels is not None:
          self._update_fused(kernels, p, state, group, packed)
        else:
          self._update(p, state, group, coef)
    return loss

  def _tpu_kernels(self, group: dict[str, Any], p: torch.Tensor, state: dict[str, Any]) -> Any:
    """The fused TPU kernel module when it applies to this parameter, else None."""
    mode = group["tpu_kernel"]
    if mode == "off" or _local(p).device.type != "tpu" or group["sr_impl"] != "bits":
      return None
    if _local(p).dtype != torch.bfloat16 or _state_format(state) == "float32":
      return None
    if self._kernels is None:
      try:
        from torchtitan.experiments.tpu.kernels import low_precision_adamw_kernels as kernels

        if not kernels.AVAILABLE:
          raise ImportError("torch_tpu is not importable")
      except ImportError:
        if mode == "on":
          raise
        kernels = False
      self._kernels = kernels
    return self._kernels or None

  def _update_fused(self, kernels: Any, p: torch.Tensor, state: dict[str, Any], group: dict[str, Any],
                    coef: torch.Tensor) -> None:
    """The same update as ``_update`` in one fused TPU program per parameter."""
    beta1, beta2 = group["betas"]
    eps = float(group["eps"])
    stochastic_states = group["state_rounding"] == "stochastic"
    p_l = _local(p)
    g_l = _local(p.grad)
    seed = torch.randint(0, 0x7FFFFFFF, (4,), dtype=torch.int32, device=p_l.device)
    seed = torch.bitwise_xor(seed, self._salt31)
    kahan = group["param_update"] == "kahan"
    residual = _local(state["residual"]) if kahan else None
    if _state_format(state) == "float8":
      bs = int(group["block_size"])
      mq, vq = _local(state["exp_avg_f8"]), _local(state["exp_avg_sq_sqrt_f8"])
      ms, vs = _local(state["exp_avg_scale"]), _local(state["exp_avg_sq_sqrt_scale"])
      mq, vq = mq.view(-1, bs), vq.view(-1, bs)
      ms, vs = ms.view(-1, 1), vs.view(-1, 1)
      if kahan:
        new_p, nmq, nms, nvq, nvs, new_r = kernels.f8_states_kahan_update(
            p_l, g_l, mq, ms, vq, vs, residual, seed, coef, float(beta1), float(beta2), eps, bs,
            stochastic_states)
        residual.copy_(new_r)
      else:
        new_p, nmq, nms, nvq, nvs = kernels.f8_states_update(
            p_l, g_l, mq, ms, vq, vs, seed, coef, float(beta1), float(beta2), eps, bs,
            group["param_update"] == "stochastic", stochastic_states)
      mq.copy_(nmq.view(mq.shape))
      ms.copy_(nms.view(ms.shape))
      vq.copy_(nvq.view(vq.shape))
      vs.copy_(nvs.view(vs.shape))
    else:
      m_l = _local(state["exp_avg"])
      v_l = _local(state["exp_avg_sq"])
      if kahan:
        new_p, nm, nv, new_r = kernels.bf16_states_kahan_update(
            p_l, g_l, m_l, v_l, residual, seed, coef, float(beta1), float(beta2), eps, stochastic_states)
        residual.copy_(new_r)
      else:
        new_p, nm, nv = kernels.bf16_states_update(
            p_l, g_l, m_l, v_l, seed, coef, float(beta1), float(beta2), eps,
            group["param_update"] == "stochastic", stochastic_states)
      m_l.copy_(nm)
      v_l.copy_(nv)
    p_l.copy_(new_p)

  def _update(self, p: torch.Tensor, state: dict[str, Any], group: dict[str, Any], coef) -> None:
    _, step_size, inv_sqrt_bc2, lr_wd, step_bound = coef
    beta1, beta2 = group["betas"]
    eps = group["eps"]
    stochastic_states = group["state_rounding"] == "stochastic"
    sr_impl = group["sr_impl"]
    fmt = _state_format(state)
    bs = group["block_size"]
    p_l = _local(p)
    g32 = _local(p.grad).to(torch.float32)

    # Moments (fp32 math). float8 codes are viewed as [blocks, block_size].
    if fmt == "float8":
      mq, vq = _local(state["exp_avg_f8"]).view(-1, bs), _local(state["exp_avg_sq_sqrt_f8"]).view(-1, bs)
      ms, vs = _local(state["exp_avg_scale"]).view(-1, 1), _local(state["exp_avg_sq_sqrt_scale"]).view(-1, 1)
      m32 = dequantize_float8_blockwise(mq, ms).view(p_l.shape)
      s32 = dequantize_float8_blockwise(vq, vs)
      v32 = (s32 * s32).view(p_l.shape)
    else:
      m32 = _local(state["exp_avg"]).to(torch.float32)
      v32 = _local(state["exp_avg_sq"]).to(torch.float32)
    m32 = m32 * beta1 + g32 * (1.0 - beta1)
    v32 = v32 * beta2 + (g32 * g32) * (1.0 - beta2)

    # Normalized update: m_hat / (sqrt(v_hat) + eps), times lr below.
    # lr * m_hat / (sqrt(v_hat) + eps) = (lr / bc1) * m / (sqrt(v) / sqrt(bc2) + eps)
    step_vec = m32 / (v32.sqrt() * inv_sqrt_bc2 + eps) * step_size
    if fmt == "float8":
      step_vec = torch.clamp(step_vec, -step_bound, step_bound)

    # Parameter.
    if p_l.dtype == torch.float32:
      p_l.sub_(p_l * lr_wd + step_vec)
    else:
      x = p_l.to(torch.float32)
      residual = state.get("residual")
      if residual is not None:
        x = x + _local(residual).to(torch.float32)
      x = x - x * lr_wd - step_vec
      mode = group["param_update"]
      if mode == "stochastic":
        new = self._round_bf16(x, True, sr_impl)
      else:
        new = x.to(p_l.dtype)
      if residual is not None:
        # The residual is itself rounded to bf16; round it stochastically
        # (with stochastic state rounding) so that sub-ulp updates are not lost
        # once the residual has grown to about half a parameter ulp.
        _local(residual).copy_(
            self._round_bf16(x - new.to(torch.float32), stochastic_states, sr_impl)
        )
      p_l.copy_(new)

    # Store the moments.
    if fmt == "float8":
      r1 = _random_bits(mq, 20, self._salt20) if stochastic_states else None
      codes, scales = quantize_float8_blockwise(m32, bs, r1)
      mq.copy_(codes)
      ms.copy_(scales)
      r2 = _random_bits(vq, 20, self._salt20 ^ 0x5A5A5) if stochastic_states else None
      codes, scales = quantize_float8_blockwise(v32.sqrt(), bs, r2)
      vq.copy_(codes)
      vs.copy_(scales)
    else:
      m_l = _local(state["exp_avg"])
      v_l = _local(state["exp_avg_sq"])
      if m_l.dtype == torch.float32:
        m_l.copy_(m32)
        v_l.copy_(v32)
      else:
        m_l.copy_(self._round_bf16(m32, stochastic_states, sr_impl))
        v_l.copy_(self._round_bf16(v32, stochastic_states, sr_impl))


def build_low_precision_adamw(params: Iterable[Any], **kwargs: Any) -> LowPrecisionAdamW:
  return LowPrecisionAdamW(params, **kwargs)


class LowPrecisionOptimizersContainer(OptimizersContainer):
  """``OptimizersContainer`` that also resolves ``"LowPrecisionAdamW"``."""

  @dataclass(kw_only=True, slots=True)
  class Config(OptimizersContainer.Config):
    pass

  @staticmethod
  def _resolve_optimizer_factory(name: str) -> Callable[..., Optimizer]:
    if name == "LowPrecisionAdamW":
      return build_low_precision_adamw
    return OptimizersContainer._resolve_optimizer_factory(name)


def low_precision_adamw_config(
    *,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 0.1,
    param_update: str = "stochastic",
    state_dtype: str = "bfloat16",
    state_rounding: str = "stochastic",
    block_size: int = 256,
    sr_impl: str = "bits",
    tpu_kernel: str = "auto",
) -> OptimizersContainer.Config:
  """One catch-all ``LowPrecisionAdamW`` group (defaults match ``default_adamw``)."""
  group = ParamGroupConfig(
      pattern=r".*",
      optimizer_name="LowPrecisionAdamW",
      optimizer_kwargs=dict(
          lr=lr,
          betas=betas,
          eps=eps,
          weight_decay=weight_decay,
          param_update=param_update,
          state_dtype=state_dtype,
          state_rounding=state_rounding,
          block_size=block_size,
          sr_impl=sr_impl,
          tpu_kernel=tpu_kernel,
      ),
  )
  return LowPrecisionOptimizersContainer.Config(param_groups=[group], implementation="for-loop")
