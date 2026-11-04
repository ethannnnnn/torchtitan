# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""How much of each optimizer update does a training run keep?

At fine-tuning learning rates most AdamW updates of bf16 weights are smaller
than half a bf16 ulp; ``torch.optim.AdamW`` on bf16 parameters rounds them away
(see ``low_precision_adamw.py``). ``UpdateProbe`` measures this inside a real
run, for the run's own gradients: on one rank it samples a few elements of each
probed parameter's local shard and keeps an fp32 "shadow" AdamW for them (fp32
master copies started from the run's weights, fp32 moments), stepped with the
same (already clipped) gradients and the same hyperparameters as the run's
optimizer (learning rate read from the parameter group at every step, betas,
eps, weight decay). After every step it reports, per parameter category and
overall,

* ``kept``: ``<d_run, d_ref> / <d_ref, d_ref>``, where ``d`` is the change of
  the sampled weights since the probe started, ``run`` the run's weights plus
  the Kahan residual when the optimizer keeps one, and ``ref`` the fp32 shadow
  master. 1.0 means the run applied fp32 AdamW's update; plain
  ``torch.optim.AdamW`` on bf16 weights keeps about 0.1 at learning rate 1e-6
  for weights of magnitude ~5e-3.
* ``kept_bf16``: the same with the bf16 weights alone (what the forward pass
  sees; with Kahan they trail the master by less than half an ulp).
* ``dev``: ``|run - ref| / |d_ref|``, the distance between the run's and the
  shadow's weights relative to the shadow's total movement.

The shadow does not feed back into training. Its cost is a few gathers per
probed tensor per step. ``install_update_probe`` hooks every
``OptimizersContainer.step`` so that any TorchTitan trainer config can turn it
on (call it while building the config, e.g. from a recipe function).
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable

import torch

__all__ = ["UpdateProbe", "install_update_probe", "probe_category", "uninstall_update_probe"]

DEFAULT_PATTERN = r"(^|\.)layers\.(0|31|61)\.|tok_embeddings|lm_head|(^|\.)norm\."
"""Probed parameter names by default: three decoder layers, embeddings, head, final norm."""


def _local(t: torch.Tensor) -> torch.Tensor:
  local = getattr(t, "_local_tensor", None)
  return local if local is not None else t


def probe_category(name: str) -> str:
  """Coarse parameter category for the per-category report."""
  if "expert" in name or re.search(r"\.w[123](_EFD|_EDF)?\b", name) and "moe" in name:
    return "experts"
  if "router" in name or ".gate." in name or name.endswith(".gate"):
    return "router"
  if "attention" in name:
    return "attention"
  if "embed" in name or "lm_head" in name:
    return "embed_head"
  return "other"


@dataclass
class _Probed:
  name: str
  category: str
  param: torch.Tensor
  optimizer: torch.optim.Optimizer
  group: dict[str, Any]
  index: tuple[torch.Tensor, ...]
  start: torch.Tensor  # fp32 run weights (+ residual) at the first probed step
  master: torch.Tensor  # fp32 shadow weights
  exp_avg: torch.Tensor
  exp_avg_sq: torch.Tensor
  prev_run: torch.Tensor
  prev_ref: torch.Tensor
  step: int = 0


class UpdateProbe:
  """fp32 shadow AdamW on sampled elements (see the module docstring)."""

  def __init__(self, *, elements: int, pattern: str = DEFAULT_PATTERN, seed: int = 0,
               log: Any = print) -> None:
    if elements <= 0:
      raise ValueError("elements must be positive")
    self.elements = int(elements)
    self.pattern = re.compile(pattern)
    self.seed = int(seed)
    self.log = log
    self.probed: list[_Probed] = []
    self.steps = 0
    self._built = False
    self._grads: list[torch.Tensor | None] = []

  # ------------------------------------------------------------------ setup

  def build(self, named_params: Iterable[tuple[str, torch.Tensor]],
            optimizers: Iterable[torch.optim.Optimizer]) -> None:
    """Pick the probed elements (call once, before the first step)."""
    groups: dict[int, tuple[torch.optim.Optimizer, dict[str, Any]]] = {}
    for opt in optimizers:
      for group in opt.param_groups:
        for p in group["params"]:
          groups[id(p)] = (opt, group)
    gen = torch.Generator().manual_seed(self.seed)
    for name, p in named_params:
      if not p.requires_grad or id(p) not in groups or not self.pattern.search(name):
        continue
      local = _local(p)
      if local.numel() == 0:
        continue
      k = min(self.elements, local.numel())
      flat = torch.randperm(local.numel(), generator=gen)[:k] if k < local.numel() else torch.arange(k)
      index = tuple(i.to(local.device) for i in torch.unravel_index(flat, tuple(local.shape)))
      opt, group = groups[id(p)]
      start = self._run_values(p, opt, index)
      zeros = torch.zeros_like(start)
      self.probed.append(_Probed(
          name=name, category=probe_category(name), param=p, optimizer=opt, group=group,
          index=index, start=start, master=start.clone(), exp_avg=zeros.clone(),
          exp_avg_sq=zeros.clone(), prev_run=start.clone(), prev_ref=start.clone()))
    self._built = True
    if self.probed:
      n = sum(e.start.numel() for e in self.probed)
      cats = sorted({e.category for e in self.probed})
      self.log(f"[UPDATE-PROBE] probing {n} elements of {len(self.probed)} tensors ({', '.join(cats)})")

  @staticmethod
  def _gather(t: torch.Tensor, index: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return _local(t).detach()[index].to(torch.float32).cpu()

  @staticmethod
  def _residual(p: torch.Tensor, opt: torch.optim.Optimizer) -> torch.Tensor | None:
    residual = opt.state.get(p, {}).get("residual") if hasattr(opt, "state") else None
    return residual if isinstance(residual, torch.Tensor) else None

  def _run_values(self, p: torch.Tensor, opt: torch.optim.Optimizer,
                  index: tuple[torch.Tensor, ...]) -> torch.Tensor:
    values = self._gather(p, index)
    residual = self._residual(p, opt)
    if residual is not None:
      values = values + self._gather(residual, index)
    return values

  @staticmethod
  def _gather_many(tensors: list[torch.Tensor | None],
                   indices: list[tuple[torch.Tensor, ...]]) -> list[torch.Tensor | None]:
    """Sampled fp32 values of several tensors with one device-to-host copy per device."""
    out: list[torch.Tensor | None] = [None] * len(tensors)
    by_device: dict[Any, list[int]] = {}
    for i, t in enumerate(tensors):
      if t is not None:
        by_device.setdefault(_local(t).device, []).append(i)
    for ids in by_device.values():
      parts = [_local(tensors[i]).detach()[indices[i]].to(torch.float32) for i in ids]
      host = torch.cat(parts).cpu()
      for i, chunk in zip(ids, torch.split(host, [p.numel() for p in parts])):
        out[i] = chunk
    return out

  # ------------------------------------------------------------------- step

  @staticmethod
  def _lr(group: dict[str, Any]) -> float:
    lr = group["lr"]
    return float(lr.item()) if isinstance(lr, torch.Tensor) else float(lr)

  def before_step(self) -> None:
    """Read the sampled gradients (the run's optimizer step consumes them)."""
    self._grads = self._gather_many([e.param.grad for e in self.probed],
                                    [e.index for e in self.probed])

  def after_step(self, step: int | None = None) -> dict[str, dict[str, float]]:
    """Advance the shadow with the gradients read in ``before_step``; report."""
    if not self.probed:
      return {}
    self.steps += 1
    sums: dict[str, list[float]] = {}
    indices = [e.index for e in self.probed]
    weights_all = self._gather_many([e.param for e in self.probed], indices)
    residual_all = self._gather_many(
        [self._residual(e.param, e.optimizer) for e in self.probed], indices)
    for e, g, weights, residual in zip(self.probed, self._grads, weights_all, residual_all):
      if g is None:
        continue
      lr = self._lr(e.group)
      if lr == 0.0:
        continue
      beta1, beta2 = e.group.get("betas", (0.9, 0.999))
      eps = float(e.group.get("eps", 1e-8))
      wd = float(e.group.get("weight_decay", 0.0))
      e.step += 1
      e.exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
      e.exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
      bc1 = 1.0 - beta1**e.step
      bc2 = 1.0 - beta2**e.step
      update = (e.exp_avg / bc1) / ((e.exp_avg_sq / bc2).sqrt() + eps)
      e.master = e.master - lr * wd * e.master - lr * update
      run = weights if residual is None else weights + residual
      d_run, d_ref, d_bf16 = run - e.start, e.master - e.start, weights - e.start
      s_run, s_ref = run - e.prev_run, e.master - e.prev_ref
      e.prev_run, e.prev_ref = run, e.master.clone()
      for key in (e.category, "all"):
        acc = sums.setdefault(key, [0.0] * 6)
        acc[0] += float((d_run * d_ref).sum())
        acc[1] += float((d_ref * d_ref).sum())
        acc[2] += float((d_bf16 * d_ref).sum())
        acc[3] += float(((run - e.master) ** 2).sum())
        acc[4] += float((s_run * s_ref).sum())
        acc[5] += float((s_ref * s_ref).sum())
    report: dict[str, dict[str, float]] = {}
    for key, (dot, ref2, dot16, dev2, sdot, sref2) in sums.items():
      if ref2 <= 0.0:
        continue
      report[key] = {
          "kept": dot / ref2,
          "kept_bf16": dot16 / ref2,
          "dev": math.sqrt(dev2 / ref2),
          "kept_step": sdot / sref2 if sref2 > 0 else float("nan"),
      }
    if report:
      parts = " ".join(
          f"{k}:kept={v['kept']:.4f},kept_bf16={v['kept_bf16']:.4f},dev={v['dev']:.4f},"
          f"kept_step={v['kept_step']:.3f}"
          for k, v in sorted(report.items(), key=lambda kv: (kv[0] != "all", kv[0]))
      )
      self.log(f"[UPDATE-PROBE] step={step if step is not None else self.steps} {parts}")
    return report


def install_update_probe(elements: int, pattern: str | None = None) -> bool:
  """Run an ``UpdateProbe`` around every ``OptimizersContainer.step`` on global rank 0.

  ``elements`` sampled elements per probed tensor (0 or less: no probe);
  ``pattern`` (default ``DEFAULT_PATTERN``) selects parameters by name. Returns
  whether a probe is installed. Uses torch.optim's global step pre/post hooks,
  so it works whether it is called before or after the optimizer container is
  built.
  """
  if elements <= 0:
    return False
  if _INSTALLED:
    return True
  pattern = pattern or DEFAULT_PATTERN
  # pylint: disable=g-import-not-at-top
  from torch.optim.optimizer import register_optimizer_step_post_hook, register_optimizer_step_pre_hook

  from torchtitan.components.optimizer.optimizer import OptimizersContainer
  # pylint: enable=g-import-not-at-top

  probes: dict[int, UpdateProbe | None] = {}

  def _probe(container) -> UpdateProbe | None:
    if id(container) not in probes:
      probe = None
      if int(os.environ.get("RANK", "0")) == 0:
        probe = UpdateProbe(elements=elements, pattern=pattern,
                            log=lambda msg: print(msg, flush=True))
        named = [(n, p) for part in container.model_parts for n, p in part.named_parameters()]
        probe.build(named, container.optimizers)
      probes[id(container)] = probe
    return probes[id(container)]

  def pre_hook(optimizer, args, kwargs):
    if isinstance(optimizer, OptimizersContainer):
      probe = _probe(optimizer)
      if probe is not None:
        probe.before_step()

  def post_hook(optimizer, args, kwargs):
    if isinstance(optimizer, OptimizersContainer):
      probe = probes.get(id(optimizer))
      if probe is not None:
        probe.after_step()

  _HANDLES.extend([register_optimizer_step_pre_hook(pre_hook),
                   register_optimizer_step_post_hook(post_hook)])
  _set_installed(True)
  return True


def uninstall_update_probe() -> None:
  """Remove the hooks of ``install_update_probe`` (tests)."""
  for handle in _HANDLES:
    handle.remove()
  _HANDLES.clear()
  _set_installed(False)


_INSTALLED = False
_HANDLES: list[Any] = []


def _set_installed(value: bool) -> None:
  global _INSTALLED
  _INSTALLED = value
