# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Write a load-balanced expert placement during training, from measured routing.

With the released weights Spark's routing is skewed: on natural text the
hottest expert of a layer receives 13 to 24 times the mean expert's routes, and
with 8-way expert parallelism the busiest rank receives about 2.1 (contiguous
placement) or 2.4 (interleaved) times the mean. A fixed receive buffer then
drops routes unless the placement balances the ranks. A placement measured on
the training text held every route on all 128 ranks of a 4x4x4 slice at a
1.5x buffer for about 13 LoRA steps; routing drifts as training proceeds, so
long runs use a 2x buffer or re-measure.

``install_placement_writer`` hooks a trainer config: train a few steps with
``SPARK_EXPERT_PLACEMENT_OUT=<path>`` (expert parallel degree 1 is fine; the
counts come from the router, before dispatch). At step
``SPARK_EXPERT_STATS_STEP`` (default 2) every rank all-reduces its MoE layers'
``tokens_per_expert_E`` counters and rank 0 bin-packs them for
``SPARK_EXPERT_STATS_EP`` (default 8) ranks with
``expert_parallel.balanced_expert_placement``, prints the skew, and writes the
JSON read by ``ep_expert_placement="file:<path>"``.

``install_ep_monitor`` prints the dispatchers' receive-buffer fill and largest
dropped-route fraction every step (rank 0, and any rank whose buffer
overflowed): ``[EP] rank=R step=S max_buffer_fill=F max_dropped_frac=D``. A fill
above 1.0 means routes were dropped on that rank.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from .expert_parallel import balanced_expert_placement, busiest_rank_load, write_expert_placement


def skew_summary(counts_LE, ep_degree: int) -> dict[str, float]:
  """Busiest-rank / mean load (contiguous and balanced placements) and hottest expert / mean."""
  # pylint: disable-next=g-import-not-at-top
  import torch

  counts = torch.as_tensor(counts_LE, dtype=torch.float64)
  balanced = balanced_expert_placement(counts, ep_degree)
  contiguous = busiest_rank_load(counts, None, ep_degree)
  greedy = busiest_rank_load(counts, balanced, ep_degree)
  hottest = (counts.max(dim=1).values / counts.mean(dim=1).clamp(min=1e-30)).tolist()
  return {"contiguous_max": max(contiguous), "contiguous_mean": sum(contiguous) / len(contiguous),
          "balanced_max": max(greedy), "balanced_mean": sum(greedy) / len(greedy),
          "hottest_expert_max": max(hottest)}


def _measure_and_write(model: Any, step: int, path: str, ep_degree: int) -> None:
  # pylint: disable-next=g-import-not-at-top
  import torch
  import torch.distributed as dist

  counters = [m.tokens_per_expert_E for m in model.modules() if hasattr(m, "tokens_per_expert_E")]
  if not counters:
    return
  local = torch.stack([(c.to_local() if hasattr(c, "to_local") else c).float() for c in counters])
  if dist.is_available() and dist.is_initialized():
    dist.all_reduce(local)
  if int(os.environ.get("RANK", "0")) != 0:
    return
  counts = local.cpu()
  summary = skew_summary(counts, ep_degree)
  print(f"[EXPERT-PLACEMENT] step={step} layers={counts.shape[0]} ep={ep_degree} busiest-rank/mean: "
        f"contiguous {summary['contiguous_max']:.2f} max / {summary['contiguous_mean']:.2f} mean, balanced "
        f"{summary['balanced_max']:.2f} / {summary['balanced_mean']:.2f}; hottest expert / mean "
        f"{summary['hottest_expert_max']:.1f}", flush=True)
  if not os.path.exists(path):
    directory = os.path.dirname(path)
    if directory:
      os.makedirs(directory, exist_ok=True)
    write_expert_placement(path, balanced_expert_placement(counts, ep_degree), ep_degree,
                           source=f"router counts summed over all ranks up to step {step}")
    print(f"[EXPERT-PLACEMENT] wrote {path}", flush=True)


def install_placement_writer(cfg: Any) -> Any:
  """Write a measured placement during training when ``SPARK_EXPERT_PLACEMENT_OUT`` is set.

  Wraps the model spec's ``parallelize_fn`` to keep a handle on the model and
  hooks TorchTitan's metrics logging (called on every rank each step).
  """
  path = os.environ.get("SPARK_EXPERT_PLACEMENT_OUT", "")
  if not path:
    return cfg
  step_to_measure = int(os.environ.get("SPARK_EXPERT_STATS_STEP", "2"))
  ep_degree = int(os.environ.get("SPARK_EXPERT_STATS_EP", "8"))
  holder: list[Any] = []
  parallelize: Callable[..., Any] = cfg.model_spec.parallelize_fn

  def parallelize_and_keep(model, *args, **kwargs):
    out = parallelize(model, *args, **kwargs)
    holder[:] = [out if out is not None else model]
    return out

  cfg.model_spec.parallelize_fn = parallelize_and_keep
  # pylint: disable-next=g-import-not-at-top
  from torchtitan.components.metrics import MetricsProcessor

  if not getattr(MetricsProcessor.log, "_expert_placement", False):
    original = MetricsProcessor.log

    def log(self, step, *args, **kwargs):
      out = original(self, step, *args, **kwargs)
      if step == step_to_measure and holder:
        _measure_and_write(holder[0], step, path, ep_degree)
      return out

    log._expert_placement = True
    MetricsProcessor.log = log
  return cfg


def install_ep_monitor() -> None:
  """Print the expert-parallel buffer fill and dropped-route fraction after every step."""
  # pylint: disable-next=g-import-not-at-top
  from torchtitan.components.metrics import MetricsProcessor

  if getattr(MetricsProcessor.log, "_ep_monitor", False):
    return
  original = MetricsProcessor.log

  def log(self, step, *args, **kwargs):
    out = original(self, step, *args, **kwargs)
    # pylint: disable-next=g-import-not-at-top
    from torchtitan.experiments.tpu import ragged_ep_dispatcher as dispatchers

    try:
      fill = dispatchers.max_buffer_fill()
      dropped = dispatchers.max_dropped_fraction()
    except Exception as error:  # pragma: no cover - monitoring must not stop training
      print(f"[EP] monitor read failed: {error!r}"[:200], flush=True)
      return out
    if fill is not None and (int(os.environ.get("RANK", "0")) == 0 or fill > 1.0):
      extra = f" max_dropped_frac={dropped:.4f}" if dropped is not None else ""
      flag = " OVERFLOW" if fill > 1.0 else ""
      print(f"[EP] rank={os.environ.get('RANK', '0')} step={step} max_buffer_fill={fill:.4f}{extra}{flag}",
            flush=True)
    return out

  log._ep_monitor = True
  MetricsProcessor.log = log
