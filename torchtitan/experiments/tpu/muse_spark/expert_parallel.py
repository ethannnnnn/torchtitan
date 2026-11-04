# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Expert parallelism (EP > 1) for the Muse Spark text decoder on TPU.

Layout (TorchTitan meshes, ``ParallelDims``):

* Routed-expert weights ``[E, ...]`` are sharded on the expert dimension over
  the ``ep`` mesh axis (``Shard(0)``, via the standard MoE sharding configs and
  ``Module.parallelize``) and FSDP-sharded over the remaining ``efsdp`` axis
  (``world / ep`` ranks per EP rank) by SimpleFSDP. Each rank all-gathers only
  its ``E / ep`` local experts (over ``efsdp``) instead of all ``E``.
* Every other weight, including the post-expert norm that all experts share
  (``moe.routed_experts.output_norm``), the router, the pooler / unpooler and
  the pre-expert norm, stays a dense SimpleFSDP parameter over all ranks. Its
  gradient is reduced over the full dense mesh, which is what the shared norm
  needs: under EP each rank's partial gradient covers a different subset of
  the routed rows, and the sum over all ranks covers each routed row of the
  global batch exactly once (see ``MuseSparkRoutedExperts``).
* Tokens move with ``RaggedAllToAllTokenDispatcher`` (``torch.tpu.ops.
  ragged_all_to_all`` on TPU; a host-split reference on CPU; dropless up to a
  fixed receive buffer) or ``AllGatherEPTokenDispatcher`` (all-gather of the
  EP group's tokens, reduce-scatter combine; no all-to-all): static shapes,
  device-side split sizes. EP rank ``g`` owns experts
  ``[g * E / ep, (g + 1) * E / ep)``, which matches the dim-0 ``Shard`` order
  of the expert weights.

The ``ep`` axis is the innermost mesh axis, so with the TPU topology mesh an EP
group of 8 is the 8 TensorCores of one TPU7x host (4 chips) and ``efsdp`` spans
hosts; EP = world size removes expert FSDP entirely.
"""

from __future__ import annotations

import spmd_types as spmd
from torch import nn

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.models.common.moe_sharding import set_moe_sharding_config

__all__ = [
    "balanced_expert_placement",
    "busiest_rank_load",
    "load_expert_placement",
    "write_expert_placement",
    "EXPERT_PARAM_LAYOUT",
    "apply_expert_data_parallel",
    "configure_expert_parallel",
]

# Dense in/out-dim placements of the canonical grouped-expert weights, only
# read on the (unsupported here) TP path; the EP path shards dim 0 on ``ep``.
EXPERT_PARAM_LAYOUT: dict[str, spmd.PerMeshAxisSpmdType] = {
    "w1_EFD": spmd.S(1),
    "w2_EDF": spmd.S(2),
    "w3_EFD": spmd.S(1),
}


def configure_expert_parallel(
    model_config,
    *,
    ep_degree: int,
    capacity_factor: float = 1.5,
    buffer_multiple: int = 1024,
    dispatcher: str = "ragged",
    placement: str = "contiguous",
) -> None:
  """Prepares every MoE layer of a ``MuseSparkModel.Config`` for EP > 1.

  Installs the expert-parallel sharding configs (expert weights ``Shard(0)``
  on ``ep``) and the EP token dispatcher. Idempotent.

  Args:
    model_config: A ``MuseSparkModel.Config`` (after any converters).
    ep_degree: Expert-parallel degree; must divide the number of experts.
    capacity_factor: Expert-side buffer rows per rank as a multiple of the
      rank's own routes (``tokens * top_k``).
    buffer_multiple: Buffer rows are rounded up to this multiple.
    dispatcher: ``"ragged"`` (``RaggedAllToAllTokenDispatcher``: ragged
      all-to-all, count matrix all-gathered), ``"ragged_a2a"`` (the same with
      the all-to-all count exchange), ``"allgather"`` / ``"allgather_bf16"``
      (``AllGatherEPTokenDispatcher``: all-gather dispatch, reduce-scatter
      combine in FP32 / BF16; suits ``ep_degree <= top_k``).
    placement: ``"contiguous"`` (EP rank g holds experts g*E/ep ...) or
      ``"interleaved"`` (EP rank g holds experts g, g+ep, ...): spreads experts
      with neighbouring ids, which the released checkpoint routes to together
      (measured: the first 64 experts take up to 3.6x the mean load at EP 8).
      The expert weight rows are then stored in physical-slot order: load the
      HF checkpoint with a reader that permutes expert rows
      (``ExpertPermutingHFStorageReader``). ``"file:<path>"`` reads a
      per-layer placement (``load_expert_placement``), e.g. one balanced from
      measured routing by ``balanced_expert_placement``.
  """
  # Imported lazily: the dispatcher modules import the TPU token dispatcher.
  # pylint: disable=g-import-not-at-top
  from torchtitan.experiments.tpu.allgather_ep_dispatcher import AllGatherEPTokenDispatcher
  from torchtitan.experiments.tpu.ragged_ep_dispatcher import RaggedAllToAllTokenDispatcher
  # pylint: enable=g-import-not-at-top

  if ep_degree <= 1:
    return
  if dispatcher not in ("ragged", "ragged_a2a", "allgather", "allgather_bf16"):
    raise ValueError(f"unknown EP dispatcher {dispatcher!r}")
  slot_maps = load_expert_placement(placement, ep_degree) if placement.startswith("file:") else None
  if slot_maps is None and placement not in ("contiguous", "interleaved"):
    raise ValueError(f"unknown expert placement {placement!r}")
  # Recorded for the checkpoint reader (expert rows are stored in slot order).
  model_config.ep_expert_placement = placement
  model_config.ep_placement_degree = ep_degree if placement != "contiguous" else 1
  for layer_id, layer in enumerate(model_config.layers):
    moe = layer.moe
    if moe is None:
      continue
    if moe.num_experts % ep_degree != 0:
      raise ValueError(
          f"layer {layer_id}: {moe.num_experts} experts are not divisible by"
          f" expert_parallel_degree={ep_degree}"
      )
    routed = moe.routed_experts
    old = routed.token_dispatcher
    slot_to_logical = None
    if slot_maps is not None:
      if str(layer_id) not in slot_maps:
        raise ValueError(f"{placement}: no slot map for MoE layer {layer_id}")
      slot_to_logical = slot_maps[str(layer_id)]
      if sorted(slot_to_logical) != list(range(moe.num_experts)):
        raise ValueError(f"{placement}: layer {layer_id} map is not a permutation of {moe.num_experts}")
    if dispatcher.startswith("allgather"):
      routed.token_dispatcher = AllGatherEPTokenDispatcher.Config(
          num_experts=old.num_experts,
          top_k=old.top_k,
          capacity_factor=capacity_factor,
          buffer_multiple=buffer_multiple,
          combine_dtype="bfloat16" if dispatcher == "allgather_bf16" else "float32",
          placement="contiguous" if slot_maps is not None else placement,
          slot_to_logical=slot_to_logical,
      )
    else:
      routed.token_dispatcher = RaggedAllToAllTokenDispatcher.Config(
          num_experts=old.num_experts,
          top_k=old.top_k,
          capacity_factor=capacity_factor,
          buffer_multiple=buffer_multiple,
          exchange="alltoall" if dispatcher == "ragged_a2a" else "allgather",
          placement="contiguous" if slot_maps is not None else placement,
          slot_to_logical=slot_to_logical,
      )
    set_moe_sharding_config(
        moe,
        enable_ep=True,
        enable_sp=False,
        expert_param_layout=EXPERT_PARAM_LAYOUT,
    )


def load_expert_placement(placement: str, ep_degree: int) -> dict[str, list[int]]:
  """Per-MoE-layer ``slot_to_logical`` maps from ``file:<path>`` (JSON).

  Format: ``{"ep": G, "num_experts": E, "slot_to_logical": {"<layer id>": [E ints]}}``;
  slots ``[g * E / G, (g + 1) * E / G)`` live on EP rank ``g``. Written e.g. by a
  load-balancing pass over measured per-expert routing counts.
  """
  # pylint: disable-next=g-import-not-at-top
  import json

  path = placement[len("file:"):]
  with open(path) as f:
    spec = json.load(f)
  if int(spec["ep"]) != ep_degree:
    raise ValueError(f"{path}: placement is for EP {spec['ep']}, not {ep_degree}")
  return {str(k): [int(x) for x in v] for k, v in spec["slot_to_logical"].items()}


def balanced_expert_placement(counts_LE, ep_degree: int) -> dict[str, list[int]]:
  """Per-layer ``slot_to_logical`` maps that balance measured expert loads.

  Greedy bin packing, heaviest expert first, into ``ep_degree`` bins of
  ``E / ep_degree`` experts each; bin ``g`` becomes the slots of EP rank ``g``.

  Args:
    counts_LE: ``(num_layers, num_experts)`` routed-token counts (e.g. the MoE
      layers' ``tokens_per_expert_E`` summed over all ranks after a step).
    ep_degree: expert-parallel degree.

  Returns:
    ``{str(layer): slot_to_logical}`` for ``configure_expert_parallel``'s
    ``placement="file:..."`` (see ``write_expert_placement``).
  """
  # pylint: disable-next=g-import-not-at-top
  import torch

  counts = torch.as_tensor(counts_LE, dtype=torch.float64).cpu()
  num_layers, num_experts = counts.shape
  if num_experts % ep_degree:
    raise ValueError(f"{num_experts} experts not divisible by EP {ep_degree}")
  local = num_experts // ep_degree
  maps = {}
  for layer in range(num_layers):
    bins: list[list[int]] = [[] for _ in range(ep_degree)]
    loads = [0.0] * ep_degree
    for expert in torch.argsort(counts[layer], descending=True, stable=True).tolist():
      g = min((g for g in range(ep_degree) if len(bins[g]) < local), key=lambda g: loads[g])
      bins[g].append(expert)
      loads[g] += float(counts[layer, expert])
    maps[str(layer)] = [e for b in bins for e in sorted(b)]
  return maps


def busiest_rank_load(counts_LE, slot_to_logical: dict[str, list[int]] | None, ep_degree: int):
  """Per layer: load of the busiest EP rank / mean rank load for a placement.

  ``slot_to_logical=None`` is the contiguous placement.
  """
  # pylint: disable-next=g-import-not-at-top
  import torch

  counts = torch.as_tensor(counts_LE, dtype=torch.float64).cpu()
  num_layers, num_experts = counts.shape
  local = num_experts // ep_degree
  out = []
  for layer in range(num_layers):
    order = (
        torch.arange(num_experts)
        if slot_to_logical is None
        else torch.tensor(slot_to_logical[str(layer)])
    )
    per_rank = counts[layer, order].reshape(ep_degree, local).sum(1)
    out.append(float(per_rank.max() / per_rank.mean().clamp(min=1e-30)))
  return out


def write_expert_placement(path: str, slot_to_logical: dict[str, list[int]], ep_degree: int, **meta) -> None:
  """Writes a placement for ``ep_expert_placement="file:<path>"``."""
  # pylint: disable-next=g-import-not-at-top
  import json
  import os

  num_experts = len(next(iter(slot_to_logical.values())))
  spec = {"ep": ep_degree, "num_experts": num_experts, **meta, "slot_to_logical": slot_to_logical}
  tmp = f"{path}.tmp{os.getpid()}"
  with open(tmp, "w") as f:
    json.dump(spec, f)
  os.replace(tmp, path)


def apply_expert_data_parallel(model: nn.Module, parallel_dims, training) -> int:
  """SimpleFSDP for the EP-sharded expert weights over the ``efsdp`` ranks.

  ``apply_simple_fsdp`` does the same for plain ``TransformerBlock`` layers,
  but skips blocks wrapped by activation checkpointing; this walks the modules
  instead, so it works under any AC wrapper. Run it before
  ``apply_simple_fsdp`` (which then leaves these modules alone).

  Returns:
    The number of expert modules wrapped.
  """
  # pylint: disable-next=g-import-not-at-top
  from torch.distributed.fsdp import MixedPrecisionPolicy

  from torchtitan.experiments.graph_trainer.simple_fsdp import data_parallel
  from torchtitan.experiments.tpu.muse_spark.model import MuseSparkRoutedExperts

  mesh_names = (
      ["dp_replicate", "efsdp"] if parallel_dims.dp_replicate_enabled else ["efsdp"]
  )
  edp_mesh = parallel_dims.get_optional_mesh(mesh_names)
  if edp_mesh is None:
    raise ValueError("expert parallelism needs the efsdp mesh")
  mode = "hybrid_shard" if parallel_dims.dp_replicate_enabled else "fully_shard"
  mp_policy = MixedPrecisionPolicy(
      param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
      reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
  )
  count = 0
  for module in model.modules():
    if not isinstance(module, MuseSparkRoutedExperts):
      continue
    experts = module.inner_experts
    if "SimpleFSDP" in type(experts).__name__:
      continue
    shard_dim = 0
    if edp_mesh["efsdp"].size() * parallel_dims.ep > experts.num_experts:
      shard_dim = 1
    module.inner_experts = data_parallel(
        experts, edp_mesh, mode, mp_policy=mp_policy, shard_dim=shard_dim
    )
    count += 1
  return count
