# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Expert-parallel token dispatch with all-gather and reduce-scatter (no all-to-all).

Every rank of an EP group of ``G`` ranks all-gathers the group's tokens (and
their routing), runs its ``E / G`` local experts on the routes that chose them,
weights and sums the outputs per token into a ``G * T`` row buffer, and a
reduce-scatter returns each rank the sum over the group's experts for its own
``T`` tokens. Only all-gather / reduce-scatter collectives are used, the same
ones SimpleFSDP runs, with static shapes and no host round trip.

Traffic per rank and direction is ``(G - 1) * T * D`` elements, versus about
``K * T * D * (G - 1) / G`` for an all-to-all dispatch, so the two match at
``G = K + 1`` and this path suits ``G <= K`` (e.g. EP 8 with top-8 routing).
The combine buffer and its reduce-scatter use ``combine_dtype`` (FP32 by
default: the per-token route sum then accumulates in FP32, as in the EP = 1
TPU path).

Local routes are gathered into a fixed buffer of ``capacity_factor * T * K``
rows (rounded up to ``buffer_multiple``), grouped by local expert. If more
routes choose this rank's experts, the routes past the buffer are dropped
(their outputs count as zero) and ``max_fill`` exceeds 1.0; nothing is written
out of bounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

from torchtitan.experiments.tpu import ragged_ep_dispatcher as _red
from torchtitan.experiments.tpu.token_dispatcher import scale_route_weights
from torchtitan.models.common.token_dispatcher import BaseEPTokenDispatcher

__all__ = ["AllGatherEPDispatchMetadata", "AllGatherEPTokenDispatcher"]

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


class _AllGatherRows(torch.autograd.Function):
    """All-gather along dim 0; the VJP reduce-scatters (sums) the gradient."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return funcol.all_gather_tensor(x.contiguous(), 0, group).wait()

    @staticmethod
    def backward(ctx, grad):
        return funcol.reduce_scatter_tensor(grad.contiguous(), "sum", 0, ctx.group).wait(), None


class _ReduceScatterRows(torch.autograd.Function):
    """Reduce-scatter (sum) along dim 0; the VJP all-gathers the gradient."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return funcol.reduce_scatter_tensor(x.contiguous(), "sum", 0, group).wait()

    @staticmethod
    def backward(ctx, grad):
        return funcol.all_gather_tensor(grad.contiguous(), 0, ctx.group).wait(), None


@dataclass(frozen=True, kw_only=True)
class AllGatherEPDispatchMetadata:
    """State carried from ``dispatch`` to ``combine``."""

    token_indices_R: torch.Tensor  # noqa: N815  row of the gathered token buffer
    valid_R: torch.Tensor  # noqa: N815  real (not padding / dropped) rows
    scores_R: torch.Tensor  # noqa: N815  route weights of the rows
    num_gathered_tokens: int


class AllGatherEPTokenDispatcher(BaseEPTokenDispatcher):
    """EP dispatch by all-gather of the group's tokens, combine by reduce-scatter.

    ``dispatch`` returns ``buffer_rows`` rows grouped by local expert; rows past
    ``sum(group_sizes)`` are padding (zeros) and the grouped matmul must honour
    ``group_sizes``. ``return_rows`` masks the padding rows of the expert
    output; a row-wise transform may run on them before ``weighted_reduce``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseEPTokenDispatcher.Config):
        capacity_factor: float = 1.5
        buffer_multiple: int = 1024
        combine_dtype: str = "float32"
        # Expert-to-rank placement ("contiguous" | "interleaved"), see
        # ragged_ep_dispatcher.physical_expert_ids; an explicit per-layer
        # slot_to_logical permutation overrides it.
        placement: str = "contiguous"
        slot_to_logical: list[int] | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        if config.combine_dtype not in _DTYPES:
            raise ValueError(f"unknown combine_dtype {config.combine_dtype!r}")
        self.capacity_factor = config.capacity_factor
        self.buffer_multiple = config.buffer_multiple
        self.combine_dtype = _DTYPES[config.combine_dtype]
        self.placement = config.placement
        self._slot_of = _red.slot_of_table(config.slot_to_logical)
        self.max_fill: torch.Tensor | None = None
        self.max_drop: torch.Tensor | None = None
        _red.track_buffer_fill(self)  # shared fill monitor (max_buffer_fill)

    def _group(self):
        if self.ep_mesh is None:
            raise ValueError("AllGatherEPTokenDispatcher requires an EP mesh")
        return self.ep_mesh.get_group()

    def buffer_rows(self, num_routes: int) -> int:
        rows = math.ceil(self.capacity_factor * num_routes)
        return math.ceil(rows / self.buffer_multiple) * self.buffer_multiple

    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, AllGatherEPDispatchMetadata]:
        del num_local_tokens_per_expert_E  # counted on the gathered routes
        group = self._group()
        ep_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        num_local_experts = self.num_experts // ep_size
        num_tokens, top_k = topk_expert_ids_TK.shape
        buffer_rows = self.buffer_rows(num_tokens * top_k)

        x_all = _AllGatherRows.apply(x_TD, group)  # [G*T, D]
        scores_all = _AllGatherRows.apply(topk_scores_TK, group)  # [G*T, K]
        with torch.no_grad():
            # int32 on the wire (TPU has no native int64).
            ids_all = funcol.all_gather_tensor(
                topk_expert_ids_TK.to(torch.int32).contiguous(), 0, group
            ).wait()
            phys = _red.physical_expert_ids(
                ids_all.reshape(-1).long(), self.num_experts, ep_size, self.placement,
                self._slot_of,
            )
            local_e = phys - rank * num_local_experts
            is_local = (local_e >= 0) & (local_e < num_local_experts)
            # Local routes first, grouped by local expert (stable: token order).
            key = torch.where(is_local, local_e, num_local_experts)
            order = torch.argsort(key, stable=True)
            if buffer_rows > order.shape[0]:
                # More buffer rows than gathered routes (small batches, or a
                # large capacity_factor / buffer_multiple): the extra rows are
                # padding, masked below like any row past the local routes.
                order = torch.cat([order, order.new_zeros(buffer_rows - order.shape[0])])
            routes_R = order[:buffer_rows]
            counts = torch.zeros(
                num_local_experts + 1, dtype=torch.int32, device=key.device
            ).scatter_add_(0, key, torch.ones_like(key, dtype=torch.int32))
            counts = counts[:num_local_experts]
            ends = torch.cumsum(counts, 0, dtype=torch.int32).clamp(max=buffer_rows)
            group_sizes_E = ends - torch.cat([ends.new_zeros(1), ends[:-1]])
            total = counts.sum()
            kept = ends[-1]
            valid_R = torch.arange(buffer_rows, device=key.device) < kept
            token_indices_R = routes_R // top_k
            fill = total.float() / buffer_rows
            self.max_fill = (
                fill if self.max_fill is None else torch.maximum(self.max_fill, fill)
            )
            drop = (total - kept).float() / total.clamp(min=1).float()
            self.max_drop = (
                drop if self.max_drop is None else torch.maximum(self.max_drop, drop)
            )
        # Padding rows are zeroed; torch.where also zeroes their gradient, so
        # unspecified values that a group-size-aware kernel leaves in padding
        # rows never reach the gathered tokens' gradient.
        routed_RD = torch.where(
            valid_R[:, None],
            x_all.index_select(0, token_indices_R),
            x_all.new_zeros(()),
        )
        scores_R = torch.where(
            valid_R,
            scores_all.reshape(-1).index_select(0, routes_R),
            scores_all.new_zeros(()),
        )
        metadata = AllGatherEPDispatchMetadata(
            token_indices_R=token_indices_R,
            valid_R=valid_R,
            scores_R=scores_R,
            num_gathered_tokens=x_all.shape[0],
        )
        return routed_RD, group_sizes_E, metadata

    def return_rows(
        self, routed_output_RD: torch.Tensor, metadata: AllGatherEPDispatchMetadata
    ) -> torch.Tensor:
        """The expert output rows with padding rows set to zero (stays on this rank)."""
        return torch.where(
            metadata.valid_R[:, None], routed_output_RD, routed_output_RD.new_zeros(())
        )

    def weighted_reduce(
        self,
        rows_RD: torch.Tensor,
        metadata: AllGatherEPDispatchMetadata,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        """Weight the rows, sum per gathered token, reduce-scatter to the owners."""
        # The TPU route-weighting VJP: BF16 weighted rows, saves the BF16 rows
        # (not an FP32 copy) and fuses the FP32 dScore reduction.
        weighted_RD = scale_route_weights(rows_RD, metadata.scores_R.float())
        summed = torch.zeros(
            metadata.num_gathered_tokens,
            rows_RD.shape[1],
            dtype=torch.float32,
            device=rows_RD.device,
        ).index_add(0, metadata.token_indices_R, weighted_RD.float())
        out_TD = _ReduceScatterRows.apply(summed.to(self.combine_dtype), self._group())
        return out_TD.to(x_TD.dtype)

    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: AllGatherEPDispatchMetadata,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        return self.weighted_reduce(
            self.return_rows(routed_output_RD, metadata), metadata, x_TD
        )
