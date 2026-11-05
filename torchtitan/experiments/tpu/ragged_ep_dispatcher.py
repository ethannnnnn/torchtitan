# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Expert-parallel token dispatch for TPU with static shapes.

The standard ``AllToAllTokenDispatcher`` moves routed tokens with a variable
split all-to-all whose split sizes are copied to the host every layer. On TPU
that host round trip serialises the step and the data-dependent shapes defeat
whole-graph compilation. This dispatcher keeps every shape static and every
split size on the device:

1. Sort the ``T * K`` routes by global expert id (the TPU ``local_reorder``
   VJP), so the rows for each destination rank are contiguous and, within a
   rank, ordered by local expert.
2. Exchange the per-(destination, local expert) row counts as int32 device
   tensors. The default ``exchange="allgather"`` all-gathers the whole
   ``(source, destination, local expert)`` count matrix once and derives every
   offset, including the backward plans, locally; ``exchange="alltoall"`` uses
   one small all-to-all for the counts and another for the receive offsets.
3. Move the rows with ``torch.tpu.ops.ragged_all_to_all`` into a receive
   buffer of fixed size ``capacity_factor * T * K`` rows (rounded up to
   ``buffer_multiple``). The buffer is rank-major: one segment per source rank,
   each segment ordered by local expert.
4. Regroup rank-major to expert-major with one row permutation, so the
   grouped matmul sees ``num_local_experts`` contiguous groups. Rows past the
   received total are padding; grouped-matmul kernels that honour
   ``group_sizes`` (MegaBlox / Tokamax) skip them.
5. ``combine`` masks the padding rows, applies the inverse permutation, sends
   rows back with a second ragged all-to-all into exactly ``T * K`` rows, and
   finishes with the fixed-K weighted reduction used by the EP=1 TPU path.

No route is dropped as long as every rank receives at most the buffer size.
Beyond that, with ``exchange="allgather"`` every rank knows the full count
matrix, so the routes that would not fit a destination's buffer are not sent
(their expert output counts as zero) and nothing is written past the buffer.
With ``exchange="alltoall"`` the receive counts are known only after the
exchange, too late to drop routes, so the buffer must cover the busiest rank:
a dispatch that would overflow it fails with a ``RuntimeError`` that names the
remedies. The check is asynchronous (``torch._assert_async``: no host round
trip, so on an accelerator the error may surface later in the step) and is
skipped under ``torch.compile``, like the RoPE position bounds check. The
received total is returned in the metadata (``received_rows``), and
``max_buffer_fill`` reports the largest demanded / buffer ratio (above 1.0:
routes were dropped); raise ``capacity_factor`` or balance the expert
placement.
``ragged_all_to_all`` is differentiable with respect to its operand, and the
row permutations use a gather-only VJP, so the backward pass mirrors the
forward with the same static shapes.

On non-TPU devices a reference exchange based on ``all_to_all_single`` with
host split sizes is used, so the logic can be tested on CPU with gloo.
"""

import math
import weakref
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

from torchtitan.experiments.tpu.token_dispatcher import (
    local_reorder,
    scale_route_weights,
    stable_grouped_reduce,
)
from torchtitan.models.common.token_dispatcher import BaseEPTokenDispatcher

# Live dispatchers, for buffer-fill monitoring (``max_buffer_fill``).
_DISPATCHERS: "weakref.WeakSet[BaseEPTokenDispatcher]" = weakref.WeakSet()


def track_buffer_fill(dispatcher: BaseEPTokenDispatcher) -> None:
    """Report a dispatcher through ``max_buffer_fill`` and ``max_dropped_fraction``.

    The dispatcher keeps its largest fill in ``max_fill`` (and, if it drops
    routes, its largest dropped fraction in ``max_drop``) as device tensors or
    None. It is held weakly.
    """
    _DISPATCHERS.add(dispatcher)


def max_dropped_fraction(reset: bool = True) -> float | None:
    """Largest fraction of a rank's demanded expert rows dropped in one dispatch.

    Tracked by ``AllGatherEPTokenDispatcher``; for this module's dispatcher a
    ``max_buffer_fill`` above 1.0 signals drops. Returns None if nothing was
    tracked since the last reset.
    """
    drops = []
    for dispatcher in list(_DISPATCHERS):
        if getattr(dispatcher, "max_drop", None) is not None:
            drops.append(dispatcher.max_drop)
            if reset:
                dispatcher.max_drop = None
    if not drops:
        return None
    return float(torch.stack(drops).max().item())


def max_buffer_fill(reset: bool = True) -> float | None:
    """Largest demanded-rows / buffer-rows ratio seen by this rank's dispatchers.

    Tracked on the device (no host sync inside the step); reading it syncs.
    A value above 1.0 means routes were dropped because they did not fit a
    receive buffer (``exchange="allgather"`` and ``AllGatherEPTokenDispatcher``;
    ``exchange="alltoall"`` raises instead, see the module docstring): raise
    ``capacity_factor`` or balance the expert placement. Returns None if no
    dispatch ran since the last reset.
    """
    fills = []
    for dispatcher in list(_DISPATCHERS):
        if dispatcher.max_fill is not None:
            fills.append(dispatcher.max_fill)
            if reset:
                dispatcher.max_fill = None
    if not fills:
        return None
    return float(torch.stack(fills).max().item())


def physical_expert_ids(
    ids: torch.Tensor,
    num_experts: int,
    ep_size: int,
    placement: str,
    slot_of=None,
) -> torch.Tensor:
    """Logical expert id -> physical slot (row of the expert weights).

    ``contiguous``: EP rank ``g`` holds experts ``[g * E_local, (g + 1) * E_local)``
    (identity). ``interleaved``: EP rank ``g`` holds logical experts
    ``g, g + G, g + 2G, ...`` in its rows ``[g * E_local, (g + 1) * E_local)``.
    ``slot_of`` (an explicit ``logical -> slot`` table, e.g. a load-balanced
    placement) overrides ``placement``.
    """
    if slot_of is not None:
        return slot_of.to(ids.device).index_select(0, ids.reshape(-1)).reshape(ids.shape)
    if placement == "contiguous" or ep_size <= 1:
        return ids
    if placement != "interleaved":
        raise ValueError(f"unknown expert placement {placement!r}")
    local = num_experts // ep_size
    return (ids % ep_size) * local + ids // ep_size


def logical_expert_of_slot(
    num_experts: int,
    ep_size: int,
    placement: str,
    device=None,
    slot_to_logical: list[int] | None = None,
) -> torch.Tensor:
    """``logical[p]`` = logical expert stored in physical slot ``p`` (int64)."""
    if slot_to_logical is not None:
        return torch.tensor(slot_to_logical, dtype=torch.long, device=device)
    slots = torch.arange(num_experts, device=device)
    if placement == "contiguous" or ep_size <= 1:
        return slots
    local = num_experts // ep_size
    return (slots % local) * ep_size + slots // local


class _SlotTable:
    """``logical -> slot`` lookup, materialized lazily on the device of use.

    Dispatchers can be constructed under ``torch.device("meta")``, so no
    tensor is created at construction time.
    """

    def __init__(self, slot_to_logical: list[int]):
        if sorted(slot_to_logical) != list(range(len(slot_to_logical))):
            raise ValueError("slot_to_logical must be a permutation")
        slot_of = [0] * len(slot_to_logical)
        for slot, logical in enumerate(slot_to_logical):
            slot_of[logical] = slot
        self.slot_of = slot_of
        self._cache: dict = {}

    def to(self, device) -> torch.Tensor:
        key = str(device)
        if key not in self._cache:
            self._cache[key] = torch.tensor(self.slot_of, dtype=torch.long, device=device)
        return self._cache[key]


def slot_of_table(slot_to_logical: list[int] | None) -> "_SlotTable | None":
    """The ``slot_of`` table of ``physical_expert_ids`` for an explicit placement.

    ``slot_to_logical[p]`` is the logical expert in physical slot ``p``; it must
    be a permutation. None (no explicit placement) gives None.
    """
    return None if slot_to_logical is None else _SlotTable(slot_to_logical)


class _PermuteRows(torch.autograd.Function):
    """``x[index]`` for a row permutation; the VJP gathers with the inverse."""

    @staticmethod
    def forward(ctx, x_ND, index_N, inverse_N):
        ctx.save_for_backward(inverse_N)
        return x_ND.index_select(0, index_N)

    @staticmethod
    def backward(ctx, grad_ND):
        (inverse_N,) = ctx.saved_tensors
        return grad_ND.index_select(0, inverse_N), None, None


def permute_rows(
    x_ND: torch.Tensor, index_N: torch.Tensor, inverse_N: torch.Tensor
) -> torch.Tensor:
    """Gather rows by a permutation whose inverse is known (gather-only VJP)."""
    return _PermuteRows.apply(x_ND, index_N, inverse_N)


def _exclusive_cumsum(v: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(v, 0, dtype=torch.int32) - v


def _exchange(t: torch.Tensor, group) -> torch.Tensor:
    """Equal-split all-to-all along dim 0 (functional, so it traces under compile)."""
    return funcol.all_to_all_single(t.contiguous(), None, None, group)


def _assert_rows_fit(rows: torch.Tensor, buffer_rows: int, message: str) -> None:
    """Fail with ``message`` if ``rows`` (a device scalar) exceeds ``buffer_rows``.

    ``torch._assert_async`` checks on the device without a host round trip: on
    CPU it raises a ``RuntimeError`` right away, on an accelerator the failure
    surfaces when the queued work runs. Skipped under ``torch.compile``.
    """
    if torch.compiler.is_compiling():
        return
    torch._assert_async(rows <= buffer_rows, message)


def plan_expert_major(
    counts_GE: torch.Tensor, buffer_rows: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Row permutations between the rank-major receive buffer and expert-major order.

    Args:
        counts_GE: int32 ``(G, E_local)``; rows received from EP rank ``g`` for
            local expert ``e``. Rank-major buffer order is ``(g, e)`` with ``g``
            outermost; expert-major order is ``(e, g)``.
        buffer_rows: receive buffer size (rows), at least ``counts_GE.sum()``.

    Returns:
        to_expert_major: ``x_em = x_rm[to_expert_major]``.
        to_rank_major: ``x_rm = x_em[to_rank_major]`` (the inverse permutation).
        group_sizes_E: int32 ``(E_local,)`` rows per local expert.
        total: int32 scalar tensor, the received row count.
        Padding rows (positions >= total) map to themselves in both orders.
    """
    G, E = counts_GE.shape
    device = counts_GE.device
    flat_rm = counts_GE.reshape(-1)
    end_rm = torch.cumsum(flat_rm, 0, dtype=torch.int32)
    start_rm = end_rm - flat_rm
    flat_em = counts_GE.t().reshape(-1)
    start_em = (
        (torch.cumsum(flat_em, 0, dtype=torch.int32) - flat_em)
        .reshape(E, G)
        .t()
        .reshape(-1)
    )
    pos = torch.arange(buffer_rows, device=device, dtype=torch.int32)
    segment = (pos[:, None] >= end_rm[None, :]).sum(1).clamp(max=G * E - 1)
    total = end_rm[-1]
    dest = torch.where(
        pos < total, start_em[segment] + (pos - start_rm[segment]), pos
    ).long()
    inverse = torch.empty_like(dest).scatter_(0, dest, pos.long())
    return inverse, dest, counts_GE.sum(0).to(torch.int32), total


def _reference_ragged_all_to_all(
    operand, output, input_offsets, send_sizes, output_offsets, recv_sizes, group
):
    """Host-split reference of ``torch.tpu.ops.ragged_all_to_all`` (tests / CPU)."""
    return _RefRaggedAllToAll.apply(
        operand, output, input_offsets, send_sizes, output_offsets, recv_sizes, group
    )


class _RefRaggedAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, operand, output, in_off, send, out_off, recv, group):
        G = dist.get_world_size(group)
        land = torch.empty_like(out_off)
        dist.all_to_all_single(land, out_off.contiguous(), group=group)
        send_l, recv_l = send.tolist(), recv.tolist()
        in_l, land_l = in_off.tolist(), land.tolist()
        packed = torch.cat([operand[in_l[j] : in_l[j] + send_l[j]] for j in range(G)])
        got = operand.new_empty(sum(recv_l), operand.shape[1])
        dist.all_to_all_single(got, packed, recv_l, send_l, group=group)
        result = output.clone()
        for j, chunk in enumerate(torch.split(got, recv_l)):
            result[land_l[j] : land_l[j] + recv_l[j]] = chunk
        ctx.group, ctx.rows = group, operand.shape[0]
        ctx.lists = (in_l, send_l, land_l, recv_l)
        return result

    @staticmethod
    def backward(ctx, grad):
        in_l, send_l, land_l, recv_l = ctx.lists
        G = len(send_l)
        packed = torch.cat([grad[land_l[j] : land_l[j] + recv_l[j]] for j in range(G)])
        got = grad.new_empty(sum(send_l), grad.shape[1])
        dist.all_to_all_single(got, packed, send_l, recv_l, group=ctx.group)
        grad_operand = grad.new_zeros(ctx.rows, grad.shape[1])
        for j, chunk in enumerate(torch.split(got, send_l)):
            grad_operand[in_l[j] : in_l[j] + send_l[j]] = chunk
        return grad_operand, None, None, None, None, None, None


def ragged_all_to_all(
    operand, output, input_offsets, send_sizes, output_offsets, recv_sizes, group
):
    """``torch.tpu.ops.ragged_all_to_all`` on TPU, the host-split reference elsewhere."""
    if operand.device.type == "tpu":
        return torch.tpu.ops.ragged_all_to_all(
            operand,
            output,
            input_offsets,
            send_sizes,
            output_offsets,
            recv_sizes,
            group.group_name,
        )
    return _reference_ragged_all_to_all(
        operand, output, input_offsets, send_sizes, output_offsets, recv_sizes, group
    )


def _raw_ragged_all_to_all(
    operand, output, input_offsets, send_sizes, output_offsets, recv_sizes, group
):
    """One ragged all-to-all, no autograd.

    On TPU the operand is detached, so ``torch.tpu.ops.ragged_all_to_all`` does
    not run its autograd wrapper's two 1-int32-per-peer all-to-alls that
    transpose the offsets for the backward (the backward plan is computed
    locally instead, see ``a2a_plans``). Elsewhere: a host-split reference.
    """
    if operand.device.type == "tpu":
        return torch.tpu.ops.ragged_all_to_all(
            operand.detach(),
            output,
            input_offsets,
            send_sizes,
            output_offsets,
            recv_sizes,
            group.group_name,
        )
    with torch.no_grad():
        return _RefRaggedAllToAll.forward(
            _NullCtx(),
            operand.detach(),
            output,
            input_offsets,
            send_sizes,
            output_offsets,
            recv_sizes,
            group,
        )


class _NullCtx:
    """Attribute sink for calling the reference forward without autograd."""


class _PlannedRaggedAllToAll(torch.autograd.Function):
    """Ragged all-to-all whose backward is another ragged all-to-all.

    ``fwd_plan`` / ``bwd_plan`` are ``(input_offsets, send_sizes,
    output_offsets, recv_sizes)`` device tensors; the backward plan is the
    transpose of the forward one (``a2a_plans``), so no collective is needed to
    derive it.
    """

    @staticmethod
    def forward(ctx, operand, out_rows, fwd_plan, bwd_plan, group):
        ctx.bwd_plan, ctx.group, ctx.rows = bwd_plan, group, operand.shape[0]
        output = operand.new_zeros(out_rows, operand.shape[1])
        return _raw_ragged_all_to_all(operand, output, *fwd_plan, group)

    @staticmethod
    def backward(ctx, grad):
        if grad is None:
            return None, None, None, None, None
        grad_operand = _raw_ragged_all_to_all(
            grad.contiguous(),
            grad.new_zeros(ctx.rows, grad.shape[1]),
            *ctx.bwd_plan,
            ctx.group,
        )
        return grad_operand, None, None, None, None


def planned_ragged_all_to_all(operand, out_rows, fwd_plan, bwd_plan, group):
    return _PlannedRaggedAllToAll.apply(operand, out_rows, fwd_plan, bwd_plan, group)


def a2a_plans(counts_GG: torch.Tensor, rank: int, sent_GG: torch.Tensor | None = None):
    """Dispatch / combine ragged all-to-all plans from the full count matrix.

    Args:
        counts_GG: int32 ``(G, G)``; ``counts_GG[s, d]`` = rows EP rank ``s``
            holds for EP rank ``d`` (identical on every rank). Senders keep
            their rows grouped by destination (expert-sorted).
        rank: this rank's EP index.
        sent_GG: rows actually sent (``<= counts_GG``; each a prefix of the
            sender's rows for that destination); default ``counts_GG``.
            Receive buffers pack the sent rows source-major.

    Returns:
        ``dispatch`` and ``combine`` plans, each ``(input_offsets, send_sizes,
        output_offsets, recv_sizes)``. The backward of dispatch is the combine
        plan and vice versa.
    """
    if sent_GG is None:
        sent_GG = counts_GG
    # sender layout: all of a rank's rows, grouped by destination
    send_offsets = _exclusive_cumsum(counts_GG[rank].contiguous())
    row_prefix = (torch.cumsum(counts_GG, 1, dtype=torch.int32) - counts_GG)[:, rank]
    # receiver layout: the sent rows, packed source-major
    send = sent_GG[rank].contiguous()
    recv = sent_GG[:, rank].contiguous()
    recv_offsets = _exclusive_cumsum(recv)
    col_prefix = (torch.cumsum(sent_GG, 0, dtype=torch.int32) - sent_GG)[rank]
    dispatch = (send_offsets, send, col_prefix.contiguous(), recv)
    combine = (recv_offsets, recv, row_prefix.contiguous(), send)
    return dispatch, combine


def cap_to_buffer(
    counts_SGE: torch.Tensor, buffer_rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows that fit each destination's receive buffer (the rest are dropped).

    Every rank computes the same result from the all-gathered counts. Each
    destination keeps the first ``buffer_rows`` rows of its source-major
    receive layout, so what a source sends is a prefix of its rows for that
    destination (contiguous in its expert-sorted order).

    Args:
        counts_SGE: int32 ``(G_src, G_dst, E_local)`` routed-row counts.
        buffer_rows: receive buffer rows per rank.

    Returns:
        sent_GG: int32 ``(G_src, G_dst)`` rows sent.
        sent_SGE: int32 ``(G_src, G_dst, E_local)`` rows sent per local expert.
    """
    counts_GG = counts_SGE.sum(2, dtype=torch.int32)
    before = torch.cumsum(counts_GG, 0, dtype=torch.int32) - counts_GG
    sent_GG = torch.minimum(torch.clamp(buffer_rows - before, min=0), counts_GG)
    expert_before = torch.cumsum(counts_SGE, 2, dtype=torch.int32) - counts_SGE
    sent_SGE = torch.minimum(
        torch.clamp(sent_GG[:, :, None] - expert_before, min=0), counts_SGE
    )
    return sent_GG, sent_SGE


@dataclass(frozen=True, kw_only=True)
class RaggedEPDispatchMetadata:
    """State carried from ``dispatch`` to ``combine``."""

    token_indices_experts_sorted_N: torch.Tensor  # noqa: N815
    topk_scores_experts_sorted_N: torch.Tensor  # noqa: N815
    send_sizes_G: torch.Tensor  # noqa: N815
    recv_sizes_G: torch.Tensor  # noqa: N815
    recv_offsets_G: torch.Tensor  # noqa: N815
    return_offsets_G: torch.Tensor  # noqa: N815
    to_expert_major: torch.Tensor
    to_rank_major: torch.Tensor
    received_rows: torch.Tensor
    num_routes: int
    dispatch_plan: tuple | None = None
    combine_plan: tuple | None = None


class RaggedAllToAllTokenDispatcher(BaseEPTokenDispatcher):
    """EP dispatcher with device-side split sizes and static shapes.

    ``dispatch`` returns an expert-major buffer of ``buffer_rows`` rows whose
    first ``sum(group_sizes)`` rows are real; the grouped matmul must honour
    ``group_sizes`` (the TPU Pallas kernels do). Requires EP > 1.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseEPTokenDispatcher.Config):
        # Receive buffer rows = capacity_factor * tokens * top_k, rounded up to
        # buffer_multiple. Routes past the buffer of the most loaded rank are
        # dropped with exchange="allgather"; with "alltoall" dispatch raises.
        capacity_factor: float = 1.5
        buffer_multiple: int = 1024
        # "allgather": one all-gather of the (G, G, E_local) count matrix, every
        # offset computed locally, backward plans derived locally (no tiny
        # all-to-alls at all). "alltoall": two small all-to-alls per dispatch
        # plus torch_tpu's offset-transposing all-to-alls in the op's autograd.
        exchange: str = "allgather"
        # Expert-to-rank placement ("contiguous" | "interleaved"); the expert
        # weights must be stored in the matching physical order. An explicit
        # slot_to_logical permutation (one per MoE layer) overrides it.
        placement: str = "contiguous"
        slot_to_logical: list[int] | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.capacity_factor = config.capacity_factor
        self.buffer_multiple = config.buffer_multiple
        if config.exchange not in ("allgather", "alltoall"):
            raise ValueError(f"unknown exchange {config.exchange!r}")
        self.exchange = config.exchange
        self.placement = config.placement
        self.slot_to_logical = config.slot_to_logical
        self._slot_of = slot_of_table(config.slot_to_logical)
        self.max_fill: torch.Tensor | None = None
        track_buffer_fill(self)

    def _group(self):
        if self.ep_mesh is None:
            raise ValueError("RaggedAllToAllTokenDispatcher requires an EP mesh")
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
    ) -> tuple[torch.Tensor, torch.Tensor, RaggedEPDispatchMetadata]:
        group = self._group()
        ep_size = dist.get_world_size(group)
        num_local_experts = self.num_experts // ep_size
        if self.placement != "contiguous" or self._slot_of is not None:
            topk_expert_ids_TK = physical_expert_ids(
                topk_expert_ids_TK, self.num_experts, ep_size, self.placement, self._slot_of
            )
            num_local_tokens_per_expert_E = num_local_tokens_per_expert_E.index_select(
                0,
                logical_expert_of_slot(
                    self.num_experts,
                    ep_size,
                    self.placement,
                    num_local_tokens_per_expert_E.device,
                    self.slot_to_logical,
                ),
            )
        routed_ND, token_indices_N, scores_N = local_reorder(
            x_TD, topk_scores_TK, topk_expert_ids_TK, self.top_k
        )
        num_routes = routed_ND.shape[0]
        buffer_rows = self.buffer_rows(num_routes)
        dispatch_plan = combine_plan = None
        with torch.no_grad():
            counts_GE = (
                num_local_tokens_per_expert_E.to(torch.int32)
                .reshape(ep_size, num_local_experts)
                .contiguous()
            )
            send_G = counts_GE.sum(1).to(torch.int32)
            if self.exchange == "allgather":
                # (src, dst, local expert) counts of the whole EP group.
                counts_SGE = funcol.all_gather_tensor(
                    counts_GE.reshape(1, -1), 0, group
                ).reshape(ep_size, ep_size, num_local_experts)
                rank = dist.get_rank(group)
                # Drop what would not fit a receive buffer (never write past it);
                # dropped routes get a zero expert output.
                sent_GG, sent_SGE = cap_to_buffer(counts_SGE, buffer_rows)
                recv_counts_GE = sent_SGE[:, rank, :].contiguous()
                demand = counts_SGE[:, rank, :].sum()
                dispatch_plan, combine_plan = a2a_plans(
                    counts_SGE.sum(2, dtype=torch.int32), rank, sent_GG
                )
                send_offsets_G, send_G, dest_offsets_G, recv_G = dispatch_plan
                recv_offsets_G, _, return_offsets_G, _ = combine_plan
            else:
                recv_counts_GE = _exchange(counts_GE, group)
                demand = recv_counts_GE.sum()
                # The receive counts arrive with the exchange, too late to
                # drop routes: never plan or write past the buffer.
                _assert_rows_fit(
                    demand,
                    buffer_rows,
                    "RaggedAllToAllTokenDispatcher(exchange='alltoall'): EP rank"
                    f" {dist.get_rank(group)} received more routed rows than its"
                    f" {buffer_rows}-row receive buffer (capacity_factor="
                    f"{self.capacity_factor} x {num_routes} routes). The"
                    " all-to-all count exchange cannot drop routes: raise"
                    " capacity_factor, balance the expert placement, or use"
                    " exchange='allgather', which drops the routes that do not"
                    " fit.",
                )
                recv_G = recv_counts_GE.sum(1).to(torch.int32)
                send_offsets_G = _exclusive_cumsum(send_G)
                recv_offsets_G = _exclusive_cumsum(recv_G)
                # Where this rank's rows land in each destination's buffer, and
                # where rows returned by each destination land here in combine.
                offsets_2G = _exchange(
                    torch.stack([recv_offsets_G, send_offsets_G], dim=1), group
                )
                dest_offsets_G = offsets_2G[:, 0].contiguous()
                return_offsets_G = offsets_2G[:, 1].contiguous()
            to_em, to_rm, group_sizes_E, received = plan_expert_major(
                recv_counts_GE, buffer_rows
            )
            # Demanded rows / buffer: > 1.0 means routes were dropped
            # ("allgather" exchange; "alltoall" raised above).
            fill = demand.float() / buffer_rows
            self.max_fill = (
                fill if self.max_fill is None else torch.maximum(self.max_fill, fill)
            )
        if dispatch_plan is not None:
            received_rm = planned_ragged_all_to_all(
                routed_ND, buffer_rows, dispatch_plan, combine_plan, group
            )
        else:
            received_rm = ragged_all_to_all(
                routed_ND,
                routed_ND.new_zeros(buffer_rows, routed_ND.shape[1]),
                send_offsets_G,
                send_G,
                dest_offsets_G,
                recv_G,
                group,
            )
        routed_input_RD = permute_rows(received_rm, to_em, to_rm)
        metadata = RaggedEPDispatchMetadata(
            token_indices_experts_sorted_N=token_indices_N,
            topk_scores_experts_sorted_N=scores_N,
            send_sizes_G=send_G,
            recv_sizes_G=recv_G,
            recv_offsets_G=recv_offsets_G,
            return_offsets_G=return_offsets_G,
            to_expert_major=to_em,
            to_rank_major=to_rm,
            received_rows=received,
            num_routes=num_routes,
            dispatch_plan=dispatch_plan,
            combine_plan=combine_plan,
        )
        return routed_input_RD, group_sizes_E, metadata

    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: RaggedEPDispatchMetadata,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        return self.weighted_reduce(
            self.return_rows(routed_output_RD, metadata), metadata, x_TD
        )

    def return_rows(
        self,
        routed_output_RD: torch.Tensor,
        metadata: RaggedEPDispatchMetadata,
    ) -> torch.Tensor:
        """Send expert outputs back to the ranks that own their tokens.

        Returns this rank's ``T * K`` route rows in the expert-sorted order of
        ``metadata.token_indices_experts_sorted_N``, before the route weights
        are applied. A row-wise transform (e.g. a norm shared by all experts)
        can run on them before ``weighted_reduce``: its input is then exactly
        this rank's routes, as with EP=1, and it never sees padding rows.
        """
        group = self._group()
        rows = routed_output_RD.shape[0]
        valid_R1 = (
            torch.arange(rows, device=routed_output_RD.device)
            < metadata.received_rows
        )[:, None]
        # Padding rows are not computed by group-size-aware kernels; zero them so
        # nothing unspecified reaches the backward of later row-wise ops.
        routed_output_RD = torch.where(
            valid_R1, routed_output_RD, torch.zeros_like(routed_output_RD)
        )
        out_rm = permute_rows(
            routed_output_RD, metadata.to_rank_major, metadata.to_expert_major
        )
        if metadata.combine_plan is not None:
            return planned_ragged_all_to_all(
                out_rm,
                metadata.num_routes,
                metadata.combine_plan,
                metadata.dispatch_plan,
                group,
            )
        return ragged_all_to_all(
            out_rm,
            out_rm.new_zeros(metadata.num_routes, out_rm.shape[1]),
            metadata.recv_offsets_G,
            metadata.recv_sizes_G,
            metadata.return_offsets_G,
            metadata.send_sizes_G,
            group,
        )

    def weighted_reduce(
        self,
        returned_ND: torch.Tensor,
        metadata: RaggedEPDispatchMetadata,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        """Weight each route by its score and sum a token's ``K`` routes in FP32."""
        weighted_ND = scale_route_weights(
            returned_ND, metadata.topk_scores_experts_sorted_N
        )
        return stable_grouped_reduce(
            weighted_ND, metadata.token_indices_experts_sorted_N, self.top_k
        ).to(x_TD.dtype)


__all__ = [
    "max_dropped_fraction",
    "logical_expert_of_slot",
    "physical_expert_ids",
    "a2a_plans",
    "cap_to_buffer",
    "max_buffer_fill",
    "planned_ragged_all_to_all",
    "RaggedAllToAllTokenDispatcher",
    "RaggedEPDispatchMetadata",
    "permute_rows",
    "plan_expert_major",
    "ragged_all_to_all",
    "slot_of_table",
    "track_buffer_fill",
]
