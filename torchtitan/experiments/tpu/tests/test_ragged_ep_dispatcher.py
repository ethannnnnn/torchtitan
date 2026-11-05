# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU (gloo) tests of the EP token dispatchers against a dense reference.

Each rank routes its own tokens to top-k of all experts; experts are sharded over
the EP group. The EP path (dispatch -> grouped SwiGLU experts -> combine) must match
a single-process computation that sees every rank's tokens and every expert, for the
output and for the gradients of the tokens and of the local expert weights.

Cases cover both count exchanges of ``RaggedAllToAllTokenDispatcher``,
``AllGatherEPTokenDispatcher``, and the contiguous, interleaved and explicit
(measured) expert placements. A receive buffer smaller than the routes must drop
routes, report it, and never write past the buffer; the all-to-all count
exchange, which cannot drop routes, must raise a clear error instead. The
planning helpers are checked on their own.

The dispatchers import the TPU token-dispatcher module, which imports TorchTPU.
Without TorchTPU, put the CPU stand-in first on the path:

  PYTHONPATH=torchtitan/experiments/tpu/tests/cpu_stubs:. \\
      python -m pytest torchtitan/experiments/tpu/tests/test_ragged_ep_dispatcher.py
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from torchtitan.experiments.tpu.ragged_ep_dispatcher import (
    _assert_rows_fit,
    _slot_of_table,
    a2a_plans,
    cap_to_buffer,
    logical_expert_of_slot,
    physical_expert_ids,
    plan_expert_major,
)

_TOP_K, _TOKENS, _DIM, _HIDDEN = 4, 64, 32, 48


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _experts(x, w13, w2, group_sizes):
    """Grouped SwiGLU over contiguous expert groups; rows past the groups get zeros."""
    out = x.new_zeros(x.shape[0], w2.shape[2])
    start = 0
    hidden = w2.shape[1]
    for e, n in enumerate(group_sizes.tolist()):
        if n:
            gate_up = x[start : start + n] @ w13[e]
            h = F.silu(gate_up[:, :hidden]) * gate_up[:, hidden:]
            out[start : start + n] = h @ w2[e]
        start += n
    return out


def _slot_to_logical(placement, num_experts, world):
    """Logical expert held by each physical slot, written out independently."""
    local = num_experts // world
    if placement == "contiguous":
        return list(range(num_experts))
    if placement == "interleaved":
        # EP rank g holds logical experts g, g + G, g + 2G, ... in its slots.
        return [s // local + (s % local) * world for s in range(num_experts)]
    # "measured": an explicit permutation, as read from a placement file.
    gen = torch.Generator().manual_seed(5)
    return torch.randperm(num_experts, generator=gen).tolist()


def _dispatcher(case, num_experts, slot_to_logical):
    from torchtitan.experiments.tpu.allgather_ep_dispatcher import (
        AllGatherEPTokenDispatcher,
    )
    from torchtitan.experiments.tpu.ragged_ep_dispatcher import (
        RaggedAllToAllTokenDispatcher,
    )

    measured = case["placement"] == "measured"
    common = dict(
        num_experts=num_experts,
        top_k=_TOP_K,
        capacity_factor=case["capacity"],
        buffer_multiple=case["buffer_multiple"],
        placement="contiguous" if measured else case["placement"],
        slot_to_logical=slot_to_logical if measured else None,
    )
    if case["dispatcher"] == "allgather":
        return AllGatherEPTokenDispatcher(AllGatherEPTokenDispatcher.Config(**common))
    return RaggedAllToAllTokenDispatcher(
        RaggedAllToAllTokenDispatcher.Config(exchange=case["exchange"], **common)
    )


def _run_case(case, rank, world, ep_mesh, num_experts):
    from torchtitan.experiments.tpu import ragged_ep_dispatcher as dispatchers

    slot_to_logical = _slot_to_logical(case["placement"], num_experts, world)
    disp = _dispatcher(case, num_experts, slot_to_logical)
    disp.wire_meshes(ep_mesh=ep_mesh)
    dispatchers.max_buffer_fill()  # reset the fill and drop monitors
    dispatchers.max_dropped_fraction()
    local = num_experts // world

    g = torch.Generator().manual_seed(100 + rank)
    x = torch.randn(_TOKENS, _DIM, generator=g).requires_grad_(True)
    logits = torch.randn(_TOKENS, num_experts, generator=g)
    ids = torch.topk(logits, _TOP_K, dim=-1).indices
    scores = torch.softmax(torch.gather(logits, 1, ids), -1)
    counts = torch.bincount(ids.reshape(-1), minlength=num_experts)
    gw = torch.Generator().manual_seed(1000 + rank)
    # This rank's experts, in physical-slot order.
    w13 = (torch.randn(local, _DIM, 2 * _HIDDEN, generator=gw) * 0.1).requires_grad_(True)
    w2 = (torch.randn(local, _HIDDEN, _DIM, generator=gw) * 0.1).requires_grad_(True)

    if case["expect_error"]:
        # Every rank overflows, so every rank stops before the next collective.
        try:
            disp.dispatch(x, scores, ids, counts)
        except RuntimeError as error:
            return {"error": str(error)}
        return {"error": None}
    routed, group_sizes, meta = disp.dispatch(x, scores, ids, counts)
    out = disp.combine(_experts(routed, w13, w2, group_sizes), meta, x)
    grad_out = torch.randn(out.shape, generator=g)
    out.backward(grad_out)
    fill = dispatchers.max_buffer_fill()
    dropped = dispatchers.max_dropped_fraction()

    def gather(t):
        parts = [torch.empty_like(t) for _ in range(world)]
        dist.all_gather(parts, t.contiguous())
        return parts

    xs, idss, scs, gos = gather(x.detach()), gather(ids), gather(scores), gather(grad_out)
    W13 = torch.cat(gather(w13.detach())).requires_grad_(True)
    W2 = torch.cat(gather(w2.detach())).requires_grad_(True)
    slot_of = torch.empty(num_experts, dtype=torch.long)
    slot_of[torch.tensor(slot_to_logical)] = torch.arange(num_experts)
    xr = xs[rank].clone().requires_grad_(True)
    ref = torch.zeros_like(xr)
    for k in range(_TOP_K):
        e = slot_of[idss[rank][:, k]]
        gate_up = torch.bmm(xr[:, None, :], W13[e]).squeeze(1)
        h = F.silu(gate_up[:, :_HIDDEN]) * gate_up[:, _HIDDEN:]
        ref = ref + torch.bmm(h[:, None, :], W2[e]).squeeze(1) * scs[rank][:, k : k + 1]
    (ref * gos[rank]).sum().backward()
    g13, g2 = W13.grad.clone(), W2.grad.clone()
    dist.all_reduce(g13)
    dist.all_reduce(g2)
    sl = slice(rank * local, (rank + 1) * local)

    def rel(a, b):
        return float((a - b).norm() / (b.norm() + 1e-30))

    return {
        "out": rel(out.detach(), ref.detach()),
        "dx": rel(x.grad, xr.grad),
        "dw13": rel(w13.grad, g13[sl]),
        "dw2": rel(w2.grad, g2[sl]),
        "received": int(group_sizes.sum()),
        "buffer": int(routed.shape[0]),
        "fill": fill,
        "dropped": dropped,
        "finite": all(
            bool(torch.isfinite(t).all()) for t in (out, x.grad, w13.grad, w2.grad)
        ),
    }


def _worker(rank, world, port, num_experts, cases, results):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("ep",))
        for name, case in cases.items():
            results[f"{name}/{rank}"] = _run_case(
                case, rank, world, mesh["ep"], num_experts
            )
    finally:
        dist.destroy_process_group()


def _case(
    dispatcher,
    placement,
    exchange="allgather",
    capacity=2.0,
    buffer_multiple=64,
    expect_error=False,
):
    return dict(
        dispatcher=dispatcher,
        placement=placement,
        exchange=exchange,
        capacity=capacity,
        buffer_multiple=buffer_multiple,
        expect_error=expect_error,
    )


_CASES = {
    2: {
        "ragged": _case("ragged", "contiguous"),
        "allgather-measured": _case("allgather", "measured"),
        # 1024 buffer rows for the 512 routes of the whole EP group.
        "ragged-big-buffer": _case("ragged", "contiguous", buffer_multiple=1024),
        "allgather-big-buffer": _case("allgather", "contiguous", buffer_multiple=1024),
    },
    4: {
        "ragged": _case("ragged", "contiguous"),
        "ragged-a2a-interleaved": _case("ragged", "interleaved", exchange="alltoall"),
        "ragged-measured": _case("ragged", "measured"),
        "allgather": _case("allgather", "contiguous"),
        "allgather-interleaved": _case("allgather", "interleaved"),
        # A 32-row buffer for about 256 routes per rank: most routes are dropped.
        "ragged-overflow": _case("ragged", "contiguous", capacity=0.1, buffer_multiple=8),
        "allgather-overflow": _case(
            "allgather", "contiguous", capacity=0.1, buffer_multiple=8
        ),
        # The same 32-row buffer with the all-to-all count exchange: no drops,
        # dispatch raises on every rank.
        "ragged-a2a-overflow": _case(
            "ragged",
            "contiguous",
            exchange="alltoall",
            capacity=0.1,
            buffer_multiple=8,
            expect_error=True,
        ),
    },
}


@pytest.mark.parametrize("world,num_experts", [(2, 8), (4, 16)])
def test_ep_dispatchers_match_dense(world, num_experts):
    manager = mp.Manager()
    results = manager.dict()
    cases = _CASES[world]
    mp.spawn(
        _worker,
        args=(world, _free_port(), num_experts, cases, results),
        nprocs=world,
        join=True,
    )
    assert len(results) == world * len(cases)
    for key, r in results.items():
        name = key.split("/")[0]
        if cases[name]["expect_error"]:
            assert r["error"] is not None, (key, r)
            assert "exchange='alltoall'" in r["error"], (key, r)
            assert "32-row receive buffer" in r["error"], (key, r)
            continue
        assert r["finite"], (key, r)
        assert r["received"] <= r["buffer"], (key, r)
        if name.endswith("-overflow"):
            # Dropped routes count as zero, so the output differs from the
            # dense one; the fill monitor reports more demand than buffer.
            assert r["fill"] > 1.0, (key, r)
            assert r["out"] > 1e-3, (key, r)
            if cases[name]["dispatcher"] == "allgather":
                assert r["dropped"] > 0.0, (key, r)
            continue
        assert r["fill"] <= 1.0, (key, r)
        for metric in ("out", "dx", "dw13", "dw2"):
            assert r[metric] < 1e-5, (key, metric, r)


def test_expert_placements_map_logical_ids_to_slots():
    num_experts, ep = 16, 4
    ids = torch.arange(num_experts)
    for placement in ("contiguous", "interleaved"):
        slot = physical_expert_ids(ids, num_experts, ep, placement)
        logical = logical_expert_of_slot(num_experts, ep, placement)
        assert torch.equal(logical[slot], ids), placement
        assert logical.tolist() == _slot_to_logical(placement, num_experts, ep)
    permutation = _slot_to_logical("measured", num_experts, ep)
    slot = physical_expert_ids(
        ids, num_experts, ep, "contiguous", _slot_of_table(permutation)
    )
    assert torch.equal(torch.tensor(permutation)[slot], ids)
    with pytest.raises(ValueError, match="permutation"):
        _slot_of_table([0, 0, 1, 2])
    with pytest.raises(ValueError, match="unknown expert placement"):
        physical_expert_ids(ids, num_experts, ep, "striped")


def test_cap_to_buffer_keeps_a_source_major_prefix_per_destination():
    gen = torch.Generator().manual_seed(3)
    counts_SGE = torch.randint(0, 9, (4, 4, 3), generator=gen, dtype=torch.int32)
    for buffer_rows in (0, 5, 17, 40, 1000):
        sent_GG, sent_SGE = cap_to_buffer(counts_SGE, buffer_rows)
        assert torch.equal(sent_SGE.sum(2, dtype=torch.int32), sent_GG)
        for dst in range(4):
            demand = counts_SGE[:, dst].reshape(-1)  # (source, local expert) order
            before = torch.cumsum(demand, 0) - demand
            kept = torch.minimum(torch.clamp(buffer_rows - before, min=0), demand)
            assert torch.equal(sent_SGE[:, dst].reshape(-1), kept), (buffer_rows, dst)
            assert int(sent_GG[:, dst].sum()) == min(int(demand.sum()), buffer_rows)


def test_assert_rows_fit_raises_only_past_the_buffer():
    _assert_rows_fit(torch.tensor(32, dtype=torch.int32), 32, "unused")
    with pytest.raises(RuntimeError, match="does not fit"):
        _assert_rows_fit(torch.tensor(33, dtype=torch.int32), 32, "does not fit")


def test_plan_expert_major_regroups_rows_by_local_expert():
    counts_GE = torch.tensor([[2, 0, 1], [1, 3, 0]], dtype=torch.int32)
    to_em, to_rm, group_sizes, total = plan_expert_major(counts_GE, 10)
    assert int(total) == 7
    assert group_sizes.tolist() == [3, 3, 1]
    assert torch.equal(to_rm[to_em], torch.arange(10))
    rank_major = [(0, 0), (0, 0), (0, 2), (1, 0), (1, 1), (1, 1), (1, 1)] + [None] * 3
    expert_major = [rank_major[i] for i in to_em.tolist()]
    # Grouped by local expert, source-major within an expert; padding rows stay.
    assert expert_major == [
        (0, 0), (0, 0), (1, 0), (1, 1), (1, 1), (1, 1), (0, 2), None, None, None
    ]


def test_a2a_plans_agree_between_senders_and_receivers():
    counts_GG = torch.tensor([[3, 1, 0], [2, 2, 5], [0, 4, 1]], dtype=torch.int32)
    sent_GG = torch.minimum(counts_GG, torch.tensor(3, dtype=torch.int32))
    plans = [a2a_plans(counts_GG, rank, sent_GG) for rank in range(3)]
    for src in range(3):
        (send_off, send, dest_off, _), _ = plans[src]
        for dst in range(3):
            (_, _, _, recv), (recv_off, return_send, return_off, _) = plans[dst]
            assert int(send[dst]) == int(recv[src]) == int(sent_GG[src, dst])
            # src's rows land in dst's buffer where dst expects rows from src,
            assert int(dest_off[dst]) == int(recv_off[src])
            # and dst sends them back to where src took them from.
            assert int(return_send[src]) == int(sent_GG[src, dst])
            assert int(return_off[src]) == int(send_off[dst])
    # Senders lay out all their rows; only a prefix per destination is sent.
    assert plans[1][0][0].tolist() == [0, 2, 4]
