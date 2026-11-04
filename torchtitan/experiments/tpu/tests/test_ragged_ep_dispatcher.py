# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU (gloo) test of RaggedAllToAllTokenDispatcher against a dense reference.

Each rank routes its own tokens to top-k of all experts; experts are sharded over
the EP group. The EP path (dispatch -> grouped SwiGLU experts -> combine) must match
a single-process computation that sees every rank's tokens and every expert, for the
output and for the gradients of the tokens and of the local expert weights.

The dispatcher imports the TPU token-dispatcher module, which imports TorchTPU.
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


def _worker(rank, world, port, num_experts, top_k, tokens, dim, hidden, capacity, results):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from torch.distributed.device_mesh import init_device_mesh

        from torchtitan.experiments.tpu.ragged_ep_dispatcher import (
            RaggedAllToAllTokenDispatcher,
        )

        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("ep",))
        disp = RaggedAllToAllTokenDispatcher(
            RaggedAllToAllTokenDispatcher.Config(
                num_experts=num_experts,
                top_k=top_k,
                capacity_factor=capacity,
                buffer_multiple=64,
            )
        )
        disp.wire_meshes(ep_mesh=mesh["ep"])
        local = num_experts // world

        g = torch.Generator().manual_seed(100 + rank)
        x = torch.randn(tokens, dim, generator=g).requires_grad_(True)
        logits = torch.randn(tokens, num_experts, generator=g)
        ids = torch.topk(logits, top_k, dim=-1).indices
        scores = torch.softmax(torch.gather(logits, 1, ids), -1)
        counts = torch.bincount(ids.reshape(-1), minlength=num_experts)
        gw = torch.Generator().manual_seed(1000 + rank)
        w13 = (torch.randn(local, dim, 2 * hidden, generator=gw) * 0.1).requires_grad_(True)
        w2 = (torch.randn(local, hidden, dim, generator=gw) * 0.1).requires_grad_(True)

        routed, group_sizes, meta = disp.dispatch(x, scores, ids, counts)
        out = disp.combine(_experts(routed, w13, w2, group_sizes), meta, x)
        grad_out = torch.randn(out.shape, generator=g)
        out.backward(grad_out)

        def gather(t):
            parts = [torch.empty_like(t) for _ in range(world)]
            dist.all_gather(parts, t.contiguous())
            return parts

        xs, idss, scs, gos = gather(x.detach()), gather(ids), gather(scores), gather(grad_out)
        W13 = torch.cat(gather(w13.detach())).requires_grad_(True)
        W2 = torch.cat(gather(w2.detach())).requires_grad_(True)
        xr = xs[rank].clone().requires_grad_(True)
        ref = torch.zeros_like(xr)
        for k in range(top_k):
            e = idss[rank][:, k]
            gate_up = torch.bmm(xr[:, None, :], W13[e]).squeeze(1)
            h = F.silu(gate_up[:, :hidden]) * gate_up[:, hidden:]
            ref = ref + torch.bmm(h[:, None, :], W2[e]).squeeze(1) * scs[rank][:, k : k + 1]
        (ref * gos[rank]).sum().backward()
        g13, g2 = W13.grad.clone(), W2.grad.clone()
        dist.all_reduce(g13)
        dist.all_reduce(g2)
        sl = slice(rank * local, (rank + 1) * local)

        def rel(a, b):
            return float((a - b).norm() / (b.norm() + 1e-30))

        results[rank] = {
            "out": rel(out.detach(), ref.detach()),
            "dx": rel(x.grad, xr.grad),
            "dw13": rel(w13.grad, g13[sl]),
            "dw2": rel(w2.grad, g2[sl]),
            "received": int(meta.received_rows),
            "buffer": int(routed.shape[0]),
        }
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world,num_experts", [(2, 8), (4, 16)])
def test_ragged_ep_dispatcher_matches_dense(world, num_experts):
    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(
        _worker,
        args=(world, _free_port(), num_experts, 4, 64, 32, 48, 2.0, results),
        nprocs=world,
        join=True,
    )
    assert len(results) == world
    for rank, r in results.items():
        assert r["received"] <= r["buffer"], (rank, r)
        for key in ("out", "dx", "dw13", "dw2"):
            assert r[key] < 1e-5, (rank, key, r)
