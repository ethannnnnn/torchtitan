# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""2-rank CPU check: ``LowPrecisionAdamW`` state through a distributed checkpoint.

Sharded (DTensor) parameters: a float8-moment weight with uneven row shards, a
weight whose last dimension is not a multiple of the block size and one whose
last dimension is sharded (both keep bf16 moments), all with Kahan residuals.
The optimizer state goes through TorchTitan's flat optimizer state dict and
``torch.distributed.checkpoint`` save/load into a freshly built optimizer, then
both optimizers take one more identical step. Checks: every state value of a
sharded parameter is a DTensor; materializing state before the load leaves the
step counter at 0; restored state equals the saved state; the next step gives
bit-identical parameters.

Run: torchrun --nproc_per_node=2 -m torchtitan.experiments.tpu.tests.low_precision_adamw_dcp_check
"""

from __future__ import annotations

import os
import shutil
import tempfile

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from torchtitan.experiments.tpu.low_precision_adamw import low_precision_adamw_config

SHAPES = {"w_f8": ((513, 512), Shard(0)), "w_odd": ((256, 300), Shard(0)), "w_col": ((512, 1024), Shard(1))}


def build(mesh, state_dtype):
  torch.manual_seed(0)
  model = torch.nn.Module()
  for name, (shape, placement) in SHAPES.items():
    full = (torch.randn(shape) * 5.5e-3).to(torch.bfloat16)
    setattr(model, name, torch.nn.Parameter(distribute_tensor(full, mesh, [placement])))
  opt = low_precision_adamw_config(lr=1e-3, param_update="kahan", state_dtype=state_dtype).build(model_parts=[model])
  return model, opt


def grads(mesh, model, step):
  g = torch.Generator().manual_seed(100 + step)
  for name, (shape, placement) in SHAPES.items():
    getattr(model, name).grad = distribute_tensor(torch.randn(shape, generator=g).to(torch.bfloat16), mesh, [placement])


def local(v):
  # Compare local shards (gloo cannot all-gather float8) and reduce the verdict.
  return v.to_local() if isinstance(v, DTensor) else v


def all_ranks(flag: bool) -> bool:
  t = torch.tensor([1 if flag else 0])
  dist.all_reduce(t, op=dist.ReduceOp.MIN)
  return bool(t.item())


def main():
  dist.init_process_group("gloo")
  rank = dist.get_rank()
  mesh = init_device_mesh("cpu", (dist.get_world_size(),))
  ok = True
  for state_dtype in ("float8", "bfloat16"):
    model_a, opt_a = build(mesh, state_dtype)
    for step in range(3):
      grads(mesh, model_a, step)
      torch.manual_seed(1000 + step)
      opt_a.step()
      opt_a.zero_grad()
    sd_a = opt_a.state_dict()
    sharded_ok = all(isinstance(v, DTensor) for k, v in sd_a.items()
                     if k.startswith("state.") and not k.endswith(".step"))
    path = [tempfile.mkdtemp(prefix="lpadamw_dcp_") if rank == 0 else None]
    dist.broadcast_object_list(path, src=0)
    dcp.save(sd_a, checkpoint_id=path[0])

    model_b, opt_b = build(mesh, state_dtype)
    with torch.no_grad():
      for name in SHAPES:
        getattr(model_b, name).copy_(getattr(model_a, name))
    sd_b = opt_b.state_dict()  # materializes the state (zero-lr step)
    steps_before = sorted({float(local(v)) for k, v in sd_b.items() if k.endswith(".step")})
    dcp.load(sd_b, checkpoint_id=path[0])
    opt_b.load_state_dict(sd_b)
    sd_b2 = opt_b.state_dict()
    restored = all_ranks(all(torch.equal(local(sd_a[k]).float(), local(sd_b2[k]).float()) for k in sd_a
                             if isinstance(sd_a[k], torch.Tensor)))
    grads(mesh, model_a, 3)
    grads(mesh, model_b, 3)
    torch.manual_seed(2000)
    opt_a.step()
    torch.manual_seed(2000)
    opt_b.step()
    same_next = all_ranks(all(torch.equal(local(getattr(model_a, n)), local(getattr(model_b, n))) for n in SHAPES))
    formats = sorted({k.rsplit(".", 1)[-1] for k in sd_a if k.startswith("state.")})
    result = dict(state_dtype=state_dtype, sharded_states_are_dtensors=sharded_ok, step_before_load=steps_before,
                  restored_equal=restored, next_step_identical=same_next, state_keys=formats)
    if rank == 0:
      print(f"[lpadamw-dcp] {result}", flush=True)
      shutil.rmtree(path[0], ignore_errors=True)
    ok = ok and sharded_ok and restored and same_next and steps_before == [0.0]
  dist.barrier()
  if rank == 0:
    print(f"[lpadamw-dcp] {'PASS' if ok else 'FAIL'}", flush=True)
  dist.destroy_process_group()
  if not ok:
    raise SystemExit(1)


if __name__ == "__main__":
  os.environ.setdefault("OMP_NUM_THREADS", "4")
  main()
