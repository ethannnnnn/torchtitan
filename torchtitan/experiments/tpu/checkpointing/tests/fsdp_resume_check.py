# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Two-rank CPU (gloo) check: bit-exact resume of an FSDP-sharded model.

A model is sharded over 2 ranks and trained with AdamW through TorchTitan's
``CheckpointManager`` (DTensor shards in DCP). Run A trains 6 steps from the
Hugging Face base weights and checkpoints at step 3; run B builds a fresh
sharded model with a different init, resumes from step 3 and trains to step 6.
Losses and the gathered parameters must be bit-identical. With ``--lora`` the
model gets rank-4 adapters and adapter-only checkpoints, ``--tuned`` uses
``tune_checkpoint_manager``.

Models (``--model``):

* ``llama3`` (default): TorchTitan's Llama 3 debugmodel sharded with FSDP2
  (``fully_shard``), LoRA from ``LoRAConverter`` on ``wqkv`` / ``wo``; rank 0
  writes the Hugging Face base checkpoint.
* ``muse_spark``: the Muse Spark port sharded with SimpleFSDP, LoRA on the
  q/k/v/o projections. Needs ``MUSE_SPARK_TINY_DIR``, the port and its LoRA
  converter (``post_training.lora``).

Run: torchrun --nproc_per_node=2 -m torchtitan.experiments.tpu.checkpointing.tests.fsdp_resume_check [--model M] [--lora] [--tuned]
"""

import argparse
import os
import shutil
import tempfile
import types

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor


def build_muse_spark(lora: bool, seed: int, world: int, hf_dir: str):
    del hf_dir  # the public tiny checkpoint
    from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
    from torchtitan.distributed import ParallelDims
    from torchtitan.experiments.tpu.muse_spark import (
        MuseSparkArgs,
        MuseSparkStateDictAdapter,
        build_muse_spark_config,
        parallelize_muse_spark,
    )

    tiny = os.environ["MUSE_SPARK_TINY_DIR"]
    args = MuseSparkArgs.from_hf_config(tiny)

    def config():
        return build_muse_spark_config(args, max_context_length=256, experts_compute_dtype=None)

    spec = types.SimpleNamespace(model=config(), state_dict_adapter=MuseSparkStateDictAdapter)
    if lora:
        from torchtitan.experiments.tpu.post_training.lora import LoRAOptions, apply_lora

        apply_lora(spec, LoRAOptions(rank=4, alpha=8.0))
    with torch.device("meta"):
        model = spec.model.build()
    dims = ParallelDims(dp_shard=world, dp_replicate=1, cp=1, tp=1, pp=1, ep=1, world_size=world)
    model = parallelize_muse_spark(
        model, parallel_dims=dims,
        training=TrainingConfig(mixed_precision_param="float32", mixed_precision_reduce="float32"),
        parallelism=ParallelismConfig(data_parallel_shard_degree=world, use_simple_fsdp=True),
        compile_config=CompileConfig(enable=False), ac_config=None, dump_folder="/tmp",
    )
    model.to_empty(device="cpu")
    torch.manual_seed(seed)
    with torch.no_grad():
        model.init_weights(buffer_device=None)
    adapter = spec.state_dict_adapter(config(), tiny)
    return model, adapter


def build_llama3(lora: bool, seed: int, world: int, hf_dir: str):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    from torchtitan.experiments.tpu.checkpointing.tests import fixtures

    model, spec = fixtures.build_llama3(lora=lora, seed=seed)  # same init on every rank
    mesh = init_device_mesh("cpu", (world,))
    for block in model.layers.values():
        fully_shard(block, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model, spec.state_dict_adapter(spec.model, hf_dir)


def write_llama3_base(hf_dir: str) -> None:
    from torchtitan.experiments.tpu.checkpointing.tests import fixtures

    base, spec = fixtures.build_llama3(lora=False, seed=0)
    fixtures.write_hf_checkpoint(hf_dir, base.state_dict(), spec)


# vocabulary and sequence length of the training batches
MODELS = {"llama3": (build_llama3, 2048, 32), "muse_spark": (build_muse_spark, 199_000, 40)}


class Counter:
    def __init__(self):
        self.step = 0

    def state_dict(self):
        return {"step": torch.tensor(self.step)}

    def load_state_dict(self, sd):
        self.step = int(sd["step"])


def manager(model, adapter, folder, interval):
    from torchtitan.components.checkpointer.dcp import CheckpointManager
    from torchtitan.components.optimizer.lr_scheduler import LRSchedulersContainer
    from torchtitan.components.optimizer.optimizer import OptimizersContainer, ParamGroupConfig

    optim = OptimizersContainer.Config(
        param_groups=[ParamGroupConfig(pattern=".*", optimizer_name="AdamW",
                                       optimizer_kwargs={"lr": 3e-3, "weight_decay": 0.01})],
        implementation="for-loop",
    ).build(model_parts=[model])
    lr = LRSchedulersContainer.Config(warmup_steps=2).build(optimizers=optim, training_steps=8)
    ts = Counter()
    ckpt = CheckpointManager(
        CheckpointManager.Config(enable=True, folder="checkpoint", interval=interval,
                                 initial_load_in_hf=True, last_save_model_only=False, keep_latest_k=0),
        dataloader=Counter(), model_parts=[model], optimizers=optim, lr_schedulers=lr,
        states={"train_state": ts}, sd_adapter=adapter, base_folder=folder,
    )
    return ckpt, optim, lr, ts


def train(model, optim, lr, ts, ckpt, until, rank, vocab, seq):
    losses = []
    while ts.step < until:
        ts.step += 1
        g = torch.Generator().manual_seed(1000 + 10 * ts.step + rank)  # each rank: its own batch
        tokens = torch.randint(0, vocab, (2, seq), generator=g)
        labels = torch.randint(0, vocab, (2, seq), generator=g)
        optim.zero_grad()
        logits = model(tokens.reshape(-1), positions=torch.arange(seq).repeat(2))
        loss = F.cross_entropy(logits.float(), labels.reshape(-1))
        loss.backward()
        optim.step()
        lr.step()
        total = loss.detach().clone()
        dist.all_reduce(total)
        losses.append(total.item())
        ckpt.save(curr_step=ts.step)
    return losses


def gathered(model):
    return {n: (p.full_tensor() if isinstance(p, DTensor) else p).detach().clone()
            for n, p in model.named_parameters()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), default="llama3")
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--tuned", action="store_true")
    a = ap.parse_args()
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    from torchtitan.experiments.tpu.checkpointing.dcp_parallel import tune_checkpoint_manager
    from torchtitan.experiments.tpu.checkpointing.lora_checkpoint import enable_adapter_only_checkpoints

    build, vocab, seq = MODELS[a.model]
    tmp = [tempfile.mkdtemp(prefix="fsdp-resume-") if rank == 0 else None]
    dist.broadcast_object_list(tmp)
    root = tmp[0]
    hf_dir = os.path.join(root, "hf")
    if a.model == "llama3" and rank == 0:
        write_llama3_base(hf_dir)
    dist.barrier()

    def setup(seed, folder, interval):
        model, adapter = build(a.lora, seed, world, hf_dir)
        ckpt, optim, lr, ts = manager(model, adapter, folder, interval)
        if a.lora:
            enable_adapter_only_checkpoints(ckpt, [model])
        if a.tuned:
            tune_checkpoint_manager(ckpt, save_threads=2, load_threads=2)
        ckpt.load()
        return model, ckpt, optim, lr, ts

    model, ckpt, optim, lr, ts = setup(1, os.path.join(root, "a"), 3)
    losses_a = train(model, optim, lr, ts, ckpt, 6, rank, vocab, seq)
    final_a = gathered(model)
    if rank == 0:
        os.makedirs(os.path.join(root, "b", "checkpoint"))
        shutil.copytree(os.path.join(root, "a", "checkpoint", "step-3"), os.path.join(root, "b", "checkpoint", "step-3"))
    dist.barrier()
    model_b, ckpt_b, optim_b, lr_b, ts_b = setup(2, os.path.join(root, "b"), 100)
    assert ts_b.step == 3, ts_b.step
    losses_b = train(model_b, optim_b, lr_b, ts_b, ckpt_b, 6, rank, vocab, seq)
    final_b = gathered(model_b)
    same_params = all(torch.equal(v, final_b[k]) for k, v in final_a.items())
    if rank == 0:
        from torch.distributed.checkpoint import FileSystemReader

        step3 = os.path.join(root, "a", "checkpoint", "step-3")
        md = FileSystemReader(step3).read_metadata().state_dict_metadata
        model_keys = [k for k in md if not k.startswith(("optimizer", "lr_scheduler", "dataloader", "train_state"))]
        files = sorted(f for f in os.listdir(step3) if f.endswith(".distcp"))
        print(f"fsdp_resume_check model={a.model} world={world} lora={a.lora} tuned={a.tuned}: "
              f"losses_a[3:]={losses_a[3:]} losses_b={losses_b} same_losses={losses_a[3:] == losses_b} "
              f"same_params={same_params} model_keys_in_ckpt={len(model_keys)} files={files}", flush=True)
        assert losses_a[3:] == losses_b and same_params
        if a.lora:
            assert model_keys and all(".lora_" in k for k in model_keys), model_keys[:5]
    # Remove the checkpoints only once every rank is done with them.
    dist.barrier()
    if rank == 0:
        shutil.rmtree(root, ignore_errors=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
