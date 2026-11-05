# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Two-rank CPU (gloo) check of ``parallelize_muse_spark`` with SimpleFSDP.

Each rank runs different packed documents through the FSDP-sharded model
(FP32 parameters and reductions); the loss is normalized by the global token
count. The reduced gradients must equal a single-process reference that runs
all documents, and must differ from the gradient of one rank's documents
alone (negative control).

Run: ``torchrun --nproc_per_node=2 -m torchtitan.experiments.tpu.muse_spark.tests.fsdp_cpu_check``
(on a host without ``torch_tpu``, with its CPU import stand-in
``torchtitan/experiments/tpu/tests/cpu_stubs`` first on ``PYTHONPATH``).
"""

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.checkpoint.state_dict import set_model_state_dict
from torch.distributed.checkpoint.state_dict import StateDictOptions
from torch.distributed.tensor import DTensor

from torchtitan.config import CompileConfig
from torchtitan.config import ParallelismConfig
from torchtitan.config import TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.tpu.muse_spark import build_muse_spark_config
from torchtitan.experiments.tpu.muse_spark import MuseSparkArgs
from torchtitan.experiments.tpu.muse_spark import MuseSparkStateDictAdapter
from torchtitan.experiments.tpu.muse_spark import parallelize_muse_spark
from torchtitan.experiments.tpu.muse_spark.tests import test_muse_spark_cpu as t


def main() -> None:
  rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
  dist.init_process_group("gloo")
  torch.manual_seed(0)
  hf_model = t._hf_model(t._SMALL_TEXT)  # pylint: disable=protected-access
  t._randomize_text_weights(hf_model, 1)  # pylint: disable=protected-access
  args = MuseSparkArgs.from_hf_config(t._SMALL_TEXT)  # pylint: disable=protected-access
  cfg = build_muse_spark_config(args, max_context_length=64, experts_compute_dtype=None)
  adapter = MuseSparkStateDictAdapter(cfg, None)

  def fresh_state():
    return adapter.from_hf({
        k: v.detach().clone()
        for k, v in hf_model.state_dict().items()
        if k.startswith("model.language_model.") or k == "lm_head.weight"
    })

  batches = [
      t._packed_inputs(lengths, 512, seed)  # pylint: disable=protected-access
      for lengths, seed in (([19, 7, 30], 2), ([11, 25, 20], 3))
  ][:world]
  total_tokens = sum(len(torch.cat(labels)) for _, labels, _ in batches)

  def loss_of(model, docs, labels, positions):
    logits = model(torch.cat(docs), positions=positions)
    return F.cross_entropy(logits, torch.cat(labels), reduction="sum") / total_tokens

  reference = cfg.build()
  reference.init_states()
  reference.load_state_dict(fresh_state(), strict=True)
  reference.train()
  sum(loss_of(reference, *batch) for batch in batches).backward()
  ref_grads = {k: p.grad for k, p in reference.named_parameters()}

  with torch.device("meta"):
    model = cfg.build()
  parallel_dims = ParallelDims(
      dp_shard=world, dp_replicate=1, cp=1, tp=1, pp=1, ep=1, world_size=world
  )
  model = parallelize_muse_spark(
      model,
      parallel_dims=parallel_dims,
      training=TrainingConfig(
          mixed_precision_param="float32", mixed_precision_reduce="float32"
      ),
      parallelism=ParallelismConfig(data_parallel_shard_degree=world, use_simple_fsdp=True),
      compile_config=CompileConfig(enable=False),
      ac_config=None,
      dump_folder="/tmp",
  )
  model.to_empty(device="cpu")
  with torch.no_grad():
    model.init_weights(buffer_device=None)
  set_model_state_dict(model, fresh_state(), options=StateDictOptions(full_state_dict=True))
  model.train()
  loss = loss_of(model, *batches[rank])
  loss.backward()
  total = loss.detach().clone()
  dist.all_reduce(total)

  worst = 0.0
  grads = {}
  for name, param in model.named_parameters():
    grad = param.grad.full_tensor() if isinstance(param.grad, DTensor) else param.grad
    grads[name] = grad
    ref = ref_grads[name]
    worst = max(worst, ((grad - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item())

  control = cfg.build()
  control.init_states()
  control.load_state_dict(fresh_state(), strict=True)
  control.train()
  loss_of(control, *batches[0]).backward()
  key = "layers.0.moe.router.gate.weight"
  control_rel = (
      (control.get_parameter(key).grad - grads[key]).abs().max() / grads[key].abs().max()
  ).item()
  if rank == 0:
    print(
        f"world={world} loss: fsdp={total.item():.7f}"
        f" reference={sum(loss_of(reference, *b) for b in batches).item():.7f};"
        f" worst relative grad diff={worst:.3g} over {len(grads)} params;"
        f" negative control relative diff={control_rel:.3g}"
    )
    assert worst < 1e-5, worst
    assert control_rel > 1e-2, control_rel
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
