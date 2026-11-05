# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU (gloo) check of Muse Spark expert parallelism (EP > 1) against EP = 1.

For each config and EP degree, every rank runs different packed documents
through ``parallelize_muse_spark`` with SimpleFSDP over all ranks and the
routed experts sharded over ``ep`` (each EP token dispatcher; the ragged
all-to-all uses its CPU reference transport), in FP32. The loss is normalized by the global token
count. Checked against a single-process EP = 1 model that runs every rank's
documents with the same weights and the TPU lane's EP = 1 dispatcher:

* the global loss, every rank's logits (= the reference logits of its
  documents), and the reduced gradient of every parameter (including the
  post-expert norm shared by all experts, whose per-rank partial gradients
  cover different routed rows under EP);
* a negative control (the gradient of one rank's documents alone differs).

Configs: ``small`` (4 layers: sliding + full, RoPE + NoPE, a dense layer,
16 experts top-4), ``tiny`` (the public tiny checkpoint, needs
``MUSE_SPARK_TINY_DIR``: 2 layers, 16 experts top-4) and ``trunc256`` (the
released model's first 4 layer types and routing hyperparameters with all
256 experts, top-8, at reduced widths). Weights are random at O(1) scale
(``tiny``: the checkpoint's), routing biases included.

Run (the CPU stand-in provides TorchTPU's import-time names; drop it from
``PYTHONPATH`` where TorchTPU is installed):
  PYTHONPATH=torchtitan/experiments/tpu/tests/cpu_stubs:. \\
  torchrun --nproc_per_node=4 -m torchtitan.experiments.tpu.muse_spark.tests.ep_cpu_check \\
      [--configs small,tiny,trunc256] [--ep 2,4] [--ac none,full]
      [--dispatchers ragged,ragged_a2a,allgather,allgather_bf16]
      [--capacity 2.0] [--placement contiguous|interleaved|file:<path>]
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import math
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
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.experiments.tpu.muse_spark import build_muse_spark_config
from torchtitan.experiments.tpu.muse_spark import MuseSparkArgs
from torchtitan.experiments.tpu.muse_spark import MuseSparkStateDictAdapter
from torchtitan.experiments.tpu.muse_spark import parallelize_muse_spark
from torchtitan.experiments.tpu.muse_spark.expert_parallel import configure_expert_parallel
from torchtitan.experiments.tpu.muse_spark.state_dict_adapter import expert_slot_to_logical

_SMALL = {
    "vocab_size": 512,
    "hidden_size": 96,
    "num_hidden_layers": 4,
    "num_attention_heads": 6,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "intermediate_size": 64,
    "layer_types": [
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
    ],
    "layer_rope_theta": [500000.0, 0.0, 0.0, 500000.0],
    "sliding_window": 7,
    "mlp_layer_types": ["sparse", "sparse", "dense", "sparse"],
    "moe_hidden_size": 48,
    "moe_intermediate_size": 40,
    "num_local_experts": 16,
    "num_experts_per_tok": 4,
    "routed_scaling_factor": 1.25,
    "qk_scale_factor": 5.4730064863875,
    "output_multiplier": 0.17677669529663687,
    "final_logit_softcapping": 20.0,
    "rms_norm_eps": 1e-5,
    "post_norm_eps": 1e-8,
    "residual_gate_temperature": 0.3,
    "rope_parameters": {"rope_type": "default", "rope_theta": 500000.0},
    "tie_word_embeddings": False,
}

# The released 816B-A42B text config's first four layers (sliding + RoPE,
# full + NoPE, sliding, sliding), routing (256 experts, top-8, scale 1.0) and
# scalars, at reduced widths so that every rank holds a full replica cheaply.
_TRUNC256 = {
    **_SMALL,
    "vocab_size": 1024,
    "hidden_size": 128,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "intermediate_size": 64,
    "layer_types": [
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
    ],
    "layer_rope_theta": [500000.0, 0.0, 500000.0, 500000.0],
    "mlp_layer_types": ["sparse"] * 4,
    "moe_hidden_size": 64,
    "moe_intermediate_size": 32,
    "num_local_experts": 256,
    "num_experts_per_tok": 8,
    "routed_scaling_factor": 1.0,
}


@torch.no_grad()
def _randomize(model: torch.nn.Module, seed: int) -> None:
  """O(1)-scale random weights, norm gains, residual gates and routing biases."""
  gen = torch.Generator().manual_seed(seed)
  for name, tensor in list(model.named_parameters()) + [
      (n, b) for n, b in model.named_buffers() if n.endswith("expert_bias_E")
  ]:
    if name.endswith("expert_bias_E"):
      std = 0.05
    elif name.endswith(".gate") and tensor.ndim == 1:
      std = 0.5
    elif tensor.ndim == 1:
      std = 0.2
    elif name.startswith("lm_head"):
      std = 40.0 / math.sqrt(tensor.shape[-1])
    elif "tok_embeddings" in name:
      std = 1.0
    elif name.endswith("router.gate.weight"):
      std = 4.0 / math.sqrt(tensor.shape[-1])
    elif tensor.ndim == 3:  # [E, out, in] experts
      std = 1.0 / math.sqrt(tensor.shape[-1])
    else:
      std = 1.0 / math.sqrt(tensor.shape[-1])
    tensor.normal_(0.0, std, generator=gen)


def _tiny_state(cfg, tiny_dir: str) -> dict[str, torch.Tensor]:
  from safetensors.torch import load_file  # pylint: disable=g-import-not-at-top

  hf = {}
  for path in sorted(glob.glob(os.path.join(tiny_dir, "*.safetensors"))):
    hf.update(load_file(path))
  hf = {
      k: v.float()
      for k, v in hf.items()
      if k.startswith("model.language_model.") or k == "lm_head.weight"
  }
  return MuseSparkStateDictAdapter(cfg, None).from_hf(hf)


def _build_config(name: str):
  if name == "tiny":
    tiny_dir = os.environ["MUSE_SPARK_TINY_DIR"]
    with open(os.path.join(tiny_dir, "config.json")) as f:
      text = json.load(f)
    text = text.get("text_config", text)
  else:
    text = {"small": _SMALL, "trunc256": _TRUNC256}[name]
  args = MuseSparkArgs.from_hf_config(text)
  cfg = build_muse_spark_config(args, max_context_length=128, experts_compute_dtype=None)
  # The TPU lane's EP = 1 dispatcher (TPU local reorder / fixed-K FP32 combine),
  # as in the training recipes; FP32 experts (MuseSparkGroupedExperts).
  # pylint: disable-next=g-import-not-at-top
  from torchtitan.experiments.tpu.token_dispatcher import TpuTokenDispatcherConverter

  cfg = TpuTokenDispatcherConverter.Config().build().convert(cfg) or cfg
  return cfg, args


def _reference_state(name: str, cfg) -> dict[str, torch.Tensor]:
  model = cfg.build()
  model.init_states()
  if name == "tiny":
    model.load_state_dict(_tiny_state(cfg, os.environ["MUSE_SPARK_TINY_DIR"]), strict=True)
  else:
    _randomize(model, seed=7)
  return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _batches(world: int, vocab: int, tokens_per_rank: int = 72):
  """Equal-size rows per rank (as the trainer's fixed-size rows), split into 3
  documents at random points (the all-gather dispatcher needs equal token
  counts across the EP group)."""
  gen = torch.Generator().manual_seed(11)
  out = []
  for rank in range(world):
    cuts = sorted(int(c) for c in torch.randint(5, tokens_per_rank - 5, (2,), generator=gen))
    cuts[1] = max(cuts[1], cuts[0] + 1)
    lengths = [cuts[0], cuts[1] - cuts[0], tokens_per_rank - cuts[1]]
    docs = [torch.randint(3, vocab, (n,), generator=gen) for n in lengths]
    labels = [torch.randint(3, vocab, (n,), generator=gen) for n in lengths]
    positions = torch.cat([torch.arange(n) for n in lengths])
    out.append((torch.cat(docs), torch.cat(labels), positions, rank))
  return out


def _check(
    name: str, ep: int, ac: str, rank: int, world: int, dispatcher: str = "ragged",
    capacity: float = 2.0, placement: str = "contiguous",
) -> dict:
  cfg, args = _build_config(name)
  state = _reference_state(name, cfg)
  batches = _batches(world, args.vocab_size)
  total_tokens = sum(len(labels) for _, labels, _, _ in batches)

  def loss_and_logits(model, tokens, labels, positions):
    logits = model(tokens, positions=positions)
    return F.cross_entropy(logits, labels, reduction="sum") / total_tokens, logits

  reference = cfg.build()
  reference.init_states()
  reference.load_state_dict(state, strict=True)
  reference.train()
  ref_loss = 0.0
  ref_logits = None
  for tokens, labels, positions, r in batches:
    loss, logits = loss_and_logits(reference, tokens, labels, positions)
    loss.backward()
    ref_loss += loss.item()
    if r == rank:
      ref_logits = logits.detach()
  ref_grads = {k: p.grad.detach().clone() for k, p in reference.named_parameters()}

  ep_cfg = copy.deepcopy(cfg)
  configure_expert_parallel(
      ep_cfg, ep_degree=ep, capacity_factor=capacity, buffer_multiple=8,
      dispatcher=dispatcher, placement=placement,
  )
  with torch.device("meta"):
    model = ep_cfg.build()
  parallel_dims = ParallelDims(
      dp_shard=world, dp_replicate=1, cp=1, tp=1, pp=1, ep=ep, world_size=world
  )
  model = parallelize_muse_spark(
      model,
      parallel_dims=parallel_dims,
      training=TrainingConfig(
          mixed_precision_param="float32", mixed_precision_reduce="float32"
      ),
      parallelism=ParallelismConfig(
          data_parallel_shard_degree=world,
          expert_parallel_degree=ep,
          use_simple_fsdp=True,
      ),
      compile_config=CompileConfig(enable=False),
      ac_config=FullAC.Config(preserve_rng_state=False) if ac == "full" else None,
      dump_folder="/tmp",
  )
  model.to_empty(device="cpu")
  with torch.no_grad():
    model.init_weights(buffer_device=None)
  # Non-contiguous placements store expert rows in physical-slot order.
  num_experts = next(l.moe.num_experts for l in cfg.layers if l.moe is not None)
  if placement.startswith("file:"):
    # pylint: disable-next=g-import-not-at-top
    from torchtitan.experiments.tpu.muse_spark.expert_parallel import load_expert_placement

    maps = {int(k): torch.tensor(v) for k, v in load_expert_placement(placement, ep).items()}
  else:
    one = torch.tensor(expert_slot_to_logical(num_experts, ep, placement))
    maps = {i: one for i in range(len(cfg.layers))}

  def _layer(key):
    return int(key.split("layers.")[1].split(".")[0])

  def to_slots(key, value):
    return value.index_select(0, maps[_layer(key)]) if "inner_experts" in key else value

  def to_logical(key, value):
    return value.index_select(0, torch.argsort(maps[_layer(key)])) if "inner_experts" in key else value

  # set_model_state_dict replaces the dict's tensors with DTensors: pass a copy.
  set_model_state_dict(
      model,
      {k: to_slots(k, v).clone() for k, v in state.items()},
      options=StateDictOptions(full_state_dict=True, strict=True),
  )

  # Layout checks: experts on (efsdp, ep) with E / ep local experts per EP rank,
  # every other parameter (incl. the shared post-expert norm) on the dense mesh.
  layout = {}
  for pname, p in model.named_parameters():
    if "inner_experts" in pname:
      assert isinstance(p, DTensor) and p.device_mesh.mesh_dim_names == ("efsdp", "ep"), (
          pname, getattr(p, "device_mesh", None))
      layout["expert_placements"] = str(tuple(p.placements))
    elif isinstance(p, DTensor):
      assert "ep" not in (p.device_mesh.mesh_dim_names or ()), pname
  dispatchers = {
      type(m.token_dispatcher).__name__
      for m in model.modules()
      if hasattr(m, "token_dispatcher")
  }
  expected = (
      "AllGatherEPTokenDispatcher"
      if dispatcher.startswith("allgather")
      else "RaggedAllToAllTokenDispatcher"
  )
  assert dispatchers == {expected}, dispatchers

  model.train()
  tokens, labels, positions, _ = batches[rank]
  loss, logits = loss_and_logits(model, tokens, labels, positions)
  loss.backward()
  total = loss.detach().clone()
  dist.all_reduce(total)
  logit_diff = (logits.detach() - ref_logits).abs().max().item()

  worst, worst_name, norm_rel = 0.0, "", None
  grads = {}
  for pname, p in model.named_parameters():
    grad = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
    key = pname.replace("_checkpoint_wrapped_module.", "")
    grad = to_logical(key, grad)  # expert rows back to logical order
    grads[key] = grad
    ref = ref_grads[key]
    rel = ((grad - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()
    if rel > worst:
      worst, worst_name = rel, key
    if key.endswith("routed_experts.output_norm.weight") and norm_rel is None:
      norm_rel = rel

  control = cfg.build()
  control.init_states()
  control.load_state_dict(state, strict=True)
  control.train()
  loss_and_logits(control, *batches[0][:3])[0].backward()
  ckey = next(k for k in grads if k.endswith("routed_experts.output_norm.weight"))
  control_rel = (
      (control.get_parameter(ckey).grad - grads[ckey]).abs().max()
      / grads[ckey].abs().max()
  ).item()
  max_logit_diff = torch.tensor(logit_diff)
  dist.all_reduce(max_logit_diff, op=dist.ReduceOp.MAX)
  return {
      "config": name,
      "ep": ep,
      "ac": ac,
      "dispatcher": dispatcher,
      "placement": placement,
      "loss_ep": total.item(),
      "loss_ref": ref_loss,
      "max_abs_logit_diff": max_logit_diff.item(),
      "worst_rel_grad_diff": worst,
      "worst_param": worst_name,
      "post_expert_norm_rel_grad_diff": norm_rel,
      "params_compared": len(grads),
      "negative_control_rel_diff": control_rel,
      **layout,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--configs", default="small,tiny,trunc256")
  parser.add_argument("--ep", default="2,4")
  parser.add_argument("--ac", default="none,full")
  # allgather_bf16 (BF16 reduce-scatter combine) is meant for BF16 models; in
  # this FP32 check its rounding is amplified by the random weights.
  parser.add_argument("--dispatchers", default="ragged,ragged_a2a,allgather")
  # < 1 forces dropped routes: then only finiteness is checked (smoke test of
  # the overflow handling), not equality with EP = 1.
  parser.add_argument("--capacity", type=float, default=2.0)
  parser.add_argument("--placement", default="contiguous")
  opts = parser.parse_args()
  rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
  dist.init_process_group("gloo")
  torch.manual_seed(0)
  failures = []
  for name in opts.configs.split(","):
    if name == "tiny" and not os.environ.get("MUSE_SPARK_TINY_DIR"):
      if rank == 0:
        print("skip tiny: MUSE_SPARK_TINY_DIR is not set", flush=True)
      continue
    for ep in [int(e) for e in opts.ep.split(",")]:
      if world % ep:
        continue
      for ac, dispatcher in [
          (a, d) for a in opts.ac.split(",") for d in opts.dispatchers.split(",")
      ]:
        result = _check(name, ep, ac, rank, world, dispatcher, opts.capacity, opts.placement)
        # FP32 everywhere: the EP and EP = 1 paths differ only in summation
        # order (rank-partitioned grouped matmuls, reduce-scatter vs local
        # sums), amplified through layers of random O(1)-scale weights with
        # soft-capped logits up to ~20 (one layer: ~2e-6).
        if opts.capacity < 1.0:
          ok = math.isfinite(result["loss_ep"]) and math.isfinite(result["worst_rel_grad_diff"])
        else:
          ok = (
              abs(result["loss_ep"] - result["loss_ref"]) < 1e-6 * max(1.0, abs(result["loss_ref"]))
              and result["max_abs_logit_diff"] < 2e-4
              and result["worst_rel_grad_diff"] < 5e-4
              and result["negative_control_rel_diff"] > 1e-2
          )
        if not ok:
          failures.append(result)
        if rank == 0:
          print(f"[EP-CHECK] {'PASS' if ok else 'FAIL'} world={world} {json.dumps(result)}", flush=True)
  dist.barrier()
  dist.destroy_process_group()
  if failures:
    raise SystemExit(f"{len(failures)} EP checks failed")


if __name__ == "__main__":
  main()
