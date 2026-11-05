# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks of the Muse Spark text port against Hugging Face Transformers.

Reference: ``transformers.MuseSparkForConditionalGeneration`` (Transformers
>= 5.18, ``model_type="muse_spark"``), eager attention, FP32.

Checks:

1. Random-init parity (always runs): a small config with every structural
   feature (sliding + full layers, RoPE + NoPE, 16 experts, top-4), all
   weights, norms, gates and the routing bias randomized at O(1) scale;
   packed documents of uneven length against per-document HF runs. Logits,
   loss and every parameter gradient (mapped through the state-dict adapter).
2. Tiny public checkpoint parity (``MUSE_SPARK_TINY_DIR`` =
   ``trl-internal-testing/tiny-MuseSparkForConditionalGeneration``).
3. Truncated real dimensions (opt-in ``MUSE_SPARK_REAL_DIMS=1``; about
   125 GB of host memory): the first two released layers (sliding + RoPE,
   full + NoPE) at hidden 8192, 128/16 heads, 256 experts of 4096x4096, the
   real vocabulary, a shrunk sliding window so that it is exercised; random
   weights shared by both models (the expert and embedding weights are frozen
   to bound memory; every other gradient is compared).
4. Released-checkpoint metadata (``MUSE_SPARK_CKPT_META_DIR`` with
   ``config.json``, ``model.safetensors.index.json`` and optionally the shard
   headers as ``*.safetensors.header.json``): the hyperparameters, the
   parameter count, and every mapped key and shape of the 62-layer model.

Run: ``python -m pytest -q torchtitan/experiments/tpu/muse_spark/tests``. The
package imports ``torch_tpu`` through ``tpu/muse_glimmer``; on a host without
it, put its CPU import stand-in first on the path:
``PYTHONPATH=torchtitan/experiments/tpu/tests/cpu_stubs:$PWD``.
"""

from __future__ import annotations

import dataclasses
import glob
import importlib.util
import json
import math
import os
import types
import unittest

import torch
import torch.nn.functional as F

from torchtitan.experiments.tpu.muse_spark import model_registry
from torchtitan.experiments.tpu.muse_spark.configs import build_muse_spark_config
from torchtitan.experiments.tpu.muse_spark.configs import MuseSparkArgs
from torchtitan.experiments.tpu.muse_spark.state_dict_adapter import hf_text_keys_for_config
from torchtitan.experiments.tpu.muse_spark.state_dict_adapter import MuseSparkStateDictAdapter

try:
  import transformers
  from transformers import MuseSparkConfig
  from transformers import MuseSparkForConditionalGeneration
except ImportError:  # pragma: no cover
  transformers = None

TINY_DIR = os.environ.get("MUSE_SPARK_TINY_DIR")
CKPT_META_DIR = os.environ.get("MUSE_SPARK_CKPT_META_DIR")
REAL_DIMS = os.environ.get("MUSE_SPARK_REAL_DIMS") == "1"

# Released 816B-A42B text decoder, counted from the checkpoint's safetensors
# headers (model.language_model.* plus lm_head.weight); 15,872 of these are
# the 62 routing-bias vectors, which TorchTitan keeps as buffers.
RELEASED_TEXT_TENSOR_ELEMENTS = 819_990_552_064
RELEASED_ROUTING_BIAS_ELEMENTS = 62 * 256

# Tiny vision / audio towers for building HF reference models around a text
# config (the text-only TorchTitan model ignores them).
_TINY_VISION = {
    "hidden_act": "gelu",
    "hidden_size": 32,
    "intermediate_size": 64,
    "layer_types": ["window_attention", "full_attention"],
    "merge_size": 2,
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "patch_size": 14,
    "patch_temporal": 2,
    "pos_emb_height": 4,
    "pos_emb_width": 4,
}
_TINY_AUDIO = {
    "conv_kernel_size": 3,
    "depths": [1, 1, 1, 1],
    "hidden_sizes": [16, 16, 32, 32],
    "mlp_ratio": 2.0,
    "num_hidden_layers": 4,
    "patch_sizes": [16, 6, 5, 4],
}

_SMALL_TEXT = {
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
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "tie_word_embeddings": False,
}


def _hf_model(
    text_config: dict, *, skip_init: bool = False, experts: str = "eager"
) -> "MuseSparkForConditionalGeneration":
  """HF reference: eager attention; ``experts`` = "eager" (the modeling file's
  per-expert loop) or "grouped_mm" (Transformers' default kernel)."""
  config = MuseSparkConfig(
      text_config=text_config,
      vision_config=_TINY_VISION,
      audio_config={**_TINY_AUDIO, "out_hidden_size": text_config["hidden_size"]},
  )
  config._attn_implementation = "eager"  # pylint: disable=protected-access
  config.text_config._attn_implementation = "eager"  # pylint: disable=protected-access
  config._experts_implementation = experts  # pylint: disable=protected-access
  config.text_config._experts_implementation = experts  # pylint: disable=protected-access
  if skip_init:  # every text weight is overwritten by _randomize_text_weights
    from transformers.initialization import no_init_weights  # pylint: disable=g-import-not-at-top

    with no_init_weights():
      return MuseSparkForConditionalGeneration(config).float()
  return MuseSparkForConditionalGeneration(config).float()


@torch.no_grad()
def _randomize_text_weights(model, seed: int) -> None:
  """O(1)-scale random text weights, norms, gates and routing biases."""
  gen = torch.Generator().manual_seed(seed)
  tensors = list(model.named_parameters()) + [
      (n, b) for n, b in model.named_buffers() if "e_score_correction_bias" in n
  ]  # not the RoPE inverse frequencies
  for name, tensor in tensors:
    if not (name.startswith("model.language_model.") or name == "lm_head.weight"):
      continue
    # In place (no full-size temporaries: the real-dims experts are 34 GB).
    if "e_score_correction_bias" in name:
      std = 0.05
    elif name.endswith("residual_gate.gate"):
      std = 0.5
    elif tensor.ndim == 1:  # zero-centered norm gains
      std = 0.2
    elif name == "lm_head.weight":
      std = 40.0 / math.sqrt(tensor.shape[-1])
    elif name.endswith("embed_tokens.weight"):
      std = 1.0
    elif name.endswith("mlp.gate.weight"):
      std = 4.0 / math.sqrt(tensor.shape[-1])
    else:  # [.., out, in]
      std = 1.0 / math.sqrt(tensor.shape[-1])
    tensor.normal_(0.0, std, generator=gen)


def _tt_model_from_hf(hf_model, args: MuseSparkArgs, *, max_context_length: int, share: bool = False):
  cfg = build_muse_spark_config(
      args, max_context_length=max_context_length, experts_compute_dtype=None
  )
  adapter = MuseSparkStateDictAdapter(cfg, None)
  hf_state = {
      k: (v if share else v.detach().clone())
      for k, v in hf_model.state_dict().items()
      if k.startswith("model.language_model.") or k == "lm_head.weight"
  }
  tt_state = adapter.from_hf(hf_state)
  if share:
    with torch.device("meta"):
      model = cfg.build()
    # Materialize the buffers (RoPE caches, MoE counters) without the
    # parameters, then alias the HF tensors (persistent buffers included).
    for module in model.modules():
      if hasattr(module, "_init_self_buffers"):
        module._init_self_buffers(buffer_device=torch.device("cpu"))  # pylint: disable=protected-access
    model.load_state_dict(tt_state, strict=True, assign=True)
  else:
    model = cfg.build()
    model.init_states()
    model.load_state_dict(tt_state, strict=True)
  return model, adapter


def _packed_inputs(lengths: list[int], vocab: int, seed: int):
  gen = torch.Generator().manual_seed(seed)
  docs = [torch.randint(3, vocab, (n,), generator=gen) for n in lengths]
  labels = [torch.randint(3, vocab, (n,), generator=gen) for n in lengths]
  positions = torch.cat([torch.arange(n) for n in lengths])
  return docs, labels, positions


def _compare(test, hf_model, tt_model, adapter, docs, labels, positions, *,
             logit_atol: float, grad_rtol: float, skip_grads=()):
  hf_logits = torch.cat(
      [hf_model(input_ids=d[None], use_cache=False).logits[0] for d in docs]
  )
  tt_logits = tt_model(torch.cat(docs), positions=positions)
  target = torch.cat(labels)
  logit_diff = (hf_logits - tt_logits).abs().max().item()
  hf_loss = F.cross_entropy(hf_logits, target)
  tt_loss = F.cross_entropy(tt_logits, target)
  hf_loss.backward()
  tt_loss.backward()
  report = {
      "max_abs_logit": hf_logits.abs().max().item(),
      "max_abs_logit_diff": logit_diff,
      "loss": (hf_loss.item(), tt_loss.item()),
  }
  test.assertLess(logit_diff, logit_atol, report)
  test.assertLess(abs(hf_loss.item() - tt_loss.item()), logit_atol, report)

  tt_grads = adapter.to_hf({
      k: (p.grad if p.grad is not None else torch.zeros_like(p))
      for k, p in tt_model.named_parameters()
      if p.requires_grad
  })
  worst = 0.0
  compared = 0
  for name, param in hf_model.named_parameters():
    if not param.requires_grad or not (
        name.startswith("model.language_model.") or name == "lm_head.weight"
    ):
      continue
    if any(s in name for s in skip_grads):
      continue
    test.assertIn(name, tt_grads)
    hf_grad = param.grad if param.grad is not None else torch.zeros_like(param)
    scale = max(hf_grad.abs().max().item(), 1e-30)
    rel = (hf_grad - tt_grads[name]).abs().max().item() / scale
    worst = max(worst, rel)
    compared += 1
    test.assertLess(rel, grad_rtol, f"{name}: relative grad diff {rel:.3g}")
  report.update(worst_rel_grad_diff=worst, grads_compared=compared)
  return report


@unittest.skipIf(transformers is None, "needs transformers with MuseSpark")
class MuseSparkHFParityTest(unittest.TestCase):

  def setUp(self):
    torch.manual_seed(0)
    self._threads = torch.get_num_threads()

  def test_random_small_parity(self):
    for experts in ("eager", "grouped_mm"):
      hf_model = _hf_model(_SMALL_TEXT, experts=experts)
      self.assertEqual(
          hf_model.model.language_model.layers[0].mlp.experts.config._experts_implementation,  # pylint: disable=protected-access
          experts,
      )
      _randomize_text_weights(hf_model, seed=1)
      args = MuseSparkArgs.from_hf_config(_SMALL_TEXT)
      tt_model, adapter = _tt_model_from_hf(hf_model, args, max_context_length=64)
      hf_model.train()
      tt_model.train()
      docs, labels, positions = _packed_inputs(
          [19, 7, 30], _SMALL_TEXT["vocab_size"], seed=2
      )
      report = _compare(
          self, hf_model, tt_model, adapter, docs, labels, positions,
          logit_atol=1e-4, grad_rtol=1e-4,
      )
      print(f"random_small_parity[hf experts={experts}]", report)
      self.assertGreater(report["max_abs_logit"], 1.0)

  def test_random_small_parity_float64(self):
    """FP64 run of the same comparison.

    The gap does not vanish in FP64 because the reference still computes the
    attention softmax and the router logits in FP32 (as does the port's
    router); layer 0's attention differs by ~2e-7 and random O(1) layers
    amplify that to ~1e-5 at the logits, the same size as the FP32 gap. So the
    FP32 gap is rounding, not a difference in the math.
    """
    hf_model = _hf_model(_SMALL_TEXT, experts="eager")
    _randomize_text_weights(hf_model, seed=1)
    hf_model = hf_model.double()
    args = MuseSparkArgs.from_hf_config(_SMALL_TEXT)
    tt_model, _ = _tt_model_from_hf(hf_model, args, max_context_length=64)
    tt_model = tt_model.double()
    for module in tt_model.modules():
      if getattr(module, "cache", None) is not None:  # RoPE tables
        module.cache = module.cache.double()
    docs, _, positions = _packed_inputs([19, 7, 30], _SMALL_TEXT["vocab_size"], seed=2)
    with torch.no_grad():
      hf_logits = torch.cat(
          [hf_model(input_ids=d[None], use_cache=False).logits[0] for d in docs]
      )
      tt_logits = tt_model(torch.cat(docs), positions=positions)
    diff = (hf_logits - tt_logits).abs().max().item()
    print("random_small_parity_float64", {"max_abs_logit_diff": diff})
    # SoftCappedLinear returns FP32 logits by design; the decoder ran in FP64.
    self.assertEqual(tt_model.norm.weight.dtype, torch.float64)
    self.assertLess(diff, 1e-4)

  def test_router_bias_changes_routing(self):
    """Negative control: a different routing bias must change the logits."""
    hf_model = _hf_model(_SMALL_TEXT)
    _randomize_text_weights(hf_model, seed=3)
    args = MuseSparkArgs.from_hf_config(_SMALL_TEXT)
    tt_model, _ = _tt_model_from_hf(hf_model, args, max_context_length=64)
    docs, _, positions = _packed_inputs([24], _SMALL_TEXT["vocab_size"], seed=4)
    with torch.no_grad():
      ref = tt_model(torch.cat(docs), positions=positions)
      tt_model.layers["0"].moe.expert_bias_E.add_(
          torch.randn(_SMALL_TEXT["num_local_experts"]) * 0.5
      )
      changed = tt_model(torch.cat(docs), positions=positions)
    self.assertGreater((ref - changed).abs().max().item(), 1e-3)

  @unittest.skipUnless(TINY_DIR, "set MUSE_SPARK_TINY_DIR")
  def test_tiny_checkpoint_parity(self):
    hf_model = MuseSparkForConditionalGeneration.from_pretrained(
        TINY_DIR,
        dtype=torch.float32,
        attn_implementation="eager",
        experts_implementation="eager",
    )
    args = MuseSparkArgs.from_hf_config(TINY_DIR)
    tt_model, adapter = _tt_model_from_hf(hf_model, args, max_context_length=256)
    hf_model.train()
    tt_model.train()
    # 70 > the 32-token window of the tiny model's sliding layer.
    docs, labels, positions = _packed_inputs([70, 13], 199_000, seed=5)
    report = _compare(
        self, hf_model, tt_model, adapter, docs, labels, positions,
        logit_atol=1e-5, grad_rtol=1e-4,
    )
    print("tiny_checkpoint_parity", report)

  @unittest.skipUnless(REAL_DIMS, "set MUSE_SPARK_REAL_DIMS=1 (~125 GB RAM)")
  def test_truncated_real_dims_parity(self):
    args = MuseSparkArgs(num_hidden_layers=2, sliding_window=48)
    text = {
        **{f: getattr(args, f) for f in (
            "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim", "layer_types",
            "layer_rope_theta", "sliding_window", "mlp_layer_types", "moe_hidden_size",
            "moe_intermediate_size", "num_local_experts", "num_experts_per_tok",
            "routed_scaling_factor", "qk_scale_factor", "output_multiplier",
            "final_logit_softcapping", "rms_norm_eps", "post_norm_eps",
            "residual_gate_temperature")},
        "rope_parameters": {"rope_type": "default", "rope_theta": args.rope_theta},
        "pad_token_id": 200018,
        "bos_token_id": 200000,
        "eos_token_id": 200001,
        "tie_word_embeddings": False,
    }
    self.assertEqual(text["layer_types"], ["sliding_attention", "full_attention"])
    self.assertEqual(text["layer_rope_theta"], [500000.0, 0.0])
    hf_model = _hf_model(text, skip_init=True)
    _randomize_text_weights(hf_model, seed=6)
    frozen = (
        "mlp.experts.gate_up_proj", "mlp.experts.down_proj", "embed_tokens",
        "lm_head",
    )
    for name, param in hf_model.named_parameters():
      if any(s in name for s in frozen) or not (
          name.startswith("model.language_model.") or name == "lm_head.weight"
      ):
        param.requires_grad_(False)
    tt_model, adapter = _tt_model_from_hf(hf_model, args, max_context_length=256, share=True)
    for name, param in tt_model.named_parameters():
      if any(s in name for s in ("inner_experts", "tok_embeddings", "lm_head")):
        param.requires_grad_(False)
    hf_model.train()
    tt_model.train()
    docs, labels, positions = _packed_inputs([100, 37], 199_000, seed=7)
    report = _compare(
        self, hf_model, tt_model, adapter, docs, labels, positions,
        logit_atol=1e-4, grad_rtol=1e-3, skip_grads=frozen,
    )
    print("truncated_real_dims_parity", report)


def _tpu_moe_kernels_importable() -> bool:
  try:
    from torchtitan.experiments.tpu.muse_spark import tpu_moe_converters  # pylint: disable=g-import-not-at-top

    tpu_moe_converters()
    return True
  except (ImportError, RuntimeError):
    return False


@unittest.skipIf(transformers is None, "needs transformers with MuseSpark")
class MuseSparkTpuMoEKernelsTest(unittest.TestCase):
  """The TPU MoE converters must not change the math.

  ``model_registry`` applies ``TpuTokenDispatcherConverter`` and
  ``TpuGroupedExpertsConverter``; with Pallas off they must match the plain
  model with the same BF16 grouped experts. Needs ``torch_tpu`` or its CPU
  import stand-in (``torchtitan/experiments/tpu/tests/cpu_stubs`` on
  ``PYTHONPATH``).
  """

  @unittest.skipUnless(_tpu_moe_kernels_importable(), "needs torch_tpu or its CPU stubs")
  def test_tpu_moe_converters_match_bf16_experts(self):
    hf_model = _hf_model(_SMALL_TEXT)
    _randomize_text_weights(hf_model, seed=1)
    hf_state = {
        k: v.detach().clone()
        for k, v in hf_model.state_dict().items()
        if k.startswith("model.language_model.") or k == "lm_head.weight"
    }
    args = MuseSparkArgs.from_hf_config(_SMALL_TEXT)
    configs = {
        "tpu": model_registry("debugmodel", args=args, seq_len=64).model,
        "plain": build_muse_spark_config(
            args, max_context_length=64, experts_compute_dtype="bfloat16"
        ),
    }
    docs, labels, positions = _packed_inputs([19, 7, 30], _SMALL_TEXT["vocab_size"], seed=2)
    results = {}
    for name, cfg in configs.items():
      model = cfg.build()
      model.init_states()
      adapter = MuseSparkStateDictAdapter(cfg, None)
      model.load_state_dict(adapter.from_hf(dict(hf_state)), strict=True)
      model.train()
      logits = model(torch.cat(docs), positions=positions)
      F.cross_entropy(logits, torch.cat(labels)).backward()
      grads = adapter.to_hf({k: p.grad for k, p in model.named_parameters()})
      results[name] = (logits.detach(), grads, model)
    tpu_experts = results["tpu"][2].layers["0"].moe.routed_experts
    self.assertEqual(type(tpu_experts.inner_experts).__name__, "TpuGroupedExperts")
    self.assertEqual(
        type(tpu_experts.token_dispatcher).__name__, "TpuAllToAllTokenDispatcher"
    )
    diff = (results["tpu"][0] - results["plain"][0]).abs().max().item()
    self.assertLess(diff, 1e-5)
    for key, grad in results["plain"][1].items():
      rel = (results["tpu"][1][key] - grad).abs().max() / grad.abs().max().clamp_min(1e-30)
      self.assertLess(rel.item(), 1e-4, key)


@unittest.skipIf(transformers is None, "needs transformers with MuseSpark")
class MuseSparkLoRATest(unittest.TestCase):
  """LoRA on q/k/v/o (``torchtitan.components.lora.LoRAConverter``)."""

  def test_lora_qkvo(self):
    from torchtitan.components.lora import LoRAConverter  # pylint: disable=g-import-not-at-top

    hf_model = _hf_model(_SMALL_TEXT)
    _randomize_text_weights(hf_model, seed=1)
    hf_state = {
        k: v.detach().clone()
        for k, v in hf_model.state_dict().items()
        if k.startswith("model.language_model.") or k == "lm_head.weight"
    }
    args = MuseSparkArgs.from_hf_config(_SMALL_TEXT)
    base_cfg = build_muse_spark_config(args, max_context_length=64, experts_compute_dtype=None)
    lora_cfg = LoRAConverter(
        LoRAConverter.Config(rank=4, alpha=8.0, target_modules=["wq", "wkv", "wo"])
    ).convert(
        build_muse_spark_config(args, max_context_length=64, experts_compute_dtype=None)
    )
    models = {}
    for name, cfg in (("base", base_cfg), ("lora", lora_cfg)):
      model = cfg.build()
      model.init_states()
      adapter = MuseSparkStateDictAdapter(cfg, None)
      missing, unexpected = model.load_state_dict(adapter.from_hf(dict(hf_state)), strict=False)
      self.assertEqual(unexpected, [])
      self.assertTrue(all(".lora_" in k for k in missing), missing)
      # Every non-LoRA key round-trips through to_hf.
      self.assertEqual(set(adapter.to_hf(model.state_dict())), set(hf_text_keys_for_config(cfg)))
      models[name] = model
    trainable = [k for k, p in models["lora"].named_parameters() if p.requires_grad]
    self.assertTrue(trainable)
    self.assertTrue(all(".lora_" in k for k in trainable), trainable)
    self.assertEqual(len(trainable), 2 * 4 * args.num_hidden_layers)  # a/b x q/k/v/o x layers
    docs, labels, positions = _packed_inputs([19, 7], _SMALL_TEXT["vocab_size"], seed=2)
    with torch.no_grad():  # lora_b starts at zero: identical outputs
      base_logits = models["base"](torch.cat(docs), positions=positions)
      lora_logits = models["lora"](torch.cat(docs), positions=positions)
    self.assertEqual((base_logits - lora_logits).abs().max().item(), 0.0)
    loss = F.cross_entropy(models["lora"](torch.cat(docs), positions=positions), torch.cat(labels))
    loss.backward()
    grads = {k: p.grad for k, p in models["lora"].named_parameters() if p.requires_grad}
    self.assertTrue(all(g is not None for g in grads.values()))
    self.assertGreater(
        max(g.abs().max().item() for k, g in grads.items() if k.endswith("lora_b.weight")), 0.0
    )


class MuseSparkConfigTest(unittest.TestCase):

  def test_flavors_build_on_meta(self):
    for flavor in ("debugmodel", "tiny-hf", "816B-A42B-2L"):
      spec = model_registry(flavor, tpu_moe_kernels=False)
      with torch.device("meta"):
        model = spec.model.build()
      self.assertGreater(sum(p.numel() for p in model.parameters()), 0)

  def test_released_parameter_count(self):
    spec = model_registry("816B-A42B", tpu_moe_kernels=False)
    with torch.device("meta"):
      model = spec.model.build()
    nparams = sum(p.numel() for p in model.parameters())
    state = sum(v.numel() for v in model.state_dict().values())
    self.assertEqual(nparams, RELEASED_TEXT_TENSOR_ELEMENTS - RELEASED_ROUTING_BIAS_ELEMENTS)
    self.assertEqual(state, RELEASED_TEXT_TENSOR_ELEMENTS)

  def test_layer_pattern(self):
    args = MuseSparkArgs()
    spec = model_registry("816B-A42B", tpu_moe_kernels=False)
    for i, layer in enumerate(spec.model.layers):
      full = i % 4 == 1
      self.assertEqual(layer.attention.window_size is None, full)
      self.assertEqual(layer.attention.rope is None, full)
      self.assertEqual(args.layer_types[i], "full_attention" if full else "sliding_attention")
    self.assertEqual(spec.model.layers[0].attention.window_size, 2048)

  @unittest.skipUnless(CKPT_META_DIR, "set MUSE_SPARK_CKPT_META_DIR")
  def test_released_config_and_keys(self):
    self.assertEqual(MuseSparkArgs.from_hf_config(CKPT_META_DIR), MuseSparkArgs())
    spec = model_registry("816B-A42B", tpu_moe_kernels=False)
    adapter = MuseSparkStateDictAdapter(spec.model, CKPT_META_DIR)  # checks the index
    with open(os.path.join(CKPT_META_DIR, "model.safetensors.index.json")) as f:
      weight_map = json.load(f)["weight_map"]
    text_keys = {
        k for k in weight_map
        if k.startswith("model.language_model.") or k == "lm_head.weight"
    }
    self.assertEqual(set(hf_text_keys_for_config(spec.model)), text_keys)
    headers = sorted(glob.glob(os.path.join(CKPT_META_DIR, "*.safetensors.header.json")))
    if not headers:
      return
    shapes = {}
    for path in headers:
      with open(path) as f:
        for key, meta in json.load(f).items():
          if key != "__metadata__":
            shapes[key] = tuple(meta["shape"])
    with torch.device("meta"):
      model = spec.model.build()
    expected = {k: tuple(v.shape) for k, v in adapter.to_hf(model.state_dict()).items()}
    self.assertEqual(set(expected), text_keys)
    for key, shape in expected.items():
      self.assertEqual(shapes[key], shape, key)

  def test_truncated_flavor_ignores_deeper_layers(self):
    spec = model_registry("816B-A42B-2L", tpu_moe_kernels=False)
    adapter = MuseSparkStateDictAdapter(spec.model, None)
    self.assertEqual(adapter._classify_hf_key(  # pylint: disable=protected-access
        "model.language_model.layers.5.mlp.gate.weight"), "excluded")
    self.assertEqual(adapter._classify_hf_key(  # pylint: disable=protected-access
        "model.vision_tower.layers.0.attn.q_proj.weight"), "excluded")
    self.assertEqual(adapter._classify_hf_key(  # pylint: disable=protected-access
        "model.language_model.layers.1.mlp.gate.weight"), "expected")
    self.assertEqual(adapter._classify_hf_key(  # pylint: disable=protected-access
        "model.language_model.layers.1.bogus.weight"), "unknown")

  def test_expert_parallel_module_is_imported_lazily(self):
    """EP > 1 and placement files need ``expert_parallel.py``; EP 1 does not."""
    if importlib.util.find_spec(
        "torchtitan.experiments.tpu.muse_spark.expert_parallel"
    ) is not None:
      self.skipTest("expert_parallel.py is present")
    from torchtitan.config import ParallelismConfig  # pylint: disable=g-import-not-at-top

    spec = model_registry("debugmodel", tpu_moe_kernels=False)
    spec.model.update_from_config(
        config=types.SimpleNamespace(parallelism=ParallelismConfig())
    )
    with self.assertRaisesRegex(NotImplementedError, "expert_parallel.py"):
      spec.model.update_from_config(
          config=types.SimpleNamespace(
              parallelism=ParallelismConfig(expert_parallel_degree=2)
          )
      )
    placed = dataclasses.replace(
        spec.model, ep_expert_placement="file:placement.json"
    )
    with self.assertRaisesRegex(NotImplementedError, "expert_parallel.py"):
      MuseSparkStateDictAdapter(placed, None)

  def test_ep_grad_reduce_dtype_reaches_the_built_model(self):
    """None by default: the expert gradients follow the training reduce dtype.

    Expert parallelism reads a recipe's override from the built model's
    config, so the field must survive ``build``.
    """
    for flavor in ("debugmodel", "tiny-hf", "816B-A42B-2L", "816B-A42B"):
      config = model_registry(flavor, tpu_moe_kernels=False).model
      self.assertIsNone(config.ep_grad_reduce_dtype, flavor)
      self.assertIn(
          "ep_grad_reduce_dtype", {f.name for f in dataclasses.fields(config)}
      )
    config = model_registry("debugmodel", tpu_moe_kernels=False).model
    config.ep_grad_reduce_dtype = "bfloat16"
    with torch.device("meta"):
      model = config.build()
    self.assertEqual(model.config.ep_grad_reduce_dtype, "bfloat16")


class ChatTemplateTest(unittest.TestCase):
  """TorchTitan's tokenizer must compile HF ``{% generation %}`` blocks.

  The Muse Spark chat template wraps assistant turns in them; with
  ``MUSE_SPARK_TINY_DIR`` the full template is rendered and compared with
  Transformers (TorchTitan adds BOS at encode time, not in the template).
  """

  def test_generation_block(self):
    from torchtitan.components.tokenizer import BaseTokenizer  # pylint: disable=g-import-not-at-top

    class _Tok(BaseTokenizer):

      def encode(self, *args, **kwargs):
        raise NotImplementedError

      def decode(self, *args, **kwargs):
        raise NotImplementedError

      def get_vocab_size(self) -> int:
        return 0

    tok = _Tok()
    tok.set_chat_template(
        "{% for m in messages %}<{{ m.role }}>"
        "{%- if m.role == 'assistant' %}{% generation %}{{ m.content }}"
        "{% endgeneration %}{% else %}{{ m.content }}{% endif %}{% endfor %}"
    )
    out = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    )
    self.assertEqual(out, "<user>hi<assistant>yo")

  @unittest.skipUnless(TINY_DIR and transformers is not None, "set MUSE_SPARK_TINY_DIR")
  def test_muse_spark_template_matches_transformers(self):
    from torchtitan.components.tokenizer import HuggingFaceTokenizer  # pylint: disable=g-import-not-at-top
    from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top

    tt_tok = HuggingFaceTokenizer(tokenizer_path=TINY_DIR)
    hf_tok = AutoTokenizer.from_pretrained(TINY_DIR)
    messages = [
        {"role": "system", "content": "Rank the products."},
        {"role": "user", "content": "Find running shoes."},
        {"role": "assistant", "content": "Here are three options."},
        {"role": "user", "content": "Cheaper?"},
    ]
    for add_generation_prompt in (True, False):
      ours = tt_tok.apply_chat_template(
          messages, add_generation_prompt=add_generation_prompt
      )
      theirs = hf_tok.apply_chat_template(
          messages, tokenize=False, add_generation_prompt=add_generation_prompt
      )
      self.assertEqual(hf_tok.bos_token + ours, theirs)


if __name__ == "__main__":
  unittest.main()
