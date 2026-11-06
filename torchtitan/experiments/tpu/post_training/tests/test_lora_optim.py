# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests: attention LoRA (conversion, freezing, merge, adapters) and the
low-bit AdamW options, on TorchTitan's llama3 debug model."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from torchtitan.components.optimizer.lr_scheduler import LRSchedulersContainer
from torchtitan.experiments.tpu.post_training import lora as lora_mod
from torchtitan.experiments.tpu.post_training.optimizers import (
    PostTrainingOptimizersContainer,
    adamw_config,
)
from torchtitan.models.llama3 import model_registry


def _spec():
  return model_registry("debugmodel", seq_len=32, attn_backend="sdpa")


def _build(spec, seed=0):
  torch.manual_seed(seed)
  model = spec.model.build()
  model.init_states(buffer_device=None)
  return model


def _tokens(seed=1):
  g = torch.Generator().manual_seed(seed)
  return torch.randint(0, 2048, (32,), generator=g)  # one flattened row


class LoRATest(unittest.TestCase):

  def setUp(self):
    self.base_spec = _spec()
    self.spec = _spec()
    self.options = lora_mod.LoRAOptions(rank=4, alpha=8.0)
    self.applied = lora_mod.apply_lora(self.spec, self.options)
    self.matched = self.applied.matched
    self.layouts = self.applied.fused_layouts

  def _pair(self):
    base = _build(self.base_spec)
    model = _build(self.spec, seed=3)
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    self.assertFalse(unexpected)
    self.assertTrue(all(lora_mod.is_lora_key(k) for k in missing))
    return base, model

  def test_targets_are_attention_projections_only(self):
    layers = len(self.base_spec.model.layers)
    self.assertEqual(len(self.matched), 2 * layers)  # fused wqkv + wo per layer
    self.assertTrue(
        all(
            m.endswith(("attention.qkv_linear.wqkv", "attention.wo"))
            for m in self.matched
        )
    )
    self.assertEqual(len(self.layouts), layers)  # one fused wqkv per layer
    self.assertEqual(
        next(iter(self.layouts.values())).n_kv_heads,
        self.base_spec.model.layers[0].attention.qkv_linear.n_kv_heads,
    )
    only_q = _spec()
    with self.assertRaises(
        ValueError
    ):  # the fused projection is not a q-only target
      lora_mod.apply_lora(only_q, lora_mod.LoRAOptions(projections=("q",)))
    with self.assertRaises(ValueError):
      lora_mod.target_regex(("x",))

  def test_freezing_and_identity_at_init(self):
    base, model = self._pair()
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    self.assertTrue(trainable)
    self.assertTrue(
        all(n.endswith(("lora_a.weight", "lora_b.weight")) for n in trainable)
    )
    report = lora_mod.lora_parameter_report(model)
    self.assertEqual(report["trainable"], report["adapter"])
    tokens = _tokens()
    torch.testing.assert_close(model(tokens), base(tokens))

  def test_merge_matches_adapter_forward(self):
    base, model = self._pair()
    with torch.no_grad():
      for name, p in model.named_parameters():
        if name.endswith("lora_b.weight"):
          p.normal_(0, 0.05)
    tokens = _tokens()
    with torch.no_grad():
      lora_out = model(tokens)
      base_out = base(tokens)
    self.assertGreater(
        float((lora_out - base_out).abs().max()), 1e-4
    )  # adapters act
    merged = lora_mod.merge_lora_state_dict(
        model.state_dict(), self.options.scaling, self.layouts
    )
    self.assertFalse(any(lora_mod.is_lora_key(k) for k in merged))
    base.load_state_dict(merged)
    with torch.no_grad():
      torch.testing.assert_close(base(tokens), lora_out, rtol=1e-4, atol=1e-5)

  def test_gradients_reach_only_adapters(self):
    _, model = self._pair()
    loss = model(_tokens()).float().pow(2).mean()
    loss.backward()
    for name, p in model.named_parameters():
      if p.requires_grad:
        self.assertIsNotNone(p.grad, name)
      else:
        self.assertIsNone(p.grad, name)
    # lora_b starts at zero, so only lora_b receives a nonzero gradient at step 0.
    grads_b = [
        p.grad.abs().sum()
        for n, p in model.named_parameters()
        if n.endswith("lora_b.weight")
    ]
    self.assertTrue(all(float(g) > 0 for g in grads_b))

  def test_state_dict_adapter(self):
    _, model = self._pair()
    with torch.no_grad():
      for name, p in model.named_parameters():
        if name.endswith("lora_b.weight"):
          p.normal_(0, 0.05)
    state = model.state_dict()
    adapter = self.spec.state_dict_adapter(self.spec.model, None)
    inner = self.base_spec.state_dict_adapter(self.base_spec.model, None)
    self.assertIsInstance(adapter, self.base_spec.state_dict_adapter)
    base_only, lora_only = lora_mod.split_lora_state_dict(state)
    hf_load = adapter.to_hf(state)
    self.assertEqual(set(hf_load), set(inner.to_hf(base_only)))
    exported = adapter.to_hf_for_export(state)
    expected = inner.to_hf(
        lora_mod.merge_lora_state_dict(
            state, self.options.scaling, self.layouts
        )
    )
    self.assertEqual(set(exported), set(expected))
    for key in expected:
      torch.testing.assert_close(exported[key], expected[key])
    changed = [k for k in expected if not torch.equal(expected[k], hf_load[k])]
    self.assertTrue(changed and all("self_attn" in k for k in changed))
    roundtrip = adapter.from_hf(hf_load)
    self.assertEqual(set(roundtrip), set(base_only))
    self.assertTrue(lora_only)

  def test_peft_layout(self):
    _, model = self._pair()
    with torch.no_grad():
      for name, p in model.named_parameters():
        if name.endswith("lora_b.weight"):
          p.normal_(0, 0.05)
    state = model.state_dict()
    adapter = self.spec.state_dict_adapter(self.spec.model, None)
    tensors = lora_mod.peft_adapter_state_dict(state, adapter=adapter)
    # The fused wqkv adapter becomes q_proj / k_proj / v_proj sharing lora_A.
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
      self.assertIn(
          f"base_model.model.model.layers.0.self_attn.{proj}.lora_A.weight",
          tensors,
      )
    a = tensors[
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    ]
    self.assertTrue(
        torch.equal(
            a,
            tensors[
                "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight"
            ],
        )
    )
    # PEFT delta (B @ A) equals the merged HF weight minus the base HF weight.
    inner = self.base_spec.state_dict_adapter(self.base_spec.model, None)
    base_hf = inner.to_hf(lora_mod.split_lora_state_dict(state)[0])
    merged_hf = adapter.to_hf_for_export(state)
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
      key = f"model.layers.1.self_attn.{proj}.weight"
      b = tensors[
          f"base_model.model.model.layers.1.self_attn.{proj}.lora_B.weight"
      ]
      a = tensors[
          f"base_model.model.model.layers.1.self_attn.{proj}.lora_A.weight"
      ]
      torch.testing.assert_close(
          merged_hf[key] - base_hf[key],
          self.options.scaling * (b @ a),
          rtol=1e-4,
          atol=1e-5,
      )
    named = lora_mod.peft_adapter_state_dict(
        state,
        hf_module_map=lambda prefix: "m." + prefix.replace(".", "_"),
        fused_layouts={},
    )
    self.assertEqual(len(named), 2 * len(lora_mod.lora_prefixes(state)))
    with tempfile.TemporaryDirectory() as tmp:
      out = lora_mod.save_peft_adapter(tmp, tensors, options=self.options)
      config = json.loads((out / "adapter_config.json").read_text())
      self.assertEqual((config["r"], config["lora_alpha"]), (4, 8.0))
      self.assertEqual(
          config["target_modules"], ["k_proj", "o_proj", "q_proj", "v_proj"]
      )
      from safetensors.torch import load_file

      self.assertEqual(
          set(load_file(str(out / "adapter_model.safetensors"))), set(tensors)
      )


class LoRAHelperTest(unittest.TestCase):
  """target_regex, merge errors and the PEFT folder, without a model."""

  PATH = r"(?:^|.*\.)(?:attention|self_attn|attn)(?:\.[^.]+)*\."

  def test_target_regex(self):
    cases = {
        ("q", "k", "v", "o"): (
            "k_proj|o_proj|q_proj|qkv_proj|v_proj|wk|wkv|wo|wq|wqkv|wv"
        ),
        ("q",): "q_proj|wq",
        ("o", "q"): "o_proj|q_proj|wo|wq",
        ("v", "k", "q"): "k_proj|q_proj|qkv_proj|v_proj|wk|wkv|wq|wqkv|wv",
        ("qkv",): "qkv_proj|wqkv",
        (): "",
    }
    for projections, names in cases.items():
      with self.subTest(projections=projections):
        self.assertEqual(
            lora_mod.target_regex(projections), self.PATH + f"(?:{names})"
        )
    self.assertEqual(
        lora_mod.target_regex(), lora_mod.target_regex(("q", "k", "v", "o"))
    )
    with self.assertRaisesRegex(ValueError, "'z'"):
      lora_mod.target_regex(("q", "z"))

  def test_merge_values_and_errors(self):
    weight = torch.arange(6.0).reshape(2, 3).to(torch.bfloat16)
    a, b = torch.full((1, 3), 0.5), torch.tensor([[1.0], [-2.0]])
    other = torch.ones(1)
    state = {
        "m.weight": weight,
        "m.lora_a.weight": a,
        "m.lora_b.weight": b,
        "other": other,
    }
    merged = lora_mod.merge_lora_state_dict(state, 2.0)
    self.assertEqual(set(merged), {"m.weight", "other"})
    self.assertIs(merged["other"], other)
    self.assertEqual(merged["m.weight"].dtype, torch.bfloat16)
    want = (weight.float() + 2.0 * (b @ a)).to(torch.bfloat16)
    self.assertTrue(torch.equal(merged["m.weight"], want))
    on_meta = {key: value.to("meta") for key, value in state.items()}
    self.assertEqual(
        lora_mod.merge_lora_state_dict(on_meta, 2.0)["m.weight"].shape, (2, 3)
    )
    broken = {
        "no lora_b": ({"m.weight": weight, "m.lora_a.weight": a}, KeyError),
        "no base weight": (
            {"m.lora_a.weight": a, "m.lora_b.weight": b},
            KeyError,
        ),
        "shape": (
            {
                "m.weight": torch.zeros(3, 3),
                "m.lora_a.weight": a,
                "m.lora_b.weight": b,
            },
            ValueError,
        ),
    }
    for label, (bad, error) in broken.items():
      with self.subTest(label), self.assertRaises(error):
        lora_mod.merge_lora_state_dict(bad, 1.0)

  def test_merge_splits_a_fused_adapter(self):
    layout = lora_mod.FusedLayout(n_heads=2, n_kv_heads=1, head_dim=1)
    a = torch.tensor([[1.0, 2.0, 3.0]])
    b = torch.tensor([[1.0], [2.0], [3.0], [4.0]])  # q, q, k, v rows
    base = {
        "blk.attn.wq.weight": torch.zeros(2, 3),
        "blk.attn.wk.weight": torch.zeros(1, 3),
        "blk.attn.wv.weight": torch.zeros(1, 3),
    }
    state = dict(
        base,
        **{"blk.attn.wqkv.lora_a.weight": a, "blk.attn.wqkv.lora_b.weight": b},
    )
    merged = lora_mod.merge_lora_state_dict(
        state, 0.5, {"blk.attn.wqkv": layout}
    )
    delta = 0.5 * (b @ a)
    self.assertEqual(set(merged), set(base))
    self.assertTrue(torch.equal(merged["blk.attn.wq.weight"], delta[:2]))
    self.assertTrue(torch.equal(merged["blk.attn.wk.weight"], delta[2:3]))
    self.assertTrue(torch.equal(merged["blk.attn.wv.weight"], delta[3:]))

  def test_peft_folder(self):
    from safetensors.torch import load_file

    shared = torch.arange(8.0).reshape(2, 4)
    prefix = "base_model.model.model.layers.0.self_attn"
    tensors = {
        f"{prefix}.q_proj.lora_A.weight": shared,
        f"{prefix}.q_proj.lora_B.weight": torch.ones(4, 2),
        f"{prefix}.k_proj.lora_A.weight": shared,
        f"{prefix}.k_proj.lora_B.weight": torch.ones(2, 4).t(),
    }
    options = lora_mod.LoRAOptions(rank=2, alpha=4.0)
    with tempfile.TemporaryDirectory() as tmpdir:
      nested = Path(tmpdir) / "runs" / "lora_adapter" / "step-3"
      out = lora_mod.save_peft_adapter(
          nested, tensors, options=options, base_model_name_or_path="/w"
      )
      self.assertEqual(out, nested)
      self.assertIsInstance(out, Path)
      raw = (out / "adapter_config.json").read_text()
      config = json.loads(raw)
      self.assertEqual(raw, json.dumps(config, indent=2))  # no newline at end
      self.assertEqual(
          list(config),
          [
              "peft_type",
              "task_type",
              "r",
              "lora_alpha",
              "lora_dropout",
              "bias",
              "target_modules",
              "base_model_name_or_path",
              "fan_in_fan_out",
              "inference_mode",
          ],
      )
      self.assertEqual(
          config,
          {
              "peft_type": "LORA",
              "task_type": "CAUSAL_LM",
              "r": 2,
              "lora_alpha": 4.0,
              "lora_dropout": 0.0,
              "bias": "none",
              "target_modules": ["k_proj", "q_proj"],
              "base_model_name_or_path": "/w",
              "fan_in_fan_out": False,
              "inference_mode": True,
          },
      )
      saved = load_file(str(out / "adapter_model.safetensors"))
      self.assertEqual(set(saved), set(tensors))
      for key, value in tensors.items():
        self.assertTrue(torch.equal(saved[key], value), key)
      again = lora_mod.save_peft_adapter(str(nested), tensors, options=options)
      self.assertEqual(again, nested)  # an existing folder is fine
      config = json.loads((again / "adapter_config.json").read_text())
      self.assertEqual(config["base_model_name_or_path"], "")


class LowBitOptimizerTest(unittest.TestCase):

  def _container(self, model, name, lr=1e-3):
    config = adamw_config(lr=lr, name=name, weight_decay=0.0)
    return config.build(model_parts=[model])

  def test_adamw8bit_step_close_to_adamw(self):
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(256, 512), torch.nn.Linear(512, 8)
    )
    ref = copy.deepcopy(model)
    start = model[0].weight.detach().clone()
    container = self._container(model, "AdamW8bit")
    self.assertIsInstance(container, PostTrainingOptimizersContainer)
    ref_opt = torch.optim.AdamW(ref.parameters(), lr=1e-3, weight_decay=0.0)
    x = torch.randn(4, 256)
    for _ in range(3):
      for m, opt in ((model, container), (ref, ref_opt)):
        opt.zero_grad()
        m(x).pow(2).mean().backward()
        opt.step()
    state = container.optimizers[0].state[model[0].weight]
    self.assertEqual(type(state["exp_avg"]).__name__, "OptimState8bit")
    self.assertEqual(state["exp_avg"].codes.dtype, torch.uint8)
    self.assertTrue(torch.isfinite(model[0].weight).all())
    # Same direction and size as full-precision AdamW from the same start.
    delta_8 = (model[0].weight - start).detach().flatten()
    delta_ref = (ref[0].weight - start).detach().flatten()
    cosine = torch.nn.functional.cosine_similarity(delta_8, delta_ref, dim=0)
    self.assertGreater(float(cosine), 0.98)
    rel = (delta_8 - delta_ref).abs().mean() / delta_ref.abs().mean()
    self.assertLess(float(rel), 0.1)

  def test_bf16_stochastic_rounding_and_scheduler(self):
    torch.manual_seed(0)
    model = torch.nn.Linear(128, 64).to(torch.bfloat16)
    container = self._container(model, "AdamWBf16", lr=1e-2)
    scheduler = LRSchedulersContainer.Config(
        warmup_steps=2, decay_type="linear"
    ).build(optimizers=container, training_steps=10)
    before = model.weight.detach().clone()
    lrs = []
    for _ in range(3):
      container.zero_grad()
      model(torch.randn(8, 128, dtype=torch.bfloat16)).float().pow(
          2
      ).mean().backward()
      container.step()
      scheduler.step()
      lrs.append(float(container.optimizers[0].param_groups[0]["lr"]))
    self.assertEqual(model.weight.dtype, torch.bfloat16)
    self.assertFalse(torch.equal(before, model.weight))
    self.assertGreater(
        len(set(lrs)), 1, lrs
    )  # the scheduler moved the tensor lr

  def test_state_dict_roundtrip(self):
    torch.manual_seed(0)
    model = torch.nn.Linear(256, 64)
    container = self._container(model, "AdamW8bit")
    model(torch.randn(2, 256)).sum().backward()
    container.step()
    flat = container.state_dict()
    self.assertTrue(flat)
    twin = copy.deepcopy(model)
    other = self._container(twin, "AdamW8bit")
    other.load_state_dict(flat)
    a = container.optimizers[0].state[model.weight]["exp_avg"]
    b = other.optimizers[0].state[twin.weight]["exp_avg"]
    torch.testing.assert_close(
        a.dequantize() if hasattr(a, "dequantize") else a,
        b.dequantize() if hasattr(b, "dequantize") else b,
    )

  def test_names(self):
    with self.assertRaises(ValueError):
      adamw_config(lr=1e-3, name="SGD")
    plain = adamw_config(lr=1e-3)
    self.assertNotIsInstance(plain, PostTrainingOptimizersContainer.Config)


class LowBitFactoryTest(unittest.TestCase):
  """The optimizer builders that ``lowbit_factory`` returns (torchao)."""

  def test_torch_switches_are_dropped(self):
    from torchtitan.experiments.tpu.post_training import optimizers

    build = optimizers.lowbit_factory("AdamW8bit")
    param = torch.nn.Parameter(torch.zeros(64, 80))
    optimizer = build([param], lr=1e-3, fused=None, foreach=False)
    self.assertEqual(type(optimizer).__name__, "AdamW8bitStep")
    self.assertFalse(optimizer.bf16_stochastic_round)

  def test_bf16_rounding_is_on_by_default(self):
    from torchtitan.experiments.tpu.post_training import optimizers

    build = optimizers.lowbit_factory("AdamWBf16")
    param = torch.nn.Parameter(torch.zeros(16, dtype=torch.bfloat16))
    self.assertTrue(build([param], lr=1e-3).bf16_stochastic_round)
    plain = build([param], lr=1e-3, bf16_stochastic_round=False, foreach=True)
    self.assertFalse(plain.bf16_stochastic_round)
    self.assertEqual(type(plain).__name__, "_AdamWStep")

  def test_unknown_name_or_backend(self):
    from torchtitan.experiments.tpu.post_training import optimizers

    with self.assertRaises(ValueError):
      optimizers.lowbit_factory("AdamW8bit", backend="xla")
    with self.assertRaises(NotImplementedError):
      optimizers.lowbit_factory("Lion")


if __name__ == "__main__":
  unittest.main()
