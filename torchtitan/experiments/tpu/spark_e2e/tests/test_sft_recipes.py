# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU checks for the end-to-end LoRA SFT recipes, the PEFT export and the tools. ASCII only.

The recipe tests build trainer configs only (no model is instantiated); the
TPU stages need TorchTPU or its import-only CPU stand-in on PYTHONPATH
(``tests/tpu_import_stubs``, as ``run_cpu.sh`` sets it) and are skipped
otherwise.
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

SPARK_E2E = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]

try:
    import torch_tpu  # noqa: F401  (the real package or the CPU import stub)

    HAVE_TPU_IMPORTS = True
except ImportError:  # pragma: no cover
    HAVE_TPU_IMPORTS = False


def _clean_env(**values: str):
    """Patch os.environ: no SPARK_* knob except the given ones."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("SPARK_")}
    env.update(values)
    return mock.patch.dict(os.environ, env, clear=True)


class NoNativeRLTest(unittest.TestCase):
    def test_no_native_rl_or_private_cross_package_imports(self):
        files = sorted(SPARK_E2E.rglob("*.py")) + [SPARK_E2E.parent / "muse_spark" / "spark_64chip_recipes.py"]
        for path in files:
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
                for name in names:
                    self.assertNotIn("native_rl", name, f"{path}: {name}")
                    self.assertFalse(name.startswith("torchtitan.experiments.tpu.qwen3_5"), f"{path}: {name}")

    @unittest.skipUnless(HAVE_TPU_IMPORTS, "needs torch_tpu or the CPU import stub on PYTHONPATH")
    def test_entry_point_imports_without_native_rl(self):
        code = ("import sys; import torchtitan.experiments.tpu.spark_e2e.train, "
                "torchtitan.experiments.tpu.spark_e2e.recipes; "
                "bad = [m for m in sys.modules if 'native_rl' in m]; assert not bad, bad; print('ok')")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600,
                                env=dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0"))
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])


@unittest.skipUnless(HAVE_TPU_IMPORTS, "needs torch_tpu or the CPU import stub on PYTHONPATH")
class RecipeTest(unittest.TestCase):
    def _recipe(self, name: str, **env: str):
        from torchtitan.experiments.tpu.spark_e2e import recipes

        with _clean_env(**env):
            return getattr(recipes, name)()

    def _lora_modules(self, cfg):
        from torchtitan.models.common.linear import Linear

        return [(path, config.rank, config.alpha)
                for path, config, _, _ in cfg.model_spec.model.traverse(Linear.Config)
                if hasattr(config, "rank") and hasattr(config, "alpha")]

    def test_cpu_stage_defaults_and_knobs(self):
        from torchtitan.experiments.tpu.post_training.sft_config import SFTTrainerConfig
        from torchtitan.experiments.tpu.spark_e2e.lora_publish import LoRATolerantMuseSparkAdapter

        cfg = self._recipe("s0_sft_tiny")
        self.assertIsInstance(cfg, SFTTrainerConfig)
        self.assertTrue(issubclass(cfg.model_spec.state_dict_adapter, LoRATolerantMuseSparkAdapter))
        self.assertEqual(cfg.sft.lora_save, "full")  # SparkSFTTrainer installs the adapter-only checkpoints
        self.assertEqual((cfg.sft.val_data_path, cfg.sft.val_every, cfg.sft.val_max_microbatches), ("", 10, 0))
        self.assertEqual(cfg.dataloader.reasoning_effort, "low")
        self.assertEqual(cfg.dataloader.data_path, "/mnt/data/sft/train.jsonl")
        self.assertEqual(cfg.hf_assets_path, "/mnt/model")
        self.assertEqual(cfg.training.steps, 4)
        self.assertEqual(cfg.training.num_tokens_per_microbatch_per_dp_rank, 4096)
        self.assertEqual(cfg.training.dtype, "float32")
        self.assertIsNone(cfg.activation_checkpoint)
        self.assertTrue(cfg.checkpoint.initial_load_in_hf)
        cfg = self._recipe("s0_sft_tiny", SPARK_STEPS="7", SPARK_LR="0.5", SPARK_SFT_SEQ="2048",
                           SPARK_SFT_MICROBATCHES="2", SPARK_AC="1", SPARK_REASONING="high",
                           SPARK_VAL_DATA="/d/val.jsonl", SPARK_VAL_EVERY="3", SPARK_VAL_BATCHES="5")
        self.assertEqual(cfg.training.steps, 7)
        self.assertEqual(cfg.optimizer.param_groups[0].optimizer_kwargs["lr"], 0.5)
        self.assertEqual(cfg.training.num_tokens_per_microbatch_per_dp_rank, 2048)
        self.assertEqual(cfg.dataloader.microbatches_per_step, 2)
        self.assertIsNotNone(cfg.activation_checkpoint)
        self.assertEqual(cfg.dataloader.reasoning_effort, "high")
        self.assertEqual((cfg.sft.val_data_path, cfg.sft.val_every, cfg.sft.val_max_microbatches),
                         ("/d/val.jsonl", 3, 5))

    def test_lora_rank_and_alpha(self):
        cfg = self._recipe("s0_sft_tiny", SPARK_LORA_RANK="8", SPARK_LORA_ALPHA="24")
        mods = self._lora_modules(cfg)
        self.assertTrue(mods)
        self.assertTrue(all(rank == 8 and alpha == 24.0 for _, rank, alpha in mods), mods[:3])
        targets = {path.rsplit(".", 1)[-1] for path, _, _ in mods}
        self.assertTrue(targets <= {"wq", "wk", "wv", "wkv", "wo"}, targets)

    def _check_tpu(self, cfg, *, layers: int, steps: int, rows: int = 1):
        from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts
        from torchtitan.experiments.tpu.muse_spark import tpu7x
        from torchtitan.experiments.tpu.token_dispatcher import TpuAllToAllTokenDispatcher

        self.assertEqual(len(cfg.model_spec.model.layers), layers)
        self.assertEqual(cfg.training.steps, steps)
        self.assertEqual(cfg.training.num_tokens_per_microbatch_per_dp_rank, rows * 8192)
        self.assertEqual(cfg.training.dtype, "bfloat16")
        self.assertEqual(cfg.lr_scheduler.warmup_steps, 2)
        self.assertEqual(cfg.parallelism.expert_parallel_degree, 1)
        self.assertEqual(cfg.parallelism.fsdp_reshard_after_forward, "always")
        self.assertTrue(cfg.parallelism.enable_tpu_topo_mesh)
        self.assertEqual((cfg.comm.init_timeout_seconds, cfg.comm.train_timeout_seconds), (1800, 1800))
        self.assertEqual(cfg.tpu_config.eager_mode, "DEFER_AND_FUSE")
        self.assertFalse(cfg.tpu_config.enable_amp)
        self.assertTrue(cfg.compile.compile_grad_norm)
        self.assertEqual(cfg.loss.num_chunks, 16)
        splash = cfg.splash_attention_kernel
        for key, value in tpu7x.SPLASH_ATTENTION.items():
            self.assertEqual(getattr(splash, key), value, key)
        experts = [e for _, e, _, _ in cfg.model_spec.model.traverse(TpuGroupedExperts.Config)]
        self.assertEqual(len(experts), layers)
        for e in experts:
            self.assertTrue(e.use_tokamax_gmm_v2)
            self.assertFalse(e.use_pallas_paired_grouped_mm)
        dispatchers = [d for _, d, _, _ in cfg.model_spec.model.traverse(TpuAllToAllTokenDispatcher.Config)]
        self.assertTrue(dispatchers and all(d.fuse_route_score_backward for d in dispatchers))
        self.assertEqual(cfg.libtpu_init_flags.get("xla_tpu_dvfs_p_state"), "7")

    def test_tpu_stages(self):
        from torchtitan.distributed.activation_checkpoint import FullAC

        cfg = self._recipe("s1_sft_trunc2")
        self._check_tpu(cfg, layers=2, steps=5)
        self.assertIsInstance(cfg.activation_checkpoint, FullAC.Config)
        self.assertFalse(cfg.compile.enable)
        cfg = self._recipe("s2_sft_full", SPARK_STEPS="60", SPARK_SFT_ROWS="2", SPARK_VAL_DATA="/mnt/data/sft/val.jsonl")
        self._check_tpu(cfg, layers=62, steps=60, rows=2)
        self.assertEqual(cfg.sft.val_data_path, "/mnt/data/sft/val.jsonl")
        cfg = self._recipe("s2_sft_full", SPARK_AC="0", SPARK_COMPILE="1")
        self.assertIsNone(cfg.activation_checkpoint)
        self.assertTrue(cfg.compile.enable)
        self._check_tpu(cfg, layers=62, steps=20)


def _spark_lora_model():
    """A stand-in with the port's attention module paths and TorchTitan LoRA parameter names."""

    class LoRALinear(torch.nn.Module):
        def __init__(self, out_features: int, in_features: int, rank: int, alpha: float):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
            self.lora_a = torch.nn.Linear(in_features, rank, bias=False)
            self.lora_b = torch.nn.Linear(rank, out_features, bias=False)
            torch.nn.init.normal_(self.lora_b.weight)
            self._lora_scaling = alpha / rank

    class QKV(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.wq, self.wk, self.wv = LoRALinear(8, 6, 2, 4.0), LoRALinear(4, 6, 2, 4.0), LoRALinear(4, 6, 2, 4.0)

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv_linear = QKV()
            self.wo = LoRALinear(6, 8, 2, 4.0)

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = Attention()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleDict({"0": Layer(), "1": Layer()})

    return Model()


class PeftAdapterTest(unittest.TestCase):
    def test_export_layout_and_round_trip(self):
        from safetensors.torch import load_file
        from torchtitan.experiments.tpu.spark_e2e import peft_adapter

        model = _spark_lora_model()
        spec = peft_adapter.adapter_spec_from_model(model)
        self.assertEqual((spec.rank, spec.alpha, spec.scaling), (2, 4.0, 2.0))
        tensors = peft_adapter.gather_peft_adapter([model], keep=True)
        self.assertEqual(len(tensors), 16)
        self.assertIn("base_model.model.model.language_model.layers.1.self_attn.k_proj.lora_B.weight", tensors)
        self.assertTrue(all(t.dtype == torch.float32 for t in tensors.values()))
        self.assertEqual(peft_adapter.gather_peft_adapter([model], keep=False), {})
        tmp = Path(tempfile.mkdtemp(prefix="spark-peft-"))
        try:
            out = peft_adapter.save_peft_dir(tmp / "adapter", tensors, spec, base="/mnt/model", extra={"step": 3})
            config = json.loads((out / "adapter_config.json").read_text())
            self.assertEqual((config["r"], config["lora_alpha"], config["peft_type"]), (2, 4.0, "LORA"))
            import re

            pattern = re.compile(config["target_modules"])
            self.assertTrue(pattern.fullmatch("model.language_model.layers.0.self_attn.q_proj"))
            self.assertFalse(pattern.fullmatch("model.vision_tower.layers.0.self_attn.q_proj"))
            self.assertEqual(json.loads((out / "export_info.json").read_text()), {"step": 3})
            self.assertEqual(sorted(load_file(str(out / "adapter_model.safetensors"))), sorted(tensors))
            fresh = _spark_lora_model()
            info = peft_adapter.load_peft_into_model(fresh, out)
            self.assertEqual(info["tensors"], 16)
            for (name, a), (_, b) in zip(model.named_parameters(), fresh.named_parameters()):
                if "lora_" in name:
                    self.assertTrue(torch.equal(a, b), name)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_rejects_unmapped_or_unpaired_adapters(self):
        from torchtitan.experiments.tpu.spark_e2e import peft_adapter

        mapping = peft_adapter.spark_module_map()
        self.assertEqual(mapping("layers.12.attention.wo"), "model.language_model.layers.12.self_attn.o_proj")
        with self.assertRaises(peft_adapter.AdapterError):
            mapping("layers.0.feed_forward.w1")
        with self.assertRaises(peft_adapter.AdapterError):
            peft_adapter.lora_module_keys(["layers.0.attention.wo.lora_a.weight"])
        self.assertEqual(peft_adapter.canonical_fqn("layers.0._checkpoint_wrapped_module.attention.wo.lora_a.weight"),
                         "layers.0.attention.wo.lora_a.weight")

    @unittest.skipUnless(HAVE_TPU_IMPORTS, "needs torch_tpu or the CPU import stub on PYTHONPATH")
    def test_module_map_agrees_with_the_checkpoint_mapping(self):
        """The export's HF module names equal what the model's state-dict adapter maps the base weights to."""
        from torchtitan.experiments.tpu.spark_e2e import peft_adapter
        from torchtitan.experiments.tpu.spark_e2e.recipes import _lora, spark_model_spec

        with _clean_env():
            spec = spark_model_spec("tiny-hf", 256, lora=_lora(), tpu_kernels=False)
        adapter = spec.state_dict_adapter(spec.model, None)
        mapping = peft_adapter.spark_module_map()
        with torch.device("meta"):
            model = spec.model.build()
        keys = [k for k in model.state_dict() if k.endswith(".lora_a.weight")]
        self.assertTrue(keys)
        for key in keys:
            module = key[: -len(".lora_a.weight")]
            weight = dict(model.named_parameters())[f"{module}.weight"]
            hf = adapter.to_hf({f"{module}.weight": torch.empty(weight.shape, device="meta")})
            self.assertEqual(list(hf), [mapping(module) + ".weight"], module)


class FastReaderTest(unittest.TestCase):
    def test_reads_contiguous_and_strided_slices(self):
        import torch.distributed.checkpoint as dcp
        from safetensors.torch import save_file
        from torchtitan.experiments.tpu.spark_e2e.tpu_knobs import FastHFStorageReader

        tmp = Path(tempfile.mkdtemp(prefix="spark-reader-"))
        try:
            weights = {"a": torch.arange(24, dtype=torch.float32).reshape(4, 6).to(torch.bfloat16),
                       "b": torch.arange(10, dtype=torch.float32)}
            save_file(weights, str(tmp / "model-00001-of-00001.safetensors"))
            reader = FastHFStorageReader(str(tmp), thread_count=2)
            loaded = {k: torch.empty_like(v) for k, v in weights.items()}
            dcp.load(loaded, storage_reader=reader, no_dist=True)
            for key in weights:
                self.assertTrue(torch.equal(loaded[key], weights[key]), key)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class CompatTest(unittest.TestCase):
    def test_generation_blocks_render(self):
        from torchtitan.components.tokenizer import HuggingFaceTokenizer
        from torchtitan.experiments.tpu.spark_e2e import compat

        compat.install()
        tokenizer = HuggingFaceTokenizer.__new__(HuggingFaceTokenizer)
        tokenizer.set_chat_template(
            "{% for m in messages %}{% if m['role'] == 'assistant' %}{% generation %}"
            "{{ m['content'] }}{% endgeneration %}{% else %}{{ m['content'] }}|{% endif %}{% endfor %}")
        text = tokenizer.apply_chat_template([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}])
        self.assertEqual(text, "q|a")


def _write_checkpoint(root: Path, layers: int) -> dict[str, torch.Tensor]:
    from safetensors.torch import save_file

    tensors = {"model.language_model.embed_tokens.weight": torch.randn(10, 4),
               "model.language_model.norm.weight": torch.randn(4), "lm_head.weight": torch.randn(10, 4),
               "model.vision_tower.proj.weight": torch.randn(3, 3)}
    for i in range(layers):
        tensors[f"model.language_model.layers.{i}.self_attn.q_proj.weight"] = torch.randn(4, 4)
        tensors[f"model.language_model.layers.{i}.mlp.experts.down_proj"] = torch.randn(2, 4, 3).to(torch.bfloat16)
    names = sorted(tensors)
    shards = {"model-00001-of-00002.safetensors": names[::2], "model-00002-of-00002.safetensors": names[1::2]}
    weight_map = {}
    for shard, keys in shards.items():
        save_file({k: tensors[k] for k in keys}, str(root / shard))
        weight_map.update({k: shard for k in keys})
    (root / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    (root / "config.json").write_text(json.dumps({"model_type": "muse_spark", "text_config": {
        "num_hidden_layers": layers, "layer_types": ["sliding_attention"] * layers,
        "mlp_layer_types": ["moe"] * layers, "layer_rope_theta": [1.0] * layers}}))
    (root / "tokenizer.json").write_text("{}")
    return tensors


class BuildTruncTest(unittest.TestCase):
    def test_first_layers_only_and_byte_identical(self):
        from safetensors.torch import load_file
        from torchtitan.experiments.tpu.spark_e2e import build_trunc

        tmp = Path(tempfile.mkdtemp(prefix="spark-trunc-"))
        try:
            src = tmp / "src"
            src.mkdir()
            tensors = _write_checkpoint(src, layers=3)
            report = build_trunc.build(src, tmp / "out", layers=2, threads=2)
            out = load_file(str(tmp / "out" / "model.safetensors"))
            expected = {k for k in tensors if "vision" not in k and ".layers.2." not in k}
            self.assertEqual(set(out), expected)
            for key in expected:
                self.assertTrue(torch.equal(out[key], tensors[key]), key)
            config = json.loads((tmp / "out" / "config.json").read_text())["text_config"]
            self.assertEqual(config["num_hidden_layers"], 2)
            self.assertEqual(len(config["layer_types"]), 2)
            index = json.loads((tmp / "out" / "model.safetensors.index.json").read_text())
            self.assertEqual(set(index["weight_map"]), expected)
            self.assertTrue((tmp / "out" / "tokenizer.json").is_file())
            self.assertEqual(report["tensors"], len(expected))
            with open(tmp / "out" / "model.safetensors", "rb") as handle:
                (header_len,) = struct.unpack("<Q", handle.read(8))
            self.assertEqual((8 + header_len) % 8, 0)
            with self.assertRaises(ValueError):
                build_trunc.build(src, tmp / "too-many", layers=4, threads=1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
