# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the layout-preserving Hugging Face export.

Model-agnostic (always run):

* ``SyntheticExportTest``: a sharded checkpoint with mixed dtypes and
  non-weight files: no-op export, multi-worker export and finalize, checks on
  replacement tensors, PEFT merges, the command line and the multi-process
  ``hf_export_check``.
* ``Llama3DcpExportTest``: DCP checkpoints of TorchTitan's Llama 3 debugmodel
  exported through its state-dict adapter (full parameter, folded LoRA,
  LoRA -> PEFT adapter -> merged export).

``HfExportTest`` needs a local copy of the public tiny test checkpoint
``trl-internal-testing/tiny-MuseSparkForConditionalGeneration`` in
``MUSE_SPARK_TINY_DIR`` (tests skip otherwise). Tests that use the TorchTitan
Muse Spark port or the PEFT library skip when those are not importable.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest

import torch

from torchtitan.experiments.tpu.checkpointing import hf_export
from torchtitan.experiments.tpu.checkpointing.hf_export import (
    DcpSource,
    HfLayout,
    PeftSource,
    StateDictSource,
    assign_files,
    export_hf,
    finalize_export,
)
from torchtitan.experiments.tpu.checkpointing.lora_checkpoint import write_peft_dir
from torchtitan.experiments.tpu.checkpointing.tests import fixtures

TINY_DIR = os.environ.get("MUSE_SPARK_TINY_DIR")


def _sha(path: str) -> str:
    """Hex SHA-256 of the file at ``path`` (what ``sha256sum`` prints)."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise equality (also for float8, which has no ``torch.equal``)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


# ---------------------------------------------------------------------------
# Synthetic checkpoint
# ---------------------------------------------------------------------------

EXTRA_FILES = {"config.json": "{}\n", "tokenizer.json": "{}\n", "chat_template.jinja": "{{ x }}\n",
               "README.md": "readme\n"}
SKIPPED_FILES = ("SHA256SUMS", ".hidden", "_staging.json", "old.partial")


def _synthetic_tensors(seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=g).to(dtype)

    tensors = {
        "model.embed_tokens.weight": rnd(64, 16),
        "model.norm.weight": rnd(16, dtype=torch.float16),
        "model.vision_tower.patch.weight": rnd(8, 3),
        "lm_head.weight": rnd(64, 16),
    }
    for i in range(2):
        p = f"model.layers.{i}."
        tensors.update({
            p + "self_attn.q_proj.weight": rnd(16, 16),
            p + "self_attn.o_proj.weight": rnd(16, 16),
            p + "mlp.gate.weight": rnd(4, 16, dtype=torch.float32),
            p + "mlp.experts.gate_up_proj": rnd(4, 32, 16),
            p + "mlp.expert_counts": torch.randint(0, 100, (4,), generator=g),
            p + "mlp.scales": torch.randint(0, 255, (16, 2), generator=g).to(torch.uint8),
            p + "mlp.fp8_weight": rnd(16, 8, dtype=torch.float32).to(torch.float8_e4m3fn),
        })
    return tensors


def _write_synthetic(directory: str, num_files: int = 3, seed: int = 0) -> dict[str, torch.Tensor]:
    """Sharded checkpoint + index + small files + files that are not part of a release."""
    from safetensors.torch import save_file

    tensors = _synthetic_tensors(seed)
    os.makedirs(directory)
    keys = sorted(tensors)
    weight_map = {}
    for i in range(num_files):
        name = f"model-{i + 1:05d}-of-{num_files:05d}.safetensors"
        part = {k: tensors[k] for j, k in enumerate(keys) if j % num_files == i}
        save_file(part, os.path.join(directory, name), metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(part, name))
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    with open(os.path.join(directory, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)
    for name, text in EXTRA_FILES.items():
        with open(os.path.join(directory, name), "w") as f:
            f.write(text)
    for name in SKIPPED_FILES:
        with open(os.path.join(directory, name), "w") as f:
            f.write("x\n")
    save_file({"stray": torch.zeros(2)}, os.path.join(directory, "stray.safetensors"))  # not in the index
    os.makedirs(os.path.join(directory, "original"))
    return tensors


def _peft(targets: list[str], shapes: dict[str, list[int]], rank: int, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    out = {}
    for key in targets:
        module = key[: -len(".weight")]
        rows, cols = shapes[key]
        out[f"base_model.model.{module}.lora_A.weight"] = (0.1 * torch.randn(rank, cols, generator=g)).to(torch.bfloat16)
        out[f"base_model.model.{module}.lora_B.weight"] = (0.1 * torch.randn(rows, rank, generator=g)).to(torch.bfloat16)
    return out


def _merged(w: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scale: float) -> torch.Tensor:
    return (w.float() + scale * (b.float() @ a.float())).to(w.dtype)


QO_TARGETS = [f"model.layers.{i}.self_attn.{p}.weight" for i in range(2) for p in ("q_proj", "o_proj")]


class SyntheticExportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hf-export-synthetic-")
        self.base = os.path.join(self.tmp, "base")
        self.tensors = _write_synthetic(self.base)
        self.layout = HfLayout(self.base)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_noop_export_is_byte_identical(self):
        out = os.path.join(self.tmp, "out")
        rep = export_hf(self.base, out, StateDictSource({}))
        fin = rep["finalize"]
        self.assertEqual((fin["num_shards"], fin["replaced_tensors"], fin["copied_tensors"]),
                         (3, 0, len(self.tensors)))
        for f in self.layout.files:
            self.assertEqual(_sha(os.path.join(self.base, f)), _sha(os.path.join(out, f)), f)
        with open(os.path.join(out, "model.safetensors.index.json")) as fo, \
                open(os.path.join(self.base, "model.safetensors.index.json")) as fb:
            self.assertEqual(json.load(fo), json.load(fb))
        for name in EXTRA_FILES:
            self.assertEqual(_sha(os.path.join(self.base, name)), _sha(os.path.join(out, name)), name)
        expected = set(self.layout.files) | set(EXTRA_FILES) | {
            "model.safetensors.index.json", "EXPORT_OK", "export_manifest.json",
            "export_manifest.worker00000-of-00001.json"}
        self.assertEqual(set(os.listdir(out)), expected)
        with open(os.path.join(out, "export_manifest.json")) as f:
            manifest = json.load(f)
        for f in self.layout.files:
            self.assertEqual(manifest["shards"][f]["sha256"], _sha(os.path.join(self.base, f)))
        self.assertEqual(sorted(manifest["extra_files"]), sorted(EXTRA_FILES))

    def test_single_file_checkpoint(self):
        from safetensors.torch import save_file

        single = os.path.join(self.tmp, "single")
        os.makedirs(single)
        part = {k: v for k, v in self.tensors.items() if "layers.0." in k}
        save_file(part, os.path.join(single, "model.safetensors"))
        layout = HfLayout(single)
        self.assertIsNone(layout.index)
        self.assertEqual(layout.files, ["model.safetensors"])
        name = "model.layers.0.self_attn.q_proj.weight"
        new = torch.randn(16, 16).to(torch.bfloat16)
        out = os.path.join(self.tmp, "out")
        self.assertEqual(export_hf(single, out, StateDictSource({name: new}))["finalize"]["replaced_tensors"], 1)
        self.assertFalse(os.path.exists(os.path.join(out, "model.safetensors.index.json")))
        self.assertTrue(_same(HfLayout(out).read_tensor(name), new))

    def test_multi_worker_export_and_finalize(self):
        assignment = assign_files(self.layout, 3)
        self.assertEqual(sorted(f for files in assignment for f in files), self.layout.files)
        trained = {
            "model.layers.1.self_attn.q_proj.weight": torch.randn(16, 16).to(torch.bfloat16),
            "model.layers.0.mlp.gate.weight": torch.randn(4, 16),  # F32 router
        }
        out = os.path.join(self.tmp, "out")
        for w in (0, 1):
            export_hf(self.base, out, StateDictSource(trained), worker=w, num_workers=3)
        with self.assertRaises(FileNotFoundError):  # worker 2 has not finished
            finalize_export(self.base, out, num_workers=3)
        export_hf(self.base, out, StateDictSource(trained), worker=2, num_workers=3)
        self.assertFalse(os.path.exists(os.path.join(out, "EXPORT_OK")))
        with self.assertRaises(ValueError):  # replaced count != what the source provides
            finalize_export(self.base, out, num_workers=3, source_keys=set(trained) | {"lm_head.weight"})
        fin = finalize_export(self.base, out, num_workers=3, source_keys=set(trained))
        self.assertEqual(fin["replaced_tensors"], 2)
        self.assertTrue(os.path.exists(os.path.join(out, "EXPORT_OK")))
        touched = {self.layout.weight_map[k] for k in trained}
        for f in self.layout.files:
            same = _sha(os.path.join(self.base, f)) == _sha(os.path.join(out, f))
            self.assertEqual(same, f not in touched, f)
        exported = HfLayout(out)
        for key, value in self.tensors.items():
            self.assertTrue(_same(exported.read_tensor(key), trained.get(key, value)), key)

    def test_replacement_checks(self):
        q = "model.layers.0.self_attn.q_proj.weight"
        with self.assertRaises(KeyError):
            export_hf(self.base, os.path.join(self.tmp, "o1"), StateDictSource({"model.missing.weight": torch.zeros(1)}))
        with self.assertRaises(ValueError):  # shape
            export_hf(self.base, os.path.join(self.tmp, "o2"), StateDictSource({q: torch.zeros(8, 16)}))
        with self.assertRaises(ValueError):  # integer tensors keep their dtype
            export_hf(self.base, os.path.join(self.tmp, "o3"), StateDictSource(
                {"model.layers.0.mlp.expert_counts": torch.zeros(4, dtype=torch.int32)}))
        fp32 = torch.randn(16, 16)
        out = os.path.join(self.tmp, "o4")
        export_hf(self.base, out, StateDictSource({q: fp32}))
        self.assertTrue(_same(HfLayout(out).read_tensor(q), fp32.to(torch.bfloat16)))  # cast to the original dtype

    def test_peft_merge(self):
        rank, alpha = 2, 4.0
        shapes = {k: self.layout.tensor_info(k)[1] for k in QO_TARGETS}
        peft = _peft(QO_TARGETS, shapes, rank)
        adapter_dir = os.path.join(self.tmp, "adapter")
        write_peft_dir(peft, adapter_dir, rank=rank, alpha=alpha)
        source = PeftSource(adapter_dir, self.layout)
        self.assertEqual(source.keys(), set(QO_TARGETS))
        self.assertEqual(source.scaling, alpha / rank)
        out = os.path.join(self.tmp, "out")
        self.assertEqual(export_hf(self.base, out, source)["finalize"]["replaced_tensors"], 4)
        exported = HfLayout(out)
        for key in QO_TARGETS:
            module = f"base_model.model.{key[: -len('.weight')]}"
            want = _merged(self.tensors[key], peft[f"{module}.lora_A.weight"], peft[f"{module}.lora_B.weight"],
                           alpha / rank)
            self.assertTrue(_same(exported.read_tensor(key), want), key)
        config_path = os.path.join(adapter_dir, "adapter_config.json")
        with open(config_path) as f:
            config = json.load(f)
        config["use_rslora"] = True
        with open(config_path, "w") as f:
            json.dump(config, f)
        self.assertAlmostEqual(PeftSource(adapter_dir).scaling, alpha / rank ** 0.5)

    def test_peft_source_checks(self):
        from safetensors.torch import save_file

        shapes = {k: self.layout.tensor_info(k)[1] for k in QO_TARGETS}
        peft = _peft(QO_TARGETS[:1], shapes, 2)
        cases = {
            "absent": (KeyError, {k.replace("layers.0", "layers.7"): v for k, v in peft.items()}),
            "no_lora_B": (ValueError, {k: v for k, v in peft.items() if "lora_A" in k}),
            "foreign": (ValueError, {**peft, "base_model.model.x.scale": torch.zeros(1)}),
        }
        for name, (error, tensors) in cases.items():
            adapter_dir = os.path.join(self.tmp, name)
            os.makedirs(adapter_dir)
            save_file(tensors, os.path.join(adapter_dir, "adapter_model.safetensors"))
            with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
                json.dump({"peft_type": "LORA", "r": 2, "lora_alpha": 4}, f)
            with self.subTest(name=name), self.assertRaises(error):
                PeftSource(adapter_dir, self.layout)
        with open(os.path.join(self.tmp, "absent", "adapter_config.json"), "w") as f:
            json.dump({"peft_type": "IA3", "r": 2}, f)
        with self.assertRaises(ValueError):
            PeftSource(os.path.join(self.tmp, "absent"))

    def test_command_line(self):
        rank, alpha = 2, 4.0
        target = QO_TARGETS[0]
        peft = _peft([target], {target: [16, 16]}, rank)
        adapter_dir = os.path.join(self.tmp, "adapter")
        write_peft_dir(peft, adapter_dir, rank=rank, alpha=alpha)
        out = os.path.join(self.tmp, "out")
        for worker in (0, 1):
            self.assertEqual(hf_export.main(["--base", self.base, "--out", out, "--peft", adapter_dir,
                                             "--worker", str(worker), "--num-workers", "2"]), 0)
        self.assertFalse(os.path.exists(os.path.join(out, "EXPORT_OK")))
        self.assertEqual(hf_export.main(["--base", self.base, "--out", out, "--num-workers", "2", "--finalize"]), 0)
        self.assertTrue(os.path.exists(os.path.join(out, "EXPORT_OK")))
        module = f"base_model.model.{target[: -len('.weight')]}"
        want = _merged(self.tensors[target], peft[f"{module}.lora_A.weight"], peft[f"{module}.lora_B.weight"],
                       alpha / rank)
        self.assertTrue(_same(HfLayout(out).read_tensor(target), want))
        for argv in (["--base", self.base, "--out", out],  # no source
                     ["--base", self.base, "--out", out, "--dcp", self.base]):  # --dcp needs --adapter-factory
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                hf_export.main(argv)

    def test_hf_export_check(self):
        """The multi-worker check script, on one host with two spawned workers and local paths."""
        from torchtitan.experiments.tpu.checkpointing import hf_export_check

        report_path = os.path.join(self.tmp, "report.json")
        rc = hf_export_check.main([
            "--base", self.base, "--out", os.path.join(self.tmp, "check"),
            "--staging-dir", os.path.join(self.tmp, "stage"), "--hosts", "1", "--host", "0",
            "--workers-per-host", "2", "--auth", "none", "--rank", "2", "--alpha", "4",
            "--targets", r"model\.layers\.\d+\.self_attn\.(q|o)_proj\.weight", "--report", report_path,
        ])
        with open(report_path) as f:
            report = json.load(f)
        self.assertEqual(rc, 0, report)
        self.assertTrue(report["header_bytes_identical"])
        self.assertTrue(report["merged_tensors_bit_exact"])
        self.assertEqual(report["merged_tensors_checked"], 4)
        self.assertEqual(report["finalize"]["replaced_tensors"], 4)


# ---------------------------------------------------------------------------
# DCP checkpoints of a TorchTitan model (Llama 3 debugmodel)
# ---------------------------------------------------------------------------


class Llama3DcpExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.ensure_process_group()

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hf-export-llama3-")
        self.base_dir = os.path.join(self.tmp, "base")
        base, spec = fixtures.build_llama3(lora=False, seed=0)
        self.base_state = {k: v.detach().clone() for k, v in base.state_dict().items()}
        fixtures.write_hf_checkpoint(self.base_dir, self.base_state, spec, num_files=3)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _lora_model(self, targets: tuple[str, ...]):
        model, spec = fixtures.build_llama3(lora=True, seed=1, targets=targets)
        missing, unexpected = model.load_state_dict(self.base_state, strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(all(".lora_" in k for k in missing), missing)
        g = torch.Generator().manual_seed(2)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name.endswith("lora_b.weight"):
                    param.copy_(0.1 * torch.randn(param.shape, generator=g))
        return model, spec.state_dict_adapter(spec.model, self.base_dir)

    def _save(self, state: dict[str, torch.Tensor]) -> str:
        import torch.distributed.checkpoint as dcp

        path = os.path.join(self.tmp, "checkpoint", "step-10")
        dcp.save(dict(state), checkpoint_id=path)
        return path

    def test_full_param_export(self):
        model, spec = fixtures.build_llama3(lora=False, seed=1)
        model.load_state_dict(self.base_state)
        g = torch.Generator().manual_seed(4)
        with torch.no_grad():
            for param in model.parameters():
                param.add_(0.02 * torch.randn(param.shape, generator=g))  # "training"
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        adapter = spec.state_dict_adapter(spec.model, self.base_dir)
        source = DcpSource(self._save(state), adapter)
        out = os.path.join(self.tmp, "hf")
        rep = export_hf(self.base_dir, out, source)
        expected = adapter.to_hf(state)
        self.assertEqual(rep["finalize"]["replaced_tensors"], len(expected))
        exported = HfLayout(out)
        for key, value in expected.items():
            self.assertTrue(_same(exported.read_tensor(key), value), key)

    def test_lora_is_folded_into_the_export(self):
        model, adapter = self._lora_model(("wo",))
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        path = self._save(state)
        with self.assertRaises(ValueError):  # LoRA tensors need the scale
            DcpSource(path, adapter)
        scale = fixtures.LORA_ALPHA / fixtures.LORA_RANK
        out = os.path.join(self.tmp, "hf")
        export_hf(self.base_dir, out, DcpSource(path, adapter, lora_scale=scale))
        merged = {k: v for k, v in state.items() if ".lora_" not in k}
        for key in [k for k in merged if k.endswith("attention.wo.weight")]:
            prefix = key[: -len(".weight")]
            merged[key] = _merged(state[key], state[f"{prefix}.lora_a.weight"], state[f"{prefix}.lora_b.weight"], scale)
        exported = HfLayout(out)
        for key, value in adapter.to_hf(merged).items():
            self.assertTrue(_same(exported.read_tensor(key), value), key)

    def test_unfoldable_lora_is_rejected(self):
        model, adapter = self._lora_model(("wqkv",))  # saved as wq/wk/wv: no wqkv.weight to fold into
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        with self.assertRaises(NotImplementedError):
            DcpSource(self._save(state), adapter, lora_scale=2.0)
        adapter_only = {k: v for k, v in state.items() if ".lora_" in k}
        shutil.rmtree(os.path.join(self.tmp, "checkpoint"))
        with self.assertRaises(NotImplementedError):  # an adapter-only checkpoint
            DcpSource(self._save(adapter_only), adapter, lora_scale=2.0)

    def test_lora_peft_merged_export(self):
        """TorchTitan LoRA -> PEFT adapter -> merged original-layout export."""
        from torchtitan.experiments.tpu.checkpointing.lora_checkpoint import save_peft_adapter

        model, adapter = self._lora_model(("wo",))
        adapter_dir = os.path.join(self.tmp, "adapter")
        save_peft_adapter([model], adapter, adapter_dir, rank=fixtures.LORA_RANK, alpha=fixtures.LORA_ALPHA,
                          dtype=None)
        out = os.path.join(self.tmp, "merged")
        layout = HfLayout(self.base_dir)
        rep = export_hf(self.base_dir, out, PeftSource(adapter_dir, layout))
        self.assertEqual(rep["finalize"]["replaced_tensors"], 6)
        state = model.state_dict()
        scale = fixtures.LORA_ALPHA / fixtures.LORA_RANK
        merged = {k: v for k, v in state.items() if ".lora_" not in k}
        for key in [k for k in merged if k.endswith("attention.wo.weight")]:
            prefix = key[: -len(".weight")]
            merged[key] = _merged(state[key], state[f"{prefix}.lora_a.weight"], state[f"{prefix}.lora_b.weight"], scale)
        exported = HfLayout(out)
        for key, value in adapter.to_hf(merged).items():
            self.assertTrue(_same(exported.read_tensor(key), value), key)


# ---------------------------------------------------------------------------
# The public tiny Muse Spark checkpoint
# ---------------------------------------------------------------------------


def _hf_model(path: str, dtype=torch.bfloat16):
    from transformers import MuseSparkForConditionalGeneration

    model = MuseSparkForConditionalGeneration.from_pretrained(path, dtype=dtype, attn_implementation="eager")
    return model.eval()


def _logits(model, ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=ids, use_cache=False).logits.float()


def _ids(n: int = 48, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 199_000, (2, n), generator=g)


def _reshard(src_dir: str, dst_dir: str, num_files: int) -> None:
    """Re-save a single-file checkpoint as ``num_files`` shards with an index."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    os.makedirs(dst_dir)
    with safe_open(os.path.join(src_dir, "model.safetensors"), "pt") as f:
        keys = sorted(f.keys())
        tensors = {k: f.get_tensor(k) for k in keys}
    weight_map = {}
    names = [f"model-{i + 1:05d}-of-{num_files:05d}.safetensors" for i in range(num_files)]
    for i, name in enumerate(names):
        part = {k: tensors[k] for j, k in enumerate(keys) if j % num_files == i}
        save_file(part, os.path.join(dst_dir, name), metadata={"format": "pt"})
        weight_map.update({k: name for k in part})
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    with open(os.path.join(dst_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)
    for name in os.listdir(src_dir):
        if not name.endswith(".safetensors") and os.path.isfile(os.path.join(src_dir, name)):
            shutil.copy(os.path.join(src_dir, name), dst_dir)


def _tiny_peft(base, rank: int, seed: int = 3) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    peft = {}
    for name, mod in base.named_modules():
        if name.startswith("model.language_model.layers.") and name.rsplit(".", 1)[-1] in (
                "q_proj", "k_proj", "v_proj", "o_proj"):
            out_f, in_f = mod.weight.shape
            peft[f"base_model.model.{name}.lora_A.weight"] = (0.1 * torch.randn(rank, in_f, generator=g)).to(torch.bfloat16)
            peft[f"base_model.model.{name}.lora_B.weight"] = (0.1 * torch.randn(out_f, rank, generator=g)).to(torch.bfloat16)
    return peft


@unittest.skipUnless(TINY_DIR, "set MUSE_SPARK_TINY_DIR")
class HfExportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hf-export-test-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_unchanged_export_is_byte_identical(self):
        out = os.path.join(self.tmp, "out")
        rep = export_hf(TINY_DIR, out, StateDictSource({}))
        self.assertEqual(rep["finalize"]["replaced_tensors"], 0)
        self.assertEqual(_sha(os.path.join(TINY_DIR, "model.safetensors")),
                         _sha(os.path.join(out, "model.safetensors")))
        for name in ("config.json", "tokenizer.json", "chat_template.jinja", "generation_config.json"):
            self.assertEqual(_sha(os.path.join(TINY_DIR, name)), _sha(os.path.join(out, name)), name)
        self.assertTrue(os.path.exists(os.path.join(out, "EXPORT_OK")))

    def test_sharded_multi_worker_export(self):
        base = os.path.join(self.tmp, "base")
        _reshard(TINY_DIR, base, 5)
        layout = HfLayout(base)
        g = torch.Generator().manual_seed(1)
        name = "model.language_model.layers.1.self_attn.q_proj.weight"
        dtype, shape, file = layout.tensor_info(name)
        new = torch.randn(shape, generator=g).to(torch.bfloat16)
        out = os.path.join(self.tmp, "out")
        for w in range(3):
            export_hf(base, out, StateDictSource({name: new}), worker=w, num_workers=3)
        with self.assertRaises(FileNotFoundError):
            finalize_export(base, os.path.join(self.tmp, "nowhere"), num_workers=3)
        fin = finalize_export(base, out, num_workers=3)
        self.assertEqual(fin["replaced_tensors"], 1)
        for f in layout.files:
            same = _sha(os.path.join(base, f)) == _sha(os.path.join(out, f))
            self.assertEqual(same, f != file, f)
        self.assertEqual(json.load(open(os.path.join(out, "model.safetensors.index.json"))),
                         json.load(open(os.path.join(base, "model.safetensors.index.json"))))
        self.assertTrue(torch.equal(HfLayout(out).read_tensor(name), new))

    def test_replaced_tensors_round_trip_through_transformers(self):
        """Trained text weights -> export -> transformers load -> identical logits."""
        ref = _hf_model(TINY_DIR)
        g = torch.Generator().manual_seed(2)
        trained = {}
        for name, p in ref.named_parameters():
            if name.startswith("model.language_model.layers.") and name.endswith(
                ("q_proj.weight", "o_proj.weight", "gate_up_proj", "mlp.gate.weight", "post_feedforward_layernorm.weight")
            ) or name == "lm_head.weight":
                with torch.no_grad():
                    p.add_(0.05 * torch.randn(p.shape, generator=g).to(p.dtype))
                trained[name] = p.detach().clone()
        self.assertTrue(any(t.dtype == torch.float32 for t in trained.values()), "router weight is F32")
        out = os.path.join(self.tmp, "out")
        rep = export_hf(TINY_DIR, out, StateDictSource(trained))
        self.assertEqual(rep["finalize"]["replaced_tensors"], len(trained))
        exported = HfLayout(out)
        for name, t in trained.items():
            self.assertTrue(torch.equal(exported.read_tensor(name), t), name)
        ids = _ids()
        loaded = _hf_model(out)
        self.assertTrue(torch.equal(_logits(loaded, ids), _logits(ref, ids)))
        base_logits = _logits(_hf_model(TINY_DIR), ids)
        self.assertGreater((base_logits - _logits(loaded, ids)).abs().max().item(), 1e-3)

    def _peft_export(self, rank: int, alpha: float):
        peft = _tiny_peft(_hf_model(TINY_DIR), rank)
        adapter_dir = os.path.join(self.tmp, "adapter")
        write_peft_dir(peft, adapter_dir, rank=rank, alpha=alpha, base_model_name_or_path=TINY_DIR)
        source = PeftSource(adapter_dir, HfLayout(TINY_DIR))
        out = os.path.join(self.tmp, "merged")
        rep = export_hf(TINY_DIR, out, source)
        return peft, adapter_dir, source, out, rep

    def test_peft_merge_export(self):
        rank, alpha = 4, 8.0
        peft, _, source, out, rep = self._peft_export(rank, alpha)
        self.assertEqual(len(source.keys()), 8)
        self.assertEqual(rep["finalize"]["replaced_tensors"], 8)
        layout, exported = HfLayout(TINY_DIR), HfLayout(out)
        scale = alpha / rank
        target_keys = sorted(source.keys())
        for key in target_keys:
            module = key[: -len(".weight")]
            a = peft[f"base_model.model.{module}.lora_A.weight"].float()
            b = peft[f"base_model.model.{module}.lora_B.weight"].float()
            want = (layout.read_tensor(key).float() + scale * (b @ a)).to(torch.bfloat16)
            self.assertTrue(torch.equal(exported.read_tensor(key), want), key)

    def test_peft_merge_matches_the_peft_library(self):
        try:
            from peft import PeftModel
        except ImportError:
            self.skipTest("peft not installed")
        _, adapter_dir, source, out, _ = self._peft_export(4, 8.0)
        exported = HfLayout(out)
        # The PEFT library reads the adapter and merges it itself (float32):
        # same tensors as the export, hence identical logits.
        peft_model = PeftModel.from_pretrained(_hf_model(TINY_DIR, torch.float32), adapter_dir).eval()
        lora_b = dict(peft_model.named_parameters())
        some = next(k for k in lora_b if "language_model" in k and k.endswith("lora_B.default.weight"))
        self.assertGreater(lora_b[some].abs().max().item(), 0.0, "PEFT did not load the adapter")
        ids = _ids()
        unmerged = _logits(peft_model, ids)
        peft_merged = peft_model.merge_and_unload().eval()
        merged_sd = peft_merged.state_dict()
        target_keys = sorted(source.keys())
        for key in target_keys:
            self.assertTrue(torch.equal(merged_sd[key].to(torch.bfloat16), exported.read_tensor(key)), key)
        with torch.no_grad():
            for key in target_keys:
                merged_sd[key].copy_(merged_sd[key].to(torch.bfloat16).float())
        ours = _logits(_hf_model(out, torch.float32), ids)
        self.assertTrue(torch.equal(ours, _logits(peft_merged, ids)))
        base_logits = _logits(_hf_model(TINY_DIR, torch.float32), ids)
        print(f"peft merge: adapter effect {(base_logits - unmerged).abs().max().item():.3e}, "
              f"bf16 rounding of merged weights {(ours - unmerged).abs().max().item():.3e}")


def _spark():
    try:
        from torchtitan.experiments.tpu.muse_spark import MuseSparkArgs, MuseSparkStateDictAdapter
        from torchtitan.experiments.tpu.muse_spark.configs import build_muse_spark_config
    except Exception:  # noqa: BLE001
        return None
    return MuseSparkArgs, MuseSparkStateDictAdapter, build_muse_spark_config


def tt_tiny_from_hf(converters=()):
    """TorchTitan Muse Spark tiny model with the public tiny checkpoint's weights."""
    MuseSparkArgs, MuseSparkStateDictAdapter, build = _spark()
    from safetensors.torch import load_file

    args = MuseSparkArgs.from_hf_config(TINY_DIR)
    cfg = build(args, max_context_length=256, experts_compute_dtype=None)
    for c in converters:
        cfg = c.build().convert(cfg) or cfg
    adapter = MuseSparkStateDictAdapter(cfg, TINY_DIR)
    hf = {k: v for k, v in load_file(os.path.join(TINY_DIR, "model.safetensors")).items()
          if k.startswith("model.language_model.") or k == "lm_head.weight"}
    model = cfg.build()
    model.init_states()
    missing, unexpected = model.load_state_dict(adapter.from_hf(hf), strict=False)
    assert not unexpected, unexpected
    assert all(".lora_" in k for k in missing), missing
    return model.float(), adapter


def tt_logits(model, ids: torch.Tensor) -> torch.Tensor:
    b, s = ids.shape
    positions = torch.arange(s).repeat(b)
    with torch.no_grad():
        return model(ids.reshape(-1), positions=positions).float().reshape(b, s, -1)


@unittest.skipUnless(TINY_DIR and _spark(), "needs MUSE_SPARK_TINY_DIR and the muse_spark port")
class DcpExportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="dcp-export-test-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_dcp_full_param_export(self):
        import torch.distributed.checkpoint as dcp

        model, adapter = tt_tiny_from_hf()
        g = torch.Generator().manual_seed(4)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.02 * torch.randn(p.shape, generator=g))  # "training"
        ckpt = os.path.join(self.tmp, "step-10")
        dcp.save({k: v for k, v in model.state_dict().items()}, checkpoint_id=ckpt)
        out = os.path.join(self.tmp, "hf")
        source = DcpSource(ckpt, adapter)
        rep = export_hf(TINY_DIR, out, source)
        n_text = sum(1 for k in HfLayout(TINY_DIR).weight_map
                     if k.startswith("model.language_model.") or k == "lm_head.weight")
        self.assertEqual(rep["finalize"]["replaced_tensors"], n_text)
        expected = adapter.to_hf({k: v.detach() for k, v in model.state_dict().items()})
        exported, base = HfLayout(out), HfLayout(TINY_DIR)
        for key, t in expected.items():
            want = t.to(hf_export.ST_DTYPES[base.tensor_info(key)[0]])
            self.assertTrue(torch.equal(exported.read_tensor(key), want), key)
        ids = _ids(70)
        hf = _logits(_hf_model(out, torch.float32), ids)
        model_bf16_weights = tt_tiny_from_hf()[0]
        model_bf16_weights.load_state_dict(adapter.from_hf({k: exported.read_tensor(k).float() for k in expected}), strict=True)
        tt = tt_logits(model_bf16_weights, ids)
        diff = (hf - tt).abs().max().item()
        print(f"dcp export: max|HF(export) - TT| = {diff:.3e} (logit scale {hf.abs().max().item():.1f})")
        self.assertLess(diff, 2e-3)


if __name__ == "__main__":
    unittest.main()
