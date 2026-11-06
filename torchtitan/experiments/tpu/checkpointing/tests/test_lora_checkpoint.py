# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for adapter-only LoRA checkpoints and PEFT adapters.

Model-agnostic (always run):

* ``Llama3AdapterOnlyTest``: TorchTitan's Llama 3 debugmodel with LoRA from
  ``LoRAConverter``. Bit-exact resume through TorchTitan's ``CheckpointManager``:
  full-parameter (stock DCP checkpoint of every parameter + optimizer) and
  LoRA adapter-only (``enable_adapter_only_checkpoints``: base re-read from the
  Hugging Face checkpoint, adapters + optimizer from DCP), also with the tuned
  DCP I/O; PEFT adapter export and import.
* ``PeftLayoutTest``: TorchTitan <-> PEFT adapter tensors, including adapters
  that permute rows (RoPE) and re-layouts that cannot be supported.
* ``CanonicalNamesTest``: activation-checkpoint and torch.compile wrappers in
  parameter names.

``ResumeTest`` repeats the resume checks on Muse Spark and needs
``MUSE_SPARK_TINY_DIR`` (public tiny test checkpoint), the Muse Spark port
(``muse_spark``) and its LoRA converter (``post_training.lora``).
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import types
import unittest

import torch
import torch.nn.functional as F

from torchtitan.experiments.tpu.checkpointing.tests import fixtures

TINY_DIR = os.environ.get("MUSE_SPARK_TINY_DIR")


def _deps():
    try:
        from torchtitan.experiments.tpu.muse_spark import MuseSparkArgs  # noqa: F401
        from torchtitan.experiments.tpu.post_training import lora  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


class _Counter:
    """Minimal Stateful object for the dataloader / train state."""

    def __init__(self):
        self.step = 0

    def state_dict(self):
        return {"step": torch.tensor(self.step)}

    def load_state_dict(self, sd):
        self.step = int(sd["step"])


def _batch(step: int, b: int = 2, s: int = 40):
    g = torch.Generator().manual_seed(1000 + step)
    tokens = torch.randint(0, 199_000, (b, s), generator=g)
    labels = torch.randint(0, 199_000, (b, s), generator=g)
    return tokens, labels


def _build(lora: bool, init_seed: int):
    from torchtitan.experiments.tpu.muse_spark import MuseSparkArgs, MuseSparkStateDictAdapter
    from torchtitan.experiments.tpu.muse_spark.configs import build_muse_spark_config
    from torchtitan.experiments.tpu.post_training.lora import LoRAOptions, apply_lora

    args = MuseSparkArgs.from_hf_config(TINY_DIR)

    def config():
        return build_muse_spark_config(args, max_context_length=256, experts_compute_dtype=None)

    spec = types.SimpleNamespace(model=config(), state_dict_adapter=MuseSparkStateDictAdapter)
    if lora:
        apply_lora(spec, LoRAOptions(rank=4, alpha=8.0))
    torch.manual_seed(init_seed)
    model = spec.model.build()
    model.init_states()
    model.float()
    # The checkpoint mapping covers the base weights only: build it from the
    # pre-LoRA config (the strict Muse Spark adapter rejects unmapped keys);
    # the LoRA wrapper strips adapter keys before mapping.
    adapter = spec.state_dict_adapter(config(), TINY_DIR)
    return model, adapter


def _manager(model, adapter, folder: str, interval: int):
    from torchtitan.components.checkpointer.dcp import CheckpointManager
    from torchtitan.components.optimizer.lr_scheduler import LRSchedulersContainer
    from torchtitan.components.optimizer.optimizer import OptimizersContainer, ParamGroupConfig

    optim = OptimizersContainer.Config(
        param_groups=[ParamGroupConfig(pattern=".*", optimizer_name="AdamW",
                                       optimizer_kwargs={"lr": 3e-3, "weight_decay": 0.01})],
        implementation="for-loop",
    ).build(model_parts=[model])
    lr = LRSchedulersContainer.Config(warmup_steps=2).build(optimizers=optim, training_steps=8)
    train_state = _Counter()
    ckpt = CheckpointManager(
        CheckpointManager.Config(enable=True, folder="checkpoint", interval=interval,
                                 initial_load_in_hf=True, last_save_model_only=False, keep_latest_k=0),
        dataloader=_Counter(), model_parts=[model], optimizers=optim, lr_schedulers=lr,
        states={"train_state": train_state}, sd_adapter=adapter, base_folder=folder,
    )
    return ckpt, optim, lr, train_state


def _train(model, optim, lr, train_state, ckpt, until: int) -> list[float]:
    losses = []
    while train_state.step < until:
        train_state.step += 1
        tokens, labels = _batch(train_state.step)
        b, s = tokens.shape
        optim.zero_grad()
        logits = model(tokens.reshape(-1), positions=torch.arange(s).repeat(b))
        loss = F.cross_entropy(logits.float(), labels.reshape(-1))
        loss.backward()
        optim.step()
        lr.step()
        losses.append(loss.item())
        ckpt.save(curr_step=train_state.step)
    return losses


def _dir_bytes(path: str) -> int:
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs)


def _model_keys(metadata) -> list[str]:
    return [k for k in metadata if not k.startswith(("optimizer", "lr_scheduler", "dataloader", "train_state"))]


class _TinyLoRABlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = torch.nn.Linear(4, 4, bias=False)
        self.lora_a = torch.nn.Linear(4, 2, bias=False)
        self.lora_b = torch.nn.Linear(2, 4, bias=False)


def _wrapped_blocks(wrap) -> torch.nn.Module:
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([wrap(_TinyLoRABlock()) for _ in range(2)])
    for name, param in model.named_parameters():
        param.requires_grad_(".lora_" in name)
    return model


class CanonicalNamesTest(unittest.TestCase):
    """Adapter-only checkpoints keep the ``state_dict()`` keys of models with wrapped blocks."""

    def test_trainable_wrapper_with_checkpoint_wrappers(self):
        """Every TPU recipe wraps its blocks for activation checkpointing."""
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

        from torchtitan.experiments.tpu.checkpointing.lora_checkpoint import TrainableModelWrapper

        model = _wrapped_blocks(checkpoint_wrapper)
        raw = [n for n, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(all("_checkpoint_wrapped_module." in n for n in raw))
        state = TrainableModelWrapper([model]).state_dict()
        self.assertEqual(sorted(state), sorted(k for k in model.state_dict() if ".lora_" in k))
        self.assertEqual(len(state), 4)

    def test_trainable_wrapper_with_compile_wrappers(self):
        """A wrapping torch.compile keeps ``_orig_mod`` in named_parameters() and state_dict()."""
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

        from torchtitan.experiments.tpu.checkpointing.lora_checkpoint import TrainableModelWrapper

        wraps = {
            "compile": torch.compile,
            "checkpoint(compile)": lambda block: checkpoint_wrapper(torch.compile(block)),
            "compile(checkpoint)": lambda block: torch.compile(checkpoint_wrapper(block)),
        }
        for label, wrap in wraps.items():
            with self.subTest(wrap=label):
                model = _wrapped_blocks(wrap)
                want = sorted(k for k in model.state_dict() if ".lora_" in k)
                self.assertEqual(len(want), 4)
                self.assertTrue(all("._orig_mod." in k for k in want), want)
                self.assertEqual(sorted(TrainableModelWrapper([model]).state_dict()), want)


class Llama3AdapterOnlyTest(unittest.TestCase):
    """Resume and PEFT checks on TorchTitan's Llama 3 debugmodel (model-agnostic code paths)."""

    @classmethod
    def setUpClass(cls):
        fixtures.ensure_process_group()

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="adapter-only-test-")
        self.hf_dir = os.path.join(self.tmp, "hf")
        base, spec = fixtures.build_llama3(lora=False, seed=0)
        self.base_state = {k: v.detach().clone() for k, v in base.state_dict().items()}
        fixtures.write_hf_checkpoint(self.hf_dir, self.base_state, spec)

    def tearDown(self):
        fixtures.restore_checkpointer_dcp()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _setup(self, *, lora: bool, seed: int, folder: str, interval: int, tuned: bool = False):
        from torchtitan.experiments.tpu.checkpointing.dcp_parallel import tune_checkpoint_manager
        from torchtitan.experiments.tpu.checkpointing.lora_checkpoint import enable_adapter_only_checkpoints

        model, spec = fixtures.build_llama3(lora=lora, seed=seed)
        ckpt, optim, lr, ts = fixtures.build_manager(model, spec, self.hf_dir, folder, interval)
        if lora:
            enable_adapter_only_checkpoints(ckpt, [model])
        if tuned:
            tune_checkpoint_manager(ckpt, save_threads=4, load_threads=4)
        ckpt.load()
        return model, ckpt, optim, lr, ts

    def _resume(self, *, lora: bool, tuned: bool = False):
        steps, save_at = 6, 3
        # Run A: from the Hugging Face base, checkpoint every 3 steps.
        model, ckpt, optim, lr, ts = self._setup(
            lora=lora, seed=1, folder=os.path.join(self.tmp, "a"), interval=save_at, tuned=tuned)
        loaded = model.state_dict()
        for key, value in self.base_state.items():
            self.assertTrue(torch.equal(loaded[key], value), f"base weight {key} not loaded")
        losses_a = fixtures.train(model, optim, lr, ts, ckpt, until=steps)
        final_a = {k: v.detach().clone() for k, v in model.state_dict().items()}
        step3 = os.path.join(self.tmp, "a", "checkpoint", f"step-{save_at}")
        self.assertTrue(os.path.isdir(step3))
        # Run B: a new process state (different init) resumes from step 3.
        os.makedirs(os.path.join(self.tmp, "b", "checkpoint"))
        shutil.copytree(step3, os.path.join(self.tmp, "b", "checkpoint", f"step-{save_at}"))
        model_b, ckpt_b, optim_b, lr_b, ts_b = self._setup(
            lora=lora, seed=2, folder=os.path.join(self.tmp, "b"), interval=100, tuned=tuned)
        self.assertEqual(ts_b.step, save_at)
        losses_b = fixtures.train(model_b, optim_b, lr_b, ts_b, ckpt_b, until=steps)
        self.assertEqual(losses_a[save_at:], losses_b, "resumed losses differ")
        final_b = model_b.state_dict()
        for key, value in final_a.items():
            self.assertTrue(torch.equal(value, final_b[key]), key)
        from torch.distributed.checkpoint import FileSystemReader

        metadata = FileSystemReader(step3).read_metadata().state_dict_metadata
        return model, ckpt, ckpt_b, metadata, step3

    def test_full_param_resume_bit_exact(self):
        model, _, _, metadata, _ = self._resume(lora=False)
        self.assertEqual(sorted(_model_keys(metadata)), sorted(model.state_dict()))

    def test_adapter_only_resume_bit_exact(self):
        model, ckpt, ckpt_b, metadata, step3 = self._resume(lora=True)
        lora_keys = sorted(k for k in model.state_dict() if ".lora_" in k)
        self.assertEqual(len(lora_keys), 24)  # 6 layers x (wqkv, wo) x (lora_a, lora_b)
        self.assertEqual(sorted(_model_keys(metadata)), lora_keys)
        self.assertIsNone(ckpt.adapter_resume_step)  # run A started from the base only
        self.assertEqual(ckpt_b.adapter_resume_step, 3)
        self.assertEqual(set(ckpt_b.adapter_load_seconds), {"base", "adapter"})
        frozen = sum(p.numel() * p.element_size() for p in model.parameters() if not p.requires_grad)
        self.assertLess(fixtures.dir_bytes(step3), frozen / 10)

    def test_adapter_only_resume_bit_exact_tuned_io(self):
        """Same, with 4 writer threads per rank and the parallel DCP reader."""
        _, _, _, _, step3 = self._resume(lora=True, tuned=True)
        files = sorted(f for f in os.listdir(step3) if f.endswith(".distcp"))
        self.assertEqual(len(files), 4, files)  # one file per writer thread

    def test_adapter_only_load_errors(self):
        from torchtitan.experiments.tpu.checkpointing.lora_checkpoint import enable_adapter_only_checkpoints

        model, spec = fixtures.build_llama3(lora=True, seed=1)
        ckpt, _, _, _ = fixtures.build_manager(model, spec, self.hf_dir, os.path.join(self.tmp, "c"), 3)
        enable_adapter_only_checkpoints(ckpt, [model])
        with self.assertRaises(FileNotFoundError):  # no checkpoint folder
            ckpt.load(step=3)
        os.makedirs(os.path.join(self.tmp, "c", "checkpoint"))
        with self.assertRaises(FileNotFoundError):  # no such step
            ckpt.load(step=3)
        self.assertTrue(ckpt.load())  # fresh start: base only
        self.assertIsNone(ckpt.adapter_resume_step)

    def test_enable_adapter_only_checks(self):
        from torchtitan.experiments.tpu.checkpointing.lora_checkpoint import enable_adapter_only_checkpoints

        model, spec = fixtures.build_llama3(lora=True, seed=1)
        disabled = types.SimpleNamespace(enable=False)
        enable_adapter_only_checkpoints(disabled, [model])  # no-op
        self.assertFalse(hasattr(disabled, "load"))
        adapter = spec.state_dict_adapter(spec.model, self.hf_dir)
        with self.assertRaises(ValueError):
            enable_adapter_only_checkpoints(
                types.SimpleNamespace(enable=True, last_save_in_hf=True, sd_adapter=adapter), [model])
        with self.assertRaises(ValueError):
            enable_adapter_only_checkpoints(
                types.SimpleNamespace(enable=True, last_save_in_hf=False, sd_adapter=None), [model])
        base, _ = fixtures.build_llama3(lora=False, seed=1)
        base.requires_grad_(False)
        with self.assertRaises(ValueError):  # nothing trainable
            enable_adapter_only_checkpoints(
                types.SimpleNamespace(enable=True, last_save_in_hf=False, sd_adapter=adapter), [base])

    def test_peft_adapter_round_trip(self):
        from torchtitan.experiments.tpu.checkpointing.lora_checkpoint import (
            load_peft_adapter,
            read_peft_adapter,
            save_peft_adapter,
        )

        model, spec = fixtures.build_llama3(lora=True, seed=3, targets=("wo",))
        adapter = spec.state_dict_adapter(spec.model, self.hf_dir)
        g = torch.Generator().manual_seed(5)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name.endswith("lora_b.weight"):
                    param.copy_(0.2 * torch.randn(param.shape, generator=g))
        out = os.path.join(self.tmp, "adapter")
        report = save_peft_adapter([model], adapter, out, rank=fixtures.LORA_RANK, alpha=fixtures.LORA_ALPHA,
                                   base_model_name_or_path="base", dtype=None)
        self.assertEqual(report["target_modules"], ["o_proj"])
        self.assertEqual(report["tensors"], 12)
        config, tensors = read_peft_adapter(out)
        self.assertEqual((config["r"], config["lora_alpha"], config["peft_type"]), (4, 8.0, "LORA"))
        self.assertIn("base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight", tensors)
        fresh, _ = fixtures.build_llama3(lora=True, seed=4, targets=("wo",))
        self.assertEqual(load_peft_adapter([fresh], adapter, out), 12)
        fresh_params = dict(fresh.named_parameters())
        for name, param in model.named_parameters():
            if ".lora_" in name:
                self.assertTrue(torch.equal(param, fresh_params[name]), name)

    def test_peft_export_rejects_fused_qkv(self):
        """A fused wqkv adapter has no one-to-one Hugging Face module: refuse rather than guess."""
        from torchtitan.experiments.tpu.checkpointing.lora_checkpoint import save_peft_adapter

        model, spec = fixtures.build_llama3(lora=True, seed=3, targets=("wqkv",))
        with self.assertRaises(NotImplementedError):
            save_peft_adapter([model], spec.state_dict_adapter(spec.model, self.hf_dir),
                              os.path.join(self.tmp, "adapter"), rank=4, alpha=8.0)


def _rope_permute(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Row permutation of Hugging Face's Llama conversion (interleaved -> half-split RoPE rows)."""
    dim1, dim2 = w.shape
    return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)


class _ToyAdapter:
    """``layers.N.wq`` -> ``q_proj`` (rows permuted for RoPE), ``layers.N.wo`` -> ``o_proj``."""

    def __init__(self, n_heads: int = 2, mode: str = "permute"):
        self.n_heads = n_heads
        self.mode = mode

    def to_hf(self, state_dict):
        out = {}
        for key, value in state_dict.items():
            match = re.fullmatch(r"layers\.(\d+)\.(wq|wo)\.weight", key)
            if match is None:
                continue  # adapter tensors are not part of the base checkpoint
            layer, name = match.groups()
            if name == "wo":
                out[f"model.layers.{layer}.self_attn.o_proj.weight"] = value
            elif self.mode == "permute":
                out[f"model.layers.{layer}.self_attn.q_proj.weight"] = _rope_permute(value, self.n_heads)
            elif self.mode == "permute_cast":  # row permutation, then a cast to bf16
                out[f"model.layers.{layer}.self_attn.q_proj.weight"] = _rope_permute(
                    value, self.n_heads).to(torch.bfloat16)
            elif self.mode == "cast":  # a rename that also casts
                out[f"model.layers.{layer}.self_attn.q_proj.weight"] = value.to(torch.bfloat16)
            elif self.mode == "view":  # a rename through view(), which needs a compatible stride
                out[f"model.layers.{layer}.self_attn.q_proj.weight"] = value.view(-1).view(value.shape)
            elif self.mode == "transpose":
                out[f"model.layers.{layer}.self_attn.q_proj.weight"] = value.t()
            elif self.mode == "flip_columns":
                out[f"model.layers.{layer}.self_attn.q_proj.weight"] = value.flip(1)
            elif self.mode == "scale":
                out[f"model.layers.{layer}.self_attn.q_proj.weight"] = 2 * value
            elif self.mode == "split":
                half = value.shape[0] // 2
                out[f"model.layers.{layer}.self_attn.q_proj.weight"] = value[:half]
                out[f"model.layers.{layer}.self_attn.q_extra.weight"] = value[half:]
            else:
                out[f"model.layers.{layer}.self_attn.q_proj.weight"] = value
        return out


def _random_lora(dim: int = 8, rank: int = 2, layers: int = 2, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    out = {}
    for layer in range(layers):
        for name in ("wq", "wo"):
            out[f"layers.{layer}.{name}.lora_a.weight"] = torch.randn(rank, dim, generator=g)
            out[f"layers.{layer}.{name}.lora_b.weight"] = torch.randn(dim, rank, generator=g)
    return out


class _LoRALinear(torch.nn.Linear):
    def __init__(self, dim: int, rank: int):
        super().__init__(dim, dim, bias=False)
        self.lora_a = torch.nn.Linear(dim, rank, bias=False)
        self.lora_b = torch.nn.Linear(rank, dim, bias=False)


class _Block(torch.nn.Module):
    def __init__(self, dim: int = 8, rank: int = 2):
        super().__init__()
        self.wq = _LoRALinear(dim, rank)
        self.wo = _LoRALinear(dim, rank)


class PeftLayoutTest(unittest.TestCase):
    def test_rename_round_trip(self):
        from torchtitan.experiments.tpu.checkpointing.lora_checkpoint import peft_to_tt_lora, tt_lora_to_peft

        lora = _random_lora(dim=512)  # more rows than bf16 represents exactly as integers
        for mode in ("rename", "cast", "view"):
            with self.subTest(mode=mode):
                peft = tt_lora_to_peft(lora, _ToyAdapter(n_heads=8, mode=mode))
                self.assertEqual(len(peft), len(lora))
                self.assertTrue(torch.equal(peft["base_model.model.model.layers.1.self_attn.q_proj.lora_B.weight"],
                                            lora["layers.1.wq.lora_b.weight"]))
                back = peft_to_tt_lora(peft, _ToyAdapter(n_heads=8, mode=mode),
                                       {k: tuple(v.shape) for k, v in lora.items()})
                self.assertEqual(sorted(back), sorted(lora))
                for key, value in lora.items():
                    self.assertTrue(torch.equal(back[key], value), key)

    def test_lora_b_rows_follow_the_adapter_permutation(self):
        from torchtitan.experiments.tpu.checkpointing.lora_checkpoint import peft_to_tt_lora, tt_lora_to_peft

        adapter = _ToyAdapter(mode="permute")
        lora = _random_lora()
        peft = tt_lora_to_peft(lora, adapter)
        scale = 3.0
        g = torch.Generator().manual_seed(1)
        for layer in range(2):
            w = torch.randn(8, 8, generator=g)
            a, b = lora[f"layers.{layer}.wq.lora_a.weight"], lora[f"layers.{layer}.wq.lora_b.weight"]
            module = f"base_model.model.model.layers.{layer}.self_attn.q_proj"
            self.assertTrue(torch.equal(peft[f"{module}.lora_A.weight"], a))
            self.assertFalse(torch.equal(peft[f"{module}.lora_B.weight"], b))
            # Merging in the HF layout equals mapping the merged TorchTitan weight.
            (want,) = adapter.to_hf({f"layers.{layer}.wq.weight": w + scale * (b @ a)}).values()
            (w_hf,) = adapter.to_hf({f"layers.{layer}.wq.weight": w}).values()
            got = w_hf + scale * (peft[f"{module}.lora_B.weight"] @ peft[f"{module}.lora_A.weight"])
            self.assertTrue(torch.equal(got, want))
        back = peft_to_tt_lora(peft, adapter, {k: tuple(v.shape) for k, v in lora.items()})
        for key, value in lora.items():
            self.assertTrue(torch.equal(back[key], value), key)

    def test_lora_b_rows_follow_a_permutation_that_casts(self):
        """Row ids above 256 are not exact in bf16; the probes read them digit by digit."""
        from torchtitan.experiments.tpu.checkpointing.lora_checkpoint import peft_to_tt_lora, tt_lora_to_peft

        adapter = _ToyAdapter(n_heads=8, mode="permute_cast")
        lora = _random_lora(dim=512)
        peft = tt_lora_to_peft(lora, adapter)
        for layer in range(2):
            a, b = lora[f"layers.{layer}.wq.lora_a.weight"], lora[f"layers.{layer}.wq.lora_b.weight"]
            module = f"base_model.model.model.layers.{layer}.self_attn.q_proj"
            self.assertTrue(torch.equal(peft[f"{module}.lora_A.weight"], a))
            self.assertTrue(torch.equal(peft[f"{module}.lora_B.weight"], _rope_permute(b, 8)))
        back = peft_to_tt_lora(peft, adapter, {k: tuple(v.shape) for k, v in lora.items()})
        for key, value in lora.items():
            self.assertTrue(torch.equal(back[key], value), key)

    def test_unsupported_relayouts_raise(self):
        from torchtitan.experiments.tpu.checkpointing.lora_checkpoint import tt_lora_to_peft

        for mode in ("transpose", "scale", "split", "flip_columns"):
            with self.subTest(mode=mode), self.assertRaises(NotImplementedError):
                tt_lora_to_peft(_random_lora(), _ToyAdapter(mode=mode))

    def test_peft_directory_round_trip(self):
        from torchtitan.experiments.tpu.checkpointing.lora_checkpoint import (
            read_peft_adapter,
            tt_lora_to_peft,
            write_peft_dir,
        )

        peft = tt_lora_to_peft(_random_lora(), _ToyAdapter(mode="permute"))
        with tempfile.TemporaryDirectory() as tmp:
            report = write_peft_dir(peft, os.path.join(tmp, "adapter"), rank=2, alpha=4.0,
                                    base_model_name_or_path="base")
            config, tensors = read_peft_adapter(os.path.join(tmp, "adapter"))
            local_url, _ = read_peft_adapter("file://" + os.path.join(tmp, "adapter"))
        self.assertEqual(report["target_modules"], ["o_proj", "q_proj"])
        self.assertEqual(config["target_modules"], ["o_proj", "q_proj"])
        self.assertEqual((config["r"], config["lora_alpha"], config["base_model_name_or_path"]), (2, 4.0, "base"))
        self.assertEqual(local_url, config)
        self.assertEqual(sorted(tensors), sorted(peft))
        for key, value in peft.items():
            self.assertTrue(torch.equal(tensors[key], value), key)

    def test_wrapped_model_peft_save_and_load(self):
        """Activation-checkpoint wrappers do not leak into PEFT names."""
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

        from torchtitan.experiments.tpu.checkpointing.lora_checkpoint import (
            gather_lora_state_dict,
            load_peft_adapter,
            save_peft_adapter,
        )

        def model(seed: int) -> torch.nn.Module:
            torch.manual_seed(seed)
            root = torch.nn.Module()
            root.layers = torch.nn.ModuleList([checkpoint_wrapper(_Block()) for _ in range(2)])
            return root

        src = model(0)
        self.assertEqual(sorted(gather_lora_state_dict(src)), sorted(_random_lora()))
        with tempfile.TemporaryDirectory() as tmp:
            report = save_peft_adapter(src, _ToyAdapter(), tmp, rank=2, alpha=4.0, dtype=None)
            self.assertEqual(report["tensors"], 8)
            dst = model(1)
            self.assertEqual(load_peft_adapter(dst, _ToyAdapter(), tmp), 8)
        dst_params = dict(dst.named_parameters())
        for name, param in src.named_parameters():
            if ".lora_" in name:
                self.assertTrue(torch.equal(param, dst_params[name]), name)


@unittest.skipUnless(TINY_DIR and _deps(), "needs MUSE_SPARK_TINY_DIR, muse_spark and post_training.lora")
class ResumeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.ensure_process_group()

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="resume-test-")

    def tearDown(self):
        fixtures.restore_checkpointer_dcp()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, lora: bool, tuned: bool = False):
        from torchtitan.experiments.tpu.checkpointing.dcp_parallel import tune_checkpoint_manager
        from torchtitan.experiments.tpu.checkpointing.lora_checkpoint import enable_adapter_only_checkpoints

        steps, save_at = 6, 3
        # Run A: uninterrupted, checkpoint every 3 steps.
        model, adapter = _build(lora, init_seed=1)
        ckpt, optim, lr, ts = _manager(model, adapter, os.path.join(self.tmp, "a"), interval=save_at)
        if lora:
            enable_adapter_only_checkpoints(ckpt, [model])
        if tuned:
            tune_checkpoint_manager(ckpt, save_threads=4, load_threads=4)
        ckpt.load()
        losses_a = _train(model, optim, lr, ts, ckpt, until=steps)
        final_a = {k: v.detach().clone() for k, v in model.state_dict().items()}
        step3 = os.path.join(self.tmp, "a", "checkpoint", f"step-{save_at}")
        self.assertTrue(os.path.isdir(step3))
        # Run B: a new process state (different init seed) resumes from step 3.
        os.makedirs(os.path.join(self.tmp, "b", "checkpoint"))
        shutil.copytree(step3, os.path.join(self.tmp, "b", "checkpoint", f"step-{save_at}"))
        model_b, adapter_b = _build(lora, init_seed=2)
        ckpt_b, optim_b, lr_b, ts_b = _manager(model_b, adapter_b, os.path.join(self.tmp, "b"), interval=100)
        if lora:
            enable_adapter_only_checkpoints(ckpt_b, [model_b])
        if tuned:
            tune_checkpoint_manager(ckpt_b, save_threads=4, load_threads=4)
        ckpt_b.load()
        self.assertEqual(ts_b.step, save_at)
        losses_b = _train(model_b, optim_b, lr_b, ts_b, ckpt_b, until=steps)
        self.assertEqual(losses_a[save_at:], losses_b, "resumed losses differ")
        final_b = model_b.state_dict()
        for k, v in final_a.items():
            self.assertTrue(torch.equal(v, final_b[k]), k)
        from torch.distributed.checkpoint import FileSystemReader

        md = FileSystemReader(step3).read_metadata().state_dict_metadata
        return losses_a, md, _dir_bytes(step3), model, adapter

    def test_full_param_resume_bit_exact(self):
        losses, md, nbytes, model, _ = self._run(lora=False)
        n_params = sum(p.numel() for p in model.parameters())
        self.assertTrue(all(k in md for k in model.state_dict()))
        print(f"full-param resume bit-exact: losses {[round(x, 4) for x in losses]}, "
              f"step-3 checkpoint {nbytes / 1e6:.1f} MB for {n_params / 1e6:.2f}M params")

    def test_full_param_resume_bit_exact_tuned_io(self):
        """Same, with 4 writer threads per rank and the parallel DCP reader."""
        losses, md, nbytes, model, _ = self._run(lora=False, tuned=True)
        step3 = os.path.join(self.tmp, "a", "checkpoint", "step-3")
        files = sorted(f for f in os.listdir(step3) if f.endswith(".distcp"))
        self.assertEqual(len(files), 4, files)  # one file per writer thread
        print(f"full-param resume bit-exact with tuned DCP I/O: {files}")

    def test_lora_adapter_only_resume_bit_exact(self):
        losses, md, nbytes, model, adapter = self._run(lora=True)
        model_keys = _model_keys(md)
        self.assertTrue(model_keys)
        self.assertTrue(all(".lora_" in k for k in model_keys), model_keys[:5])
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"lora adapter-only resume bit-exact: losses {[round(x, 4) for x in losses]}, "
              f"step-3 checkpoint {nbytes / 1e6:.2f} MB ({len(model_keys)} adapter tensors, "
              f"{trainable} trainable vs {frozen} frozen params)")

    def test_lora_to_peft_to_merged_hf(self):
        """TT LoRA weights -> PEFT adapter -> merged original-layout export -> transformers."""
        from transformers import MuseSparkForConditionalGeneration

        from torchtitan.experiments.tpu.checkpointing.hf_export import HfLayout, PeftSource, export_hf
        from torchtitan.experiments.tpu.checkpointing.lora_checkpoint import (
            load_peft_adapter,
            save_peft_adapter,
        )

        model, adapter = _build(lora=True, init_seed=3)
        from safetensors.torch import load_file

        hf = {k: v for k, v in load_file(os.path.join(TINY_DIR, "model.safetensors")).items()
              if k.startswith("model.language_model.") or k == "lm_head.weight"}
        model.load_state_dict(adapter.from_hf(hf), strict=False)
        g = torch.Generator().manual_seed(5)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n.endswith("lora_b.weight"):
                    p.copy_(0.2 * torch.randn(p.shape, generator=g))
        adir = os.path.join(self.tmp, "adapter")
        rep = save_peft_adapter([model], adapter, adir, rank=4, alpha=8.0,
                                base_model_name_or_path="tiny", dtype=None)
        self.assertEqual(rep["target_modules"], ["k_proj", "o_proj", "q_proj", "v_proj"])
        # Round trip into a fresh LoRA model.
        fresh, adapter2 = _build(lora=True, init_seed=4)
        self.assertEqual(load_peft_adapter([fresh], adapter2, adir), rep["tensors"])
        for n, p in model.named_parameters():
            if ".lora_" in n:
                self.assertTrue(torch.equal(p, dict(fresh.named_parameters())[n]), n)
        out = os.path.join(self.tmp, "merged")
        export_hf(TINY_DIR, out, PeftSource(adir, HfLayout(TINY_DIR)))
        hf_model = MuseSparkForConditionalGeneration.from_pretrained(
            out, dtype=torch.float32, attn_implementation="eager").eval()
        tokens, _ = _batch(0, s=70)
        b, s = tokens.shape
        pos = torch.arange(s).repeat(b)
        with torch.no_grad():
            unmerged = model(tokens.reshape(-1), positions=pos).float()
            # Reference: the same model with W <- bf16(W + scale * B @ A), B <- 0,
            # i.e. exactly the weights the original-layout (bf16) export holds.
            for name, mod in model.named_modules():
                if hasattr(mod, "lora_a") and hasattr(mod, "lora_b"):
                    w = mod.weight
                    merged = w.float() + (8.0 / 4) * (mod.lora_b.weight.float() @ mod.lora_a.weight.float())
                    w.copy_(merged.to(torch.bfloat16).float())
                    mod.lora_b.weight.zero_()
            want = model(tokens.reshape(-1), positions=pos).float()
            got = hf_model(input_ids=tokens, use_cache=False).logits.float().reshape(b * s, -1)
            base = MuseSparkForConditionalGeneration.from_pretrained(
                TINY_DIR, dtype=torch.float32, attn_implementation="eager").eval()
            base_logits = base(input_ids=tokens, use_cache=False).logits.float().reshape(b * s, -1)
        err = (got - want).abs().max().item()
        print(f"TT LoRA -> PEFT -> merged HF export: max|HF(export) - TT(merged weights)| {err:.3e}; "
              f"adapter effect {(base_logits - unmerged).abs().max().item():.3e}; "
              f"bf16 rounding of merged weights {(want - unmerged).abs().max().item():.3e}")
        self.assertLess(err, 1e-5)


if __name__ == "__main__":
    unittest.main()
