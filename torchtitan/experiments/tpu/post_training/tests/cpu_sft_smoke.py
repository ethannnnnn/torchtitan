# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU end-to-end smoke of chat SFT (no TPU).

Runs the real SFT path -- ConfigManager CLI parse, SFTTPUTrainer on
TPUTrainer, HF initial load, ChatSFTDataLoader (Muse Spark chat template,
assistant-only labels, packed documents), ChunkedLossWrapper(CrossEntropyLoss)
with the global supervised-token denominator, and checkpoint saves -- on CPU
with gloo, world size 1. The TPU Qwen3.5 debug model (random weights) in five
variants: full-parameter AdamW, LoRA (r 8) on the attention projections with
TorchTitan's full DCP save, LoRA with adapter saves, full-parameter AdamW8bit,
and bf16 weights with AdamWBf16; with ``--spark-checkpoint`` (and the Muse
Spark TPU package installed), the Muse Spark port loaded from the public tiny
checkpoint, full-parameter and LoRA. Every variant must lower the training
loss on a small repeated dataset of short templated replies; LoRA runs must
leave every base weight bit-identical to the checkpoint, and LoRA runs with
adapter saves must write the adapter-only resume state (which needs
``checkpointing/lora_checkpoint.py``; without it they write the PEFT adapter
only, ``sft.lora_save=peft``).

    python -m torchtitan.experiments.tpu.post_training.tests.cpu_sft_smoke \\
        --tokenizer-dir <Muse Spark tokenizer dir> --work <new dir> \\
        [--spark-checkpoint <public tiny Muse Spark snapshot>]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import shutil
import sys

import torch

from torchtitan.experiments.tpu.post_training.tests import smoke_common

PRODUCTS = ("trail running shoe", "waterproof jacket", "wool socks", "camping stove",
            "yoga mat", "steel water bottle")
SYSTEM_PROMPT = "Say in one or two short sentences whether the product matches the query."

# Expected PEFT tensor names of the adapter exports: q/k/v/o of every
# attention layer (Qwen3.5 debug: layers 3 and 7 use full attention).
EXPECTED_ADAPTER_KEYS = {
    "qwen_debug": {
        f"base_model.model.model.layers.{layer}.self_attn.{proj}_proj.lora_{ab}.weight"
        for layer in (3, 7) for proj in "qkvo" for ab in "AB"
    },
    "spark_tiny": {
        f"base_model.model.model.language_model.layers.{layer}.self_attn.{proj}_proj.lora_{ab}.weight"
        for layer in (0, 1) for proj in "qkvo" for ab in "AB"
    },
}


def write_conversations(path: pathlib.Path, n: int = 24) -> None:
    """``n`` short single-turn conversations with templated replies."""
    rng = random.Random(0)
    rows = []
    for i in range(n):
        product = PRODUCTS[i % len(PRODUCTS)]
        query = product.split()[-1] + "s"
        exact = i % 3 == 0
        reply = (f"The shopper wants {query} and the {product} is one. "
                 + ("It is an exact match." if exact else "It is a close substitute."))
        rows.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {query}\nProduct: {product}\n"
                                        f"Shopper: {rng.randint(1, 9)}"},
            {"role": "assistant", "content": reply,
             "reasoning_content": "Compare the query with the product."},
        ]})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def run(config_name: str, work: pathlib.Path, assets: pathlib.Path, data: pathlib.Path,
        template_dir: pathlib.Path, extra: list[str] | None = None) -> dict:
    from torchtitan.config import ConfigManager
    from torchtitan.experiments.tpu.post_training import sft_train

    dump = work / "dump"
    smoke_common.single_rank_env()
    argv = [
        "--module=torchtitan.experiments.tpu.post_training.tests.cpu_recipes",
        f"--config={config_name}", f"--hf_assets_path={assets}",
        f"--dataloader.data-path={data}", f"--dataloader.hf-tokenizer-path={template_dir}",
        f"--dump_folder={dump}", *(extra or []),
    ]
    sft_train.start_trainer(ConfigManager().parse_args(argv))
    return json.loads((dump / "sft-records-rank0.json").read_text())


def dcp_model_state(checkpoint_dir: pathlib.Path) -> dict[str, torch.Tensor]:
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    out = checkpoint_dir.parent / f"{checkpoint_dir.name}.pt"
    dcp_to_torch_save(str(checkpoint_dir), str(out))
    state = torch.load(out, weights_only=False)
    out.unlink()
    return {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}


def check_adapter_export(name: str, kind: str, adapter_dir: pathlib.Path) -> dict[str, bool]:
    """The PEFT adapter written by a LoRA run with adapter saves."""
    from safetensors.torch import load_file

    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not (config_path.is_file() and weights_path.is_file()):
        return {f"{name}: PEFT adapter written at the last step": False}
    config = json.loads(config_path.read_text())
    tensors = load_file(str(weights_path))
    print(f"[SFT] {name} adapter: {len(tensors)} tensors, r={config['r']}, "
          f"alpha={config['lora_alpha']}, modules={config['target_modules']}", flush=True)
    return {
        f"{name}: PEFT adapter written at the last step": True,
        f"{name}: adapter config r 8 / alpha 16": (config["r"], config["lora_alpha"]) == (8, 16.0),
        f"{name}: adapter holds q/k/v/o A and B of every attention layer": (
            set(tensors) == EXPECTED_ADAPTER_KEYS[kind]),
        f"{name}: adapter lora_B trained": all(
            float(v.abs().sum()) > 0 for k, v in tensors.items() if ".lora_B." in k),
    }


def expected_base_state(kind: str, assets: pathlib.Path,
                        initial: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The model-keyed base weights the run started from."""
    if kind == "qwen_debug":
        return smoke_common.debug_base_state(initial)
    from safetensors.torch import load_file

    from torchtitan.experiments.tpu.muse_spark import model_registry

    spec = model_registry("tiny-hf", seq_len=512, tpu_moe_kernels=False)
    hf = {}
    for path in sorted(assets.glob("*.safetensors")):
        hf.update(load_file(str(path)))
    return spec.state_dict_adapter(spec.model, str(assets)).from_hf(hf)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--spark-checkpoint", default="",
                    help="public tiny Muse Spark snapshot (weights + tokenizer); "
                         "adds the Muse Spark runs when the Muse Spark package is installed")
    args = ap.parse_args()
    work = pathlib.Path(args.work)
    if work.exists():
        shutil.rmtree(work)
    tokenizer_dir = pathlib.Path(args.tokenizer_dir)
    qwen_assets, data = work / "assets", work / "sft.jsonl"
    qwen_initial = smoke_common.make_debug_checkpoint(qwen_assets, tokenizer_dir)
    write_conversations(data)
    # (config, model kind, assets, minimum relative loss drop, LoRA, lora_save)
    runs = [
        ("sft_debug_smoke", "qwen_debug", qwen_assets, 0.3, False, ""),
        ("sft_debug_lora_smoke", "qwen_debug", qwen_assets, 0.3, True, "full"),
        ("sft_debug_lora_adapter_smoke", "qwen_debug", qwen_assets, 0.3, True, "adapter"),
        ("sft_debug_adamw8bit_smoke", "qwen_debug", qwen_assets, 0.3, False, ""),
        ("sft_debug_bf16_smoke", "qwen_debug", qwen_assets, 0.3, False, ""),
    ]
    if args.spark_checkpoint and not smoke_common.has_muse_spark():
        print("[SFT] --spark-checkpoint given but the Muse Spark package is not installed: "
              "skipping the Muse Spark runs", flush=True)
    elif args.spark_checkpoint:
        # The public tiny checkpoint starts near uniform over 202048 tokens
        # (|logit| ~ 0.02 behind the 0.177 output multiplier), so its loss
        # moves slowly: require a steady decrease rather than a large drop.
        spark = pathlib.Path(args.spark_checkpoint)
        runs += [
            ("sft_spark_tiny_smoke", "spark_tiny", spark, 0.01, False, ""),
            ("sft_spark_tiny_lora_smoke", "spark_tiny", spark, 0.0, True, "adapter"),
        ]
    resume_state = smoke_common.has_lora_checkpoint()
    if not resume_state:
        print("[SFT] checkpointing.lora_checkpoint is not installed: the LoRA runs with "
              "adapter saves write the PEFT adapter only (sft.lora_save=peft)", flush=True)
    checks: dict[str, bool] = {}
    for name, kind, assets, drop, lora, lora_save in runs:
        extra = []
        if lora_save == "adapter" and not resume_state:
            lora_save, extra = "peft", ["--sft.lora-save=peft"]
        # The chat template comes from the tokenizer directory (the debug
        # checkpoint has none, see smoke_common.TOKENIZER_FILES) or the snapshot.
        template_dir = tokenizer_dir if kind == "qwen_debug" else assets
        records = run(name, work / name, assets, data, template_dir, extra)
        steps = records["steps"]
        losses = [s["loss_local"] for s in steps]
        print(f"[SFT] {name} losses {[round(x, 4) for x in losses]} parameters "
              f"{records['parameters']}", flush=True)
        checks[f"{name}: 8 steps"] = [s["step"] for s in steps] == list(range(1, 9))
        checks[f"{name}: finite loss"] = all(torch.isfinite(torch.tensor(losses)))
        if drop >= 0.1:
            # Fast-learning runs overfit and oscillate at the end: compare the
            # mean of the last two steps with the first.
            checks[f"{name}: loss falls by > {drop:.0%}"] = (
                sum(losses[-2:]) / 2 < (1.0 - drop) * losses[0])
        else:
            # Each step's loss is measured on a different packed batch. For the
            # 7k-parameter tiny LoRA the late per-step gain (~2e-4) is below the
            # batch-to-batch variation, so allow one step that does not fall.
            falls = sum(b < a for a, b in zip(losses, losses[1:]))
            checks[f"{name}: loss falls overall and at all but at most one step"] = (
                losses[-1] < losses[0] and falls >= len(losses) - 2)
            if drop > 0:
                checks[f"{name}: loss falls by > {drop:.0%}"] = losses[-1] < (1.0 - drop) * losses[0]
        checks[f"{name}: supervised tokens every step"] = all(
            s["valid_tokens_local"] > 0 and s["documents"] > 0 for s in steps)
        checkpoint = work / name / "dump" / "checkpoint" / "step-8"
        adapter_dir = work / name / "dump" / "lora_adapter" / "step-8"
        if lora and lora_save == "adapter":
            # Adapter-only resume state: adapters + optimizer + LR + dataloader + step, no base.
            checks[f"{name}: DCP checkpoint at the last step"] = checkpoint.is_dir()
            state = dcp_model_state(checkpoint) if checkpoint.is_dir() else {}
            model_keys = [k for k in state if not k.startswith(
                ("optimizer", "lr_scheduler", "dataloader", "train_state"))]
            checks[f"{name}: checkpoint holds only the adapters (adapter-only saves)"] = bool(
                model_keys) and all(k.endswith(("lora_a.weight", "lora_b.weight")) for k in model_keys)
            checks.update(check_adapter_export(name, kind, adapter_dir))
        elif lora and lora_save == "peft":
            checks[f"{name}: no DCP checkpoint (PEFT saves only)"] = not checkpoint.exists()
            checks.update(check_adapter_export(name, kind, adapter_dir))
        else:
            checks[f"{name}: DCP checkpoint at the last step"] = checkpoint.is_dir()
        if not lora:
            continue
        params = records["parameters"]
        checks[f"{name}: only adapters train"] = (
            params["trainable"] == params["adapter"] > 0 and params["frozen"] > 0)
        if checkpoint.is_dir() and lora_save == "full":
            state = dcp_model_state(checkpoint)
            expected = expected_base_state(kind, assets, qwen_initial)
            base = {k: v for k, v in state.items() if k in expected}
            adapters = [k for k in state if k.endswith(("lora_a.weight", "lora_b.weight"))]
            print(f"[SFT] {name} checkpoint: {len(base)} base tensors, "
                  f"{len(adapters)} adapter tensors", flush=True)
            checks[f"{name}: checkpoint holds every base weight and the adapters"] = (
                len(base) == len(expected) and len(adapters) > 0)
            checks[f"{name}: base weights bit-identical to the HF checkpoint"] = all(
                torch.equal(base[k].to(expected[k].dtype), expected[k]) for k in base)
            checks[f"{name}: adapters trained (lora_b nonzero)"] = all(
                float(state[k].abs().sum()) > 0 for k in adapters if k.endswith("lora_b.weight"))
    return smoke_common.report(checks, "SFT")


if __name__ == "__main__":
    sys.exit(main())
