# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""End-to-end LoRA SFT on CPU through the real trainer and the public tiny checkpoint. ASCII only.

Runs ``spark_e2e.train`` with the ``s0_sft_tiny`` recipe under ``torchrun``
(gloo, one or two ranks) on short generated chat records, the way the TPU
stages run (HF initial load, LoRA on q/k/v/o, adapter-only checkpoints, PEFT
exports, held-out loss):

* run A: ``--steps`` steps, ``--checkpoint.interval=2``,
  ``SPARK_EXPORT_EVERY=3`` and a held-out set (``SPARK_VAL_DATA``,
  ``SPARK_VAL_EVERY=2``);
* run B: a fresh run that starts from run A's last PEFT adapter
  (``SPARK_ADAPTER_INIT``) with a learning rate of 0, so its own export must
  equal that adapter bit for bit.

Checks: both runs exit 0; PEFT exports at every checkpoint step, every
``SPARK_EXPORT_EVERY`` steps and the last step, in the PEFT layout (HF module
names, rank 16 / alpha 32, an anchored ``target_modules`` regex); LoRA-only
DCP checkpoints (adapter tensors plus optimizer, LR schedule, dataloader and
train state, under 2 MB); held-out losses before the first step, every two
steps and after the last; finite training losses; run B restores run A's
adapter exactly.

    PYTHONPATH=<repo>/torchtitan/experiments/tpu/spark_e2e/tests/tpu_import_stubs:<repo> \\
      TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m torchtitan.experiments.tpu.spark_e2e.tests.cpu_sft_smoke \\
        --model <tiny-muse-spark> --work <new dir> [--nproc 2]

Prints ``[SMOKE] PASS`` and writes ``<work>/smoke_summary.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ENTRY = "torchtitan.experiments.tpu.spark_e2e.train"
RECIPES = "torchtitan.experiments.tpu.spark_e2e.recipes"
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
COLORS = ("red", "blue", "green", "black", "white", "silver")
ITEMS = ("mug", "lamp", "chair", "kettle", "scarf", "backpack", "notebook", "speaker")


def write_conversations(path: Path, n: int = 32, seed: int = 0) -> None:
    """Short, deterministic chat records with one assistant reply each."""
    rows = []
    for i in range(n):
        color, item = COLORS[(i + seed) % len(COLORS)], ITEMS[(3 * i + seed) % len(ITEMS)]
        rows.append({"messages": [
            {"role": "system", "content": "Answer with one short sentence."},
            {"role": "user", "content": f"Describe a {color} {item} in a few words. Request {i}."},
            {"role": "assistant", "content": f"A {color} {item} that looks simple and works well."},
        ]})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def run_trainer(repo: str, nproc: int, model: str, data: Path, dump: Path, log: Path,
                extra: list[str], env: dict[str, str]) -> int:
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={nproc}",
           "-m", ENTRY, f"--module={RECIPES}", "--config=s0_sft_tiny", f"--hf_assets_path={model}",
           f"--dataloader.data-path={data}", f"--dump_folder={dump}", *extra]
    run_env = {k: v for k, v in os.environ.items() if not k.startswith("SPARK_")}
    run_env.update(env)
    run_env.setdefault("OMP_NUM_THREADS", "8")
    run_env.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    with log.open("w") as handle:
        return subprocess.run(cmd, env=run_env, stdout=handle, stderr=subprocess.STDOUT, cwd=repo).returncode


def adapter_steps(dump: Path) -> list[int]:
    root = dump / "adapter"
    return sorted(int(p.name.split("-")[1]) for p in root.glob("step-*")) if root.is_dir() else []


def checkpoint_keys(step_dir: Path) -> list[str]:
    from torch.distributed.checkpoint import FileSystemReader

    return sorted(FileSystemReader(str(step_dir)).read_metadata().state_dict_metadata)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="tiny Muse Spark checkpoint directory")
    ap.add_argument("--work", required=True, help="new working directory")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--steps", type=int, default=4)
    args = ap.parse_args()
    if args.steps < 4:
        ap.error("--steps must be at least 4 (exports at 2, 3 and the last step)")
    import torch
    from safetensors.torch import load_file

    repo = str(Path(__file__).resolve().parents[5])
    work = Path(args.work)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    train, val = work / "data" / "train.jsonl", work / "data" / "val.jsonl"
    write_conversations(train, 32)
    write_conversations(val, 8, seed=5)
    checks: dict[str, bool] = {}
    summary: dict = {"args": vars(args), "checks": checks}

    # ---- run A ----------------------------------------------------------
    dump_a = work / "a"
    rc_a = run_trainer(repo, args.nproc, args.model, train, dump_a, work / "a.log", ["--checkpoint.interval=2"], {
        "SPARK_STEPS": str(args.steps), "SPARK_LR": "1e-2", "SPARK_EXPORT_EVERY": "3",
        "SPARK_VAL_DATA": str(val), "SPARK_VAL_EVERY": "2"})
    checks["run A exit 0"] = rc_a == 0
    expected_exports = sorted({s for s in range(1, args.steps + 1) if s % 2 == 0 or s % 3 == 0} | {args.steps})
    summary["exports_a"] = adapter_steps(dump_a)
    checks[f"PEFT exports at steps {expected_exports}"] = summary["exports_a"] == expected_exports
    last = dump_a / "adapter" / f"step-{args.steps}"
    tensors = load_file(str(last / "adapter_model.safetensors")) if (last / "adapter_model.safetensors").is_file() else {}
    config = json.loads((last / "adapter_config.json").read_text()) if (last / "adapter_config.json").is_file() else {}
    model_config = json.loads((Path(args.model) / "config.json").read_text())
    layers = model_config.get("text_config", model_config)["num_hidden_layers"]
    names = {f"base_model.model.model.language_model.layers.{i}.self_attn.{p}.lora_{ab}.weight"
             for i in range(layers) for p in PROJECTIONS for ab in "AB"}
    checks["PEFT tensors: q/k/v/o of every layer, HF names, float32"] = (
        set(tensors) == names and all(t.dtype == torch.float32 for t in tensors.values()))
    checks["PEFT config: r 16, alpha 32, anchored target_modules"] = (
        config.get("r") == 16 and config.get("lora_alpha") == 32.0
        and bool(re.fullmatch(config.get("target_modules", "^$"), "model.language_model.layers.0.self_attn.q_proj"))
        and not re.fullmatch(config.get("target_modules", "^$"), "model.vision_tower.layers.0.self_attn.q_proj"))
    ckpt_steps = sorted(int(p.name.split("-")[1]) for p in (dump_a / "checkpoint").glob("step-*")) \
        if (dump_a / "checkpoint").is_dir() else []
    summary["checkpoints_a"] = ckpt_steps
    checks["LoRA-only checkpoints at every interval and the last step"] = ckpt_steps == sorted(
        {s for s in range(2, args.steps + 1, 2)} | {args.steps})
    if ckpt_steps:
        step_dir = dump_a / "checkpoint" / f"step-{ckpt_steps[-1]}"
        keys = checkpoint_keys(step_dir)
        model_keys = [k for k in keys if not k.startswith(("optimizer", "lr_scheduler", "dataloader", "train_state"))]
        nbytes = sum(p.stat().st_size for p in step_dir.rglob("*") if p.is_file())
        summary["checkpoint"] = {"keys": len(keys), "model_keys": len(model_keys), "bytes": nbytes}
        checks["checkpoint model keys are adapter tensors only"] = bool(model_keys) and all(
            ".lora_a." in k or ".lora_b." in k for k in model_keys)
        checks["checkpoint holds optimizer, lr_scheduler, dataloader, train_state"] = all(
            any(k.startswith(p) for k in keys) for p in ("optimizer", "lr_scheduler", "dataloader", "train_state"))
        checks["checkpoint under 2 MB"] = nbytes < 2 * 1024**2
    val_path = dump_a / "sft-val-rank0.json"
    evals = json.loads(val_path.read_text())["evals"] if val_path.is_file() else []
    summary["val_a"] = evals
    expected_val = sorted({0, args.steps} | set(range(2, args.steps + 1, 2)))
    checks[f"held-out loss at steps {expected_val}"] = (
        [e["step"] for e in evals] == expected_val and all(math.isfinite(e["loss"]) for e in evals))
    records_path = dump_a / "sft-records-rank0.json"
    steps = json.loads(records_path.read_text())["steps"] if records_path.is_file() else []
    summary["train_a"] = steps
    checks["finite training loss every step"] = (
        [r["step"] for r in steps] == list(range(1, args.steps + 1))
        and all(math.isfinite(r.get("loss_local", float("nan"))) for r in steps))

    # ---- run B: start from run A's adapter, learning rate 0 --------------
    dump_b = work / "b"
    rc_b = run_trainer(repo, args.nproc, args.model, train, dump_b, work / "b.log", [], {
        "SPARK_STEPS": "1", "SPARK_LR": "0", "SPARK_ADAPTER_INIT": str(last)})
    checks["run B exit 0"] = rc_b == 0
    log_b = (work / "b.log").read_text()
    checks["run B logged the adapter init"] = "adapter initialized from" in log_b
    restored = dump_b / "adapter" / "step-1" / "adapter_model.safetensors"
    tensors_b = load_file(str(restored)) if restored.is_file() else {}
    checks["run B export == run A's adapter (bitwise)"] = bool(tensors) and sorted(tensors_b) == sorted(tensors) and all(
        torch.equal(tensors[k], tensors_b[k]) for k in tensors)

    summary["pass"] = all(checks.values())
    (work / "smoke_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    for name, ok in checks.items():
        print(f"[SMOKE] {'ok  ' if ok else 'FAIL'} {name}")
    print(f"[SMOKE] {'PASS' if summary['pass'] else 'FAIL'} nproc={args.nproc} steps={args.steps}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
