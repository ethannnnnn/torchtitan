# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""LoRA SFT kill-and-resume check on CPU through the real trainer. ASCII only.

Three runs of the same LoRA SFT recipe, each a fresh ``torchrun`` (gloo, one
or two ranks), on the public tiny Muse Spark checkpoint:

* ``ref``: ``--steps`` steps, a checkpoint every ``--interval`` steps;
* ``crash``: the same in a new dump folder, killed with SIGKILL on every rank
  inside step ``--kill-step`` (``SFT_FAULT_KILL_STEP``, after that step's
  batches were drawn);
* ``resume``: relaunched with the crash run's dump folder: base weights from
  the Hugging Face checkpoint, adapters + optimizer + LR schedule + dataloader
  + step from the newest complete ``checkpoint/step-N``.

With ``--partial`` a fourth run copies the crash folder, plants an incomplete
newer checkpoint (``step-M`` without DCP ``.metadata``, what a kill during a
save leaves), and must resume from the complete one.

Checks (every rank, every step; ``SFT_RECORDS_ALL_RANKS`` records): the crash
run matches ``ref`` before the kill; the resumed run starts at the step after
the checkpoint and matches ``ref`` bit for bit afterwards (loss share,
gradient norm, learning rate, the SHA-256 of every microbatch and its
document ids); the final PEFT adapters are identical; the checkpoint holds
only adapter tensors plus training state.

Start it from the repository root, with the TorchTPU CPU stand-in
(``torchtitan/experiments/tpu/tests/cpu_stubs``) ahead of the repository root
on ``PYTHONPATH`` and ``TORCH_DEVICE_BACKEND_AUTOLOAD=0`` in the environment,
which is how ``run_cpu.sh`` launches it::

    python -m torchtitan.experiments.tpu.spark_e2e.tests.cpu_sft_resume \\
        --model <tiny-muse-spark> --work <new dir> [--data <chat jsonl>] [--nproc 2] [--ac] [--partial]

Without ``--data`` the runs train on short generated chat records
(``cpu_sft_smoke.write_conversations``).

Prints ``[RESUME] PASS`` and writes ``<work>/resume_summary.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ENTRIES = {
    # entry: (python module, config module, config, extra CLI)
    "e2e": ("torchtitan.experiments.tpu.spark_e2e.train",
            "torchtitan.experiments.tpu.spark_e2e.recipes", "s0_sft_tiny", []),
}


def run(args, name: str, dump: Path, *, kill_step: int = 0) -> dict:
    module, config_module, config, extra = ENTRIES[args.entry]
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={args.nproc}",
           "-m", module, f"--module={config_module}", f"--config={config}",
           f"--hf_assets_path={args.model}", f"--dataloader.data-path={args.data}",
           f"--dump_folder={dump}", f"--training.steps={args.steps}",
           f"--checkpoint.interval={args.interval}",
           # Multi-threaded CPU reductions are not bitwise reproducible from run
           # to run; deterministic algorithms make two fresh runs identical.
           "--debug.deterministic", *extra]
    env = {k: v for k, v in os.environ.items() if not k.startswith("SPARK_")}
    env.update(SFT_RECORDS_ALL_RANKS="1", SPARK_LR=str(args.lr), SPARK_AC="1" if args.ac else "0")
    env.setdefault("OMP_NUM_THREADS", "8")
    env.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    env.pop("SFT_FAULT_KILL_STEP", None)
    if kill_step:
        env["SFT_FAULT_KILL_STEP"] = str(kill_step)
    log = Path(args.work) / f"{name}.log"
    with log.open("w") as handle:
        rc = subprocess.run(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT, cwd=args.repo).returncode
    return {"name": name, "rc": rc, "log": str(log), "records": read_records(dump)}


def read_records(dump: Path) -> dict[int, dict[int, dict]]:
    """{rank: {step: record}}; a later line for the same step wins."""
    out: dict[int, dict[int, dict]] = {}
    root = dump / "sft-records"
    for path in sorted(root.glob("rank-*.jsonl")) if root.is_dir() else []:
        rank = int(path.stem.split("-")[1])
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                out.setdefault(rank, {})[int(rec["step"])] = rec
    return out


FIELDS = ("loss_local", "grad_norm", "lr", "batches", "epoch", "documents", "tokens", "valid_tokens_local")


def _digest_and_docs(entry: dict) -> tuple:
    return entry.get("sha256"), entry.get("docs")


def _batch_draws(record: dict) -> list[tuple]:
    """(sha256, docs) of each microbatch a step drew, in order; microbatch numbers restart on resume."""
    return list(map(_digest_and_docs, record.get("batches", [])))


def compare(a: dict, b: dict, steps: range) -> list[str]:
    diffs = []
    if sorted(a) != sorted(b):
        return [f"ranks differ: {sorted(a)} vs {sorted(b)}"]
    for rank in sorted(a):
        for step in steps:
            ra, rb = a[rank].get(step), b[rank].get(step)
            if ra is None or rb is None:
                diffs.append(f"rank {rank} step {step}: missing ({ra is None}, {rb is None})")
                continue
            for field in FIELDS:
                va, vb = (_batch_draws(ra), _batch_draws(rb)) if field == "batches" else (ra.get(field), rb.get(field))
                if va != vb:
                    diffs.append(f"rank {rank} step {step} {field}: {va!r} != {vb!r}")
    return diffs


def checkpoint_keys(step_dir: Path) -> list[str]:
    from torch.distributed.checkpoint import FileSystemReader

    return sorted(FileSystemReader(str(step_dir)).read_metadata().state_dict_metadata)


def adapter_tensors(dump: Path, step: int, entry: str) -> dict:
    from safetensors.torch import load_file

    return load_file(str(dump / "adapter" / f"step-{step}" / "adapter_model.safetensors"))


def _resume_arguments() -> argparse.Namespace:
    resume_parser = argparse.ArgumentParser(
        description="Kill a CPU LoRA SFT run mid-training, resume it and compare it with an uninterrupted run.")
    resume_parser.add_argument("--model", required=True, help="directory of the tiny Muse Spark checkpoint")
    resume_parser.add_argument("--data", default="", help="chat records as JSONL (default: generate short ones)")
    resume_parser.add_argument("--work", required=True, help="working directory, removed first if it exists")
    resume_parser.add_argument("--entry", choices=sorted(ENTRIES), default="e2e", help="training entry to run")
    resume_parser.add_argument("--nproc", type=int, default=1, help="ranks per torchrun launch")
    resume_parser.add_argument("--steps", type=int, default=6, help="steps of the reference and resumed runs")
    resume_parser.add_argument("--interval", type=int, default=3, help="checkpoint every this many steps")
    resume_parser.add_argument("--kill-step", type=int, default=5, help="step during which the crash run dies")
    resume_parser.add_argument("--lr", type=float, default=0.01, help="learning rate of every run")
    resume_parser.add_argument("--ac", action="store_true", help="turn on full activation checkpointing")
    resume_parser.add_argument("--partial", action="store_true",
                               help="also resume past an incomplete newer checkpoint")
    parsed = resume_parser.parse_args()
    if not parsed.interval < parsed.kill_step <= parsed.steps:
        resume_parser.error("the steps must satisfy --interval < --kill-step <= --steps, so that a checkpoint "
                            "exists before the kill and the reference run gets past it")
    return parsed


def _fresh_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def _print_checks(checks: dict[str, bool], failed: list[str]) -> None:
    for name in checks:
        marker = "FAIL" if name in failed else "ok"
        print(f"[RESUME] {marker:<4} {name}")


def main() -> int:
    args = _resume_arguments()
    import torch  # noqa: F401  (without torch this fails here, before any run)

    args.repo = str(Path(__file__).resolve().parents[5])
    work = _fresh_dir(Path(args.work))
    if not args.data:
        from torchtitan.experiments.tpu.spark_e2e.tests.cpu_sft_smoke import write_conversations

        args.data = str(work / "conversations.jsonl")
        write_conversations(Path(args.data), 48)
    ckpt_step = (args.kill_step - 1) // args.interval * args.interval

    ref = run(args, "ref", work / "ref")
    crash = run(args, "crash", work / "crash", kill_step=args.kill_step)
    crash_records = read_records(work / "crash")  # before the resume appends to the same files
    resume = run(args, "resume", work / "crash")
    summary: dict = {"args": {k: v for k, v in vars(args).items()}, "checkpoint_step": ckpt_step,
                     "rc": {"ref": ref["rc"], "crash": crash["rc"], "resume": resume["rc"]}, "checks": {}}
    checks = summary["checks"]
    checks["ref run exit 0"] = ref["rc"] == 0
    checks["crash run killed (rc != 0)"] = crash["rc"] != 0
    resume_log = Path(resume["log"]).read_text()
    checks[f"resume restored adapter-only state of step {ckpt_step}"] = (
        f"resumed adapter-only state of step {ckpt_step}" in resume_log)
    checks["resume reloaded the base from Hugging Face"] = "[ckpt] base weights loaded from" in resume_log
    checks["resume run exit 0"] = resume["rc"] == 0
    ranks = sorted(ref["records"])
    checks[f"records from {args.nproc} rank(s)"] = len(ranks) == args.nproc
    killed_steps = sorted({s for r in crash_records.values() for s in r})
    checks[f"crash run stopped before finishing step {args.kill_step}"] = (
        bool(killed_steps) and max(killed_steps) == args.kill_step - 1)
    pre = compare(ref["records"], crash_records, range(1, args.kill_step))
    checks["crash run == ref before the kill (bitwise)"] = not pre
    post = compare(ref["records"], resume["records"], range(ckpt_step + 1, args.steps + 1))
    checks[f"resumed steps {ckpt_step + 1}..{args.steps} == ref (bitwise: loss, grad norm, lr, batches)"] = not post
    a_ref = adapter_tensors(work / "ref", args.steps, args.entry)
    a_res = adapter_tensors(work / "crash", args.steps, args.entry)
    checks["final PEFT adapter identical"] = (
        sorted(a_ref) == sorted(a_res) and all(bool((a_ref[k] == a_res[k]).all()) for k in a_ref))
    keys = checkpoint_keys(work / "crash" / "checkpoint" / f"step-{ckpt_step}")
    model_keys = [k for k in keys if not k.startswith(("optimizer", "lr_scheduler", "dataloader", "train_state"))]
    checks["checkpoint model keys are LoRA tensors only"] = bool(model_keys) and all(
        ".lora_a." in k or ".lora_b." in k for k in model_keys)
    checks["checkpoint has optimizer, lr_scheduler, dataloader, train_state"] = all(
        any(k.startswith(p) for k in keys) for p in ("optimizer", "lr_scheduler", "dataloader", "train_state"))
    nbytes = sum(p.stat().st_size for p in (work / "crash" / "checkpoint" / f"step-{ckpt_step}").rglob("*"))
    summary["checkpoint"] = {"step": ckpt_step, "keys": len(keys), "model_keys": len(model_keys), "bytes": nbytes}
    summary["diffs"] = {"pre_kill": pre[:20], "post_resume": post[:20]}
    summary["losses"] = {
        name: {str(s): rec["records"][ranks[0]][s].get("loss_local") for s in sorted(rec["records"].get(ranks[0], {}))}
        for name, rec in (("ref", ref), ("resume", resume))} if ranks else {}

    if args.partial:
        part = work / "partial"
        shutil.copytree(work / "crash", part, ignore=shutil.ignore_patterns("sft-records", "adapter", "lora_adapter"))
        newest = max(int(p.name.split("-")[1]) for p in (part / "checkpoint").glob("step-*"))
        planted = part / "checkpoint" / f"step-{newest + 1}"
        shutil.copytree(part / "checkpoint" / f"step-{ckpt_step}", planted)
        (planted / ".metadata").unlink()
        # Drop the completed run's later checkpoints: the incomplete one is the newest.
        for p in (part / "checkpoint").glob("step-*"):
            if int(p.name.split("-")[1]) not in (ckpt_step, newest + 1):
                shutil.rmtree(p)
        partial = run(args, "partial", part)
        plog = Path(partial["log"]).read_text()
        checks[f"partial: incomplete step-{newest + 1} skipped, resumed step {ckpt_step}"] = (
            f"resumed adapter-only state of step {ckpt_step}" in plog and partial["rc"] == 0)
        pdiff = compare(ref["records"], partial["records"], range(ckpt_step + 1, args.steps + 1))
        checks["partial: resumed steps == ref (bitwise)"] = not pdiff
        summary["diffs"]["partial"] = pdiff[:20]

    failed = [name for name in checks if not checks[name]]
    summary["pass"] = not failed
    with open(work / "resume_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    _print_checks(checks, failed)
    for kind in summary["diffs"]:
        for diff in summary["diffs"][kind][:5]:
            print(f"[RESUME] diff {kind}: {diff}")
    verdict = "FAIL" if failed else "PASS"
    print(f"[RESUME] {verdict} entry={args.entry} nproc={args.nproc} ac={args.ac} ckpt_step={ckpt_step} "
          f"checkpoint={summary['checkpoint']}")
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
