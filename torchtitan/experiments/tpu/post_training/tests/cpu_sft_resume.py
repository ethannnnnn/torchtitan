# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LoRA SFT kill-and-resume check on CPU through the real trainer.

Three runs of the same LoRA SFT recipe (``sft.lora_save=adapter``), each a
fresh ``torchrun`` (gloo, one or two ranks):

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

The default model is the TPU Qwen3.5 debug model of ``cpu_recipes`` (random
weights written next to the runs); ``--spark-checkpoint`` uses the Muse Spark
port on the public tiny checkpoint instead (needs the Muse Spark TPU package).
Adapter-only resume needs ``checkpointing/lora_checkpoint.py``.

    python -m torchtitan.experiments.tpu.post_training.tests.cpu_sft_resume \\
        --tokenizer-dir <Muse Spark tokenizer dir> --work <new dir> \\
        [--spark-checkpoint <tiny Muse Spark snapshot>] [--nproc 2] [--partial]

Prints ``[RESUME] PASS`` and writes ``<work>/resume_summary.json``.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(args, name: str, dump: Path, *, kill_step: int = 0) -> dict:
  cmd = [
      sys.executable,
      "-m",
      "torch.distributed.run",
      "--standalone",
      f"--nproc-per-node={args.nproc}",
      "-m",
      "torchtitan.experiments.tpu.post_training.sft_train",
      "--module=torchtitan.experiments.tpu.post_training.tests.cpu_recipes",
      f"--config={args.config}",
      f"--hf_assets_path={args.model}",
      f"--dataloader.data-path={args.data}",
      f"--dataloader.hf-tokenizer-path={args.template_dir}",
      f"--dump_folder={dump}",
      f"--training.steps={args.steps}",
      f"--checkpoint.interval={args.interval}",
      # Multi-threaded CPU reductions are not bitwise reproducible from run
      # to run; deterministic algorithms make two fresh runs identical.
      "--debug.deterministic",
  ]
  env = dict(os.environ, SFT_RECORDS_ALL_RANKS="1")
  env.setdefault("OMP_NUM_THREADS", "8")
  env.pop("SFT_FAULT_KILL_STEP", None)
  env.pop("SFT_RECORDS_DIR", None)
  if kill_step:
    env["SFT_FAULT_KILL_STEP"] = str(kill_step)
  log = Path(args.work) / f"{name}.log"
  with log.open("w") as handle:
    rc = subprocess.run(
        cmd, env=env, stdout=handle, stderr=subprocess.STDOUT, cwd=args.repo
    ).returncode
  return {
      "name": name,
      "rc": rc,
      "log": str(log),
      "records": read_records(dump),
  }


def read_records(dump) -> dict[int, dict[int, dict]]:
  """``{rank: {step: record}}`` from ``<dump>/sft-records/rank-*.jsonl``.

  Blank lines are skipped, a later line for a step replaces an earlier one,
  and a file without records adds no rank.
  """
  import collections
  import glob

  pattern = os.path.join(dump, "sft-records", "rank-*.jsonl")
  by_rank = collections.defaultdict(dict)
  for name in sorted(glob.glob(pattern)):
    rank = int(os.path.basename(name)[len("rank-") : -len(".jsonl")])
    with open(name, encoding="utf-8") as lines:
      for text in lines:
        if text.strip():
          row = json.loads(text)
          by_rank[rank][row["step"]] = row
  return dict(by_rank)


FIELDS = (
    "loss_local",
    "grad_norm",
    "lr",
    "batches",
    "epoch",
    "documents",
    "tokens",
    "valid_tokens_local",
)


def _batch_identity(record: dict) -> list[tuple]:
  """``(sha256, docs)`` of every microbatch a per-rank record lists.

  ``compare`` checks these instead of whole entries: the per-process
  microbatch counter of an entry restarts after a resume.
  """
  return list(
      map(
          lambda entry: (entry.get("sha256"), entry.get("docs")),
          record.get("batches", []),
      )
  )


def _step_differences(rank: int, step: int, mine: dict, theirs: dict) -> list:
  """Difference lines of one rank and step (``compare``'s format)."""
  absent = (step not in mine, step not in theirs)
  if any(absent):
    return [f"rank {rank} step {step}: missing {absent}"]
  left, right = mine[step], theirs[step]
  lines = []
  for name in FIELDS:
    if name == "batches":
      a, b = _batch_identity(left), _batch_identity(right)
    else:
      a, b = left.get(name), right.get(name)
    if a != b:
      lines.append(f"rank {rank} step {step} {name}: {a!r} != {b!r}")
  return lines


def compare(a: dict, b: dict, steps) -> list[str]:
  """Every difference between two ``read_records`` results over ``steps``."""
  import itertools

  if set(a) != set(b):
    return [f"ranks differ: {sorted(a)} vs {sorted(b)}"]
  found = []
  for rank, step in itertools.product(sorted(a), steps):
    found += _step_differences(rank, step, a[rank], b[rank])
  return found


def checkpoint_keys(step_dir: Path) -> list[str]:
  from torch.distributed.checkpoint import FileSystemReader

  return sorted(
      FileSystemReader(str(step_dir)).read_metadata().state_dict_metadata
  )


def adapter_tensors(dump: Path, step: int) -> dict:
  from safetensors.torch import load_file

  return load_file(
      str(dump / "lora_adapter" / f"step-{step}" / "adapter_model.safetensors")
  )


def _recipe_microbatches(config_name: str) -> int:
  """``dataloader.microbatches_per_step`` of a ``cpu_recipes`` recipe."""
  recipes = importlib.import_module(
      "torchtitan.experiments.tpu.post_training.tests.cpu_recipes"
  )
  build = vars(recipes)[config_name]
  return int(build().dataloader.microbatches_per_step)


def _parse_cli() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Kill a LoRA SFT run mid-step, resume it from its newest complete "
          "adapter-only checkpoint and compare it with an uninterrupted run."
      )
  )
  parser.add_argument(
      "--tokenizer-dir",
      required=True,
      help="Muse Spark tokenizer folder, chat template included",
  )
  parser.add_argument(
      "--work", required=True, help="output folder; emptied before the runs"
  )
  parser.add_argument(
      "--spark-checkpoint",
      default="",
      help="tiny Muse Spark snapshot (default: the Qwen3.5 debug model)",
  )
  parser.add_argument(
      "--nproc", type=int, default=1, help="ranks of every torchrun"
  )
  parser.add_argument(
      "--steps", type=int, default=6, help="optimizer steps of a full run"
  )
  parser.add_argument(
      "--interval", type=int, default=3, help="steps between checkpoints"
  )
  parser.add_argument(
      "--kill-step",
      type=int,
      default=5,
      help="step during which the crash run is killed",
  )
  parser.add_argument(
      "--partial",
      action="store_true",
      help="also resume next to a planted, incomplete newer checkpoint",
  )
  args = parser.parse_args()
  if not args.interval < args.kill_step <= args.steps:
    parser.error("the runs need --interval < --kill-step <= --steps")
  return args


def _checkout_root() -> str:
  """The repository root, five folders above the folder of this file."""
  return os.path.realpath(
      os.path.join(os.path.dirname(__file__), *[os.pardir] * 5)
  )


def _emptied(folder: str) -> Path:
  """``folder`` re-created empty; anything it held before is deleted."""
  with contextlib.suppress(FileNotFoundError):
    shutil.rmtree(folder)
  os.makedirs(folder)
  return Path(folder)


def main() -> int:
  args = _parse_cli()
  # torch first: if it is missing, the check stops before touching any folder.
  import torch  # noqa: F401

  from torchtitan.experiments.tpu.post_training.tests import (
      cpu_sft_smoke,
      smoke_common,
  )

  args.repo = _checkout_root()
  work = _emptied(os.path.abspath(args.work))
  args.data = str(work / "conversations.jsonl")
  cpu_sft_smoke.write_conversations(Path(args.data))
  if args.spark_checkpoint:
    args.config, args.model = "sft_spark_tiny_lora_smoke", args.spark_checkpoint
    args.template_dir = args.spark_checkpoint
  else:
    args.config, args.model = "sft_debug_lora_adapter_smoke", str(
        work / "assets"
    )
    args.template_dir = args.tokenizer_dir
    smoke_common.make_debug_checkpoint(
        Path(args.model), Path(args.tokenizer_dir)
    )
  ckpt_step = (args.kill_step - 1) // args.interval * args.interval

  ref = run(args, "ref", work / "ref")
  crash = run(args, "crash", work / "crash", kill_step=args.kill_step)
  crash_records = read_records(
      work / "crash"
  )  # before the resume appends to the same files
  resume = run(args, "resume", work / "crash")
  summary: dict = {
      "args": {k: v for k, v in vars(args).items()},
      "checkpoint_step": ckpt_step,
      "rc": {"ref": ref["rc"], "crash": crash["rc"], "resume": resume["rc"]},
      "checks": {},
  }
  checks = summary["checks"]
  checks["ref run exit 0"] = ref["rc"] == 0
  checks["crash run killed (rc != 0)"] = crash["rc"] != 0
  resume_log = Path(resume["log"]).read_text()
  checks[f"resume restored adapter-only state of step {ckpt_step}"] = (
      f"resumed adapter-only state of step {ckpt_step}" in resume_log
  )
  checks["resume reloaded the base from Hugging Face"] = (
      "[ckpt] base weights loaded from" in resume_log
  )
  checks["resume run exit 0"] = resume["rc"] == 0
  ranks = sorted(ref["records"])
  checks[f"records from {args.nproc} rank(s)"] = len(ranks) == args.nproc
  # The trainer takes its gradient accumulation from the loader: each step of
  # each rank trains exactly microbatches_per_step microbatches.
  per_step = _recipe_microbatches(args.config)
  drawn = {
      len(rec.get("batches", []))
      for by_step in ref["records"].values()
      for rec in by_step.values()
  }
  checks[f"ref: {per_step} microbatches in every step of every rank"] = (
      drawn == {per_step}
  )
  killed_steps = sorted({s for r in crash_records.values() for s in r})
  checks[f"crash run stopped before finishing step {args.kill_step}"] = (
      bool(killed_steps) and max(killed_steps) == args.kill_step - 1
  )
  pre = compare(ref["records"], crash_records, range(1, args.kill_step))
  checks["crash run == ref before the kill (bitwise)"] = not pre
  post = compare(
      ref["records"], resume["records"], range(ckpt_step + 1, args.steps + 1)
  )
  checks[
      f"resumed steps {ckpt_step + 1}..{args.steps} == ref (bitwise: loss, grad norm, lr, batches)"
  ] = not post
  if checks["ref run exit 0"] and checks["resume run exit 0"]:
    a_ref = adapter_tensors(work / "ref", args.steps)
    a_res = adapter_tensors(work / "crash", args.steps)
    checks["final PEFT adapter identical"] = sorted(a_ref) == sorted(
        a_res
    ) and all(bool((a_ref[k] == a_res[k]).all()) for k in a_ref)
  else:
    checks["final PEFT adapter identical"] = False
  step_dir = work / "crash" / "checkpoint" / f"step-{ckpt_step}"
  keys = checkpoint_keys(step_dir) if (step_dir / ".metadata").is_file() else []
  model_keys = [
      k
      for k in keys
      if not k.startswith(
          ("optimizer", "lr_scheduler", "dataloader", "train_state")
      )
  ]
  checks["checkpoint model keys are LoRA tensors only"] = bool(
      model_keys
  ) and all(".lora_a." in k or ".lora_b." in k for k in model_keys)
  checks["checkpoint has optimizer, lr_scheduler, dataloader, train_state"] = (
      all(
          any(k.startswith(p) for k in keys)
          for p in ("optimizer", "lr_scheduler", "dataloader", "train_state")
      )
  )
  nbytes = (
      sum(p.stat().st_size for p in step_dir.rglob("*"))
      if step_dir.is_dir()
      else 0
  )
  summary["checkpoint"] = {
      "step": ckpt_step,
      "keys": len(keys),
      "model_keys": len(model_keys),
      "bytes": nbytes,
  }
  summary["diffs"] = {"pre_kill": pre[:20], "post_resume": post[:20]}
  summary["losses"] = (
      {
          name: {
              str(s): rec["records"][ranks[0]][s].get("loss_local")
              for s in sorted(rec["records"].get(ranks[0], {}))
          }
          for name, rec in (("ref", ref), ("resume", resume))
      }
      if ranks
      else {}
  )

  if args.partial:
    part = work / "partial"
    shutil.copytree(
        work / "crash",
        part,
        ignore=shutil.ignore_patterns("sft-records", "lora_adapter"),
    )
    newest = max(
        int(p.name.split("-")[1]) for p in (part / "checkpoint").glob("step-*")
    )
    planted = part / "checkpoint" / f"step-{newest + 1}"
    shutil.copytree(part / "checkpoint" / f"step-{ckpt_step}", planted)
    (planted / ".metadata").unlink()
    # Drop the completed run's later checkpoints: the incomplete one is the newest.
    for p in (part / "checkpoint").glob("step-*"):
      if int(p.name.split("-")[1]) not in (ckpt_step, newest + 1):
        shutil.rmtree(p)
    partial = run(args, "partial", part)
    plog = Path(partial["log"]).read_text()
    checks[
        f"partial: incomplete step-{newest + 1} skipped, resumed step {ckpt_step}"
    ] = (
        f"resumed adapter-only state of step {ckpt_step}" in plog
        and partial["rc"] == 0
    )
    pdiff = compare(
        ref["records"], partial["records"], range(ckpt_step + 1, args.steps + 1)
    )
    checks["partial: resumed steps == ref (bitwise)"] = not pdiff
    summary["diffs"]["partial"] = pdiff[:20]

  failed = [name for name in checks if not checks[name]]
  summary["pass"] = not failed
  with open(work / "resume_summary.json", "w") as handle:
    json.dump(summary, handle, indent=2, default=str)
  _print_outcome(summary, failed, args)
  return int(bool(failed))


def _print_outcome(summary: dict, failed: list[str], args) -> None:
  """The check lines, up to five differences per kind and the verdict."""
  for name in summary["checks"]:
    marker = "FAIL" if name in failed else "ok"
    print(f"[RESUME] {marker:<4} {name}")
  diffs = summary["diffs"]
  for kind in diffs:
    for diff in diffs[kind][:5]:
      print(f"[RESUME] diff {kind}: {diff}")
  verdict = "FAIL" if failed else "PASS"
  print(
      f"[RESUME] {verdict} config={args.config} nproc={args.nproc} "
      f"ckpt_step={summary['checkpoint_step']} "
      f"checkpoint={summary['checkpoint']}"
  )


if __name__ == "__main__":
  raise SystemExit(main())
