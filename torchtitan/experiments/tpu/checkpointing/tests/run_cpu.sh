#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# CPU tests for the checkpointing package. Never downloads anything.
# usage: run_cpu.sh [--tiny-checkpoint DIR]
#   --tiny-checkpoint  local copy of trl-internal-testing/tiny-MuseSparkForConditionalGeneration
#                      (revision c9e83e8784d89e7e7f66f78e15adf10e3a93fdbb); without it the
#                      Muse Spark round-trip and resume checks are skipped.
# Needs torch (CPU), safetensors, fsspec, pytest. Optional: transformers >= 5.18 (Muse Spark
# round trips), peft (parity with the PEFT library's merge), google-auth (credential tests);
# the Muse Spark resume / LoRA checks also need the muse_spark port and post_training.lora.
# Where torch_tpu is not installed, those import TorchTPU through the shared import-only
# stand-in torchtitan/experiments/tpu/tests/cpu_stubs when the tree has it.
set -euo pipefail
TEST_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$TEST_DIR/../../../../.." && pwd)
CPU_PYTHON=${CPU_PYTHON:-python3}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tiny-checkpoint) [[ $# -ge 2 ]] || { echo 'missing DIR' >&2; exit 2; }
      MUSE_SPARK_TINY_DIR=$(cd "$2" && pwd); export MUSE_SPARK_TINY_DIR; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
STUBS="$REPO_ROOT/torchtitan/experiments/tpu/tests/cpu_stubs"
# The stand-in refuses to load next to a real torch_tpu: use it only where none is installed.
if [[ -d "$STUBS/torch_tpu" ]] && "$CPU_PYTHON" -c \
    'import importlib.util, sys; sys.exit(importlib.util.find_spec("torch_tpu") is not None)'; then
  export PYTHONPATH="$STUBS:$PYTHONPATH"
fi
export PJRT_DEVICE=CPU JAX_PLATFORMS=cpu TORCH_DEVICE_BACKEND_AUTOLOAD=0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
cd "$REPO_ROOT"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
"$CPU_PYTHON" -m pytest -q -p no:cacheprovider torchtitan/experiments/tpu/checkpointing/tests
# Two-rank FSDP (DTensor shards in DCP) bit-exact resume through TorchTitan's CheckpointManager.
models=(llama3)
if [[ -n "${MUSE_SPARK_TINY_DIR:-}" ]] && "$CPU_PYTHON" -c "import torchtitan.experiments.tpu.muse_spark, torchtitan.experiments.tpu.post_training.lora" 2>/dev/null; then
  models+=(muse_spark)
fi
for model in "${models[@]}"; do
  for flags in "" "--tuned" "--lora" "--lora --tuned"; do
    # shellcheck disable=SC2086
    if "$CPU_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=2 \
        -m torchtitan.experiments.tpu.checkpointing.tests.fsdp_resume_check --model "$model" $flags \
        > "$TMP/fsdp_resume_check.log" 2>&1; then
      grep "^fsdp_resume_check" "$TMP/fsdp_resume_check.log"
    else
      tail -40 "$TMP/fsdp_resume_check.log"
      echo "fsdp_resume_check --model $model $flags FAILED"
      exit 1
    fi
  done
done
# Multi-process smoke test of the I/O benchmark (4 gloo ranks, local files only).
"$CPU_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=4 \
  torchtitan/experiments/tpu/checkpointing/dcp_io_bench.py \
  --device cpu --preset tiny --ref-devices 4 --methods fs,fsspec,stage \
  --fs-root "$TMP/fs" --gs-root "file://$TMP/remote" --stage-root "$TMP/stage" --auth none \
  --cal-layers 2 --threads 1,4 --par-threads 4 --raw-read --budget-s 300 --d2h-probe-gb 0.001 \
  --out "$TMP/results.json" > "$TMP/bench.log" 2>&1 || { tail -50 "$TMP/bench.log"; exit 1; }
"$CPU_PYTHON" - "$TMP/results.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
bad = [p for p in r["phases"] if not p.get("verify_bit_exact") or "error" in p]
print(f"dcp_io_bench smoke: {len(r['phases'])} phases, {len(bad)} failed")
sys.exit(1 if bad or not r["phases"] else 0)
EOF
