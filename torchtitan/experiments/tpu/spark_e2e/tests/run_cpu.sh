#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# CPU tests for the end-to-end LoRA SFT recipes (spark_e2e), the 64-chip step
# recipes (muse_spark/spark_64chip_recipes.py) and the launch kit
# (muse_spark/launch). Never downloads anything.
# usage: run_cpu.sh [--tiny-checkpoint DIR] [--nproc N]
#   --tiny-checkpoint  local copy of trl-internal-testing/tiny-MuseSparkForConditionalGeneration
#                      (revision c9e83e8784d89e7e7f66f78e15adf10e3a93fdbb): adds the torchrun SFT
#                      smoke and the kill-and-resume check through spark_e2e.train
#   --nproc            ranks for those runs (default 2)
# Without TorchTPU the TPU modules are imported through the shared CPU stand-in
# in torchtitan/experiments/tpu/tests/cpu_stubs (kernels raise if called;
# nothing here calls them).
set -euo pipefail
TEST_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$TEST_DIR/../../../../.." && pwd)
CPU_PYTHON=${CPU_PYTHON:-python3}
TINY= NPROC=2
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tiny-checkpoint) [[ $# -ge 2 ]] || { echo 'missing DIR' >&2; exit 2; }; TINY=$(cd "$2" && pwd); shift 2 ;;
    --nproc) [[ $# -ge 2 ]] || { echo 'missing N' >&2; exit 2; }; NPROC=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if ! "$CPU_PYTHON" -c 'import torch_tpu' >/dev/null 2>&1; then
  STUBS=$REPO_ROOT/torchtitan/experiments/tpu/tests/cpu_stubs
  [[ -d "$STUBS/torch_tpu" ]] || { echo "no torch_tpu and no CPU stand-in at $STUBS" >&2; exit 2; }
  export PYTHONPATH="$STUBS:$PYTHONPATH"
fi
export PJRT_DEVICE=CPU JAX_PLATFORMS=cpu TORCH_DEVICE_BACKEND_AUTOLOAD=0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
cd "$REPO_ROOT"
# One process per module: the step recipes patch checkpoint classes when imported.
rc=0
for test in muse_spark/tests/test_launch.py muse_spark/tests/test_spark_64chip_recipes.py \
            spark_e2e/tests/test_sft_recipes.py; do
  echo "== $test"
  "$CPU_PYTHON" -m pytest -q -p no:cacheprovider "torchtitan/experiments/tpu/$test" || rc=1
done
if [[ -n "$TINY" ]]; then
  work=$(mktemp -d "${TMPDIR:-/tmp}/spark-e2e-cpu.XXXXXX")
  echo "== cpu_sft_smoke (nproc $NPROC, work $work)"
  "$CPU_PYTHON" -m torchtitan.experiments.tpu.spark_e2e.tests.cpu_sft_smoke \
    --model "$TINY" --work "$work/smoke" --nproc "$NPROC" || rc=1
  echo "== cpu_sft_resume (nproc $NPROC, activation checkpointing, partial checkpoint)"
  "$CPU_PYTHON" -m torchtitan.experiments.tpu.spark_e2e.tests.cpu_sft_resume \
    --model "$TINY" --work "$work/resume" --nproc "$NPROC" --ac --partial || rc=1
fi
exit $rc
