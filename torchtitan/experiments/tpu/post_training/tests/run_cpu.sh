#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# CPU checks of the post-training SFT layer. Needs a local Muse Spark
# tokenizer directory (tokenizer.json, tokenizer_config.json,
# chat_template.jinja, generation_config.json), e.g. a snapshot of the public
# tiny test model trl-internal-testing/tiny-MuseSparkForConditionalGeneration.
# When that directory also holds the tiny model's safetensors and the Muse
# Spark TPU package is installed, the smoke adds Muse Spark runs on the public
# tiny checkpoint. Never downloads anything. Without TorchTPU, the import-only
# stand-in in cpu_stubs/ satisfies the TPU modules' imports.
set -euo pipefail
TEST_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$TEST_DIR/../../../../.." && pwd)
CPU_PYTHON=${CPU_PYTHON:-python3}
tokenizer_dir=${SPARK_TEST_ASSETS:-}
work_root=
keep_work=0
usage() {
  echo 'usage: run_cpu.sh --tokenizer-dir MUSE_SPARK_DIR [--work NEW_DIR] [--unit-only]' >&2
  exit 2
}
unit_only=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tokenizer-dir) [[ $# -ge 2 ]] || usage; tokenizer_dir=$2; shift 2 ;;
    --work) [[ $# -ge 2 ]] || usage; work_root=$2; keep_work=1; shift 2 ;;
    --unit-only) unit_only=1; shift ;;
    *) usage ;;
  esac
done
[[ -d "$tokenizer_dir" ]] || usage
tokenizer_dir=$(cd "$tokenizer_dir" && pwd)
if [[ -n "$work_root" ]]; then
  [[ ! -e "$work_root" ]] || { echo '--work must name a new directory' >&2; exit 2; }
  mkdir -p "$work_root"
  work_root=$(cd "$work_root" && pwd)
else
  work_root=$(mktemp -d /tmp/post-training-tests.XXXXXX)
fi
finish() {
  if [[ "$keep_work" -eq 0 ]]; then rm -rf "$work_root"; else echo "CPU smoke artifacts: $work_root"; fi
}
trap finish EXIT
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if ! "$CPU_PYTHON" -c 'import torch_tpu' 2>/dev/null; then
  export PYTHONPATH="$TEST_DIR/cpu_stubs:$PYTHONPATH"
fi
export PJRT_DEVICE=CPU JAX_PLATFORMS=cpu TORCH_DEVICE_BACKEND_AUTOLOAD=0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export SPARK_TEST_ASSETS="$tokenizer_dir"
cd "$REPO_ROOT"
PKG=torchtitan.experiments.tpu.post_training.tests
"$CPU_PYTHON" -m unittest "$PKG.test_lora_optim" "$PKG.test_sft_data" "$PKG.test_recipes"
if [[ "$unit_only" -eq 1 ]]; then
  exit 0
fi
spark=()
if compgen -G "$tokenizer_dir/*.safetensors" > /dev/null; then
  spark=(--spark-checkpoint "$tokenizer_dir")
fi
"$CPU_PYTHON" -m "$PKG.cpu_sft_smoke" --tokenizer-dir "$tokenizer_dir" --work "$work_root/sft" "${spark[@]}"
