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
# tiny checkpoint. Never downloads anything. Without TorchTPU, the shared
# import-only stand-in torchtitan/experiments/tpu/tests/cpu_stubs satisfies the
# TPU modules' imports. Each process uses OMP_NUM_THREADS threads (default 8).
set -euo pipefail
REPO_ROOT=$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../..")
: "${CPU_PYTHON:=python3}"
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
if (( keep_work == 0 )); then
  work_root=$(mktemp -d /tmp/sft-cpu-checks.XXXXXX)
elif [[ -e "$work_root" ]]; then
  echo "run_cpu.sh: $work_root already exists; --work takes a new folder" >&2
  exit 2
else
  mkdir -p -- "$work_root"
  work_root=$(realpath -- "$work_root")
fi
clean_up_work() {
  if (( keep_work )); then
    echo "run_cpu.sh: outputs kept in $work_root"
  else
    rm -rf -- "$work_root"
  fi
}
trap clean_up_work EXIT
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if ! "$CPU_PYTHON" -c 'import torch_tpu' 2>/dev/null; then
  stubs="$REPO_ROOT/torchtitan/experiments/tpu/tests/cpu_stubs"
  if [[ ! -d "$stubs/torch_tpu" ]]; then
    echo "torch_tpu is not installed and its CPU stand-in $stubs is missing" >&2
    exit 2
  fi
  export PYTHONPATH="$stubs:$PYTHONPATH"
fi
# A process otherwise starts one thread per core, which oversubscribes a
# shared host as soon as several run (the resume check runs two ranks).
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PJRT_DEVICE=CPU JAX_PLATFORMS=cpu TORCH_DEVICE_BACKEND_AUTOLOAD=0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export SPARK_TEST_ASSETS="$tokenizer_dir"
cd "$REPO_ROOT"
PKG=torchtitan.experiments.tpu.post_training.tests
"$CPU_PYTHON" -m unittest "$PKG.test_lora_optim" "$PKG.test_sft_data" "$PKG.test_recipes" \
  "$PKG.test_sft_validation" "$PKG.test_sft_train" "$PKG.test_smoke_common"
if [[ "$unit_only" -eq 1 ]]; then
  exit 0
fi
spark=()
if compgen -G "$tokenizer_dir/*.safetensors" > /dev/null; then
  spark=(--spark-checkpoint "$tokenizer_dir")
fi
"$CPU_PYTHON" -m "$PKG.cpu_sft_smoke" --tokenizer-dir "$tokenizer_dir" --work "$work_root/sft" "${spark[@]}"
# The kill-and-resume check needs the adapter-only checkpoints of
# ../checkpointing/lora_checkpoint.py. It is skipped only when that module is
# absent; one that is present but fails to import stops the runner here.
lora_checkpoint=$("$CPU_PYTHON" -c "from $PKG.smoke_common import has_lora_checkpoint; print(int(has_lora_checkpoint()))")
if [[ "$lora_checkpoint" == 1 ]]; then
  # SFT_CHECKPOINT_FLUSH=1: every rank leaves each save together, as on TPU.
  SFT_CHECKPOINT_FLUSH=1 "$CPU_PYTHON" -m "$PKG.cpu_sft_resume" --tokenizer-dir "$tokenizer_dir" \
    --work "$work_root/resume" --nproc 2 --steps 6 --interval 2 --kill-step 4 --partial
else
  echo 'skipping cpu_sft_resume: checkpointing/lora_checkpoint.py (adapter-only checkpoints) is not present'
fi
