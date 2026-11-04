#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# In-pod runner for one Muse Spark recipe on a TPU7x slice (one process per
# host, eight TensorCore ranks per host). Started by bootstrap.sh from the
# source tree it verified against the candidate's git tree.
#
# Inputs (environment):
#   SPARK_CONFIG           recipe name (required)
#   SPARK_ENTRY            python module to run   (default torchtitan.experiments.tpu.train)
#   SPARK_CONFIG_MODULE    TorchTitan --module    (default torchtitan.experiments.tpu.muse_spark)
#   SPARK_EXTRA_ARGS       extra trainer flags, shell-quoted
#   SPARK_MODEL / SPARK_DATA / SPARK_OUT   mounted checkpoint, dataset, output dirs
#   SPARK_RUNTIME_MODE     auto | mounted | preinstalled (default auto: mounted when
#                          SPARK_RUNTIME_DIR holds SPARK_RUNTIME_NAME-lock.txt; see runtime_install.sh)
#   SPARK_PIP_OVERLAY      optional hash-pinned requirements file (path in the source tree)
#                          installed with --no-deps --require-hashes --only-binary=:all:
#   SPARK_PIP_FIND_LINKS   install SPARK_PIP_OVERLAY offline from this directory
#   SPARK_PIP_INDEX_URL    package index otherwise (default https://pypi.org/simple)
#   SPARK_LOG              runner log (default /work/trainer.log)
#   SPARK_DRY_RUN=1        print the trainer command instead of running it (local checks)
#   NNODES, JOB_COMPLETION_INDEX, WORKERS_0_HOSTNAME   multi-host rendezvous
# Outputs under SPARK_OUT: stdout-hostN.log, versions-hostN.json, pip-freeze-hostN.txt,
# exit_code-hostN, plus whatever the trainer writes (--dump_folder).
set -euo pipefail
ulimit -c 0
ulimit -n 65536 2>/dev/null || true
SRC=${SPARK_SRC_DIR:-/work/tt}
MODEL=${SPARK_MODEL:-/mnt/model}
DATA=${SPARK_DATA:-/mnt/data}
OUT=${SPARK_OUT:-/mnt/out}
ENTRY=${SPARK_ENTRY:-torchtitan.experiments.tpu.train}
CONFIG_MODULE=${SPARK_CONFIG_MODULE:-torchtitan.experiments.tpu.muse_spark}
CONFIG=${SPARK_CONFIG:?set SPARK_CONFIG to a recipe name}
NNODES=${NNODES:-1}
NODE_RANK=${JOB_COMPLETION_INDEX:-${TPU_WORKER_ID:-0}}
HOST=host$NODE_RANK
BUDGET_S=${SPARK_BUDGET_S:-2400}
RUNTIME_DIR=${SPARK_RUNTIME_DIR:-/mnt/runtime}
RUNTIME_NAME=${SPARK_RUNTIME_NAME:-runtime}
RUNTIME_MODE=${SPARK_RUNTIME_MODE:-auto}
TRAINER_PYTHON=${TRAINER_PYTHON:-python3}
LOG=${SPARK_LOG:-/work/trainer.log}
mkdir -p "$(dirname "$LOG")" "$OUT"
: > "$LOG"
( while sleep 30; do cp -f "$LOG" "$OUT/stdout-$HOST.log" 2>/dev/null || true; done ) &
log_sync_pid=$!
finish() {
  local rc=$?
  trap - EXIT
  kill "$log_sync_pid" 2>/dev/null || true
  echo "[SPARK:EXIT] rc=$rc $(date -u +%FT%TZ)"
  exec 1>&3 2>&4
  wait "$log_tee_pid" || true
  cp -f "$LOG" "$OUT/stdout-$HOST.log" || true
  printf '%s\n' "$rc" > "$OUT/exit_code-$HOST"
  exit "$rc"
}
trap finish EXIT
exec 3>&1 4>&2
exec > >(tee -a "$LOG" >&3) 2>&1
log_tee_pid=$!

echo "[SPARK:START] run=${SPARK_RUN:-spark} config=$CONFIG $HOST/$NNODES $(date -u +%FT%TZ)"
if [[ -n "${SPARK_TOPOLOGY:-}" && -n "${TPU_TOPOLOGY:-}" && "$SPARK_TOPOLOGY" != "$TPU_TOPOLOGY" ]]; then
  echo "[SPARK:GATE] slice topology $TPU_TOPOLOGY does not match the rendered $SPARK_TOPOLOGY" >&2
  exit 5
fi
[[ -d "$SRC/torchtitan" ]] || { echo "[SPARK:GATE] missing source tree $SRC" >&2; exit 5; }
[[ -f "$MODEL/config.json" ]] || { echo "[SPARK:GATE] missing $MODEL/config.json" >&2; exit 5; }
if [[ -f "$MODEL/SHA256SUMS" && "${SPARK_VERIFY_MODEL:-0}" == 1 ]]; then
  (cd "$MODEL" && sha256sum -c --quiet SHA256SUMS)
fi

if [[ "$RUNTIME_MODE" == auto ]]; then
  if [[ -f "$RUNTIME_DIR/$RUNTIME_NAME-lock.txt" ]]; then RUNTIME_MODE=mounted; else RUNTIME_MODE=preinstalled; fi
fi
case "$RUNTIME_MODE" in
  mounted)
    export SPARK_RUNTIME_DIR=$RUNTIME_DIR SPARK_RUNTIME_NAME=$RUNTIME_NAME
    # shellcheck source=runtime_install.sh
    source "$SRC/torchtitan/experiments/tpu/muse_spark/launch/runtime_install.sh"
    ;;
  preinstalled) ;;
  *) echo 'SPARK_RUNTIME_MODE must be auto, mounted, or preinstalled' >&2; exit 2 ;;
esac
echo "[SPARK:RUNTIME] mode=$RUNTIME_MODE python=$(command -v "$TRAINER_PYTHON")"

if [[ -n "${SPARK_PIP_OVERLAY:-}" ]]; then
  # Exact wheels on top of the runtime: --no-deps (nothing else changes),
  # --require-hashes (pip refuses any file whose SHA-256 is not listed).
  overlay="$SRC/$SPARK_PIP_OVERLAY"
  [[ -f "$overlay" ]] || { echo "[SPARK:GATE] missing pip overlay $overlay" >&2; exit 5; }
  source_args=(--index-url "${SPARK_PIP_INDEX_URL:-https://pypi.org/simple}")
  [[ -n "${SPARK_PIP_FIND_LINKS:-}" ]] && source_args=(--no-index --find-links "$SPARK_PIP_FIND_LINKS")
  echo "[SPARK:PIP] installing $SPARK_PIP_OVERLAY (${source_args[*]})"
  timeout 1200 "$TRAINER_PYTHON" -m pip install -q --no-deps --require-hashes --only-binary=:all: \
    --no-cache-dir --disable-pip-version-check "${source_args[@]}" -r "$overlay"
  "$TRAINER_PYTHON" -m pip check || echo "[SPARK:PIP] pip check reports the conflicts above (expected for --no-deps overlays)"
fi
"$TRAINER_PYTHON" -m pip freeze --all > "$OUT/pip-freeze-$HOST.txt" 2>/dev/null || true

export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
if [[ -z "${TPU_WORKER_HOSTNAMES:-}" && -n "${WORKERS_0_HOSTNAME:-}" ]]; then
  export TPU_WORKER_HOSTNAMES=$WORKERS_0_HOSTNAME
fi
# Compile caches live in memory: TPU node boot disks have only ~20 GB free.
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT=${TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT:-/dev/shm/cache/tier3}
mkdir -p "$TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT" /dev/shm/torch_tpu_cache 2>/dev/null || true
cd "$SRC"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$TRAINER_PYTHON" - "$OUT/versions-$HOST.json" <<'PY'
import importlib.metadata as md, json, os, platform, sys
versions = {"python": platform.python_version()}
for name in ("torch", "torch_tpu", "libtpu", "jax", "jaxlib", "tokamax", "transformers", "safetensors"):
    try:
        versions[name] = md.version(name)
    except md.PackageNotFoundError:
        versions[name] = None
for key in ("SPARK_BASE_COMMIT", "SPARK_CANDIDATE_COMMIT", "SPARK_CANDIDATE_TREE"):
    if os.environ.get(key):
        versions[key[len("SPARK_"):].lower()] = os.environ[key]
line = json.dumps(versions, sort_keys=True)
print("[SPARK:VERSIONS] " + line, flush=True)
open(sys.argv[1], "w").write(line + "\n")
PY

RDZV=(--nnodes="$NNODES" --node-rank="$NODE_RANK")
if (( NNODES > 1 )); then
  : "${WORKERS_0_HOSTNAME:?set the rank-0 host name}"
  if [[ "${SPARK_DRY_RUN:-0}" != 1 ]]; then
    for _ in $(seq 120); do getent hosts "$WORKERS_0_HOSTNAME" >/dev/null && break; sleep 5; done
  fi
  RDZV+=(--rdzv-backend=static --rdzv-endpoint="$WORKERS_0_HOSTNAME:29501")
else
  RDZV+=(--master-addr=127.0.0.1 --master-port=29501)
fi
CLI_EXTRA=()
readarray -d '' -t CLI_EXTRA < <("$TRAINER_PYTHON" - "${SPARK_EXTRA_ARGS:-}" <<'PY'
import shlex, sys
for arg in shlex.split(sys.argv[1]):
    sys.stdout.buffer.write(arg.encode() + b"\0")
PY
)
if [[ "${SPARK_MODE:-}" == export ]]; then
  # Merged Hugging Face export of a LoRA adapter into the released layout: every
  # host exports its share of the shards from $MODEL to $OUT/hf (one worker per
  # host), then host 0 waits for every worker manifest and finalizes (index,
  # small files, export_manifest.json, EXPORT_OK).
  PEFT_DIR="$DATA/$CONFIG"
  EXPORT_OUT=${SPARK_EXPORT_OUT:-$OUT/hf}
  [[ -f "$PEFT_DIR/adapter_model.safetensors" ]] || { echo "[SPARK:GATE] no PEFT adapter at $PEFT_DIR" >&2; exit 5; }
  echo "[SPARK:EXPORT] peft=$PEFT_DIR out=$EXPORT_OUT worker $NODE_RANK of $NNODES"
  export_cmd=("$TRAINER_PYTHON" -m torchtitan.experiments.tpu.checkpointing.hf_export
              --base "$MODEL" --out "$EXPORT_OUT" --peft "$PEFT_DIR"
              --worker "$NODE_RANK" --num-workers "$NNODES" "${CLI_EXTRA[@]}")
  if [[ "${SPARK_DRY_RUN:-0}" == 1 ]]; then
    printf '[SPARK:DRY_RUN]'; printf ' %q' "${export_cmd[@]}"; echo
    exit 0
  fi
  start=$(date +%s)
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 timeout --signal=TERM --kill-after=30 "$BUDGET_S" "${export_cmd[@]}"
  if (( NODE_RANK == 0 )); then
    manifest_glob="$EXPORT_OUT/export_manifest.worker*-of-$(printf '%05d' "$NNODES").json"
    while (( $(compgen -G "$manifest_glob" | wc -l) < NNODES )); do
      (( $(date +%s) - start < BUDGET_S )) || { echo '[SPARK:EXPORT] timed out waiting for workers' >&2; exit 4; }
      sleep 10
    done
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$TRAINER_PYTHON" -m torchtitan.experiments.tpu.checkpointing.hf_export \
      --base "$MODEL" --out "$EXPORT_OUT" --num-workers "$NNODES" --finalize
  fi
  echo "[SPARK:EXPORT] done in $(( $(date +%s) - start )) s"
  exit 0
fi
train_cmd=("$TRAINER_PYTHON" -m torch.distributed.run "${RDZV[@]}" --nproc-per-node=8
           -m "$ENTRY" --module="$CONFIG_MODULE" --config="$CONFIG"
           --hf_assets_path="$MODEL" --dump_folder="$OUT" "${CLI_EXTRA[@]}")
echo "[SPARK:RUN] $ENTRY --module=$CONFIG_MODULE --config=$CONFIG ${CLI_EXTRA[*]}"
if [[ "${SPARK_DRY_RUN:-0}" == 1 ]]; then
  printf '[SPARK:DRY_RUN]'; printf ' %q' "${train_cmd[@]}"; echo
  exit 0
fi
timeout --signal=TERM --kill-after=30 "$BUDGET_S" "${train_cmd[@]}"
