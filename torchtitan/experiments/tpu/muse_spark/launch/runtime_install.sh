#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Offline install of a mounted, digest-pinned Python wheelhouse into a fresh
# virtual environment. Sourced by spark_pod.sh when SPARK_RUNTIME_MODE is
# "mounted"; nothing is downloaded.
#
# Runtime directory layout (NAME = SPARK_RUNTIME_NAME):
#   NAME-lock.txt        exact pins, one "package==version" per line (pip freeze format)
#   NAME-wheels.sha256   "<sha256>  <file>" for every wheel under wheels/
#   NAME-debs.sha256     optional: "<sha256>  <file>" for system packages under debs/
#   wheels/, debs/
#
# Inputs (environment):
#   SPARK_RUNTIME_DIR            mounted runtime directory
#   SPARK_RUNTIME_NAME           file prefix (default runtime)
#   SPARK_RUNTIME_LOCK_SHA256    SHA-256 of NAME-lock.txt (required)
#   SPARK_RUNTIME_WHEELS_SHA256  SHA-256 of NAME-wheels.sha256 (required)
#   SPARK_RUNTIME_DEBS_SHA256    SHA-256 of NAME-debs.sha256 (required when that file exists;
#                                when set, the file must exist)
#   SPARK_RUNTIME_VENV           target environment (default /opt/spark-runtime)
#   SPARK_RUNTIME_WORK           local scratch (default /work/runtime)
#   SPARK_TCMALLOC               allocator preloaded when present
#                                (default /usr/lib/x86_64-linux-gnu/libtcmalloc.so.4)
# Sets TRAINER_PYTHON and PATH for the caller, and LD_PRELOAD when the
# allocator library exists after the install.

spark_runtime_install() {
  local dir=${SPARK_RUNTIME_DIR:?set SPARK_RUNTIME_DIR}
  local name=${SPARK_RUNTIME_NAME:-runtime}
  local venv=${SPARK_RUNTIME_VENV:-/opt/spark-runtime}
  local work=${SPARK_RUNTIME_WORK:-/work/runtime}
  local lock="$dir/$name-lock.txt" wheels="$dir/$name-wheels.sha256" debs="$dir/$name-debs.sha256"
  local file
  : "${SPARK_RUNTIME_LOCK_SHA256:?pin the runtime lock file (render.py --runtime-lock-sha256)}"
  : "${SPARK_RUNTIME_WHEELS_SHA256:?pin the wheel digest list (render.py --runtime-wheels-sha256)}"
  mkdir -p "$work/wheels" "$work/debs"
  printf '%s  %s\n' "$SPARK_RUNTIME_LOCK_SHA256" "$lock" "$SPARK_RUNTIME_WHEELS_SHA256" "$wheels" \
    | sha256sum --quiet -c - || { echo "[SPARK:RUNTIME] lock or wheel list does not match its pin" >&2; return 3; }
  if [[ -n "${SPARK_RUNTIME_DEBS_SHA256:-}" && ! -f "$debs" ]]; then
    echo "[SPARK:RUNTIME] the system package list is pinned but $debs is missing" >&2
    return 3
  fi
  if [[ -f "$debs" ]]; then
    : "${SPARK_RUNTIME_DEBS_SHA256:?pin the system package list (render.py --runtime-debs-sha256)}"
    printf '%s  %s\n' "$SPARK_RUNTIME_DEBS_SHA256" "$debs" | sha256sum --quiet -c - \
      || { echo "[SPARK:RUNTIME] system package list does not match its pin" >&2; return 3; }
  fi
  # Copy to local disk first and verify every file there (the mount is read
  # once; pip and dpkg only see verified bytes).
  while read -r _ file; do
    file=${file#\*}
    [[ -n "$file" && "$file" != */* ]] || { echo "[SPARK:RUNTIME] bad wheel entry: $file" >&2; return 3; }
    cp "$dir/wheels/$file" "$work/wheels/"
  done < "$wheels"
  (cd "$work/wheels" && sha256sum --quiet -c "$wheels") || return 3
  if [[ -f "$debs" ]]; then
    while read -r _ file; do
      file=${file#\*}
      [[ -n "$file" && "$file" != */* ]] || { echo "[SPARK:RUNTIME] bad package entry: $file" >&2; return 3; }
      cp "$dir/debs/$file" "$work/debs/"
    done < "$debs"
    (cd "$work/debs" && sha256sum --quiet -c "$debs") || return 3
    dpkg -i "$work"/debs/*.deb > "$work/dpkg.log" 2>&1 || { tail -5 "$work/dpkg.log" >&2; return 3; }
  fi
  "${SPARK_BOOTSTRAP_PYTHON:-python3}" -m venv "$venv"
  "$venv/bin/python" -m pip install -q --no-index --no-deps --find-links "$work/wheels" -r "$lock"
  "$venv/bin/python" -m pip list --format=freeze > "$work/installed.txt"
  "$venv/bin/python" - "$lock" "$work/installed.txt" <<'PY'
import re
import sys


def pins(path):
    out = {}
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, sep, version = line.partition("==")
        if not sep:
            raise SystemExit(f"{path}: not an exact pin: {line!r}")
        key = re.sub(r"[-_.]+", "-", name).lower()
        if key != "pip":
            out[key] = version.strip()
    return out


want, got = pins(sys.argv[1]), pins(sys.argv[2])
if want != got:
    diff = sorted(set(want.items()) ^ set(got.items()))
    raise SystemExit(f"[SPARK:RUNTIME] installed set differs from the lock: {diff[:10]}")
print(f"[SPARK:RUNTIME] {len(got)} packages installed offline, identical to the lock", flush=True)
PY
  TRAINER_PYTHON=$venv/bin/python
  export PATH="$venv/bin:$PATH"
  local allocator=${SPARK_TCMALLOC:-/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4}
  if [[ -f "$allocator" ]]; then
    export LD_PRELOAD=$allocator
  fi
}

spark_runtime_install
