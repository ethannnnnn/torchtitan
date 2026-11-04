#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Container entry point rendered into the trainer JobSet (inline code
# delivery). Checks the slice wiring, fetches the public base commit, applies
# the code overlay carried by the mounted ConfigMaps (build_overlay.py), and
# refuses to continue unless the resulting git tree is exactly the candidate's
# tree. Then hands off to spark_pod.sh from that verified tree.
#
# Inputs (environment, set by render.py): SPARK_BASE_REPO, SPARK_BASE_COMMIT,
# SPARK_BASE_TREE, SPARK_CANDIDATE_TREE, SPARK_OVERLAY_SHA256, SPARK_OVERLAY_CHUNKS,
# NNODES.
# SPARK_SRC_DIR (default /work/tt) and SPARK_OVERLAY_DIR (default /mnt/overlay)
# move the tree and the overlay; SPARK_BOOTSTRAP_ONLY=1 stops after the check.
set -euo pipefail
ulimit -c 0
: "${SPARK_BASE_REPO:?set the base repository URL}"
: "${SPARK_BASE_COMMIT:?set the base commit}"
: "${SPARK_BASE_TREE:?set the base tree}"
: "${SPARK_CANDIDATE_TREE:?set the expected candidate tree}"
: "${SPARK_OVERLAY_SHA256:?set the overlay SHA-256}"
: "${SPARK_OVERLAY_CHUNKS:?set the number of overlay chunks}"
: "${NNODES:?set the host count}"
if (( NNODES > 1 )) && [[ "${SPARK_BOOTSTRAP_ONLY:-0}" != 1 ]]; then
  : "${TPU_PROCESS_ADDRESSES:?GKE must inject TPU_PROCESS_ADDRESSES}"
  IFS=',' read -r -a process_addresses <<< "$TPU_PROCESS_ADDRESSES"
  if [[ ${#process_addresses[@]} -ne "$NNODES" ]]; then
    echo "[SPARK:GATE] injected process-address count ${#process_addresses[@]} != NNODES $NNODES" >&2
    exit 5
  fi
fi
SRC=${SPARK_SRC_DIR:-/work/tt}
OVERLAY_DIR=${SPARK_OVERLAY_DIR:-/mnt/overlay}
WORK=$(dirname "$SRC")
mkdir -p "$WORK"

if ! command -v git >/dev/null 2>&1; then
  echo "[SPARK:CODE] git not found; installing it"
  { timeout 180 apt-get update -qq && DEBIAN_FRONTEND=noninteractive timeout 300 \
      apt-get install -y -qq --no-install-recommends git ca-certificates; } >/dev/null 2>&1 || true
fi
command -v git >/dev/null 2>&1 || { echo "[SPARK:GATE] git is needed to fetch the base commit" >&2; exit 5; }

# 1. The public base commit, checked against its commit and tree ids.
rm -rf "$SRC"
git init -q "$SRC"
git -C "$SRC" remote add origin "$SPARK_BASE_REPO"
fetched=0
for attempt in 1 2 3 4 5; do
  if timeout 600 git -C "$SRC" fetch -q --depth=1 origin "$SPARK_BASE_COMMIT"; then
    fetched=1
    break
  fi
  echo "[SPARK:CODE] fetch attempt $attempt of $SPARK_BASE_COMMIT failed; retrying" >&2
  sleep $(( attempt * 15 ))
done
(( fetched )) || { echo "[SPARK:GATE] cannot fetch $SPARK_BASE_COMMIT from $SPARK_BASE_REPO" >&2; exit 5; }
git -C "$SRC" -c advice.detachedHead=false checkout -q --detach FETCH_HEAD
if [[ "$(git -C "$SRC" rev-parse HEAD)" != "$SPARK_BASE_COMMIT" \
      || "$(git -C "$SRC" rev-parse 'HEAD^{tree}')" != "$SPARK_BASE_TREE" ]]; then
  echo "[SPARK:GATE] fetched base is not $SPARK_BASE_COMMIT / tree $SPARK_BASE_TREE" >&2
  exit 5
fi

# 2. The overlay: reassemble the ConfigMap chunks, check the digest, check every
#    file against the manifest (path, git mode, blob id), apply it.
: > "$WORK/overlay.tar.gz"
for (( chunk = 0; chunk < SPARK_OVERLAY_CHUNKS; chunk++ )); do
  cat "$OVERLAY_DIR/overlay.tar.gz.$chunk" >> "$WORK/overlay.tar.gz"
done
printf '%s  %s\n' "$SPARK_OVERLAY_SHA256" "$WORK/overlay.tar.gz" | sha256sum --quiet -c - \
  || { echo "[SPARK:GATE] overlay digest mismatch" >&2; exit 5; }
python3 - "$SRC" "$WORK/overlay.tar.gz" "$OVERLAY_DIR/overlay.json" "$WORK/overlay.paths" <<'PY'
import hashlib
import json
import os
import sys
import tarfile

src, archive, manifest_path, paths_out = sys.argv[1:]
manifest = json.load(open(manifest_path))
for key, env in (("base_commit", "SPARK_BASE_COMMIT"), ("candidate_tree", "SPARK_CANDIDATE_TREE"),
                 ("sha256", "SPARK_OVERLAY_SHA256")):
    if manifest[key] != os.environ[env]:
        sys.exit(f"[SPARK:GATE] overlay.json {key} {manifest[key]} != {env} {os.environ[env]}")
expected = {f["path"]: f for f in manifest["files"]}
root = os.path.realpath(src)


def target(path):
    full = os.path.realpath(os.path.join(root, os.path.dirname(path)))
    if os.path.isabs(path) or ".." in path.split("/") or not (full + os.sep).startswith(root + os.sep):
        sys.exit(f"[SPARK:GATE] overlay path escapes the tree: {path!r}")
    return os.path.join(root, path)


def remove(path):
    if os.path.islink(path) or os.path.isfile(path):
        os.remove(path)
    parent = os.path.dirname(path)
    while parent != root and os.path.isdir(parent) and not os.listdir(parent):
        os.rmdir(parent)
        parent = os.path.dirname(parent)


for path in manifest["deleted"]:
    remove(target(path))
seen = set()
with tarfile.open(archive, "r:gz") as tar:
    for member in tar:
        info = expected.get(member.name)
        if info is None or member.name in seen:
            sys.exit(f"[SPARK:GATE] unexpected overlay member {member.name!r}")
        seen.add(member.name)
        path = target(member.name)
        if os.path.lexists(path) and (os.path.islink(path) or not os.path.isdir(path)):
            os.remove(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if member.issym():
            data = member.linkname.encode()
            if info["mode"] != "120000":
                sys.exit(f"[SPARK:GATE] {member.name}: symlink, manifest mode {info['mode']}")
            os.symlink(member.linkname, path)
        elif member.isfile():
            data = tar.extractfile(member).read()
            mode = "100755" if member.mode & 0o111 else "100644"
            if mode != info["mode"]:
                sys.exit(f"[SPARK:GATE] {member.name}: mode {mode}, manifest {info['mode']}")
            with open(path, "wb") as handle:
                handle.write(data)
            os.chmod(path, 0o755 if mode == "100755" else 0o644)
        else:
            sys.exit(f"[SPARK:GATE] {member.name}: unsupported member type")
        blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        if blob != info["blob"]:
            sys.exit(f"[SPARK:GATE] {member.name}: blob {blob}, manifest {info['blob']}")
if seen != set(expected):
    sys.exit(f"[SPARK:GATE] overlay lacks {sorted(set(expected) - seen)[:5]}")
with open(paths_out, "wb") as handle:
    for path in sorted(set(expected) | set(manifest["deleted"])):
        handle.write(path.encode() + b"\0")
print(f"[SPARK:CODE] overlay applied: {len(expected)} files, {len(manifest['deleted'])} deletions", flush=True)
PY

# 3. The resulting tree must be the candidate's tree, bit for bit.
git -C "$SRC" --literal-pathspecs add -A -f --pathspec-from-file="$WORK/overlay.paths" --pathspec-file-nul
tree=$(git -C "$SRC" write-tree)
if [[ "$tree" != "$SPARK_CANDIDATE_TREE" ]]; then
  echo "[SPARK:GATE] source tree $tree != candidate tree $SPARK_CANDIDATE_TREE" >&2
  exit 5
fi
echo "[SPARK:CODE] base $SPARK_BASE_COMMIT + overlay = candidate tree $tree"
[[ "${SPARK_BOOTSTRAP_ONLY:-0}" == 1 ]] && exit 0
export SPARK_SRC_DIR=$SRC
exec /bin/bash "$SRC/torchtitan/experiments/tpu/muse_spark/launch/spark_pod.sh"
