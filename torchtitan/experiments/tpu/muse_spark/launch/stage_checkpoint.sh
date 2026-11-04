#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Stage the public Muse Spark Open checkpoint from Hugging Face to Cloud
# Storage one file at a time, so local disk only needs room for one shard
# (about 26 GB). Every file is checked against the SHA-256 that the Hub
# records for it before upload. A SHA256SUMS file and a _STAGED_OK marker are
# written last.
#
# Requirements: you accepted the Muse Model License on the model page with your
# own account; HF_TOKEN holds your own token; huggingface_hub >= 1.0 (the `hf`
# command) and gcloud are installed. Nothing here bypasses the license gate.
#
# usage: stage_checkpoint.sh --revision REV --dest gs://<GCS_BUCKET>/muse-spark/model/Muse-Spark-Open-816B-A42B \
#          [--repo meta-models/Muse-Spark-Open-816B-A42B] [--work DIR]
set -euo pipefail
repo=meta-models/Muse-Spark-Open-816B-A42B
revision= dest= work=
while [[ $# -gt 0 ]]; do
  [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }
  case "$1" in
    --repo) repo=$2 ;;
    --revision) revision=$2 ;;
    --dest) dest=${2%/} ;;
    --work) work=$2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift 2
done
[[ -n "$revision" && "$dest" == gs://* ]] || { echo 'need --revision and --dest gs://...' >&2; exit 2; }
: "${HF_TOKEN:?export your own Hugging Face token (license accepted)}"
work=${work:-$(mktemp -d)}
mkdir -p "$work"
# File list and Hub-recorded SHA-256 (LFS files) for the pinned revision.
python3 - "$repo" "$revision" "$work/hub-files.tsv" <<'PY'
import sys
from huggingface_hub import HfApi
repo, revision, out = sys.argv[1:]
api = HfApi()
rows = []
for item in api.list_repo_tree(repo, revision=revision, recursive=True, expand=True):
    if not hasattr(item, "size") or item.path.startswith("."):
        continue
    lfs = getattr(item, "lfs", None)
    rows.append((item.path, str(item.size), lfs.sha256 if lfs else "-"))
with open(out, "w") as fh:
    for row in sorted(rows):
        fh.write("\t".join(row) + "\n")
print(f"{len(rows)} files at {revision}")
PY
: > "$work/SHA256SUMS.partial"
while IFS=$'\t' read -r path size hub_sha; do
  if gcloud storage objects describe "$dest/$path.ok" >/dev/null 2>&1; then
    gcloud storage cat "$dest/$path.ok" >> "$work/SHA256SUMS.partial"
    echo "skip (already staged): $path"
    continue
  fi
  hf download "$repo" "$path" --revision "$revision" --local-dir "$work/files" >/dev/null
  local_sha=$(sha256sum "$work/files/$path" | cut -d' ' -f1)
  if [[ "$hub_sha" != "-" && "$hub_sha" != "$local_sha" ]]; then
    echo "checksum mismatch for $path" >&2
    exit 3
  fi
  [[ $(stat -c %s "$work/files/$path") == "$size" ]] || { echo "size mismatch for $path" >&2; exit 3; }
  gcloud storage cp "$work/files/$path" "$dest/$path"
  printf '%s  %s\n' "$local_sha" "$path" | tee -a "$work/SHA256SUMS.partial" | gcloud storage cp - "$dest/$path.ok"
  rm -f "$work/files/$path"
done < "$work/hub-files.tsv"
sort -k2 "$work/SHA256SUMS.partial" > "$work/SHA256SUMS"
gcloud storage cp "$work/SHA256SUMS" "$dest/SHA256SUMS"
printf 'repo=%s\nrevision=%s\n' "$repo" "$revision" | gcloud storage cp - "$dest/_STAGED_OK"
echo "staged $repo@$revision to $dest"
