#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Build the inline code overlay that trainer pods apply to a public base commit. Local only.

Pods do not download a code archive. Each pod fetches the public base commit
(``--base``, for example a commit of the upstream repository) and applies an
overlay of the candidate's changed files, which the rendered manifest carries
in ConfigMaps (``render.py --code-overlay``). The pod then rebuilds the git
tree and refuses to start unless it equals the candidate's tree.

This tool writes ``OUT_DIR/overlay.tar.gz`` (every file the candidate adds or
changes relative to the base, deterministic: sorted, mode 0644 / 0755 or
symlink as in git, mtime 0, gzip without a timestamp) and
``OUT_DIR/overlay.json`` (base and candidate commits and trees, the file list
with git modes and blob ids, deleted paths, the archive's SHA-256 and size,
and the candidate's blob ids of the launch files ``render.py`` copies into
the manifest: the container entry point and the JobSet template). Nothing
is uploaded and nothing outside OUT_DIR is written.

    python build_overlay.py --repo . --base <BASE_COMMIT> --candidate HEAD --out ./overlay
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path

KIND = "muse_spark_code_overlay_v1"
MAX_OVERLAY_BYTES = 3 * 1024 * 1024
"""Compressed overlay limit: it travels in ConfigMaps (about 1 MiB each)."""
FILE_MODES = {"100644": 0o644, "100755": 0o755, "120000": 0o777}
SHA1 = re.compile(r"[0-9a-f]{40}")
KIT_DIR = "torchtitan/experiments/tpu/muse_spark/launch"
KIT_FILES = ("bootstrap.sh", "trainer_jobset.json.in")
"""Launch files that render.py takes from its own directory into the rendered manifest."""


def blob_id(data: bytes) -> str:
    """The git blob id of ``data``."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def git(repo: Path, *args: str, binary: bool = False) -> str | bytes:
    out = subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout
    return out if binary else out.decode().strip()


def resolve(repo: Path, rev: str) -> tuple[str, str]:
    commit = git(repo, "rev-parse", "--verify", f"{rev}^{{commit}}")
    return commit, git(repo, "rev-parse", f"{commit}^{{tree}}")


def changed_paths(repo: Path, base: str, candidate: str) -> tuple[list[dict], list[str]]:
    """Files the candidate adds or changes (with git mode and blob id) and paths it deletes."""
    raw = git(repo, "diff-tree", "-r", "-z", "--no-renames", "--raw", base, candidate, binary=True)
    fields = raw.split(b"\0")
    changed, deleted = [], []
    i = 0
    while i < len(fields) - 1:
        meta = fields[i].decode()
        path = fields[i + 1].decode()
        i += 2
        if not meta.startswith(":"):
            raise ValueError(f"unexpected diff-tree record {meta!r}")
        _, new_mode, _, new_blob, status = meta[1:].split(" ")
        if "\n" in path or path.startswith("/") or ".." in path.split("/"):
            raise ValueError(f"refusing path {path!r}")
        if status == "D":
            deleted.append(path)
            continue
        if new_mode not in FILE_MODES:
            raise ValueError(f"{path}: git mode {new_mode} (submodules and others are not supported)")
        changed.append({"path": path, "status": status, "mode": new_mode, "blob": new_blob})
    return sorted(changed, key=lambda f: f["path"]), sorted(deleted)


def kit_blobs(repo: Path, commit: str) -> dict[str, str]:
    """Blob ids of the launch files in ``commit``, whether or not the candidate changes them."""
    blobs = {}
    for name in KIT_FILES:
        path = f"{KIT_DIR}/{name}"
        found = subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", "-q", f"{commit}:{path}"],
                               capture_output=True, text=True)
        if found.returncode == 0:
            blobs[path] = found.stdout.strip()
    return blobs


def build_archive(repo: Path, files: list[dict]) -> bytes:
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for item in files:
            data = git(repo, "cat-file", "blob", item["blob"], binary=True)
            info = tarfile.TarInfo(item["path"])
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = FILE_MODES[item["mode"]]
            if item["mode"] == "120000":
                info.type = tarfile.SYMTYPE
                info.linkname = data.decode()
                archive.addfile(info)
            else:
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    gz_buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=gz_buffer, compresslevel=9, mtime=0) as gz:
        gz.write(tar_buffer.getvalue())
    return gz_buffer.getvalue()


def build(repo: Path, base: str, candidate: str, out: Path) -> dict:
    base_commit, base_tree = resolve(repo, base)
    candidate_commit, candidate_tree = resolve(repo, candidate)
    if base_tree == candidate_tree:
        raise ValueError("the candidate's tree equals the base tree; nothing to overlay")
    files, deleted = changed_paths(repo, base_commit, candidate_commit)
    archive = build_archive(repo, files)
    if len(archive) > MAX_OVERLAY_BYTES:
        raise ValueError(f"overlay is {len(archive)} bytes compressed, over the {MAX_OVERLAY_BYTES}-byte limit")
    out.mkdir(parents=True, exist_ok=False)
    (out / "overlay.tar.gz").write_bytes(archive)
    manifest = {
        "kind": KIND,
        "base_commit": base_commit, "base_tree": base_tree,
        "candidate_commit": candidate_commit, "candidate_tree": candidate_tree,
        "sha256": hashlib.sha256(archive).hexdigest(), "bytes": len(archive),
        "files": files, "deleted": deleted,
        "kit": kit_blobs(repo, candidate_commit),
        "method": "git diff-tree -r --no-renames BASE CANDIDATE; tar of the changed blobs (mtime 0, git modes), gzip -n",
    }
    (out / "overlay.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_manifest(directory: Path) -> tuple[dict, bytes]:
    """The manifest and archive of an overlay directory, checked for consistency."""
    manifest = json.loads((directory / "overlay.json").read_text())
    archive = (directory / "overlay.tar.gz").read_bytes()
    if manifest.get("kind") != KIND:
        raise ValueError(f"{directory}/overlay.json is not a {KIND} manifest")
    for key in ("base_commit", "base_tree", "candidate_commit", "candidate_tree"):
        if not SHA1.fullmatch(str(manifest.get(key, ""))):
            raise ValueError(f"overlay.json: {key} must be a full SHA-1")
    if hashlib.sha256(archive).hexdigest() != manifest["sha256"] or len(archive) != manifest["bytes"]:
        raise ValueError("overlay.tar.gz does not match overlay.json")
    return manifest, archive


_OVERLAY_OPTIONS = (
    ("--repo", {"type": Path, "default": Path("."),
                "help": "git checkout that holds both commits (default: the current directory)"}),
    ("--base", {"required": True, "help": "public base commit that every pod fetches before applying the overlay"}),
    ("--candidate", {"default": "HEAD", "help": "commit whose tree the pods must end up with (default: HEAD)"}),
    ("--out", {"type": Path, "required": True, "help": "output directory to create; it must not exist yet"}),
)


def main() -> int:
    overlay_cli = argparse.ArgumentParser(
        description="Package a candidate commit's changes against a public base as the pods' inline code overlay.")
    for flag, settings in _OVERLAY_OPTIONS:
        overlay_cli.add_argument(flag, **settings)
    args = overlay_cli.parse_args()
    manifest = build(args.repo.resolve(), args.base, args.candidate, args.out)
    print(f"overlay: {len(manifest['files'])} files, {len(manifest['deleted'])} deletions, "
          f"{manifest['bytes']} bytes; base {manifest['base_commit']} -> tree {manifest['candidate_tree']}")
    print(manifest["sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
