#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Render a Muse Spark trainer JobSet for a TPU7x slice. Never contacts a cluster.

Every site-specific value (bucket, project, namespace, queue, image, storage
directories) is a command-line argument; the ``.json.in`` template keeps
``<PLACEHOLDER>`` names and rendering fails if any placeholder survives.

Code reaches the pods inline: ``--code-overlay`` is the output of
``build_overlay.py`` (the candidate's changed files relative to a public base
commit). It is split into ConfigMaps that the pods mount; each pod fetches the
base commit, applies the overlay and starts only if the resulting git tree is
the candidate's tree (``bootstrap.sh``). The container command is this
directory's ``bootstrap.sh`` and the JobSet comes from this directory's
template, so rendering stops unless both are the candidate's own copies (the
blob ids in the overlay manifest).

Output goes to a new directory: ``trainer.json`` (a ``v1`` List: the overlay
ConfigMaps, then the suspended JobSet), ``own-configmaps.sh`` (run once after
``kubectl create``: makes the JobSet the owner of its code ConfigMaps, so they
are deleted with it, also when ``ttlSecondsAfterFinished`` removes a finished
JobSet) and ``receipt.json`` (geometry, code, inputs, file digests).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE))
import build_overlay  # noqa: E402

PLACEHOLDER = re.compile(r"<[A-Z][A-Z0-9_]*>")
DNS_LABEL = re.compile(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?")
ENV_NAME = re.compile(r"[A-Z_][A-Z0-9_]*")
SECRET_NAME = re.compile(r"TOKEN|SECRET|PASSWORD|CREDENTIAL|PRIVATE_KEY")
SHA256 = re.compile(r"[0-9a-f]{64}")
# In container arguments the kubelet replaces $(NAME) by the value of the container variable NAME
# and $$ by $; the embedded bootstrap.sh must reach the pod unchanged.
KUBELET_EXPANSION = re.compile(r"\$\$|\$\(([A-Za-z_][A-Za-z0-9_]*)\)")
DEFAULT_BASE_REPO = "https://github.com/google-pytorch/torchtitan.git"
CHUNK_BYTES = 700 * 1024
"""Raw overlay bytes per ConfigMap; base64 keeps each object under the 1 MiB ConfigMap limit."""
CONFIGMAP_LIMIT = 1024 * 1024 - 16 * 1024

# TPU7x slice shapes: hosts (four chips, eight TensorCores each) and accelerator type.
TOPOLOGIES = {
    "2x2x1": (1, "tpu7x-8"),
    "2x2x2": (2, "tpu7x-16"),
    "2x2x4": (4, "tpu7x-32"),
    "2x4x4": (8, "tpu7x-64"),
    "4x4x4": (16, "tpu7x-128"),
    "4x4x8": (32, "tpu7x-256"),
    "4x8x8": (64, "tpu7x-512"),
    "8x8x8": (128, "tpu7x-1024"),
}
MODES = {
    # mode: (entry module, TorchTitan --module)
    "train": ("torchtitan.experiments.tpu.train", "torchtitan.experiments.tpu.muse_spark"),
    "step": ("torchtitan.experiments.tpu.train", "torchtitan.experiments.tpu.muse_spark.spark_64chip_recipes"),
    "sft": ("torchtitan.experiments.tpu.post_training.sft_train",
            "torchtitan.experiments.tpu.post_training.config_registry"),
    "e2e": ("torchtitan.experiments.tpu.spark_e2e.train", "torchtitan.experiments.tpu.spark_e2e.recipes"),
    # Merged Hugging Face export of a LoRA run (no training): one export worker
    # per host (checkpointing/hf_export.py), host 0 finalizes. --config names
    # the PEFT adapter directory relative to the data mount (see spark_pod.sh).
    "export": ("torchtitan.experiments.tpu.checkpointing.hf_export", "-"),
}
RUNTIME_MODES = ("auto", "mounted", "preinstalled")


def substitute(value, settings):
    if isinstance(value, dict):
        return {k: substitute(v, settings) for k, v in value.items()}
    if isinstance(value, list):
        return [substitute(v, settings) for v in value]
    if isinstance(value, str):
        if value in settings:
            return settings[value]
        return PLACEHOLDER.sub(lambda m: str(settings.get(m.group(), m.group())), value)
    return value


def render_template(name, settings):
    result = substitute(json.loads((HERE / name).read_text()), settings)
    unresolved = sorted(set(PLACEHOLDER.findall(json.dumps(result))))
    if unresolved:
        raise ValueError("unresolved template values: " + ", ".join(unresolved))
    return result


def _relative_dir(value: str) -> bool:
    return not value.startswith("/") and all(part not in ("", ".", "..") for part in value.split("/"))


def check_kit(manifest: dict) -> dict[str, str]:
    """Blob ids of this directory's launch files, which must be the candidate's (see build_overlay.KIT_FILES).

    bootstrap.sh is the script that checks the candidate's tree in every pod; an
    entry script or pod template from another revision or a dirty checkout
    would bypass that check, so it has to come from the verified candidate too.
    """
    recorded = {f["path"]: f["blob"] for f in manifest["files"]}
    recorded.update(manifest.get("kit", {}))
    blobs = {}
    for name in build_overlay.KIT_FILES:
        path = f"{build_overlay.KIT_DIR}/{name}"
        have = build_overlay.blob_id((HERE / name).read_bytes())
        want = recorded.get(path)
        if want is None:
            raise ValueError(f"the overlay does not record the candidate's {path}; "
                             "build it again with this checkout's build_overlay.py")
        if have != want:
            raise ValueError(f"{HERE / name} (blob {have[:12]}) differs from the candidate's {path} "
                             f"(blob {want[:12]}); render from a clean checkout of the candidate "
                             f"{manifest['candidate_commit'][:12]}")
        blobs[path] = have
    return blobs


def check_command(command: str, env_names: set[str]) -> None:
    """The kubelet must pass the container command on unchanged (bootstrap.sh compares itself with the tree)."""
    for match in KUBELET_EXPANSION.finditer(command):
        if match.group(1) is None or match.group(1) in env_names:
            raise SystemExit(f"bootstrap.sh contains {match.group()!r}, which the kubelet would rewrite in the "
                             "container command; write it differently")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=sorted(MODES), required=True)
    p.add_argument("--run", required=True, help="JobSet name (DNS label, at most 40 characters)")
    p.add_argument("--topology", choices=sorted(TOPOLOGIES), required=True)
    p.add_argument("--config", required=True,
                   help="recipe name in the config module; with --mode export, the PEFT adapter "
                        "directory relative to --data-dir (e.g. adapter/step-60)")
    p.add_argument("--config-module", help="TorchTitan --module (default depends on --mode)")
    p.add_argument("--entry", help="python entry module (default depends on --mode)")
    for option in ("bucket", "namespace", "local-queue", "project", "trainer-image"):
        p.add_argument("--" + option, required=True)
    p.add_argument("--code-overlay", type=Path, required=True,
                   help="directory written by build_overlay.py (overlay.tar.gz + overlay.json)")
    p.add_argument("--base-repo", default=DEFAULT_BASE_REPO,
                   help="public git repository the pods fetch the overlay's base commit from")
    p.add_argument("--prefix", default="muse-spark", help="storage prefix inside the bucket (runs/ goes here)")
    p.add_argument("--model-dir", default="muse-spark/model/Muse-Spark-Open-816B-A42B",
                   help="checkpoint directory inside the bucket")
    p.add_argument("--data-dir", required=True,
                   help="dataset directory inside the bucket, mounted at /mnt/data (the SFT recipes read "
                        "sft/train.jsonl there)")
    p.add_argument("--runtime-dir", help="mounted wheelhouse directory inside the bucket (default <prefix>/runtime)")
    p.add_argument("--runtime-mode", choices=RUNTIME_MODES, default="auto",
                   help="mounted: install the wheelhouse offline (runtime_install.sh); preinstalled: use the image; "
                        "auto: mounted when the wheelhouse has <runtime-name>-lock.txt")
    p.add_argument("--runtime-name", default="runtime", help="file prefix of the wheelhouse lists")
    p.add_argument("--runtime-lock-sha256", help="SHA-256 of <runtime-name>-lock.txt")
    p.add_argument("--runtime-wheels-sha256", help="SHA-256 of <runtime-name>-wheels.sha256")
    p.add_argument("--runtime-debs-sha256", help="SHA-256 of <runtime-name>-debs.sha256 (if the wheelhouse has one)")
    p.add_argument("--pip-overlay",
                   help="hash-pinned requirements file (path in the repository) installed in the pod with "
                        "--no-deps --require-hashes, e.g. "
                        "torchtitan/experiments/tpu/muse_spark/launch/pip-pins-jax0.12.0.1-tokamax0.0.17.txt")
    p.add_argument("--pip-find-links", help="pod directory to install --pip-overlay from, offline (e.g. /mnt/runtime)")
    p.add_argument("--pip-index-url", help="package index for --pip-overlay (pod default: https://pypi.org/simple)")
    p.add_argument("--extra-args", default="",
                   help="extra trainer flags, shell-quoted; pass as --extra-args='--flag=value ...'")
    p.add_argument("--budget-seconds", type=int, default=2400, help="trainer wall-clock budget inside the pod")
    p.add_argument("--deadline-seconds", type=int, default=2700, help="Job activeDeadlineSeconds")
    p.add_argument("--shm-size", default="64Gi", help="memory-backed /dev/shm size")
    p.add_argument("--work-size", default="15Gi",
                   help="/work emptyDir limit; TPU node boot disks often have only ~20 GB free")
    p.add_argument("--label", action="append", default=[], metavar="KEY=VALUE",
                   help="extra label on the JobSet, its pods and the ConfigMaps (repeatable)")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                   help="extra trainer environment variable, e.g. SPARK_STEPS=20 (repeatable; not for secrets)")
    p.add_argument("--service-account", default="default")
    p.add_argument("--priority-class", help="optional pod priority class chosen by the cluster operator")
    p.add_argument("--out", type=Path, required=True, help="new output directory")
    args = p.parse_args(argv)
    for option, value in vars(args).items():
        if isinstance(value, str) and (PLACEHOLDER.search(value) or any(ord(c) < 32 for c in value)):
            p.error(f"--{option.replace('_', '-')} must be a concrete value without control characters")
    labels = {}
    for item in args.label:
        key, sep, value = item.partition("=")
        if not sep or not re.fullmatch(r"([a-z0-9.-]+/)?[A-Za-z0-9]([-A-Za-z0-9_.]*[A-Za-z0-9])?", key) \
                or not re.fullmatch(r"([A-Za-z0-9]([-A-Za-z0-9_.]*[A-Za-z0-9])?)?", value) or len(value) > 63:
            p.error(f"--label {item!r} must be KEY=VALUE with Kubernetes label syntax")
        labels[key] = value
    args.labels = labels
    env = {}
    for item in args.env:
        key, sep, value = item.partition("=")
        if not sep or not ENV_NAME.fullmatch(key) or PLACEHOLDER.search(value) or any(ord(c) < 32 for c in value):
            p.error(f"--env {item!r} must be KEY=VALUE with an upper-case name and a concrete value")
        if SECRET_NAME.search(key):
            p.error(f"--env {key}: pass credentials through a Kubernetes Secret, not the rendered manifest")
        env[key] = value
    args.environment = env
    for option, limit in (("run", 40), ("namespace", 63), ("local_queue", 63), ("service_account", 63)):
        value = getattr(args, option)
        if not DNS_LABEL.fullmatch(value) or len(value) > limit:
            p.error(f"--{option.replace('_', '-')} must be a DNS label of at most {limit} characters")
    if not args.base_repo.startswith("https://"):
        p.error("--base-repo must be an https:// URL")
    for option in ("prefix", "model_dir", "data_dir", "runtime_dir"):
        value = getattr(args, option)
        if value is not None and not _relative_dir(value):
            p.error(f"--{option.replace('_', '-')} must be a relative path without empty or dot components")
    for option in ("runtime_lock_sha256", "runtime_wheels_sha256", "runtime_debs_sha256"):
        value = getattr(args, option)
        if value is not None and not SHA256.fullmatch(value):
            p.error(f"--{option.replace('_', '-')} must be a lowercase SHA-256 digest")
    if args.runtime_mode == "mounted" and not (args.runtime_lock_sha256 and args.runtime_wheels_sha256):
        p.error("--runtime-mode mounted needs --runtime-lock-sha256 and --runtime-wheels-sha256")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.runtime_name):
        p.error("--runtime-name must be a plain file prefix")
    if args.pip_overlay is not None and not _relative_dir(args.pip_overlay):
        p.error("--pip-overlay must be a path inside the repository")
    if (args.pip_find_links or args.pip_index_url) and not args.pip_overlay:
        p.error("--pip-find-links and --pip-index-url need --pip-overlay")
    if args.pip_find_links is not None and not args.pip_find_links.startswith("/"):
        p.error("--pip-find-links must be an absolute path inside the pod")
    if args.pip_index_url is not None and not args.pip_index_url.startswith("https://"):
        p.error("--pip-index-url must be an https:// URL")
    if min(args.budget_seconds, args.deadline_seconds) <= 0 or args.budget_seconds >= args.deadline_seconds:
        p.error("need 0 < --budget-seconds < --deadline-seconds")
    try:
        args.overlay_manifest, args.overlay_archive = build_overlay.load_manifest(args.code_overlay)
        args.kit = check_kit(args.overlay_manifest)
    except (OSError, ValueError, KeyError) as exc:
        p.error(f"--code-overlay: {exc}")
    if args.pip_overlay is not None:
        overlay_files = {f["path"] for f in args.overlay_manifest["files"]}
        if args.pip_overlay not in overlay_files and not (REPO_ROOT / args.pip_overlay).is_file():
            p.error(f"--pip-overlay {args.pip_overlay} is neither in the code overlay nor in this checkout")
    return args


def overlay_configmaps(args, labels: dict[str, str]) -> list[dict]:
    """The overlay archive in ConfigMaps of at most CHUNK_BYTES raw bytes; the first also holds overlay.json."""
    archive = args.overlay_archive
    chunks = [archive[i:i + CHUNK_BYTES] for i in range(0, len(archive), CHUNK_BYTES)]
    manifest_text = json.dumps(args.overlay_manifest, indent=1) + "\n"
    maps = []
    for index, chunk in enumerate(chunks):
        item = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": f"{args.run}-code-{index}", "namespace": args.namespace, "labels": dict(labels)},
            "binaryData": {f"overlay.tar.gz.{index}": base64.b64encode(chunk).decode()},
        }
        if index == 0:
            item["data"] = {"overlay.json": manifest_text}
        size = len(json.dumps(item))
        if size > CONFIGMAP_LIMIT:
            raise SystemExit(f"ConfigMap {index} would be {size} bytes, over the {CONFIGMAP_LIMIT}-byte limit")
        maps.append(item)
    return maps


def projected_sources(maps: list[dict]) -> list[dict]:
    sources = []
    for index, item in enumerate(maps):
        keys = [f"overlay.tar.gz.{index}"] + (["overlay.json"] if index == 0 else [])
        sources.append({"configMap": {"name": item["metadata"]["name"],
                                      "items": [{"key": key, "path": key} for key in keys]}})
    return sources


OWNER_SCRIPT = """#!/usr/bin/env bash
# Written by render.py for JobSet {run} in namespace {namespace}. Run it once after
# `kubectl create -f trainer.json`: it makes the JobSet the owner of its code ConfigMaps,
# so Kubernetes deletes them together with the JobSet (deleted by hand, or by
# ttlSecondsAfterFinished after the run). Arguments go to every kubectl call,
# e.g. --context NAME.
set -euo pipefail
uid=$(kubectl "$@" -n {namespace} get jobset {run} -o jsonpath='{{.metadata.uid}}')
[[ -n "$uid" ]] || {{ echo "JobSet {run} not found in {namespace}" >&2; exit 1; }}
patch='{{"metadata":{{"ownerReferences":[{{"apiVersion":"{api_version}","kind":"JobSet","name":"{name}","uid":"'"$uid"'"}}]}}}}'
for configmap in {configmaps}; do
  kubectl "$@" -n {namespace} patch configmap "$configmap" --type=merge -p "$patch"
done
echo "the code ConfigMaps of {run} now belong to JobSet uid $uid"
"""


def owner_script(jobset: dict, maps: list[dict]) -> str:
    """``own-configmaps.sh``: ownerReferences from the code ConfigMaps to the JobSet (needs its uid)."""
    meta = jobset["metadata"]
    return OWNER_SCRIPT.format(
        run=shlex.quote(meta["name"]), namespace=shlex.quote(meta["namespace"]),
        name=meta["name"], api_version=jobset["apiVersion"],
        configmaps=" ".join(shlex.quote(m["metadata"]["name"]) for m in maps))


def main(argv=None):
    args = parse_args(argv)
    hosts, accelerator = TOPOLOGIES[args.topology]
    entry = args.entry or MODES[args.mode][0]
    config_module = args.config_module or MODES[args.mode][1]
    manifest = args.overlay_manifest
    runtime_dir = args.runtime_dir or f"{args.prefix}/runtime"
    base_labels = {"app": "muse-spark-trainer", "muse-spark-run": args.run, "muse-spark-mode": args.mode}
    maps = overlay_configmaps(args, {**base_labels, "component": "code", **args.labels})
    command = (HERE / "bootstrap.sh").read_text()
    settings = {
        "<RUN_NAME>": args.run, "<MODE>": args.mode, "<ENTRY_MODULE>": entry,
        "<CONFIG_MODULE>": config_module, "<CONFIG>": args.config, "<EXTRA_ARGS>": args.extra_args,
        "<TPU_TOPOLOGY>": args.topology,
        "<BASE_REPO>": args.base_repo, "<BASE_COMMIT>": manifest["base_commit"], "<BASE_TREE>": manifest["base_tree"],
        "<CANDIDATE_COMMIT>": manifest["candidate_commit"], "<CANDIDATE_TREE>": manifest["candidate_tree"],
        "<OVERLAY_SHA256>": manifest["sha256"], "<OVERLAY_CHUNKS>": str(len(maps)),
        "<OVERLAY_SOURCES>": projected_sources(maps),
        "<BUDGET_S>": str(args.budget_seconds), "<DEADLINE_S>": args.deadline_seconds,
        "<NNODES>": hosts, "<GCS_BUCKET>": args.bucket, "<PREFIX>": args.prefix,
        "<MODEL_DIR>": args.model_dir, "<DATA_DIR>": args.data_dir,
        "<RUNTIME_DIR>": runtime_dir, "<K8S_NAMESPACE>": args.namespace,
        "<LOCAL_QUEUE>": args.local_queue, "<GCP_PROJECT>": args.project, "<TRAINER_IMAGE>": args.trainer_image,
        "<SERVICE_ACCOUNT>": args.service_account, "<SHM_SIZE>": args.shm_size, "<WORK_SIZE>": args.work_size,
        "<TRAINER_COMMAND>": command,
    }
    trainer = render_template("trainer_jobset.json.in", settings)
    pod_spec = trainer["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]
    container_env = pod_spec["containers"][0]["env"]
    # NNODES is an environment string; the Job counts are integers.
    for env in container_env:
        if env["name"] == "NNODES":
            env["value"] = str(hosts)
    kit_env = {"SPARK_RUNTIME_MODE": args.runtime_mode, "SPARK_RUNTIME_NAME": args.runtime_name}
    for option, name in (("runtime_lock_sha256", "SPARK_RUNTIME_LOCK_SHA256"),
                         ("runtime_wheels_sha256", "SPARK_RUNTIME_WHEELS_SHA256"),
                         ("runtime_debs_sha256", "SPARK_RUNTIME_DEBS_SHA256"),
                         ("pip_overlay", "SPARK_PIP_OVERLAY"), ("pip_find_links", "SPARK_PIP_FIND_LINKS"),
                         ("pip_index_url", "SPARK_PIP_INDEX_URL")):
        if getattr(args, option):
            kit_env[name] = getattr(args, option)
    reserved = sorted(set(args.environment) & ({env["name"] for env in container_env} | set(kit_env)))
    if reserved:
        raise SystemExit("--env cannot override variables the kit sets (use the dedicated options): "
                         + ", ".join(reserved))
    container_env.extend({"name": key, "value": value} for key, value in kit_env.items())
    container_env.extend({"name": key, "value": value} for key, value in sorted(args.environment.items()))
    check_command(command, {env["name"] for env in container_env})
    if args.priority_class:
        pod_spec["priorityClassName"] = args.priority_class
    if args.labels:
        trainer["metadata"]["labels"].update(args.labels)
        trainer["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["metadata"]["labels"].update(args.labels)
    document = {"apiVersion": "v1", "kind": "List", "items": [*maps, trainer]}
    receipt = {
        "kind": "muse-spark-render", "status": "RENDERED", "mode": args.mode, "run": args.run,
        "topology": args.topology, "hosts": hosts, "chips": 4 * hosts, "ranks": 8 * hosts,
        "accelerator_type": accelerator, "entry": entry, "config_module": config_module, "config": args.config,
        "extra_args": args.extra_args, "image": args.trainer_image,
        "code": {
            "delivery": "inline: public base commit + ConfigMap overlay, git tree checked in every pod",
            "base_repo": args.base_repo, "base_commit": manifest["base_commit"], "base_tree": manifest["base_tree"],
            "candidate_commit": manifest["candidate_commit"], "candidate_tree": manifest["candidate_tree"],
            "overlay_sha256": manifest["sha256"], "overlay_bytes": manifest["bytes"],
            "files": len(manifest["files"]), "deleted": len(manifest["deleted"]),
            "configmaps": [m["metadata"]["name"] for m in maps],
            "configmap_owner_script": "own-configmaps.sh",
            "launch_files": args.kit,
        },
        "runtime": {"mode": args.runtime_mode, "dir": f"gs://{args.bucket}/{runtime_dir}/", "name": args.runtime_name,
                    "pins": {k: v for k, v in kit_env.items() if k.endswith("_SHA256")},
                    "pip_overlay": args.pip_overlay, "pip_find_links": args.pip_find_links,
                    "pip_index_url": args.pip_index_url},
        "inputs": {"model": f"gs://{args.bucket}/{args.model_dir}/", "data": f"gs://{args.bucket}/{args.data_dir}/"},
        "outputs": f"gs://{args.bucket}/{args.prefix}/runs/{args.run}/",
        "namespace": args.namespace, "queue": args.local_queue, "project": args.project,
        "priority_class": args.priority_class,
        "budget_seconds": args.budget_seconds, "deadline_seconds": args.deadline_seconds, "labels": args.labels,
        "environment": args.environment,
        "unresolved_placeholders": [],
    }
    args.out.mkdir(parents=True, exist_ok=False, mode=0o700)
    data = (json.dumps(document, indent=2) + "\n").encode()
    fd = os.open(args.out / "trainer.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    script = owner_script(trainer, maps).encode()
    (args.out / "own-configmaps.sh").write_bytes(script)
    (args.out / "own-configmaps.sh").chmod(0o700)
    receipt["files"] = {"trainer.json": hashlib.sha256(data).hexdigest(),
                        "own-configmaps.sh": hashlib.sha256(script).hexdigest()}
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"rendered {args.mode} {args.config} on {args.topology} ({hosts} hosts, {4 * hosts} chips) "
          f"under {args.out}: {len(maps)} code ConfigMap(s) + the JobSet, suspended until the queue admits it; "
          f"candidate tree {manifest['candidate_tree']}. After `kubectl create -f {args.out / 'trainer.json'}`, "
          f"run `bash {args.out / 'own-configmaps.sh'} [--context NAME]` so the ConfigMaps go with the JobSet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
