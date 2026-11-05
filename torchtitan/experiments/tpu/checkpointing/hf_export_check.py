#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Full-scale check of the layout-preserving export on a real checkpoint.

Merges a deterministic random LoRA adapter (default rank 16 on the q/k/v/o
projections of every text layer) into a Hugging Face checkpoint with
``hf_export`` on several hosts and workers, then verifies the result:

* every output shard has the original header bytes and size;
* shards the adapter does not touch are byte-identical to the original
  (their streamed SHA-256 equals the published per-shard SHA-256, if given);
* sampled merged tensors equal ``bf16(W + alpha / r * B @ A)`` recomputed from
  the original weights.

Run one process per host (``--host i --hosts n``); each starts
``--workers-per-host`` export workers. Host 0 finalizes and verifies.
Example::

    python -m torchtitan.experiments.tpu.checkpointing.hf_export_check \\
        --base gs://<GCS_BUCKET>/muse-spark/model/Muse-Spark-Open-816B-A42B \\
        --out gs://<GCS_BUCKET>/muse-spark/runs/<RUN_NAME>/export-check \\
        --staging-dir /stage --workers-per-host 4 --hosts $NNODES --host $JOB_COMPLETION_INDEX \\
        --public-sha256 SHA256SUMS --report /tmp/export-check.json --cleanup
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import sys
import time

import torch

DEFAULT_TARGETS = r"model\.language_model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj\.weight"


def _modules():
    try:
        from torchtitan.experiments.tpu.checkpointing import gcs, hf_export, lora_checkpoint
    except Exception:  # noqa: BLE001 - standalone use next to the other files
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import gcs  # type: ignore
        import hf_export  # type: ignore
        import lora_checkpoint  # type: ignore
    return gcs, hf_export, lora_checkpoint


def make_adapter(layout, targets: str, rank: int, seed: int) -> dict[str, torch.Tensor]:
    """Deterministic random PEFT tensors for every weight matching ``targets``."""
    pat = re.compile(targets)
    out = {}
    for key in sorted(k for k in layout.weight_map if pat.fullmatch(k)):
        _, shape, _ = layout.tensor_info(key)
        module = key[: -len(".weight")]
        h = int(hashlib.sha256(f"{seed}:{module}".encode()).hexdigest()[:15], 16)
        g = torch.Generator().manual_seed(h)
        out[f"base_model.model.{module}.lora_A.weight"] = (torch.randn(rank, shape[1], generator=g) / shape[1] ** 0.5).to(torch.bfloat16)
        out[f"base_model.model.{module}.lora_B.weight"] = (0.02 * torch.randn(shape[0], rank, generator=g)).to(torch.bfloat16)
    return out


def _auth(gcs, mode: str, probe_uri: str | None = None):
    """(credentials, storage client, fsspec storage options); ``none`` = local paths only."""
    if mode == "none":
        return None, None, {}
    creds, _, _ = gcs.resolve_credentials(mode, impersonate=os.environ.get("GCS_IMPERSONATE_SA") or None,
                                          probe_uri=probe_uri)
    return creds, gcs.storage_client(creds), {"token": creds}


def _worker(args, worker: int, num_workers: int, adapter_dir: str, q) -> None:
    try:
        gcs, hf_export, _ = _modules()
        creds, client, storage = _auth(gcs, args.auth, args.out if args.out.startswith("gs://") else None)
        layout = hf_export.HfLayout(args.base, storage)
        source = hf_export.PeftSource(adapter_dir, layout)
        rep = hf_export.export_hf(
            args.base, args.out, source, worker=worker, num_workers=num_workers,
            storage_options=storage, out_storage_options=storage, staging_dir=args.staging_dir,
            gcs_client=client, sha256=True, finalize=False)
        q.put((worker, {k: v for k, v in rep.items() if k != "source"}))
    except Exception as exc:  # noqa: BLE001
        import traceback

        q.put((worker, {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-3000:]}))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--staging-dir", required=True)
    p.add_argument("--hosts", type=int, default=int(os.environ.get("NNODES", "1")))
    p.add_argument("--host", type=int, default=int(os.environ.get("JOB_COMPLETION_INDEX", "0")))
    p.add_argument("--workers-per-host", type=int, default=4)
    p.add_argument("--targets", default=DEFAULT_TARGETS)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--public-sha256", default="", help="sha256sum-style list of the original shards")
    p.add_argument("--verify-tensors", type=int, default=8)
    p.add_argument("--report", default="")
    p.add_argument("--cleanup", action="store_true", help="delete the export after verifying")
    p.add_argument("--finalize-timeout-s", type=float, default=3000)
    p.add_argument("--auth", default="auto")
    args = p.parse_args(argv)

    t0 = time.time()
    gcs, hf_export, lora_checkpoint = _modules()
    creds, client, storage = _auth(gcs, args.auth, args.out if args.out.startswith("gs://") else None)
    mode = args.auth
    layout = hf_export.HfLayout(args.base, storage)
    adapter = make_adapter(layout, args.targets, args.rank, args.seed)
    adapter_dir = os.path.join(args.staging_dir, "adapter")
    lora_checkpoint.write_peft_dir(adapter, adapter_dir, rank=args.rank, alpha=args.alpha,
                                   base_model_name_or_path=args.base)
    report: dict = {"kind": "hf_export_check_v1", "base": args.base, "hosts": args.hosts,
                    "workers_per_host": args.workers_per_host, "auth": mode,
                    "adapter_tensors": len(adapter), "adapter_params": sum(v.numel() for v in adapter.values()),
                    "shards": len(layout.files), "setup_s": round(time.time() - t0, 1)}
    print(f"[export-check host{args.host}] adapter {report['adapter_tensors']} tensors, "
          f"{report['adapter_params'] / 1e6:.1f}M params; {len(layout.files)} shards", flush=True)

    num_workers = args.hosts * args.workers_per_host
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    t_export = time.time()
    for j in range(args.workers_per_host):
        w = args.host * args.workers_per_host + j
        pr = ctx.Process(target=_worker, args=(args, w, num_workers, adapter_dir, q))
        pr.start()
        procs.append(pr)
    results = dict(q.get() for _ in procs)
    for pr in procs:
        pr.join()
    report["host_export_s"] = round(time.time() - t_export, 1)
    report["workers"] = results
    errors = {w: r["error"] for w, r in results.items() if "error" in r}
    print(f"[export-check host{args.host}] workers done in {report['host_export_s']}s; errors={errors}", flush=True)
    if errors or args.host != 0:
        if args.report:
            with open(args.report, "w") as f:
                json.dump(report, f, indent=1)
        return 1 if errors else 0

    # Host 0: wait for every worker manifest, then finalize and verify.
    import fsspec

    fs, _ = fsspec.core.url_to_fs(args.out if "://" in args.out else "file://" + os.path.abspath(args.out), **storage)
    deadline = time.time() + args.finalize_timeout_s
    while True:
        names = [hf_export._join(args.out, f"export_manifest.worker{w:05d}-of-{num_workers:05d}.json")
                 for w in range(num_workers)]
        missing = [n for n in names if not fs.exists(n)]
        if not missing:
            break
        if time.time() > deadline:
            raise TimeoutError(f"{len(missing)} worker manifests missing")
        time.sleep(10)
        fs.invalidate_cache()
    report["all_workers_s"] = round(time.time() - t_export, 1)
    t = time.time()
    report["finalize"] = hf_export.finalize_export(args.base, args.out, num_workers=num_workers,
                                                   storage_options=storage, out_storage_options=storage)
    report["finalize_s"] = round(time.time() - t, 1)
    manifest = hf_export._read_json(hf_export._join(args.out, hf_export.MANIFEST_NAME), storage)
    out_layout = hf_export.HfLayout(args.out, storage)

    # 1) headers and sizes
    bad_headers = [f for f in layout.files if out_layout.header(f)[0] != layout.header(f)[0]]
    report["header_bytes_identical"] = not bad_headers
    # 2) untouched shards byte-identical (streamed sha256 vs the published one)
    touched = {layout.weight_map[k[len("base_model.model."):-len(".lora_A.weight")] + ".weight"]
               for k in adapter if k.endswith(".lora_A.weight")}
    report["shards_touched"] = len(touched)
    if args.public_sha256:
        public = {}
        with open(args.public_sha256) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    public[parts[1]] = parts[0]
        untouched = [f for f in layout.files if f not in touched]
        same = [f for f in untouched if manifest["shards"][f]["sha256"] == public.get(f)]
        changed = [f for f in touched if manifest["shards"][f]["sha256"] != public.get(f)]
        report["untouched_shards_sha256_equal_public"] = f"{len(same)}/{len(untouched)}"
        report["touched_shards_sha256_differ"] = f"{len(changed)}/{len(touched)}"
    # 3) sampled merged tensors
    scale = args.alpha / args.rank
    keys = sorted(k[len("base_model.model."):-len(".lora_A.weight")] + ".weight"
                  for k in adapter if k.endswith(".lora_A.weight"))
    step = max(1, len(keys) // max(1, args.verify_tensors))
    checked, mismatched = 0, []
    for key in keys[::step][: args.verify_tensors]:
        module = key[: -len(".weight")]
        a = adapter[f"base_model.model.{module}.lora_A.weight"].float()
        b = adapter[f"base_model.model.{module}.lora_B.weight"].float()
        w = layout.read_tensor(key)
        want = (w.float() + scale * (b @ a)).to(w.dtype)
        got = out_layout.read_tensor(key)
        checked += 1
        if not torch.equal(want, got):
            mismatched.append(key)
    report["merged_tensors_checked"] = checked
    report["merged_tensors_bit_exact"] = not mismatched
    report["bytes"] = manifest["bytes"]
    report["export_GBps_aggregate"] = round(manifest["bytes"] / report["all_workers_s"] / 1e9, 3)
    if args.cleanup:
        t = time.time()
        if client is not None and args.out.startswith("gs://"):
            report["deleted_objects"] = gcs.delete_prefix(client, gcs.parse_gs_uri(args.out))
        else:
            import shutil

            shutil.rmtree(args.out, ignore_errors=True)
        report["cleanup_s"] = round(time.time() - t, 1)
    report["wall_s"] = round(time.time() - t0, 1)
    print("[export-check] " + json.dumps({k: v for k, v in report.items() if k != "workers"}), flush=True)
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=1)
    ok = report["header_bytes_identical"] and report["merged_tensors_bit_exact"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
