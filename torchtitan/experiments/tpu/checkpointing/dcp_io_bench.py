#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DCP checkpoint I/O benchmark for multi-host TPU slices (also runs on CPU).

The benchmark builds a synthetic training state shaped like a large MoE
decoder: bf16 parameters plus two 1-byte optimizer moments per parameter
(8-bit AdamW). Every tensor is a ``DTensor`` sharded on dim 0 over all ranks,
and each rank's local shard has the size it would have on a slice of
``--ref-devices`` devices. Running on a smaller slice therefore reproduces
the per-device (and per-host) checkpoint volume of the large slice; total
volume scales with the number of hosts, which is what a proportional test
needs.

Methods (``--methods``):

``fs``      ``torch.distributed.checkpoint`` (DCP) ``FileSystemWriter`` /
            ``FileSystemReader`` on a POSIX path, e.g. a Cloud Storage FUSE
            mount. Reads may use a second mount of the same prefix
            (``--fs-read-root``) so that the kernel page cache of the
            writing mount cannot serve them.
``fsspec``  DCP ``FsspecWriter`` / ``FsspecReader`` on a ``gs://`` URL (gcsfs,
            direct Cloud Storage).
``stage``   DCP save to local storage (e.g. a memory-backed volume), then
            every rank uploads its own files with parallel multipart
            uploads; restore downloads with parallel ranged reads and loads
            locally. The save blocks training only for the local part.

Each load goes into freshly zeroed tensors and is verified bit-exact against
the saved tensors. Results (JSON) are written by rank 0 after every phase.

Example (per host, under torchrun with 8 ranks per TPU7x host)::

    python -m torch.distributed.run --nnodes=$NNODES --nproc-per-node=8 ... \\
        dcp_io_bench.py --device tpu --preset moe-816b --ref-devices 128 \\
        --methods fs,fsspec,stage --fs-root /mnt/ckpt-w --fs-read-root /mnt/ckpt-r \\
        --gs-root gs://<GCS_BUCKET>/<PREFIX>/bench --stage-root /stage \\
        --out /mnt/out/results.json
"""

from __future__ import annotations

import argparse
import functools
import glob
import json
import math
import os
import shutil
import socket
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any, Callable

T0 = time.time()


# ---------------------------------------------------------------------------
# Synthetic state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoEDecoderShape:
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    experts: int
    moe_hidden: int
    moe_inter: int
    vocab: int


PRESETS = {
    # 62 sparse layers, 256 routed experts in a 4096-wide expert space,
    # gated attention output, untied embeddings (Muse Spark Open 816B-A42B).
    "moe-816b": MoEDecoderShape(62, 8192, 128, 16, 64, 256, 4096, 4096, 202048),
    "tiny": MoEDecoderShape(4, 64, 4, 2, 16, 16, 32, 32, 512),
}


def param_shapes(s: MoEDecoderShape, layers: int) -> list[tuple[str, tuple[int, ...], str]]:
    """(name, global shape, dtype) of every parameter of the first ``layers``."""
    q = s.heads * s.head_dim
    kv = s.kv_heads * s.head_dim
    out = [
        ("tok_embeddings.weight", (s.vocab, s.hidden), "bf16"),
        ("norm.weight", (s.hidden,), "bf16"),
        ("lm_head.weight", (s.vocab, s.hidden), "bf16"),
    ]
    for i in range(layers):
        p = f"layers.{i}."
        out += [
            (p + "attention_norm.weight", (s.hidden,), "bf16"),
            (p + "attention.wq.weight", (q, s.hidden), "bf16"),
            (p + "attention.wk.weight", (kv, s.hidden), "bf16"),
            (p + "attention.wv.weight", (kv, s.hidden), "bf16"),
            (p + "attention.wo.weight", (s.hidden, q), "bf16"),
            (p + "attention.output_gate.weight", (q, s.hidden), "bf16"),
            (p + "attention_residual.gate", (s.hidden,), "bf16"),
            (p + "ffn_norm.weight", (s.hidden,), "bf16"),
            (p + "post_ffn_norm.weight", (s.hidden,), "bf16"),
            (p + "ffn_residual.gate", (s.hidden,), "bf16"),
            (p + "moe.router.weight", (s.experts, s.hidden), "fp32"),
            (p + "moe.expert_bias", (s.experts,), "fp32"),
            (p + "moe.pre_proj.weight", (s.moe_hidden, s.hidden), "bf16"),
            (p + "moe.pre_norm.weight", (s.moe_hidden,), "bf16"),
            (p + "moe.experts.w1", (s.experts, s.moe_inter, s.moe_hidden), "bf16"),
            (p + "moe.experts.w3", (s.experts, s.moe_inter, s.moe_hidden), "bf16"),
            (p + "moe.experts.w2", (s.experts, s.moe_hidden, s.moe_inter), "bf16"),
            (p + "moe.experts.out_norm.weight", (s.moe_hidden,), "bf16"),
            (p + "moe.post_proj.weight", (s.hidden, s.moe_hidden), "bf16"),
        ]
    return out


def state_layout(
    s: MoEDecoderShape, layers: int, ref_devices: int, world: int, moments: bool = True
) -> list[tuple[str, tuple[int, ...], tuple[int, ...], str]]:
    """(key, local shape, global shape, dtype) for every saved tensor."""
    items = []
    for name, shape, dtype in param_shapes(s, layers):
        rows = math.ceil(shape[0] / ref_devices)
        local = (rows, *shape[1:])
        glob_ = (rows * world, *shape[1:])
        items.append((f"model.{name}", local, glob_, dtype))
        if moments:
            items.append((f"optim.{name}.exp_avg", local, glob_, "u8"))
            items.append((f"optim.{name}.exp_avg_sq", local, glob_, "u8"))
    return items


def full_model_bytes(s: MoEDecoderShape, moments: bool = True) -> int:
    total = 0
    for _, shape, dtype in param_shapes(s, s.layers):
        n = math.prod(shape)
        total += n * (4 if dtype == "fp32" else 2) + (2 * n if moments else 0)
    return total


# ---------------------------------------------------------------------------
# Distributed plumbing
# ---------------------------------------------------------------------------


class Ctx:
    def __init__(self, device_kind: str):
        import torch
        import torch.distributed as dist

        self.torch = torch
        self.dist = dist
        self.kind = device_kind
        self.rank = int(os.environ.get("RANK", "0"))
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        self.node = self.rank // max(self.local_world, 1)
        self.nodes = max(self.world // max(self.local_world, 1), 1)
        self._tpu_sync = None
        if device_kind == "tpu":
            import torch_tpu  # noqa: F401

            try:
                from torch_tpu._internal import sync as tpu_sync

                self._tpu_sync = tpu_sync
            except Exception:
                self._tpu_sync = None
            self.device = torch.device("tpu")
            backend = "tpu_dist"
        else:
            self.device = torch.device("cpu")
            backend = "gloo"
        t = time.time()
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, rank=self.rank, world_size=self.world)
        self.init_s = time.time() - t
        from torch.distributed.device_mesh import init_device_mesh

        self.mesh = init_device_mesh(device_kind, (self.world,))

    def sync(self) -> None:
        if self._tpu_sync is not None:
            self._tpu_sync.synchronize(wait=True)
        elif self.kind == "cpu":
            return
        else:
            self.torch.accelerator.synchronize()

    def barrier(self) -> None:
        t = self.torch.ones(1, device=self.device)
        self.dist.all_reduce(t)
        float(t.cpu()[0])

    def gather(self, obj: Any) -> list[Any]:
        out: list[Any] = [None] * self.world
        self.dist.all_gather_object(out, obj)
        return out

    def log(self, msg: str, all_ranks: bool = False) -> None:
        if all_ranks or self.rank == 0:
            print(
                f"[CKB r{self.rank} n{self.node} {time.strftime('%H:%M:%S')} "
                f"+{time.time() - T0:.1f}s] {msg}",
                flush=True,
            )


def _dtype(ctx: Ctx, name: str):
    t = ctx.torch
    return {"bf16": t.bfloat16, "fp32": t.float32, "u8": t.uint8}[name]


def _fill_random_(ctx: Ctx, t) -> None:
    torch = ctx.torch
    if t.dtype == torch.uint8:
        try:
            t.copy_(torch.randint(0, 256, tuple(t.shape), dtype=torch.uint8, device=t.device))
            return
        except Exception:  # some backends lack a uint8 randint; use scaled uniform floats
            pass
        t.copy_((torch.rand(tuple(t.shape), device=t.device) * 255.0).to(torch.uint8))
    else:
        t.normal_()


def build_state(ctx: Ctx, layout, fill: bool) -> dict[str, Any]:
    from torch.distributed.tensor import DTensor, Shard

    torch = ctx.torch
    state = {}
    for key, local, glob_, dtype in layout:
        if fill:
            t = torch.empty(local, dtype=_dtype(ctx, dtype), device=ctx.device)
            _fill_random_(ctx, t)
        else:
            t = torch.zeros(local, dtype=_dtype(ctx, dtype), device=ctx.device)
        stride = [1] * len(glob_)
        for i in range(len(glob_) - 2, -1, -1):
            stride[i] = stride[i + 1] * glob_[i + 1]
        state[key] = DTensor.from_local(
            t, ctx.mesh, [Shard(0)], run_check=False, shape=glob_, stride=tuple(stride)
        )
    ctx.sync()
    return state


def local_bytes(state: dict[str, Any]) -> int:
    return sum(v.to_local().numel() * v.to_local().element_size() for v in state.values())


def verify(ctx: Ctx, saved: dict[str, Any], loaded: dict[str, Any]) -> list[str]:
    torch = ctx.torch
    bad = []
    for key, ref in saved.items():
        if not torch.equal(ref.to_local(), loaded[key].to_local()):
            bad.append(key)
    return bad


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------


def _fsspec_kwargs(args, creds) -> dict[str, Any]:
    kw: dict[str, Any] = {}
    if creds is not None:
        kw["token"] = creds
    if args.gcsfs_block_mb:
        kw["block_size"] = args.gcsfs_block_mb << 20
    return kw


def _transfer_fns(args, run: str):
    """(upload, download, remote root) for gs:// or, in tests, file:// roots."""
    if args.gs_root.startswith("file://"):
        root = args.gs_root[len("file://"):].rstrip("/") + "/" + run

        class _Local:
            def __init__(self, path):
                self.path = path

            def join(self, name):
                return _Local(os.path.join(self.path, name))

        def up(client, src, dst, **kw):
            os.makedirs(os.path.dirname(dst.path), exist_ok=True)
            shutil.copyfile(src, dst.path)
            return os.path.getsize(src)

        def down(client, src, dst, **kw):
            shutil.copyfile(src.path, dst)
            return os.path.getsize(dst)

        return up, down, _Local(root)
    gcs = _import_gcs()
    return gcs.upload_file, gcs.download_file, gcs.parse_gs_uri(args.gs_root).join(run)


def fsspec_reader(url: str, cache_type: str, **kw):
    """DCP ``FsspecReader`` whose files are opened with an explicit cache type.

    Recent gcsfs releases default to an ``adaptive`` read cache that only
    exists in matching fsspec releases; a runtime that pins an older fsspec
    fails on every read unless the cache type is given explicitly.
    """
    from contextlib import contextmanager

    from torch.distributed.checkpoint import FsspecReader

    reader = FsspecReader(url, **kw)
    if url.startswith("file://"):
        return reader
    wrapper = reader.fs

    @contextmanager
    def create_stream(path, mode):
        with wrapper.fs.open(path, mode, cache_type=cache_type) as stream:
            yield stream

    wrapper.create_stream = create_stream
    return reader


def raw_read_peer(ctx: Ctx, fs, root: str, fsspec_mode: bool = False, cache_type: str = "readahead") -> int:
    """Read (and discard) the files a rank on another host wrote: raw I/O rate."""
    peer = (ctx.rank + ctx.local_world) % ctx.world
    n = 0
    if fsspec_mode:
        names = [p for p in fs.ls(root.split("://", 1)[1], detail=False)
                 if os.path.basename(p).startswith(f"__{peer}_")]
        for p in names:
            opts = {} if root.startswith("file://") else {"cache_type": "none"}
            with fs.open(p, "rb", **opts) as f:
                while True:
                    b = f.read(256 << 20)
                    if not b:
                        break
                    n += len(b)
    else:
        for p in sorted(glob.glob(os.path.join(root, f"__{peer}_*"))):
            with open(p, "rb", buffering=0) as f:
                buf = bytearray(64 << 20)
                while True:
                    k = f.readinto(buf)
                    if not k:
                        break
                    n += k
    return n


def run_method(ctx: Ctx, args, method: str, state, layout, tag: str, threads: int, creds, client,
               full: bool = False) -> dict:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
    from torch.distributed.checkpoint.api import CheckpointException

    try:
        from torchtitan.experiments.tpu.checkpointing.dcp_parallel import (
            ParallelFileSystemReader,
            ParallelFsspecReader,
        )
    except Exception:  # standalone script
        from dcp_parallel import ParallelFileSystemReader, ParallelFsspecReader  # type: ignore

    rec: dict[str, Any] = {"method": method, "tag": tag, "threads": threads}
    my_bytes = local_bytes(state)
    per_rank = ctx.gather(my_bytes)
    total = sum(per_rank)
    rec["bytes_total"] = total
    rec["bytes_per_host"] = total / ctx.nodes
    rec["keys"] = len(state)
    run = f"{args.run_id}/{tag}-{method}-t{threads}"

    def timed(fn: Callable[[], Any]) -> float:
        """Time ``fn`` on every rank; if any rank fails, every rank raises."""
        ctx.barrier()
        t = time.perf_counter()
        err = None
        try:
            fn()
            ctx.sync()
        except (Exception, CheckpointException) as exc:
            err = f"{type(exc).__name__}: {str(exc)[:300]}"
            ctx.log(f"rank-local failure: {err}\n{traceback.format_exc()[-2000:]}", all_ranks=True)
        errs = [e for e in ctx.gather(err) if e]
        if errs:
            raise RuntimeError(f"{len(errs)} rank(s) failed: {errs[0]}")
        ctx.barrier()
        return time.perf_counter() - t

    def load_and_verify(key: str, make_reader: Callable[[], Any]) -> None:
        fresh = build_state(ctx, layout, fill=False)
        rec[key] = timed(lambda: dcp.load(fresh, storage_reader=make_reader()))
        bad = [b for lst in ctx.gather(verify(ctx, state, fresh)) for b in lst]
        rec[key.replace("_s", "_bit_exact")] = not bad
        if bad:
            rec.setdefault("verify_examples", sorted(set(bad))[:5])
        del fresh

    try:
        if method == "fs":
            wpath = os.path.join(args.fs_root, run)
            rpath = os.path.join(args.fs_read_root or args.fs_root, run)
            if ctx.rank == 0:
                os.makedirs(wpath, exist_ok=True)
            ctx.barrier()
            rec["save_s"] = timed(lambda: dcp.save(state, storage_writer=FileSystemWriter(
                wpath, single_file_per_rank=True, sync_files=True, thread_count=threads)))
            if not (full and args.par_threads):
                load_and_verify("load_s", lambda: FileSystemReader(rpath))
            if args.par_threads:
                load_and_verify("load_par_s", lambda: ParallelFileSystemReader(rpath, read_threads=args.par_threads))
            if args.raw_read and not full:
                rec["raw_read_s"] = timed(lambda: raw_read_peer(ctx, None, rpath))
        elif method == "fsspec":
            url = args.gs_root.rstrip("/") + "/" + run
            kw = _fsspec_kwargs(args, creds)
            rec["save_s"] = timed(lambda: dcp.save(state, storage_writer=_fsspec_writer(
                url, threads, **kw)))
            if not (full and args.par_threads):
                load_and_verify("load_s", lambda: fsspec_reader(url, args.gcsfs_cache, **kw))
            if args.par_threads:
                load_and_verify("load_par_s", lambda: ParallelFsspecReader(
                    url, read_threads=args.par_threads, cache_type=args.gcsfs_cache, **kw))
            if args.raw_read and not full:
                fs = fsspec_reader(url, args.gcsfs_cache, **kw).fs.fs
                rec["raw_read_s"] = timed(lambda: raw_read_peer(ctx, fs, url, fsspec_mode=True))
        elif method == "stage":
            upload_file, download_file, remote = _transfer_fns(args, run)
            local = os.path.join(args.stage_root, run)
            os.makedirs(local, exist_ok=True)
            ctx.barrier()
            rec["save_local_s"] = timed(lambda: dcp.save(state, storage_writer=FileSystemWriter(
                local, single_file_per_rank=True, sync_files=False, thread_count=threads)))
            mine = sorted(glob.glob(os.path.join(local, f"__{ctx.rank}_*")))
            meta = os.path.join(local, ".metadata")

            def upload():
                for path in mine:
                    upload_file(client, path, remote.join(os.path.basename(path)),
                                chunk_size=args.part_mb << 20, max_workers=args.parts)
                if ctx.rank == 0:
                    upload_file(client, meta, remote.join(".metadata"))

            rec["upload_s"] = timed(upload)

            def drop_local():
                for path in mine:
                    os.remove(path)
                if ctx.local_rank == 0 and os.path.exists(meta):
                    os.remove(meta)

            timed(drop_local)

            def download():
                for path in mine:
                    download_file(client, remote.join(os.path.basename(path)), path,
                                  chunk_size=args.part_mb << 20, max_workers=args.parts)
                if ctx.local_rank == 0:
                    download_file(client, remote.join(".metadata"), meta)

            rec["download_s"] = timed(download)
            if not (full and args.par_threads):
                load_and_verify("load_local_s", lambda: FileSystemReader(local))
            if args.par_threads:
                load_and_verify("load_local_par_s", lambda: ParallelFileSystemReader(
                    local, read_threads=args.par_threads))
            rec["save_s"] = rec["save_local_s"] + rec["upload_s"]
            rec["load_s"] = rec["download_s"] + rec.get("load_local_par_s", rec.get("load_local_s", 0.0))
        else:
            raise ValueError(method)
        exact = [v for k, v in rec.items() if k.endswith("_bit_exact")]
        rec["verify_bit_exact"] = bool(exact) and all(exact)
    except (Exception, CheckpointException) as exc:  # record, go on
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:600]}"
        ctx.log(f"{method} {tag} FAILED: {rec['error']}")
    finally:
        try:
            cleanup(ctx, args, method, run, client, creds)
        except (Exception, CheckpointException) as exc:
            rec["cleanup_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    for k in [k for k in rec if k.endswith("_s") and isinstance(rec[k], float)]:
        rec[k] = round(rec[k], 3)
        rec[k[:-2] + "_GBps_per_host"] = round(rec["bytes_per_host"] / rec[k] / 1e9, 3) if rec[k] > 0 else None
    return rec


def _fsspec_writer(url: str, threads: int, **kw):
    from torch.distributed.checkpoint import FsspecWriter

    return FsspecWriter(url, single_file_per_rank=True, sync_files=False, thread_count=threads, **kw)


def cleanup(ctx: Ctx, args, method: str, run: str, client, creds) -> None:
    ctx.barrier()
    if method == "stage":
        local = os.path.join(args.stage_root, run)
        for path in glob.glob(os.path.join(local, f"__{ctx.rank}_*")):
            try:
                os.remove(path)
            except OSError:
                pass
        ctx.barrier()
        if ctx.local_rank == 0:
            shutil.rmtree(local, ignore_errors=True)
    if not args.keep and ctx.rank == 0:
        if method == "fs":
            shutil.rmtree(os.path.join(args.fs_root, run), ignore_errors=True)
        elif method in ("fsspec", "stage") and args.gs_root.startswith("file://"):
            shutil.rmtree(args.gs_root[len("file://"):].rstrip("/") + "/" + run, ignore_errors=True)
        elif method in ("fsspec", "stage") and client is not None:
            gcs = _import_gcs()
            gcs.delete_prefix(client, gcs.parse_gs_uri(args.gs_root).join(run))
    ctx.barrier()


def d2h_probe(ctx: Ctx, state, limit_bytes: int) -> dict:
    """Device->host and host->device copy rate of this rank's shards."""
    torch = ctx.torch
    picked, n = [], 0
    for v in state.values():
        t = v.to_local()
        if n >= limit_bytes:
            break
        picked.append(t)
        n += t.numel() * t.element_size()
    ctx.barrier()
    t0 = time.perf_counter()
    host = [t.cpu() for t in picked]
    d2h = time.perf_counter() - t0
    ctx.barrier()
    t0 = time.perf_counter()
    back = [h.to(ctx.device) for h in host]
    ctx.sync()
    h2d = time.perf_counter() - t0
    del host, back
    per = ctx.gather({"bytes": n, "d2h_s": d2h, "h2d_s": h2d})
    hosts = ctx.nodes
    b = sum(p["bytes"] for p in per)
    return {
        "bytes_per_rank": n,
        "d2h_GBps_per_host": round(b / hosts / max(p["d2h_s"] for p in per) / 1e9, 3),
        "h2d_GBps_per_host": round(b / hosts / max(p["h2d_s"] for p in per) / 1e9, 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@functools.cache
def _import_gcs():
    """Import ``gcs.py`` from this package, or from next to this file."""
    try:
        from torchtitan.experiments.tpu.checkpointing import gcs
    except Exception:  # running as a standalone script
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import gcs  # type: ignore[no-redef]
    return gcs


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", choices=("tpu", "cpu"), default="tpu",
                   help="where the synthetic state lives: TorchTPU devices, or the host with the gloo backend")
    p.add_argument("--preset", choices=sorted(PRESETS), default="moe-816b",
                   help="decoder shape whose per-device checkpoint volume is reproduced")
    p.add_argument("--ref-devices", type=int, default=128,
                   help="per-device shard sizes are those of a slice with this many devices")
    p.add_argument("--no-moments", action="store_true", help="parameters only (no optimizer state)")
    p.add_argument("--methods", default="fs,fsspec,stage", help="comma-separated save/load methods to time")
    p.add_argument("--cal-layers", type=int, default=4, help="layers in the calibration pass (0 = skip)")
    p.add_argument("--full-layers", type=int, default=-1, help="layers in the main pass (-1 = all)")
    p.add_argument("--min-full-layers", type=int, default=8,
                   help="fewest layers the main pass may be cut down to when the budget runs short")
    p.add_argument("--threads", default="1", help="writer thread counts to calibrate, e.g. 1,4")
    p.add_argument("--fs-root", default="", help="POSIX folder for the fs method, for example a FUSE mount")
    p.add_argument("--fs-read-root", default="", help="second mount of --fs-root for reads")
    p.add_argument("--gs-root", default="",
                   help="Cloud Storage prefix for the fsspec and stage methods (tests pass a file:// root)")
    p.add_argument("--stage-root", default="", help="local folder the stage method saves into before uploading")
    p.add_argument("--gcsfs-block-mb", type=int, default=0,
                   help="gcsfs block size in MiB (0 keeps the gcsfs default)")
    p.add_argument("--gcsfs-cache", default="readahead", help="fsspec cache type for DCP reads")
    p.add_argument("--raw-read", action="store_true", help="also time raw reads of a peer host's files")
    p.add_argument("--par-threads", type=int, default=0,
                   help="also time loads with the parallel DCP reader using this many threads per rank")
    p.add_argument("--parts", type=int, default=8, help="parallel parts per object (stage)")
    p.add_argument("--part-mb", type=int, default=64, help="size in MiB of each part of a stage transfer")
    p.add_argument("--auth", choices=("auto", "ksa", "default", "none"), default="auto",
                   help="Cloud Storage credential mode; none skips credentials (for file:// roots)")
    p.add_argument("--ksa-token-path", default="/var/run/secrets/gcs-ksa/token",
                   help="projected Kubernetes service account token read by the ksa mode")
    p.add_argument("--wi-audience", default=None,
                   help="audience for the token exchange (derived from the token when not given)")
    p.add_argument("--impersonate", default=os.environ.get("GCS_IMPERSONATE_SA") or None,
                   help="Google service account bound to the pod's Kubernetes service account")
    p.add_argument("--budget-s", type=float, default=2000.0, help="wall budget from process start")
    p.add_argument("--d2h-probe-gb", type=float, default=2.0,
                   help="GB of this rank's shards copied device -> host and back as a probe (0 = skip)")
    p.add_argument("--keep", action="store_true", help="keep written checkpoints")
    p.add_argument("--run-id", default=time.strftime("ckb-%Y%m%d-%H%M%S"),
                   help="name of this run's subfolder under each storage root")
    p.add_argument("--out", default="", help="results JSON that rank 0 rewrites as phases finish (empty: none)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.device == "tpu":
        try:  # rewrites TPU_* env per rank before the first device use
            from torchtitan.experiments.tpu import distributed_utils

            distributed_utils.maybe_init_distributed()
        except Exception as exc:
            print(f"[CKB] maybe_init_distributed skipped: {exc!r}", flush=True)
    ctx = Ctx(args.device)
    gcs = _import_gcs()
    shape = PRESETS[args.preset]
    methods = [m for m in args.methods.split(",") if m]
    threads = [int(t) for t in args.threads.split(",") if t]
    results: dict[str, Any] = {
        "kind": "dcp_io_bench_v1", "args": vars(args), "world": ctx.world, "hosts": ctx.nodes,
        "ranks_per_host": ctx.local_world, "init_process_group_s": round(ctx.init_s, 3),
        "shape": asdict(shape), "hostname0": socket.gethostname(),
        "full_model_state_bytes": full_model_bytes(shape, not args.no_moments),
        "phases": [],
    }
    import torch

    results["versions"] = {"torch": torch.__version__}
    for mod in ("torch_tpu", "gcsfs", "fsspec", "google.cloud.storage"):
        try:
            m = __import__(mod, fromlist=["x"])
            results["versions"][mod] = getattr(m, "__version__", "?")
        except Exception:
            pass

    def dump():
        if ctx.rank == 0 and args.out:
            tmp = args.out + ".tmp"
            with open(tmp, "w") as f:
                json.dump(results, f, indent=1, default=str)
            os.replace(tmp, args.out)

    creds = client = None
    if any(m in ("fsspec", "stage") for m in methods) and args.auth != "none":
        try:
            creds, mode, notes = gcs.resolve_credentials(
                args.auth, ksa_token_path=args.ksa_token_path, audience=args.wi_audience,
                impersonate=args.impersonate, probe_uri=args.gs_root)
            client = gcs.storage_client(creds)
            results["auth"] = {"mode": mode, "notes": notes}
        except Exception as exc:
            results["auth"] = {"error": f"{type(exc).__name__}: {str(exc)[:800]}"}
            methods = [m for m in methods if m not in ("fsspec", "stage")]
        auth_ok = ctx.gather(creds is not None)
        if not all(auth_ok):
            methods = [m for m in methods if m not in ("fsspec", "stage")]
            results["auth"]["all_ranks_ok"] = False
    ctx.log(f"world={ctx.world} hosts={ctx.nodes} methods={methods} auth={results.get('auth')}")
    dump()

    full_layers = shape.layers if args.full_layers < 0 else args.full_layers
    t = time.time()
    state = build_state(ctx, state_layout(shape, full_layers, args.ref_devices, ctx.world,
                                          not args.no_moments), fill=True)
    results["alloc_fill_s"] = round(time.time() - t, 3)
    full_keys = list(state)
    ctx.log(f"state: {len(state)} tensors, {local_bytes(state) / 1e9:.2f} GB per rank "
            f"({full_layers} layers), alloc+fill {results['alloc_fill_s']}s")

    if args.d2h_probe_gb > 0:
        try:
            results["d2h_probe"] = d2h_probe(ctx, state, int(args.d2h_probe_gb * 1e9))
            ctx.log(f"d2h probe {results['d2h_probe']}")
        except Exception as exc:
            results["d2h_probe"] = {"error": f"{type(exc).__name__}: {exc}"}
        dump()

    def subset(layers: int) -> tuple[dict, list]:
        lay = state_layout(shape, layers, args.ref_devices, ctx.world, not args.no_moments)
        return {k: state[k] for k, *_ in lay}, lay

    best: dict[str, int] = {}
    cal_rate: dict[str, float] = {}
    if args.cal_layers > 0:
        sub, lay = subset(min(args.cal_layers, full_layers))
        for method in methods:
            for th in threads:
                if time.time() - T0 > args.budget_s:
                    break
                rec = run_method(ctx, args, method, sub, lay, f"cal{args.cal_layers}", th, creds, client)
                results["phases"].append(rec)
                ctx.log(f"PHASE {json.dumps(rec)}")
                dump()
                if "error" not in rec and rec.get("verify_bit_exact"):
                    # What a full phase runs: save + (parallel, if measured) load.
                    if method == "stage":
                        load = rec["load_s"]
                    else:
                        load = rec.get("load_par_s", rec.get("load_s"))
                    rate = rec["bytes_per_host"] / (rec["save_s"] + load)
                    if rate > cal_rate.get(method, 0.0):
                        cal_rate[method], best[method] = rate, th

    todo = [m for m in methods if args.cal_layers <= 0 or m in cal_rate]
    for i, method in enumerate(todo):
        th = best.get(method, threads[0])
        # Split what is left evenly over the remaining methods.
        remaining = (args.budget_s - (time.time() - T0)) / (len(todo) - i)
        layers = full_layers
        if method in cal_rate:
            per_layer = local_bytes(subset(1)[0]) - local_bytes(subset(0)[0])
            base = local_bytes(subset(0)[0])
            host_bytes = lambda n: (base + n * per_layer) * ctx.local_world  # noqa: E731
            est = lambda n: host_bytes(n) / cal_rate[method] * 1.25 + 60  # noqa: E731
            while layers > args.min_full_layers and est(layers) > remaining:
                layers = max(args.min_full_layers, layers // 2)
            if est(layers) > remaining:
                ctx.log(f"skip full {method}: est {est(layers):.0f}s > remaining {remaining:.0f}s")
                results["phases"].append({"method": method, "tag": "full", "skipped": True,
                                          "est_s": round(est(layers)), "remaining_s": round(remaining)})
                dump()
                continue
        sub, lay = subset(layers)
        rec = run_method(ctx, args, method, sub, lay, f"full{layers}", th, creds, client, full=True)
        rec["layers"] = layers
        results["phases"].append(rec)
        ctx.log(f"PHASE {json.dumps(rec)}")
        dump()

    del state, full_keys
    results["wall_s"] = round(time.time() - T0, 1)
    dump()
    ctx.log("DONE")
    ctx.barrier()
    ctx.dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
