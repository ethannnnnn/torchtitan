# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Faster DCP I/O for large checkpoints on Cloud Storage (FUSE or gcsfs).

Measured on TPU7x hosts (see ``dcp_io_bench``), the stock DCP paths are
limited by per-rank serialization, not by storage:

* ``FileSystemWriter`` defaults to one writer thread per rank; with 8
  threads per rank a save is ~3.4x faster on a Cloud Storage FUSE mount.
* ``FileSystemReader`` reads every item of a rank sequentially through
  ``torch.load`` on a lazy file view (many small reads on a remote file).

``ParallelFileSystemReader`` / ``ParallelFsspecReader`` read each item with
one large ranged read into memory, deserialize it, and copy it to its
destination from a pool of threads (each with its own file handles).
``tune_checkpoint_manager`` wires both into a TorchTitan
``CheckpointManager`` without changing its configuration surface.
"""

from __future__ import annotations

import io
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import torch
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.planner import LoadItemType, LoadPlan, LoadPlanner
from torch.futures import Future

__all__ = ["ParallelReadMixin", "ParallelFileSystemReader", "ParallelFsspecReader", "tune_checkpoint_manager"]


def _read_exact(stream, offset: int, length: int) -> bytes:
    stream.seek(offset)
    chunks, got = [], 0
    while got < length:
        b = stream.read(length - got)
        if not b:
            raise EOFError(f"short read: {got} of {length} bytes at offset {offset}")
        chunks.append(b)
        got += len(b)
    return chunks[0] if len(chunks) == 1 else b"".join(chunks)


class ParallelReadMixin:
    """``read_data`` with whole-item ranged reads processed by a thread pool."""

    read_threads: int = 8

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        from torch.distributed._shard._utils import narrow_tensor_by_index

        items = []
        for req in plan.items:
            md = self.storage_data[req.storage_index]
            items.append((md.relative_path, md.offset, md.length, req, md))
        items.sort(key=lambda x: (x[0], x[1]))
        n = max(1, min(self.read_threads, len(items)))
        # Contiguous chunks balanced by bytes keep each thread's reads sequential.
        total = sum(x[2] for x in items) or 1
        chunks = [list() for _ in range(n)]
        acc, i = 0, 0
        for it in items:
            if i < n - 1 and acc >= (i + 1) * total / n:
                i += 1
            chunks[i].append(it)
            acc += it[2]
        lock = threading.Lock()

        def work(chunk) -> None:
            opened: dict[str, tuple[Any, Any]] = {}
            try:
                for rel, offset, length, req, md in chunk:
                    if rel not in opened:
                        cm = self.fs.create_stream(self.fs.concat_path(self.path, rel), "rb")
                        opened[rel] = (cm, cm.__enter__())
                    raw = io.BytesIO(_read_exact(opened[rel][1], offset, length))
                    src = self.transforms.transform_load_stream(req, md.transform_descriptors or (), raw)
                    if req.type == LoadItemType.BYTE_IO:
                        data = io.BytesIO(src.read(-1))
                        with lock:
                            planner.load_bytes(req, data)
                        continue
                    if not src.seekable():
                        src = io.BytesIO(src.read(-1))
                    tensor = cast(torch.Tensor, torch.load(src, map_location="cpu", weights_only=True))
                    tensor = narrow_tensor_by_index(tensor, req.storage_offsets, req.lengths)
                    with lock:
                        target = planner.resolve_tensor(req).detach()
                    if target.size() != tensor.size():
                        raise AssertionError(f"req {req.storage_index} size {target.size()} vs {tensor.size()}")
                    target.copy_(tensor)
                    with lock:
                        planner.commit_tensor(req, target)
            finally:
                for cm, _ in opened.values():
                    cm.__exit__(None, None, None)

        with ThreadPoolExecutor(max_workers=n, thread_name_prefix="dcp-read") as pool:
            for fut in [pool.submit(work, c) for c in chunks if c]:
                fut.result()
        done: Future = Future()
        done.set_result(None)
        return done


class ParallelFileSystemReader(ParallelReadMixin, FileSystemReader):
    def __init__(self, path, read_threads: int = 8, **kwargs):
        super().__init__(path, **kwargs)
        self.read_threads = read_threads


class ParallelFsspecReader(ParallelReadMixin):
    """Parallel variant of ``FsspecReader`` (``gs://`` via gcsfs).

    ``cache_type`` is passed to every ``open`` (recent gcsfs releases default to
    a cache type that older fsspec releases do not have).
    """

    def __new__(cls, url: str, read_threads: int = 8, cache_type: str | None = "readahead", **kwargs):
        from contextlib import contextmanager

        from torch.distributed.checkpoint import FsspecReader

        klass = type("ParallelFsspecReaderImpl", (ParallelReadMixin, FsspecReader), {})
        reader = klass(url, **kwargs)
        reader.read_threads = read_threads
        if cache_type is not None and not str(url).startswith("file://"):
            wrapper = reader.fs

            @contextmanager
            def create_stream(path, mode):
                with wrapper.fs.open(path, mode, cache_type=cache_type) as stream:
                    yield stream

            wrapper.create_stream = create_stream
        return reader


def tune_checkpoint_manager(checkpointer: Any, *, save_threads: int = 8, load_threads: int = 8) -> None:
    """Use multi-threaded DCP writers/readers in a TorchTitan ``CheckpointManager``.

    Only plain DCP checkpoints on local paths or FUSE mounts are affected
    (Hugging Face loads and saves keep their own readers/writers).
    """
    if not getattr(checkpointer, "enable", False):
        return
    import torch.distributed.checkpoint as dcp

    original_load = checkpointer._load_checkpoint

    def load_checkpoint(states, checkpoint_id, *, from_hf, from_quantized):
        if from_hf or "://" in str(checkpoint_id):
            return original_load(states, checkpoint_id, from_hf=from_hf, from_quantized=from_quantized)
        state_dict = checkpointer._flattened_model_states_sd(states)
        dcp.load(state_dict, storage_reader=ParallelFileSystemReader(checkpoint_id, read_threads=load_threads))
        if "model" in states:
            states["model"].load_state_dict(state_dict)

    checkpointer._load_checkpoint = load_checkpoint
    original_save = dcp.save

    def save(state_dict, *, checkpoint_id=None, storage_writer=None, **kwargs):
        if storage_writer is None and checkpoint_id is not None and "://" not in str(checkpoint_id):
            storage_writer = FileSystemWriter(checkpoint_id, thread_count=save_threads)
            checkpoint_id = None
        return original_save(state_dict, checkpoint_id=checkpoint_id, storage_writer=storage_writer, **kwargs)

    # CheckpointManager.dcp_save calls dcp.save / dcp.async_save through the module.
    checkpointer._tuned_dcp_save = save
    _patch_module_save(checkpointer, save)


def _patch_module_save(checkpointer: Any, save) -> None:
    import importlib

    mod = importlib.import_module(type(checkpointer).__module__)
    target = getattr(mod, "dcp", None)
    if target is None:
        return
    if getattr(target, "_tuned", False):
        target.save = save  # latest settings win (one manager per process in practice)
        return

    class _DcpProxy:
        _tuned = True

        def __getattr__(self, name):
            return getattr(target, name)

    proxy = _DcpProxy()
    proxy.save = save
    setattr(mod, "dcp", proxy)
