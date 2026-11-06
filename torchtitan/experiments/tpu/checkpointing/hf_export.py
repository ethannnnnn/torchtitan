# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Layout-preserving Hugging Face export of trained weights, at any scale.

The output directory is a drop-in replacement for the original Hugging Face
checkpoint directory: the same ``model-XXXXX-of-YYYYY.safetensors`` files,
the same ``model.safetensors.index.json``, the same tensor-to-file map,
dtypes, shapes, header bytes and data offsets, plus every non-weight file
(config, tokenizer, chat template, license, ...). Only the bytes of tensors
that training changed differ; tensors the run never touched (for example a
vision tower of a text-only fine-tune, or every non-adapted weight of a LoRA
run) are byte-copied. Serving stacks that read the original checkpoint (vLLM,
transformers, quantizers) therefore read the export the same way.

How it scales: every output shard is produced independently by streaming --
the original header is written verbatim, then each tensor's bytes in data
offset order, either the replacement tensor (cast to the original dtype) or a
raw byte-range copy from the original shard. Peak memory is the replacement
tensors of one shard. ``--worker i --num-workers n`` splits the shards over
``n`` processes or hosts (balanced by bytes); ``--finalize`` then checks that
every shard landed and writes the index, the small files and ``EXPORT_OK``.

Sources of replacement tensors:

* ``--dcp DIR``: a full-parameter TorchTitan DCP checkpoint. Only the keys a
  shard needs are read (``_load_state_dict_from_keys``) and converted with
  the model's ``StateDictAdapter.to_hf``. If the checkpoint also holds LoRA
  adapters (``<module>.lora_a.weight`` / ``lora_b.weight``), they are folded
  into their base weight (``W + scale * B @ A``, in float32).
* ``--peft DIR``: a PEFT LoRA adapter (``adapter_model.safetensors`` +
  ``adapter_config.json``); each targeted weight becomes ``W + scale * B @ A``
  computed in float32 from the original weight and cast back.
* ``StateDictSource``: an in-memory HF-keyed state dict (tests, small models).

Paths may be local (including Cloud Storage FUSE mounts) or ``gs://`` URLs
(read through fsspec/gcsfs; written through a local staging file and a
parallel multipart upload).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import shutil
import struct
import sys
import tempfile
import time
from collections import defaultdict
from typing import Any, Callable, Iterable, Protocol

import torch

__all__ = [
    "HfLayout",
    "ReplacementSource",
    "StateDictSource",
    "PeftSource",
    "DcpSource",
    "export_hf",
    "finalize_export",
    "assign_files",
    "discover_hf_dependencies",
]

INDEX_NAME = "model.safetensors.index.json"
SINGLE_NAME = "model.safetensors"
OK_NAME = "EXPORT_OK"
MANIFEST_NAME = "export_manifest.json"

ST_DTYPES: dict[str, torch.dtype] = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "U16": torch.uint16,
    "I32": torch.int32,
    "U32": torch.uint32,
    "I64": torch.int64,
    "U64": torch.uint64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}

_COPY_CHUNK = 64 << 20


# ---------------------------------------------------------------------------
# Paths (local, FUSE mounts, or fsspec URLs)
# ---------------------------------------------------------------------------


def _is_url(path: str) -> bool:
    return "://" in path and not path.startswith("file://")


def _local(path: str) -> str:
    return path.removeprefix("file://")


def _join(root: str, name: str) -> str:
    return root.rstrip("/") + "/" + name if _is_url(root) else os.path.join(_local(root), name)


def _open(path: str, mode: str, storage_options: dict | None = None):
    if _is_url(path):
        import fsspec

        fs, p = fsspec.core.url_to_fs(path, **(storage_options or {}))
        kwargs = {"block_size": 64 << 20}
        if "r" in mode:
            # Explicit cache type: recent gcsfs defaults to one that older fsspec lacks.
            kwargs["cache_type"] = "readahead"
        return fs.open(p, mode, **kwargs)
    return open(_local(path), mode)


def _exists(path: str, storage_options: dict | None = None) -> bool:
    if _is_url(path):
        import fsspec

        fs, p = fsspec.core.url_to_fs(path, **(storage_options or {}))
        return fs.exists(p)
    return os.path.exists(_local(path))


def _listdir(path: str, storage_options: dict | None = None) -> list[str]:
    if _is_url(path):
        import fsspec

        fs, p = fsspec.core.url_to_fs(path, **(storage_options or {}))
        return sorted(os.path.basename(x.rstrip("/")) for x in fs.ls(p, detail=False))
    return sorted(os.listdir(_local(path)))


def _read_json(path: str, storage_options: dict | None = None) -> Any:
    with _open(path, "rb", storage_options) as f:
        return json.loads(f.read())


def _copy_file(src: str, dst: str, storage_options: dict | None = None) -> None:
    with _open(src, "rb", storage_options) as fi, _open(dst, "wb", storage_options) as fo:
        shutil.copyfileobj(fi, fo, _COPY_CHUNK)


# ---------------------------------------------------------------------------
# Original layout
# ---------------------------------------------------------------------------


class HfLayout:
    """Tensor-to-file map and per-shard headers of a HF safetensors checkpoint."""

    def __init__(self, root: str, storage_options: dict | None = None):
        self.root = root
        self.storage_options = storage_options
        self._headers: dict[str, tuple[bytes, dict]] = {}
        index_path = _join(root, INDEX_NAME)
        if _exists(index_path, storage_options):
            self.index = _read_json(index_path, storage_options)
            self.weight_map: dict[str, str] = dict(self.index["weight_map"])
        elif _exists(_join(root, SINGLE_NAME), storage_options):
            self.index = None
            _, header = self.header(SINGLE_NAME)
            self.weight_map = {k: SINGLE_NAME for k in header if k != "__metadata__"}
        else:
            raise FileNotFoundError(f"no {INDEX_NAME} or {SINGLE_NAME} under {root}")
        self.files = sorted(set(self.weight_map.values()))

    def header(self, file: str) -> tuple[bytes, dict]:
        """(verbatim header bytes incl. the 8-byte length, parsed header)."""
        if file not in self._headers:
            with _open(_join(self.root, file), "rb", self.storage_options) as f:
                prefix = f.read(8)
                (n,) = struct.unpack("<Q", prefix)
                raw = f.read(n)
            self._headers[file] = (prefix + raw, json.loads(raw))
        return self._headers[file]

    def entries(self, file: str) -> list[tuple[str, str, list[int], int, int]]:
        """(name, dtype, shape, begin, end) in data-offset order."""
        _, header = self.header(file)
        out = [
            (name, e["dtype"], list(e["shape"]), e["data_offsets"][0], e["data_offsets"][1])
            for name, e in header.items()
            if name != "__metadata__"
        ]
        return sorted(out, key=lambda x: x[3])

    def file_bytes(self, file: str) -> int:
        hb, header = self.header(file)
        end = max((e["data_offsets"][1] for k, e in header.items() if k != "__metadata__"), default=0)
        return len(hb) + end

    def tensor_info(self, name: str) -> tuple[str, list[int], str]:
        file = self.weight_map[name]
        e = self.header(file)[1][name]
        return e["dtype"], list(e["shape"]), file

    def read_tensor(self, name: str, f=None, path: str | None = None) -> torch.Tensor:
        """One tensor of the checkpoint (``path``: a local copy of its shard)."""
        file = self.weight_map[name]
        hb, header = self.header(file)
        e = header[name]
        begin, end = e["data_offsets"]
        own = f is None
        if own:
            f = _open(path or _join(self.root, file), "rb", None if path else self.storage_options)
        try:
            f.seek(len(hb) + begin)
            buf = bytearray(end - begin)
            view = memoryview(buf)
            got = 0
            while got < len(buf):
                n = f.readinto(view[got:]) if hasattr(f, "readinto") else None
                if n is None:
                    chunk = f.read(len(buf) - got)
                    n = len(chunk)
                    view[got : got + n] = chunk
                if not n:
                    raise EOFError(f"short read for {name} in {file}")
                got += n
        finally:
            if own:
                f.close()
        dtype = ST_DTYPES[e["dtype"]]
        t = torch.frombuffer(buf, dtype=torch.uint8) if len(buf) else torch.empty(0, dtype=torch.uint8)
        return t.view(dtype).reshape(e["shape"])

    # Staging/bookkeeping files that are not part of a checkpoint release (and
    # checksum lists that would be stale for the export).
    EXCLUDE = ("*.ok", "_*", ".*", "SHA256SUMS*", "MANIFEST.json", "export_manifest*", "*.partial")

    def extra_files(self) -> list[str]:
        """Non-weight files to carry over (config, tokenizer, license, ...)."""
        import fnmatch

        skip = set(self.files) | {INDEX_NAME, OK_NAME, MANIFEST_NAME}
        out = []
        for name in _listdir(self.root, self.storage_options):
            if name in skip or any(fnmatch.fnmatch(name, pat) for pat in self.EXCLUDE):
                continue
            if name.endswith(".safetensors"):
                continue  # stray shards not in the index are not part of the checkpoint
            path = _join(self.root, name)
            if not _is_url(path) and os.path.isdir(_local(path)):
                continue
            out.append(name)
        return out


# ---------------------------------------------------------------------------
# Replacement sources
# ---------------------------------------------------------------------------


class ReplacementSource(Protocol):
    def keys(self) -> set[str]:
        """Every HF key this source replaces."""

    def replacements(
        self, names: list[str], read_base: Callable[[str], torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Replacement tensors (HF keys) for ``names`` (a subset of ``keys()``)."""

    def describe(self) -> dict[str, Any]:
        ...


class StateDictSource:
    """In-memory HF-keyed tensors."""

    def __init__(self, state_dict: dict[str, torch.Tensor]):
        self.state_dict = state_dict

    def keys(self) -> set[str]:
        return set(self.state_dict)

    def replacements(self, names, read_base):
        return {n: self.state_dict[n] for n in names}

    def describe(self):
        return {"kind": "state_dict", "tensors": len(self.state_dict)}


def lora_scaling(config: dict[str, Any]) -> float:
    r = int(config["r"])
    alpha = float(config.get("lora_alpha", r))
    if config.get("use_rslora"):
        return alpha / (r ** 0.5)
    return alpha / r


class PeftSource:
    """Merge a PEFT LoRA adapter into the base weights it targets."""

    _KEY = re.compile(r"^(?:base_model\.model\.)?(.+)\.lora_([AB])(?:\.[^.]+)?\.weight$")

    def __init__(self, adapter_dir: str, layout: HfLayout | None = None, storage_options: dict | None = None):
        from safetensors.torch import load as st_load

        self.adapter_dir = adapter_dir
        self.config = _read_json(_join(adapter_dir, "adapter_config.json"), storage_options)
        if self.config.get("peft_type", "LORA").upper() != "LORA":
            raise ValueError(f"unsupported peft_type {self.config.get('peft_type')}")
        with _open(_join(adapter_dir, "adapter_model.safetensors"), "rb", storage_options) as f:
            tensors = st_load(f.read())
        self.scaling = lora_scaling(self.config)
        pairs: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        for key, value in tensors.items():
            m = self._KEY.match(key)
            if m is None:
                raise ValueError(f"unexpected adapter tensor {key}")
            pairs[m.group(1)][m.group(2)] = value
        self.adapters: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for module, ab in pairs.items():
            if set(ab) != {"A", "B"}:
                raise ValueError(f"adapter {module} lacks lora_A or lora_B")
            self.adapters[f"{module}.weight"] = (ab["A"], ab["B"])
        if layout is not None:
            missing = sorted(set(self.adapters) - set(layout.weight_map))
            if missing:
                raise KeyError(f"adapter targets weights absent from the base checkpoint: {missing[:5]}")

    def keys(self) -> set[str]:
        return set(self.adapters)

    def replacements(self, names, read_base):
        out = {}
        for name in names:
            a, b = self.adapters[name]
            w = read_base(name)
            if tuple(b.shape) != (w.shape[0], a.shape[0]) or a.shape[1] != w.shape[1]:
                raise ValueError(f"adapter shapes {tuple(a.shape)}/{tuple(b.shape)} vs weight {tuple(w.shape)} for {name}")
            delta = torch.matmul(b.to(torch.float32), a.to(torch.float32))
            out[name] = (w.to(torch.float32) + self.scaling * delta).to(w.dtype)
        return out

    def describe(self):
        return {"kind": "peft_lora_merge", "adapter_dir": self.adapter_dir, "r": self.config.get("r"),
                "lora_alpha": self.config.get("lora_alpha"), "scaling": self.scaling,
                "targets": len(self.adapters)}


def _is_lora(key: str) -> bool:
    return key.endswith(".lora_a.weight") or key.endswith(".lora_b.weight")


def _layer_of(key: str) -> str:
    m = re.search(r"(?:^|\.)layers\.(\d+)\.", key)
    return m.group(1) if m else "-"


def discover_hf_dependencies(adapter: Any, meta: dict[str, torch.Tensor]) -> dict[str, list[str]]:
    """HF key -> model keys it is computed from, found by probing ``adapter.to_hf``.

    One-to-one renames are found by converting single keys. Keys that do not
    convert alone (fused tensors such as a packed gate/up expert weight) are
    converted together with the other such keys of the same layer, and every
    HF key that comes out depends on that group. Only meta tensors are used.
    """
    to_hf = getattr(adapter, "to_hf_base", adapter.to_hf)
    deps: dict[str, list[str]] = {}
    pending: dict[str, list[str]] = defaultdict(list)
    for key, value in meta.items():
        if _is_lora(key):
            continue
        try:
            out = to_hf({key: value})
        except Exception:  # needs its partners
            pending[_layer_of(key)].append(key)
            continue
        if len(out) == 1:
            (hf_key, hf_val), = out.items()
            if hf_val.numel() == value.numel():
                deps[hf_key] = [key]
                continue
        if len(out) == 0:
            continue  # the adapter drops this key
        pending[_layer_of(key)].append(key)
    for layer, keys in pending.items():
        out = to_hf({k: meta[k] for k in keys})
        for hf_key in out:
            if hf_key in deps:
                raise ValueError(f"{hf_key} produced both alone and by group {keys}")
            deps[hf_key] = list(keys)
    return deps


class DcpSource:
    """Trained tensors from a TorchTitan DCP checkpoint, converted to HF keys."""

    def __init__(
        self,
        dcp_dir: str,
        adapter: Any,
        *,
        storage_options: dict | None = None,
        lora_scale: float | None = None,
    ):
        self.dcp_dir = dcp_dir
        self.adapter = adapter
        self.storage_options = storage_options or {}
        self.lora_scale = lora_scale
        md = self._reader().read_metadata().state_dict_metadata
        tensor_md = {k: v for k, v in md.items() if hasattr(v, "size") and hasattr(v, "properties")}
        model_keys = {k: v for k, v in tensor_md.items() if not k.startswith(("optimizer.", "lr_scheduler", "dataloader", "train_state"))}
        # TorchTitan flattens model keys to the top level; older layouts prefix "model.".
        if model_keys and all(k.startswith("model.") for k in model_keys):
            self.prefix = "model."
        else:
            self.prefix = ""
        self.meta = {
            k[len(self.prefix):]: torch.empty(tuple(v.size), dtype=v.properties.dtype, device="meta")
            for k, v in model_keys.items()
        }
        self.lora = {k for k in self.meta if _is_lora(k)}
        # Fail before any shard is written: every adapter must fold into a base
        # weight of the same module (a fused projection whose module saves its
        # weight as parts, e.g. ``wqkv`` saved as ``wq``/``wk``/``wv``, cannot).
        prefixes = {k.rsplit(".lora_", 1)[0] for k in self.lora}
        unpaired = sorted(p for p in prefixes if not {f"{p}.lora_a.weight", f"{p}.lora_b.weight"} <= self.lora)
        if unpaired:
            raise ValueError(f"LoRA adapters without lora_a or lora_b: {unpaired[:5]}")
        unfoldable = sorted(p for p in prefixes if f"{p}.weight" not in self.meta)
        if unfoldable:
            raise NotImplementedError(
                f"LoRA adapters with no base weight of the same module in the checkpoint: {unfoldable[:5]} "
                "(for an adapter-only checkpoint, write a PEFT adapter and merge it with --peft)")
        if self.lora and lora_scale is None:
            raise ValueError("checkpoint holds LoRA adapters; pass lora_scale (alpha / r)")
        self.deps = discover_hf_dependencies(adapter, self.meta)

    def _reader(self):
        from torch.distributed.checkpoint import FileSystemReader, FsspecReader

        if _is_url(self.dcp_dir):
            return FsspecReader(self.dcp_dir, **self.storage_options)
        return FileSystemReader(_local(self.dcp_dir))

    def keys(self) -> set[str]:
        return set(self.deps)

    def _load(self, keys: set[str]) -> dict[str, torch.Tensor]:
        from torch.distributed.checkpoint.state_dict_loader import _load_state_dict_from_keys

        loaded = _load_state_dict_from_keys({self.prefix + k for k in keys}, storage_reader=self._reader())
        flat: dict[str, torch.Tensor] = {}

        def walk(prefix: str, obj: Any) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    walk(f"{prefix}{k}." if prefix or k else k, v)
            else:
                flat[prefix[:-1]] = obj

        walk("", loaded)
        return {k[len(self.prefix):]: v for k, v in flat.items() if k[len(self.prefix):] in keys}

    def replacements(self, names, read_base):
        groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for name in names:
            groups[tuple(self.deps[name])].append(name)
        need: set[str] = set()
        for g in groups:
            need.update(g)
            for k in g:
                base = k[: -len(".weight")] if k.endswith(".weight") else None
                if base and f"{base}.lora_a.weight" in self.lora:
                    need.update({f"{base}.lora_a.weight", f"{base}.lora_b.weight"})
        loaded = self._load(need)
        to_hf = getattr(self.adapter, "to_hf_base", self.adapter.to_hf)
        out: dict[str, torch.Tensor] = {}
        for g, hf_names in groups.items():
            part = {}
            for k in g:
                t = loaded[k]
                base = k[: -len(".weight")] if k.endswith(".weight") else None
                if base and f"{base}.lora_a.weight" in loaded:
                    if self.lora_scale is None:
                        raise ValueError("checkpoint holds LoRA adapters; pass lora_scale (alpha / r)")
                    a = loaded[f"{base}.lora_a.weight"].to(torch.float32)
                    b = loaded[f"{base}.lora_b.weight"].to(torch.float32)
                    t = (t.to(torch.float32) + self.lora_scale * (b @ a)).to(t.dtype)
                part[k] = t
            hf = to_hf(part)
            for n in hf_names:
                out[n] = hf[n]
        return out

    def describe(self):
        return {"kind": "dcp", "dcp_dir": self.dcp_dir, "model_tensors": len(self.meta),
                "lora_tensors": len(self.lora), "hf_tensors": len(self.deps), "key_prefix": self.prefix}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def assign_files(layout: HfLayout, num_workers: int) -> list[list[str]]:
    """Greedy largest-first assignment of shard files to workers by bytes."""
    loads = [0] * num_workers
    out = [list() for _ in range(num_workers)]
    for file in sorted(layout.files, key=lambda f: (-layout.file_bytes(f), f)):
        i = min(range(num_workers), key=lambda j: (loads[j], j))
        out[i].append(file)
        loads[i] += layout.file_bytes(file)
    return [sorted(x) for x in out]


def _tensor_bytes(t: torch.Tensor) -> memoryview:
    t = t.detach().to("cpu").contiguous()
    if t.numel() == 0:
        return memoryview(b"")
    return memoryview(t.view(torch.uint8).reshape(-1).numpy())


class _Writer:
    """Sequential output stream with an optional running SHA-256."""

    def __init__(self, f, sha: bool):
        self.f = f
        self.h = hashlib.sha256() if sha else None
        self.n = 0

    def write(self, data) -> None:
        self.f.write(data)
        if self.h is not None:
            self.h.update(data)
        self.n += len(data)


def _export_one(layout: HfLayout, file: str, source: ReplacementSource, keys: set[str], dst, sha: bool,
                local_src: str | None = None) -> dict:
    hb, header = layout.header(file)
    entries = layout.entries(file)
    names = [e[0] for e in entries]
    replaced_names = [n for n in names if n in keys]
    src_path = local_src or _join(layout.root, file)
    t0 = time.perf_counter()
    with _open(src_path, "rb", None if local_src else layout.storage_options) as src:
        repl = (source.replacements(replaced_names, lambda n: layout.read_tensor(n, path=local_src))
                if replaced_names else {})
        t_src = time.perf_counter() - t0
        w = _Writer(dst, sha)
        w.write(hb)
        pos = 0
        for name, dtype, shape, begin, end in entries:
            if begin != pos:
                raise ValueError(f"{file}: non-contiguous data at {name}")
            if name in repl:
                t = repl[name]
                want = ST_DTYPES[dtype]
                if list(t.shape) != shape:
                    raise ValueError(f"{name}: trained shape {list(t.shape)} != original {shape}")
                if not t.is_floating_point() and t.dtype != want:
                    raise ValueError(f"{name}: dtype {t.dtype} != original {want}")
                t = t.to(want)
                data = _tensor_bytes(t)
                if len(data) != end - begin:
                    raise ValueError(f"{name}: {len(data)} bytes != slot {end - begin}")
                w.write(data)
            else:
                src.seek(len(hb) + begin)
                left = end - begin
                while left:
                    chunk = src.read(min(left, _COPY_CHUNK))
                    if not chunk:
                        raise EOFError(f"{file}: short read at {name}")
                    w.write(chunk)
                    left -= len(chunk)
            pos = end
    return {
        "file": file, "bytes": w.n, "tensors": len(names), "replaced": len(repl),
        "copied": len(names) - len(repl), "sha256": w.h.hexdigest() if w.h else None,
        "source_s": round(t_src, 3), "total_s": round(time.perf_counter() - t0, 3),
    }


def export_hf(
    base_dir: str,
    out_dir: str,
    source: ReplacementSource,
    *,
    worker: int = 0,
    num_workers: int = 1,
    storage_options: dict | None = None,
    out_storage_options: dict | None = None,
    staging_dir: str | None = None,
    gcs_client: Any = None,
    sha256: bool = True,
    allow_unreplaced: Iterable[str] | None = None,
    finalize: bool | None = None,
) -> dict:
    """Export the shards assigned to ``worker``; returns that worker's report.

    With ``num_workers == 1`` (or ``finalize=True``) the export is also
    finalized (index, small files, manifest, ``EXPORT_OK``).
    """
    layout = HfLayout(base_dir, storage_options)
    keys = source.keys()
    unknown = sorted(keys - set(layout.weight_map))
    if unknown:
        raise KeyError(f"{len(unknown)} trained tensors have no slot in the original layout, e.g. {unknown[:5]}")
    files = assign_files(layout, num_workers)[worker]
    if not _is_url(out_dir):
        os.makedirs(_local(out_dir), exist_ok=True)
    report = {"worker": worker, "num_workers": num_workers, "files": []}
    t0 = time.perf_counter()
    gcs = _gcs() if gcs_client is not None else None
    for file in files:
        dst_path = _join(out_dir, file)
        local_src = None
        t_dl = None
        if staging_dir and gcs is not None and base_dir.startswith("gs://"):
            local_src = os.path.join(staging_dir, f"base-{worker}-{file}")
            t = time.perf_counter()
            gcs.download_file(gcs_client, gcs.parse_gs_uri(_join(base_dir, file)), local_src)
            t_dl = round(time.perf_counter() - t, 3)
        if out_dir.startswith("gs://") and gcs is not None:
            with tempfile.NamedTemporaryFile(dir=staging_dir, suffix=".safetensors", delete=False) as tmp:
                rec = _export_one(layout, file, source, keys, tmp, sha256, local_src)
            t = time.perf_counter()
            gcs.upload_file(gcs_client, tmp.name, gcs.parse_gs_uri(dst_path))
            rec["upload_s"] = round(time.perf_counter() - t, 3)
            os.remove(tmp.name)
        else:
            tmp_path = dst_path + ".partial"
            with _open(tmp_path, "wb", out_storage_options) as f:
                rec = _export_one(layout, file, source, keys, f, sha256, local_src)
            if _is_url(dst_path):
                import fsspec

                fs, p = fsspec.core.url_to_fs(dst_path, **(out_storage_options or {}))
                fs.mv(fs._strip_protocol(tmp_path), p)
            else:
                os.replace(_local(tmp_path), _local(dst_path))
        if local_src is not None:
            os.remove(local_src)
            rec["download_s"] = t_dl
        report["files"].append(rec)
    report["seconds"] = round(time.perf_counter() - t0, 3)
    report["bytes"] = sum(r["bytes"] for r in report["files"])
    report["source"] = source.describe()
    report["replaced_in_worker"] = sum(r["replaced"] for r in report["files"])
    worker_manifest = _join(out_dir, f"export_manifest.worker{worker:05d}-of-{num_workers:05d}.json")
    with _open(worker_manifest, "wb", out_storage_options) as f:
        f.write(json.dumps(report, indent=1).encode())
    if finalize if finalize is not None else num_workers == 1:
        report["finalize"] = finalize_export(
            base_dir, out_dir, num_workers=num_workers, source_keys=keys,
            storage_options=storage_options, out_storage_options=out_storage_options,
            allow_unreplaced=allow_unreplaced)
    return report


def finalize_export(
    base_dir: str,
    out_dir: str,
    *,
    num_workers: int,
    source_keys: set[str] | None = None,
    storage_options: dict | None = None,
    out_storage_options: dict | None = None,
    allow_unreplaced: Iterable[str] | None = None,
) -> dict:
    """Check every worker's shards, then write the index, small files, manifest and ``EXPORT_OK``."""
    layout = HfLayout(base_dir, storage_options)
    reports = []
    for w in range(num_workers):
        path = _join(out_dir, f"export_manifest.worker{w:05d}-of-{num_workers:05d}.json")
        if not _exists(path, out_storage_options):
            raise FileNotFoundError(f"worker {w} has not finished: {path}")
        reports.append(_read_json(path, out_storage_options))
    done = {r["file"]: r for rep in reports for r in rep["files"]}
    missing = sorted(set(layout.files) - set(done))
    if missing:
        raise FileNotFoundError(f"shards not exported: {missing[:5]}")
    for file, rec in done.items():
        if rec["bytes"] != layout.file_bytes(file):
            raise ValueError(f"{file}: {rec['bytes']} bytes != original {layout.file_bytes(file)}")
    if layout.index is not None:
        with _open(_join(out_dir, INDEX_NAME), "wb", out_storage_options) as f:
            f.write(json.dumps(layout.index, indent=2).encode() + b"\n")
    extras = layout.extra_files()
    for name in extras:
        _copy_file(_join(base_dir, name), _join(out_dir, name), out_storage_options or storage_options)
    replaced = sum(r["replaced"] for r in done.values())
    manifest = {
        "kind": "hf_layout_preserving_export_v1",
        "base_dir": base_dir,
        "num_shards": len(layout.files),
        "num_tensors": len(layout.weight_map),
        "replaced_tensors": replaced,
        "copied_tensors": len(layout.weight_map) - replaced,
        "extra_files": extras,
        "source": reports[0].get("source"),
        "bytes": sum(r["bytes"] for r in done.values()),
        "shards": {f: {k: done[f][k] for k in ("bytes", "sha256", "replaced")} for f in sorted(done)},
        "workers": [{"worker": r["worker"], "seconds": r["seconds"], "bytes": r["bytes"]} for r in reports],
    }
    if source_keys is not None and replaced != len(source_keys):
        raise ValueError(f"replaced {replaced} tensors but the source provides {len(source_keys)}")
    with _open(_join(out_dir, MANIFEST_NAME), "wb", out_storage_options) as f:
        f.write(json.dumps(manifest, indent=1).encode())
    with _open(_join(out_dir, OK_NAME), "wb", out_storage_options) as f:
        f.write(b"ok\n")
    return {k: manifest[k] for k in ("num_shards", "num_tensors", "replaced_tensors", "copied_tensors", "bytes")}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_factory(spec: str) -> Callable[..., Any]:
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn)


def _gcs():
    """Cloud Storage helpers: the package module, or the ``gcs.py`` beside this file when run as a script."""
    try:
        from torchtitan.experiments.tpu.checkpointing import gcs
    except Exception:  # standalone use
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import gcs  # type: ignore[no-redef]
    return gcs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="original HF checkpoint dir (local, FUSE mount, or gs://)")
    p.add_argument("--out", required=True, help="output dir (local, FUSE mount, or gs://)")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--dcp", help="full-parameter DCP checkpoint dir (e.g. .../checkpoint/step-N)")
    src.add_argument("--peft", help="PEFT LoRA adapter dir to merge")
    p.add_argument("--adapter-factory", default="",
                   help="module:function(hf_assets_path) -> StateDictAdapter (required with --dcp)")
    p.add_argument("--lora-scale", type=float, default=None, help="alpha / r, if the DCP holds adapters")
    p.add_argument("--worker", type=int, default=int(os.environ.get("JOB_COMPLETION_INDEX", "0")))
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--finalize", action="store_true", help="only finalize (after all workers finished)")
    p.add_argument("--staging-dir", default=None, help="local dir for shards bound for gs://")
    p.add_argument("--no-sha256", action="store_true")
    p.add_argument("--auth", default="auto", choices=("auto", "ksa", "default"))
    args = p.parse_args(argv)
    if not args.finalize and not (args.dcp or args.peft):
        p.error("one of --dcp / --peft is required (or --finalize)")

    storage: dict = {}
    client = None
    if any(x and x.startswith("gs://") for x in (args.base, args.out, args.dcp, args.peft)):
        gcs = _gcs()
        creds, mode, notes = gcs.resolve_credentials(
            args.auth, impersonate=os.environ.get("GCS_IMPERSONATE_SA") or None,
            probe_uri=args.out if args.out.startswith("gs://") else None)
        storage = {"token": creds}
        client = gcs.storage_client(creds)
    if args.finalize:
        print(json.dumps(finalize_export(args.base, args.out, num_workers=args.num_workers,
                                         storage_options=storage, out_storage_options=storage), indent=1))
        return 0
    layout = HfLayout(args.base, storage)
    if args.peft:
        source: ReplacementSource = PeftSource(args.peft, layout, storage)
    else:
        if not args.adapter_factory:
            p.error("--dcp needs --adapter-factory")
        adapter = _load_factory(args.adapter_factory)(args.base)
        source = DcpSource(args.dcp, adapter, storage_options=storage, lora_scale=args.lora_scale)
    report = export_hf(
        args.base, args.out, source, worker=args.worker, num_workers=args.num_workers,
        storage_options=storage, out_storage_options=storage, staging_dir=args.staging_dir,
        gcs_client=client, sha256=not args.no_sha256)
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
