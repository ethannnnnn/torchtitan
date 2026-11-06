#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Build a truncated Muse Spark text checkpoint (first N layers) from a sharded one. ASCII only.

Reads only the needed tensors (``model.language_model.layers.<i>.*`` for
i < N, the embedding, the final norm and ``lm_head.weight``) from a
Hugging Face sharded checkpoint, with one large positional read per tensor
(fast on a Cloud Storage FUSE mount, unlike mmap page faults), and writes
``model.safetensors`` + index + a config whose text decoder has N layers
(``num_hidden_layers``, ``layer_types``, ``mlp_layer_types``,
``layer_rope_theta`` truncated) to ``--out``. Tokenizer, chat template and
generation config are copied. Vision / audio tensors are not copied (serve
with the text-only architecture). Standard library + torch-free.

    python build_trunc.py --src /mnt/model --out /dev/shm/model-trunc2 --layers 2 --threads 16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TEXT = "model.language_model."
LAYER = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
COPY = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
        "generation_config.json", "processor_config.json", "LICENSE.md", "USE_POLICY.md")
PER_LAYER_LISTS = ("layer_types", "mlp_layer_types", "layer_rope_theta")


def wanted(key: str, layers: int) -> bool:
    if key == "lm_head.weight":
        return True
    if not key.startswith(TEXT):
        return False
    match = LAYER.match(key)
    return match is None or int(match.group(1)) < layers


def read_header(path: Path) -> tuple[int, dict]:
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        return 8 + length, json.loads(handle.read(length))


def truncated_config(config: dict, layers: int) -> dict:
    """A copy of ``config`` whose text decoder keeps only its first ``layers`` layers."""
    import copy

    result = copy.deepcopy(config)
    text = result["text_config"] if "text_config" in result else result
    available = text["num_hidden_layers"]
    if layers > available:
        raise ValueError(f"asked for {layers} decoder layers, but the source config has only {available}")
    text["num_hidden_layers"] = layers
    text.update({name: text[name][:layers] for name in PER_LAYER_LISTS if isinstance(text.get(name), list)})
    return result


def build(src: Path, out: Path, layers: int, threads: int) -> dict:
    started = time.time()
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    keys = sorted(k for k in index if wanted(k, layers))
    by_file: dict[str, list[str]] = {}
    for key in keys:
        by_file.setdefault(index[key], []).append(key)
    os.makedirs(out, exist_ok=True)
    headers = {}

    def header_of(name: str):
        base, header = read_header(src / name)
        return name, base, header

    with ThreadPoolExecutor(max_workers=threads) as pool:
        for name, base, header in pool.map(header_of, sorted(by_file)):
            headers[name] = (base, header)
    # Output layout: tensors in key order, 8-byte aligned is not required by safetensors.
    plan = []
    offset = 0
    for key in keys:
        base, header = headers[index[key]]
        meta = header[key]
        begin, end = meta["data_offsets"]
        plan.append((key, index[key], base + begin, end - begin, meta["dtype"], meta["shape"], offset))
        offset += end - begin
    out_header = {"__metadata__": {"format": "pt", "source": "build_trunc", "layers": str(layers)}}
    for key, _, _, size, dtype, shape, start in plan:
        out_header[key] = {"dtype": dtype, "shape": shape, "data_offsets": [start, start + size]}
    header_bytes = json.dumps(out_header, separators=(",", ":")).encode()
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)
    target = out / "model.safetensors"
    data_start = 8 + len(header_bytes)
    with open(target, "wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        handle.truncate(data_start + offset)

    def copy_one(item) -> int:
        key, name, src_off, size, _, _, dst_off = item
        chunk = 256 * 1024 * 1024
        done = 0
        with open(src / name, "rb", buffering=0) as fin, open(target, "r+b", buffering=0) as fout:
            while done < size:
                n = min(chunk, size - done)
                fin.seek(src_off + done)
                buf = fin.read(n)
                if len(buf) != n:
                    raise IOError(f"short read {key} in {name}")
                fout.seek(data_start + dst_off + done)
                fout.write(buf)
                done += n
        return size

    with ThreadPoolExecutor(max_workers=threads) as pool:
        copied = sum(pool.map(copy_one, plan))
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": copied}, "weight_map": {k: "model.safetensors" for k in keys}}, indent=1))
    config = json.loads((src / "config.json").read_text())
    (out / "config.json").write_text(json.dumps(truncated_config(config, layers), indent=2))
    for name in COPY:
        if (src / name).is_file():
            shutil.copy(src / name, out / name)
    seconds = time.time() - started
    return {"tensors": len(keys), "bytes": copied, "files_read": len(by_file), "seconds": round(seconds, 1),
            "gb_per_s": round(copied / max(seconds, 1e-9) / 1e9, 2), "out": str(out)}


_CLI_FLAGS = (
    ("--src", {"required": True, "help": "sharded Hugging Face checkpoint directory to read from"}),
    ("--out", {"required": True, "help": "directory that receives the truncated checkpoint"}),
    ("--layers", {"type": int, "required": True, "help": "how many leading decoder layers to keep"}),
    ("--threads", {"type": int, "default": 16, "help": "files read in parallel (default: %(default)s)"}),
)


def main() -> int:
    trunc_cli = argparse.ArgumentParser(
        description="Copy the first N decoder layers of a sharded Muse Spark checkpoint into a small one.")
    for flag, settings in _CLI_FLAGS:
        trunc_cli.add_argument(flag, **settings)
    args = trunc_cli.parse_args()
    report = build(Path(args.src), Path(args.out), args.layers, args.threads)
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
