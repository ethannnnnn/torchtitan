# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""TPU7x settings for Muse Spark trainer configs. ASCII only.

Collected from the 64-chip bring-up (measured on TPU7x):

* The shared Spark TPU7x settings of ``muse_spark/tpu7x.py``: Tokamax Splash
  attention with the Spark blocks, Tokamax GMM-v2 for the routed experts
  (or K/N-tiled MegaBlox grouped matmuls with the paired W1/W3 kernel off:
  Spark experts are 4096 x 4096 and untiled tiles exceed the 64 MiB scoped
  VMEM), ``xla_tpu_scoped_vmem_limit_kib=65472`` and
  ``xla_tpu_dvfs_p_state=7``. The TPU token dispatcher computes the routing
  score gradient in one fused FP32 reduction.
* ``DEFER_AND_FUSE`` eager mode with a compiled gradient norm, full
  activation checkpointing, SimpleFSDP resharding after forward and the TPU
  topology-aware mesh, 30 minute collective timeouts.
* ``FastHFStorageReader``: one positional read per contiguous slice of the
  safetensors payload (dim-0 / expert-dim FSDP shards) instead of mmap page
  faults through Cloud Storage FUSE (77.9 GB/s vs 0.92 GB/s on 128 ranks).
"""

from __future__ import annotations

import json
import os
import struct
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.distributed.checkpoint import HuggingFaceStorageReader

_ST_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}


class FastHFStorageReader(HuggingFaceStorageReader):
    """``HuggingFaceStorageReader`` with large positional reads for contiguous slices."""

    def read_data(self, plan, planner):
        from safetensors import safe_open
        from torch.futures import Future

        per_file: dict[str, list] = {}
        for req in plan.items:
            per_file.setdefault(self.storage_data[req.storage_index].relative_path, []).append(req)

        def do_file(item):
            file_name, reqs = item
            with open(file_name, "rb", buffering=0) as fh:
                hlen = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(hlen))
                for req in reqs:
                    meta = header[req.storage_index.fqn]
                    shape, dtype = meta["shape"], _ST_DTYPES.get(meta["dtype"])
                    offs, lens = list(req.storage_offsets), list(req.lengths)
                    contiguous = dtype is not None and all(
                        o == 0 and n == d for o, n, d in zip(offs[1:], lens[1:], shape[1:]))
                    if contiguous:
                        row = dtype.itemsize
                        for d in shape[1:]:
                            row *= d
                        buf = bytearray(lens[0] * row)
                        view = memoryview(buf)
                        fh.seek(8 + hlen + meta["data_offsets"][0] + offs[0] * row)
                        got = 0
                        while got < len(buf):
                            n = fh.readinto(view[got:])
                            if not n:
                                raise IOError(f"short read: {req.storage_index.fqn} in {file_name}")
                            got += n
                        tensor = (torch.frombuffer(buf, dtype=dtype).view(lens) if buf
                                  else torch.empty(lens, dtype=dtype))
                    else:
                        with safe_open(filename=file_name, framework="pt") as f:
                            tensor = f.get_slice(req.storage_index.fqn)[
                                tuple(slice(o, o + n) for o, n in zip(offs, lens))]
                    target = planner.resolve_tensor(req).detach()
                    if target.size() != tensor.size():
                        raise AssertionError(f"{req.storage_index} size {target.size()} vs {tensor.size()}")
                    target.copy_(tensor)
                    planner.commit_tensor(req, target)
            return len(reqs)

        with ThreadPoolExecutor(max_workers=max(1, self.thread_count)) as pool:
            list(pool.map(do_file, per_file.items()))
        fut: Future = Future()
        fut.set_result(None)
        return fut


def fast_reader(path: str, from_quantized: bool = False) -> HuggingFaceStorageReader:
    return FastHFStorageReader(path, thread_count=int(os.environ.get("SPARK_HF_READER_THREADS", "8")))


def fuse_route_score_backward(cfg) -> None:
    """Routing-score gradient of the TPU token dispatcher in one fused FP32 reduction.

    The dispatcher setting of the TPU MoE recipes; the expert matmul settings
    come from ``muse_spark/tpu7x.py``. A no-op for models built without the
    TPU token dispatcher.
    """
    from torchtitan.experiments.tpu.token_dispatcher import TpuAllToAllTokenDispatcher

    for _, dispatcher, _, _ in cfg.model_spec.model.traverse(TpuAllToAllTokenDispatcher.Config):
        dispatcher.fuse_route_score_backward = True


def apply_tpu_knobs(cfg) -> None:
    """Mutates a TPUTrainerConfig built for a Muse Spark spec with TPU kernels."""
    from torchtitan.distributed.activation_checkpoint import FullAC
    from torchtitan.experiments.tpu.muse_spark import tpu7x
    from torchtitan.experiments.tpu.tpu_job_config import TPUConfig

    fuse_route_score_backward(cfg)
    # Tokamax Splash attention with the Spark blocks, Tokamax GMM-v2 experts,
    # 64 MiB scoped VMEM and power state 7 (muse_spark/tpu7x.py).
    tpu7x.apply(cfg)
    cfg.tpu_config = TPUConfig(enable_amp=False, eager_mode="DEFER_AND_FUSE")
    # The compiled gradient norm needs expert parallelism 1 (tpu_job_config).
    cfg.compile.compile_grad_norm = True
    if os.environ.get("SPARK_AC", "1") == "1":
        cfg.activation_checkpoint = FullAC.Config(preserve_rng_state=False)
    cfg.parallelism.fsdp_reshard_after_forward = "always"
    cfg.parallelism.enable_tpu_topo_mesh = True
    cfg.comm.init_timeout_seconds = 1800
    cfg.comm.train_timeout_seconds = 1800
