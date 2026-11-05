# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Muse Spark Open 816B-A42B step recipes for one TPU7x 4x4x4 slice (64 chips, 128 ranks).

  spark_816b_lora_64chip   LoRA r16 / alpha 32 on q, k, v, o; frozen bf16 base; AdamW on the adapters.
  spark_816b_full_64chip   full-parameter AdamW with bf16 parameters, gradients and states (no fp32 master).
  *_random                 same, random init instead of the HF checkpoint (memory / speed without weights).
  *_16k, *_32k             two or four rows per rank.
  spark_816b_lora_64chip_ep8_random   LoRA, random weights, expert parallelism 8 (all-gather dispatcher);
                           spark_816b_lora_64chip_tokamax_random is its EP = 1 baseline.
  spark_816b_lora_64chip_stock_splash_random   EP = 1 with the stock Splash patch (attention A/B);
                           spark_816b_lora_64chip_megablox_random: tiled MegaBlox experts (GMM A/B).
  spark_816b_lora_64chip_measure_placement   released weights, EP = 1: measures routing and writes a
                           load-balanced EP-8 placement (SPARK_EXPERT_PLACEMENT_OUT).
  spark_816b_{lora,full}_64chip_ep8   released weights, EP 8 with that placement
                           (SPARK_EXPERT_PLACEMENT_FILE), BF16 all-gather combine, capacity 2 (LoRA) / 1.5.

Shared settings: packed 8,192-token rows, one row per rank per step (1,048,576 tokens per step);
SimpleFSDP ZeRO-3 over all ranks (reshard after forward); experts FSDP-sharded (expert parallel
degree 1) unless the recipe enables expert parallelism; full activation checkpointing; the TPU7x
settings of tpu7x.py (Tokamax Splash attention with the Spark blocks, tiled Pallas grouped matmuls,
64 MiB scoped VMEM, power state 7) and the Pallas fused linear cross-entropy; torch_tpu
DEFER_AND_FUSE eager mode; real HF weights loaded with TorchTitan's DCP HuggingFaceStorageReader path (each rank reads
its own slices). Paths come from the launcher: --hf_assets_path, --checkpoint.initial_load_path,
--dataloader.dataset_path (a c4_test-format data.json; synthetic text is fine).
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
import re
import string

import torch

from torchtitan.components.checkpointer.dcp import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lora import LoRAConverter
from torchtitan.components.lora import _make_frozen_config
from torchtitan.config import derive
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer.optimizer import default_adamw
from torchtitan.components.validate import Validator
from torchtitan.config.configs import CommConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.experiments.tpu.datasets import c4_test_dataloader
from torchtitan.experiments.tpu.muse_spark import model_registry
from torchtitan.experiments.tpu.muse_spark import expert_placement
from torchtitan.experiments.tpu.muse_spark import tpu7x
from torchtitan.experiments.tpu.muse_spark import state_dict_adapter as _sda
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
from torchtitan.experiments.tpu.overrides.fused_loss import FusedLossLinearWrapper
from torchtitan.experiments.tpu.token_dispatcher import TpuAllToAllTokenDispatcher
from torchtitan.experiments.tpu.tpu_job_config import (
    SplashAttentionKernelConfig,
    TPUConfig,
    TPUTrainerConfig,
    TpuTrainerCompileConfig,
)
from torch.distributed.checkpoint import HuggingFaceStorageReader

def _install_hbm_logger() -> None:
  """Print device-level HBM (free/total and allocator peaks) with every TorchTitan metrics log.

  TorchTitan's own ``memory:`` field on torch_tpu 0.2.0 under-reports (it showed 4.0 GiB for a
  model whose bf16 shard alone is 11.9 GiB per device), so log torch.accelerator's view too.
  """
  try:
    from torchtitan.components.metrics import MetricsProcessor
  except Exception:  # pragma: no cover
    return
  if getattr(MetricsProcessor.log, "_sp_hbm", False):
    return
  orig = MetricsProcessor.log

  def log(self, step, *args, **kwargs):
    out = orig(self, step, *args, **kwargs)
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
      info = {}
      for name in ("memory_allocated", "max_memory_allocated", "memory_reserved", "max_memory_reserved"):
        try:
          info[name + "_GiB"] = round(getattr(torch.accelerator, name)() / 2**30, 2)
        except Exception as e:  # pragma: no cover
          info[name] = repr(e)[:80]
      try:
        free, total = torch.accelerator.get_memory_info()
        info["used_GiB"] = round((total - free) / 2**30, 2)
        info["total_GiB"] = round(total / 2**30, 2)
      except Exception as e:  # pragma: no cover
        info["get_memory_info"] = repr(e)[:80]
      print(f"[SPMEM] rank={os.environ.get('RANK')} step={step} {json.dumps(info)}", flush=True)
    return out

  log._sp_hbm = True
  MetricsProcessor.log = log


_install_hbm_logger()


def _install_ckpt_cache_drop() -> None:
  """Free TorchTitan's cached model state dict after every checkpoint load and synchronous save.

  ``ModelWrapper`` caches ``model.state_dict()``; with the TPU kernel-native expert layout the
  state-dict hook returns fresh contiguous canonical copies of every expert weight, so the cache
  pins ~11.6 GiB per device on the full model for the whole run (VERIFIED-ON-TPU: resident 66.4 vs
  54.8 GiB, full-param OOM at step 2). The cache only serves asynchronous saves (stable staging
  buffers); with synchronous saves it is dropped after each load and each save.
  """
  if os.environ.get("SPARK_DROP_CKPT_CACHE", "1") != "1":
    return
  try:
    from torchtitan.components.checkpointer import base as _ckb
  except Exception:  # pragma: no cover
    return
  wrapper = getattr(_ckb, "ModelWrapper", None)
  if wrapper is None or getattr(wrapper.load_state_dict, "_sp_drop", False):
    return
  orig = wrapper.load_state_dict

  def load_state_dict(self, state_dict):
    orig(self, state_dict)
    self.cached_state_dict = {}
    import gc

    gc.collect()

  load_state_dict._sp_drop = True
  wrapper.load_state_dict = load_state_dict

  orig_save = CheckpointManager._save

  def _save(self, curr_step, last_step=False):
    saved = orig_save(self, curr_step, last_step)
    model = self.states.get("model") if isinstance(getattr(self, "states", None), dict) else None
    if saved and str(getattr(self.async_mode, "value", self.async_mode)) == "disabled" and model is not None:
      model.cached_state_dict = {}
      import gc

      gc.collect()
      # Every rank leaves the save together (checkpointing/flush.py): otherwise ranks 1..N-1
      # wait in DCP's last collective until rank 0's next readback, and hang if rank 0 dies first.
      device = next((p.device for m in getattr(model, "model", []) for p in m.parameters()), None)
      if device is not None and device.type == "tpu":
        from torchtitan.experiments.tpu.checkpointing.flush import device_flush

        seconds = device_flush(device)
        if int(os.environ.get("RANK", "0")) == 0:
          print(f"[SPCKPT] step={curr_step} every rank left the save (flush {seconds:.3f} s)", flush=True)
    return saved

  CheckpointManager._save = _save


_install_ckpt_cache_drop()


def _install_ckpt_io_threads() -> None:
  """SPARK_CKPT_IO_THREADS=N: N DCP writer threads and the parallel reader per rank.

  Stock DCP writes with one thread per rank and reads item by item; at the 64-chip full-parameter
  size (~0.3 TB per host) that is ~7 minutes each way through Cloud Storage FUSE, about 6x and 15x
  slower than with 8 threads (checkpointing/README.md). Applies to every CheckpointManager built in
  this process (``checkpointing.dcp_parallel.tune_checkpoint_manager``).
  """
  threads = int(os.environ.get("SPARK_CKPT_IO_THREADS", "0") or 0)
  if threads <= 0 or getattr(CheckpointManager.__init__, "_sp_io", False):
    return
  from torchtitan.experiments.tpu.checkpointing.dcp_parallel import tune_checkpoint_manager

  orig_init = CheckpointManager.__init__

  def __init__(self, *args, **kwargs):
    orig_init(self, *args, **kwargs)
    tune_checkpoint_manager(self, save_threads=threads, load_threads=threads)

  __init__._sp_io = True
  CheckpointManager.__init__ = __init__


_install_ckpt_io_threads()


def _install_resume_check() -> None:
  """TT_FAULT_KILL_STEP / TT_BATCH_RECORDS_DIR test hooks (checkpointing/resume_check.py)."""
  from torchtitan.experiments.tpu.checkpointing import resume_check

  resume_check.install()


_install_resume_check()

SEQ_LEN = 8192
LORA_TARGETS = ["wq", "wkv", "wo"]  # QKVLinear builds wk and wv from the `wkv` config
SYNTH_PATH = os.environ.get("SPARK_SYNTH_DATA", "/dev/shm/spark-synthetic/data.json")


def ensure_synthetic_text(path: str = SYNTH_PATH, docs: int = 6000, seed: int = 20261030) -> str:
  """Writes a deterministic synthetic text corpus (c4_test JSON-lines format) if absent.

  Random pseudo-words, ~6 M words: enough for several 1 M-token steps on 128 ranks. Every
  process may race to create it; each writes a private temp file and renames atomically, and
  all writers produce identical bytes.
  """
  if os.path.exists(path):
    return path
  os.makedirs(os.path.dirname(path), exist_ok=True)
  rng = random.Random(seed)
  vocab = ["".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(2, 9))) for _ in range(60000)]
  tmp = f"{path}.tmp{os.getpid()}"
  with open(tmp, "w") as f:
    for _ in range(docs):
      n = rng.randint(600, 1400)
      words = [rng.choice(vocab) for _ in range(n)]
      sents, i = [], 0
      while i < n:
        k = rng.randint(8, 20)
        sent = " ".join(words[i:i + k])
        sents.append(sent[:1].upper() + sent[1:] + ".")
        i += k
      f.write(json.dumps({"text": " ".join(sents)}) + "\n")
  os.replace(tmp, path)
  return path


_ST_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}
READER_THREADS = int(os.environ.get("SPARK_HF_READER_THREADS", "8"))
# HF expert tensors ([256, ...] fused per layer) and their layer id, for non-contiguous placements.
_EXPERT_HF_KEY = re.compile(r".*\.mlp\.experts\.(gate_up_proj|down_proj)$")
_EXPERT_LAYER = re.compile(r".*\.layers\.(\d+)\.mlp\.experts\.")


class FastHFStorageReader(HuggingFaceStorageReader):
  """``HuggingFaceStorageReader`` with large positional reads for contiguous slices.

  The stock reader slices through ``safetensors.safe_open`` (mmap), so every page fault
  becomes a small read on a GCSFuse mount: ~7 MB/s per rank, ~0.9 GB/s on 128 ranks
  (measured on a 4x4x4 slice). FSDP dim-0 and expert-dim shards are contiguous
  byte ranges of the safetensors payload, so each is read with one ``readinto``; anything
  else falls back to the stock path. Files are processed by ``thread_count`` threads.

  With a non-contiguous expert placement (expert parallelism with ``ep_expert_placement``
  ``"interleaved"`` or ``"file:..."``) physical expert row ``p`` of every expert tensor is
  read from logical row ``slot_to_logical[p]`` of the checkpoint, one ``readinto`` per row.
  """

  def __init__(self, path, *, thread_count=8, slot_to_logical=None, **kwargs):
    super().__init__(path, thread_count=thread_count, **kwargs)
    # One map for every layer, or {layer_id: map} (load-balanced placements).
    self.slot_to_logical = slot_to_logical
    if isinstance(slot_to_logical, (list, tuple)) and list(slot_to_logical) == sorted(slot_to_logical):
      self.slot_to_logical = None

  def _slot_map(self, fqn):
    if isinstance(self.slot_to_logical, dict):
      return self.slot_to_logical[str(int(_EXPERT_LAYER.match(fqn).group(1)))]
    return self.slot_to_logical

  def read_metadata(self):
    """Stock metadata from plain header reads (no safetensors mmap per file); timed."""
    import json as _json
    import struct
    import time as _time

    try:
      from torch.distributed.checkpoint._hf_utils import _HFStorageInfo
    except ImportError:  # older layouts
      from torch.distributed.checkpoint.hf_storage import _HFStorageInfo
    from torch.distributed.checkpoint.metadata import (
        ChunkStorageMetadata,
        Metadata,
        MetadataIndex,
        StorageMeta,
        TensorProperties,
        TensorStorageMetadata,
    )

    t0 = _time.time()
    files = sorted(f for f in self.fs.ls(self.path) if f.endswith(".safetensors"))
    sd_md, storage = {}, {}
    for fn in files:
      with open(fn, "rb", buffering=0) as fh:
        header = _json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
      header.pop("__metadata__", None)
      for key, meta in header.items():
        dtype = _ST_DTYPES.get(meta["dtype"])
        if dtype is None:
          continue
        shape = torch.Size(meta["shape"])
        sd_md[key] = TensorStorageMetadata(
            properties=TensorProperties(dtype=dtype),
            size=shape,
            chunks=[ChunkStorageMetadata(offsets=torch.Size([0] * len(shape)), sizes=shape)],
        )
        storage[MetadataIndex(fqn=key, offset=[0] * len(shape))] = _HFStorageInfo(
            relative_path=fn, shape=shape, dtype=dtype
        )
    md = Metadata(state_dict_metadata=sd_md, storage_data=storage)
    if getattr(md, "storage_meta", None) is None:
      md.storage_meta = StorageMeta()
    md.storage_meta.load_id = self.load_id
    if int(os.environ.get("RANK", "0")) == 0:
      print(f"[SPLOAD] read_metadata: {len(files)} files, {len(sd_md)} tensors in {_time.time() - t0:.1f}s", flush=True)
    return md

  def read_data(self, plan, planner):
    import json as _json
    import struct
    from concurrent.futures import ThreadPoolExecutor

    from torch.futures import Future

    from safetensors import safe_open

    per_file: dict[str, list] = {}
    for req in plan.items:
      per_file.setdefault(self.storage_data[req.storage_index].relative_path, []).append(req)

    def do_file(item):
      file_name, reqs = item
      with open(file_name, "rb", buffering=0) as fh:
        hlen = struct.unpack("<Q", fh.read(8))[0]
        header = _json.loads(fh.read(hlen))
        for req in reqs:
          meta = header[req.storage_index.fqn]
          shape, dtype = meta["shape"], _ST_DTYPES.get(meta["dtype"])
          offs, lens = list(req.storage_offsets), list(req.lengths)
          contiguous = dtype is not None and all(
              o == 0 and n == d for o, n, d in zip(offs[1:], lens[1:], shape[1:])
          )
          if contiguous:
            row = dtype.itemsize
            for d in shape[1:]:
              row *= d
            buf = bytearray(lens[0] * row)
            view = memoryview(buf)
            permuted = self.slot_to_logical is not None and _EXPERT_HF_KEY.match(req.storage_index.fqn)
            # (file row, buffer row, rows) runs: one run, or one per expert row.
            s2l = self._slot_map(req.storage_index.fqn) if permuted else None
            runs = (
                [(s2l[p], p - offs[0], 1) for p in range(offs[0], offs[0] + lens[0])]
                if permuted else [(offs[0], 0, lens[0])]
            )
            for file_row, buf_row, nrows in runs:
              fh.seek(8 + hlen + meta["data_offsets"][0] + file_row * row)
              got, end = buf_row * row, (buf_row + nrows) * row
              while got < end:
                n = fh.readinto(view[got:end])
                if not n:
                  raise IOError(f"short read: {req.storage_index.fqn} in {file_name}")
                got += n
            tensor = torch.frombuffer(buf, dtype=dtype).view(lens) if buf else torch.empty(lens, dtype=dtype)
          elif dtype is not None and len(shape) == 2 and offs[0] == 0 and lens[0] == shape[0]:
            # Column slice of a 2-D tensor (vocab-uneven embedding / lm_head are sharded on
            # dim 1). Every rank needs bytes from every row, and all ranks hit the same two
            # objects, so one process per host copies the whole tensor into a host-shared
            # /dev/shm file (flock-guarded) and every local rank memory-maps its columns.
            import fcntl

            import numpy as np

            rows, cols = shape
            c0, nc = offs[1], lens[1]
            nbytes = rows * cols * dtype.itemsize
            cache_dir = os.environ.get("SPARK_HF_SHM_CACHE", "/dev/shm/spark-hf-cache")
            os.makedirs(cache_dir, exist_ok=True)
            cpath = os.path.join(cache_dir, f"{os.path.basename(file_name)}.{req.storage_index.fqn}.bin")
            if not (os.path.exists(cpath) and os.path.getsize(cpath) == nbytes):
              with open(cpath + ".lock", "w") as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                try:
                  if not (os.path.exists(cpath) and os.path.getsize(cpath) == nbytes):
                    tmp = f"{cpath}.tmp{os.getpid()}"
                    fh.seek(8 + hlen + meta["data_offsets"][0])
                    left = nbytes
                    with open(tmp, "wb") as out:
                      while left:
                        chunk = fh.read(min(left, 256 << 20))
                        if not chunk:
                          raise IOError(f"short read: {req.storage_index.fqn} in {file_name}")
                        out.write(chunk)
                        left -= len(chunk)
                    os.replace(tmp, cpath)
                finally:
                  fcntl.flock(lk, fcntl.LOCK_UN)
            np_dtype = {2: np.uint16, 4: np.uint32}[dtype.itemsize]
            mm = np.memmap(cpath, dtype=np_dtype, mode="r", shape=(rows, cols))
            cols_np = np.ascontiguousarray(mm[:, c0:c0 + nc])
            del mm
            tensor = torch.from_numpy(cols_np).view(dtype)
          else:
            if self.slot_to_logical is not None and _EXPERT_HF_KEY.match(req.storage_index.fqn):
              raise NotImplementedError("permuted expert rows need contiguous dim-0 slices")
            with safe_open(filename=file_name, framework="pt") as f:
              tensor = f.get_slice(req.storage_index.fqn)[
                  tuple(slice(o, o + n) for o, n in zip(offs, lens))
              ]
          target = planner.resolve_tensor(req).detach()
          if target.size() != tensor.size():
            raise AssertionError(f"{req.storage_index} size {target.size()} vs {tensor.size()}")
          target.copy_(tensor)
          planner.commit_tensor(req, target)
      return len(reqs)

    import time as _time

    t0 = _time.time()
    with ThreadPoolExecutor(max_workers=max(1, self.thread_count)) as pool:
      list(pool.map(do_file, per_file.items()))
    if int(os.environ.get("RANK", "0")) == 0:
      nbytes = sum(t.numel() * t.element_size() for t in [planner.resolve_tensor(r) for r in plan.items])
      print(f"[SPLOAD] read_data rank0: {len(plan.items)} slices from {len(per_file)} files, "
            f"{nbytes / 2**30:.2f} GiB in {_time.time() - t0:.1f}s ({self.thread_count} threads)", flush=True)
    fut: Future = Future()
    fut.set_result(None)
    return fut


def _fast_reader(path: str, from_quantized: bool = False, slot_to_logical=None) -> HuggingFaceStorageReader:
  return FastHFStorageReader(path, thread_count=READER_THREADS, slot_to_logical=slot_to_logical)


def _slot_to_logical(model_config):
  """Expert slot -> logical checkpoint row map of the model (None for the contiguous placement)."""
  moe = next((l.moe for l in model_config.layers if l.moe is not None), None)
  placement = getattr(model_config, "ep_expert_placement", "contiguous")
  if moe is None or placement == "contiguous":
    return None
  degree = getattr(model_config, "ep_placement_degree", 1)
  if placement.startswith("file:"):
    # pylint: disable-next=g-import-not-at-top
    from torchtitan.experiments.tpu.muse_spark.expert_parallel import load_expert_placement

    return load_expert_placement(placement, degree)
  return _sda.expert_slot_to_logical(moe.num_experts, degree, placement)


class FastSparkAdapter(_sda.MuseSparkStateDictAdapter):
  """The Muse Spark HF adapter with the fast reader (expert placement aware).

  Serves LoRA and full-parameter recipes alike: the base adapter leaves LoRA
  adapter weights out of its checks and of ``to_hf``. The released checkpoint
  has no LoRA tensors; they keep their initialization (A random, B zero) and
  the checkpointer loads the model non-strictly.
  """

  def get_hf_storage_reader(self, path, from_quantized=False):
    return _fast_reader(path, from_quantized, _slot_to_logical(self.model_config))


def _enable_spark_fused_ce(cfg: TPUTrainerConfig, *, frozen: bool) -> None:
  """Pallas fused linear cross-entropy for Spark's soft-capped head.

  The stock override (``fused_loss_soft_capped_lm_head``) matches only the exact
  ``SoftCappedLinear.Config`` type; Spark's head is the ``MuseSparkLMHead`` subclass
  (and, under LoRA, a frozen subclass of that), so it is swapped here directly. The
  head's output multiplier and soft cap move into the loss kernel; under LoRA the head
  stays frozen.
  """
  lm = cfg.model_spec.model.lm_head
  new = derive(
      lm,
      FusedLossLinearWrapper.Config,
      soft_cap=lm.output_soft_cap_temp,
      output_multiplier=lm.output_multiplier,
  )
  cfg.model_spec.model.lm_head = _make_frozen_config(new) if frozen else new
  cfg.loss = PallasCrossEntropyLoss.Config(
      soft_cap=lm.output_soft_cap_temp, output_multiplier=lm.output_multiplier
  )


def _fuse_route_score_backward(cfg: TPUTrainerConfig) -> None:
  """Routing-score gradient of the TPU token dispatcher in one fused FP32 reduction.

  The dispatcher setting of the TPU MoE recipes; the expert matmul settings come from tpu7x.py.
  A no-op for models built without the TPU token dispatcher.
  """
  for _, dispatcher, _, _ in cfg.model_spec.model.traverse(TpuAllToAllTokenDispatcher.Config):
    dispatcher.fuse_route_score_backward = True


# Expert matmul tiles, attention kernel and libtpu flags come from tpu7x.py (shared by every Spark
# TPU recipe): Spark's 4096 x 4096 experts overflow the 64 MiB scoped VMEM with the paired W1/W3
# kernel or untiled K/N, so the plain Pallas GMM runs with K and N tiled (512 / 1024 / 1024).


def _base(*, lora: bool, load_hf: bool) -> TPUTrainerConfig:
  converters = (
      [LoRAConverter.Config(rank=16, alpha=32.0, target_modules=list(LORA_TARGETS))] if lora else []
  )
  spec = model_registry("816B-A42B", seq_len=SEQ_LEN, converters=converters)
  spec = dataclasses.replace(spec, state_dict_adapter=FastSparkAdapter)
  cfg = TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="/mnt/model",
      metrics=MetricsProcessor.Config(log_freq=1),
      model_spec=spec,
      optimizer=default_adamw(lr=1e-4 if lora else 1e-6, implementation="foreach"),
      lr_scheduler=LRSchedulersContainer.Config(warmup_steps=1, decay_ratio=None),
      training=TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=SEQ_LEN,
          max_context_length=SEQ_LEN,
          steps=4,
          dtype="bfloat16",
          mixed_precision_param="bfloat16",
          # Full-param: reduce in bf16; an fp32 reduction would double the per-layer
          # unsharded expert gradients (EP=1 makes every layer's gradients 24 GiB).
          mixed_precision_reduce="float32" if lora else "bfloat16",
      ),
      dataloader=c4_test_dataloader(dataset_path=ensure_synthetic_text()),
      parallelism=ParallelismConfig(
          data_parallel_replicate_degree=1,
          data_parallel_shard_degree=-1,
          tensor_parallel_degree=1,
          pipeline_parallel_degree=1,
          context_parallel_degree=1,
          expert_parallel_degree=1,
          fsdp_reshard_after_forward="always",
          use_simple_fsdp=True,
          enable_tpu_topo_mesh=True,
      ),
      comm=CommConfig(init_timeout_seconds=1800, train_timeout_seconds=1800),
      activation_checkpoint=FullAC.Config(preserve_rng_state=False),
      compile=TpuTrainerCompileConfig(enable=False, backend="tpu", compile_grad_norm=True),
      validator=Validator.Config(enable=False),
      checkpoint=CheckpointManager.Config(
          enable=load_hf,
          initial_load_path="/mnt/model" if load_hf else None,
          initial_load_in_hf=load_hf,
          initial_load_model_only=True,
          load_only=True,
          interval=10**9,
      ),
      tpu_config=TPUConfig(enable_amp=False, eager_mode="DEFER_AND_FUSE"),
  )
  _fuse_route_score_backward(cfg)
  # Tokamax Splash attention, tiled Pallas grouped matmuls, scoped VMEM and power-state flags.
  tpu7x.apply(cfg)
  _enable_spark_fused_ce(cfg, frozen=lora)
  return cfg


def spark_816b_lora_64chip() -> TPUTrainerConfig:
  return _base(lora=True, load_hf=True)


def spark_816b_full_64chip() -> TPUTrainerConfig:
  return _base(lora=False, load_hf=True)


def spark_816b_lora_64chip_random() -> TPUTrainerConfig:
  return _base(lora=True, load_hf=False)


def spark_816b_full_64chip_random() -> TPUTrainerConfig:
  return _base(lora=False, load_hf=False)


def _tokens(cfg: TPUTrainerConfig, tokens_per_rank: int) -> TPUTrainerConfig:
  cfg.training.num_tokens_per_microbatch_per_dp_rank = tokens_per_rank
  return cfg


# Token-scaling probes (rows of 8,192 per rank): with EP=1 every MoE layer all-gathers its
# full expert weights once per pass whatever the token count, so step(n) ~ T_fixed + n * T_token.
def spark_816b_lora_64chip_random_16k() -> TPUTrainerConfig:
  return _tokens(_base(lora=True, load_hf=False), 2 * SEQ_LEN)


def spark_816b_lora_64chip_random_32k() -> TPUTrainerConfig:
  return _tokens(_base(lora=True, load_hf=False), 4 * SEQ_LEN)


def spark_816b_lora_64chip_16k() -> TPUTrainerConfig:
  return _tokens(_base(lora=True, load_hf=True), 2 * SEQ_LEN)


def _stock_splash(cfg: TPUTrainerConfig) -> TPUTrainerConfig:
  """The stock JAX Splash patch at its default tiles (the earlier default; A/B baseline)."""
  cfg.splash_attention_kernel = SplashAttentionKernelConfig(use_splash_attention_kernel=True)
  return cfg


def _expert_parallel(cfg: TPUTrainerConfig, degree: int = 8, capacity_factor: float = 2.0) -> TPUTrainerConfig:
  """Shard the routed experts over `degree` ranks (FSDP over the rest) with the all-gather dispatcher.

  The all-gather / reduce-scatter dispatcher moves as much data as an all-to-all dispatch at
  EP = top-k + 1, so it suits EP <= 8. The receive buffer holds capacity_factor x the routes of
  the EP group's tokens; routes past it are dropped.
  """
  cfg.parallelism.expert_parallel_degree = degree
  # Dense and expert gradients live on different meshes; TorchTitan's EP-aware clipping is used
  # (the compiled gradient norm needs expert parallelism 1).
  cfg.compile.compile_grad_norm = False
  cfg.model_spec.model.ep_capacity_factor = capacity_factor
  cfg.model_spec.model.ep_dispatcher = "allgather"
  # Per-step "[EP] ... max_buffer_fill=F max_dropped_frac=D" lines (F > 1.0: routes dropped on that rank).
  expert_placement.install_ep_monitor()
  return cfg


# Expert parallelism 8 with random weights (contiguous placement, capacity factor 2).
def spark_816b_lora_64chip_ep8_random() -> TPUTrainerConfig:
  return _expert_parallel(_base(lora=True, load_hf=False))


# Real weights with expert parallelism. With the released weights routing is skewed (the busiest
# EP-8 rank receives 2-2.5x the mean routes with the contiguous or interleaved placement), so a
# fixed buffer drops routes unless the placement is measured on the training text. Step 1: run
# spark_816b_lora_64chip_measure_placement (EP = 1, a few steps) with SPARK_EXPERT_PLACEMENT_OUT
# set; step 2: run the *_ep8 recipes with SPARK_EXPERT_PLACEMENT_FILE pointing at that file. With a
# measured placement, the BF16 all-gather combine and GMM-v2, a 1.5x buffer dropped no route for about
# 13 LoRA steps; routing drifts later, so LoRA uses 2x (full-parameter keeps 1.5x for HBM headroom)
# (expert_placement.py, docs/KNOWN_GAPS.md).
def _placement_file() -> str:
  path = os.environ.get("SPARK_EXPERT_PLACEMENT_FILE", "")
  if not path:
    raise ValueError(
        "real-weight expert parallelism needs a measured placement: set SPARK_EXPERT_PLACEMENT_FILE "
        "(written by spark_816b_lora_64chip_measure_placement with SPARK_EXPERT_PLACEMENT_OUT)")
  return path


def _expert_parallel_measured(cfg: TPUTrainerConfig, capacity_factor: float) -> TPUTrainerConfig:
  _expert_parallel(cfg, degree=8, capacity_factor=capacity_factor)
  cfg.model_spec.model.ep_dispatcher = "allgather_bf16"
  cfg.model_spec.model.ep_expert_placement = "file:" + _placement_file()
  return cfg


def spark_816b_lora_64chip_measure_placement() -> TPUTrainerConfig:
  """EP = 1, released weights: writes a load-balanced EP-8 placement at step 2."""
  if not os.environ.get("SPARK_EXPERT_PLACEMENT_OUT"):
    raise ValueError("set SPARK_EXPERT_PLACEMENT_OUT to the placement file to write")
  cfg = _base(lora=True, load_hf=True)
  cfg.training.steps = max(cfg.training.steps, int(os.environ.get("SPARK_EXPERT_STATS_STEP", "2")))
  return expert_placement.install_placement_writer(cfg)


def spark_816b_lora_64chip_ep8() -> TPUTrainerConfig:
  return _expert_parallel_measured(_base(lora=True, load_hf=True), capacity_factor=2.0)


def spark_816b_full_64chip_ep8() -> TPUTrainerConfig:
  return _expert_parallel_measured(_base(lora=False, load_hf=True), capacity_factor=1.5)


def spark_816b_lora_64chip_tokamax_random() -> TPUTrainerConfig:
  """Kept for continuity: every recipe now uses the Tokamax Splash settings."""
  return _base(lora=True, load_hf=False)


def spark_816b_lora_64chip_stock_splash_random() -> TPUTrainerConfig:
  """EP = 1, random weights, stock Splash at its default tiles (attention A/B)."""
  return _stock_splash(_base(lora=True, load_hf=False))


def spark_816b_lora_64chip_megablox_random() -> TPUTrainerConfig:
  """EP = 1, random weights, tiled MegaBlox grouped matmuls instead of GMM-v2 (expert-kernel A/B)."""
  cfg = _base(lora=True, load_hf=False)
  tpu7x.tile_expert_matmuls(cfg, gmm_v2=False)
  return cfg
