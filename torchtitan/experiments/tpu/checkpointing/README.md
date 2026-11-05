# Checkpointing at scale (TPU)

Tools for checkpointing very large post-training runs (for example the
816B-parameter Muse Spark Open MoE on TPU7x slices) and for handing trained
weights to Hugging Face based serving and evaluation. Nothing here is model
specific: models plug in through their TorchTitan `StateDictAdapter`.

| Module | What it does |
|---|---|
| `lora_checkpoint.py` | Adapter-only checkpoints for LoRA runs; PEFT adapter export/import from sharded parameters |
| `hf_export.py` | Streaming export of trained weights in the *original* safetensors layout (from DCP, or by merging a PEFT adapter) |
| `hf_export_check.py` | Full-scale multi-host check of the export on a real checkpoint |
| `dcp_parallel.py` | Multi-threaded DCP reader and writer settings for a `CheckpointManager` |
| `dcp_io_bench.py` | DCP save/load throughput benchmark (Cloud Storage FUSE, `gs://` via gcsfs, local staging + parallel upload) |
| `gcs.py` | Cloud Storage credentials for GKE pods and parallel multipart transfers |
| `flush.py` | Makes every rank leave a DCP save together on TorchTPU |
| `resume_check.py` | Environment-gated kill and batch-record hooks for checkpoint/resume tests of TorchTitan trainers |

## LoRA runs

A LoRA run only changes its adapters. Writing the whole model at every
checkpoint (1.6 TB of bf16 weights for an 816B model) is wasted I/O.

```python
from torchtitan.experiments.tpu.checkpointing.lora_checkpoint import (
    enable_adapter_only_checkpoints, save_peft_adapter)

enable_adapter_only_checkpoints(checkpointer, model_parts)  # before checkpointer.load()
...
save_peft_adapter(model_parts, sd_adapter, "gs://<GCS_BUCKET>/<RUN_NAME>/adapter",
                  rank=16, alpha=32.0, storage_options={"token": creds})
```

* Saves: the trainable parameters, optimizer, LR scheduler, dataloader and
  train state (DCP). Rank-16 adapters on q/k/v/o of a 62-layer, 8192-wide model
  are 50.8M parameters: 0.10 GB in bf16, about 0.6 GB with fp32 AdamW moments.
* Loads: the frozen base from the Hugging Face checkpoint (through the model's
  adapter, which must skip `*.lora_a.weight` / `*.lora_b.weight`), then the
  newest adapter checkpoint. Afterwards `checkpointer.adapter_resume_step` is
  the restored step (`None` on a fresh start) and
  `checkpointer.adapter_load_seconds` the base and adapter load times.
* `save_peft_adapter` writes a standard PEFT LoRA directory. Engines that load
  LoRA adapters take it directly; `hf_export --peft` merges it into the original
  layout for engines that need merged weights. Merging into bf16 weights rounds
  them, so serve the adapter when the engine supports it.
* PEFT module names come from the model's adapter, and `lora_B` rows follow any
  row permutation the adapter applies to the base weight (for example a RoPE
  permutation of q/k), also when it casts the weight. An adapter on a weight
  that does not map one-to-one (such as a fused `wqkv` projection), or whose
  columns or values the model's adapter changes, is refused; export those with
  the model's own PEFT layout code.

## Export trained weights in the original layout

The output is a drop-in replacement for the original checkpoint directory:
same shard files, `model.safetensors.index.json`, header bytes, data offsets
and dtypes, plus the non-weight files. Untouched tensors (for example the
vision and audio towers of a text-only fine-tune) are byte-copied.

```bash
# merged LoRA, 16 workers (one command per worker; e.g. 4 per host on 4 hosts)
python -m torchtitan.experiments.tpu.checkpointing.hf_export \
  --base gs://<GCS_BUCKET>/muse-spark/model/Muse-Spark-Open-816B-A42B \
  --out gs://<GCS_BUCKET>/muse-spark/runs/<RUN_NAME>/hf \
  --peft gs://<GCS_BUCKET>/muse-spark/runs/<RUN_NAME>/adapter \
  --worker $I --num-workers 16 --staging-dir /stage
# then once, after every worker finished
python -m torchtitan.experiments.tpu.checkpointing.hf_export --base ... --out ... --peft ... \
  --num-workers 16 --finalize

# full-parameter run: read the DCP checkpoint through the model's adapter
python -m torchtitan.experiments.tpu.checkpointing.hf_export --base ... --out ... \
  --dcp gs://<GCS_BUCKET>/muse-spark/runs/<RUN_NAME>/checkpoint/step-<N> \
  --adapter-factory <module>:<function>   # function(hf_assets_path) -> StateDictAdapter
```

With `--staging-dir`, each shard is downloaded with parallel ranged reads,
rewritten locally and uploaded as a parallel multipart upload (a memory-backed
volume of about 2 x the largest shard per worker is enough). `--finalize`
checks that every shard landed with its original size, then writes the index,
small files, `export_manifest.json` (per-shard SHA-256) and `EXPORT_OK`.

A DCP checkpoint that also holds LoRA adapters is folded (`W + scale * B @ A`
in float32, `--lora-scale` = alpha / r) into the base weight of the same
module. Adapters without such a weight (a fused projection saved as its parts,
or an adapter-only checkpoint) are refused before anything is written: write a
PEFT adapter and merge it with `--peft` instead.

## Faster DCP checkpoints

```python
from torchtitan.experiments.tpu.checkpointing.dcp_parallel import tune_checkpoint_manager
tune_checkpoint_manager(checkpointer, save_threads=8, load_threads=8)
```

`FileSystemWriter` with 8 threads per rank instead of 1, and a reader that
fetches each item with one ranged read and processes items from a thread
pool. Works on local paths and Cloud Storage FUSE mounts. For `gs://` paths
use `ParallelFsspecReader` directly.

Runtime note: gcsfs 2026.10 requires fsspec >= 2026.9; with an older fsspec
every gcsfs read fails unless a cache type is passed explicitly (the readers
here do that).

`tune_checkpoint_manager` routes the checkpointer module's `dcp.save` through
the multi-threaded writer for the whole process (the latest settings win);
asynchronous saves, Hugging Face loads and saves, and remote paths keep the
stock readers and writers.

## Leaving a save together (TorchTPU)

TorchTPU defers device work until a host readback, so the last collective of a
DCP save (the coordinator's broadcast of the result) stays queued on rank 0,
which returns at once while every other rank waits for it until rank 0's next
readback. If rank 0 dies in between, the other ranks hang in the save although
the checkpoint is complete.

```python
from torchtitan.experiments.tpu.checkpointing.flush import flush_after_saves
flush_after_saves(checkpointer, device)
```

After every save that wrote, each rank all-reduces one int32 on the device and
reads it back, so every rank leaves the save together. It changes no numerics.

## Resume checks

`resume_check.install()` patches `torchtitan.trainer.Trainer` only when the
environment asks for it:

* `TT_FAULT_KILL_STEP=N`: every rank sends itself SIGKILL when step N draws its
  first batch, a hard crash for restart tests (relaunch with the same dump
  folder).
* `TT_BATCH_RECORDS_DIR=<dir>`: every rank appends one JSON line per step with
  the SHA-256 of each batch it drew to `<dir>/rank-NNNNN.jsonl`, and rank 0 logs
  `[resume-check] step=N loss=... grad_norm=...` at full precision, so two runs
  can be compared batch by batch and bit for bit across a restart.

The hooks live in `Trainer.batch_generator` and `Trainer.train_step`, so a
subclass gets them only through those methods: one that overrides
`batch_generator` without calling the base method is never killed and hashes
no batches, and one that overrides `train_step` that way (for example
`TpuGraphTrainer` with the optimizer in the graph) writes no batch records.

## Benchmark

```bash
python -m torch.distributed.run --nnodes=$NNODES --nproc-per-node=8 ... \
  torchtitan/experiments/tpu/checkpointing/dcp_io_bench.py --device tpu \
  --preset moe-816b --ref-devices 128 --methods fs,fsspec,stage \
  --fs-root /mnt/ckpt-w --fs-read-root /mnt/ckpt-r \
  --gs-root gs://<GCS_BUCKET>/<PREFIX>/bench --stage-root /stage \
  --threads 8 --par-threads 8 --out /mnt/out/results.json
```

The synthetic state has the per-device shard sizes of a `--ref-devices`
slice (bf16 parameters plus two 1-byte AdamW moments: 3.28 TB for 816B), so a
small slice measures the per-host rates of the large one. Every load is
verified bit-exact. `/mnt/ckpt-w` and `/mnt/ckpt-r` are two Cloud Storage FUSE
mounts of the same prefix, so reads cannot be served from the writer's page
cache. Pods with `hostNetwork: true` need explicit credentials for direct
Cloud Storage access (`gcs.resolve_credentials`: service account token
exchange, optionally impersonating the Google service account bound to the
Kubernetes service account).

## Measured on TPU7x

`dcp_io_bench` on a 2x2x4 slice (4 hosts, 8 ranks per host), synthetic state
with the per-device shard sizes of a 128-device (64-chip) slice: 25.6 GB per
rank, 205 GB per host, 3543 tensors, every load bit-exact. The per-host volume
equals that of a 64-chip slice holding the 816B model's bf16 parameters and
8-bit AdamW moments (3.28 TB), so per-host times carry over to 64 chips as long
as the bucket's aggregate bandwidth does (reads may be limited by the
project's Cloud Storage egress quota, 200 Gbps per region by default).

| Path (205 GB per host) | Save | Restore |
|---|---|---|
| Cloud Storage FUSE, stock DCP (1 writer thread, stock reader) | ~420 s (0.49 GB/s/host)* | ~445 s (0.46 GB/s/host)* |
| Cloud Storage FUSE, 8 writer threads, `ParallelFileSystemReader` | 71 s (2.9 GB/s/host) | 29 s (7.1 GB/s/host) |
| `gs://` (gcsfs), 8 writer threads, `ParallelFsspecReader` | 68 s (3.0 GB/s/host) | 61 s (3.4 GB/s/host) |
| Local memory volume, then parallel multipart upload / download | 20 s blocking + 44 s upload | 75 s download + 15 s load |

\* measured at 14 GB per host (same tensor shapes, 4 of 62 layers).
Device-to-host copies run at ~14.7 GB/s per host, so the stock DCP rates
are limited by the per-rank sequential reader/writer, not by the TPU or the
storage service.

## Tests

```bash
bash torchtitan/experiments/tpu/checkpointing/tests/run_cpu.sh [--tiny-checkpoint <dir>]
```

The model-agnostic tests run on CPU in a few minutes with TorchTitan's Llama 3
debugmodel (LoRA from `LoRAConverter`) and synthetic checkpoints: bit-exact
resume through `CheckpointManager` (full parameter, LoRA adapter-only, tuned
I/O; one process and two FSDP2 ranks), PEFT adapter export/import (including
RoPE row permutations), byte-identical no-op and multi-worker exports, PEFT
merges, DCP -> HF export through the model's adapter, the parallel DCP readers,
the save flush and the resume-check hooks.

`<dir>` is a local copy of the public test checkpoint
`trl-internal-testing/tiny-MuseSparkForConditionalGeneration`. With it (and the
Muse Spark port), the tests also cover a transformers round trip with
bit-identical logits, parity with the PEFT library's merge, DCP -> HF export of
the Muse Spark port, and bit-exact resume of the Muse Spark port (full
parameter, tuned I/O, LoRA adapter-only). On a machine without TorchTPU the
runner imports the TPU modules these checks need through the shared CPU
stand-in `torchtitan/experiments/tpu/tests/cpu_stubs`, if the tree has it.
