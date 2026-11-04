# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checkpointing utilities for large post-training runs on TPU slices.

* ``lora_checkpoint``: adapter-only checkpoints for LoRA training (resume
  state is the adapters plus their optimizer state; the frozen base is
  re-read from the Hugging Face checkpoint) and PEFT adapter export/import.
* ``hf_export``: streaming, layout-preserving export of trained weights to
  the original Hugging Face safetensors layout (same shard files, same
  tensor-to-file map, same dtypes), either from a full-parameter DCP
  checkpoint or by merging a PEFT LoRA adapter into the base checkpoint.
  Work is split by shard file, so many hosts can export in parallel;
  ``hf_export_check`` checks it at full scale on a real checkpoint.
* ``dcp_parallel``: multi-threaded DCP writer and parallel DCP readers for a
  TorchTitan ``CheckpointManager``.
* ``dcp_io_bench``: DCP save/load throughput benchmark against a POSIX path
  (e.g. a Cloud Storage FUSE mount), a ``gs://`` URL (gcsfs), or local
  staging plus parallel object upload.
* ``gcs``: Cloud Storage credentials and parallel transfer helpers.
* ``flush``: makes every rank leave a DCP save together on TorchTPU.

Nothing in this package is model specific; models plug in through their
TorchTitan ``StateDictAdapter``.
"""
