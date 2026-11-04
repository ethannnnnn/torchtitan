# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checkpointing utilities for large post-training runs on TPU slices.

* ``dcp_parallel``: multi-threaded DCP writer and parallel DCP readers for a
  TorchTitan ``CheckpointManager``.
* ``dcp_io_bench``: DCP save/load throughput benchmark against a POSIX path
  (e.g. a Cloud Storage FUSE mount), a ``gs://`` URL (gcsfs), or local
  staging plus parallel object upload.
* ``gcs``: Cloud Storage credentials and parallel transfer helpers.

Nothing in this package is model specific; models plug in through their
TorchTitan ``StateDictAdapter``.
"""
