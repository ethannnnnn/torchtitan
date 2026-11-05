# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Import-only stand-in for TorchTPU (``torch_tpu``), for CPU test runs.

Many TPU-lane modules import TorchTPU at module level even when a test only
runs their CPU code paths: kernel modules register their JAX functions through
``torch_tpu._internal.pallas`` (``jax_op``, ``custom_jax_kernel``), trainer and
graph-trainer modules import the ``StaticCompiler`` class, and the scan
backward patch imports ``torch_tpu._internal.compile.scan``. With this
directory first on ``PYTHONPATH`` those modules import, and their CPU tests
run, on a machine without TorchTPU::

    PYTHONPATH=torchtitan/experiments/tpu/tests/cpu_stubs:. \\
        python -m pytest torchtitan/experiments/tpu/muse_spark/tests

Only names that TPU-lane modules use at import time exist. Calling a kernel or
constructing a compiler raises ``TorchTpuUnavailableError`` (a
``NotImplementedError``, so also a ``RuntimeError``); nothing here registers a
device or runs on a TPU. Kernel registration hooks are accepted and ignored. As
in TorchTPU, a ``jax_op``'s ``_init_fn.jit_fn`` is the ``jax.jit``-compiled
JAX function with its non-array arguments static, so a test can still trace a
kernel's output shapes with ``jax.eval_shape``. The package refuses to load
where a real ``torch_tpu`` is importable, so it never shadows one.
"""

import importlib.machinery
import os
import sys

IS_IMPORT_STUB = True
"""Marks this stand-in; the real package has no such attribute."""

STAND_IN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
"""The directory to put on ``PYTHONPATH`` (it holds this package)."""


class TorchTpuUnavailableError(NotImplementedError):
  """A TorchTPU kernel or compiler was used where only this stand-in exists."""


def unavailable(name: str):
  """Returns a function that raises ``TorchTpuUnavailableError`` for ``name``."""

  def call(*args, **kwargs):
    del args, kwargs
    raise TorchTpuUnavailableError(
        f"{name} needs TorchTPU; this host only has its CPU import stand-in"
        f" ({STAND_IN_ROOT})"
    )

  call.__name__ = call.__qualname__ = name.rsplit(".", 1)[-1]
  return call


def _installed_torch_tpu_spec():
  """Spec of a ``torch_tpu`` package importable without this stand-in."""
  root = os.path.realpath(STAND_IN_ROOT)
  others = [
      entry for entry in sys.path
      if os.path.realpath(entry or os.curdir) != root
  ]
  spec = importlib.machinery.PathFinder.find_spec("torch_tpu", others)
  # A directory named torch_tpu without __init__.py is no installation.
  return spec if spec is not None and spec.origin is not None else None


if _installed_torch_tpu_spec() is not None:
  raise ImportError(
      "torch_tpu is installed; remove the CPU stand-in directory"
      f" {STAND_IN_ROOT} from PYTHONPATH instead of shadowing it"
  )
