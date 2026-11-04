# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Import-only stand-in for ``torch_tpu`` on hosts without it (CPU tests).

Several TPU-lane modules import a few ``torch_tpu`` names at import time: the
Pallas op registration decorator (``torch_tpu._internal.pallas.jax_op``) and
the static compiler class. This package defines exactly those names so the CPU
tests can import the model package, the TPU MoE converters and the trainer
configs. Calling a TPU kernel or the compiler raises ``RuntimeError``.

Usage, on a host without ``torch_tpu`` only::

  PYTHONPATH=torchtitan/experiments/tpu/muse_spark/tests/cpu_stubs:$PWD \\
      python -m pytest -q torchtitan/experiments/tpu/muse_spark/tests
"""

import importlib.machinery
import os
import sys

IS_IMPORT_STUB = True


def _installed_torch_tpu_spec():
  """Spec of a real ``torch_tpu`` on ``sys.path`` past this stub, if any."""
  stub_root = os.path.realpath(os.path.dirname(os.path.dirname(__file__)))
  others = [
      entry
      for entry in sys.path
      if os.path.realpath(entry or os.curdir) != stub_root
  ]
  return importlib.machinery.PathFinder.find_spec("torch_tpu", others)


if _installed_torch_tpu_spec() is not None:
  raise ImportError(
      "torch_tpu is installed on this host; remove the muse_spark tests"
      " cpu_stubs directory from PYTHONPATH instead of shadowing it."
  )
