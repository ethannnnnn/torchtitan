# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Import-only stand-in for ``torch_tpu``, for CPU test runs.

Some TPU modules import TorchTPU at module level, for example the Pallas
kernel wrappers behind ``torchtitan.experiments.tpu.token_dispatcher`` and the
graph trainer's ``StaticCompiler``, even when a test only runs their CPU code
paths. With this directory first on ``PYTHONPATH`` such tests also run on a
machine without TorchTPU::

    PYTHONPATH=torchtitan/experiments/tpu/tests/cpu_stubs:. \\
        python -m pytest torchtitan/experiments/tpu/tests/test_ragged_ep_dispatcher.py

Only the names those modules use at import time exist, and calling a TPU
kernel raises. The stand-in refuses to load when a real ``torch_tpu`` is
installed, so it never shadows one.
"""

import importlib.machinery
import os
import sys

_STUB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OTHER_PATHS = [
    entry
    for entry in sys.path
    if os.path.abspath(entry or os.curdir) != _STUB_ROOT
]
if importlib.machinery.PathFinder.find_spec("torch_tpu", _OTHER_PATHS) is not None:
    raise ImportError(
        "torch_tpu is installed; remove the CPU stand-in directory"
        f" {_STUB_ROOT} from PYTHONPATH"
    )
