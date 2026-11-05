# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Import-only stand-in for TorchTPU, for CPU tests. ASCII only.

Some TorchTitan TPU modules import TorchTPU at module level (kernel
registration through ``torch_tpu._internal.pallas``), so a CPU machine
without TorchTPU cannot even build a TPU recipe's config. Put the parent
directory first on ``PYTHONPATH`` to import those modules on CPU: every
TorchTPU kernel raises when called, and nothing here runs on a TPU. The
package refuses to load where a real TorchTPU is installed, so it never
shadows one.
"""

import importlib.machinery
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_other_paths = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
if importlib.machinery.PathFinder.find_spec("torch_tpu", _other_paths) is not None:
    raise ImportError("a real torch_tpu is installed; remove the CPU import stub directory "
                      f"{_HERE} from PYTHONPATH")

IS_IMPORT_STUB = True
