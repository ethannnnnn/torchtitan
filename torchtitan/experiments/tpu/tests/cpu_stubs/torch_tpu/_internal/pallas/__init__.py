# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU import stand-in for ``torch_tpu._internal.pallas`` (see ``torch_tpu/__init__.py``).

Both import forms of the real package work: ``from torch_tpu._internal import
pallas`` followed by ``pallas.jax_op``, and ``from torch_tpu._internal.pallas
import pallas`` (the submodule that defines the names).
"""

from . import pallas
from .pallas import custom_jax_kernel
from .pallas import jax_op
from .pallas import JaxKernel
from .pallas import TpuOnlyOp

__all__ = ["JaxKernel", "TpuOnlyOp", "custom_jax_kernel", "jax_op", "pallas"]
