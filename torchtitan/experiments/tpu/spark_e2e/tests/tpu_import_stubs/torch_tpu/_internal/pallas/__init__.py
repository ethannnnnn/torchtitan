# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Import-only stand-in for ``torch_tpu._internal.pallas`` (see ``torch_tpu/__init__.py``)."""

from . import pallas
from .pallas import custom_jax_kernel, jax_op, StubKernelError

__all__ = ["StubKernelError", "custom_jax_kernel", "jax_op", "pallas"]
