# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Import-only stand-in for ``torch_tpu._internal.compile.compiler`` (see ``torch_tpu/__init__.py``)."""

from __future__ import annotations

from typing import Any

from torch_tpu._internal.pallas.pallas import StubKernelError


class StaticCompiler:
    """Raises when used: graph compilation needs the real TorchTPU."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise StubKernelError("StaticCompiler needs TorchTPU (CPU import stub in use)")
