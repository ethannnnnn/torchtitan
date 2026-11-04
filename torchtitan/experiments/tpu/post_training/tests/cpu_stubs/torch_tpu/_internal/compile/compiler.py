# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stand-in for ``torch_tpu._internal.compile.compiler`` (see ``torch_tpu/__init__.py``)."""

from torch_tpu import unavailable


class StaticCompiler:
    __init__ = unavailable("torch_tpu._internal.compile.compiler.StaticCompiler")
