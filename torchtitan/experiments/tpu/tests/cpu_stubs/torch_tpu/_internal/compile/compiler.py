# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU stand-in for ``torch_tpu._internal.compile.compiler``: ``StaticCompiler``.

TPU trainer modules import ``StaticCompiler`` at module level (for example
through ``muse_glimmer``'s config registry); the import succeeds here and
constructing the compiler raises.
"""


class StaticCompiler:
    """TorchTPU's whole-graph compiler, which needs TorchTPU."""

    def __init__(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "StaticCompiler needs TorchTPU; this is the CPU stand-in for torch_tpu"
        )
