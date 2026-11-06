# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU import stand-in for ``torch_tpu._internal.compile.compiler``.

TPU trainer modules import ``StaticCompiler`` at module level (for example
through the graph trainer and ``muse_glimmer``'s config registry); the import
succeeds here and constructing the compiler raises.
"""

from torch_tpu import unavailable


class StaticCompiler:
  """TorchTPU's whole-graph compiler, which needs TorchTPU."""

  __init__ = unavailable("torch_tpu._internal.compile.compiler.StaticCompiler")
