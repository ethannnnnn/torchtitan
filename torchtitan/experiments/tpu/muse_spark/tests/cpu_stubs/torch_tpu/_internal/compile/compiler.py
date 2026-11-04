# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stand-in for ``torch_tpu._internal.compile.compiler``."""


class StaticCompiler:
  """Importable placeholder; constructing it raises."""

  def __init__(self, *args, **kwargs):
    del args, kwargs
    raise RuntimeError(
        "StaticCompiler needs torch_tpu, which is only an import stub on this"
        " host."
    )
