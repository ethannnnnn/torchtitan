# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stand-in for ``torch_tpu._internal.pallas`` (see ``torch_tpu/__init__.py``).

``jax_op`` returns an op whose registration hooks (``register_autograd``,
``register_fake``, ...) do nothing, so kernel modules import; calling an op or
a JAX kernel bridge raises.
"""

import sys

from torch_tpu import unavailable


def _passthrough(fn=None, *args, **kwargs):
    """A registration hook, used directly or as a decorator."""
    del args, kwargs
    return fn if fn is not None else (lambda f: f)


class _JaxOp:
    def __init__(self, name: str, fn):
        self.name = name
        self.__wrapped__ = fn
        self._call = unavailable(f"torch_tpu._internal.pallas.jax_op({name!r})")

    def __call__(self, *args, **kwargs):
        return self._call(*args, **kwargs)

    register_autograd = staticmethod(_passthrough)
    register_fake = staticmethod(_passthrough)
    register_kernel = staticmethod(_passthrough)
    register_vmap = staticmethod(_passthrough)
    register_torch_dispatch = staticmethod(_passthrough)


def jax_op(name: str, *args, **kwargs):
    del args, kwargs

    def register(fn):
        return _JaxOp(name, fn)

    return register


custom_jax_kernel = unavailable("torch_tpu._internal.pallas.custom_jax_kernel")

# ``from torch_tpu._internal.pallas import pallas`` resolves to this module.
pallas = sys.modules[__name__]
