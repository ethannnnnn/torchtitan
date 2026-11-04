# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU stand-in for ``torch_tpu._internal.pallas``: ``jax_op`` only.

Kernel modules decorate their JAX functions with ``jax_op`` and register an
autograd formula on the result when they are imported. Here the decorated
function becomes a ``TpuOnlyOp`` that accepts those registrations and raises
when it is called.
"""

from collections.abc import Callable
from typing import Any


class TpuOnlyOp:
    """A ``jax_op`` kernel that cannot run without TorchTPU."""

    def __init__(self, name: str, fn: Callable[..., Any]):
        self.name = name
        self.__wrapped__ = fn
        self.__doc__ = fn.__doc__

    def register_autograd(self, backward, /, *, setup_context=None) -> None:
        del backward, setup_context  # The kernel never runs.

    def register_fake(self, fn):
        return fn

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"{self.name} is a TPU kernel and this is the CPU stand-in for"
            " torch_tpu; run on TPU or use the operation's CPU path"
        )


def jax_op(name: str, fn: Callable[..., Any] | None = None, /, **kwargs: Any):
    """Same call forms as ``torch_tpu._internal.pallas.jax_op``.

    ``jax_op(name)`` returns a decorator and ``jax_op(name, fn)`` wraps ``fn``
    directly; keyword arguments (donation, mesh, partition specs) are ignored.
    """
    del kwargs

    def decorate(function: Callable[..., Any]) -> TpuOnlyOp:
        return TpuOnlyOp(name, function)

    return decorate if fn is None else decorate(fn)
