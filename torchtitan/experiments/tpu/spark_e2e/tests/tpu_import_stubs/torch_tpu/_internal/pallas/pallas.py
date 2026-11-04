# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Import-only stand-in for TorchTPU's JAX kernel bridge (see ``torch_tpu/__init__.py``).

``jax_op(name, fn)`` (also usable as a decorator) and ``custom_jax_kernel``
return callables that raise: TPU kernels cannot run on CPU. The autograd and
fake-tensor registration hooks the kernel modules call at import time accept
anything and do nothing.
"""

from __future__ import annotations

from typing import Any, Callable


class StubKernelError(NotImplementedError):
    """A TorchTPU kernel was called on a machine that only has the import stub."""


class _StubOp:
    def __init__(self, name: str, fn: Callable[..., Any] | None = None) -> None:
        self.name = name
        self.fn = fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise StubKernelError(f"TPU kernel {self.name!r} called with the CPU import stub of torch_tpu")

    def register_autograd(self, *args: Any, **kwargs: Any) -> None:
        return None

    def register_fake(self, fn: Callable[..., Any] | None = None) -> Any:
        return fn

    def register_kernel(self, *args: Any, **kwargs: Any) -> Any:
        return args[-1] if args and callable(args[-1]) else (lambda fn: fn)


def jax_op(name: str, fn: Callable[..., Any] | None = None, **kwargs: Any) -> Any:
    if fn is None:
        return lambda inner: _StubOp(name, inner)
    return _StubOp(name, fn)


def custom_jax_kernel(fn: Callable[..., Any] | None = None, *args: Any, **kwargs: Any) -> _StubOp:
    return _StubOp(getattr(fn, "__name__", "custom_jax_kernel"), fn)
