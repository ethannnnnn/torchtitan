# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stand-in for ``torch_tpu._internal.pallas``: ops register, never run."""

from typing import Any, Callable


class UnavailableTpuOp:
  """What ``@jax_op(name)`` returns here: importable, not callable."""

  def __init__(self, name: str, fn: Callable[..., Any]):
    self.op_name = name
    self.__wrapped__ = fn
    self.__name__ = getattr(fn, "__name__", name)
    self.__doc__ = getattr(fn, "__doc__", None)

  def register_autograd(self, backward, *, setup_context=None) -> None:
    del backward, setup_context  # Nothing runs on this host.

  def __call__(self, *args, **kwargs):
    del args, kwargs
    raise RuntimeError(
        f"TPU kernel {self.op_name!r} called, but torch_tpu is only an import"
        " stub on this host."
    )


def jax_op(name: str, *args, **kwargs) -> Callable[..., UnavailableTpuOp]:
  """Decorator factory with the signature shape of the real ``jax_op``."""
  del args, kwargs

  def decorate(fn: Callable[..., Any]) -> UnavailableTpuOp:
    return UnavailableTpuOp(name, fn)

  return decorate
