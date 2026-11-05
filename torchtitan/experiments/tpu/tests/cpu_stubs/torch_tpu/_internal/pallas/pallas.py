# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU import stand-in for ``torch_tpu._internal.pallas.pallas``.

Kernel modules wrap their JAX functions when they are imported, with
``jax_op(name)`` / ``jax_op(name, fn)`` or ``custom_jax_kernel(fn)``, and then
register autograd and fake-tensor formulas on the result. Here that result is
a ``TpuOnlyOp`` (``jax_op``) or a ``JaxKernel`` (``custom_jax_kernel``): every
registration is accepted and ignored, and calling the kernel raises
``TorchTpuUnavailableError``.

As in TorchTPU, ``op._init_fn`` is the op's ``JaxKernel``, and its ``jit_fn``
is ``jax.jit`` of the JAX function with every non-array argument static. For
``jax_op`` the static arguments follow from the annotations: parameters
annotated ``jax.Array`` or ``torch.Tensor`` (also optional, or a sequence of
them) are arrays. Tests use ``jit_fn`` to trace a kernel's output shapes with
``jax.eval_shape``, which needs no TPU; JAX is imported only then.
"""

from __future__ import annotations

import collections.abc
import inspect
import types
import typing
from typing import Any, Callable

from torch_tpu import TorchTpuUnavailableError


def _passthrough(fn: Callable[..., Any] | None) -> Any:
  """``fn`` itself or, when ``fn`` is None, a decorator returning its argument."""
  return (lambda function: function) if fn is None else fn


class _TpuOnlyCallable:
  """A TorchTPU kernel here: registration hooks do nothing, calls raise.

  The hooks take the call forms of ``torch.library.CustomOpDef``'s.
  """

  name: str

  def __call__(self, *args, **kwargs):
    del args, kwargs
    raise TorchTpuUnavailableError(
        f"TPU kernel {self.name!r} called, but this host only has the CPU"
        " import stand-in for TorchTPU; run on TPU or use the operation's"
        " CPU path"
    )

  def register_autograd(self, backward, /, *, setup_context=None) -> None:
    del backward, setup_context

  def register_fake(self, fn, /):
    return fn

  def register_kernel(self, device_types, fn=None, /):
    del device_types
    return _passthrough(fn)

  def register_torch_dispatch(self, torch_dispatch_class, fn=None, /):
    del torch_dispatch_class
    return _passthrough(fn)

  def register_vmap(self, func=None, /):
    return _passthrough(func)

  def register_autocast(self, device_type, cast_inputs) -> None:
    del device_type, cast_inputs


def _is_array_annotation(annotation: Any) -> bool:
  """Whether TorchTPU passes a parameter with this annotation as a tensor."""
  # pylint: disable=g-import-not-at-top
  import jax
  import torch

  # pylint: enable=g-import-not-at-top
  origin = typing.get_origin(annotation)
  args = typing.get_args(annotation)
  if origin is typing.Annotated:
    return _is_array_annotation(args[0])
  if origin in (typing.Union, types.UnionType):  # Optional[X], X | None
    present = [arg for arg in args if arg is not type(None)]
    return len(present) == 1 and _is_array_annotation(present[0])
  if origin in (list, tuple, collections.abc.Sequence):
    elements = [arg for arg in args if arg is not Ellipsis]
    return bool(elements) and all(map(_is_array_annotation, elements))
  return annotation in (jax.Array, torch.Tensor)


def _static_arguments(
    fn: Callable[..., Any],
) -> tuple[tuple[int, ...], tuple[str, ...]]:
  """Non-array parameters of ``fn``: positional ones by index, keyword-only
  ones by name (TorchTPU's ``jax_op`` rule)."""
  signature = inspect.signature(inspect.unwrap(fn), eval_str=True)
  argnums, argnames = [], []
  for index, param in enumerate(signature.parameters.values()):
    if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
      continue
    if _is_array_annotation(param.annotation):
      continue
    if param.kind is param.KEYWORD_ONLY:
      argnames.append(param.name)
    else:
      argnums.append(index)
  return tuple(argnums), tuple(argnames)


class JaxKernel(_TpuOnlyCallable):
  """A JAX function registered as a TorchTPU kernel; it cannot run here.

  What ``custom_jax_kernel`` returns, and a ``jax_op``'s ``_init_fn``.
  ``jit_fn``, built on first use, is ``jax.jit`` of the function with the
  given static arguments, or with the ones inferred from its annotations when
  none are given.
  """

  def __init__(
      self,
      name: str,
      fn: Callable[..., Any],
      static_argnums: tuple[int, ...] | None = None,
      static_argnames: tuple[str, ...] | None = None,
  ):
    self.name = name
    self.fn = fn
    self.__wrapped__ = fn
    if static_argnums is None and static_argnames is None:
      self._static = None
    else:
      self._static = (tuple(static_argnums or ()), tuple(static_argnames or ()))
    self._jit_fn = None

  @property
  def jit_fn(self) -> Any:
    if self._jit_fn is None:
      import jax  # pylint: disable=g-import-not-at-top

      argnums, argnames = self._static or _static_arguments(self.fn)
      self._jit_fn = jax.jit(
          self.fn, static_argnums=argnums, static_argnames=argnames
      )
    return self._jit_fn


class TpuOnlyOp(_TpuOnlyCallable):
  """What ``jax_op`` returns: a TorchTPU custom op that cannot run here."""

  def __init__(self, name: str, fn: Callable[..., Any]):
    self.name = name
    self._init_fn = JaxKernel(name, fn)
    self.__wrapped__ = fn
    self.__name__ = getattr(fn, "__name__", name)
    self.__doc__ = getattr(fn, "__doc__", None)


def jax_op(name: str, fn: Callable[..., Any] | None = None, /, **kwargs: Any):
  """``jax_op(name)`` returns a decorator, ``jax_op(name, fn)`` wraps ``fn``.

  The real ``jax_op``'s keyword arguments (argument donation, mesh, partition
  specs) are accepted and ignored.
  """
  del kwargs

  def decorate(function: Callable[..., Any]) -> TpuOnlyOp:
    return TpuOnlyOp(name, function)

  return decorate if fn is None else decorate(fn)


def custom_jax_kernel(
    jax_fn: Callable[..., Any] | None = None,
    name: str | None = None,
    static_argnums: tuple[int, ...] = (),
    *args: Any,
    static_argnames: tuple[str, ...] = (),
    **kwargs: Any,
):
  """``custom_jax_kernel(fn)`` returns a ``JaxKernel``; without ``fn``, a decorator.

  Arguments other than the name and the static arguments (argument donation,
  mesh, partition specs) are accepted and ignored.
  """
  del args, kwargs

  def wrap(function: Callable[..., Any]) -> JaxKernel:
    kernel_name = name or getattr(function, "__name__", "custom_jax_kernel")
    return JaxKernel(kernel_name, function, static_argnums, static_argnames)

  return wrap if jax_fn is None else wrap(jax_fn)
