# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checks of the CPU import stand-in for TorchTPU in ``cpu_stubs/``.

Each check runs a fresh interpreter with the stand-in first on ``PYTHONPATH``,
so the stand-in never enters this process. Where a real ``torch_tpu`` is
installed the stand-in refuses to load, and only that refusal is checked.

Run: ``python -m torchtitan.experiments.tpu.tests.test_cpu_stubs``.
"""

import importlib.machinery
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

from absl.testing import absltest

_STAND_IN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpu_stubs")
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), *[os.pardir] * 4)
)


def _torch_tpu_installed() -> bool:
  """Whether a real ``torch_tpu`` is importable here, ignoring the stand-in."""
  root = os.path.realpath(_STAND_IN)
  paths = [p for p in sys.path if os.path.realpath(p or os.curdir) != root]
  spec = importlib.machinery.PathFinder.find_spec("torch_tpu", paths)
  return spec is not None and spec.origin is not None


def _run(code: str, *paths: str) -> subprocess.CompletedProcess:
  """Runs ``code`` in a new interpreter: the stand-in, ``paths``, the repo."""
  env = dict(os.environ, JAX_PLATFORMS="cpu", TORCH_DEVICE_BACKEND_AUTOLOAD="0")
  env["PYTHONPATH"] = os.pathsep.join([_STAND_IN, *paths, _REPO_ROOT])
  return subprocess.run(
      [sys.executable, "-c", textwrap.dedent(code)],
      env=env,
      capture_output=True,
      text=True,
      timeout=900,
      check=False,
  )


class CpuStandInGuardTest(absltest.TestCase):

  def test_refuses_to_shadow_an_installed_torch_tpu(self):
    with tempfile.TemporaryDirectory() as other:
      package = os.path.join(other, "torch_tpu")
      os.mkdir(package)
      with open(os.path.join(package, "__init__.py"), "w"):
        pass
      result = _run("import torch_tpu", other)
    self.assertNotEqual(result.returncode, 0, result.stdout)
    self.assertIn("remove the CPU stand-in directory", result.stderr)


@unittest.skipIf(
    _torch_tpu_installed(),
    "torch_tpu is installed, so the stand-in refuses to load",
)
class CpuStandInTest(absltest.TestCase):

  def _check(self, code: str) -> None:
    result = _run(code)
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(result.stdout.strip().splitlines()[-1:], ["ok"])

  def test_import_forms_and_compiler(self):
    self._check("""
        import torch_tpu
        from torch_tpu._internal import pallas
        from torch_tpu._internal.compile.compiler import StaticCompiler
        from torch_tpu._internal.compile.scan import SCAN_REVERSE_ANNOTATION
        from torch_tpu._internal.pallas import pallas as pallas_module

        assert torch_tpu.IS_IMPORT_STUB
        assert issubclass(torch_tpu.TorchTpuUnavailableError, RuntimeError)
        assert pallas.jax_op is pallas_module.jax_op
        assert pallas.custom_jax_kernel is pallas_module.custom_jax_kernel
        assert isinstance(SCAN_REVERSE_ANNOTATION, str)
        try:
          StaticCompiler(object())
        except torch_tpu.TorchTpuUnavailableError as error:
          assert "StaticCompiler" in str(error), error
        else:
          raise AssertionError("StaticCompiler() did not raise")
        print("ok")
    """)

  def test_jax_op_registers_raises_and_traces(self):
    self._check("""
        import jax
        import jax.numpy as jnp
        import torch_tpu
        from torch_tpu._internal import pallas

        @pallas.jax_op("cpu_stand_in_test::scale")
        def scale(x: jax.Array, factor: int, *, transpose: bool = False) -> jax.Array:
          '''Scales x.'''
          return (x.T if transpose else x) * factor

        direct = pallas.jax_op(
            "cpu_stand_in_test::scale_direct", scale.__wrapped__, donate_argnums=(0,)
        )
        assert (scale.__name__, scale.__doc__) == ("scale", "Scales x.")
        scale.register_autograd(
            lambda ctx, grad: (grad, None), setup_context=lambda *args: None
        )
        assert scale.register_fake(len) is len
        assert scale.register_kernel("cpu")(len) is len
        assert scale.register_kernel("cpu", len) is len
        assert scale.register_vmap()(len) is len
        for op in (scale, direct):
          try:
            op(1, 2)
          except torch_tpu.TorchTpuUnavailableError as error:
            assert "cpu_stand_in_test::scale" in str(error), error
          else:
            raise AssertionError("the op ran")
        # As in TorchTPU, the int and bool parameters are static.
        out = jax.eval_shape(
            scale._init_fn.jit_fn,
            jax.ShapeDtypeStruct((2, 3), jnp.bfloat16),
            4,
            transpose=True,
        )
        assert (out.shape, out.dtype) == ((3, 2), jnp.bfloat16), out
        print("ok")
    """)

  def test_custom_jax_kernel(self):
    self._check("""
        import jax
        import jax.numpy as jnp
        import torch_tpu
        from torch_tpu._internal import pallas

        def repeat(x, times):
          return jnp.tile(x, times)

        kernel = pallas.custom_jax_kernel(repeat, static_argnums=(1,))
        named = pallas.custom_jax_kernel(name="cpu_stand_in_repeat", static_argnums=(1,))(
            repeat
        )
        assert (kernel.name, named.name) == ("repeat", "cpu_stand_in_repeat")
        assert kernel.register_fake(len) is len
        out = jax.eval_shape(named.jit_fn, jax.ShapeDtypeStruct((4,), jnp.float32), 3)
        assert out.shape == (12,), out
        try:
          kernel(1, 2)
        except torch_tpu.TorchTpuUnavailableError:
          pass
        else:
          raise AssertionError("the kernel ran")
        print("ok")
    """)

  def test_tpu_lane_modules_import(self):
    """Modules that use the stand-in's names at import time import."""
    self._check("""
        import importlib

        for name in (
            "torchtitan.experiments.tpu.kernels.splash_attention",
            "torchtitan.experiments.tpu.kernels.expert_segment_matmul",
            "torchtitan.experiments.tpu.scan_backward_patch",
            "torchtitan.experiments.tpu.compiled_optimizers.separate_graph",
        ):
          importlib.import_module(name)
        print("ok")
    """)


if __name__ == "__main__":
  absltest.main()
