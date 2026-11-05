# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the frozen-weight backward of the expert grouped matmul.

``expert_segment_matmul`` imports ``torch_tpu._internal.pallas``. These tests
load a private copy of it against a small CPU stand-in for that module. The
stand-in's ``jax_op`` registers the JAX function as a ``torch.library`` custom
op, as torch_tpu does, and runs it on CPU with the Pallas TPU kernels in
interpret mode. The module's autograd code therefore runs unchanged without a
TPU, and an installed torch_tpu is neither used nor modified.
``test_expert_segment_matmul`` covers the kernels on TPU.
"""

import collections
import importlib.util
import inspect
import sys
import types
import typing
import uuid

from absl.testing import absltest
from absl.testing import parameterized
from functorch.compile import make_boxed_func
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import torch
from torch._dynamo.backends.common import aot_autograd

_MODULE = "torchtitan.experiments.tpu.kernels.expert_segment_matmul"

# Kernel invocations of the stand-in ops, by op name (e.g. "grouped_mm_dgrad").
_KERNEL_CALLS = collections.Counter()
_JAX_DTYPES = {
    torch.bfloat16: jnp.bfloat16,
    torch.float32: jnp.float32,
    torch.int32: jnp.int32,
}
_TORCH_DTYPES = {jnp.dtype(v): k for k, v in _JAX_DTYPES.items()}


def _torch_annotation(annotation):
    if annotation is jax.Array:
        return torch.Tensor
    if typing.get_origin(annotation) is tuple:
        return tuple[tuple(map(_torch_annotation, typing.get_args(annotation)))]
    return annotation


def _to_jax(value):
    if isinstance(value, torch.Tensor):
        return jax.dlpack.from_dlpack(value.detach().contiguous())
    return value


def _to_shape_dtype(value):
    if isinstance(value, torch.Tensor):
        return jax.ShapeDtypeStruct(value.shape, _JAX_DTYPES[value.dtype])
    return value


def _to_torch(array: jax.Array) -> torch.Tensor:
    # Copy, so that autograd never writes into a buffer owned by JAX.
    return torch.from_dlpack(array).clone()


def _cpu_jax_op(name: str):
    """CPU stand-in for ``torch_tpu._internal.pallas.jax_op``.

    Registers ``fn`` as a ``torch.library`` custom op (in a fresh namespace, so
    it never collides with torch_tpu's ops) whose kernel runs ``fn`` under
    ``jax.jit`` on CPU, with Pallas TPU kernels in interpret mode.
    """
    namespace, op_name = name.split("::")
    qualname = f"{namespace}_cpu_{uuid.uuid4().hex[:8]}::{op_name}"

    def register(fn):
        signature = inspect.signature(fn, eval_str=True)
        params = list(signature.parameters.values())
        static_argnums = tuple(
            i for i, param in enumerate(params) if param.annotation is not jax.Array
        )
        jit_fn = jax.jit(fn, static_argnums=static_argnums)

        def kernel(*args):
            _KERNEL_CALLS[op_name] += 1
            with pltpu.force_tpu_interpret_mode():
                out = jit_fn(*map(_to_jax, args))
            return jax.tree.map(_to_torch, out)

        def fake(*args):
            shapes = jax.eval_shape(jit_fn, *map(_to_shape_dtype, args))
            return jax.tree.map(
                lambda s: torch.empty(s.shape, dtype=_TORCH_DTYPES[s.dtype]), shapes
            )

        kernel.__signature__ = signature.replace(
            parameters=[
                param.replace(annotation=_torch_annotation(param.annotation))
                for param in params
            ],
            return_annotation=_torch_annotation(signature.return_annotation),
        )
        # torch_tpu's kernels expose the jitted JAX function the same way.
        kernel.jit_fn = jit_fn
        op = torch.library.custom_op(qualname, kernel, mutates_args=())
        op.register_fake(fake)
        return op

    return register


def _load_on_cpu():
    """Returns a private copy of the module, bound to the CPU stand-in."""
    pallas = types.ModuleType("torch_tpu._internal.pallas")
    pallas.jax_op = _cpu_jax_op
    internal = types.ModuleType("torch_tpu._internal")
    internal.pallas = pallas
    package = types.ModuleType("torch_tpu")
    package._internal = internal
    stand_ins = {
        "torch_tpu": package,
        "torch_tpu._internal": internal,
        "torch_tpu._internal.pallas": pallas,
    }
    spec = importlib.util.find_spec(_MODULE)
    module = importlib.util.module_from_spec(spec)
    previous = {name: sys.modules.get(name) for name in stand_ins}
    sys.modules.update(stand_ins)
    try:
        spec.loader.exec_module(module)
    finally:
        for name, value in previous.items():
            if value is None:
                del sys.modules[name]
            else:
                sys.modules[name] = value
    return module


# One empty expert, and group boundaries that are not multiples of _TILE_M.
_GROUP_SIZES = (96, 0, 160)
_TILE_M = 128
_DIM_K = 128
_DIM_N = 256


class FrozenExpertWeightsTest(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.kernels = _load_on_cpu()

    def setUp(self):
        super().setUp()
        torch.compiler.reset()
        _KERNEL_CALLS.clear()
        self._make_tensors(_DIM_K, _DIM_N)

    def _make_tensors(self, dim_k, dim_n):
        generator = torch.Generator().manual_seed(0)
        rows = sum(_GROUP_SIZES)
        self.group_sizes = torch.tensor(_GROUP_SIZES, dtype=torch.int32)
        self.lhs = torch.randn(rows, dim_k, generator=generator).bfloat16()
        self.weights_EKN = torch.randn(
            len(_GROUP_SIZES), dim_k, dim_n, generator=generator
        ).bfloat16()
        self.grad_output = torch.randn(rows, dim_n, generator=generator).bfloat16()

    def _inputs(self, *, transpose_rhs, train_weights):
        lhs = self.lhs.clone().requires_grad_()
        if transpose_rhs:
            rhs = self.weights_EKN.transpose(-2, -1).contiguous()
        else:
            rhs = self.weights_EKN.clone()
        return lhs, rhs.requires_grad_(train_weights)

    def _forward_backward(
        self, *, transpose_rhs, train_weights, grouped_mm=None, **tiles
    ):
        """Returns the output and the gradients of lhs and rhs."""
        grouped_mm = grouped_mm or self.kernels.segment_matmul_pallas_2d
        lhs, rhs = self._inputs(
            transpose_rhs=transpose_rhs, train_weights=train_weights
        )
        out = grouped_mm(
            lhs,
            rhs,
            self.group_sizes,
            tile_m=_TILE_M,
            transpose_rhs=transpose_rhs,
            **tiles,
        )
        out.backward(self.grad_output)
        return out.detach(), lhs.grad, rhs.grad

    def _fp32_reference(self):
        """FP32 dInput (rows, K) and dWeight (E, K, N), one expert at a time."""
        dinput, dweight = [], []
        for lhs, grad, weight in zip(
            self.lhs.split(_GROUP_SIZES),
            self.grad_output.split(_GROUP_SIZES),
            self.weights_EKN,
        ):
            dinput.append(grad.float() @ weight.float().T)
            dweight.append(lhs.float().T @ grad.float())
        return torch.cat(dinput), torch.stack(dweight)

    @parameterized.named_parameters(("kn", False), ("nk", True))
    def test_frozen_weights_get_the_dinput_of_the_full_backward(self, transpose_rhs):
        out_full, dinput_full, dweight_full = self._forward_backward(
            transpose_rhs=transpose_rhs, train_weights=True
        )
        self.assertEqual(_KERNEL_CALLS, {"grouped_mm": 1, "grouped_mm_bwd": 1})

        _KERNEL_CALLS.clear()
        out, dinput, dweight = self._forward_backward(
            transpose_rhs=transpose_rhs, train_weights=False
        )
        # Only the dInput GMM runs, and the frozen weights get no gradient.
        self.assertEqual(_KERNEL_CALLS, {"grouped_mm": 1, "grouped_mm_dgrad": 1})
        self.assertIsNone(dweight)
        # The same GMM with the same tiling as in the full backward.
        torch.testing.assert_close(out, out_full, rtol=0, atol=0)
        torch.testing.assert_close(dinput, dinput_full, rtol=0, atol=0)

        # Both backwards are also right, to about one BF16 ULP.
        want_dinput, want_dweight = self._fp32_reference()
        if transpose_rhs:
            dweight_full = dweight_full.transpose(-2, -1)
        torch.testing.assert_close(dinput.float(), want_dinput, rtol=8e-3, atol=1e-3)
        torch.testing.assert_close(
            dweight_full.float(), want_dweight, rtol=8e-3, atol=1e-3
        )

    @parameterized.named_parameters(("kn", False), ("nk", True))
    def test_frozen_weights_get_the_full_dinput_with_k_and_n_tiles(self, transpose_rhs):
        """K and N tiled below their sizes, with a different tile for each.

        The default zero tiles resolve to the full dimensions, so they cannot
        tell the K and N tiles of the dInput GMM apart.
        """
        self._make_tensors(dim_k=256, dim_n=512)
        tiles = {"tile_k": 128, "tile_n": 256}
        _, dinput_full, _ = self._forward_backward(
            transpose_rhs=transpose_rhs, train_weights=True, **tiles
        )
        _KERNEL_CALLS.clear()
        _, dinput, dweight = self._forward_backward(
            transpose_rhs=transpose_rhs, train_weights=False, **tiles
        )

        self.assertEqual(_KERNEL_CALLS, {"grouped_mm": 1, "grouped_mm_dgrad": 1})
        self.assertIsNone(dweight)
        torch.testing.assert_close(dinput, dinput_full, rtol=0, atol=0)
        want_dinput, _ = self._fp32_reference()
        torch.testing.assert_close(dinput.float(), want_dinput, rtol=8e-3, atol=1e-3)

    @parameterized.named_parameters(("trainable", True), ("frozen", False))
    def test_lhs_is_saved_for_backward_only_for_trainable_weights(self, train_weights):
        lhs, rhs = self._inputs(transpose_rhs=False, train_weights=train_weights)
        saved = []

        def pack(tensor):
            saved.append(tuple(tensor.shape))
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            self.kernels.segment_matmul_pallas_2d(
                lhs, rhs, self.group_sizes, tile_m=_TILE_M
            )

        want = [tuple(rhs.shape), tuple(self.group_sizes.shape)]
        if train_weights:
            want.insert(0, tuple(lhs.shape))
        self.assertEqual(saved, want)

    def test_frozen_weights_skip_dweight_under_aot_autograd(self):
        """AOTAutograd traces the frozen branch, as torch.compile training does."""
        _, dinput_eager, _ = self._forward_backward(
            transpose_rhs=False, train_weights=False
        )
        graphs = {}

        def capture(name):
            def compiler(graph_module, example_inputs):
                del example_inputs
                graphs[name] = graph_module
                return make_boxed_func(graph_module.forward)

            return compiler

        backend = aot_autograd(
            fw_compiler=capture("forward"), bw_compiler=capture("backward")
        )
        grouped_mm = torch.compile(
            self.kernels.segment_matmul_pallas_2d, backend=backend, fullgraph=True
        )
        _KERNEL_CALLS.clear()
        _, dinput, dweight = self._forward_backward(
            transpose_rhs=False, train_weights=False, grouped_mm=grouped_mm
        )

        self.assertEqual(_KERNEL_CALLS, {"grouped_mm": 1, "grouped_mm_dgrad": 1})
        self.assertIsNone(dweight)
        torch.testing.assert_close(dinput, dinput_eager, rtol=0, atol=0)
        backward = graphs["backward"].graph
        targets = {
            str(node.target) for node in backward.nodes if node.op == "call_function"
        }
        self.assertTrue(any("grouped_mm_dgrad" in t for t in targets), targets)
        self.assertFalse(any("grouped_mm_bwd" in t for t in targets), targets)
        # The backward graph reads no LHS-shaped tensor, so the compiled
        # forward keeps none alive for it. (AOTAutograd saves only what the
        # backward graph reads; the saved-tensors test covers eager mode.)
        backward_input_shapes = [
            tuple(getattr(node.meta.get("val"), "shape", ()))
            for node in backward.nodes
            if node.op == "placeholder"
        ]
        self.assertNotIn(tuple(self.lhs.shape), backward_input_shapes)


if __name__ == "__main__":
    absltest.main()
