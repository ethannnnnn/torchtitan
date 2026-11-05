# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the Tokamax GMM-v2 expert grouped matmul.

``test_gmm_v2_grouped_mm`` checks the kernel numerics on a TPU. These tests
need no TPU, so CI runs them in the CPU shards. They check what the module asks
Tokamax for and the autograd bookkeeping around the kernel calls:

* each ``tokamax.ragged_dot_general`` call names one GMM-v2 op with heuristic
  tiling, the dimension numbers and output dtype of its GEMM, and
  ``num_actual_groups`` on the dWeight call only;
* the three JAX functions trace through the real Tokamax kernels for an
  abstract TPU7x device, with the expected output shapes and dtypes;
* the backward keeps the LHS only for trainable weights, and frozen weights
  take the dInput-only kernel.
"""

import inspect
from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import tokamax
from tokamax._src.ops.ragged_dot import base
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_v2 as v2
import torch

from torchtitan.experiments.tpu.kernels import gmm_v2_grouped_mm

# An abstract TPU7x device: Tokamax picks its TPU7x tiles and the kernels trace
# on a CPU host. Nothing is lowered or run.
_TPU7X = jax.sharding.AbstractMesh(
    (),
    (),
    abstract_device=jax.sharding.AbstractDevice(
        device_kind="TPU7x", num_cores=1, platform="tpu"
    ),
)


def _jax_function(op):
    """The jitted JAX function behind a ``jax_op``, as torch_tpu keeps it."""
    return op._init_fn.jit_fn


class RaggedDotCallTest(absltest.TestCase):
    """What each GEMM passes to ``tokamax.ragged_dot_general``.

    The JAX functions run eagerly with the Tokamax dispatch replaced by XLA's
    ragged dot, which runs on CPU, so both the recorded arguments and the math
    they select are checked. Activations are BF16 and weights FP32, so each
    output dtype shows which operand it follows.
    """

    def setUp(self):
        super().setUp()
        rng = np.random.default_rng(0)
        # An empty group, and two rows past sum(sizes): a partially filled buffer.
        self.sizes = np.array([5, 0, 7, 4], np.int32)
        rows, k, n = int(self.sizes.sum()) + 2, 8, 12
        self.lhs = jnp.asarray(rng.standard_normal((rows, k)), jnp.bfloat16)
        self.rhs = jnp.asarray(
            rng.standard_normal((len(self.sizes), k, n)), jnp.float32
        )
        self.dout = jnp.asarray(rng.standard_normal((rows, n)), jnp.bfloat16)
        self.group_sizes = jnp.asarray(self.sizes)
        self.calls = []

    def _ragged_dot_general(
        self,
        lhs,
        rhs,
        group_sizes,
        *,
        ragged_dot_dimension_numbers,
        preferred_element_type,
        implementation,
    ):
        self.calls.append(
            SimpleNamespace(
                implementation=implementation,
                dimension_numbers=ragged_dot_dimension_numbers,
                out_dtype=preferred_element_type,
                group_sizes=group_sizes,
            )
        )
        out = jax.lax.ragged_dot_general(
            lhs.astype(jnp.float32),
            rhs.astype(jnp.float32),
            group_sizes,
            ragged_dot_dimension_numbers,
            preferred_element_type=jnp.float32,
        )
        return out.astype(preferred_element_type)

    def _run(self, op, *args):
        # Eagerly: a cached trace of the jitted function would skip the patch.
        fn = inspect.unwrap(_jax_function(op))
        with mock.patch.object(tokamax, "ragged_dot_general", self._ragged_dot_general):
            return fn(*args)

    def _assert_call(self, call, dimension_numbers, out_dtype, num_actual_groups=None):
        # A single op, so Tokamax has no other implementation to fall back to.
        self.assertLen(call.implementation, 1)
        op = call.implementation[0]
        self.assertIsInstance(op, v2.PallasMosaicTpuV2RaggedDot)
        # No tile sizes: the kernel's VMEM-aware heuristic picks them per call.
        self.assertEqual(op.config, v2.Config())
        self.assertEqual(op.num_actual_groups, num_actual_groups)
        self.assertEqual(call.dimension_numbers, dimension_numbers)
        self.assertEqual(call.out_dtype, out_dtype)
        np.testing.assert_array_equal(call.group_sizes, self.sizes)

    def _reference(self):
        """FP32 per-expert output and dInput (visited rows) and dWeight."""
        x = np.asarray(self.lhs, np.float32)
        w = np.asarray(self.rhs, np.float32)
        dy = np.asarray(self.dout, np.float32)
        out, dlhs, drhs, start = [], [], [], 0
        for e, size in enumerate(self.sizes):
            rows = slice(start, start + size)
            out.append(x[rows] @ w[e])
            dlhs.append(dy[rows] @ w[e].T)
            drhs.append(x[rows].T @ dy[rows])
            start += size
        return np.concatenate(out), np.concatenate(dlhs), np.stack(drhs)

    def _assert_visited_rows_close(self, actual, expected):
        visited = int(self.sizes.sum())
        np.testing.assert_allclose(
            np.asarray(actual[:visited], np.float32), expected, rtol=1e-2, atol=1e-2
        )

    def test_forward(self):
        out = self._run(
            gmm_v2_grouped_mm._gmm_v2_fwd, self.lhs, self.rhs, self.group_sizes
        )

        self.assertLen(self.calls, 1)
        self._assert_call(self.calls[0], base.DEFAULT_RAGGED_DOT_DIM_NUMS, jnp.bfloat16)
        self.assertEqual(out.dtype, jnp.bfloat16)
        want_out, _, _ = self._reference()
        self._assert_visited_rows_close(out, want_out)

    def test_backward(self):
        dlhs, drhs = self._run(
            gmm_v2_grouped_mm._gmm_v2_bwd,
            self.dout,
            self.lhs,
            self.rhs,
            self.group_sizes,
        )

        self.assertLen(self.calls, 2)
        dinput_call, dweight_call = self.calls
        self._assert_call(dinput_call, base.TRANS_RHS_RAGGED_DOT_DIM_NUMS, jnp.bfloat16)
        # Tokamax's dWeight (tgmm) pass needs the local expert count.
        self._assert_call(
            dweight_call,
            base.RAGGED_CONTRACTING_DOT_DIM_NUMS,
            jnp.float32,
            num_actual_groups=len(self.sizes),
        )
        self.assertEqual(dlhs.dtype, jnp.bfloat16)
        self.assertEqual(drhs.dtype, jnp.float32)
        _, want_dlhs, want_drhs = self._reference()
        self._assert_visited_rows_close(dlhs, want_dlhs)
        np.testing.assert_allclose(drhs, want_drhs, rtol=1e-5, atol=1e-5)

    def test_dinput_only_backward(self):
        dlhs = self._run(
            gmm_v2_grouped_mm._gmm_v2_dgrad, self.dout, self.rhs, self.group_sizes
        )

        self.assertLen(self.calls, 1)
        self._assert_call(
            self.calls[0], base.TRANS_RHS_RAGGED_DOT_DIM_NUMS, jnp.bfloat16
        )
        self.assertEqual(dlhs.dtype, jnp.bfloat16)
        _, want_dlhs, _ = self._reference()
        self._assert_visited_rows_close(dlhs, want_dlhs)


class Tpu7xTraceTest(parameterized.TestCase):
    """The real Tokamax kernels trace for TPU7x on a CPU host."""

    @parameterized.parameters(jnp.bfloat16, jnp.float32)
    def test_output_shapes_and_dtypes(self, dtype):
        # K != N, so a call that contracts the wrong axis fails to trace.
        rows, k, n, experts = 512, 256, 384, 4
        lhs = jax.ShapeDtypeStruct((rows, k), dtype)
        rhs = jax.ShapeDtypeStruct((experts, k, n), dtype)
        dout = jax.ShapeDtypeStruct((rows, n), dtype)
        group_sizes = jax.ShapeDtypeStruct((experts,), jnp.int32)
        with jax.sharding.use_abstract_mesh(_TPU7X):
            out = jax.eval_shape(
                _jax_function(gmm_v2_grouped_mm._gmm_v2_fwd), lhs, rhs, group_sizes
            )
            # Tokamax rejects a dWeight call without num_actual_groups.
            dlhs, drhs = jax.eval_shape(
                _jax_function(gmm_v2_grouped_mm._gmm_v2_bwd),
                dout,
                lhs,
                rhs,
                group_sizes,
            )
            dgrad = jax.eval_shape(
                _jax_function(gmm_v2_grouped_mm._gmm_v2_dgrad), dout, rhs, group_sizes
            )

        self.assertEqual((out.shape, out.dtype), ((rows, n), dtype))
        self.assertEqual((dlhs.shape, dlhs.dtype), ((rows, k), dtype))
        self.assertEqual((drhs.shape, drhs.dtype), ((experts, k, n), dtype))
        self.assertEqual((dgrad.shape, dgrad.dtype), ((rows, k), dtype))


class GmmV2AutogradBookkeepingTest(absltest.TestCase):
    """What the backward keeps alive and which kernels it calls."""

    def setUp(self):
        super().setUp()
        self.lhs = torch.ones(4, 2)
        self.rhs = torch.ones(1, 2, 3)
        self.group_sizes = torch.tensor([4], dtype=torch.int32)
        self.grad_output = torch.ones(4, 3)

    def _setup(self, needs_input_grad):
        ctx = SimpleNamespace()
        if needs_input_grad is not None:
            ctx.needs_input_grad = needs_input_grad

        def save_for_backward(*tensors):
            ctx.saved_tensors = tensors

        ctx.save_for_backward = save_for_backward
        gmm_v2_grouped_mm._setup_context(
            ctx, (self.lhs, self.rhs, self.group_sizes), None
        )
        return ctx

    def _backward(self, ctx):
        with mock.patch.object(
            gmm_v2_grouped_mm, "_gmm_v2_dgrad", return_value="dx"
        ) as dgrad, mock.patch.object(
            gmm_v2_grouped_mm, "_gmm_v2_bwd", return_value=("dx", "dw")
        ) as bwd:
            grads = gmm_v2_grouped_mm._backward(ctx, self.grad_output)
        return grads, dgrad, bwd

    def test_frozen_weights_keep_only_the_weights(self):
        ctx = self._setup((True, False, False))
        self.assertFalse(ctx.lhs_saved)
        self.assertEqual(len(ctx.saved_tensors), 2)
        self.assertIs(ctx.saved_tensors[0], self.rhs)
        grads, dgrad, bwd = self._backward(ctx)
        self.assertEqual(grads, ("dx", None, None))
        dgrad.assert_called_once_with(self.grad_output, self.rhs, self.group_sizes)
        bwd.assert_not_called()

    def test_trainable_weights_run_the_full_backward(self):
        ctx = self._setup((True, True, False))
        self.assertTrue(ctx.lhs_saved)
        self.assertEqual(len(ctx.saved_tensors), 3)
        grads, dgrad, bwd = self._backward(ctx)
        self.assertEqual(grads, ("dx", "dw", None))
        bwd.assert_called_once_with(
            self.grad_output, self.lhs, self.rhs, self.group_sizes
        )
        dgrad.assert_not_called()

    def test_unknown_requires_grad_keeps_the_lhs(self):
        ctx = self._setup(None)
        self.assertTrue(ctx.lhs_saved)
        grads, _, bwd = self._backward(ctx)
        self.assertEqual(grads, ("dx", "dw", None))
        bwd.assert_called_once()

    def test_weight_grad_without_saved_lhs_raises(self):
        ctx = self._setup((True, False, False))
        ctx.needs_input_grad = (True, True, False)
        with self.assertRaisesRegex(RuntimeError, "LHS not saved"):
            gmm_v2_grouped_mm._backward(ctx, self.grad_output)

    def test_group_sizes_reach_the_kernel_as_int32(self):
        with mock.patch.object(
            gmm_v2_grouped_mm, "_gmm_v2_fwd", return_value="out"
        ) as fwd:
            out = gmm_v2_grouped_mm.gmm_v2_grouped_mm(
                self.lhs, self.rhs, self.group_sizes.long()
            )
        self.assertEqual(out, "out")
        self.assertEqual(fwd.call_args.args[2].dtype, torch.int32)


if __name__ == "__main__":
    absltest.main()
