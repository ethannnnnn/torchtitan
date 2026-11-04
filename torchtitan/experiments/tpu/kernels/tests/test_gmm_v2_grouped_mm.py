# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the Tokamax GMM-v2 expert grouped matmul.

The kernel tests need a TPU (``--device=tpu``) and skip on other devices. The
autograd bookkeeping tests replace the kernels with stand-ins and run anywhere.
"""

from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest
import torch

from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu.kernels import gmm_v2_grouped_mm


def _reference(lhs, rhs, sizes, grad_output):
    """FP32 per-group matmul and its gradients, over the visited rows only."""
    lhs = lhs.float().requires_grad_()
    rhs = rhs.float().requires_grad_()
    outs, start = [], 0
    for e, n in enumerate(sizes):
        outs.append(lhs[start : start + n] @ rhs[e])
        start += n
    out = torch.cat(outs)
    out.backward(grad_output[:start].float())
    return out, lhs.grad[:start], rhs.grad


def _inputs(sizes, rows, k, n):
    generator = torch.Generator().manual_seed(0)
    lhs = torch.randn(rows, k, generator=generator).bfloat16()
    rhs = (torch.randn(len(sizes), k, n, generator=generator) * 0.05).bfloat16()
    grad_output = torch.randn(rows, n, generator=generator).bfloat16()
    return lhs, rhs, grad_output


def _skip_unless_tpu(test):
    if test.accelerator_device_type != device_type.AcceleratorDeviceType.TPU:
        test.skipTest("Tokamax GMM-v2 is a TPU-only Pallas kernel.")


class GmmV2GroupedMmTest(base_device_test.BaseAcceleratorDeviceTest):
    def setUp(self):
        super().setUp()
        _skip_unless_tpu(self)

    def _check(self, sizes, rows, k, n):
        device = self.accelerator_device
        lhs, rhs, grad_output = _inputs(sizes, rows, k, n)
        want, want_dlhs, want_drhs = _reference(lhs, rhs, sizes, grad_output)

        lhs_d = lhs.to(device).requires_grad_()
        rhs_d = rhs.to(device).requires_grad_()
        group_sizes = torch.tensor(sizes, dtype=torch.int32, device=device)
        out = gmm_v2_grouped_mm.gmm_v2_grouped_mm(lhs_d, rhs_d, group_sizes)
        out.backward(grad_output.to(device))
        visited = sum(sizes)

        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(rhs_d.grad.dtype, torch.bfloat16)
        torch.testing.assert_close(
            out[:visited].float().cpu(), want, rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(
            lhs_d.grad[:visited].float().cpu(), want_dlhs, rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(
            rhs_d.grad.float().cpu(), want_drhs, rtol=2e-2, atol=5e-2
        )

    def test_uneven_groups_with_empty_group(self):
        self._check([0, 700, 1100, 300, 1, 1000, 500, 495], 4096, 512, 1024)

    def test_partially_filled_row_buffer(self):
        # Rows past sum(group_sizes) (a fixed-size dispatch buffer) are not visited.
        self._check([1000, 0, 1500, 600], 4096, 512, 512)

    def test_frozen_weights_skip_the_dweight_pass(self):
        # Frozen weights (e.g. LoRA) take the dInput-only backward. It makes the
        # same dInput kernel call as the full backward, so dInput is identical.
        device = self.accelerator_device
        sizes = [1000, 0, 1500, 600]
        lhs, rhs, grad_output = _inputs(sizes, 4096, 512, 512)
        group_sizes = torch.tensor(sizes, dtype=torch.int32, device=device)
        visited = sum(sizes)

        lhs_full = lhs.to(device, copy=True).requires_grad_()
        rhs_full = rhs.to(device, copy=True).requires_grad_()
        gmm_v2_grouped_mm.gmm_v2_grouped_mm(lhs_full, rhs_full, group_sizes).backward(
            grad_output.to(device)
        )
        lhs_frozen = lhs.to(device, copy=True).requires_grad_()
        rhs_frozen = rhs.to(device, copy=True)
        gmm_v2_grouped_mm.gmm_v2_grouped_mm(
            lhs_frozen, rhs_frozen, group_sizes
        ).backward(grad_output.to(device))

        self.assertIsNone(rhs_frozen.grad)
        torch.testing.assert_close(
            lhs_frozen.grad[:visited].cpu(),
            lhs_full.grad[:visited].cpu(),
            rtol=0,
            atol=0,
        )


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
