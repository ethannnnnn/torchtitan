# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tokamax GMM-v2 grouped matmul for TPU expert layers, forward and backward.

For expert ``e`` the forward computes ``Y_e = X_e @ W_e`` with weights in the
kernel-native ``(E, K, N)`` layout. The backward computes ``dX_e = dY_e @ W_e^T``
and ``dW_e = X_e^T @ dY_e`` with the same Pallas kernel family (Tokamax
``pallas_mosaic_tpu_v2``), each call using the kernel's built-in VMEM-aware
tiling heuristic, so no per-shape tile tuning is needed. Rows past
``sum(group_sizes)`` are not visited (their outputs are unspecified), which lets
callers pass a fixed-size, partially filled routed-row buffer.

On TPU7x at expert shapes of 4096 x 8192 and 4096 x 4096 with 65,536 routed
rows, this path measured faster than the MegaBlox v1 kernels for all three
GEMMs and does not exceed VMEM at large K/N (where full-axis MegaBlox tiles do).
"""

import jax
import torch
from torch_tpu._internal import pallas


def _ragged_dot(
    lhs, rhs, group_sizes, dimension_numbers, out_dtype, num_actual_groups=None
):
    import tokamax
    from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_v2 as v2

    # The op class is used instead of ``implementation="mosaic_tpu_v2"``
    # because the dWeight (tgmm) pass needs ``num_actual_groups``, which only
    # the op carries. ``Config()`` leaves the tiles unset, so each call uses
    # the kernel's VMEM-aware tiling heuristic.
    kwargs = {}
    if num_actual_groups is not None:
        kwargs["num_actual_groups"] = num_actual_groups
    op = v2.PallasMosaicTpuV2RaggedDot(config=v2.Config(), **kwargs)
    # ragged_dot_general iterates `implementation`; pass the op as a 1-tuple.
    return tokamax.ragged_dot_general(
        lhs,
        rhs,
        group_sizes,
        ragged_dot_dimension_numbers=dimension_numbers,
        preferred_element_type=out_dtype,
        implementation=(op,),
    )


@pallas.jax_op("tt_experts::gmm_v2")
def _gmm_v2_fwd(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array) -> jax.Array:
    from tokamax._src.ops.ragged_dot import base

    return _ragged_dot(
        lhs, rhs, group_sizes, base.DEFAULT_RAGGED_DOT_DIM_NUMS, lhs.dtype
    )


@pallas.jax_op("tt_experts::gmm_v2_bwd")
def _gmm_v2_bwd(
    dout: jax.Array, lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array
) -> tuple[jax.Array, jax.Array]:
    from tokamax._src.ops.ragged_dot import base

    grad_lhs = _ragged_dot(
        dout, rhs, group_sizes, base.TRANS_RHS_RAGGED_DOT_DIM_NUMS, lhs.dtype
    )
    grad_rhs = _ragged_dot(
        lhs,
        dout,
        group_sizes,
        base.RAGGED_CONTRACTING_DOT_DIM_NUMS,
        rhs.dtype,
        num_actual_groups=rhs.shape[0],
    )
    return grad_lhs, grad_rhs.astype(rhs.dtype)


@pallas.jax_op("tt_experts::gmm_v2_dgrad")
def _gmm_v2_dgrad(dout: jax.Array, rhs: jax.Array, group_sizes: jax.Array) -> jax.Array:
    """dInput only, for frozen expert weights (e.g. LoRA): no dWeight pass."""
    from tokamax._src.ops.ragged_dot import base

    return _ragged_dot(
        dout, rhs, group_sizes, base.TRANS_RHS_RAGGED_DOT_DIM_NUMS, dout.dtype
    )


def _weight_needs_grad(ctx) -> bool:
    needs = getattr(ctx, "needs_input_grad", None)
    return True if needs is None else bool(needs[1])


def _setup_context(ctx, inputs, output):
    del output
    lhs, rhs, group_sizes = inputs
    # Frozen weights (e.g. LoRA): dInput needs only the weights, so the LHS
    # rows are not kept alive for the backward.
    ctx.lhs_saved = _weight_needs_grad(ctx)
    if ctx.lhs_saved:
        ctx.save_for_backward(lhs, rhs, group_sizes)
    else:
        ctx.save_for_backward(rhs, group_sizes)


def _backward(ctx, grad_output):
    if not ctx.lhs_saved:
        if _weight_needs_grad(ctx):
            raise RuntimeError("gmm_v2: weight gradient requested but LHS not saved")
        rhs, group_sizes = ctx.saved_tensors
        return _gmm_v2_dgrad(grad_output, rhs, group_sizes), None, None
    lhs, rhs, group_sizes = ctx.saved_tensors
    if not _weight_needs_grad(ctx):
        return _gmm_v2_dgrad(grad_output, rhs, group_sizes), None, None
    grad_lhs, grad_rhs = _gmm_v2_bwd(grad_output, lhs, rhs, group_sizes)
    return grad_lhs, grad_rhs, None


_gmm_v2_fwd.register_autograd(_backward, setup_context=_setup_context)


def gmm_v2_grouped_mm(
    lhs: torch.Tensor, rhs: torch.Tensor, group_sizes: torch.Tensor
) -> torch.Tensor:
    """Per-expert matmul ``lhs[rows of e] @ rhs[e]`` for ``rhs`` in ``(E, K, N)``.

    ``group_sizes`` stays a runtime int32 device tensor (no host sync). The
    output dtype follows ``lhs``; the dWeight is returned in ``rhs.dtype``.
    """
    if group_sizes.dtype != torch.int32:
        group_sizes = group_sizes.to(torch.int32)
    return _gmm_v2_fwd(lhs, rhs, group_sizes)


__all__ = ["gmm_v2_grouped_mm"]
