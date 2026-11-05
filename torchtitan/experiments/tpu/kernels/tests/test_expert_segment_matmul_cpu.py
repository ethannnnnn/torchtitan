# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks that segment_matmul_pallas_2d computes only dInput for frozen experts.

``segment_matmul_pallas_2d`` multiplies the token rows of each routed expert
(``lhs``, split by ``group_sizes``) with that expert's weight matrix (``rhs``).
When ``rhs`` has ``requires_grad=False``, as for base weights frozen under LoRA,
its backward must compute dInput, the gradient of ``lhs``, and nothing else.
That dInput must match, bit for bit, the dInput that the backward returns for
trainable weights; ``rhs.grad`` must stay ``None``; and the forward must not keep
``lhs`` for the backward. The tests verify this in eager mode and through
AOTAutograd, which ``torch.compile`` uses to trace the backward of a compiled
training step.

Importing ``expert_segment_matmul`` needs TorchTPU, since the kernel module
imports ``torch_tpu._internal.pallas`` at load time. With TorchTPU installed,
run this file as a module (``python -m`` and its dotted name) from the checkout
root. On a host without TorchTPU, run it from the checkout root with two
``PYTHONPATH`` entries: first the directory of the import-only ``torch_tpu``
stand-in that the CPU tests of TPU modules share, then the root itself::

    PYTHONPATH=torchtitan/experiments/tpu/tests/cpu_stubs:. \\
      python -m torchtitan.experiments.tpu.kernels.tests.test_expert_segment_matmul_cpu

Any directory holding the stand-in's ``torch_tpu`` package can take the first
place. Without any importable ``torch_tpu``, the whole test class is skipped,
and the skip reason shows the command above. Other import errors, such as a
missing JAX or a stand-in that refuses to load beside a real TorchTPU, are
reported as errors rather than skips.

Each TorchTPU op of the module (``_grouped_mm_fwd``, ``_grouped_mm_bwd``,
``_grouped_mm_dgrad``) keeps the ``jax.jit`` form of the JAX function it wraps
at ``_init_fn.jit_fn``. Around that function the test defines a separate CPU op
with ``torch.library.custom_op``, whose implementation calls it inside
``jax.experimental.pallas.tpu.force_tpu_interpret_mode()``; the Pallas kernels
are then interpreted on JAX's CPU backend. For the duration of each test,
``unittest.mock`` swaps the three CPU ops in for those module attributes.
``register_autograd`` gives the CPU forward op the module's own ``_backward`` and
``_setup_context``, so ``segment_matmul_pallas_2d``, ``_setup_context`` and
``_backward`` execute as written in the module.

A CPU run cannot check how the kernels lower and compile for a TPU, their speed
or memory use, or the module's own ``register_autograd`` call on its TorchTPU
forward op. The TPU test ``test_expert_segment_matmul`` runs the kernels and
that registration on a TPU.
"""

import collections
import dataclasses
import inspect
import typing
import unittest
from unittest import mock
import uuid

from absl.testing import absltest
from absl.testing import parameterized
from functorch.compile import make_boxed_func
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import torch
from torch._dynamo.backends.common import aot_autograd

try:
    from torchtitan.experiments.tpu.kernels import expert_segment_matmul as esm
except ModuleNotFoundError as missing:
    # Only a missing TorchTPU skips the tests; any other import error fails.
    if missing.name != "torch_tpu":
        raise
    esm = None

_NO_TORCH_TPU = (
    "torch_tpu cannot be imported: install TorchTPU, or run from the repository"
    " root with the CPU stand-in first on PYTHONPATH: PYTHONPATH="
    "torchtitan/experiments/tpu/tests/cpu_stubs:. python -m"
    " torchtitan.experiments.tpu.kernels.tests.test_expert_segment_matmul_cpu"
)

# Module attribute of each TorchTPU op -> the op's short name.
_OP_SHORT_NAMES = {
    "_grouped_mm_fwd": "grouped_mm",
    "_grouped_mm_bwd": "grouped_mm_bwd",
    "_grouped_mm_dgrad": "grouped_mm_dgrad",
}

# Three experts: the middle one receives no rows, and the boundary at row 96
# falls inside the first M tile.
_ROWS_PER_EXPERT = (96, 0, 160)
_TILE_M = 128
# Against the FP32 reference: about one bfloat16 rounding step at magnitude 1.
_REF_RTOL = 8e-3
_REF_ATOL = 1e-3


def _as_jax_array(tensor: torch.Tensor) -> jax.Array:
    """A JAX array with the tensor's values, shape and dtype, via DLPack."""
    return jnp.from_dlpack(tensor.detach().contiguous())


def _as_cpu_tensor(array: jax.Array) -> torch.Tensor:
    """A CPU tensor holding its own copy of a JAX result.

    Autograd may update a returned tensor in place; the copy keeps that away
    from the buffer JAX handed over.
    """
    return torch.from_dlpack(array).clone()


def _jax_dtype(dtype: torch.dtype) -> jnp.dtype:
    """The JAX dtype of the same name, e.g. bfloat16 for ``torch.bfloat16``."""
    return jnp.dtype(getattr(jnp, str(dtype).removeprefix("torch.")))


def _torch_dtype(dtype: jnp.dtype) -> torch.dtype:
    """The torch dtype of the same name as a JAX dtype."""
    return getattr(torch, jnp.dtype(dtype).name)


def _schema_type(annotation: object) -> str:
    """Op schema type for an annotation of a kernel's JAX function."""
    if annotation is jax.Array:
        return "Tensor"
    if annotation in (int, bool):
        return annotation.__name__
    raise TypeError(f"no op schema type for the annotation {annotation!r}")


def _schema_of(fn) -> str:
    """An op schema with ``fn``'s parameters, in order, and its results."""
    signature = inspect.signature(inspect.unwrap(fn), eval_str=True)
    params = ", ".join(
        f"{_schema_type(param.annotation)} {param.name}"
        for param in signature.parameters.values()
    )
    result = signature.return_annotation
    if typing.get_origin(result) is tuple:
        returns = "(" + ", ".join(map(_schema_type, typing.get_args(result))) + ")"
    else:
        returns = _schema_type(result)
    return f"({params}) -> {returns}"


def _jit_fn(module, attr: str):
    """The jitted JAX function that one of the module's TorchTPU ops wraps."""
    op = getattr(module, attr)
    try:
        return op._init_fn.jit_fn
    except AttributeError as error:
        raise AssertionError(
            f"op tt_experts::{_OP_SHORT_NAMES[attr]} ({module.__name__}.{attr})"
            " has no _init_fn.jit_fn; the TorchTPU op contract that this test"
            " relies on has changed"
        ) from error


class _CpuKernelOps:
    """CPU ``torch.library`` ops that stand in for the module's TorchTPU ops.

    Each op runs the jitted JAX function of its TorchTPU op on CPU tensors with
    Pallas TPU interpret mode on, and counts its runs under its short name. Its
    fake implementation only traces the output shapes and dtypes.
    """

    def __init__(self, module):
        # A namespace of its own per instance: setting the test class up again,
        # or importing this file under a second module name, defines new ops
        # instead of redefining existing ones.
        self.namespace = f"esm_cpu_test_{uuid.uuid4().hex[:12]}"
        self.calls = collections.Counter()
        self.ops = {
            attr: self._define(short_name, _jit_fn(module, attr))
            for attr, short_name in _OP_SHORT_NAMES.items()
        }
        self.ops["_grouped_mm_fwd"].register_autograd(
            module._backward, setup_context=module._setup_context
        )

    def overload(self, short_name: str):
        """The op's default overload, the form that traced graphs call."""
        return getattr(getattr(torch.ops, self.namespace), short_name).default

    def _define(self, short_name: str, jit_fn):
        def run(*args):
            inputs = [
                _as_jax_array(arg) if isinstance(arg, torch.Tensor) else arg
                for arg in args
            ]
            # The forced mode also reaches pallas_calls made with
            # interpret=False, such as the module's dWeight kernel.
            with pltpu.force_tpu_interpret_mode():
                result = jax.block_until_ready(jit_fn(*inputs))
            outputs = jax.tree.map(_as_cpu_tensor, result)
            self.calls[short_name] += 1
            return outputs

        def trace_outputs(*args):
            device = next(arg.device for arg in args if isinstance(arg, torch.Tensor))
            specs = [
                (
                    jax.ShapeDtypeStruct(tuple(arg.shape), _jax_dtype(arg.dtype))
                    if isinstance(arg, torch.Tensor)
                    else arg
                )
                for arg in args
            ]
            return jax.tree.map(
                lambda out: torch.empty(
                    out.shape, dtype=_torch_dtype(out.dtype), device=device
                ),
                jax.eval_shape(jit_fn, *specs),
            )

        op = torch.library.custom_op(
            f"{self.namespace}::{short_name}",
            run,
            mutates_args=(),
            schema=_schema_of(jit_fn),
        )
        op.register_fake(trace_outputs)
        return op


@dataclasses.dataclass(frozen=True)
class _Case:
    """Inputs of one grouped matmul; ``weights`` is in the kernel layout."""

    lhs: torch.Tensor  # (rows, K)
    weights: torch.Tensor  # (E, K, N)
    dout: torch.Tensor  # (rows, N)
    group_sizes: torch.Tensor  # (E,), int32
    tile_k: int
    tile_n: int

    def weights_for(self, transpose_rhs: bool) -> torch.Tensor:
        """The weights in the layout that ``transpose_rhs`` selects."""
        if transpose_rhs:
            return self.weights.transpose(-2, -1).contiguous()
        return self.weights

    def fp32_reference(self):
        """Output, dInput and dWeight (kernel layout), per expert in FP32."""
        sizes = list(_ROWS_PER_EXPERT)
        lhs_rows = self.lhs.float().split(sizes)
        dout_rows = self.dout.float().split(sizes)
        weights = self.weights.float()
        out = torch.cat([x @ w for x, w in zip(lhs_rows, weights)])
        dinput = torch.cat([dy @ w.T for dy, w in zip(dout_rows, weights)])
        dweight = torch.stack([x.T @ dy for x, dy in zip(lhs_rows, dout_rows)])
        return out, dinput, dweight


def _make_case(seed: int, k: int, n: int, tile_k: int = 0, tile_n: int = 0):
    generator = torch.Generator().manual_seed(seed)
    rows = sum(_ROWS_PER_EXPERT)

    def normal(*shape):
        return torch.randn(shape, generator=generator).to(torch.bfloat16)

    return _Case(
        lhs=normal(rows, k),
        weights=normal(len(_ROWS_PER_EXPERT), k, n),
        dout=normal(rows, n),
        group_sizes=torch.tensor(_ROWS_PER_EXPERT, dtype=torch.int32),
        tile_k=tile_k,
        tile_n=tile_n,
    )


class _Outcome(typing.NamedTuple):
    out: torch.Tensor
    dinput: torch.Tensor
    dweight: torch.Tensor | None
    calls: dict[str, int]


@unittest.skipIf(esm is None, _NO_TORCH_TPU)
class FrozenExpertBackwardCpuTest(parameterized.TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tpu_ops = {attr: getattr(esm, attr) for attr in _OP_SHORT_NAMES}
        cls.cpu = _CpuKernelOps(esm)
        cls.whole_tiles = _make_case(seed=7, k=128, n=256)
        cls.split_tiles = _make_case(seed=8, k=256, n=512, tile_k=128, tile_n=256)

    def setUp(self):
        super().setUp()
        torch.compiler.reset()
        self._expect_tpu_ops_bound()
        # Added before the binding's own cleanup, so it runs after the binding
        # has been undone.
        self.addCleanup(self._expect_tpu_ops_bound)
        binding = mock.patch.multiple(esm, **self.cpu.ops)
        binding.start()
        self.addCleanup(binding.stop)
        self.cpu.calls.clear()

    def _expect_tpu_ops_bound(self):
        for attr, tpu_op in self.tpu_ops.items():
            self.assertIs(getattr(esm, attr), tpu_op, f"{attr} is not restored")

    def _train_step(self, case, *, frozen, transpose_rhs):
        """One forward and backward; the results and the op runs they took."""
        lhs = case.lhs.clone().requires_grad_()
        weights = case.weights_for(transpose_rhs).clone().requires_grad_(not frozen)
        self.cpu.calls.clear()
        out = esm.segment_matmul_pallas_2d(
            lhs,
            weights,
            case.group_sizes,
            tile_m=_TILE_M,
            tile_k=case.tile_k,
            tile_n=case.tile_n,
            transpose_rhs=transpose_rhs,
        )
        out.backward(case.dout)
        return _Outcome(out.detach(), lhs.grad, weights.grad, dict(self.cpu.calls))

    def _check_frozen_against_trainable(self, case, transpose_rhs):
        trainable = self._train_step(case, frozen=False, transpose_rhs=transpose_rhs)
        self.assertEqual(trainable.calls, {"grouped_mm": 1, "grouped_mm_bwd": 1})
        self.assertIsNotNone(trainable.dweight)
        frozen = self._train_step(case, frozen=True, transpose_rhs=transpose_rhs)
        self.assertEqual(frozen.calls, {"grouped_mm": 1, "grouped_mm_dgrad": 1})
        self.assertIsNone(frozen.dweight)
        torch.testing.assert_close(frozen.out, trainable.out, rtol=0, atol=0)
        torch.testing.assert_close(frozen.dinput, trainable.dinput, rtol=0, atol=0)

        want_out, want_dinput, want_dweight = case.fp32_reference()
        dweight = trainable.dweight
        if transpose_rhs:
            dweight = dweight.transpose(-2, -1)
        for name, got, want in (
            ("output", frozen.out, want_out),
            ("frozen dInput", frozen.dinput, want_dinput),
            ("trainable dWeight", dweight, want_dweight),
        ):
            torch.testing.assert_close(
                got.float(),
                want,
                rtol=_REF_RTOL,
                atol=_REF_ATOL,
                msg=lambda detail, name=name: f"{name} vs FP32: {detail}",
            )

    @parameterized.named_parameters(
        ("kernel_layout", False), ("transposed_layout", True)
    )
    def test_frozen_weights_run_dgrad_and_keep_the_dinput(self, transpose_rhs):
        self._check_frozen_against_trainable(self.whole_tiles, transpose_rhs)

    @parameterized.named_parameters(
        ("kernel_layout", False), ("transposed_layout", True)
    )
    def test_dgrad_with_unequal_k_and_n_tiles(self, transpose_rhs):
        # Zero K/N tiles make each dimension a single tile, which would hide a
        # dInput GMM that mixes up the two tile sizes.
        self._check_frozen_against_trainable(self.split_tiles, transpose_rhs)

    def _shapes_saved_by_forward(self, case, *, frozen):
        stash = []

        def pack(tensor):
            stash.append(tensor)
            return len(stash) - 1

        lhs = case.lhs.clone().requires_grad_()
        weights = case.weights.clone().requires_grad_(not frozen)
        with torch.autograd.graph.saved_tensors_hooks(pack, stash.__getitem__):
            esm.segment_matmul_pallas_2d(lhs, weights, case.group_sizes, tile_m=_TILE_M)
        return [tuple(tensor.shape) for tensor in stash]

    def test_forward_keeps_the_lhs_only_for_trainable_weights(self):
        case = self.whole_tiles
        lhs, weights, group_sizes = (
            tuple(tensor.shape) for tensor in (case.lhs, case.weights, case.group_sizes)
        )
        self.assertEqual(
            self._shapes_saved_by_forward(case, frozen=False),
            [lhs, weights, group_sizes],
        )
        self.assertEqual(
            self._shapes_saved_by_forward(case, frozen=True), [weights, group_sizes]
        )

    def test_aot_autograd_takes_the_dgrad_branch(self):
        case = self.whole_tiles
        eager = self._train_step(case, frozen=True, transpose_rhs=False)

        graphs = {}

        def keep_graph(kind):
            def compile_graph(graph_module, example_inputs):
                del example_inputs
                graphs[kind] = graph_module
                return make_boxed_func(graph_module.forward)

            return compile_graph

        compiled = torch.compile(
            esm.segment_matmul_pallas_2d,
            backend=aot_autograd(
                fw_compiler=keep_graph("forward"), bw_compiler=keep_graph("backward")
            ),
            fullgraph=True,
        )
        lhs = case.lhs.clone().requires_grad_()
        weights = case.weights.clone()
        self.cpu.calls.clear()
        compiled(lhs, weights, case.group_sizes, tile_m=_TILE_M).backward(case.dout)

        self.assertEqual(dict(self.cpu.calls), {"grouped_mm": 1, "grouped_mm_dgrad": 1})
        self.assertIsNone(weights.grad)
        torch.testing.assert_close(lhs.grad, eager.dinput, rtol=0, atol=0)

        backward = graphs["backward"].graph
        called = [node.target for node in backward.nodes if node.op == "call_function"]
        self.assertIn(self.cpu.overload("grouped_mm_dgrad"), called)
        self.assertNotIn(self.cpu.overload("grouped_mm_bwd"), called)
        inputs = [
            node.meta.get("val") for node in backward.find_nodes(op="placeholder")
        ]
        input_shapes = [
            tuple(value.shape) for value in inputs if isinstance(value, torch.Tensor)
        ]
        self.assertTrue(input_shapes)
        self.assertNotIn(tuple(lhs.shape), input_shapes)

    def test_cpu_ops_are_bound_only_while_a_test_runs(self):
        # setUp saw the TorchTPU ops before binding the CPU ops, and a cleanup
        # looks again once the binding is undone.
        for attr, cpu_op in self.cpu.ops.items():
            self.assertIs(getattr(esm, attr), cpu_op)
            self.assertIsNot(cpu_op, self.tpu_ops[attr])


if __name__ == "__main__":
    absltest.main()
