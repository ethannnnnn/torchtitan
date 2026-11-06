# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Muse Spark Open (text decoder) on TPU.

A TorchTitan port of the text decoder of the public
``meta-models/Muse-Spark-Open-816B-A42B`` checkpoint (Hugging Face
Transformers ``model_type="muse_spark"``). Vision and audio towers are not
built. See ``README.md`` for the architecture, flavors, checkpoint mapping and
validation.

``model_registry(flavor)`` returns a ``ModelSpec`` with:

* ``MuseSparkModel`` configured for the TPU lane: window-aware,
  document-masked SDPA (``SlidingWindowScaledDotProductInnerAttention``,
  picked up by the Splash attention patch) and rotate-half RoPE.
* The TPU MoE converters (``TpuTokenDispatcherConverter`` and
  ``TpuGroupedExpertsConverter``) unless ``tpu_moe_kernels=False``.
* ``parallelize_muse_spark`` (AC, per-block compile, SimpleFSDP / FSDP2).
* ``MuseSparkStateDictAdapter`` for the HF checkpoint.
"""

from __future__ import annotations

from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .configs import build_muse_spark_config
from .configs import MUSE_SPARK_FLAVORS
from .configs import muse_spark_flavor_args
from .configs import MuseSparkArgs
from .infra.parallelize import parallelize_muse_spark
from .model import MuseSparkModel
from .state_dict_adapter import MuseSparkStateDictAdapter

__all__ = [
    "MUSE_SPARK_FLAVORS",
    "MuseSparkArgs",
    "MuseSparkModel",
    "MuseSparkStateDictAdapter",
    "build_muse_spark_config",
    "model_registry",
    "muse_spark_flavor_args",
    "parallelize_muse_spark",
]

# Released checkpoint limit (``max_position_embeddings``).
MAX_CONTEXT_LENGTH = 1_048_576


def tpu_moe_converters() -> list[ModelConfigConverter.Config]:
  """The TPU MoE kernels used by the Qwen3.5 MoE TPU recipes.

  Imported lazily: the kernel modules import ``torch_tpu`` (CPU tests can use
  its import stand-in in ``torchtitan/experiments/tpu/tests/cpu_stubs``).
  """
  # pylint: disable=g-import-not-at-top
  from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExpertsConverter
  from torchtitan.experiments.tpu.token_dispatcher import TpuTokenDispatcherConverter

  return [
      TpuTokenDispatcherConverter.Config(),
      TpuGroupedExpertsConverter.Config(),
  ]


def model_registry(
    flavor: str,
    *,
    seq_len: int | None = None,
    num_layers: int | None = None,
    args: MuseSparkArgs | None = None,
    converters: list[ModelConfigConverter.Config] | None = None,
    tpu_moe_kernels: bool = True,
    load_balance_coeff: float | None = None,
    residual_dtype: str | None = None,
    branch_dtype: str | None = None,
) -> ModelSpec:
  """Returns the TPU ``ModelSpec`` for a Muse Spark flavor.

  Args:
    flavor: A key of ``MUSE_SPARK_FLAVORS`` (e.g. ``816B-A42B``,
      ``816B-A42B-2L``, ``debugmodel``, ``tiny-hf``).
    seq_len: Training context length (sizes the RoPE tables); defaults to the
      flavor's default.
    num_layers: Optional truncation to the first N layers of the flavor.
    args: Optional explicit hyperparameters (e.g. ``MuseSparkArgs.from_hf_config``);
      overrides the flavor's.
    converters: Extra model-config converters, applied after the TPU MoE ones.
    tpu_moe_kernels: Apply the TPU token-dispatcher and grouped-expert
      converters.
    load_balance_coeff: Expert-bias update coefficient; None keeps the
      checkpoint's routing bias fixed.
    residual_dtype: Optional residual-stream dtype (e.g. ``"float32"``).
    branch_dtype: Optional sublayer input dtype (with ``residual_dtype``).

  Returns:
    The model spec.
  """
  if flavor not in MUSE_SPARK_FLAVORS:
    raise ValueError(
        f"Unknown Muse Spark flavor {flavor!r}; available: {sorted(MUSE_SPARK_FLAVORS)}"
    )
  if args is None:
    args = muse_spark_flavor_args(flavor)
  if num_layers is not None:
    if num_layers > args.num_hidden_layers:
      raise ValueError(
          f"num_layers={num_layers} exceeds the flavor's {args.num_hidden_layers}"
      )
    args = MuseSparkArgs(
        **{
            **{f: getattr(args, f) for f in args.__dataclass_fields__},
            "num_hidden_layers": num_layers,
        }
    )
  context_len = seq_len or MUSE_SPARK_FLAVORS[flavor][1]
  if context_len > MAX_CONTEXT_LENGTH:
    raise ValueError(
        f"seq_len {context_len} exceeds the checkpoint limit {MAX_CONTEXT_LENGTH}"
    )
  config = build_muse_spark_config(
      args,
      max_context_length=context_len,
      load_balance_coeff=load_balance_coeff,
      residual_dtype=residual_dtype,
      branch_dtype=branch_dtype,
  )
  all_converters = (tpu_moe_converters() if tpu_moe_kernels else []) + list(
      converters or []
  )
  if all_converters:
    validate_converter_order(all_converters)
    for converter in all_converters:
      config = converter.build().convert(config) or config
  return ModelSpec(
      name="muse_spark",
      flavor=flavor if num_layers is None else f"{flavor}@{num_layers}L",
      model=config,
      max_context_length=context_len,
      parallelize_fn=parallelize_muse_spark,
      pipelining_fn=None,
      post_optimizer_build_fn=None,
      state_dict_adapter=MuseSparkStateDictAdapter,
  )
