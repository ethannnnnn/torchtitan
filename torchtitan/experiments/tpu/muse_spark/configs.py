# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Hyperparameter names and defaults mirror the public Hugging Face
# ``MuseSparkTextConfig`` (Apache License 2.0); initialization follows the
# BSD-3 upstream TorchTitan Muse Spark port.

"""Muse Spark Open text-model configs: hyperparameters, builders and flavors.

``MuseSparkArgs`` holds the text hyperparameters under the Hugging Face
``text_config`` names (``config.json`` of the public checkpoint), so any
checkpoint's config can be loaded with ``MuseSparkArgs.from_hf_config``.
``build_muse_spark_config`` turns the arguments into a ``MuseSparkModel.Config``
for the TPU lane.

Flavors (``MUSE_SPARK_FLAVORS``):

* ``816B-A42B``: the released text decoder (62 layers, 256 experts, top-8).
* ``816B-A42B-{2,4,8}L``: its first 2 / 4 / 8 layers, for truncated-model
  tests that load the real checkpoint (the adapter skips deeper layers).
* ``debugmodel``: 4 layers, 8 experts, real vocabulary (random init).
* ``tiny-hf``: the public ``trl-internal-testing`` tiny test checkpoint.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

import torch.nn as nn

from torchtitan.experiments.tpu.muse_glimmer.attention import SlidingWindowScaledDotProductInnerAttention
from torchtitan.models.common import CosSinRoPE
from torchtitan.models.common.attention import QKVLinear
from torchtitan.models.common.config_utils import make_token_dispatcher_config
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.linear import RouterGateLinear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.param_init import depth_scaled_std

from .model import MuseSparkAttention
from .model import MuseSparkEmbedding
from .model import MuseSparkGroupedExperts
from .model import MuseSparkInputScaledRMSNorm
from .model import MuseSparkLMHead
from .model import MuseSparkModel
from .model import MuseSparkMoE
from .model import MuseSparkResidualGate
from .model import MuseSparkRMSNorm
from .model import MuseSparkRoutedExperts
from .model import MuseSparkRouter
from .model import MuseSparkTransformerBlock

__all__ = [
    "MUSE_SPARK_FLAVORS",
    "MuseSparkArgs",
    "build_muse_spark_config",
    "muse_spark_flavor_args",
]

FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"


@dataclasses.dataclass(kw_only=True)
class MuseSparkArgs:
  """Text-decoder hyperparameters, named as in the HF ``text_config``."""

  vocab_size: int = 202048
  hidden_size: int = 8192
  intermediate_size: int = 16384  # dense-FFN layers only
  num_hidden_layers: int = 62
  num_attention_heads: int = 128
  num_key_value_heads: int = 16
  head_dim: int = 64
  layer_types: list[str] = dataclasses.field(default_factory=list)
  layer_rope_theta: list[float] = dataclasses.field(default_factory=list)
  rope_theta: float = 500_000.0
  sliding_window: int = 2048
  mlp_layer_types: list[str] = dataclasses.field(default_factory=list)
  moe_hidden_size: int = 4096
  moe_intermediate_size: int = 4096
  num_local_experts: int = 256
  num_experts_per_tok: int = 8
  num_shared_experts: int = 0
  routed_scaling_factor: float = 1.0
  qk_scale_factor: float = 5.4730064863875
  output_multiplier: float = 1.0 / math.sqrt(32.0)
  final_logit_softcapping: float = 20.0
  rms_norm_eps: float = 1e-5
  post_norm_eps: float = 1e-8
  residual_gate_temperature: float = 0.3
  tie_word_embeddings: bool = False
  attention_bias: bool = False
  hidden_act: str = "silu"

  def __post_init__(self) -> None:
    n = self.num_hidden_layers
    if not self.layer_types:
      # Released pattern: layer i is global iff i % 4 == 1.
      self.layer_types = [
          FULL_ATTENTION if i % 4 == 1 else SLIDING_ATTENTION for i in range(n)
      ]
    if not self.layer_rope_theta:
      self.layer_rope_theta = [
          0.0 if t == FULL_ATTENTION else self.rope_theta for t in self.layer_types
      ]
    if not self.mlp_layer_types:
      self.mlp_layer_types = ["sparse"] * n
    # HF configs may list more entries than layers (truncated checkpoints).
    self.layer_types = list(self.layer_types[:n])
    self.layer_rope_theta = [float(t) for t in self.layer_rope_theta[:n]]
    self.mlp_layer_types = list(self.mlp_layer_types[:n])
    for name in ("layer_types", "layer_rope_theta", "mlp_layer_types"):
      if len(getattr(self, name)) != n:
        raise ValueError(f"{name} needs {n} entries, got {getattr(self, name)!r}")
    if bad := set(self.layer_types) - {FULL_ATTENTION, SLIDING_ATTENTION}:
      raise ValueError(f"unsupported layer_types {sorted(bad)}")
    if bad := set(self.mlp_layer_types) - {"sparse", "dense"}:
      raise ValueError(f"unsupported mlp_layer_types {sorted(bad)}")
    # The reference model computes one rotary table from the global theta and
    # uses the per-layer value only to switch RoPE off (theta == 0).
    if bad := {t for t in self.layer_rope_theta if t and t != self.rope_theta}:
      raise ValueError(
          f"per-layer rope theta {sorted(bad)} differs from rope_theta"
          f" {self.rope_theta}; only 0 (NoPE) or the global theta is supported"
      )
    if self.num_shared_experts:
      raise ValueError("Muse Spark Open has no shared experts")
    if self.tie_word_embeddings:
      raise ValueError("Muse Spark Open does not tie embeddings")
    if self.attention_bias:
      raise ValueError("Muse Spark Open attention has no biases")
    if self.hidden_act != "silu":
      raise ValueError(f"unsupported hidden_act {self.hidden_act!r}")

  @classmethod
  def from_hf_config(
      cls, config: Mapping[str, Any] | str, *, num_hidden_layers: int | None = None
  ) -> "MuseSparkArgs":
    """Reads the text hyperparameters of an HF ``config.json``.

    Args:
      config: The parsed config (top level or ``text_config``), or a path to
        ``config.json`` or to the directory holding it.
      num_hidden_layers: Optional truncation to the first N layers.

    Returns:
      The arguments, with per-layer lists cut to the layer count.
    """
    if isinstance(config, str):
      path = os.path.join(config, "config.json") if os.path.isdir(config) else config
      with open(path) as f:
        config = json.load(f)
    text = dict(config.get("text_config", config))
    rope = text.get("rope_parameters") or {}
    if rope.get("rope_type", "default") != "default":
      raise ValueError(f"unsupported rope_parameters {rope!r}")
    names = {f.name for f in dataclasses.fields(cls)}
    kwargs = {k: v for k, v in text.items() if k in names}
    if "rope_theta" in rope:
      kwargs["rope_theta"] = float(rope["rope_theta"])
    if num_hidden_layers is not None:
      if num_hidden_layers > kwargs.get("num_hidden_layers", num_hidden_layers):
        raise ValueError("cannot extend the checkpoint's layer count")
      kwargs["num_hidden_layers"] = num_hidden_layers
    return cls(**kwargs)


# Initialization (random-init flavors only; checkpoints overwrite all of it).
_LINEAR_INIT: dict[str, Callable] = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
}
_EMBEDDING_INIT: dict[str, Callable] = {
    "weight": partial(nn.init.normal_, std=1.0),
}
_GAIN_NORM_INIT: dict[str, Callable] = {"weight": nn.init.zeros_}


def _depth_init(layer_id: int) -> dict[str, Callable]:
  return {
      "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id))
  }


def _experts_init(layer_id: int) -> dict[str, Callable]:
  depth = partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id))
  return {
      "w1_EFD": partial(nn.init.trunc_normal_, std=0.02),
      "w2_EDF": depth,
      "w3_EFD": depth,
  }


def _output_linear_init(dim: int) -> dict[str, Callable]:
  std = dim**-0.5
  return {
      "weight": partial(nn.init.trunc_normal_, std=std, a=-3 * std, b=3 * std)
  }


def _residual_gate_init(layer_id: int, temperature: float, eps: float):
  # Start each branch at beta = 1 / (2 (layer_id + 1)).
  beta = min(max(1.0 / (2.0 * (layer_id + 1)), eps), 1.0 - eps)
  return {"gate": partial(nn.init.constant_, val=temperature * math.log(beta / (1.0 - beta)))}


def _norm(
    dim: int, eps: float, *, weight: bool, gain_center: float = 1.0
) -> MuseSparkRMSNorm.Config:
  return MuseSparkRMSNorm.Config(
      normalized_shape=dim,
      eps=eps,
      elementwise_affine=weight,
      gain_center=gain_center,
      param_init=_GAIN_NORM_INIT if weight else None,
  )


def _input_scaled_norm(dim: int, eps: float) -> MuseSparkInputScaledRMSNorm.Config:
  return MuseSparkInputScaledRMSNorm.Config(
      normalized_shape=dim,
      eps=eps,
      elementwise_affine=True,
      gain_center=1.0,
      param_init=_GAIN_NORM_INIT,
  )


def _attention(
    args: MuseSparkArgs, layer_id: int, max_context_length: int
) -> MuseSparkAttention.Config:
  dim, hd = args.hidden_size, args.head_dim
  nh, nkv = args.num_attention_heads, args.num_key_value_heads
  window = (
      args.sliding_window
      if args.layer_types[layer_id] == SLIDING_ATTENTION
      else None
  )
  use_rope = bool(args.layer_rope_theta[layer_id])
  return MuseSparkAttention.Config(
      n_heads=nh,
      n_kv_heads=nkv,
      head_dim=hd,
      dim=dim,
      qkv_linear=QKVLinear.Config(
          head_dim=hd,
          wq=Linear.Config(in_features=dim, out_features=nh * hd, param_init=_LINEAR_INIT),
          wkv=Linear.Config(in_features=dim, out_features=nkv * hd, param_init=_LINEAR_INIT),
      ),
      wo=Linear.Config(
          in_features=nh * hd, out_features=dim, param_init=_depth_init(layer_id)
      ),
      qk_norm=_norm(hd, args.rms_norm_eps, weight=False),
      inner_attention=SlidingWindowScaledDotProductInnerAttention.Config(
          window_size=window
      ),
      rope=(
          CosSinRoPE.Config(
              dim=hd, max_context_length=max_context_length, theta=args.rope_theta
          )
          if use_rope
          else None
      ),
      scale_query_by=args.qk_scale_factor,
      output_norm=_norm(hd, args.rms_norm_eps, weight=False),
      output_gate=Linear.Config(
          in_features=dim, out_features=nh * hd, param_init=_LINEAR_INIT
      ),
      window_size=window,
  )


def _moe(
    args: MuseSparkArgs,
    layer_id: int,
    *,
    load_balance_coeff: float | None,
    experts_compute_dtype: str | None,
    moe_comm_backend: str,
) -> MuseSparkMoE.Config:
  dim, cdim = args.hidden_size, args.moe_hidden_size
  num_experts, top_k = args.num_local_experts, args.num_experts_per_tok
  experts_kwargs = dict(
      dim=cdim,
      hidden_dim=args.moe_intermediate_size,
      num_experts=num_experts,
      param_init=_experts_init(layer_id),
  )
  if experts_compute_dtype == "bfloat16":
    inner_experts = GroupedExperts.Config(**experts_kwargs)
  else:
    inner_experts = MuseSparkGroupedExperts.Config(
        **experts_kwargs, compute_dtype=experts_compute_dtype
    )
  return MuseSparkMoE.Config(
      num_experts=num_experts,
      load_balance_coeff=load_balance_coeff,
      router=MuseSparkRouter.Config(
          num_experts=num_experts,
          gate=RouterGateLinear.Config(
              in_features=dim,
              out_features=num_experts,
              param_init=_depth_init(layer_id),
          ),
          top_k=top_k,
          score_func="sigmoid",
          route_norm=True,
          route_scale=args.routed_scaling_factor,
          logit_multiplier=args.output_multiplier,
          route_norm_eps=1e-15,
      ),
      routed_experts=MuseSparkRoutedExperts.Config(
          inner_experts=inner_experts,
          token_dispatcher=make_token_dispatcher_config(
              num_experts=num_experts,
              top_k=top_k,
              comm_backend=moe_comm_backend,
              hidden_dim=cdim,
          ),
          output_norm=_input_scaled_norm(cdim, args.post_norm_eps),
      ),
      pooler=Linear.Config(in_features=dim, out_features=cdim, param_init=_LINEAR_INIT),
      pre_expert_norm=_norm(cdim, args.rms_norm_eps, weight=True),
      unpooler=Linear.Config(
          in_features=cdim, out_features=dim, param_init=_depth_init(layer_id)
      ),
  )


def _dense_ffn(args: MuseSparkArgs, layer_id: int) -> FeedForward.Config:
  dim, hidden = args.hidden_size, args.intermediate_size
  return FeedForward.Config(
      w1=Linear.Config(in_features=dim, out_features=hidden, param_init=_LINEAR_INIT),
      w2=Linear.Config(
          in_features=hidden, out_features=dim, param_init=_depth_init(layer_id)
      ),
      w3=Linear.Config(
          in_features=dim, out_features=hidden, param_init=_depth_init(layer_id)
      ),
  )


def build_muse_spark_config(
    args: MuseSparkArgs,
    *,
    max_context_length: int,
    load_balance_coeff: float | None = None,
    experts_compute_dtype: str | None = "bfloat16",
    moe_comm_backend: str = "standard",
    residual_dtype: str | None = None,
    branch_dtype: str | None = None,
    final_norm_input_dtype: str | None = None,
) -> MuseSparkModel.Config:
  """Builds the TPU-lane ``MuseSparkModel.Config`` for ``args``.

  Args:
    args: Text-decoder hyperparameters.
    max_context_length: RoPE table length (the training context length).
    load_balance_coeff: Expert-bias update coefficient; None (default) keeps
      the checkpoint's expert bias fixed, as fine-tuning should.
    experts_compute_dtype: ``"bfloat16"`` uses the upstream grouped experts
      (BF16 grouped matmuls, the TPU converters' input); anything else uses
      ``MuseSparkGroupedExperts`` in that dtype (None = the input dtype).
    moe_comm_backend: Token dispatcher backend (``"standard"``).
    residual_dtype: Residual-stream dtype; None follows the embeddings.
    branch_dtype: Sublayer input dtype; None follows the residual stream.
    final_norm_input_dtype: Optional cast before the final norm.

  Returns:
    The model config.
  """
  layers = []
  for layer_id in range(args.num_hidden_layers):
    sparse = args.mlp_layer_types[layer_id] == "sparse"
    gate_init = _residual_gate_init(
        layer_id, args.residual_gate_temperature, 1e-3
    )
    layers.append(
        MuseSparkTransformerBlock.Config(
            attention=_attention(args, layer_id, max_context_length),
            attention_norm=_norm(args.hidden_size, args.rms_norm_eps, weight=True),
            post_attention_norm=_norm(
                args.hidden_size, args.post_norm_eps, weight=False
            ),
            attention_residual=MuseSparkResidualGate.Config(
                dim=args.hidden_size,
                temperature=args.residual_gate_temperature,
                param_init=gate_init,
            ),
            ffn_norm=_norm(args.hidden_size, args.rms_norm_eps, weight=True),
            post_ffn_norm=_input_scaled_norm(args.hidden_size, args.post_norm_eps),
            ffn_residual=MuseSparkResidualGate.Config(
                dim=args.hidden_size,
                temperature=args.residual_gate_temperature,
                param_init=gate_init,
            ),
            moe=(
                _moe(
                    args,
                    layer_id,
                    load_balance_coeff=load_balance_coeff,
                    experts_compute_dtype=experts_compute_dtype,
                    moe_comm_backend=moe_comm_backend,
                )
                if sparse
                else None
            ),
            feed_forward=None if sparse else _dense_ffn(args, layer_id),
            branch_dtype=branch_dtype,
        )
    )
  return MuseSparkModel.Config(
      dim=args.hidden_size,
      vocab_size=args.vocab_size,
      tok_embeddings=MuseSparkEmbedding.Config(
          embedding=Embedding.Config(
              num_embeddings=args.vocab_size,
              embedding_dim=args.hidden_size,
              param_init=_EMBEDDING_INIT,
          ),
          norm=_norm(args.hidden_size, args.rms_norm_eps, weight=False),
      ),
      layers=layers,
      norm=dataclasses.replace(
          _norm(args.hidden_size, args.rms_norm_eps, weight=True, gain_center=0.0),
          param_init={"weight": nn.init.ones_},
      ),
      lm_head=MuseSparkLMHead.Config(
          in_features=args.hidden_size,
          out_features=args.vocab_size,
          param_init=_output_linear_init(args.hidden_size),
          output_multiplier=args.output_multiplier,
          output_soft_cap_temp=args.final_logit_softcapping,
      ),
      residual_dtype=residual_dtype,
      final_norm_input_dtype=final_norm_input_dtype,
  )


def _released_args(num_hidden_layers: int = 62) -> MuseSparkArgs:
  """The released 816B-A42B text decoder (``config.json`` text_config)."""
  return MuseSparkArgs(num_hidden_layers=num_hidden_layers)


def _debug_args() -> MuseSparkArgs:
  return MuseSparkArgs(
      hidden_size=256,
      num_hidden_layers=4,
      num_attention_heads=8,
      num_key_value_heads=4,
      head_dim=32,
      sliding_window=128,
      moe_hidden_size=128,
      moe_intermediate_size=256,
      num_local_experts=8,
      num_experts_per_tok=2,
      qk_scale_factor=43.7840518911 / math.sqrt(32),
      output_multiplier=1.0,
  )


def _tiny_hf_args() -> MuseSparkArgs:
  """``trl-internal-testing/tiny-MuseSparkForConditionalGeneration``."""
  return MuseSparkArgs(
      hidden_size=64,
      intermediate_size=32,
      num_hidden_layers=2,
      num_attention_heads=4,
      num_key_value_heads=2,
      head_dim=16,
      layer_types=[FULL_ATTENTION, SLIDING_ATTENTION],
      layer_rope_theta=[500_000.0, 0.0],
      sliding_window=32,
      moe_hidden_size=16,
      moe_intermediate_size=32,
      num_local_experts=16,
      num_experts_per_tok=4,
  )


# flavor -> (arguments factory, default context length)
MUSE_SPARK_FLAVORS: dict[str, tuple[Callable[[], MuseSparkArgs], int]] = {
    "816B-A42B": (_released_args, 8192),
    "816B-A42B-2L": (partial(_released_args, 2), 8192),
    "816B-A42B-4L": (partial(_released_args, 4), 8192),
    "816B-A42B-8L": (partial(_released_args, 8), 8192),
    "debugmodel": (_debug_args, 2048),
    "tiny-hf": (_tiny_hf_args, 2048),
}


def muse_spark_flavor_args(flavor: str) -> MuseSparkArgs:
  if flavor not in MUSE_SPARK_FLAVORS:
    raise ValueError(
        f"Unknown Muse Spark flavor {flavor!r}; available: {sorted(MUSE_SPARK_FLAVORS)}"
    )
  return MUSE_SPARK_FLAVORS[flavor][0]()
