# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# The layer math follows the public Muse Spark Open text model in Hugging Face
# Transformers (src/transformers/models/muse_spark, Copyright 2026 the
# HuggingFace Team, Apache License 2.0) and the BSD-3 upstream TorchTitan port
# (torchtitan/models/muse_spark in pytorch/torchtitan). Parts of the module
# structure are adapted from that BSD-3 port to this fork's APIs and TPU kernels.

"""Muse Spark Open text decoder modules for the TorchTitan TPU lane.

The text decoder of ``meta-models/Muse-Spark-Open-816B-A42B`` (Transformers
``model_type="muse_spark"``) differs from a Llama-style decoder in these ways,
all implemented here:

* Gain-centered RMSNorms: the effective scale is ``weight + gain_center``
  (``gain_center=1`` everywhere except the final norm). Every norm computes in
  FP32 and returns the input dtype.
* Token embeddings are followed by a scaleless RMSNorm.
* Attention: GQA with scaleless per-head Q/K RMSNorm, the query multiplied by
  ``qk_scale_factor`` after its norm, RoPE (rotate-half) only on layers whose
  ``layer_rope_theta`` is non-zero (NoPE otherwise), causal sliding-window
  attention on ``sliding_attention`` layers, a scaleless per-head RMSNorm on
  the attention output, and a sigmoid output gate before the output
  projection.
* Post-sublayer norms: a scaleless RMSNorm after attention and an
  input-scaled RMSNorm (scale first, normalize second) after the FFN.
* Norm-preserving residual gates replace plain residual additions:
  ``alpha * residual + beta * branch`` with ``beta = sigmoid(g / T)`` and
  ``alpha = sqrt(max(sigmoid(-g / T) * (1 + beta), eps))``.
* Compressed MoE: the router reads the full hidden state; the expert input is
  pooled from ``hidden_size`` to ``moe_hidden_size`` and gain-normalized, each
  routed SwiGLU expert output is input-scale normalized (one norm shared by
  all experts) before the routing weights are applied, and the sum is
  unpooled back to ``hidden_size``. Routing uses FP32 sigmoid scores of the
  logits times ``output_multiplier``; the expert bias only affects selection.
* The LM head scales the logits by ``output_multiplier`` before a tanh soft
  cap (``muse_glimmer.model.SoftCappedLinear``).

All tensors use TorchTitan's folded token layout (``[T, ...]``); packed
documents are identified by ``positions`` restarting at 0.
"""

from dataclasses import dataclass
import importlib
import importlib.util
from types import ModuleType
from typing import cast

import torch
import torch.nn.functional as F
import torch_remat as remat
from torch import nn
from torch.distributed.tensor import DTensor

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.attention import GQAttention
from torchtitan.models.common.attention import segment_ids_from_positions
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.decoder import TransformerBlock
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import _tokens_per_expert
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.moe import MoELoadBalanceAuxLoss
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.moe import TokenChoiceTopKRouter
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.muse_glimmer.model import SoftCappedLinear
from torchtitan.models.utils import get_nparams_and_active_nparams
from torchtitan.models.utils import quadratic_attention_flops_per_token
from torchtitan.protocols.module import Module

__all__ = [
    "MuseSparkAttention",
    "MuseSparkEmbedding",
    "MuseSparkGroupedExperts",
    "MuseSparkInputScaledRMSNorm",
    "MuseSparkLMHead",
    "MuseSparkMoE",
    "MuseSparkModel",
    "MuseSparkRMSNorm",
    "MuseSparkResidualGate",
    "MuseSparkRoutedExperts",
    "MuseSparkRouter",
    "MuseSparkTransformerBlock",
]


def _maybe_dtype(name: str | None) -> torch.dtype | None:
  return None if name is None else TORCH_DTYPE_MAP[name]


_EXPERT_PARALLEL_MODULE = "torchtitan.experiments.tpu.muse_spark.expert_parallel"


def expert_parallel_module() -> ModuleType:
  """Imports ``muse_spark.expert_parallel`` (expert parallelism > 1).

  Imported only when expert parallelism or a non-contiguous expert placement
  is requested, so the model, the checkpoint mapping and FSDP training
  (expert parallelism 1) do not depend on it.

  Raises:
    NotImplementedError: If the expert-parallel module is not available.
  """
  if importlib.util.find_spec(_EXPERT_PARALLEL_MODULE) is None:
    raise NotImplementedError(
        "Muse Spark expert parallelism (expert_parallel_degree > 1 or an"
        " expert placement file) needs muse_spark/expert_parallel.py, which"
        " is not available; use expert parallelism 1 (FSDP-sharded experts)."
    )
  return importlib.import_module(_EXPERT_PARALLEL_MODULE)


class MuseSparkRMSNorm(RMSNorm):
  """RMSNorm in FP32 with effective scale ``weight + gain_center``.

  ``elementwise_affine=False`` gives the scaleless norm used for Q/K, the
  attention output, the token embeddings and the post-attention branch. With
  a weight, ``gain_center=1`` is the gain-centered norm (checkpoint weights
  are stored zero-centered) and ``gain_center=0`` the plain final norm.

  The FP32 normalization is multiplied by ``(weight + gain_center)`` computed
  in the weight's dtype and then upcast, and the result is cast back to the
  input dtype, matching the reference model's rounding in BF16.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(RMSNorm.Config):
    gain_center: float = 1.0

  def __init__(self, config: Config):
    super().__init__(config)
    self.gain_center = config.gain_center

  def _effective_weight(self) -> torch.Tensor | None:
    if self.weight is None:
      return None
    return (self.weight + self.gain_center).float()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = F.rms_norm(x.float(), self.normalized_shape, None, self.eps)
    weight = self._effective_weight()
    if weight is not None:
      out = out * weight
    return out.to(x.dtype)


class MuseSparkInputScaledRMSNorm(MuseSparkRMSNorm):
  """Scale first, then normalize: ``rms_norm(x * (weight + gain_center))``.

  The scaled input is rounded to the input dtype before the FP32
  normalization (as in the reference model), so the learned scale acts on
  the direction of the vector, not on its norm. Used after each FFN and on
  every routed expert output.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(MuseSparkRMSNorm.Config):
    pass

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    weight = self._effective_weight()
    scaled = x.float() if weight is None else x.float() * weight
    out = F.rms_norm(
        scaled.to(x.dtype).float(), self.normalized_shape, None, self.eps
    )
    return out.to(x.dtype)


class MuseSparkResidualGate(Module):
  """Norm-preserving residual connection with a learned per-channel gate.

  ``out = alpha * residual + beta * branch`` where ``w = gate / temperature``,
  ``beta = sigmoid(w)`` and ``alpha = sqrt(max(sigmoid(-w) * (1 + beta),
  eps))``. The coefficients are computed in FP32 from the gate parameter and
  the result is returned in the residual's dtype.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(Module.Config):
    dim: int
    temperature: float = 0.3
    eps: float = 1e-3

  def __init__(self, config: Config):
    super().__init__()
    self.temperature = config.temperature
    self.eps = config.eps
    self.gate = nn.Parameter(torch.empty(config.dim))

  def forward(self, residual: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
    w = self.gate.float() / self.temperature
    beta = torch.sigmoid(w)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=self.eps))
    return (alpha * residual + beta * branch).to(residual.dtype)


class MuseSparkEmbedding(Module):
  """Token embedding followed by a scaleless RMSNorm.

  The norm is a child that runs after the embedding child, so under vocab
  parallelism it sees the reduced embedding output (same structure as
  ``muse_glimmer.model.EmbeddingWithNorm``).
  """

  @dataclass(kw_only=True, slots=True)
  class Config(Module.Config):
    embedding: Embedding.Config
    norm: MuseSparkRMSNorm.Config

  def __init__(self, config: Config) -> None:
    super().__init__()
    self.embedding = config.embedding.build()
    self.norm = config.norm.build()

  def forward(self, tokens: torch.Tensor) -> torch.Tensor:
    return self.norm(self.embedding(tokens))


class MuseSparkLMHead(SoftCappedLinear):
  """``SoftCappedLinear``: ``cap * tanh(logits * output_multiplier / cap)``.

  A distinct class name only; the TPU fused-loss override recognizes the
  ``SoftCappedLinear`` base and carries the multiplier and cap into the
  fused cross-entropy kernel.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(SoftCappedLinear.Config):
    pass


class MuseSparkAttention(GQAttention):
  """Muse Spark GQA attention (see the module docstring).

  ``rope=None`` makes a NoPE layer. The inner attention receives ``positions``
  and ``segment_ids`` so the TPU SDPA / Splash paths mask packed documents
  and apply the layer's sliding window.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(GQAttention.Config):
    scale_query_by: float
    output_norm: MuseSparkRMSNorm.Config | None = None
    output_gate: Linear.Config | None = None
    # Reference-model window: a query attends to itself and the previous
    # ``window_size - 1`` tokens. None = full causal attention.
    window_size: int | None = None

    @property
    def sliding_window_size(self) -> int | None:
      # Read by generator wrappers that configure per-layer windows.
      return self.window_size

  def __init__(self, config: Config):
    super().__init__(config)
    self.scale_query_by = config.scale_query_by
    self.window_size = config.window_size
    self.output_norm = (
        config.output_norm.build() if config.output_norm is not None else None
    )
    self.output_gate = (
        config.output_gate.build() if config.output_gate is not None else None
    )

  def forward(
      self,
      x_TD: torch.Tensor,
      attention_masks: AttentionMasksType | None,
      positions: torch.Tensor | None = None,
      segment_ids: torch.Tensor | None = None,
  ) -> torch.Tensor:
    num_tokens = x_TD.shape[0]
    xq_THK, xk_THK, xv_THV = self.qkv_linear(x_TD)

    if self.q_norm is not None or self.k_norm is not None:
      if self.q_norm is None or self.k_norm is None:
        raise ValueError("MuseSparkAttention needs both q_norm and k_norm.")
      xq_THK = self.q_norm(xq_THK) * self.scale_query_by
      xk_THK = self.k_norm(xk_THK)

    if self.rope is not None:
      xq_THK, xk_THK = self.rope(xq_THK, xk_THK, positions)

    out_THV = remat.region(
        self.inner_attention,
        self.remat_region_name("inner_attention"),
        recompute=self.remat_should_recompute("inner_attention"),
    )(
        xq_THK,
        xk_THK,
        xv_THV,
        attention_masks=attention_masks,
        scale=self.scaling,
        enable_gqa=self.enable_gqa,
        positions=positions,
        segment_ids=segment_ids,
    )
    out_THV = out_THV.contiguous()
    if self.output_norm is not None:
      out_THV = self.output_norm(out_THV)
    out_TO = out_THV.view(num_tokens, -1)
    if self.output_gate is not None:
      out_TO = out_TO * torch.sigmoid(self.output_gate(x_TD))
    return self.wo(out_TO)


class MuseSparkRouter(TokenChoiceTopKRouter):
  """Sigmoid top-k router with a logit multiplier.

  ``scores = sigmoid(gate(x) * logit_multiplier)`` in FP32 (``gate`` is a
  ``RouterGateLinear`` with FP32 output). Experts are chosen by
  ``scores + expert_bias``; the combine weights are the chosen unbiased
  scores, normalized by their sum plus ``route_norm_eps`` and multiplied by
  ``route_scale``.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(TokenChoiceTopKRouter.Config):
    logit_multiplier: float = 1.0
    route_norm_eps: float = 1e-15

  def __init__(self, config: Config):
    if config.score_func != "sigmoid":
      raise ValueError("MuseSparkRouter uses sigmoid scores")
    if config.num_expert_groups is not None:
      raise ValueError("MuseSparkRouter does not use node-limited routing")
    super().__init__(config)
    self.logit_multiplier = config.logit_multiplier
    self.route_norm_eps = config.route_norm_eps

  def forward(
      self,
      x_TD: torch.Tensor,
      expert_bias_E: torch.Tensor | None = None,
      **router_kwargs,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    logits_TE = self.gate(x_TD)
    scores_TE = torch.sigmoid(logits_TE.float() * self.logit_multiplier)

    if self._debug_force_load_balance:
      topk_expert_ids_TK, topk_scores_TK = self._debug_force_load_balance_routing(
          scores_TE
      )
    else:
      # Routing choices must remain identical between forward and replay.
      topk_expert_ids_TK = remat.region(
          self._select_experts,
          "routing_decision",
          recompute=False,
      )(scores_TE, expert_bias_E, **router_kwargs)
      remat.recompute_needs_tensor(topk_expert_ids_TK)
      topk_scores_TK = scores_TE.gather(dim=-1, index=topk_expert_ids_TK)

    if self.route_norm:
      denominator = topk_scores_TK.sum(dim=-1, keepdim=True) + self.route_norm_eps
      topk_scores_TK = topk_scores_TK / denominator
    topk_scores_TK = topk_scores_TK * self.route_scale

    routing_map_TE = (
        torch.zeros_like(scores_TE, dtype=torch.bool).scatter_(
            -1, topk_expert_ids_TK, True
        )
        if self.build_routing_map
        else None
    )
    if self.training and self.aux_loss is not None:
      aux_kwargs = (
          {"expert_ids_TK": topk_expert_ids_TK}
          if isinstance(self.aux_loss, MoELoadBalanceAuxLoss)
          else {}
      )
      topk_scores_TK = self.aux_loss(
          scores_TE, routing_map_TE, carrier=topk_scores_TK, **aux_kwargs
      )
    return topk_scores_TK, topk_expert_ids_TK, routing_map_TE


class MuseSparkGroupedExperts(GroupedExperts):
  """``GroupedExperts`` with a configurable compute dtype.

  Upstream ``GroupedExperts`` always runs the grouped matmuls in BF16.
  ``compute_dtype=None`` computes in the input dtype instead; the reference
  parity tests use it to compare in FP32. Training configs use the plain
  ``GroupedExperts`` (or the TPU kernels selected by
  ``TpuGroupedExpertsConverter``).
  """

  @dataclass(kw_only=True, slots=True)
  class Config(GroupedExperts.Config):
    compute_dtype: str | None = None

  def __init__(self, config: Config):
    super().__init__(config)
    self.compute_dtype = _maybe_dtype(config.compute_dtype)

  def forward(
      self,
      x_RD: torch.Tensor,
      num_tokens_per_expert_E: torch.Tensor,
  ) -> torch.Tensor:
    w1, w2, w3 = self.w1_EFD, self.w2_EDF, self.w3_EFD
    if isinstance(w1, DTensor):
      w1, w2, w3 = w1.to_local(), w2.to_local(), w3.to_local()
    dtype = self.compute_dtype or x_RD.dtype
    offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
    x = x_RD.to(dtype)

    def grouped_mm(a: torch.Tensor, weight_EOI: torch.Tensor) -> torch.Tensor:
      if dtype in (torch.float32, torch.bfloat16, torch.float16):
        return torch._grouped_mm(
            a, weight_EOI.to(dtype).transpose(-2, -1), offs=offsets_E
        )
      # Reference loop for other dtypes (e.g. FP64 checks); syncs the host.
      rows = torch.split(a, num_tokens_per_expert_E.tolist())
      return torch.cat(
          [r @ weight_EOI[e].to(dtype).transpose(-2, -1) for e, r in enumerate(rows)]
      )

    h_RF = F.silu(grouped_mm(x, w1)) * grouped_mm(x, w3)
    return grouped_mm(h_RF, w2).type_as(x_RD)


def _ep_dispatch(dispatcher) -> bool:
  """True for an EP dispatcher wired to an EP mesh (EP > 1)."""
  return getattr(dispatcher, "ep_mesh", None) is not None


class MuseSparkRoutedExperts(RoutedExperts):
  """``RoutedExperts`` that normalizes each routed expert output.

  The input-scaled RMSNorm ``output_norm`` is shared by all experts and runs
  on every routed row after the expert FFN and before the routing weights are
  applied; it is row-wise, so the row order does not matter.

  EP = 1: the norm runs on the expert-major rows before ``combine``.

  EP > 1: the norm runs on the token-owning rank, on the rows returned by the
  dispatcher's ``return_rows`` and before its ``weighted_reduce``. Its input
  is then exactly this rank's ``T * K`` routes, as with EP = 1, never the
  padding rows of a fixed-size receive buffer. ``output_norm.weight`` stays a
  dense (non-expert) parameter: it is data-parallel over every rank, so the
  data-parallel gradient reduction (SimpleFSDP reduce-scatter over the full
  dense mesh, which includes the EP ranks) sums its per-rank partial
  gradients into the gradient of all routed rows of the global batch. It must
  not be sharded or reduced with the expert weights, whose gradients are
  reduced only over the expert-data-parallel (``efsdp``) ranks.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(RoutedExperts.Config):
    output_norm: MuseSparkInputScaledRMSNorm.Config

  def __init__(self, config: Config):
    super().__init__(config)
    self.output_norm = config.output_norm.build()

  def forward(
      self,
      x_TD: torch.Tensor,
      topk_scores_TK: torch.Tensor,
      topk_expert_ids_TK: torch.Tensor,
      num_local_tokens_per_expert_E: torch.Tensor,
  ) -> torch.Tensor:
    (
        routed_input_RD,
        num_global_tokens_per_local_expert_e,
        metadata,
    ) = self.token_dispatcher.dispatch(
        x_TD,
        topk_scores_TK,
        topk_expert_ids_TK,
        num_local_tokens_per_expert_E,
    )
    routed_output_RD = self.inner_experts(
        routed_input_RD, num_global_tokens_per_local_expert_e
    )
    if _ep_dispatch(self.token_dispatcher):
      returned_ND = self.token_dispatcher.return_rows(routed_output_RD, metadata)
      return self.token_dispatcher.weighted_reduce(
          self.output_norm(returned_ND), metadata, x_TD
      )
    routed_output_RD = self.output_norm(routed_output_RD)
    return self.token_dispatcher.combine(routed_output_RD, metadata, x_TD)

  def parallelize(self, parallel_dims) -> None:
    super().parallelize(parallel_dims)
    if _ep_dispatch(self.token_dispatcher) and not (
        hasattr(self.token_dispatcher, "return_rows")
        and hasattr(self.token_dispatcher, "weighted_reduce")
    ):
      raise NotImplementedError(
          "Muse Spark with expert parallelism needs a token dispatcher with"
          " return_rows / weighted_reduce (the shared post-expert norm runs"
          " between them), e.g. RaggedAllToAllTokenDispatcher; got"
          f" {type(self.token_dispatcher).__name__}."
      )


class MuseSparkMoE(MoE):
  """Compressed MoE: route at ``dim``, run experts at ``moe_dim``.

  ``expert_bias_E`` is always a persistent FP32 buffer (the checkpoint's
  ``e_score_correction_bias``). With ``load_balance_coeff=None`` it stays
  fixed; otherwise the standard optimizer hook updates it from the routed
  token counts.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(MoE.Config):
    pooler: Linear.Config
    pre_expert_norm: MuseSparkRMSNorm.Config
    unpooler: Linear.Config

  def __init__(self, config: Config):
    if config.shared_experts is not None:
      raise ValueError("Muse Spark MoE has no shared experts")
    super().__init__(config)
    self.num_experts = config.num_experts
    self.pooler = config.pooler.build()
    self.pre_expert_norm = config.pre_expert_norm.build()
    self.unpooler = config.unpooler.build()
    if self.expert_bias_E is None:
      del self.expert_bias_E
      self.register_buffer(
          "expert_bias_E",
          torch.zeros(self.num_experts, dtype=torch.float32),
          persistent=True,
      )

  def forward(self, x_TD: torch.Tensor, **router_kwargs) -> torch.Tensor:
    topk_scores_TK, topk_expert_ids_TK, routing_map_TE = self.router(
        x_TD, self.expert_bias_E, **router_kwargs
    )
    num_local_tokens_per_expert_E = (
        routing_map_TE.sum(dim=0)
        if routing_map_TE is not None
        else _tokens_per_expert(topk_expert_ids_TK, self.num_experts)
    )
    if self.training:
      with torch.no_grad():
        self.tokens_per_expert_E.add_(num_local_tokens_per_expert_E)

    expert_input_TC = self.pre_expert_norm(self.pooler(x_TD))
    out_TC = self.routed_experts(
        expert_input_TC,
        topk_scores_TK,
        topk_expert_ids_TK,
        num_local_tokens_per_expert_E,
    )
    return self.unpooler(out_TC)

  def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
    super()._init_self_buffers(buffer_device=buffer_device)
    if self.load_balance_coeff is None:
      if buffer_device is None:
        buffer_device = self.tokens_per_expert_E.device
      with torch.device(buffer_device):
        self.expert_bias_E = torch.zeros(self.num_experts, dtype=torch.float32)


class MuseSparkTransformerBlock(TransformerBlock):
  """Muse Spark decoder block.

  ``h = gate_a(x, post_attn_norm(attn(attn_norm(x))))``
  ``out = gate_f(h, post_ffn_norm(ffn(ffn_norm(h))))``

  ``branch_dtype`` optionally casts the sublayer inputs (e.g. BF16 branches on
  an FP32 residual stream); None keeps the residual dtype.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(TransformerBlock.Config):
    # pyrefly: ignore [bad-override]
    attention: MuseSparkAttention.Config
    post_attention_norm: MuseSparkRMSNorm.Config
    attention_residual: MuseSparkResidualGate.Config
    post_ffn_norm: MuseSparkInputScaledRMSNorm.Config
    ffn_residual: MuseSparkResidualGate.Config
    branch_dtype: str | None = None

    def __post_init__(self) -> None:
      if (self.moe is None) == (self.feed_forward is None):
        raise ValueError(
            "MuseSparkTransformerBlock needs exactly one of moe / feed_forward"
        )

  def __init__(self, config: Config):
    super().__init__()
    self.attention = config.attention.build()
    self.attention_norm = config.attention_norm.build()
    self.post_attention_norm = config.post_attention_norm.build()
    self.attention_residual = config.attention_residual.build()
    self.ffn_norm = config.ffn_norm.build()
    self.post_ffn_norm = config.post_ffn_norm.build()
    self.ffn_residual = config.ffn_residual.build()
    self.moe_enabled = config.moe is not None
    if self.moe_enabled:
      self.moe = config.moe.build()
    else:
      self.feed_forward = config.feed_forward.build()
    self.branch_dtype = _maybe_dtype(config.branch_dtype)

  def _branch(self, x: torch.Tensor) -> torch.Tensor:
    return x if self.branch_dtype is None else x.to(self.branch_dtype)

  def forward(
      self,
      x: torch.Tensor,
      attention_masks: AttentionMasksType | None,
      positions: torch.Tensor | None = None,
      segment_ids: torch.Tensor | None = None,
  ) -> torch.Tensor:
    h = self.attention(
        self._branch(self.attention_norm(x)),
        attention_masks,
        positions,
        segment_ids,
    )
    x = self.attention_residual(x, self.post_attention_norm(h))
    h = self._branch(self.ffn_norm(x))
    h = self.moe(h) if self.moe_enabled else self.feed_forward(h)
    return self.ffn_residual(x, self.post_ffn_norm(h))


class MuseSparkModel(Decoder):
  """Muse Spark Open text decoder.

  Layers find packed documents and their sliding windows from ``positions``
  (``SlidingWindowScaledDotProductInnerAttention``), so no attention masks
  are built: ``get_attention_masks`` returns None.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(Decoder.Config):
    dim: int = 8192
    vocab_size: int = 202048
    # pyrefly: ignore [bad-override]
    tok_embeddings: MuseSparkEmbedding.Config
    # pyrefly: ignore [bad-override]
    norm: MuseSparkRMSNorm.Config
    # None keeps the residual stream in the embedding output dtype, as the
    # reference model does; "float32" keeps an FP32 stream (then set each
    # block's ``branch_dtype`` to the compute dtype).
    residual_dtype: str | None = None
    final_norm_input_dtype: str | None = None
    # Expert parallelism (expert_parallel_degree > 1): receive-buffer size of
    # the ragged all-to-all dispatcher, in multiples of a rank's own routes.
    ep_capacity_factor: float = 1.5
    ep_buffer_multiple: int = 1024
    # "allgather" | "allgather_bf16" | "ragged" | "ragged_a2a" (see
    # expert_parallel.configure_expert_parallel).
    ep_dispatcher: str = "allgather"
    # "contiguous" | "interleaved" expert-to-rank placement; with
    # "interleaved" the expert weight rows are stored in physical-slot order
    # and ep_placement_degree records the EP degree they were permuted for.
    ep_expert_placement: str = "contiguous"
    ep_placement_degree: int = 1
    # Dtype of the expert weights' gradient reduce-scatter over the efsdp
    # ranks (expert parallelism > 1): None follows
    # training.mixed_precision_reduce; "bfloat16" or "float32" overrides it
    # for the EP-sharded experts only.
    ep_grad_reduce_dtype: str | None = None

    def __post_init__(self) -> None:
      if self.enable_weight_tying:
        raise ValueError("Muse Spark does not tie embeddings and lm_head")

    def update_from_config(self, *, config, **kwargs) -> None:
      Decoder.Config.update_from_config(self, config=config, **kwargs)
      parallelism = config.parallelism
      if parallelism.expert_parallel_degree > 1:
        expert_parallel_module().configure_expert_parallel(
            self,
            ep_degree=parallelism.expert_parallel_degree,
            capacity_factor=self.ep_capacity_factor,
            buffer_multiple=self.ep_buffer_multiple,
            dispatcher=self.ep_dispatcher,
            placement=self.ep_expert_placement,
        )
      if parallelism.tensor_parallel_degree > 1:
        raise NotImplementedError(
            "Muse Spark: tensor parallelism is not supported on the TPU lane"
            " yet; use FSDP (data_parallel_shard_degree)"
        )
      if parallelism.context_parallel_degree > 1:
        raise NotImplementedError(
            "Muse Spark: context parallelism is not supported yet"
        )

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, int]:
      nparams, active_nparams = get_nparams_and_active_nparams(model)
      attention_op_flops = 0
      for layer in self.layers:
        attention = layer.attention
        head_dim = (
            attention.head_dim
            if attention.head_dim is not None
            else attention.dim // attention.n_heads
        )
        attention_op_flops += quadratic_attention_flops_per_token(
            num_heads=attention.n_heads,
            qk_head_dim=head_dim,
            v_head_dim=head_dim,
            seq_len=seq_len,
            sliding_window_size=attention.window_size,
        )
      return nparams, 6 * active_nparams + attention_op_flops

  def __init__(self, config: Config) -> None:
    super().__init__(config)
    self.residual_dtype = _maybe_dtype(config.residual_dtype)
    self.final_norm_input_dtype = _maybe_dtype(config.final_norm_input_dtype)

  def forward(
      self,
      tokens: torch.Tensor,
      positions: torch.Tensor | None = None,
      attention_masks: AttentionMasksType | None = None,
  ):
    h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
    if self.residual_dtype is not None:
      h = h.to(self.residual_dtype)
    # Once per step, above the per-layer activation-checkpoint regions.
    segment_ids = (
        segment_ids_from_positions(positions) if positions is not None else None
    )
    for layer in self.layers.values():
      h = layer(h, attention_masks, positions, segment_ids)

    if self.norm is not None:
      if self.final_norm_input_dtype is not None:
        h = h.to(self.final_norm_input_dtype)
      h = self.norm(h)
    if self._skip_lm_head:
      return h
    return self.lm_head(h) if self.lm_head is not None else h

  def get_attention_masks(
      self,
      positions: torch.Tensor,
      *,
      padding_mask: torch.Tensor | None = None,
      max_num_documents: int | None = None,
      max_context_length: int | None = None,
  ) -> None:
    """Returns None: every layer derives its mask from ``positions``."""
    del positions, padding_mask, max_num_documents, max_context_length
    return None


def moe_layers(model: nn.Module) -> list[MuseSparkMoE]:
  """Every ``MuseSparkMoE`` in ``model``."""
  return [cast(MuseSparkMoE, m) for m in model.modules() if isinstance(m, MuseSparkMoE)]
