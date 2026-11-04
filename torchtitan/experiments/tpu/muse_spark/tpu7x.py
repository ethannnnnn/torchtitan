# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TPU7x kernel settings shared by every Muse Spark TPU recipe. ASCII only.

One place for the settings measured on TPU7x with Spark's real dimensions, so
that the bring-up configs (``config_registry.py``) and the full-model and
fine-tuning recipes built on them cannot drift apart:

* Attention: the Tokamax Splash kernel through TorchTitan's patch with the
  blocks and knobs measured on Spark's attention shapes (128 query / 16
  key-value heads of dimension 64, sliding window 2,048, NoPE full layers,
  packed documents). One TensorCore, one 8,192-token packed row, forward plus
  backward per layer: 16.8 ms versus 36.5 ms for the stock Splash patch at its
  default 512 tiles and 20.0 ms for Tokamax at its default tiles, at 2.3e-3
  relative L2 error against an fp32 reference (natural exponent; base 2 doubles
  the bf16 error with Spark's logits). On the full model (one 4x4x4 slice,
  LoRA, expert parallelism 1) it cut the step from 20.84 to 18.92 s.
  ``max_logit_const=0`` is safe only because Spark's QK-norm and fixed 5.473
  query scale bound |logit| to about 44; do not copy it to models without
  QK-norm. Rows must be a multiple of 1,024 tokens.
* Expert grouped matmuls: Tokamax GMM-v2 (forward, input and weight
  gradients with the kernel's own VMEM-aware tiling; it skips padding rows).
  With the released weights on natural text, LoRA at expert parallelism 1 ran
  at 14.89 s per 1,048,576-token step versus 18.48 s with the MegaBlox kernel
  (one 4x4x4 slice). ``gmm_v2=False`` falls back to the plain Pallas (MegaBlox)
  GMM with K and N tiled (512 / 1,024 / 1,024): Spark's experts are
  4,096 x 4,096, and the paired W1/W3 kernel or untiled K/N stage whole expert
  weights and need 74 to 104 MiB of the 64 MiB scoped VMEM, which does not
  compile.
* libtpu: ``xla_tpu_scoped_vmem_limit_kib=65472`` (the full 64 MiB) and
  ``tpu_flags.TPU7X_THROUGHPUT_P_STATE`` (power state 7: dense bf16 GEMMs
  948 -> 1,100 TFLOP/s per TensorCore; full-model LoRA step 18.92 -> 18.2 s
  with identical losses).

Two of these settings need config fields that are separate TorchTitan changes:
``SplashAttentionKernelConfig.sa_use_base2_exp`` / ``sa_max_logit_const``
(Tokamax Splash knobs) and ``TpuGroupedExperts.Config.use_tokamax_gmm_v2``
(Tokamax GMM-v2 experts). ``apply`` checks for them: without the Splash fields
the kernel keeps its defaults for those two knobs (base-2 softmax, online row
max), and without GMM-v2 the experts use the tiled MegaBlox kernels. Both
fallbacks are logged as warnings.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from torchtitan.experiments.tpu import tpu_flags
from torchtitan.tools.logging import logger

SPLASH_ATTENTION = dict(
    use_tokamax_splash_attention_kernel=True,
    sa_block_q=512, sa_block_kv=1024, sa_block_kv_compute=512,               # forward
    sa_block_q_dkv=1024, sa_block_kv_dkv=1024, sa_block_kv_dkv_compute=512,  # backward
    sa_use_fused_bwd_kernel=True,
    sa_q_layout="SEQ_MINOR", sa_k_layout="SEQ_MINOR", sa_v_layout="SEQ_MINOR",
    sa_use_base2_exp=False,
    sa_max_logit_const=0.0,
    sa_num_stacked_q_heads=4,
)
"""Tokamax Splash settings for Spark attention (see the module docstring)."""

GMM_TILES = (512, 1024, 1024)
"""(m, k, n) tiles of the MegaBlox grouped matmuls (fallback), also the dWeight tiling."""

GMM_V2 = True
"""Tokamax GMM-v2 for the routed experts (default); False = tiled MegaBlox."""

LIBTPU_FLAGS = tpu_flags.merge_flags(
    tpu_flags.TPU7X_THROUGHPUT_P_STATE,
    {"xla_tpu_scoped_vmem_limit_kib": "65472"},
)


def _field_names(config_cls: type) -> set[str]:
  return {field.name for field in dataclasses.fields(config_cls)}


def splash_attention_config() -> Any:
  """``SplashAttentionKernelConfig`` with the Spark Tokamax settings.

  Settings whose field this TorchTitan does not have are left out (logged).
  """
  # pylint: disable-next=g-import-not-at-top
  from torchtitan.experiments.tpu.tpu_job_config import SplashAttentionKernelConfig

  supported = _field_names(SplashAttentionKernelConfig)
  missing = sorted(set(SPLASH_ATTENTION) - supported)
  if missing:
    logger.warning(
        "Muse Spark TPU7x settings: SplashAttentionKernelConfig has no %s;"
        " the attention kernel keeps its defaults for them.",
        missing,
    )
  return SplashAttentionKernelConfig(
      **{k: v for k, v in SPLASH_ATTENTION.items() if k in supported}
  )


def tile_expert_matmuls(cfg: Any, *, gmm_v2: bool = GMM_V2) -> None:
  """Tokamax GMM-v2 (or unpaired, K/N-tiled MegaBlox) with kernel-native weights.

  Falls back to the tiled MegaBlox kernels (logged) if this TorchTitan has no
  Tokamax GMM-v2 expert path.
  """
  # pylint: disable-next=g-import-not-at-top
  from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts

  has_gmm_v2 = "use_tokamax_gmm_v2" in _field_names(TpuGroupedExperts.Config)
  if gmm_v2 and not has_gmm_v2:
    logger.warning(
        "Muse Spark TPU7x settings: TpuGroupedExperts has no Tokamax GMM-v2"
        " path (use_tokamax_gmm_v2); the experts use the tiled MegaBlox"
        " grouped matmuls instead."
    )
    gmm_v2 = False
  for _, experts, _, _ in cfg.model_spec.model.traverse(TpuGroupedExperts.Config):
    experts.use_pallas_grouped_mm = True
    experts.use_pallas_paired_grouped_mm = False
    experts.use_kernel_native_weight_layout = True
    (experts.pallas_grouped_mm_tile_m, experts.pallas_grouped_mm_tile_k,
     experts.pallas_grouped_mm_tile_n) = GMM_TILES
    experts.pallas_grouped_mm_tgmm_tiling = GMM_TILES
    experts.pallas_grouped_mm_w2_tgmm_tiling = GMM_TILES
    experts.pallas_grouped_mm_w2_transposed_tgmm_tiling = GMM_TILES
    if has_gmm_v2:
      experts.use_tokamax_gmm_v2 = gmm_v2


def apply(cfg: Any, *, gmm_v2: bool = GMM_V2) -> Any:
  """Attention kernel, expert matmul kernel and libtpu flags for a Spark TPU config."""
  cfg.splash_attention_kernel = splash_attention_config()
  tile_expert_matmuls(cfg, gmm_v2=gmm_v2)
  cfg.libtpu_init_flags = tpu_flags.merge_flags(cfg.libtpu_init_flags or {}, LIBTPU_FLAGS)
  return cfg
