# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TPU checks of the Splash attention paths for Muse Spark Open attention shapes.

Muse Spark Open text attention (Hugging Face Transformers ``muse_spark``):
GQA with 128 query / 16 KV heads, head_dim 64, a scaleless RMSNorm on q and k,
q multiplied by ``qk_scale_factor`` (5.473) after its norm and before RoPE,
softmax scale ``head_dim ** -0.5``, sliding-window layers (window 2048, i.e. a
query sees itself and the previous 2047 tokens; RoPE theta 5e5) and full-causal
NoPE layers. Training packs several documents into one row, so packed
documents must be masked (segment ids derived from ``positions`` resets).

Reference: ``transformers`` eager attention with Transformers' own causal /
sliding-window masks built from packed ``position_ids``, in FP32 on CPU, forward
and backward (out, dq, dk, dv).

Paths under test (the ones a TorchTitan TPU model can use today):

* ``patch_stock``:   ``SlidingWindowScaledDotProductInnerAttention`` patched by
  ``workarounds.use_splash_attention_patch`` (JAX splash kernel).
* ``patch_tokamax``: the same module patched by
  ``workarounds.use_tokamax_splash_attention_patch`` (Tokamax splash kernel).
* ``patch_tokamax_spark``: the Tokamax patch with the settings recommended for
  Spark: natural-exp softmax, ``max_logit_const=0`` (QK-norm bounds the logits
  to about +-44, so exp(logit) stays inside fp32 without a running max), 4
  query heads of a KV group stacked per block (q block 512) and SEQ_MINOR
  q/k/v layouts.
* ``sbd``:           ``kernels.splash_bias_dropout.sbd_splash_attention``.

Run on a TPU host (one TensorCore is enough)::

  python -m torchtitan.experiments.tpu.muse_spark.tests.test_attention_kernels_tpu --device=tpu

Environment: ``MUSE_SPARK_ATTN_TEST_SEQ`` (default 4096),
``MUSE_SPARK_ATTN_TEST_HEADS`` (default 16 query heads; KV heads = heads / 8).
"""

import dataclasses
import os
import types

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch

from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu.tpu_job_config import SplashAttentionKernelConfig

HEAD_DIM = 64
QK_SCALE = 5.4730064863875
WINDOW = 2048  # Transformers convention: 0 <= q - k < WINDOW
SEQ = int(os.environ.get("MUSE_SPARK_ATTN_TEST_SEQ", "4096"))
HEADS = int(os.environ.get("MUSE_SPARK_ATTN_TEST_HEADS", "16"))
KV_HEADS = max(1, HEADS // 8)
# Packed row: uneven documents, one longer than the window (so the window bites
# inside a document) and one that straddles block boundaries.
DOCS = {
    4096: [1100, 2600, 396],
    8192: [1880, 1650, 1420, 2600, 642],
}
# rel-L2 bounds vs the FP32 reference. Transformers' own bf16 eager attention
# is ~1.4e-2 (out) / ~1.7e-2 (grads) on these inputs; the kernels are well
# inside that. The window / segment controls below exceed these by >100x
# (max abs) or >50x (rel-L2).
TOL = dict(out=1e-2, dq=1.5e-2, dk=1.5e-2, dv=1e-2)


def _doc_lens(seq):
  if seq in DOCS:
    return DOCS[seq]
  third = seq // 3
  return [third, seq - 2 * third, third]


def _text_config(seq):
  from transformers.models.muse_spark.configuration_muse_spark import MuseSparkTextConfig

  cfg = MuseSparkTextConfig(
      hidden_size=8192, num_attention_heads=128, num_key_value_heads=16,
      head_dim=HEAD_DIM, num_hidden_layers=2,
      layer_types=["sliding_attention", "full_attention"],
      layer_rope_theta=[500000.0, 0], sliding_window=WINDOW,
      rope_parameters={"rope_theta": 500000.0, "rope_type": "default"},
      rms_norm_eps=1e-5, qk_scale_factor=QK_SCALE, mlp_layer_types=["sparse"] * 2,
      max_position_embeddings=max(seq, 8192))
  cfg._attn_implementation = "eager"
  return cfg


def _inputs(layer, seq, heads, kv_heads, seed=0):
  """q, k, v exactly as the Transformers attention module feeds its kernel."""
  from transformers.models.muse_spark import modeling_muse_spark as hf

  g = torch.Generator().manual_seed(seed)
  bf = torch.bfloat16
  q = torch.randn(1, heads, seq, HEAD_DIM, generator=g).to(bf)
  k = torch.randn(1, kv_heads, seq, HEAD_DIM, generator=g).to(bf)
  v = torch.randn(1, kv_heads, seq, HEAD_DIM, generator=g).to(bf)
  d_out = torch.randn(1, heads, seq, HEAD_DIM, generator=g).to(bf)
  lens = _doc_lens(seq)
  positions = torch.cat([torch.arange(n) for n in lens])[None]
  segment_ids = torch.cat(
      [torch.full((n,), i, dtype=torch.int32) for i, n in enumerate(lens)])[None]
  norm = hf.MuseSparkRMSNorm(eps=1e-5, with_scale=False)
  q = norm(q) * QK_SCALE
  k = norm(k)
  if layer == "sliding":  # RoPE layers; full layers are NoPE
    cos, sin = hf.MuseSparkTextRotaryEmbedding(_text_config(seq))(v, positions)
    q, k = hf.apply_rotary_pos_emb(q, k, cos, sin)
  return dict(q=q, k=k, v=v, d_out=d_out, positions=positions, segment_ids=segment_ids)


def _reference(inp, layer):
  """Transformers eager attention + masks, FP32, per KV group; heads-first outputs."""
  from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
  from transformers.models.muse_spark import modeling_muse_spark as hf

  q, k, v, d_out = inp["q"], inp["k"], inp["v"], inp["d_out"]
  _, heads, seq, _ = q.shape
  kv_heads = k.shape[1]
  rep = heads // kv_heads
  mask_args = dict(config=_text_config(seq), inputs_embeds=torch.zeros(1, seq, 1),
                   attention_mask=None, past_key_values=None,
                   position_ids=inp["positions"])
  mask = (create_sliding_window_causal_mask(**mask_args) if layer == "sliding"
          else create_causal_mask(**mask_args))
  module = types.SimpleNamespace(num_key_value_groups=rep, training=False)
  outs = {n: [] for n in ("out", "dq", "dk", "dv")}
  for g in range(kv_heads):
    qs = q[:, g * rep:(g + 1) * rep].float().requires_grad_(True)
    ks = k[:, g:g + 1].float().requires_grad_(True)
    vs = v[:, g:g + 1].float().requires_grad_(True)
    o, _ = hf.eager_attention_forward(module, qs, ks, vs, mask, scaling=HEAD_DIM ** -0.5)
    o = o.transpose(1, 2)
    o.backward(d_out[:, g * rep:(g + 1) * rep].float())
    for n, t in (("out", o), ("dq", qs.grad), ("dk", ks.grad), ("dv", vs.grad)):
      outs[n].append(t.detach())
  return {n: torch.cat(ts, dim=1).numpy() for n, ts in outs.items()}


def _splash_config(block_q=1024, block_kv=1024, block_kv_compute=512, **tokamax_knobs):
  """SplashAttentionKernelConfig-like namespace (fwd blocks; dkv blocks 1024/1024/512)."""
  cfg = dict(
      sa_block_q=block_q, sa_block_kv=block_kv, sa_block_dkv=1024,
      sa_block_kv_compute=block_kv_compute, sa_block_q_dkv=1024,
      sa_block_kv_dkv=1024, sa_block_kv_dkv_compute=512,
      sa_block_q_dq=1024, sa_block_kv_dq=1024, sa_use_fused_bwd_kernel=True,
      sa_q_layout="HEAD_DIM_MINOR", sa_k_layout="HEAD_DIM_MINOR",
      sa_v_layout="HEAD_DIM_MINOR", use_vmap_bwd=False,
      sa_use_document_aware_block_skipping=True)
  cfg.update({f"sa_{k}": v for k, v in tokamax_knobs.items()})
  return types.SimpleNamespace(**cfg)


# Recommended Spark settings for the Tokamax path (see the module docstring).
SPARK_TOKAMAX = dict(block_q=512, block_kv=1024, block_kv_compute=512, use_base2_exp=False,
                     max_logit_const=0.0, num_stacked_q_heads=4, q_layout="SEQ_MINOR",
                     k_layout="SEQ_MINOR", v_layout="SEQ_MINOR")
# The two Tokamax knobs above that are not plain block / layout settings; they
# need the matching SplashAttentionKernelConfig fields (a separate change).
SPARK_TOKAMAX_KNOBS = ("use_base2_exp", "max_logit_const")


def _spark_tokamax_knobs_available():
  fields = {f.name for f in dataclasses.fields(SplashAttentionKernelConfig)}
  return all(f"sa_{knob}" in fields for knob in SPARK_TOKAMAX_KNOBS)


def _attend(path, q, k, v, positions, segment_ids, window, splash_cfg):
  """(B, S, H, D) in and out, through one TorchTitan TPU attention entry point."""
  if path in ("patch_stock", "patch_tokamax", "patch_tokamax_spark"):
    from torchtitan.experiments.tpu import workarounds
    from torchtitan.experiments.tpu.kernels import tokamax_splash_attention as tsa
    from torchtitan.experiments.tpu.muse_glimmer.attention import (
        SlidingWindowScaledDotProductInnerAttention,
    )

    parent = torch.nn.Module()
    parent.inner = SlidingWindowScaledDotProductInnerAttention.Config(
        window_size=window).build()
    if path == "patch_stock":
      workarounds.use_splash_attention_patch(parent, splash_cfg)
    else:
      # TokamaxConfig is process-global and its update() skips None: reset the
      # knobs so one test's settings do not leak into the next.
      if _spark_tokamax_knobs_available():
        tsa.TokamaxConfig.use_base2_exp = True
        tsa.TokamaxConfig.max_logit_const = None
      tsa.TokamaxConfig.num_stacked_q_heads = 1
      if path == "patch_tokamax_spark":
        splash_cfg = _splash_config(**SPARK_TOKAMAX)
      workarounds.use_tokamax_splash_attention_patch(parent, splash_cfg)
    return parent.inner(q, k, v, scale=HEAD_DIM ** -0.5, enable_gqa=q.shape[2] != k.shape[2],
                        is_causal=True, positions=positions)
  if path == "sbd":
    from torchtitan.experiments.tpu.kernels import splash_bias_dropout as sbd

    sbd.SbdConfig.update(
        block_q=splash_cfg.sa_block_q, block_kv=splash_cfg.sa_block_kv,
        block_kv_compute=splash_cfg.sa_block_kv_compute, block_q_dkv=splash_cfg.sa_block_q_dkv,
        block_kv_dkv=splash_cfg.sa_block_kv_dkv,
        block_kv_dkv_compute=splash_cfg.sa_block_kv_dkv_compute)
    return sbd.sbd_splash_attention(
        q, k, v, segment_ids=segment_ids, is_causal=True, scale=HEAD_DIM ** -0.5,
        local_window_size=None if window is None else window - 1)
  raise ValueError(path)


def _rel_l2(a, ref):
  return float(np.linalg.norm((a - ref).ravel()) / np.linalg.norm(ref.ravel()))


class MuseSparkSplashAttentionTest(base_device_test.BaseAcceleratorDeviceTest):

  def _run(self, path, layer, inp, window, splash_cfg=None, use_segments=True):
    dev = self.accelerator_device
    to_bshd = lambda t: t.transpose(1, 2).contiguous().to(dev)
    q = to_bshd(inp["q"]).requires_grad_(True)
    k = to_bshd(inp["k"]).requires_grad_(True)
    v = to_bshd(inp["v"]).requires_grad_(True)
    positions = inp["positions"].to(dev) if use_segments else None
    segment_ids = inp["segment_ids"].to(dev) if use_segments else None
    out = _attend(path, q, k, v, positions, segment_ids, window,
                  splash_cfg or _splash_config())
    out.backward(to_bshd(inp["d_out"]))
    back = lambda t: t.detach().float().cpu().transpose(1, 2).numpy()
    return dict(out=back(out), dq=back(q.grad), dk=back(k.grad), dv=back(v.grad))

  @parameterized.product(
      path=("patch_stock", "patch_tokamax", "patch_tokamax_spark", "sbd"),
      layer=("sliding", "full"))
  def test_matches_transformers_reference(self, path, layer):
    if self.accelerator_device.type != "tpu":
      self.skipTest("TPU kernels")
    if path == "patch_tokamax_spark" and not _spark_tokamax_knobs_available():
      self.skipTest(
          "needs SplashAttentionKernelConfig.sa_use_base2_exp and"
          " sa_max_logit_const (Tokamax Splash knobs)"
      )
    inp = _inputs(layer, SEQ, HEADS, KV_HEADS)
    ref = _reference(inp, layer)
    got = self._run(path, layer, inp, WINDOW if layer == "sliding" else None)
    for name, tol in TOL.items():
      self.assertTrue(np.isfinite(got[name]).all(), f"{path}/{layer}/{name} not finite")
      err = _rel_l2(got[name], ref[name])
      print(f"[muse_spark attention] {path} {layer} {name}: rel_l2={err:.3e} (tol {tol})")
      self.assertLess(err, tol, f"{path}/{layer}/{name}")

  def test_controls_are_detected(self):
    """The reference distinguishes a wrong window and missing document masking."""
    if self.accelerator_device.type != "tpu":
      self.skipTest("TPU kernels")
    inp = _inputs("sliding", SEQ, HEADS, KV_HEADS)
    ref = _reference(inp, "sliding")
    off_by_one = self._run("sbd", "sliding", inp, WINDOW + 1)
    self.assertGreater(np.abs(off_by_one["out"] - ref["out"]).max(), 0.5)
    no_segments = self._run("patch_stock", "sliding", inp, WINDOW, use_segments=False)
    self.assertGreater(_rel_l2(no_segments["out"], ref["out"]), 0.1)


if __name__ == "__main__":
  absltest.main()
