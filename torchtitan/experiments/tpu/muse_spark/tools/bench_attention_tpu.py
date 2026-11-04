# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Times Muse Spark Open attention (fwd and fwd+bwd) through the TorchTitan TPU entry points.

One packed row per call on one TensorCore, 128 query / 16 KV heads, head_dim 64, for the
sliding-window (2048) and the full-causal layer types, through:

* ``patch_stock``:   SlidingWindowScaledDotProductInnerAttention + use_splash_attention_patch
* ``patch_tokamax``: SlidingWindowScaledDotProductInnerAttention + use_tokamax_splash_attention_patch
* ``patch_tokamax_spark``: the Tokamax patch with q block 512, natural exp, max_logit_const=0,
  4 stacked query heads per KV group and SEQ_MINOR q/k/v layouts (recommended for Spark)
* ``sbd``:           kernels.splash_bias_dropout.sbd_splash_attention

Example (any TPU7x host)::

  python torchtitan/experiments/tpu/muse_spark/tools/bench_attention_tpu.py \
      --seq 8192 --docs 1880,1650,1420,2600,642 --blocks 1024:1024:512

``--docs`` lists the packed document lengths (must sum to ``--seq``); use a single value
for an unpacked row. Results are wall-clock medians with a device sync around each call.
"""

import argparse
import dataclasses
import json
import statistics
import time
import types

import torch

HEAD_DIM = 64
WINDOW = 2048


def _sync():
  from torch_tpu._internal import sync

  sync.synchronize(wait=True)


def _splash_config(bq, bkv, bkvc, bqd=None, bkvd=None, bkvcd=None, **tokamax_knobs):
  bqd, bkvd, bkvcd = bqd or bq, bkvd or bkv, bkvcd or bkvc
  cfg = dict(
      sa_block_q=bq, sa_block_kv=bkv, sa_block_dkv=bkvd, sa_block_kv_compute=bkvc,
      sa_block_q_dkv=bqd, sa_block_kv_dkv=bkvd, sa_block_kv_dkv_compute=bkvcd,
      sa_block_q_dq=bqd, sa_block_kv_dq=bkvd, sa_use_fused_bwd_kernel=True,
      sa_q_layout="HEAD_DIM_MINOR", sa_k_layout="HEAD_DIM_MINOR",
      sa_v_layout="HEAD_DIM_MINOR", use_vmap_bwd=False,
      sa_use_document_aware_block_skipping=True)
  cfg.update({f"sa_{k}": v for k, v in tokamax_knobs.items()})
  return types.SimpleNamespace(**cfg)


SPARK_TOKAMAX = dict(use_base2_exp=False, max_logit_const=0.0, num_stacked_q_heads=4,
                     q_layout="SEQ_MINOR", k_layout="SEQ_MINOR", v_layout="SEQ_MINOR")


def _spark_tokamax_knobs_available():
  """Whether the Tokamax patch reads use_base2_exp / max_logit_const."""
  from torchtitan.experiments.tpu.tpu_job_config import SplashAttentionKernelConfig

  fields = {f.name for f in dataclasses.fields(SplashAttentionKernelConfig)}
  return {"sa_use_base2_exp", "sa_max_logit_const"} <= fields


def _make_attend(path, window, cfg):
  if path in ("patch_stock", "patch_tokamax", "patch_tokamax_spark"):
    from torchtitan.experiments.tpu import workarounds
    from torchtitan.experiments.tpu.kernels import tokamax_splash_attention as tsa
    from torchtitan.experiments.tpu.muse_glimmer.attention import (
        SlidingWindowScaledDotProductInnerAttention,
    )

    parent = torch.nn.Module()
    parent.inner = SlidingWindowScaledDotProductInnerAttention.Config(window_size=window).build()
    if path == "patch_stock":
      workarounds.use_splash_attention_patch(parent, cfg)
    else:
      if _spark_tokamax_knobs_available():  # process-global; update() skips None
        tsa.TokamaxConfig.use_base2_exp = True
        tsa.TokamaxConfig.max_logit_const = None
      tsa.TokamaxConfig.num_stacked_q_heads = 1
      if path == "patch_tokamax_spark":
        cfg = _splash_config(512, cfg.sa_block_kv, cfg.sa_block_kv_compute, cfg.sa_block_q_dkv,
                             cfg.sa_block_kv_dkv, cfg.sa_block_kv_dkv_compute, **SPARK_TOKAMAX)
      workarounds.use_tokamax_splash_attention_patch(parent, cfg)

    def attend(q, k, v, positions, segment_ids):
      return parent.inner(q, k, v, scale=HEAD_DIM ** -0.5, enable_gqa=True, is_causal=True,
                          positions=positions)
    return attend
  if path == "sbd":
    from torchtitan.experiments.tpu.kernels import splash_bias_dropout as sbd

    sbd.SbdConfig.update(block_q=cfg.sa_block_q, block_kv=cfg.sa_block_kv,
                         block_kv_compute=cfg.sa_block_kv_compute,
                         block_q_dkv=cfg.sa_block_q_dkv, block_kv_dkv=cfg.sa_block_kv_dkv,
                         block_kv_dkv_compute=cfg.sa_block_kv_dkv_compute)

    def attend(q, k, v, positions, segment_ids):
      return sbd.sbd_splash_attention(q, k, v, segment_ids=segment_ids, is_causal=True,
                                      scale=HEAD_DIM ** -0.5,
                                      local_window_size=None if window is None else window - 1)
    return attend
  raise ValueError(path)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--seq", type=int, default=8192)
  ap.add_argument("--heads", type=int, default=128)
  ap.add_argument("--kv-heads", type=int, default=16)
  ap.add_argument("--docs", default="1880,1650,1420,2600,642")
  ap.add_argument("--blocks", default="1024:1024:512",
                  help="block_q:block_kv:block_kv_compute[:block_q_dkv:block_kv_dkv:block_kv_dkv_compute]")
  ap.add_argument("--paths", default="patch_stock,patch_tokamax,patch_tokamax_spark,sbd")
  ap.add_argument("--iters", type=int, default=10)
  ap.add_argument("--json", default="")
  a = ap.parse_args()

  docs = [int(x) for x in a.docs.split(",")]
  assert sum(docs) == a.seq, f"--docs must sum to --seq ({sum(docs)} != {a.seq})"
  dev = torch.device("tpu")
  positions = torch.cat([torch.arange(n) for n in docs])[None].to(dev)
  segment_ids = torch.cat([torch.full((n,), i, dtype=torch.int32)
                           for i, n in enumerate(docs)])[None].to(dev)
  g = torch.Generator().manual_seed(0)
  mk = lambda h: torch.randn(1, a.seq, h, HEAD_DIM, generator=g).to(torch.bfloat16).to(dev)
  q0, k0, v0, d_out = mk(a.heads), mk(a.kv_heads), mk(a.kv_heads), mk(a.heads)
  cfg = _splash_config(*map(int, a.blocks.split(":")))
  paths = a.paths.split(",")
  if "patch_tokamax_spark" in paths and not _spark_tokamax_knobs_available():
    print("skipping patch_tokamax_spark: SplashAttentionKernelConfig has no"
          " sa_use_base2_exp / sa_max_logit_const", flush=True)
    paths.remove("patch_tokamax_spark")
  rows = []
  for path in paths:
    for layer, window in (("sliding", WINDOW), ("full", None)):
      attend = _make_attend(path, window, cfg)
      q, k, v = (t.detach().clone().requires_grad_(True) for t in (q0, k0, v0))
      t_fwd, t_all = [], []
      for it in range(a.iters + 2):  # 2 warm-up iterations (compile)
        _sync()
        t0 = time.perf_counter()
        out = attend(q, k, v, positions, segment_ids)
        _sync()
        t1 = time.perf_counter()
        out.backward(d_out)
        _sync()
        t2 = time.perf_counter()
        q.grad = k.grad = v.grad = None
        if it >= 2:
          t_fwd.append(t1 - t0)
          t_all.append(t2 - t0)
      row = dict(path=path, layer=layer, seq=a.seq, docs=docs, blocks=a.blocks,
                 fwd_ms=round(statistics.median(t_fwd) * 1e3, 3),
                 fwd_bwd_ms=round(statistics.median(t_all) * 1e3, 3))
      rows.append(row)
      print(f"{path:14s} {layer:8s} fwd {row['fwd_ms']:8.3f} ms   fwd+bwd {row['fwd_bwd_ms']:8.3f} ms",
            flush=True)
  if a.json:
    with open(a.json, "w") as fh:
      json.dump(rows, fh, indent=1)


if __name__ == "__main__":
  main()
