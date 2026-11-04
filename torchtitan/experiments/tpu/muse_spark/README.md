# Muse Spark Open (text decoder) on TPU

A native TorchTitan port of the text decoder of
[`meta-models/Muse-Spark-Open-816B-A42B`](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B)
(Hugging Face Transformers `model_type="muse_spark"`, Transformers >= 5.18) for
the TPU lane. The vision and audio towers are not built; their checkpoint
tensors are ignored.

The implementation follows the public Transformers modeling code
(Apache-2.0) and the BSD-3 upstream TorchTitan port
(`torchtitan/models/muse_spark` in pytorch/torchtitan), adapted to this
fork's module APIs and to the TPU attention / MoE kernels. Once the fork
rebases onto an upstream that contains `torchtitan/models/muse_spark`, this
package should become a thin TPU wrapper around it, as `tpu/muse_glimmer` is.

## Architecture (816B-A42B)

| | |
|---|---|
| layers | 62, all MoE |
| hidden / vocab | 8192 / 202,048 (untied LM head) |
| attention | 128 query heads, 16 KV heads, head dim 64 |
| layer pattern | layer `i` is global, NoPE when `i % 4 == 1`; all others sliding window 2048 with RoPE (theta 5e5, rotate-half) |
| MoE | 256 routed experts, top-8, no shared experts; router on the 8192-d input; experts run at 4096 (pooled + gain-normalized input), SwiGLU with 4096 hidden |
| parameters | 819,990,536,192 text parameters (+ 62 routing-bias buffers), 42.8B active per token |

Differences from a Llama-style decoder, all in `model.py`:

* Gain-centered RMSNorms (`weight + 1`, checkpoint stores the zero-centered
  weight), computed in FP32; the final norm uses a plain weight.
* Token embeddings followed by a scaleless RMSNorm.
* Scaleless per-head Q/K RMSNorm, the query multiplied by
  `qk_scale_factor = 5.473` (on top of `1/sqrt(head_dim)`), a scaleless
  per-head RMSNorm on the attention output, and a sigmoid output gate
  (`gate_proj`) before `o_proj`.
* A scaleless RMSNorm after attention and an input-scaled RMSNorm
  (`rms_norm(x * (1 + w))`) after the FFN.
* Norm-preserving residual gates instead of residual additions:
  `alpha * x + beta * branch`, `beta = sigmoid(g / 0.3)`,
  `alpha = sqrt(max(sigmoid(-g / 0.3) * (1 + beta), 1e-3))`.
* Routing: FP32 `sigmoid(logits / sqrt(32))`; experts are selected on
  `score + e_score_correction_bias`, combined with the unbiased selected
  scores normalized to sum 1. Every routed expert output goes through one
  shared input-scaled RMSNorm before weighting.
* LM head: `20 * tanh(logits / sqrt(32) / 20)` (`SoftCappedLinear`).

## Usage

```python
from torchtitan.experiments.tpu.muse_spark import model_registry, MuseSparkArgs

spec = model_registry("816B-A42B", seq_len=8192)   # full model
spec = model_registry("816B-A42B-2L")              # first 2 layers (also -4L, -8L)
spec = model_registry("816B-A42B", num_layers=6)   # any truncation
spec = model_registry("816B-A42B", args=MuseSparkArgs.from_hf_config("<HF_ASSETS>"))
spec = model_registry("debugmodel")                # 4 layers, 8 experts, random init
spec = model_registry("tiny-hf")                   # trl-internal-testing/tiny-MuseSparkForConditionalGeneration
```

* `seq_len` sizes the per-layer RoPE tables; use the training context length.
* `tpu_moe_kernels=True` (default) applies `TpuTokenDispatcherConverter` and
  `TpuGroupedExpertsConverter`; they import `torch_tpu`, so pass
  `tpu_moe_kernels=False` on machines without it.
* `load_balance_coeff=None` (default) keeps the checkpoint's routing bias
  fixed, as fine-tuning should; set it to update the bias from token counts.
* `residual_dtype=None` (default) keeps the residual stream in the compute
  dtype, as the reference model does; `residual_dtype="float32"` with
  `branch_dtype="bfloat16"` keeps an FP32 stream.

### Checkpoint

Point `hf_assets_path` at a folder with `config.json`,
`model.safetensors.index.json`, the 65 shards and the tokenizer, and set
`--checkpoint.initial_load_in_hf`. `MuseSparkStateDictAdapter` maps
`model.language_model.*` and `lm_head.weight`; the packed
`experts.gate_up_proj [E, 2F, D]` (gate rows first) becomes `w1_EFD` / `w3_EFD`.
Both sides use rotate-half RoPE, so Q/K are not permuted. The adapter checks
the index against the model's keys, ignores the towers and the layers a
truncated flavor does not build, and rejects anything else.

## Parallelism

FSDP / HSDP (SimpleFSDP or FSDP2) with FSDP-sharded experts (expert
parallelism 1), as in the TPU Qwen3.5-35B-A3B recipes. The 202,048-row
vocabulary switches to dim-1 sharding when dim 0 is uneven (e.g. FSDP 128).

Expert parallelism (`--parallelism.expert_parallel_degree N` > 1) is not part
of this package yet: it needs `expert_parallel.py` (EP token dispatchers and
expert placement), which `MuseSparkModel.Config.update_from_config`,
`parallelize_muse_spark` and the state-dict adapter import only when EP or a
placement file is requested; without it they raise `NotImplementedError`. The
model already keeps the shared post-expert norm on the token-owning rank under
EP (`MuseSparkRoutedExperts`), and the adapter reads HF checkpoints into a
non-contiguous expert placement (`ExpertPermutingHFStorageReader`).

Not supported yet (raise `NotImplementedError`): tensor, context and pipeline
parallelism.

## Validation (CPU, FP32, Transformers eager reference)

```bash
# The package imports torch_tpu through tpu/muse_glimmer. On a host without
# torch_tpu, put the import stubs (tests/cpu_stubs) first on the path.
export PYTHONPATH=torchtitan/experiments/tpu/muse_spark/tests/cpu_stubs:$PWD
MUSE_SPARK_TINY_DIR=<tiny checkpoint dir> \
MUSE_SPARK_CKPT_META_DIR=<dir with config.json, index, *.safetensors.header.json> \
python -m pytest -q torchtitan/experiments/tpu/muse_spark/tests --ignore-glob='*_tpu.py'
# Real dimensions (~125 GB host RAM): MUSE_SPARK_REAL_DIMS=1
# SimpleFSDP: torchrun --nproc_per_node=2 -m torchtitan.experiments.tpu.muse_spark.tests.fsdp_cpu_check
```

| check | result |
|---|---|
| random 4-layer config (sliding/full x RoPE/NoPE, a dense layer, 16 experts top-4, packed documents; reference experts eager and grouped_mm) | max abs logit diff 2.5e-5 (logits up to 7.0); worst relative gradient diff 2.2e-5 over all 67 parameters |
| public tiny checkpoint | max abs logit diff 1.4e-8 (logits up to 0.022); worst relative gradient diff 1.1e-6 |
| first two released layers at real dimensions (256 experts, real vocabulary; experts, embedding and head frozen to bound memory) | max abs logit diff 2.9e-5 (logits up to 7.4); losses equal; worst relative gradient diff 1.2e-5 over the other 31 parameters |
| released checkpoint metadata (opt-in, `MUSE_SPARK_CKPT_META_DIR`) | all 1,119 text tensor names match the index and every shape matches the shard headers (measured with the original port; not re-run for this change); the text parameter count, 819,990,552,064 elements with the routing biases, is checked on every run |
| TPU MoE converters (Pallas off) | logits within 1e-5 and gradients within 1e-4 (relative) of the plain BF16-expert model |
| SimpleFSDP, 2 ranks | gradients identical to a single-process reference (worst relative diff 0 over 70 parameters); negative control 0.89 |

An FP64 run leaves the same ~2e-5 gap: the reference computes the attention
softmax and router logits in FP32, so the FP32 gap is rounding, not a math
difference.

## Open work

* Expert parallelism (`expert_parallel.py`, see Parallelism) and TP for the
  128-head attention.
* `bld_layout` (the TPU `[B, L]` block layout) does not apply to these blocks.
* Under BF16 mixed precision the router weight and residual gates are cast to
  BF16 for compute; the reference keeps them in FP32. Keeping them in FP32
  under SimpleFSDP would tighten routing parity with FP32-router generators.
