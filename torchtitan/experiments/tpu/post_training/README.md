# Post-training (chat SFT) for TorchTitan on TPU

This package is a model-agnostic layer for supervised fine-tuning (SFT) of chat
models with native TorchTitan on TPU. It plugs into any TorchTitan `ModelSpec`:
the Muse Spark Open recipes use the Muse Spark TPU package (`../muse_spark/`),
and the CPU checks use the TPU Qwen3.5 debug model (`../qwen3_5/`).

| Piece | File | What it does |
|---|---|---|
| Chat rendering | `chat_template.py` | The checkpoint's own chat template, pinned `current_date`, optional `reasoning_effort`, assistant-only token masks, final-message extraction from sampled replies |
| SFT data | `sft_data.py` | `ChatSFTDataLoader`: packed rows with document boundaries, assistant-only labels, `num_valid_tokens` for the global token denominator, resumable, per-microbatch fingerprints |
| SFT trainer | `sft_train.py`, `sft_config.py` | `SFTTPUTrainer`: restores the chunked loss, LoRA adapter saves (PEFT export plus adapter-only resume state), held-out loss, per-step records, resume-check hooks |
| LoRA | `lora.py` | TorchTitan `LoRAConverter` on attention q/k/v/o by full module path; LoRA-aware state-dict adapter (load skips adapters, export merges them); merge and PEFT export helpers, including fused `wqkv` |
| Optimizers | `optimizers.py` | AdamW, `AdamW8bit` (8-bit moments) and `AdamWBf16` (bf16 weights, stochastic rounding) |
| Recipes | `config_registry.py` | `build_sft_config`, the helpers other recipe modules build on, and the registered Muse Spark LoRA SFT recipe |

## Model contract

A model plugs in through `resolve_model_spec(name, flavor, seq_len)`: the
built-in name `muse_spark`, or `module.path:function` for any registry with the
signature `function(flavor, seq_len=..., **kwargs)` (the CPU checks use
`torchtitan.experiments.tpu.qwen3_5:model_registry`). The recipes rely on these
properties of the `ModelSpec`:

1. `preprocess_inputs` derives document masks from `positions`, which restart
   at 0 at every packed document. Each document lies inside one
   `max_context_length` row.
2. The model has an `lm_head` module that produces the logits, including any
   output scaling or soft-capping (Muse Spark:
   `20 * tanh(z * output_multiplier / 20)`). `ChunkedLossWrapper` runs the model
   with `_skip_lm_head=True` and applies `lm_head` per chunk, so a logit
   transform applied outside `lm_head` would change the loss.
3. LoRA targets are `Linear` configs under an `attention`, `self_attn` or `attn`
   module named `wq`/`wk`/`wv`/`wkv`/`wo` (TorchTitan) or
   `q_proj`/`k_proj`/`v_proj`/`o_proj` (Hugging Face). A fused `wqkv`
   (`FusedQKVLinear`) is supported for merge and PEFT export.
4. The state-dict adapter maps TorchTitan keys to Hugging Face keys. Strict
   adapters that check a checkpoint against the model's keys must keep that
   check in `expected_model_shapes` (the LoRA wrapper removes adapter keys from
   it). `to_hf_for_export`, when present, is used for merged-weight export.

## SFT

Records are JSONL or parquet rows with `messages` (system, user, assistant and
tool turns). Each conversation is rendered once by the checkpoint's chat
template. With `mask_mode="auto"` the loader uses the template's
`{% generation %}` blocks (Hugging Face `return_assistant_tokens_mask`),
otherwise it checks that each assistant turn's prompt is a token prefix of the
conversation. The generation prompt that opens an assistant turn
(`<|start|>assistant` for Muse Spark) is not trained by default, because at
inference the prompt supplies it. Targets are shifted
(`labels[t] = input[t + 1]` where that token is an assistant token); document
ends and padding are ignored. The chat template is read with Hugging Face
`transformers` from `--dataloader.hf-tokenizer-path`, by default the trainer's
`--hf_assets_path`.

Documents are packed first-fit decreasing into fixed rows, at most
`max_documents_per_row` per row. Positions restart per document and the tail
pad continues the last document's positions. The loss is the summed
cross-entropy over supervised tokens divided by the step's global count of
supervised tokens (all microbatches and data-parallel ranks). Data-parallel
ranks read disjoint strided shards of one shuffled epoch order; the dataloader
state (epoch, cursor, pending documents) makes a resumed run serve exactly the
batches it would have served.

This loader complements the single-turn chat datasets of `TPUGrainDataLoader`
(`--dataloader.dataset gsm8k|chat_json`, upstream `ChatProcessor` masks): it
takes multi-turn and tool conversations, renders them with the checkpoint's own
template (including template options such as `reasoning_effort`) and trains
every assistant turn.

### LoRA saves and resume

`--sft.lora-save` selects what a LoRA run writes at each checkpoint step
(`--checkpoint.interval` and the last step):

* `adapter` (default): TorchTitan's DCP save of the adapter-only resume state
  (adapter tensors, their optimizer state, the LR schedule, the dataloader
  position and the step; never the frozen base) to
  `<dump_folder>/checkpoint/step-N`, plus the adapter in the PEFT layout
  (`adapter_model.safetensors` + `adapter_config.json`, module names from the
  model's state-dict adapter) in `<dump_folder>/lora_adapter/step-N`. Every
  start reads the base from the Hugging Face checkpoint and then restores the
  newest complete `step-N`, so relaunching with the same dump folder resumes a
  killed run. The base is the released checkpoint, so a full save would rewrite
  about 1.6 TB at 816B. Needs the adapter-only checkpoint support of
  `../checkpointing/lora_checkpoint.py`.
* `peft`: only the PEFT adapter, no resume state.
* `full`: TorchTitan's DCP save of the whole model and training state.
* `none`: no saves.

Runs without LoRA always use TorchTitan's saves. On TPU every rank leaves each
save together (`../checkpointing/flush.py`; `SFT_CHECKPOINT_FLUSH=1` forces it
on any device, `0` turns it off).

Resume checks: with `SFT_RECORDS_ALL_RANKS=1` every rank appends one JSON line
per step to `<dump_folder>/sft-records/rank-NNNNN.jsonl` (its loss share, the
gradient norm, the learning rate, and a SHA-256 fingerprint plus the document
ids of every microbatch it trained on), so two runs can be compared batch by
batch (`SFT_RECORDS_DIR` moves them). `SFT_FAULT_KILL_STEP=N` sends SIGKILL to
every rank inside the forward/backward of step N, after that step's batches
were drawn.

### Held-out loss

`--sft.val-data-path` (records in the training layout) adds a held-out loss:
before the first step, every `--sft.val-every` steps (default 10; 0: only
before the first and after the last step) and after the last step, a
forward-only pass over every held-out record (packed like the training data,
each data-parallel rank a strided share; ranks with fewer microbatches add
all-padding ones) gives the summed cross-entropy of the assistant tokens
divided by their count. It is logged as `validate step: N loss: X` and written
by rank 0 to `<dump_folder>/sft-val-rank0.json`. `--sft.val-max-microbatches`
caps the microbatches per rank and pass.

### Launch

```bash
torchrun --nproc_per_node=<RANKS_PER_HOST> -m torchtitan.experiments.tpu.post_training.sft_train \
  --module torchtitan.experiments.tpu.post_training.config_registry \
  --config muse_spark_816b_a42b_sft_lora_tpu7x \
  --hf_assets_path <MODEL_DIR> --dataloader.data-path <SFT_JSONL> --dump_folder <OUT_DIR>
```

Registered SFT recipe (it needs the Muse Spark TPU package):
`muse_spark_816b_a42b_sft_lora_tpu7x` (LoRA r 16, alpha 32 on q/k/v/o over a
frozen bf16 base, AdamW on the adapters, lr 1e-4 with warmup and cosine
decay). Set the flavor with `SPARK_FLAVOR` (default `816B-A42B`; truncated
flavors such as `816B-A42B-2L` for small slices), the expert-parallel degree
with `SPARK_EP`, and `SPARK_COMPILE=1` to compile per block.

The Spark recipe applies TPU7x settings measured with the 816B model on one
4x4x4 slice (`spark_tpu_settings`): full activation checkpointing, reshard
after forward, `DEFER_AND_FUSE`, compiled gradient norm, the shared kernel
settings of `muse_spark/tpu7x.py`, the TPU topology mesh, and the Pallas fused
linear cross-entropy with the head's multiplier and soft cap
(`FusedLossLinearWrapper` + `PallasCrossEntropyLoss`).

Full-parameter SFT of the same model (`LowPrecisionAdamW`, fp32 gradient
reduction, expert parallelism with a measured expert placement) has its own
recipes in `full_param_sft.py`
(`--module torchtitan.experiments.tpu.post_training.full_param_sft`; its module
docstring lists them).

## Optimizers

| Name | Weights | Moments | Bytes per parameter (weights + grads + moments) |
|---|---|---|---|
| `AdamW` with `training.dtype=float32` | fp32 | fp32 | 16 |
| `AdamW` with `training.dtype=bfloat16` | bf16 | bf16 | 8 |
| `AdamWBf16` with `training.dtype=bfloat16` | bf16, stochastic rounding | bf16 | 8 |
| `AdamW8bit` with `training.dtype=bfloat16` | bf16 | 8-bit block-wise | 6 |

At 816B parameters, fp32 masters need about 13 TB, beyond one 64-chip TPU7x
slice; the bf16 variants fit. The low-bit variants run torchao's per-parameter
AdamW update; select `lowbit_compile_backend` `""` (eager), `"tpu"` or
`"inductor"`. `AdamWBf16` and `AdamW8bit` are verified on CPU only.

## CPU checks

```bash
export SPARK_TEST_ASSETS=<public tiny Muse Spark snapshot>   # tokenizer + chat template
bash torchtitan/experiments/tpu/post_training/tests/run_cpu.sh --tokenizer-dir "$SPARK_TEST_ASSETS" --work <new dir>
```

The runner covers the unit tests (chat masks, packing, LoRA, optimizers,
recipes, held-out loss) and end-to-end runs of the real trainer on CPU: the SFT
smoke (full, LoRA with full and adapter saves, AdamW8bit, AdamWBf16) and, with
the adapter-only checkpoint support of `../checkpointing/` installed, a LoRA
kill-and-resume check on two ranks. With the Muse Spark TPU package installed
and the tiny checkpoint's safetensors in the snapshot, the smoke and the
held-out loss test add Muse Spark runs. Without TorchTPU, the runner puts the
shared import-only stand-in `../tests/cpu_stubs/` on `PYTHONPATH`. Each process
uses `OMP_NUM_THREADS` threads (default 8).

## Sources

* LoRA: Hu et al., arXiv:2106.09685.
* 8-bit optimizers: Dettmers et al., arXiv:2110.02861.
