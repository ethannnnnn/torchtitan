# Reproducing Muse Spark Open LoRA SFT on TPU7x

Every site-specific value is a placeholder: `<GCS_BUCKET>`, `<GCP_PROJECT>`,
`<K8S_NAMESPACE>`, `<LOCAL_QUEUE>`, `<PRIORITY_CLASS>`, `<TRAINER_IMAGE>`,
`<KUBE_CONTEXT>`, `<RUN_NAME>`, `<BASE_COMMIT>`. Commands run from the
repository root unless noted. The launch kit is described in
[`../launch/README.md`](../launch/README.md).

## 0. What you need

- A GKE cluster with TPU7x capacity: 4 chips (`2x2x1`) for the two-layer
  smoke and 64 chips (`4x4x4`, 16 hosts, 128 ranks) for the full 62-layer
  model. With dynamic slice provisioning, 64-chip slices came up in under a
  minute while some small slices were never provisioned.
- JobSet, Kueue, and the GCSFuse CSI driver; a Kubernetes service account that
  can read your model, data, and runtime prefixes and write the run prefix.
  The pods need outbound HTTPS: they fetch the base commit from the public
  repository and, with `--pip-overlay`, wheels from PyPI.
- A Hugging Face account that accepted the Muse Model License, with your own token.
- Local tools: Python 3.12+, `git`, `gcloud`, `kubectl`, `huggingface_hub`.

## 1. CPU checks

Use a CPU environment with Python 3.12+, torch (CPU wheel), transformers 5.18
or newer, safetensors, numpy, jinja2, and pytest, plus a local copy of the
public tiny checkpoint:

```bash
hf download trl-internal-testing/tiny-MuseSparkForConditionalGeneration \
  --revision c9e83e8784d89e7e7f66f78e15adf10e3a93fdbb --local-dir ./tiny-muse-spark
bash torchtitan/experiments/tpu/spark_e2e/tests/run_cpu.sh --tiny-checkpoint ./tiny-muse-spark
```

This builds every SFT and step recipe and checks its settings, checks the
PEFT export and the truncated-checkpoint builder, and runs the launch kit
end to end on a local repository (overlay, render, pod bootstrap, offline
runtime install, runner). With the tiny checkpoint it also trains LoRA SFT
on CPU through the same entry point as the TPU stages (two ranks, LoRA-only
checkpoints, PEFT exports, held-out loss, adapter initialization) and checks
that a run killed mid-training and relaunched continues bit for bit like an
uninterrupted one. Without TorchTPU the runner imports the TPU modules through
the CPU stand-in in `spark_e2e/tests/tpu_import_stubs`.

## 2. Trainer runtime

The trainer needs Python with the TorchTPU runtime (torch CPU wheel,
torch_tpu, libtpu, JAX, Tokamax, transformers, safetensors) and `git`. Either
bake it into `<TRAINER_IMAGE>`, or mount a digest-pinned wheelhouse and use a
stock Python image (`--runtime-dir` and the `--runtime-*-sha256` pins; see the
launch README). `--pip-overlay
torchtitan/experiments/tpu/muse_spark/launch/pip-pins-jax0.12.0.1-tokamax0.0.17.txt`
moves either one to JAX and jaxlib 0.12.0.1 with Tokamax 0.0.17, the versions
`torchtitan/experiments/tpu/requirements.txt` pins.

## 3. Stage the checkpoint

```bash
export HF_TOKEN=...    # your own token
bash torchtitan/experiments/tpu/muse_spark/launch/stage_checkpoint.sh \
  --revision 0fb02478b02565b2ce7fc57e6051c91172774ac9 \
  --dest gs://<GCS_BUCKET>/muse-spark/model/Muse-Spark-Open-816B-A42B
```

About 1.7 TB, verified file by file against the Hub's SHA-256. Run it on a VM
or pod in the bucket's region; it needs local disk for one 26 GB shard at a
time. The two-layer recipe reads only the first two layers of the same
checkpoint; `spark_e2e/build_trunc.py --src <checkpoint> --out <dir> --layers 2`
writes a truncated copy if you prefer a small one.

## 4. Data

The SFT recipes read chat records, one JSON object per line with a
`messages` list (`system` / `user` / `assistant` turns), from
`/mnt/data/sft/train.jsonl` (the `--data-dir` mount; override with
`--dataloader.data-path`). The loss covers the assistant tokens only; rows are
packed to 8,192 tokens with document boundaries; the Muse chat template renders
with `SPARK_REASONING` (default `low`). Optionally put a held-out set in the
same layout next to it (for example `sft/val.jsonl`) and pass
`--env SPARK_VAL_DATA=/mnt/data/sft/val.jsonl`.

```bash
gcloud storage cp train.jsonl val.jsonl gs://<GCS_BUCKET>/muse-spark/data/<DATASET>/sft/
```

## 5. Code

Commit your work, then pack what the candidate changes relative to a public
base commit that your branch starts from:

```bash
python torchtitan/experiments/tpu/muse_spark/launch/build_overlay.py \
  --repo . --base <BASE_COMMIT> --candidate HEAD --out ./overlay
```

Nothing is uploaded: the overlay travels in the rendered manifest, and every
pod checks that base commit + overlay is exactly the candidate's git tree
before it starts.

## 6. LoRA SFT

The staged recipes in [`../../spark_e2e/recipes.py`](../../spark_e2e/recipes.py)
train LoRA rank 16 / alpha 32 on q/k/v/o with a frozen bf16 base, keep
LoRA-only checkpoints (adapters, their optimizer state, LR schedule,
dataloader position and step) and export the adapter in the PEFT layout at
every checkpoint and at the last step. Kernels: Tokamax Splash attention with
the Spark blocks, Tokamax GMM-v2 experts, libtpu power state 7
([`../tpu7x.py`](../tpu7x.py)); expert parallelism 1.

Common arguments (fill in your values):

```bash
SITE=(--bucket <GCS_BUCKET> --namespace <K8S_NAMESPACE> --local-queue <LOCAL_QUEUE>
      --project <GCP_PROJECT> --trainer-image <TRAINER_IMAGE> --priority-class <PRIORITY_CLASS>
      --model-dir muse-spark/model/Muse-Spark-Open-816B-A42B --data-dir muse-spark/data/<DATASET>
      --code-overlay ./overlay
      --pip-overlay torchtitan/experiments/tpu/muse_spark/launch/pip-pins-jax0.12.0.1-tokamax0.0.17.txt)
```

### Two-layer smoke (`s1_sft_trunc2`, one 4-chip host)

```bash
python torchtitan/experiments/tpu/muse_spark/launch/render.py "${SITE[@]}" \
  --mode e2e --run <RUN_NAME> --topology 2x2x1 --config s1_sft_trunc2 \
  --env SPARK_STEPS=5 --out ./rendered-2l
kubectl --context <KUBE_CONTEXT> -n <K8S_NAMESPACE> create -f ./rendered-2l/trainer.json
```

### Full model (`s2_sft_full`, one `4x4x4` slice)

```bash
python torchtitan/experiments/tpu/muse_spark/launch/render.py "${SITE[@]}" \
  --mode e2e --run <RUN_NAME> --topology 4x4x4 --config s2_sft_full \
  --env SPARK_STEPS=60 --env SPARK_VAL_DATA=/mnt/data/sft/val.jsonl --env SPARK_VAL_EVERY=10 \
  --extra-args='--dataloader.data-path=/mnt/data/sft/train.jsonl --checkpoint.interval=20' \
  --budget-seconds 4200 --deadline-seconds 4500 --out ./rendered-sft
kubectl --context <KUBE_CONTEXT> -n <K8S_NAMESPACE> create -f ./rendered-sft/trainer.json
```

Watch host 0's log (`kubectl logs` of pod `<RUN_NAME>-trainer-0-0-*`, or
`runs/<RUN_NAME>/stdout-host0.log`) for `[SPARK:CODE] ... = candidate tree`,
`[SPARK:PIP]`, `[SPARK:IMPORTS] N of N modules imported`, `[SPARK:VERSIONS]`,
`Finished loading the checkpoint`, the `step: N loss: ...` lines and the
`[sft] validation step=N loss=...` lines. Outputs land under
`gs://<GCS_BUCKET>/muse-spark/runs/<RUN_NAME>/`: `checkpoint/step-N`
(LoRA-only DCP), `adapter/step-N` (PEFT), `sft-val-rank0.json`,
`sft-records-rank0.json`, and per host `stdout-hostN.log`,
`versions-hostN.json` (package versions and the code tree),
`pip-freeze-hostN.txt`, `exit_code-hostN`.

A relaunch with the same `--run` and prefix resumes: every start reads the
base from the Hugging Face checkpoint, then the newest complete
`checkpoint/step-N`. To start from an adapter instead (for example an earlier
run's export copied under the data prefix), pass
`--env SPARK_ADAPTER_INIT=/mnt/data/<ADAPTER_DIR>`.

The other knobs (`SPARK_LR`, `SPARK_LORA_RANK`, `SPARK_SFT_ROWS` for two
packed rows per rank, `SPARK_EXPORT_EVERY`, ...) are listed in the
`spark_e2e/recipes.py` docstring; set them with `--env`.

Measured with exactly this full-model launch on one `4x4x4` slice (released
weights, JAX 0.11.1 runtime, one 8,192-token row per rank, 1,048,576 tokens
per step): checkpoint load 208 s, step 1 23 s, then 14.6 to 14.8 s per step
(about 1,117 tokens/s/chip); the steps right after a checkpoint (21, 41) took
longer because of the save and the PEFT export. A first step on a cold
compilation cache takes several minutes.

### Merged Hugging Face export

To merge a PEFT adapter into the released checkpoint's layout (for engines
without LoRA support), copy the adapter under the data prefix and render
`--mode export --config <ADAPTER_DIR>` (one export worker per host; host 0
writes the index, the small files, `export_manifest.json` with per-shard
SHA-256, and `EXPORT_OK` under `runs/<RUN_NAME>/hf/`).

### Full-model step recipes and expert parallelism

[`../spark_64chip_recipes.py`](../spark_64chip_recipes.py) measures memory and
step time on any text in the `c4_test` JSON-lines format
(`--dataloader.dataset_path`; synthetic text is generated when none is given):
`--mode step --config spark_816b_lora_64chip` (LoRA), `spark_816b_lora_64chip_16k`
(two rows per rank), `spark_816b_full_64chip` (full parameter, plain bf16
AdamW, for memory and speed only), and random-weight variants. They load the
checkpoint with a parallel positional reader that also shares the
vocabulary-sharded tensors through `/dev/shm`, one reader per host.
`SPARK_CKPT_IO_THREADS=N` uses the multi-threaded DCP writer and parallel
reader; `TT_FAULT_KILL_STEP` / `TT_BATCH_RECORDS_DIR` enable the resume-check
hooks of `checkpointing/resume_check.py`.

Expert parallelism 8 with the released weights needs an expert placement
measured on your training text (routing is skewed):

```bash
# 1. EP 1, a few steps: writes a load-balanced EP-8 placement at step 2
python torchtitan/experiments/tpu/muse_spark/launch/render.py "${SITE[@]}" --mode step \
  --config spark_816b_lora_64chip_measure_placement --run <RUN_NAME>-place --topology 4x4x4 \
  --env SPARK_EXPERT_PLACEMENT_OUT=/mnt/out/placement_ep8.json \
  --extra-args='--dataloader.dataset_path=/mnt/data/<TEXT>.json --training.steps=3' --out ./rendered-place
# 2. EP 8 with that placement (copy placement_ep8.json from the first run's output to the data prefix)
python torchtitan/experiments/tpu/muse_spark/launch/render.py "${SITE[@]}" --mode step \
  --config spark_816b_lora_64chip_ep8 --run <RUN_NAME>-ep8 --topology 4x4x4 \
  --env SPARK_EXPERT_PLACEMENT_FILE=/mnt/data/placement_ep8.json \
  --extra-args='--dataloader.dataset_path=/mnt/data/<TEXT>.json' --out ./rendered-ep8
```

Watch the `[EP] ... max_buffer_fill=... max_dropped_frac=...` lines (a fill
above 1 means routes were dropped on that rank).

## 7. Collect results and clean up

Capture every pod's logs (including previous restarts) and `kubectl describe`
output before deleting anything, then delete the JobSet and its code
ConfigMaps:

```bash
kubectl --context <KUBE_CONTEXT> -n <K8S_NAMESPACE> delete jobset <RUN_NAME>
kubectl --context <KUBE_CONTEXT> -n <K8S_NAMESPACE> delete configmap -l muse-spark-run=<RUN_NAME>
```
