# Launching Muse Spark recipes on TPU7x (GKE)

These files turn a recipe name into a reviewed, suspended JobSet. Nothing here
contacts a cluster or a bucket except `stage_checkpoint.sh`, which you run
yourself with your own credentials. Every site-specific value (bucket,
project, namespace, queue, image, storage directories) is a command-line
argument; nothing site-specific is stored in these files.

| File | Purpose |
|---|---|
| `stage_checkpoint.sh` | Copy the public checkpoint from Hugging Face to `gs://<GCS_BUCKET>/...`, one shard at a time, verified against the Hub's SHA-256 |
| `build_overlay.py` | The code the pods run: the candidate commit's changed files relative to a public base commit, as a deterministic archive plus a manifest (commits, trees, git modes, blob ids) |
| `render.py` | Renders `trainer_jobset.json.in` for a slice shape and a recipe, with the overlay in ConfigMaps; rejects unresolved placeholders |
| `trainer_jobset.json.in` | One multi-host trainer JobSet: four chips and eight ranks per host, the overlay ConfigMaps, GCSFuse mounts for runtime, model, data, and outputs |
| `bootstrap.sh` | Container entry point: checks the slice wiring, fetches the base commit, applies the overlay, and starts only if the git tree equals the candidate's |
| `spark_pod.sh` | In-pod runner: optional offline runtime install, optional hash-pinned wheel overlay, version record and import check, `torch.distributed.run` (or the merged-export workers) |
| `runtime_install.sh` | Offline install of a mounted, digest-pinned wheelhouse into a fresh virtual environment |
| `pip-pins-*.txt` | Hash-pinned wheels for `--pip-overlay` (JAX/jaxlib 0.12.0.1 with Tokamax 0.0.17, or Tokamax 0.0.17 alone) |

## Cluster prerequisites

A GKE cluster with TPU7x node pools or dynamic slice provisioning, the
JobSet and Kueue controllers, the GCSFuse CSI driver, and a Kubernetes service
account with read access to the runtime, model, and data prefixes and write
access to the run prefix. The pods fetch the base commit from the public
repository (`--base-repo`, default `https://github.com/google-pytorch/torchtitan.git`)
and, with `--pip-overlay` and no `--pip-find-links`, wheels from PyPI, so
they need outbound HTTPS. The template carries the labels used by dynamic
slice provisioning (`x.tpu.cloud.google.com/slice-autoprovisioning`,
`x.tpu.google.com/inject-slice-selector`) and
`cloud.google.com/gke-accelerator-topology-mode: PROVISION_ONLY`; remove them
if your cluster uses static TPU node pools.

## Storage layout

```
gs://<GCS_BUCKET>/muse-spark/
  runtime/                              # optional mounted wheelhouse (runtime_install.sh)
  model/Muse-Spark-Open-816B-A42B/      # from stage_checkpoint.sh
  data/shopping-ctr-v1/                 # SFT records: sft/train.jsonl, sft/val.jsonl
  runs/<RUN_NAME>/                      # outputs: logs, versions, checkpoints, adapters
```

No code goes to the bucket.

## Inline code delivery

`build_overlay.py` compares the candidate commit (your branch) with a public
base commit and packs every file the candidate adds or changes; deletions are
listed in the manifest. `render.py` splits the archive into ConfigMaps of at
most 700 KiB (each stays under the 1 MiB ConfigMap limit; the overlay may be
up to 3 MiB compressed) that every trainer pod mounts at `/mnt/overlay`. In
each pod `bootstrap.sh`:

1. fetches the base commit (`git fetch --depth=1`) and checks its commit and
   tree ids;
2. reassembles the archive and checks its SHA-256;
3. applies the deletions and the files, checking each file's path, git mode,
   and blob id against the manifest;
4. stages the changed paths and compares `git write-tree` with the
   candidate's tree, and stops unless they are equal;
5. starts `spark_pod.sh` from that tree.

So the pods run exactly the candidate's tree, and the rendered manifest
records which tree that is (`receipt.json`, JobSet annotations, and the
`versions-hostN.json` of every host).

## Render and launch

```bash
python torchtitan/experiments/tpu/muse_spark/launch/build_overlay.py \
  --repo . --base <BASE_COMMIT> --candidate HEAD --out ./overlay
python torchtitan/experiments/tpu/muse_spark/launch/render.py --mode e2e --run <RUN_NAME> \
  --topology 4x4x4 --config s2_sft_full --code-overlay ./overlay \
  --bucket <GCS_BUCKET> --namespace <K8S_NAMESPACE> --local-queue <LOCAL_QUEUE> \
  --project <GCP_PROJECT> --trainer-image <TRAINER_IMAGE> \
  --pip-overlay torchtitan/experiments/tpu/muse_spark/launch/pip-pins-jax0.12.0.1-tokamax0.0.17.txt \
  --env SPARK_STEPS=20 --out ./rendered
kubectl --context <KUBE_CONTEXT> -n <K8S_NAMESPACE> create -f ./rendered/trainer.json
```

`<BASE_COMMIT>` must be a commit the pods can fetch from `--base-repo` (for
example the upstream commit your branch starts from). `trainer.json` is a
`v1` List: the code ConfigMaps (`<RUN_NAME>-code-N`) followed by the JobSet,
which stays suspended until the queue admits it. Delete the ConfigMaps
together with the JobSet: `kubectl delete jobset <RUN_NAME>` and
`kubectl delete configmap -l muse-spark-run=<RUN_NAME>`.

`--mode` picks the entry point and the TorchTitan `--module`; `--entry` and
`--config-module` override them.

| Mode | Entry | Configs |
|---|---|---|
| `train` | `torchtitan.experiments.tpu.train` | `muse_spark/config_registry.py` (debug, tiny checkpoint, first two layers) |
| `step` | `torchtitan.experiments.tpu.train` | `muse_spark/spark_64chip_recipes.py` (full model, one 64-chip slice) |
| `sft` | `post_training.sft_train` | `post_training/config_registry.py` (model-agnostic chat SFT recipes) |
| `e2e` | `spark_e2e.train` | `spark_e2e/recipes.py` (LoRA SFT: `s1_sft_trunc2` on small slices, `s2_sft_full` on one 4x4x4) |
| `export` | `checkpointing.hf_export` | `--config` names a PEFT adapter directory under the data mount; merges it into the released checkpoint, one export worker per host |

`--priority-class` sets the pod priority class when your operator requires
one, `--label KEY=VALUE` adds labels to the JobSet, its pods and the
ConfigMaps, and `--env KEY=VALUE` adds a trainer environment variable such as
the `SPARK_*` knobs of `spark_e2e/recipes.py` (variables the kit sets itself
and names that look like credentials are rejected; use a Kubernetes Secret
for tokens). TPU node boot disks often have only about 20 GB free, so the
`/work` scratch volume defaults to 15 Gi (`--work-size`). The pod budget
(`--budget-seconds`, default 2,400) must be shorter than the Job deadline
(`--deadline-seconds`, default 2,700). Logs, `versions-hostN.json`,
`pip-freeze-hostN.txt`, and `exit_code-hostN` land under `runs/<RUN_NAME>/`.
Collect pod logs and `kubectl describe` output before deleting a JobSet.

Before the trainer starts, every host imports `torch_tpu`, `jax`, `tokamax`,
`torchtitan.experiments.tpu.muse_spark`, the config module and the entry
module once, in one process that touches no device (the export module in
export mode), and records the result in `versions-hostN.json`; a failed
import stops the pod with `[SPARK:GATE]` instead of a traceback from every
rank. `--env SPARK_IMPORT_MODULES='...'` changes the list,
`--env SPARK_IMPORT_TIMEOUT_S=N` bounds the check (default 600 s) and
`--env SPARK_IMPORT_CHECK=0` skips it.

## Trainer runtime

The trainer image needs Python 3.12 or newer with the TorchTPU runtime (torch
CPU wheel, torch_tpu, libtpu, JAX, transformers, safetensors) and `git`
(`bootstrap.sh` installs `git` with `apt-get` when it is missing). Three ways
to provide the Python packages:

* Bake them into `--trainer-image` (`--runtime-mode preinstalled`).
* Mount a wheelhouse at `gs://<GCS_BUCKET>/<runtime-dir>/` and use a stock
  Python image: with `--runtime-mode mounted` (or `auto`, when the wheelhouse
  has `<runtime-name>-lock.txt`) `runtime_install.sh` checks the lock file and
  the digest lists against `--runtime-lock-sha256`, `--runtime-wheels-sha256`
  and `--runtime-debs-sha256`, copies and verifies every file, installs the
  system packages and the wheels offline into a fresh virtual environment,
  and refuses an installed set that differs from the lock.
* Either way, `--pip-overlay FILE` installs a hash-pinned requirements file
  from the repository on top, with `--no-deps --require-hashes
  --only-binary=:all:` (from PyPI, `--pip-index-url`, or offline from
  `--pip-find-links`), then records `pip check` and `pip freeze`.
  `pip-pins-jax0.12.0.1-tokamax0.0.17.txt` brings JAX and jaxlib 0.12.0.1 and
  Tokamax 0.0.17, the versions `torchtitan/experiments/tpu/requirements.txt`
  pins.

The recipes were brought up on Python 3.14, torch 2.15.0 (CPU wheel),
torch_tpu 0.2.0, libtpu 0.0.50, JAX 0.11.1, Tokamax 0.0.17, and transformers
5.20.2. torchtitan's `requirements.txt` pins JAX 0.12.0.1;
`pip-pins-jax0.12.0.1-tokamax0.0.17.txt` moves such a runtime to it.
