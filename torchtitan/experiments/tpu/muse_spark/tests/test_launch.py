# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU checks for the launch kit: overlay builder, renderer, pod bootstrap, runtime installer, runner.

Needs git, bash and coreutils (``sha256sum``); the bootstrap, runtime and
runner tests also need bash >= 4.4. Nothing contacts a cluster or the network:
the bootstrap test fetches its base commit from a local repository.
"""
from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

LAUNCH = Path(__file__).resolve().parents[1] / "launch"
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(LAUNCH))
import build_overlay  # noqa: E402
import render  # noqa: E402

BASE_ARGS = [
    "--bucket", "example-bucket", "--namespace", "example-ns", "--local-queue", "example-queue",
    "--project", "example-project", "--trainer-image", "registry.example/trainer@sha256:" + "a" * 64,
    "--data-dir", "example/data/chat-sft",
]
PINS = "torchtitan/experiments/tpu/muse_spark/launch/pip-pins-jax0.12.0.1-tokamax0.0.17.txt"
HAVE_GIT = shutil.which("git") is not None
HAVE_SHA256SUM = shutil.which("sha256sum") is not None


def _bash_ok() -> bool:
    if shutil.which("bash") is None:
        return False
    out = subprocess.run(["bash", "-c", "echo ${BASH_VERSINFO[0]}.${BASH_VERSINFO[1]}"],
                         capture_output=True, text=True).stdout.strip()
    major, minor = (int(x) for x in out.split("."))
    return (major, minor) >= (4, 4)


HAVE_BASH = _bash_ok()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com",
                           "-c", "commit.gpgsign=false", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


def make_repo(root: Path) -> tuple[Path, str, str]:
    """A repository with a base commit and a candidate commit that adds, changes, deletes,
    changes a mode, adds a symlink and replaces a directory with a file."""
    repo = root / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "keep.txt").write_text("unchanged\n")
    (repo / "change.py").write_text("x = 1\n")
    (repo / "drop.txt").write_text("deleted later\n")
    (repo / "mode.sh").write_text("echo hi\n")
    (repo / "dir").mkdir()
    (repo / "dir" / "inner.txt").write_text("inner\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")
    base = git(repo, "rev-parse", "HEAD")
    (repo / "change.py").write_text("x = 2\n")
    (repo / "drop.txt").unlink()
    (repo / "mode.sh").chmod(0o755)
    (repo / "new").mkdir()
    (repo / "new" / "added.py").write_text("print('new')\n")
    os.symlink("added.py", repo / "new" / "link.py")
    shutil.rmtree(repo / "dir")
    (repo / "dir").write_text("now a file\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "candidate")
    return repo, base, git(repo, "rev-parse", "HEAD")


@unittest.skipUnless(HAVE_GIT, "git not installed")
class LaunchTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="spark-launch-"))
        cls.repo, cls.base, cls.candidate = make_repo(cls.tmp)
        cls.overlay = cls.tmp / "overlay"
        cls.manifest = build_overlay.build(cls.repo, cls.base, cls.candidate, cls.overlay)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _render(self, *extra, overlay: Path | None = None):
        out = self.tmp / f"r{len(list(self.tmp.glob('r*')))}"
        argv = [*BASE_ARGS, "--code-overlay", str(overlay or self.overlay), *extra, "--out", str(out)]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(render.main(argv), 0)
        document = json.loads((out / "trainer.json").read_text())
        self.assertEqual((document["apiVersion"], document["kind"]), ("v1", "List"))
        maps = [item for item in document["items"] if item["kind"] == "ConfigMap"]
        jobsets = [item for item in document["items"] if item["kind"] == "JobSet"]
        self.assertEqual(len(jobsets), 1)
        self.assertIs(document["items"][-1], jobsets[0])  # ConfigMaps first: the pods mount them
        return jobsets[0], maps, json.loads((out / "receipt.json").read_text()), out


def pod_of(jobset: dict) -> dict:
    return jobset["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]


def env_of(jobset: dict) -> dict:
    return {e["name"]: e.get("value") for e in pod_of(jobset)["containers"][0]["env"]}


class OverlayTest(LaunchTestCase):
    def test_manifest_lists_changes_with_git_modes(self):
        files = {f["path"]: f for f in self.manifest["files"]}
        self.assertEqual(sorted(files), ["change.py", "dir", "mode.sh", "new/added.py", "new/link.py"])
        self.assertEqual(files["mode.sh"]["mode"], "100755")
        self.assertEqual(files["new/link.py"]["mode"], "120000")
        self.assertEqual(self.manifest["deleted"], ["dir/inner.txt", "drop.txt"])
        self.assertEqual(self.manifest["base_commit"], self.base)
        self.assertEqual(self.manifest["candidate_tree"], git(self.repo, "rev-parse", f"{self.candidate}^{{tree}}"))
        for item in files.values():
            self.assertEqual(item["blob"], git(self.repo, "rev-parse", f"{self.candidate}:{item['path']}"))

    def test_archive_is_deterministic_and_checked(self):
        again = build_overlay.build(self.repo, self.base, self.candidate, self.tmp / "overlay-again")
        self.assertEqual(again["sha256"], self.manifest["sha256"])
        self.assertEqual((self.tmp / "overlay-again" / "overlay.tar.gz").read_bytes(),
                         (self.overlay / "overlay.tar.gz").read_bytes())
        tampered = self.tmp / "overlay-tampered"
        shutil.copytree(self.overlay, tampered)
        data = bytearray((tampered / "overlay.tar.gz").read_bytes())
        data[-1] ^= 1
        (tampered / "overlay.tar.gz").write_bytes(bytes(data))
        with self.assertRaises(ValueError):
            build_overlay.load_manifest(tampered)

    def test_refuses_an_empty_overlay(self):
        with self.assertRaises(ValueError):
            build_overlay.build(self.repo, self.candidate, self.candidate, self.tmp / "never")
        self.assertFalse((self.tmp / "never").exists())


class RenderTest(LaunchTestCase):
    def test_every_topology_and_mode_renders_without_placeholders(self):
        for topology, (hosts, accelerator) in render.TOPOLOGIES.items():
            for mode in render.MODES:
                jobset, maps, receipt, out = self._render(
                    "--mode", mode, "--run", f"t-{mode}", "--topology", topology, "--config", "example_config")
                text = json.dumps([jobset, *maps])
                self.assertIsNone(render.PLACEHOLDER.search(text), text[:200])
                job = jobset["spec"]["replicatedJobs"][0]["template"]["spec"]
                self.assertEqual((job["completions"], job["parallelism"]), (hosts, hosts))
                self.assertTrue(jobset["spec"]["suspend"])
                pod = pod_of(jobset)
                self.assertEqual(pod["nodeSelector"]["cloud.google.com/gke-tpu-topology"], topology)
                env = env_of(jobset)
                self.assertEqual(env["NNODES"], str(hosts))
                self.assertEqual(env["SPARK_ENTRY"], render.MODES[mode][0])
                self.assertEqual(env["SPARK_CANDIDATE_TREE"], self.manifest["candidate_tree"])
                self.assertEqual(env["SPARK_OVERLAY_CHUNKS"], str(len(maps)))
                self.assertEqual(receipt["chips"], 4 * hosts)
                self.assertEqual(receipt["accelerator_type"], accelerator)
                for name in ("trainer.json", "own-configmaps.sh"):
                    digest = hashlib.sha256((out / name).read_bytes()).hexdigest()
                    self.assertEqual(receipt["files"][name], digest, name)

    def test_code_is_inline_and_mounted_from_the_configmaps(self):
        jobset, maps, receipt, _ = self._render("--mode", "e2e", "--run", "inline", "--topology", "2x2x1",
                                                "--config", "s1_sft_trunc2")
        pod = pod_of(jobset)
        volumes = {v["name"]: v for v in pod["volumes"]}
        self.assertNotIn("code", volumes)  # no code archive in the bucket
        sources = volumes["overlay"]["projected"]["sources"]
        self.assertEqual([s["configMap"]["name"] for s in sources], [m["metadata"]["name"] for m in maps])
        mounts = {m["name"]: m["mountPath"] for m in pod["containers"][0]["volumeMounts"]}
        self.assertEqual(mounts["overlay"], "/mnt/overlay")
        archive = b"".join(base64.b64decode(m["binaryData"][f"overlay.tar.gz.{i}"]) for i, m in enumerate(maps))
        self.assertEqual(archive, (self.overlay / "overlay.tar.gz").read_bytes())
        self.assertEqual(json.loads(maps[0]["data"]["overlay.json"]), self.manifest)
        env = env_of(jobset)
        self.assertEqual(env["SPARK_BASE_REPO"], render.DEFAULT_BASE_REPO)
        self.assertEqual(env["SPARK_BASE_COMMIT"], self.base)
        self.assertEqual(env["SPARK_OVERLAY_SHA256"], self.manifest["sha256"])
        self.assertEqual(receipt["code"]["candidate_commit"], self.candidate)
        command = pod["containers"][0]["args"][0]
        self.assertEqual(command, (LAUNCH / "bootstrap.sh").read_text())
        self.assertEqual(maps[0]["metadata"]["labels"]["muse-spark-run"], "inline")

    @unittest.skipIf(shutil.which("bash") is None, "bash not installed")
    def test_owner_script_hands_the_configmaps_to_the_jobset(self):
        jobset, maps, receipt, out = self._render("--mode", "e2e", "--run", "owned", "--topology", "2x2x1",
                                                  "--config", "s1_sft_trunc2")
        script = out / "own-configmaps.sh"
        self.assertEqual(receipt["code"]["configmap_owner_script"], script.name)
        subprocess.run(["bash", "-n", str(script)], check=True)
        # A stand-in kubectl that records its arguments and knows the JobSet's uid.
        bindir = self.tmp / "fake-kubectl"
        bindir.mkdir()
        calls = self.tmp / "kubectl-calls.jsonl"
        (bindir / "kubectl").write_text(
            f"#!{sys.executable}\nimport json, sys\n"
            f"open({str(calls)!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "if 'get' in sys.argv: print('0f0f0f0f-1111-2222-3333-444455556666', end='')\n")
        (bindir / "kubectl").chmod(0o755)
        result = subprocess.run(["bash", str(script), "--context", "example-context"], capture_output=True, text=True,
                                env=dict(os.environ, PATH=f"{bindir}:{os.environ.get('PATH', '')}"), timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        recorded = [json.loads(line) for line in calls.read_text().splitlines()]
        self.assertEqual(recorded[0], ["--context", "example-context", "-n", "example-ns", "get", "jobset", "owned",
                                       "-o", "jsonpath={.metadata.uid}"])
        patched = []
        for call in recorded[1:]:
            self.assertEqual(call[:6], ["--context", "example-context", "-n", "example-ns", "patch", "configmap"])
            patched.append(call[6])
            self.assertEqual(call[7:9], ["--type=merge", "-p"])
            owner, = json.loads(call[9])["metadata"]["ownerReferences"]
            self.assertEqual(owner, {"apiVersion": jobset["apiVersion"], "kind": "JobSet", "name": "owned",
                                     "uid": "0f0f0f0f-1111-2222-3333-444455556666"})
        self.assertEqual(patched, [m["metadata"]["name"] for m in maps])

    def test_large_overlay_uses_several_configmaps(self):
        big = self.tmp / "big-repo"
        subprocess.run(["git", "init", "-q", str(big)], check=True)
        (big / "a.txt").write_text("a\n")
        git(big, "add", "-A")
        git(big, "commit", "-q", "-m", "base")
        base = git(big, "rev-parse", "HEAD")
        (big / "blob.bin").write_bytes(os.urandom(int(2.2 * render.CHUNK_BYTES)))
        git(big, "add", "-A")
        git(big, "commit", "-q", "-m", "big")
        overlay = self.tmp / "big-overlay"
        build_overlay.build(big, base, "HEAD", overlay)
        jobset, maps, receipt, _ = self._render("--mode", "e2e", "--run", "big", "--topology", "2x2x1",
                                                "--config", "c", overlay=overlay)
        self.assertEqual(len(maps), 3)
        for item in maps:
            self.assertLess(len(json.dumps(item)), render.CONFIGMAP_LIMIT)
        archive = b"".join(base64.b64decode(m["binaryData"][f"overlay.tar.gz.{i}"]) for i, m in enumerate(maps))
        self.assertEqual(archive, (overlay / "overlay.tar.gz").read_bytes())
        self.assertEqual(env_of(jobset)["SPARK_OVERLAY_CHUNKS"], "3")
        self.assertEqual(len(receipt["code"]["configmaps"]), 3)

    def test_priority_class_labels_work_size_and_extra_args(self):
        jobset, maps, _, _ = self._render("--mode", "sft", "--run", "t", "--topology", "2x2x1", "--config", "c",
                                          "--priority-class", "example-priority", "--extra-args=--training.steps=3",
                                          "--label", "team=example", "--work-size", "10Gi")
        template = jobset["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]
        pod = template["spec"]
        self.assertEqual(pod["priorityClassName"], "example-priority")
        self.assertEqual(jobset["metadata"]["labels"]["team"], "example")
        self.assertEqual(template["metadata"]["labels"]["team"], "example")
        self.assertTrue(all(m["metadata"]["labels"]["team"] == "example" for m in maps))
        work = [v for v in pod["volumes"] if v["name"] == "work"][0]
        self.assertEqual(work["emptyDir"]["sizeLimit"], "10Gi")
        self.assertEqual(env_of(jobset)["SPARK_EXTRA_ARGS"], "--training.steps=3")

    def test_runtime_and_pip_overlay_settings(self):
        jobset, _, receipt, _ = self._render(
            "--mode", "e2e", "--run", "t", "--topology", "4x4x4", "--config", "c",
            "--runtime-dir", "example/runtime", "--runtime-name", "r1", "--runtime-mode", "mounted",
            "--runtime-lock-sha256", "1" * 64, "--runtime-wheels-sha256", "2" * 64,
            "--runtime-debs-sha256", "3" * 64, "--pip-overlay", PINS)
        env = env_of(jobset)
        self.assertEqual(env["SPARK_RUNTIME_MODE"], "mounted")
        self.assertEqual(env["SPARK_RUNTIME_NAME"], "r1")
        self.assertEqual(env["SPARK_RUNTIME_LOCK_SHA256"], "1" * 64)
        self.assertEqual(env["SPARK_RUNTIME_WHEELS_SHA256"], "2" * 64)
        self.assertEqual(env["SPARK_RUNTIME_DEBS_SHA256"], "3" * 64)
        self.assertEqual(env["SPARK_PIP_OVERLAY"], PINS)
        self.assertNotIn("SPARK_PIP_FIND_LINKS", env)
        volumes = {v["name"]: v for v in pod_of(jobset)["volumes"]}
        self.assertEqual(volumes["runtime"]["csi"]["volumeAttributes"]["mountOptions"],
                         "implicit-dirs,only-dir=example/runtime")
        self.assertEqual(receipt["runtime"]["pip_overlay"], PINS)

    def test_extra_environment(self):
        jobset, _, receipt, _ = self._render("--mode", "e2e", "--run", "t", "--topology", "4x4x4", "--config", "c",
                                             "--env", "SPARK_STEPS=20", "--env", "SPARK_ADAPTER_INIT=/mnt/data/adapter")
        names = [e["name"] for e in pod_of(jobset)["containers"][0]["env"]]
        self.assertEqual(len(names), len(set(names)))
        env = env_of(jobset)
        self.assertEqual(env["SPARK_STEPS"], "20")
        self.assertEqual(env["SPARK_ADAPTER_INIT"], "/mnt/data/adapter")
        self.assertEqual(receipt["environment"], {"SPARK_STEPS": "20", "SPARK_ADAPTER_INIT": "/mnt/data/adapter"})

    def test_rejects_placeholders_and_bad_values(self):
        bad = [
            ["--bucket", "<GCS_BUCKET>"],
            ["--run", "Bad_Name"],
            ["--model-dir", "/abs/path"],
            ["--budget-seconds", "3000", "--deadline-seconds", "2700"],
            ["--label", "bad label=x"],
            ["--env", "lower_case=1"],
            ["--env", "SPARK_DIR=<GCS_BUCKET>"],
            ["--env", "HF_TOKEN=x"],
            ["--env", "NNODES=3"],
            ["--env", "SPARK_CONFIG=other"],
            ["--env", "SPARK_CANDIDATE_TREE=" + "0" * 40],
            ["--env", "SPARK_RUNTIME_MODE=mounted"],
            ["--base-repo", "git@example.com:repo.git"],
            ["--runtime-mode", "mounted"],
            ["--runtime-lock-sha256", "abc"],
            ["--pip-overlay", "../outside.txt"],
            ["--pip-overlay", "does/not/exist.txt"],
            ["--pip-find-links", "/mnt/runtime"],
            ["--code-overlay", "/nonexistent"],
        ]
        for override in bad:
            args = ["--mode", "sft", "--run", "ok", "--topology", "2x2x1", "--config", "c", *BASE_ARGS,
                    "--code-overlay", str(self.overlay), *override, "--out", str(self.tmp / "never")]
            with self.assertRaises(SystemExit, msg=override), contextlib.redirect_stderr(io.StringIO()):
                render.main(args)
            self.assertFalse((self.tmp / "never").exists())
        # No default dataset: the data directory is always explicit.
        index = BASE_ARGS.index("--data-dir")
        without_data = BASE_ARGS[:index] + BASE_ARGS[index + 2:]
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            render.main(["--mode", "sft", "--run", "ok", "--topology", "2x2x1", "--config", "c", *without_data,
                         "--code-overlay", str(self.overlay), "--out", str(self.tmp / "never")])
        self.assertFalse((self.tmp / "never").exists())

    def test_templates_use_only_known_placeholders(self):
        known = {"<RUN_NAME>", "<MODE>", "<ENTRY_MODULE>", "<CONFIG_MODULE>", "<CONFIG>", "<EXTRA_ARGS>",
                 "<TPU_TOPOLOGY>", "<BASE_REPO>", "<BASE_COMMIT>", "<BASE_TREE>", "<CANDIDATE_COMMIT>",
                 "<CANDIDATE_TREE>", "<OVERLAY_SHA256>", "<OVERLAY_CHUNKS>", "<OVERLAY_SOURCES>", "<BUDGET_S>",
                 "<DEADLINE_S>", "<NNODES>", "<GCS_BUCKET>", "<PREFIX>", "<MODEL_DIR>", "<DATA_DIR>",
                 "<RUNTIME_DIR>", "<K8S_NAMESPACE>", "<LOCAL_QUEUE>", "<GCP_PROJECT>", "<TRAINER_IMAGE>",
                 "<SERVICE_ACCOUNT>", "<SHM_SIZE>", "<WORK_SIZE>", "<TRAINER_COMMAND>"}
        for template in LAUNCH.glob("*.json.in"):
            found = set(render.PLACEHOLDER.findall(template.read_text()))
            self.assertTrue(found <= known, (template.name, sorted(found - known)))

    def test_full_model_lora_sft_launch(self):
        """The 4x4x4 LoRA SFT launch of the reproduction guide (example site values)."""
        jobset, _, receipt, _ = self._render(
            "--mode", "e2e", "--run", "example-sft", "--topology", "4x4x4", "--config", "s2_sft_full",
            "--prefix", "example/sft", "--model-dir", "example/model/Muse-Spark-Open-816B-A42B",
            "--data-dir", "example/data/chat-sft", "--priority-class", "example-priority",
            "--label", "owner=example", "--env", "SPARK_STEPS=60", "--env", "SPARK_VAL_DATA=/mnt/data/sft/val.jsonl",
            "--env", "SPARK_VAL_EVERY=10",
            "--extra-args=--dataloader.data-path=/mnt/data/sft/train.jsonl --checkpoint.interval=20",
            "--budget-seconds", "4200", "--deadline-seconds", "4500",
            "--runtime-name", "r1", "--runtime-lock-sha256", "1" * 64, "--runtime-wheels-sha256", "2" * 64,
            "--runtime-debs-sha256", "3" * 64, "--pip-overlay", PINS)
        self.assertEqual((receipt["hosts"], receipt["chips"], receipt["ranks"]), (16, 64, 128))
        self.assertEqual(receipt["accelerator_type"], "tpu7x-128")
        env = env_of(jobset)
        expected = {
            "SPARK_RUN": "example-sft", "SPARK_MODE": "e2e",
            "SPARK_ENTRY": "torchtitan.experiments.tpu.spark_e2e.train",
            "SPARK_CONFIG_MODULE": "torchtitan.experiments.tpu.spark_e2e.recipes", "SPARK_CONFIG": "s2_sft_full",
            "SPARK_EXTRA_ARGS": "--dataloader.data-path=/mnt/data/sft/train.jsonl --checkpoint.interval=20",
            "SPARK_TOPOLOGY": "4x4x4", "SPARK_BUDGET_S": "4200", "NNODES": "16",
            "SPARK_STEPS": "60", "SPARK_VAL_DATA": "/mnt/data/sft/val.jsonl", "SPARK_VAL_EVERY": "10",
            "WORKERS_0_HOSTNAME": "example-sft-trainer-0-0.example-sft",
        }
        for key, value in expected.items():
            self.assertEqual(env[key], value, key)
        pod = pod_of(jobset)
        self.assertEqual(pod["priorityClassName"], "example-priority")
        self.assertEqual(jobset["spec"]["replicatedJobs"][0]["template"]["spec"]["activeDeadlineSeconds"], 4500)
        options = {v["name"]: v["csi"]["volumeAttributes"]["mountOptions"] for v in pod["volumes"] if "csi" in v}
        self.assertEqual(options, {
            "runtime": "implicit-dirs,only-dir=example/sft/runtime",
            "model": "implicit-dirs,only-dir=example/model/Muse-Spark-Open-816B-A42B",
            "data": "implicit-dirs,only-dir=example/data/chat-sft",
            "out": "implicit-dirs,only-dir=example/sft/runs/example-sft",
        })
        self.assertEqual(pod["containers"][0]["resources"]["limits"]["google.com/tpu"], "4")
        self.assertEqual(jobset["metadata"]["labels"]["owner"], "example")
        self.assertEqual(receipt["outputs"], "gs://example-bucket/example/sft/runs/example-sft/")

    def test_two_layer_smoke_on_one_host(self):
        jobset, _, receipt, _ = self._render("--mode", "e2e", "--run", "example-2l", "--topology", "2x2x1",
                                             "--config", "s1_sft_trunc2", "--env", "SPARK_STEPS=5")
        env = env_of(jobset)
        self.assertEqual((env["NNODES"], env["SPARK_CONFIG"], env["SPARK_STEPS"]), ("1", "s1_sft_trunc2", "5"))
        self.assertEqual((receipt["hosts"], receipt["ranks"]), (1, 8))


@unittest.skipUnless(HAVE_GIT and HAVE_SHA256SUM and HAVE_BASH, "needs git, sha256sum and bash >= 4.4")
class BootstrapTest(LaunchTestCase):
    """bootstrap.sh against a local base repository, with the ConfigMaps written out as the kubelet would."""

    def _run(self, maps: list[dict], env: dict, name: str, **overrides) -> subprocess.CompletedProcess:
        root = self.tmp / name
        overlay_dir = root / "overlay"
        overlay_dir.mkdir(parents=True)
        for item in maps:
            for key, value in item.get("binaryData", {}).items():
                (overlay_dir / key).write_bytes(base64.b64decode(value))
            for key, value in item.get("data", {}).items():
                (overlay_dir / key).write_text(value)
        python_dir = root / "bin"
        python_dir.mkdir()
        os.symlink(sys.executable, python_dir / "python3")
        run_env = dict(os.environ)
        run_env.update({k: v for k, v in env.items() if v is not None})
        run_env.update(SPARK_BASE_REPO=f"file://{self.repo}", SPARK_SRC_DIR=str(root / "work" / "tt"),
                       SPARK_OVERLAY_DIR=str(overlay_dir), SPARK_BOOTSTRAP_ONLY="1",
                       PATH=f"{python_dir}:{os.environ.get('PATH', '')}")
        run_env.update(overrides)
        return subprocess.run(["bash", str(LAUNCH / "bootstrap.sh")], env=run_env, capture_output=True, text=True,
                              timeout=300)

    def _rendered(self, run: str, topology: str = "2x2x2"):
        jobset, maps, _, _ = self._render("--mode", "e2e", "--run", run, "--topology", topology,
                                          "--config", "s2_sft_full")
        return jobset, maps, env_of(jobset)

    def test_base_plus_overlay_equals_the_candidate_tree(self):
        git(self.repo, "config", "uploadpack.allowAnySHA1InWant", "true")
        _, maps, env = self._rendered("boot-ok")
        result = self._run(maps, env, "ok")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"= candidate tree {self.manifest['candidate_tree']}", result.stdout)
        src = self.tmp / "ok" / "work" / "tt"
        self.assertEqual((src / "change.py").read_text(), "x = 2\n")
        self.assertFalse((src / "drop.txt").exists())
        self.assertTrue(os.access(src / "mode.sh", os.X_OK))
        self.assertEqual(os.readlink(src / "new" / "link.py"), "added.py")
        self.assertEqual((src / "dir").read_text(), "now a file\n")
        self.assertEqual(git(src, "write-tree"), self.manifest["candidate_tree"])

    def test_refuses_a_tampered_overlay_or_a_wrong_base(self):
        git(self.repo, "config", "uploadpack.allowAnySHA1InWant", "true")
        _, maps, env = self._rendered("boot-bad")
        tampered = json.loads(json.dumps(maps))
        chunk = bytearray(base64.b64decode(tampered[0]["binaryData"]["overlay.tar.gz.0"]))
        chunk[len(chunk) // 2] ^= 0xFF
        tampered[0]["binaryData"]["overlay.tar.gz.0"] = base64.b64encode(bytes(chunk)).decode()
        cases = {
            "digest": (tampered, {}),
            "tree": (maps, {"SPARK_CANDIDATE_TREE": "0" * 40}),
            "base-tree": (maps, {"SPARK_BASE_TREE": "0" * 40}),
            "base-commit": (maps, {"SPARK_BASE_COMMIT": self.candidate}),
        }
        for name, (case_maps, overrides) in cases.items():
            with self.subTest(case=name):
                result = self._run(case_maps, env, f"bad-{name}", **overrides)
                self.assertNotEqual(result.returncode, 0, name)
                self.assertIn("[SPARK:GATE]", result.stderr + result.stdout, name)

    def test_multi_host_needs_the_injected_process_addresses(self):
        _, maps, env = self._rendered("boot-hosts", topology="2x2x2")
        result = self._run(maps, env, "hosts", SPARK_BOOTSTRAP_ONLY="0", TPU_PROCESS_ADDRESSES="a:1")
        self.assertEqual(result.returncode, 5, result.stdout + result.stderr)
        self.assertIn("process-address count 1 != NNODES 2", result.stderr)


def _wheel(directory: Path, name: str, version: str) -> Path:
    """A minimal pure-Python wheel (a zip with dist-info), built without network or build tools."""
    path = directory / f"{name}-{version}-py3-none-any.whl"
    dist = f"{name}-{version}.dist-info"
    files = {
        f"{name}/__init__.py": f"VERSION = {version!r}\n",
        f"{dist}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        f"{dist}/WHEEL": "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    record = []
    with zipfile.ZipFile(path, "w") as archive:
        for arcname, text in files.items():
            data = text.encode()
            archive.writestr(arcname, data)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            record.append(f"{arcname},sha256={digest},{len(data)}")
        record.append(f"{dist}/RECORD,,")
        archive.writestr(f"{dist}/RECORD", "\n".join(record) + "\n")
    return path


def _venv_with_pip() -> bool:
    probe = Path(tempfile.mkdtemp(prefix="spark-venv-probe-"))
    try:
        ok = subprocess.run([sys.executable, "-m", "venv", str(probe / "v")], capture_output=True).returncode == 0
        return ok and (probe / "v" / "bin" / "pip").exists()
    finally:
        shutil.rmtree(probe, ignore_errors=True)


@unittest.skipUnless(HAVE_SHA256SUM and HAVE_BASH, "needs sha256sum and bash >= 4.4")
class RuntimeInstallTest(unittest.TestCase):
    """runtime_install.sh: offline install of a digest-pinned wheelhouse, refusing anything else."""

    @classmethod
    def setUpClass(cls):
        if not _venv_with_pip():
            raise unittest.SkipTest("python -m venv cannot create an environment with pip here")
        cls.tmp = Path(tempfile.mkdtemp(prefix="spark-runtime-"))
        runtime = cls.tmp / "runtime"
        (runtime / "wheels").mkdir(parents=True)
        wheel = _wheel(runtime / "wheels", "sparkdemo", "1.2.3")
        (runtime / "rt-lock.txt").write_text("sparkdemo==1.2.3\n")
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        (runtime / "rt-wheels.sha256").write_text(f"{digest}  {wheel.name}\n")
        cls.runtime = runtime

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _install(self, name: str, **pins) -> subprocess.CompletedProcess:
        env = dict(os.environ, SPARK_RUNTIME_DIR=str(self.runtime), SPARK_RUNTIME_NAME="rt",
                   SPARK_RUNTIME_VENV=str(self.tmp / name / "venv"), SPARK_RUNTIME_WORK=str(self.tmp / name / "work"),
                   SPARK_BOOTSTRAP_PYTHON=sys.executable, SPARK_TCMALLOC=str(self.tmp / "no-allocator"))
        env.update(pins)
        script = (f"set -euo pipefail; source {LAUNCH / 'runtime_install.sh'}; "
                  "\"$TRAINER_PYTHON\" -c 'import sparkdemo; print(\"VERSION\", sparkdemo.VERSION)'")
        return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=600)

    def _pins(self) -> dict:
        return {"SPARK_RUNTIME_LOCK_SHA256": hashlib.sha256((self.runtime / "rt-lock.txt").read_bytes()).hexdigest(),
                "SPARK_RUNTIME_WHEELS_SHA256":
                    hashlib.sha256((self.runtime / "rt-wheels.sha256").read_bytes()).hexdigest()}

    def test_installs_the_pinned_wheelhouse_offline(self):
        result = self._install("ok", **self._pins())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("VERSION 1.2.3", result.stdout)
        self.assertIn("identical to the lock", result.stdout)

    def test_refuses_unpinned_or_mismatched_inputs(self):
        pins = self._pins()
        self.assertNotEqual(self._install("nopin").returncode, 0)
        bad = dict(pins, SPARK_RUNTIME_LOCK_SHA256="0" * 64)
        result = self._install("badpin", **bad)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match its pin", result.stderr)


@unittest.skipUnless(HAVE_BASH, "needs bash >= 4.4")
class SparkPodDryRunTest(unittest.TestCase):
    """spark_pod.sh builds the trainer and export commands from the rendered environment."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="spark-pod-"))
        (self.tmp / "model").mkdir()
        (self.tmp / "model" / "config.json").write_text("{}\n")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, *, wrapper: str | None = None, **env) -> subprocess.CompletedProcess:
        run_env = dict(os.environ, SPARK_SRC_DIR=str(REPO_ROOT), SPARK_MODEL=str(self.tmp / "model"),
                       SPARK_DATA=str(self.tmp / "data"), SPARK_OUT=str(self.tmp / "out"),
                       SPARK_LOG=str(self.tmp / "work" / "trainer.log"), SPARK_RUNTIME_MODE="preinstalled",
                       TRAINER_PYTHON=sys.executable, SPARK_DRY_RUN="1",
                       TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT=str(self.tmp / "cache"),
                       # The default list imports TorchTPU, JAX and Tokamax; the command tests do not need them.
                       SPARK_IMPORT_MODULES="json")
        run_env.update(env)
        run_env = {k: v for k, v in run_env.items() if v is not None}
        command = ["bash", str(LAUNCH / "spark_pod.sh")]
        if wrapper:  # shell code run first, then exec of the runner
            command = ["bash", "-c", wrapper + ' && exec bash "$0"', str(LAUNCH / "spark_pod.sh")]
        return subprocess.run(command, env=run_env, capture_output=True, text=True, timeout=600)

    def _versions(self, host: int = 0) -> dict:
        return json.loads((self.tmp / "out" / f"versions-host{host}.json").read_text())

    def test_multi_host_training_command(self):
        result = self._run(SPARK_CONFIG="s2_sft_full", SPARK_ENTRY="torchtitan.experiments.tpu.spark_e2e.train",
                           SPARK_CONFIG_MODULE="torchtitan.experiments.tpu.spark_e2e.recipes",
                           SPARK_EXTRA_ARGS="--checkpoint.interval=20 --dataloader.data-path='/mnt/data/sft/train.jsonl'",
                           NNODES="16", JOB_COMPLETION_INDEX="3", WORKERS_0_HOSTNAME="r-trainer-0-0.r")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        line = [l for l in result.stdout.splitlines() if l.startswith("[SPARK:DRY_RUN]")][0]
        for piece in ("-m torch.distributed.run --nnodes=16 --node-rank=3 --rdzv-backend=static "
                      "--rdzv-endpoint=r-trainer-0-0.r:29501 --nproc-per-node=8 "
                      "-m torchtitan.experiments.tpu.spark_e2e.train "
                      "--module=torchtitan.experiments.tpu.spark_e2e.recipes --config=s2_sft_full",
                      f"--dump_folder={self.tmp / 'out'} --checkpoint.interval=20 "
                      "--dataloader.data-path=/mnt/data/sft/train.jsonl"):
            self.assertIn(piece, line)
        self.assertEqual((self.tmp / "out" / "exit_code-host3").read_text().strip(), "0")
        versions = self._versions(3)
        self.assertIn("python", versions)
        self.assertEqual(versions["imports"], {"json": "ok"})
        self.assertIn("[SPARK:IMPORTS] 1 of 1 modules imported", result.stdout)
        self.assertTrue((self.tmp / "out" / "stdout-host3.log").is_file())

    def test_export_mode_command(self):
        adapter = self.tmp / "data" / "adapter" / "step-60"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(b"")
        result = self._run(SPARK_MODE="export", SPARK_CONFIG="adapter/step-60", NNODES="2", JOB_COMPLETION_INDEX="1",
                           WORKERS_0_HOSTNAME="r-trainer-0-0.r")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        line = [l for l in result.stdout.splitlines() if l.startswith("[SPARK:DRY_RUN]")][0]
        self.assertIn(f"-m torchtitan.experiments.tpu.checkpointing.hf_export --base {self.tmp / 'model'} "
                      f"--out {self.tmp / 'out' / 'hf'} --peft {adapter} --worker 1 --num-workers 2", line)

    def test_import_check_stops_the_pod_before_the_trainer(self):
        result = self._run(SPARK_CONFIG="c", SPARK_IMPORT_MODULES="json spark_missing_module_for_tests")
        self.assertEqual(result.returncode, 5, result.stdout + result.stderr)
        self.assertIn("[SPARK:IMPORTS] spark_missing_module_for_tests: ModuleNotFoundError", result.stdout)
        self.assertIn("[SPARK:GATE] import check failed", result.stdout + result.stderr)
        self.assertNotIn("[SPARK:DRY_RUN]", result.stdout)
        imports = self._versions()["imports"]
        self.assertEqual(imports["json"], "ok")
        self.assertTrue(imports["spark_missing_module_for_tests"].startswith("ModuleNotFoundError"))
        self.assertEqual((self.tmp / "out" / "exit_code-host0").read_text().strip(), "5")

    def test_default_import_list(self):
        result = self._run(SPARK_CONFIG="c", SPARK_IMPORT_MODULES=None, SPARK_CONFIG_MODULE="spark_missing_config_module",
                           SPARK_ENTRY="spark_missing_entry_module")
        self.assertEqual(result.returncode, 5, result.stdout + result.stderr)
        imports = self._versions()["imports"]
        self.assertEqual(set(imports), {"torch_tpu", "jax", "tokamax", "torchtitan.experiments.tpu.muse_spark",
                                        "spark_missing_config_module", "spark_missing_entry_module"})
        self.assertTrue(imports["spark_missing_entry_module"].startswith("ModuleNotFoundError"))
        result = self._run(SPARK_MODE="export", SPARK_CONFIG="adapter/none", SPARK_IMPORT_MODULES=None)
        self.assertEqual(set(self._versions()["imports"]), {"torchtitan.experiments.tpu.checkpointing.hf_export"})
        # SPARK_IMPORT_CHECK=0 skips the check.
        result = self._run(SPARK_CONFIG="c", SPARK_IMPORT_MODULES=None, SPARK_IMPORT_CHECK="0",
                           SPARK_CONFIG_MODULE="spark_missing_config_module")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("[SPARK:DRY_RUN]", result.stdout)
        self.assertEqual(self._versions()["imports"], {})

    def test_import_check_is_bounded_in_time(self):
        modules = self.tmp / "modules"
        modules.mkdir()
        (modules / "spark_slow_module_for_tests.py").write_text("import time\ntime.sleep(60)\n")
        result = self._run(SPARK_CONFIG="c", SPARK_IMPORT_MODULES="spark_slow_module_for_tests",
                           SPARK_IMPORT_TIMEOUT_S="3",
                           PYTHONPATH=os.pathsep.join(filter(None, [str(modules), os.environ.get("PYTHONPATH")])))
        self.assertEqual(result.returncode, 5, result.stdout + result.stderr)
        self.assertIn("[SPARK:IMPORTS] the import check did not finish within 3 s", result.stdout)
        self.assertIn("python", self._versions())  # written before the imports

    def test_low_open_file_limit_is_reported(self):
        result = self._run(SPARK_CONFIG="c", wrapper="ulimit -Sn 1024 && ulimit -Hn 1024")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("[SPARK:WARN] cannot raise the open-file limit to 65536; it stays 1024", result.stdout)

    def test_gates(self):
        result = self._run(SPARK_CONFIG="c", SPARK_TOPOLOGY="4x4x4", TPU_TOPOLOGY="2x2x1")
        self.assertEqual(result.returncode, 5)
        self.assertIn("does not match the rendered", result.stdout + result.stderr)
        (self.tmp / "model" / "config.json").unlink()
        result = self._run(SPARK_CONFIG="c")
        self.assertEqual(result.returncode, 5)
        self.assertEqual((self.tmp / "out" / "exit_code-host0").read_text().strip(), "5")


class ShellTest(unittest.TestCase):
    def test_shell_syntax(self):
        if shutil.which("bash") is None:
            self.skipTest("bash not installed")
        for script in sorted(LAUNCH.glob("*.sh")):
            subprocess.run(["bash", "-n", str(script)], check=True)

    def test_pip_pins_are_exact_and_hashed(self):
        for pins in LAUNCH.glob("pip-pins-*.txt"):
            requirements = re.findall(r"^([A-Za-z0-9_.-]+)==(\S+)", pins.read_text(), re.M)
            self.assertTrue(requirements, pins.name)
            blocks = re.split(r"^(?=[A-Za-z0-9_.-]+==)", pins.read_text(), flags=re.M)[1:]
            for block in blocks:
                self.assertRegex(block, r"--hash=sha256:[0-9a-f]{64}", pins.name)


if __name__ == "__main__":
    unittest.main()
