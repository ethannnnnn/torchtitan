# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests of the smoke helpers: the debug checkpoint writer, the check
report and the record comparison of the kill-and-resume check."""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import tempfile
import unittest

import torch

from torchtitan.experiments.tpu.post_training.tests import smoke_common


def _bits(tensor: torch.Tensor) -> torch.Tensor:
  """The tensor's bytes, for bit-for-bit comparisons."""
  return tensor.detach().contiguous().reshape(-1).view(torch.uint8)


class DebugCheckpointTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.tmp = tempfile.TemporaryDirectory()
    root = pathlib.Path(cls.tmp.name)
    cls.tokenizer_dir = root / "tokenizer"
    cls.tokenizer_dir.mkdir()
    # Only some of the tokenizer files exist, plus a chat template.
    cls.present = smoke_common.TOKENIZER_FILES[:2]
    for name in cls.present:
      (cls.tokenizer_dir / name).write_text(json.dumps({"name": name}))
    (cls.tokenizer_dir / "chat_template.jinja").write_text("{{ messages }}")
    cls.out = root / "checkpoint"
    # The caller's generator state around the call, for test_caller_rng_kept.
    torch.manual_seed(1234)
    cls.rng_before = torch.random.get_rng_state()
    cls.tensors = smoke_common.make_debug_checkpoint(
        cls.out, cls.tokenizer_dir
    )
    cls.rng_after = torch.random.get_rng_state()

  @classmethod
  def tearDownClass(cls):
    cls.tmp.cleanup()

  def test_caller_rng_kept(self):
    # The weights are seeded in a forked generator scope: the caller's
    # global generator is neither re-seeded nor advanced by the call.
    self.assertTrue(torch.equal(self.rng_after, self.rng_before))

  def _shard(self) -> pathlib.Path:
    shards = sorted(self.out.glob("*.safetensors"))
    self.assertEqual(len(shards), 1, shards)
    return shards[0]

  def test_one_shard_and_an_index_of_every_key(self):
    shard = self._shard()
    with open(self.out / "model.safetensors.index.json") as handle:
      index = json.load(handle)
    self.assertEqual(index["metadata"], {})
    self.assertEqual(
        index["weight_map"], {key: shard.name for key in self.tensors}
    )

  def test_dtypes(self):
    wide = {
        key
        for key in self.tensors
        if key.endswith(smoke_common.FLOAT32_KEY_SUFFIXES)
    }
    # The debug model has GatedDeltaNet layers, so both families occur.
    for suffix in smoke_common.FLOAT32_KEY_SUFFIXES:
      self.assertTrue(any(key.endswith(suffix) for key in wide), suffix)
    for key, tensor in self.tensors.items():
      want = torch.float32 if key in wide else torch.bfloat16
      self.assertEqual(tensor.dtype, want, key)

  def test_file_holds_the_returned_tensors(self):
    from safetensors.torch import load_file

    on_disk = load_file(str(self._shard()))
    self.assertEqual(sorted(on_disk), sorted(self.tensors))
    for key, tensor in self.tensors.items():
      self.assertEqual(on_disk[key].dtype, tensor.dtype, key)
      self.assertEqual(on_disk[key].shape, tensor.shape, key)
      self.assertTrue(torch.equal(_bits(on_disk[key]), _bits(tensor)), key)

  def test_tokenizer_files_copied_when_present_and_no_template(self):
    extra = {p.name for p in self.out.iterdir()} - {
        self._shard().name,
        "model.safetensors.index.json",
    }
    self.assertEqual(extra, set(self.present))
    for name in self.present:
      self.assertEqual(
          (self.out / name).read_text(),
          (self.tokenizer_dir / name).read_text(),
      )

  def test_two_calls_write_identical_tensors(self):
    again = smoke_common.make_debug_checkpoint(
        pathlib.Path(self.tmp.name) / "again", self.tokenizer_dir
    )
    self.assertEqual(sorted(again), sorted(self.tensors))
    for key, tensor in self.tensors.items():
      self.assertTrue(torch.equal(_bits(again[key]), _bits(tensor)), key)


class ReportTest(unittest.TestCase):

  def _report(self, checks: dict[str, bool]) -> tuple[int, list[str]]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
      code = smoke_common.report(checks, "UNIT")
    return code, out.getvalue().splitlines()

  @staticmethod
  def _markers(line: str) -> list[str]:
    return [word for word in line.split() if word in ("OK", "FAIL")]

  def test_every_check_holds(self):
    code, lines = self._report({"alpha holds": True, "beta holds": True})
    self.assertEqual(code, 0)
    self.assertEqual(len(lines), 3)
    for line, name in zip(lines, ("alpha holds", "beta holds")):
      self.assertIn("UNIT", line)
      self.assertIn(name, line)
      self.assertEqual(self._markers(line), ["OK"])
    self.assertIn("UNIT", lines[-1])
    self.assertIn("PASS", lines[-1])

  def test_one_check_fails(self):
    code, lines = self._report({"alpha holds": True, "beta breaks": False})
    self.assertEqual(code, 1)
    self.assertEqual(len(lines), 3)
    self.assertEqual(self._markers(lines[0]), ["OK"])
    self.assertEqual(self._markers(lines[1]), ["FAIL"])
    self.assertIn("beta breaks", lines[1])
    self.assertIn("UNIT", lines[-1])
    self.assertIn("FAIL", lines[-1])
    self.assertNotIn("PASS", lines[-1])

  def test_no_checks(self):
    code, lines = self._report({})
    self.assertEqual(code, 0)
    self.assertEqual(len(lines), 1)
    self.assertIn("UNIT", lines[0])
    self.assertIn("PASS", lines[0])

  def test_exact_lines(self):
    # Logs and reviews count these lines, so their bytes are fixed.
    cases = [
        ({"alpha": True}, "T", ["[T] OK   alpha", "[T] PASS"], 0),
        (
            {"alpha": True, "beta gamma": False},
            "T",
            ["[T] OK   alpha", "[T] FAIL beta gamma", "[T] FAIL"],
            1,
        ),
        ({}, "E", ["[E] PASS"], 0),
    ]
    for checks, tag, want_lines, want_code in cases:
      with self.subTest(checks=checks):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
          code = smoke_common.report(checks, tag)
        self.assertEqual(out.getvalue().splitlines(), want_lines)
        self.assertTrue(out.getvalue().endswith("\n"))
        self.assertEqual(code, want_code)


class ResumeRecordsTest(unittest.TestCase):
  """``read_records`` and ``compare`` of ``cpu_sft_resume``."""

  def test_read_records(self):
    from torchtitan.experiments.tpu.post_training.tests import cpu_sft_resume

    first = {"step": 1, "loss_local": 1.5}
    second = {"step": 2, "loss_local": 1.25}
    rewritten = {"step": 1, "loss_local": 0.75}
    with tempfile.TemporaryDirectory() as tmpdir:
      self.assertEqual(cpu_sft_resume.read_records(tmpdir), {})
      folder = pathlib.Path(tmpdir, "sft-records")
      folder.mkdir()
      (folder / "rank-00003.jsonl").write_text(
          "".join(
              json.dumps(row) + "\n\n" for row in (first, second, rewritten)
          )
      )
      (folder / "rank-00000.jsonl").write_text("\n  \n")
      (folder / "notes.txt").write_text("not a rank file")
      records = cpu_sft_resume.read_records(pathlib.Path(tmpdir))
    self.assertEqual(records, {3: {1: rewritten, 2: second}})

  def test_compare_lines(self):
    from torchtitan.experiments.tpu.post_training.tests import cpu_sft_resume

    compare = cpu_sft_resume.compare

    def step(loss, sha, counter=1, epoch=0):
      batch = {"microbatch": counter, "sha256": sha, "docs": [4, 1]}
      return {"step": 1, "loss_local": loss, "epoch": epoch, "batches": [batch]}

    ref = {0: {1: step(2.5, "aa")}}
    self.assertEqual(compare(ref, {}, [1]), ["ranks differ: [0] vs []"])
    self.assertEqual(
        compare(ref, {0: {}}, range(1, 3)),
        [
            "rank 0 step 1: missing (False, True)",
            "rank 0 step 2: missing (True, True)",
        ],
    )
    # The per-process microbatch counter is not compared.
    self.assertEqual(compare(ref, {0: {1: step(2.5, "aa", counter=7)}}, [1]), [])
    self.assertEqual(
        compare(ref, {0: {1: step(2.0, "bb", epoch=1)}}, [1]),
        [
            "rank 0 step 1 loss_local: 2.5 != 2.0",
            "rank 0 step 1 batches: [('aa', [4, 1])] != [('bb', [4, 1])]",
            "rank 0 step 1 epoch: 0 != 1",
        ],
    )


if __name__ == "__main__":
  unittest.main()
