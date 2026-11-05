# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU check of the SFT held-out loss.

Runs a LoRA SFT recipe of ``cpu_recipes`` through ``sft_train`` with
``--sft.val-data-path``, ``--sft.val-every=1`` and
``--checkpoint.interval=1`` on one CPU rank, then checks that

* the held-out loss is logged before the first step and after every step,
  over the complete held-out set;
* the step-0 held-out loss (base model, LoRA B = 0) equals an independent
  computation over the same records (one unpacked forward per record, summed
  cross-entropy of the assistant tokens over their count), with the same
  token count;
* every checkpoint step is saved.

The TPU Qwen3.5 debug model needs ``SPARK_TEST_ASSETS`` (a Muse Spark
tokenizer with its chat template); its reference forward uses the same
TorchTitan model loaded from the same Hugging Face weights. With the Muse
Spark TPU package installed and the public tiny Muse Spark checkpoint in
``SPARK_TEST_ASSETS``, a second case compares with ``transformers``'
implementation of the tiny checkpoint.
"""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
import unittest

ASSETS = os.environ.get("SPARK_TEST_ASSETS", "")


def _has_tiny_checkpoint() -> bool:
    return bool(ASSETS) and any(pathlib.Path(ASSETS).glob("*.safetensors"))


def _write_data(work: pathlib.Path) -> list[str]:
    """16 training records and 8 held-out records; returns the held-out lines."""
    from torchtitan.experiments.tpu.post_training.tests import cpu_sft_smoke

    cpu_sft_smoke.write_conversations(work / "train.jsonl", n=16)
    cpu_sft_smoke.write_conversations(work / "all.jsonl", n=24)
    val_lines = (work / "all.jsonl").read_text().splitlines()[16:]
    (work / "val.jsonl").write_text("\n".join(val_lines) + "\n")
    return val_lines


def _train(work: pathlib.Path, config: str, assets: pathlib.Path, template_dir: pathlib.Path,
           extra: list[str]) -> pathlib.Path:
    from torchtitan.config import ConfigManager
    from torchtitan.experiments.tpu.post_training import sft_train
    from torchtitan.experiments.tpu.post_training.tests import smoke_common

    dump = work / "sft"
    smoke_common.single_rank_env()
    sft_train.start_trainer(ConfigManager().parse_args([
        "--module=torchtitan.experiments.tpu.post_training.tests.cpu_recipes",
        f"--config={config}", f"--hf_assets_path={assets}",
        f"--dataloader.data-path={work / 'train.jsonl'}",
        f"--dataloader.hf-tokenizer-path={template_dir}",
        f"--sft.val-data-path={work / 'val.jsonl'}", "--sft.val-every=1",
        f"--dump_folder={dump}", "--training.steps=2", "--checkpoint.interval=1", *extra,
    ]))
    return dump


def _reference(val_lines: list[str], tokenizer_dir: str, logits_fn) -> tuple[float, int]:
    """Summed assistant-token cross-entropy over the held-out records, and the count."""
    import torch

    from torchtitan.experiments.tpu.post_training import chat_template as ct

    tok = ct.load_hf_tokenizer(tokenizer_dir)
    # The CPU recipes render with reasoning effort "low".
    options = ct.ChatTemplateOptions(reasoning_effort="low")
    total, count = 0.0, 0
    for line in val_lines:
        rendered = ct.render_chat(tok, json.loads(line)["messages"], options, mask_mode="auto")
        ids = torch.tensor(rendered.input_ids)
        with torch.no_grad():
            logits = logits_fn(ids).float()
        mask = torch.tensor(rendered.assistant_mask[1:], dtype=torch.bool)
        targets = ids[1:]
        nll = -torch.log_softmax(logits[:-1], dim=-1)[torch.arange(len(targets)), targets]
        total += float(nll[mask].sum())
        count += int(mask.sum())
    return total, count


@unittest.skipUnless(ASSETS, "needs SPARK_TEST_ASSETS (Muse Spark tokenizer and chat template)")
class SFTValidationTest(unittest.TestCase):
    def _check(self, dump: pathlib.Path, total: float, count: int) -> None:
        val = json.loads((dump / "sft-val-rank0.json").read_text())
        self.assertEqual([e["step"] for e in val["evals"]], [0, 1, 2])
        self.assertTrue(val["set"]["complete"])
        self.assertEqual(val["set"]["records"], 8)
        for step in (1, 2):
            self.assertTrue((dump / "checkpoint" / f"step-{step}").is_dir()
                            or (dump / "lora_adapter" / f"step-{step}").is_dir(), step)
        first = val["evals"][0]
        self.assertEqual(first["supervised_tokens"], count)
        self.assertAlmostEqual(first["loss"], total / count, delta=1e-4 * max(1.0, total / count))
        self.assertTrue(all(e["loss"] > 0 for e in val["evals"]))

    def test_heldout_loss_debug_model(self):
        import torch

        from torchtitan.experiments.tpu.post_training.tests import smoke_common
        from torchtitan.experiments.tpu.unflatten_shim import set_unflatten_seq_len

        with tempfile.TemporaryDirectory() as tmp:
            work = pathlib.Path(tmp)
            val_lines = _write_data(work)
            assets = work / "assets"
            initial = smoke_common.make_debug_checkpoint(assets, pathlib.Path(ASSETS))
            # LoRA with TorchTitan's full saves: no extra checkpoint support needed.
            dump = _train(work, "sft_debug_lora_smoke", assets, pathlib.Path(ASSETS), [])
            for step in (1, 2):
                self.assertTrue((dump / "checkpoint" / f"step-{step}").is_dir(), step)

            spec = smoke_common.debug_model_spec()
            model = spec.model.build()
            missing, unexpected = model.load_state_dict(
                smoke_common.debug_base_state(initial), strict=False)
            self.assertEqual((missing, unexpected), ([], []))
            model.eval()

            def logits(ids: torch.Tensor) -> torch.Tensor:
                with set_unflatten_seq_len(len(ids)):
                    return model(ids, positions=torch.arange(len(ids)))

            total, count = _reference(val_lines, ASSETS, logits)
            self._check(dump, total, count)

    @unittest.skipUnless(_has_tiny_checkpoint(), "needs the tiny Muse Spark checkpoint in SPARK_TEST_ASSETS")
    def test_heldout_loss_spark_tiny(self):
        import torch

        from torchtitan.experiments.tpu.post_training.tests import smoke_common

        if not smoke_common.has_muse_spark():
            self.skipTest("needs the Muse Spark TPU package")
        try:
            from transformers import MuseSparkForConditionalGeneration
        except ImportError:  # pragma: no cover - depends on the installed transformers
            self.skipTest("needs transformers with MuseSpark (5.20.0 or newer)")

        with tempfile.TemporaryDirectory() as tmp:
            work = pathlib.Path(tmp)
            val_lines = _write_data(work)
            dump = _train(work, "sft_spark_tiny_lora_smoke", pathlib.Path(ASSETS),
                          pathlib.Path(ASSETS), ["--sft.lora-save=peft"])
            hf = MuseSparkForConditionalGeneration.from_pretrained(ASSETS, dtype=torch.float32).eval()
            total, count = _reference(val_lines, ASSETS, lambda ids: hf(input_ids=ids[None]).logits[0])
            self._check(dump, total, count)


if __name__ == "__main__":
    unittest.main()
