# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the build-time and load-time checks of the Muse Spark adapter.

They drive ``MuseSparkStateDictAdapter`` without any checkpoint file: the
comparison of a folder's ``model.safetensors.index.json`` with the text
decoder, the three labels of the key classifier, what ``from_hf`` skips,
refuses and reports, the shape table under LoRA and after a caller narrows
it, the self-check of the name mapping at construction, and
``permutes_experts``. Index files go to temporary folders; tensors live on
the meta device except where storage sharing is under test. On a host
without TorchTPU, put the shared stand-in first on the path, as for the
other model tests: ``PYTHONPATH=torchtitan/experiments/tpu/tests/cpu_stubs:$PWD``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import shutil
import tempfile
import unittest
from unittest import mock

import torch

from torchtitan.components.lora import LoRAConverter
from torchtitan.experiments.tpu.muse_spark import model_registry
from torchtitan.experiments.tpu.muse_spark import state_dict_adapter as spark_adapter
from torchtitan.experiments.tpu.muse_spark.state_dict_adapter import (
    hf_text_keys_for_config,
    MuseSparkStateDictAdapter,
)

# One tensor name under each vision / audio tower root of the released checkpoint.
_SPARK_TOWER_SAMPLES = (
    "model.vision_tower.blocks.0.attn.qkv.weight",
    "model.vision_adapter.fc1.weight",
    "model.vision_projection.weight",
    "model.audio_tower.layers.0.conv.weight",
)
_SPARK_ROUTED = "moe.routed_experts.inner_experts"


def _config(flavor: str = "debugmodel", num_layers: int | None = None):
  return model_registry(flavor, num_layers=num_layers, tpu_moe_kernels=False).model


def _write_spark_index(folder: str, names) -> None:
  """Writes ``model.safetensors.index.json`` into ``folder``, listing ``names``.

  Every name points at one shard file that is never created: the base adapter
  takes nothing from that file name but its shard number.
  """
  contents = {
      "weight_map": dict.fromkeys(sorted(names), "model-00001-of-00001.safetensors")
  }
  with open(os.path.join(folder, "model.safetensors.index.json"), "w") as f:
    json.dump(contents, f, indent=2)


def _spark_meta_checkpoint(adapter) -> dict[str, torch.Tensor]:
  """Tensors on the meta device for every checkpoint name the decoder loads.

  A renamed tensor takes the shape of its TorchTitan counterpart. Each MoE
  layer adds one packed ``gate_up_proj`` whose dim 1 stacks the gate rows on
  top of the up rows.
  """
  checkpoint = {
      hf_name: torch.empty(adapter.expected_model_shapes[tt_name], device="meta")
      for hf_name, tt_name in adapter.from_hf_map.items()
  }
  for idx in adapter.moe_layers:
    stem = f"layers.{idx}.{_SPARK_ROUTED}"
    experts, gate_rows, dim = adapter.expected_model_shapes[f"{stem}.w1_EFD"]
    up_rows = adapter.expected_model_shapes[f"{stem}.w3_EFD"][1]
    packed = f"model.language_model.layers.{idx}.mlp.experts.gate_up_proj"
    checkpoint[packed] = torch.empty(
        experts, gate_rows + up_rows, dim, device="meta"
    )
  return checkpoint


class SparkIndexFileTest(unittest.TestCase):
  """Construction holds model.safetensors.index.json against the decoder."""

  def setUp(self):
    self.folder = tempfile.mkdtemp(prefix="spark-index-")
    self.addCleanup(shutil.rmtree, self.folder, True)

  def test_full_index_with_towers_and_deeper_layers_is_accepted(self):
    config = _config("816B-A42B-2L")
    loaded = hf_text_keys_for_config(config)
    extra = [
        n.replace(".layers.1.", ".layers.5.") for n in loaded if ".layers.1." in n
    ]
    self.assertTrue(extra)
    names = {*loaded, *_SPARK_TOWER_SAMPLES, *extra}
    _write_spark_index(self.folder, names)
    adapter = MuseSparkStateDictAdapter(config, self.folder)
    self.assertEqual(set(adapter.fqn_to_index_mapping), names)

  def test_index_without_a_decoder_tensor_fails(self):
    config = _config("816B-A42B-2L")
    dropped = "model.language_model.layers.1.self_attn.o_proj.weight"
    names = set(hf_text_keys_for_config(config))
    self.assertIn(dropped, names)
    _write_spark_index(self.folder, names - {dropped})
    with self.assertRaises(ValueError) as caught:
      MuseSparkStateDictAdapter(config, self.folder)
    message = str(caught.exception)
    self.assertIn("MuseSparkStateDictAdapter", message)
    self.assertIn(self.folder, message)
    self.assertIn(f"1 {[dropped]}", message)  # the count, then the name

  def test_index_with_an_unplaceable_name_fails(self):
    config = _config("816B-A42B-2L")
    loaded = set(hf_text_keys_for_config(config))
    for sample in (
        "model.multi_modal_projector.linear_1.weight",  # neither decoder nor tower
        "model.language_model.layers.0.mlp.gate.bias",  # unmapped, inside layer 0
    ):
      with self.subTest(sample=sample):
        _write_spark_index(self.folder, loaded | {sample})
        with self.assertRaises(ValueError) as caught:
          MuseSparkStateDictAdapter(config, self.folder)
        self.assertIn(sample, str(caught.exception))

  def test_folder_without_index_defers_the_comparison_to_from_hf(self):
    with self.assertLogs(level=logging.INFO) as captured:
      adapter = MuseSparkStateDictAdapter(_config(num_layers=2), self.folder)
    self.assertIsNone(adapter.fqn_to_index_mapping)
    # captured.output lines read "LEVEL:logger:message". The base adapter may
    # also WARN that the index file is absent; only INFO lines are counted.
    about_folder = [
        line
        for line in captured.output
        if line.startswith("INFO:") and self.folder in line
    ]
    # One INFO line names the folder, and it hands the comparison to from_hf.
    self.assertEqual(
        ["from_hf" in line for line in about_folder],
        [True],
        f"INFO lines naming the folder: {about_folder}",
    )


class SparkFromHfTest(unittest.TestCase):
  """from_hf on a 2-layer cut of the debug model."""

  def setUp(self):
    self.adapter = MuseSparkStateDictAdapter(_config(num_layers=2), None)

  def test_round_trip_keeps_tensors_and_splits_experts_into_views(self):
    gen = torch.Generator().manual_seed(0)
    state = {}
    for name, shape in self.adapter.expected_model_shapes.items():
      # Embedding and head span the real vocabulary; meta keeps them free.
      if math.prod(shape) > 1 << 22:
        state[name] = torch.empty(shape, device="meta")
      else:
        state[name] = torch.randn(shape, generator=gen)
    checkpoint = self.adapter.to_hf(state)
    restored = self.adapter.from_hf(checkpoint)
    self.assertEqual(restored.keys(), state.keys())
    for tt_key, before in state.items():
      after = restored[tt_key]
      if tt_key in self.adapter.to_hf_map:
        # A renamed tensor comes back as the very object that went in.
        self.assertIs(after, before, f"{tt_key} was not passed through")
      else:
        self.assertTrue(torch.equal(after, before), f"{tt_key} changed values")
    self.assertTrue(self.adapter.moe_layers)
    for idx in self.adapter.moe_layers:
      packed = checkpoint[f"model.language_model.layers.{idx}.mlp.experts.gate_up_proj"]
      mid = packed.shape[1] // 2
      gate = restored[f"layers.{idx}.{_SPARK_ROUTED}.w1_EFD"]
      up = restored[f"layers.{idx}.{_SPARK_ROUTED}.w3_EFD"]
      for view in (gate, up):
        self.assertEqual(
            view.untyped_storage().data_ptr(), packed.untyped_storage().data_ptr()
        )
      self.assertTrue(torch.equal(gate, packed[:, :mid]))
      self.assertTrue(torch.equal(up, packed[:, mid:]))

  def test_tower_tensors_and_deeper_layers_are_skipped(self):
    checkpoint = _spark_meta_checkpoint(self.adapter)
    # Layers 2 and 3 of the 4-layer debug model lie past this cut.
    cut = [n for n in hf_text_keys_for_config(_config()) if n not in checkpoint]
    self.assertTrue(cut)
    self.assertTrue(all(".layers.2." in n or ".layers.3." in n for n in cut))
    extra = {
        name: torch.empty(1, device="meta")
        for name in (*cut, *_SPARK_TOWER_SAMPLES)
    }
    loaded = self.adapter.from_hf({**checkpoint, **extra})
    self.assertEqual(set(loaded), set(self.adapter.expected_model_shapes))

  def test_unknown_names_report_their_count_and_ten_sorted_names(self):
    checkpoint = _spark_meta_checkpoint(self.adapter)
    samples = [f"model.language_model.layers.0.probe_{i}.weight" for i in range(12)]
    for name in reversed(samples):
      checkpoint[name] = torch.empty(1, device="meta")
    with self.assertRaises(ValueError) as caught:
      self.adapter.from_hf(checkpoint)
    message = str(caught.exception)
    ordered = sorted(samples)  # probe_0, probe_1, probe_10, probe_11, probe_2, ...
    self.assertIn(f"12 {ordered[:10]}", message)
    for name in ordered[10:]:
      self.assertNotIn(name, message)

  def test_absent_and_misshaped_tensors_share_one_error(self):
    checkpoint = _spark_meta_checkpoint(self.adapter)
    del checkpoint["model.language_model.norm.weight"]
    checkpoint["model.language_model.layers.1.self_attn.o_proj.weight"] = (
        torch.empty(3, 5, device="meta")
    )
    with self.assertRaises(ValueError) as caught:
      self.adapter.from_hf(checkpoint)
    message = str(caught.exception)
    wo = "layers.1.attention.wo.weight"
    self.assertIn(f"1 {['norm.weight']}", message)
    self.assertIn(f"{wo} (3, 5) vs {self.adapter.expected_model_shapes[wo]}", message)

  def test_shape_table_is_read_on_every_call(self):
    checkpoint = _spark_meta_checkpoint(self.adapter)
    kept = dict(self.adapter.expected_model_shapes)
    del kept["norm.weight"]
    self.adapter.expected_model_shapes = kept
    subset = {
        k: v for k, v in checkpoint.items() if k != "model.language_model.norm.weight"
    }
    self.assertNotIn("norm.weight", self.adapter.from_hf(subset))
    with self.assertRaises(ValueError) as caught:  # a KeyError would escape this
      self.adapter.from_hf(checkpoint)
    self.assertIn("'norm.weight'", str(caught.exception))


class SparkLoRAShapeTableTest(unittest.TestCase):
  """LoRA matrices stay out of the shape table and out of both conversions."""

  def test_lora_converted_decoder(self):
    base = MuseSparkStateDictAdapter(_config(num_layers=2), None)
    lora_cfg = LoRAConverter(
        LoRAConverter.Config(rank=4, alpha=8.0, target_modules=["wq", "wkv", "wo"])
    ).convert(_config(num_layers=2))
    adapter = MuseSparkStateDictAdapter(lora_cfg, None)
    self.assertFalse([n for n in adapter.expected_model_shapes if ".lora_" in n])
    self.assertEqual(adapter.expected_model_shapes, base.expected_model_shapes)
    loaded = adapter.from_hf(_spark_meta_checkpoint(base))
    self.assertEqual(set(loaded), set(base.expected_model_shapes))
    with torch.device("meta"):
      state_dict = lora_cfg.build().state_dict()
    self.assertTrue([n for n in state_dict if ".lora_" in n])
    self.assertEqual(
        set(adapter.to_hf(state_dict)), set(hf_text_keys_for_config(lora_cfg))
    )


class SparkMappingSelfCheckTest(unittest.TestCase):

  def test_mapping_without_a_decoder_tensor_fails_construction(self):
    original = spark_adapter._build_from_hf_map  # pylint: disable=protected-access

    def patched(model_config):
      remapped = original(model_config)
      del remapped["model.language_model.norm.weight"]
      return remapped

    with mock.patch.object(spark_adapter, "_build_from_hf_map", patched):
      with self.assertRaises(ValueError) as caught:
        MuseSparkStateDictAdapter(_config(num_layers=2), None)
    message = str(caught.exception)
    self.assertIn("construction", message)
    self.assertIn("'norm.weight'", message)


class SparkPermutesExpertsTest(unittest.TestCase):

  def test_flag_follows_the_slot_maps_at_call_time(self):
    config = _config(num_layers=2)
    adapter = MuseSparkStateDictAdapter(config, None)
    self.assertFalse(adapter.permutes_experts)  # contiguous placement
    contiguous = list(range(8))
    swapped = [1, 0, *contiguous[2:]]
    for layout, flag in (
        (swapped, True),
        ({"0": contiguous, "1": list(contiguous)}, False),
        ({"0": contiguous, "1": swapped}, True),
        ([], False),
        (contiguous, False),
    ):
      with self.subTest(layout=layout):
        adapter.expert_slot_to_logical = layout
        self.assertIs(adapter.permutes_experts, flag)
    interleaved = dataclasses.replace(
        config, ep_expert_placement="interleaved", ep_placement_degree=2
    )
    self.assertTrue(MuseSparkStateDictAdapter(interleaved, None).permutes_experts)


class SparkClassifierAgreementTest(unittest.TestCase):
  """Each label predicts what from_hf and the index file check do."""

  def test_labels_match_both_checks(self):
    config = _config(num_layers=2)
    adapter = MuseSparkStateDictAdapter(config, None)
    checkpoint = _spark_meta_checkpoint(adapter)
    loaded = set(hf_text_keys_for_config(config))
    folder = tempfile.mkdtemp(prefix="spark-labels-")
    self.addCleanup(shutil.rmtree, folder, True)
    samples = (
        ("model.language_model.layers.1.mlp.gate.weight", "expected"),
        ("model.language_model.layers.0.mlp.experts.gate_up_proj", "expected"),
        ("lm_head.weight", "expected"),
        ("model.vision_projection.weight", "excluded"),
        ("model.audio_tower.layers.0.conv.weight", "excluded"),
        ("model.language_model.layers.2.self_attn.q_proj.weight", "excluded"),
        ("model.language_model.layers.1.self_attn.q_proj.bias", "unknown"),
        ("model.language_model.rotary_emb.inv_freq", "unknown"),
        ("model.image_newline", "unknown"),
    )
    for hf_name, label in samples:
      with self.subTest(hf_name=hf_name):
        self.assertEqual(
            adapter._classify_hf_key(hf_name), label  # pylint: disable=protected-access
        )
        if label == "expected":
          # Loading needs the name, and an index file without it is refused.
          with self.assertRaises(ValueError):
            adapter.from_hf({k: v for k, v in checkpoint.items() if k != hf_name})
          _write_spark_index(folder, loaded - {hf_name})
          with self.assertRaises(ValueError) as caught:
            MuseSparkStateDictAdapter(config, folder)
          self.assertIn(hf_name, str(caught.exception))
          continue
        full = {**checkpoint, hf_name: torch.empty(2, device="meta")}
        _write_spark_index(folder, loaded | {hf_name})
        if label == "excluded":
          self.assertEqual(
              set(adapter.from_hf(full)), set(adapter.expected_model_shapes)
          )
          MuseSparkStateDictAdapter(config, folder)
          continue
        with self.assertRaises(ValueError) as caught:
          adapter.from_hf(full)
        self.assertIn(hf_name, str(caught.exception))
        with self.assertRaises(ValueError) as caught:
          MuseSparkStateDictAdapter(config, folder)
        self.assertIn(hf_name, str(caught.exception))


if __name__ == "__main__":
  unittest.main()
