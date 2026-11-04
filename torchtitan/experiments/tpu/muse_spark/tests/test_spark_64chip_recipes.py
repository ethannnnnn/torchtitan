# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU checks that the 64-chip step recipes (``spark_64chip_recipes``) build with the intended settings.

Needs TorchTPU or its import-only CPU stand-in on PYTHONPATH
(``spark_e2e/tests/tpu_import_stubs``, as ``spark_e2e/tests/run_cpu.sh`` sets it)
because the recipes enable the TPU MoE kernels; no model is instantiated.
"""
from __future__ import annotations

import os
import tempfile
import unittest

try:
  import torch_tpu  # noqa: F401  (the CPU stubs or the real package)
  HAVE_TPU_IMPORTS = True
except ImportError:  # pragma: no cover
  HAVE_TPU_IMPORTS = False


@unittest.skipUnless(HAVE_TPU_IMPORTS, "needs torch_tpu or the CPU stubs on PYTHONPATH")
class StepRecipesTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.tmp = tempfile.mkdtemp(prefix="step-recipes-")
    path = os.path.join(cls.tmp, "data.json")
    with open(path, "w") as handle:
      handle.write('{"text": "A tiny corpus."}\n')
    os.environ["SPARK_SYNTH_DATA"] = path

  def _check(self, name, *, lora, load_hf, rows, ep=1, tokamax=True, gmm_v2=True,
             dispatcher="allgather", capacity=2.0):
    from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts
    from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
    from torchtitan.experiments.tpu.muse_spark import spark_64chip_recipes as recipes
    from torchtitan.experiments.tpu.muse_spark import tpu7x

    cfg = getattr(recipes, name)()
    self.assertEqual(cfg.training.num_tokens_per_microbatch_per_dp_rank, rows * recipes.SEQ_LEN)
    self.assertEqual(cfg.parallelism.expert_parallel_degree, ep)
    self.assertEqual(cfg.compile.compile_grad_norm, ep == 1)
    self.assertEqual(cfg.tpu_config.eager_mode, "DEFER_AND_FUSE")
    from torchtitan.experiments.tpu.token_dispatcher import TpuAllToAllTokenDispatcher

    dispatchers = [d for _, d, _, _ in cfg.model_spec.model.traverse(TpuAllToAllTokenDispatcher.Config)]
    self.assertTrue(dispatchers)
    self.assertTrue(all(d.fuse_route_score_backward for d in dispatchers))
    if ep > 1:
      self.assertEqual(cfg.model_spec.model.ep_dispatcher, dispatcher)
      self.assertEqual(cfg.model_spec.model.ep_capacity_factor, capacity)
    splash = cfg.splash_attention_kernel
    self.assertEqual(bool(getattr(splash, "use_tokamax_splash_attention_kernel", False)), tokamax)
    if tokamax:
      for key, value in tpu7x.SPLASH_ATTENTION.items():
        self.assertEqual(getattr(splash, key), value, key)
    else:
      self.assertTrue(splash.use_splash_attention_kernel)
    self.assertTrue(cfg.parallelism.use_simple_fsdp)
    adapter = recipes.LoRATolerantSparkAdapter if lora else recipes.FastSparkAdapter
    self.assertIs(cfg.model_spec.state_dict_adapter, adapter)
    self.assertEqual(cfg.checkpoint.initial_load_in_hf, load_hf)
    self.assertEqual(cfg.training.mixed_precision_reduce, "float32" if lora else "bfloat16")
    self.assertIsInstance(cfg.loss, PallasCrossEntropyLoss.Config)
    self.assertEqual(cfg.libtpu_init_flags.get("xla_tpu_scoped_vmem_limit_kib"), "65472")
    self.assertEqual(cfg.libtpu_init_flags.get("xla_tpu_dvfs_p_state"), "7")
    experts = [e for _, e, _, _ in cfg.model_spec.model.traverse(TpuGroupedExperts.Config)]
    self.assertEqual(len(experts), 62)
    for e in experts:
      self.assertEqual(e.use_tokamax_gmm_v2, gmm_v2)
      self.assertTrue(e.use_kernel_native_weight_layout)
      self.assertFalse(e.use_pallas_paired_grouped_mm)
      self.assertEqual((e.pallas_grouped_mm_tile_m, e.pallas_grouped_mm_tile_k, e.pallas_grouped_mm_tile_n),
                       (512, 1024, 1024))

  def test_recipes(self):
    cases = [
        ("spark_816b_lora_64chip", True, True, 1),
        ("spark_816b_full_64chip", False, True, 1),
        ("spark_816b_lora_64chip_random", True, False, 1),
        ("spark_816b_full_64chip_random", False, False, 1),
        ("spark_816b_lora_64chip_16k", True, True, 2),
        ("spark_816b_lora_64chip_random_32k", True, False, 4),
    ]
    for name, lora, load_hf, rows in cases:
      with self.subTest(name=name):
        self._check(name, lora=lora, load_hf=load_hf, rows=rows)
    for name, load_hf, ep in (("spark_816b_lora_64chip_ep8_random", False, 8),
                              ("spark_816b_lora_64chip_tokamax_random", False, 1)):
      with self.subTest(name=name):
        self._check(name, lora=True, load_hf=load_hf, rows=1, ep=ep)
    with self.subTest(name="spark_816b_lora_64chip_stock_splash_random"):
      self._check("spark_816b_lora_64chip_stock_splash_random", lora=True, load_hf=False, rows=1,
                  tokamax=False)
    with self.subTest(name="spark_816b_lora_64chip_megablox_random"):
      self._check("spark_816b_lora_64chip_megablox_random", lora=True, load_hf=False, rows=1,
                  gmm_v2=False)

  def test_fast_reader_reads_permuted_expert_rows(self):
    import types

    import torch
    import torch.distributed.checkpoint as dcp
    from safetensors.torch import save_file
    from torchtitan.experiments.tpu.muse_spark import spark_64chip_recipes as recipes

    tmp = tempfile.mkdtemp(prefix="permuted-reader-")
    expert = "model.language_model.layers.3.mlp.experts.gate_up_proj"
    down = "model.language_model.layers.3.mlp.experts.down_proj"
    norm = "model.language_model.norm.weight"
    gate_up = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3).to(torch.bfloat16)
    down_w = (100 + torch.arange(24, dtype=torch.float32)).reshape(4, 3, 2).to(torch.bfloat16)
    norm_w = torch.ones(5, dtype=torch.bfloat16)
    save_file({expert: gate_up, down: down_w, norm: norm_w}, os.path.join(tmp, "model-00001-of-00001.safetensors"))
    order = [2, 0, 3, 1]
    for slot_map in ({"3": order}, None):
      reader = recipes.FastHFStorageReader(tmp, thread_count=2, slot_to_logical=slot_map)
      loaded = {expert: torch.empty_like(gate_up), down: torch.empty_like(down_w), norm: torch.empty_like(norm_w)}
      dcp.load(loaded, storage_reader=reader, no_dist=True)
      rows = order if slot_map else list(range(4))
      self.assertTrue(torch.equal(loaded[expert], gate_up[rows]))
      self.assertTrue(torch.equal(loaded[down], down_w[rows]))
      self.assertTrue(torch.equal(loaded[norm], norm_w))
    # The adapters hand the model's placement to the reader.
    placement = os.path.join(tmp, "placement.json")
    with open(placement, "w") as handle:
      handle.write('{"ep": 4, "num_experts": 4, "slot_to_logical": {"3": [2, 0, 3, 1]}}')
    layer = types.SimpleNamespace(moe=types.SimpleNamespace(num_experts=4))
    config = types.SimpleNamespace(layers=[layer], ep_expert_placement="file:" + placement, ep_placement_degree=4)
    self.assertEqual(recipes._slot_to_logical(config), {"3": order})
    config.ep_expert_placement = "contiguous"
    self.assertIsNone(recipes._slot_to_logical(config))

  def test_measured_placement_recipes(self):
    from torchtitan.experiments.tpu.muse_spark import spark_64chip_recipes as recipes

    for name in ("spark_816b_lora_64chip_ep8", "spark_816b_full_64chip_ep8",
                 "spark_816b_lora_64chip_measure_placement"):
      os.environ.pop("SPARK_EXPERT_PLACEMENT_FILE", None)
      os.environ.pop("SPARK_EXPERT_PLACEMENT_OUT", None)
      with self.subTest(name=name), self.assertRaises(ValueError):
        getattr(recipes, name)()
    placement = os.path.join(self.tmp, "placement.json")
    os.environ["SPARK_EXPERT_PLACEMENT_FILE"] = placement
    try:
      for name, lora, capacity in (("spark_816b_lora_64chip_ep8", True, 2.0),
                                   ("spark_816b_full_64chip_ep8", False, 1.5)):
        with self.subTest(name=name):
          self._check(name, lora=lora, load_hf=True, rows=1, ep=8, dispatcher="allgather_bf16",
                      capacity=capacity)
          cfg = getattr(recipes, name)()
          self.assertEqual(cfg.model_spec.model.ep_expert_placement, "file:" + placement)
    finally:
      os.environ.pop("SPARK_EXPERT_PLACEMENT_FILE", None)
    os.environ["SPARK_EXPERT_PLACEMENT_OUT"] = placement
    try:
      cfg = recipes.spark_816b_lora_64chip_measure_placement()
      self.assertEqual(cfg.parallelism.expert_parallel_degree, 1)
      self.assertTrue(cfg.checkpoint.initial_load_in_hf)
      self.assertGreaterEqual(cfg.training.steps, 2)
    finally:
      os.environ.pop("SPARK_EXPERT_PLACEMENT_OUT", None)


if __name__ == "__main__":
  unittest.main()
