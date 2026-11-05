# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests: the full-parameter SFT recipes resolve to consistent trainer configs.

The recipes build on the post-training SFT layer (``config_registry``,
``lora``, ``sft_config``, ``sft_data``) and on the Muse Spark TPU package
(``muse_spark``: the model, expert placement and the step recipes, whose
module imports ``checkpointing``). Building the configs imports the TPU MoE
kernel modules, which need TorchTPU or an import-only CPU stand-in for it on
PYTHONPATH. The tests skip, naming the missing modules, unless all of these
are importable.

Run: python -m pytest -q torchtitan/experiments/tpu/post_training/tests/test_full_param_sft.py
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import unittest

# Modules from other changes (and TorchTPU) that the recipes import, directly or
# through the step recipes' module.
_DEPENDENCIES = (
    "torch_tpu",
    "torchtitan.experiments.tpu.post_training.config_registry",
    "torchtitan.experiments.tpu.post_training.lora",
    "torchtitan.experiments.tpu.post_training.sft_config",
    "torchtitan.experiments.tpu.post_training.sft_data",
    "torchtitan.experiments.tpu.muse_spark",
    "torchtitan.experiments.tpu.muse_spark.expert_placement",
    "torchtitan.experiments.tpu.muse_spark.spark_64chip_recipes",
    "torchtitan.experiments.tpu.checkpointing.resume_check",
)


def _available(name: str) -> bool:
  try:
    spec = importlib.util.find_spec(name)
  except ImportError:  # a parent package is missing
    return False
  # A directory without __init__.py is found as a namespace package (origin
  # None): sibling files without the package itself.
  return spec is not None and spec.origin is not None


MISSING_DEPENDENCIES = [name for name in _DEPENDENCIES if not _available(name)]

# Recipe environment for the expert-parallel / placement recipes (paths are not read here).
RECIPE_ENV = {
    "SPARK_FLAVOR": "816B-A42B-2L",
    "SPARK_EXPERT_PLACEMENT_FILE": "/placement/placement_ep8.json",
    "SPARK_EXPERT_PLACEMENT_OUT": "/out/placement_ep8.json",
}
FULL_SFT_ENV = (
    "SPARK_OPTIMIZER",
    "SPARK_LR",
    "SPARK_WARMUP",
    "SPARK_LR_MIN",
    "SPARK_EP",
    "SPARK_EP_CAPACITY",
    "SPARK_GRAD_REDUCE",
    "SPARK_EXPERT_GRAD_REDUCE",
    "SPARK_UPDATE_PROBE",
    "SPARK_UPDATE_PROBE_PARAMS",
    "SPARK_EXPERT_PLACEMENT",
    "SPARK_SEQ",
    "SPARK_COMPILE",
)


def _lora_configs(model_config):
  from torchtitan.protocols.module import Module

  return [
      fqn
      for fqn, cfg, _, _ in model_config.traverse(Module.Config, recurse=True)
      if type(cfg).__name__ == "Config"
      and hasattr(cfg, "rank")
      and hasattr(cfg, "alpha")
  ]


def _has_ep_grad_reduce_dtype(model_config) -> bool:
  return "ep_grad_reduce_dtype" in {
      f.name for f in dataclasses.fields(model_config)
  }


@unittest.skipIf(
    MISSING_DEPENDENCIES, "not importable: " + ", ".join(MISSING_DEPENDENCIES)
)
class FullParamSFTRecipeTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    from torchtitan.experiments.tpu.post_training import full_param_sft

    cls.fp = full_param_sft
    cls.recipes = sorted(
        n
        for n in dir(full_param_sft)
        if n.startswith("muse_spark_816b_a42b_ctr")
    )
    cls._saved = {
        k: os.environ.get(k) for k in list(RECIPE_ENV) + list(FULL_SFT_ENV)
    }
    for key in FULL_SFT_ENV:
      os.environ.pop(key, None)
    os.environ.update(RECIPE_ENV)

  @classmethod
  def tearDownClass(cls):
    for key, value in cls._saved.items():
      if value is None:
        os.environ.pop(key, None)
      else:
        os.environ[key] = value

  def _build(self, name, **env):
    os.environ.update(env)
    try:
      return getattr(self.fp, name)()
    finally:
      for key in env:
        os.environ.pop(key, None)

  def test_all_recipes_resolve(self):
    from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts
    from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
    from torchtitan.experiments.tpu.overrides.fused_loss import FusedLossLinearWrapper
    from torchtitan.experiments.tpu.post_training.sft_data import ChatSFTDataLoader

    self.assertEqual(len(self.recipes), 8, self.recipes)
    for name in self.recipes:
      with self.subTest(name=name):
        config = getattr(self.fp, name)()
        self.assertEqual(config.model_spec.name, "muse_spark")
        self.assertEqual(config.model_spec.flavor, "816B-A42B-2L")
        self.assertEqual(config.training.max_context_length, 8192)
        # TPU7x settings measured with the 816B model on one 4x4x4 slice.
        splash = config.splash_attention_kernel
        self.assertTrue(splash.use_tokamax_splash_attention_kernel)
        self.assertEqual(
            (splash.sa_block_q, splash.sa_block_kv, splash.sa_block_q_dkv),
            (512, 1024, 1024),
        )
        # The Tokamax Splash knobs are config fields of a separate change;
        # without them the TPU7x settings keep the kernel defaults.
        if hasattr(splash, "sa_use_base2_exp"):
          self.assertFalse(splash.sa_use_base2_exp)
        self.assertEqual(
            config.libtpu_init_flags.get("xla_tpu_dvfs_p_state"), "7"
        )
        self.assertTrue(config.parallelism.enable_tpu_topo_mesh)
        self.assertEqual(
            config.parallelism.fsdp_reshard_after_forward, "always"
        )
        self.assertEqual(
            type(config.activation_checkpoint).__qualname__, "FullAC.Config"
        )
        experts = [
            e
            for _, e, _, _ in config.model_spec.model.traverse(
                TpuGroupedExperts.Config
            )
        ]
        self.assertEqual(len(experts), 2)
        for e in experts:
          self.assertTrue(e.use_pallas_grouped_mm)
        # fp32 gradient reduction everywhere (SPARK_GRAD_REDUCE unset).
        self.assertEqual(config.training.mixed_precision_reduce, "float32")
        group = config.optimizer.param_groups[0]
        self.assertEqual(group.optimizer_kwargs["betas"], (0.9, 0.95))
        self.assertEqual(group.optimizer_kwargs["weight_decay"], 0.0)
        lora = _lora_configs(config.model_spec.model)
        if "lora" in name or "measure_placement" in name:
          # q, wkv (k and v) and o in each of the 2 layers.
          self.assertEqual(len(lora), 6, lora)
        else:
          self.assertEqual(lora, [])
        self.assertEqual(
            config.training.dtype,
            "float32" if "_full_fp32_" in name else "bfloat16",
        )
        # SFT view: chat rows, Pallas fused CE carrying the head's multiplier and soft cap.
        self.assertIsInstance(config.dataloader, ChatSFTDataLoader.Config)
        self.assertEqual(config.dataloader.reasoning_effort, "low")
        head = config.model_spec.model.lm_head
        self.assertIsInstance(head, FusedLossLinearWrapper.Config)
        self.assertIsInstance(config.loss, PallasCrossEntropyLoss.Config)
        self.assertEqual(
            (head.soft_cap, head.output_multiplier),
            (config.loss.soft_cap, config.loss.output_multiplier),
        )
        self.assertEqual(
            config.sft.lora_save, "full" if "_sft_full" in name else "none"
        )

  def test_full_parameter_sft_recipes(self):
    from torchtitan.experiments.tpu.low_precision_adamw import LowPrecisionOptimizersContainer
    from torchtitan.experiments.tpu.muse_spark.spark_64chip_recipes import FastSparkAdapter

    expected = {
        "muse_spark_816b_a42b_ctr_sft_full_tpu7x": (
            "kahan",
            "float8",
            "bfloat16",
        ),
        "muse_spark_816b_a42b_ctr_sft_full_sr8_tpu7x": (
            "stochastic",
            "float8",
            "bfloat16",
        ),
        "muse_spark_816b_a42b_ctr_sft_full_adamw8bit_tpu7x": (
            "stochastic",
            "float8",
            "bfloat16",
        ),
        "muse_spark_816b_a42b_ctr_sft_full_bf16sr_tpu7x": (
            "stochastic",
            "bfloat16",
            "bfloat16",
        ),
        "muse_spark_816b_a42b_ctr_sft_full_bf16_tpu7x": (
            None,
            None,
            "bfloat16",
        ),
        "muse_spark_816b_a42b_ctr_sft_full_fp32_tpu7x": (None, None, "float32"),
    }
    for name, (update, states, dtype) in expected.items():
      with self.subTest(name=name):
        config = getattr(self.fp, name)()
        group = config.optimizer.param_groups[0]
        kwargs = group.optimizer_kwargs
        if update is None:
          self.assertEqual(group.optimizer_name, "AdamW")
        else:
          self.assertEqual(group.optimizer_name, "LowPrecisionAdamW")
          self.assertIsInstance(
              config.optimizer, LowPrecisionOptimizersContainer.Config
          )
          self.assertEqual(
              (kwargs["param_update"], kwargs["state_dtype"]), (update, states)
          )
          self.assertEqual(config.optimizer.implementation, "for-loop")
        self.assertEqual(kwargs["lr"], 1e-6)
        self.assertEqual(config.training.dtype, dtype)
        # fp32 gradient reduction, EP 8 with the measured placement, eager grad norm.
        self.assertEqual(config.training.mixed_precision_reduce, "float32")
        self.assertEqual(config.parallelism.expert_parallel_degree, 8)
        self.assertFalse(config.compile.compile_grad_norm)
        model = config.model_spec.model
        self.assertEqual(model.ep_dispatcher, "allgather_bf16")
        self.assertEqual(model.ep_capacity_factor, 2.0)
        self.assertEqual(
            model.ep_expert_placement, "file:/placement/placement_ep8.json"
        )
        if _has_ep_grad_reduce_dtype(model):
          self.assertIsNone(model.ep_grad_reduce_dtype)
        self.assertTrue(
            issubclass(config.model_spec.state_dict_adapter, FastSparkAdapter)
        )
        self.assertTrue(config.checkpoint.load_only)
        self.assertEqual(config.comm.init_timeout_seconds, 1800)
        self.assertEqual(
            (
                config.lr_scheduler.warmup_steps,
                config.lr_scheduler.min_lr_factor,
            ),
            (2, 1.0),
        )
        self.assertEqual(
            config.training.num_tokens_per_microbatch_per_dp_rank, 8192
        )
        self.assertEqual(_lora_configs(model), [])

  def test_full_parameter_sft_environment(self):
    name = "muse_spark_816b_a42b_ctr_sft_full_tpu7x"
    config = self._build(
        name,
        SPARK_OPTIMIZER="sr8",
        SPARK_LR="3e-6",
        SPARK_WARMUP="0",
        SPARK_SEQ="4096",
        SPARK_EP_CAPACITY="1.5",
        SPARK_GRAD_REDUCE="bfloat16",
    )
    kwargs = config.optimizer.param_groups[0].optimizer_kwargs
    self.assertEqual(
        (kwargs["param_update"], kwargs["state_dtype"], kwargs["lr"]),
        ("stochastic", "float8", 3e-6),
    )
    self.assertEqual(config.lr_scheduler.warmup_steps, 0)
    self.assertEqual(
        (
            config.training.max_context_length,
            config.training.num_tokens_per_microbatch_per_dp_rank,
        ),
        (4096, 4096),
    )
    self.assertEqual(config.model_spec.model.ep_capacity_factor, 1.5)
    self.assertEqual(config.training.mixed_precision_reduce, "bfloat16")
    for bad in (
        {"SPARK_OPTIMIZER": "adam"},
        {"SPARK_GRAD_REDUCE": "float16"},
        {"SPARK_EXPERT_GRAD_REDUCE": "float16"},
    ):
      with self.subTest(bad=bad), self.assertRaises(ValueError):
        self._build(name, **bad)
    # The expert gradients' reduce dtype needs the model config's ep_grad_reduce_dtype.
    model = getattr(self.fp, name)().model_spec.model
    if _has_ep_grad_reduce_dtype(model):
      config = self._build(name, SPARK_EXPERT_GRAD_REDUCE="bfloat16")
      self.assertEqual(config.model_spec.model.ep_grad_reduce_dtype, "bfloat16")
      self.assertEqual(config.training.mixed_precision_reduce, "float32")
      config = self._build(
          name, SPARK_GRAD_REDUCE="bfloat16", SPARK_EXPERT_GRAD_REDUCE="float32"
      )
      self.assertEqual(config.model_spec.model.ep_grad_reduce_dtype, "float32")
      self.assertEqual(config.training.mixed_precision_reduce, "bfloat16")
    else:
      with self.assertRaises(ValueError):
        self._build(name, SPARK_EXPERT_GRAD_REDUCE="bfloat16")
    # Expert parallelism without a measured placement is refused; EP 1 needs none.
    saved = os.environ.pop("SPARK_EXPERT_PLACEMENT_FILE")
    try:
      with self.assertRaises(ValueError):
        getattr(self.fp, name)()
      config = self._build(name, SPARK_EP="1")
      self.assertEqual(config.parallelism.expert_parallel_degree, 1)
      self.assertEqual(
          config.model_spec.model.ep_expert_placement, "contiguous"
      )
    finally:
      os.environ["SPARK_EXPERT_PLACEMENT_FILE"] = saved

  def test_recipe_names_registered_only_here(self):
    # One recipe per name: the SFT layer's registry (--module
    # torchtitan.experiments.tpu.post_training) registers no other recipe
    # under any of this module's names.
    from torchtitan.experiments.tpu.post_training import config_registry

    for name in self.recipes:
      with self.subTest(name=name):
        other = getattr(config_registry, name, None)
        self.assertTrue(
            other is None or other is getattr(self.fp, name),
            f"config_registry registers its own {name}",
        )

  def test_update_probe_from_environment(self):
    from unittest import mock

    from torchtitan.experiments.tpu import update_probe

    name = "muse_spark_816b_a42b_ctr_sft_full_tpu7x"
    try:
      self._build(name)
      self.assertFalse(update_probe._INSTALLED)
      self._build(name, SPARK_UPDATE_PROBE="16")
      self.assertTrue(update_probe._INSTALLED)
    finally:
      update_probe.uninstall_update_probe()
    # The recipe's own parameter selection unless SPARK_UPDATE_PROBE_PARAMS is set.
    with mock.patch.object(self.fp, "install_update_probe") as install:
      self._build(name, SPARK_UPDATE_PROBE="16")
      self._build(
          name, SPARK_UPDATE_PROBE="8", SPARK_UPDATE_PROBE_PARAMS=r"layers\.1\."
      )
    self.assertEqual(
        install.call_args_list,
        [
            mock.call(16, self.fp.SPARK_UPDATE_PROBE_PATTERN),
            mock.call(8, r"layers\.1\."),
        ],
    )

  def test_placement_and_lora_comparison_recipes(self):
    config = self.fp.muse_spark_816b_a42b_ctr_sft_measure_placement_tpu7x()
    self.assertEqual(config.parallelism.expert_parallel_degree, 1)
    self.assertEqual(config.training.steps, 3)
    self.assertTrue(config.checkpoint.load_only)
    self.assertEqual(len(_lora_configs(config.model_spec.model)), 6)
    saved = os.environ.pop("SPARK_EXPERT_PLACEMENT_OUT")
    try:
      with self.assertRaises(ValueError):
        self.fp.muse_spark_816b_a42b_ctr_sft_measure_placement_tpu7x()
    finally:
      os.environ["SPARK_EXPERT_PLACEMENT_OUT"] = saved
    config = self.fp.muse_spark_816b_a42b_ctr_sft_lora_ep8_tpu7x()
    self.assertEqual(config.parallelism.expert_parallel_degree, 8)
    self.assertEqual(
        config.optimizer.param_groups[0].optimizer_kwargs["lr"], 1e-4
    )
    self.assertEqual(config.lr_scheduler.warmup_steps, 2)
    self.assertEqual(config.training.mixed_precision_reduce, "float32")


if __name__ == "__main__":
  unittest.main()
