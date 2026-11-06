# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests: every registered SFT recipe resolves to a consistent trainer config.

The Muse Spark recipe checks need the Muse Spark TPU package and are skipped
without it.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
from torchtitan.experiments.tpu.overrides.fused_loss import FusedLossLinearWrapper
from torchtitan.experiments.tpu.post_training import config_registry as cr
from torchtitan.experiments.tpu.post_training.sft_config import (
    SFTConfig,
    SFTTrainerConfig,
    as_sft_config,
    validated_lora_save,
)
from torchtitan.experiments.tpu.post_training.sft_data import ChatSFTDataLoader
from torchtitan.experiments.tpu.post_training.tests import cpu_recipes, smoke_common
from torchtitan.experiments.tpu.tpu_job_config import TpuTrainerCompileConfig

SPARK_RECIPES = sorted(
    n for n in dir(cr) if n.startswith("muse_spark_816b_a42b_")
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


@unittest.skipUnless(
    smoke_common.has_muse_spark(), "needs the Muse Spark TPU package"
)
class SparkRecipeTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls._flavor = os.environ.get("SPARK_FLAVOR")
    os.environ["SPARK_FLAVOR"] = "816B-A42B-2L"

  @classmethod
  def tearDownClass(cls):
    if cls._flavor is None:
      os.environ.pop("SPARK_FLAVOR", None)
    else:
      os.environ["SPARK_FLAVOR"] = cls._flavor

  def test_all_spark_recipes_resolve(self):
    from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts

    # The full-parameter recipes are registered in full_param_sft.py.
    self.assertEqual(SPARK_RECIPES, ["muse_spark_816b_a42b_sft_lora_tpu7x"])
    for name in SPARK_RECIPES:
      with self.subTest(name=name):
        config = getattr(cr, name)()
        self.assertIsInstance(config, SFTTrainerConfig)
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
        # The Tokamax Splash knobs are a separate change; check them when present.
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
        self.assertIsInstance(config.compile, TpuTrainerCompileConfig)
        self.assertTrue(config.compile.compile_grad_norm)
        self.assertEqual(config.tpu_config.eager_mode, "DEFER_AND_FUSE")
        experts = [
            e
            for _, e, _, _ in config.model_spec.model.traverse(
                TpuGroupedExperts.Config
            )
        ]
        self.assertEqual(len(experts), 2)
        for e in experts:
          self.assertTrue(e.use_pallas_grouped_mm)
          self.assertFalse(e.use_pallas_paired_grouped_mm)
          self.assertEqual(
              (
                  e.pallas_grouped_mm_tile_m,
                  e.pallas_grouped_mm_tile_k,
                  e.pallas_grouped_mm_tile_n,
              ),
              (512, 1024, 1024),
          )
        self.assertEqual(config.training.mixed_precision_reduce, "float32")
        group = config.optimizer.param_groups[0]
        self.assertEqual(group.optimizer_kwargs["betas"], (0.9, 0.95))
        self.assertEqual(group.optimizer_kwargs["weight_decay"], 0.0)
        lora = _lora_configs(config.model_spec.model)
        # q, wkv (k and v) and o in each of the 2 layers.
        self.assertEqual(len(lora), 6, lora)
        self.assertTrue(
            all(
                f.endswith(("qkv_linear.wq", "qkv_linear.wkv", "attention.wo"))
                for f in lora
            )
        )
        self.assertEqual(config.training.dtype, "bfloat16")

  def test_full_param_settings(self):
    # The full-parameter recipes of full_param_sft.py build on these.
    config = cr.build_sft_config(
        cr.SFTRecipe(
            model=cr.spark(),
            shape=cr.SFT_FULL_SHAPE,
            precision=cr.BF16,
            parallel=cr.spark_parallel(),
        )
    )
    self.assertEqual(config.training.mixed_precision_reduce, "float32")
    cr.spark_tpu_settings(
        config, full_param=True, fused_ce=True, frozen_head=False
    )
    self.assertEqual(config.training.mixed_precision_reduce, "bfloat16")
    self.assertEqual(_lora_configs(config.model_spec.model), [])
    self.assertIsInstance(
        config.model_spec.model.lm_head, FusedLossLinearWrapper.Config
    )

  def test_sft_recipes(self):
    for name in SPARK_RECIPES:
      with self.subTest(name=name):
        config = getattr(cr, name)()
        self.assertIsInstance(config.dataloader, ChatSFTDataLoader.Config)
        self.assertEqual(config.dataloader.reasoning_effort, "low")
        # Pallas fused CE carries the head's multiplier and soft cap.
        head = config.model_spec.model.lm_head
        self.assertIsInstance(head, FusedLossLinearWrapper.Config)
        self.assertIsInstance(config.loss, PallasCrossEntropyLoss.Config)
        self.assertEqual(config.loss.soft_cap, 20.0)
        self.assertEqual(
            (head.soft_cap, head.output_multiplier),
            (config.loss.soft_cap, config.loss.output_multiplier),
        )
        self.assertAlmostEqual(
            config.loss.output_multiplier, 0.1767766952966369, places=6
        )
        self.assertEqual(config.sft.lora_save, "adapter")


class BuilderTest(unittest.TestCase):

  def test_resolver_errors(self):
    with self.assertRaises(ValueError):
      cr.resolve_model_spec("nope", "x", 128)

  def test_public_helpers(self):
    # Other recipe modules build on these; the underscore names are aliases.
    self.assertIs(cr._base_config, cr.base_config)
    self.assertIs(cr._spark, cr.spark)
    self.assertIs(cr._spark_parallel, cr.spark_parallel)
    env = {
        "SPARK_FLAVOR": "816B-A42B-2L",
        "SPARK_EP": "8",
        "SPARK_COMPILE": "1",
    }
    with mock.patch.dict(os.environ, env):
      self.assertEqual(cr.spark(), cr.ModelChoice("muse_spark", "816B-A42B-2L"))
      parallel = cr.spark_parallel()
    self.assertEqual(parallel.expert_parallel, 8)
    self.assertTrue(parallel.compile_model)
    self.assertEqual(parallel.activation_checkpoint, "full")

  def test_sft_config_modes(self):
    base = cpu_recipes.sft_debug_smoke()
    self.assertEqual(
        as_sft_config(base, SFTConfig(lora_save="none")).sft.lora_save, "none"
    )
    self.assertEqual(
        as_sft_config(base, SFTConfig(lora_save="peft")).sft.lora_save, "peft"
    )
    with self.assertRaises(ValueError):
      as_sft_config(base, SFTConfig(lora_save="weights"))
    self.assertEqual(cpu_recipes.sft_debug_lora_smoke().sft.lora_save, "full")
    self.assertEqual(
        cpu_recipes.sft_debug_lora_adapter_smoke().sft.lora_save, "adapter"
    )

  def test_sft_config_over_an_sft_base(self):
    base = cpu_recipes.sft_debug_lora_smoke()  # already an SFTTrainerConfig
    self.assertEqual(base.sft.lora_save, "full")
    sft = SFTConfig(lora_save="none", val_every=3)
    config = as_sft_config(base, sft)
    self.assertIsInstance(config, SFTTrainerConfig)
    self.assertIs(config.sft, sft)
    # The TPU trainer sections are shared, not copied; the base keeps its sft.
    self.assertIs(config.training, base.training)
    self.assertIs(config.dataloader, base.dataloader)
    self.assertEqual(base.sft.lora_save, "full")
    for bad in (
        SFTConfig(lora_save="weights"),
        SFTConfig(val_every=-1),
        SFTConfig(val_max_microbatches=-1),
    ):
      with self.subTest(sft=bad), self.assertRaises(ValueError):
        as_sft_config(base, bad)

  def test_lora_save_validator(self):
    for mode in ("adapter", "peft", "full", "none"):
      self.assertEqual(validated_lora_save(mode), mode)
    with self.assertRaisesRegex(ValueError, "weights"):
      validated_lora_save("weights")

  def test_debug_recipe(self):
    config = cpu_recipes.sft_debug_smoke()
    self.assertIsInstance(config, SFTTrainerConfig)
    self.assertEqual(config.model_spec.name, "qwen3_5")
    self.assertIsInstance(config.loss, ChunkedLossWrapper.Config)
    self.assertEqual(config.loss.num_chunks, 4)
    self.assertIsInstance(config.dataloader, ChatSFTDataLoader.Config)
    self.assertEqual(config.dataloader.microbatches_per_step, 2)
    self.assertEqual(
        config.training.num_tokens_per_microbatch_per_dp_rank, 1024
    )
    # The TPU trainer's compile policy: no model compile on CPU, eager
    # gradient norm, and no TorchTPU eager mode.
    self.assertIsInstance(config.compile, TpuTrainerCompileConfig)
    self.assertFalse(config.compile.enable)
    self.assertEqual(config.compile.mode, "layer")
    self.assertFalse(config.compile.compile_grad_norm)
    self.assertEqual(config.tpu_config.eager_mode, "")
    self.assertFalse(config.tpu_config.enable_amp)
    self.assertEqual(_lora_configs(config.model_spec.model), [])

  def test_debug_lora_targets(self):
    config = cpu_recipes.sft_debug_lora_smoke()
    lora = _lora_configs(config.model_spec.model)
    # wq, wk, wv and wo of the two full-attention layers; never GatedDeltaNet.
    self.assertEqual(
        sorted(lora),
        sorted(
            f"layers.{layer}.attention.{proj}"
            for layer in (3, 7)
            for proj in ("wq", "wk", "wv", "wo")
        ),
    )

  def test_compile_policy(self):
    shape = cr.TrainShape(seq_len=512, num_chunks=4)
    config = cr.build_sft_config(
        cr.SFTRecipe(
            model=cpu_recipes.DEBUG,
            shape=shape,
            parallel=cr.Parallel(compile_model=True, compile_grad_norm=True),
        )
    )
    self.assertTrue(config.compile.enable)
    self.assertEqual(config.compile.components, ["model"])
    self.assertEqual(config.compile.backend, "tpu")
    self.assertTrue(config.compile.compile_grad_norm)
    self.assertEqual(config.tpu_config.eager_mode, "DEFER_AND_FUSE")


class CPURecipeTest(unittest.TestCase):

  def test_cpu_recipes_resolve(self):
    names = [n for n in dir(cpu_recipes) if n.endswith("_smoke")]
    self.assertGreaterEqual(len(names), 5)
    for name in names:
      if "spark" in name and not smoke_common.has_muse_spark():
        continue
      with self.subTest(name=name):
        config = getattr(cpu_recipes, name)()
        self.assertEqual(config.training.max_context_length, 512)


class BaseConfigTest(unittest.TestCase):
  """The sections of ``base_config``, which every SFT recipe builds on."""

  SHAPE = cr.TrainShape(
      seq_len=512,
      rows_per_microbatch=2,
      num_chunks=4,
      steps=7,
      warmup_steps=2,
      lr=3e-4,
      lr_decay="linear",
      lr_min_factor=0.5,
      betas=(0.8, 0.9),
      weight_decay=0.1,
      max_norm=0.5,
  )

  def _config(self, parallel=None, shape=None):
    from torchtitan.components.loss import CrossEntropyLoss

    self.loss = ChunkedLossWrapper.Config(
        num_chunks=4, loss_fn=CrossEntropyLoss.Config()
    )
    self.dataloader = ChatSFTDataLoader.Config(data_path="/data/sft.jsonl")
    return cr.base_config(
        model_spec=smoke_common.debug_model_spec(),
        hf_assets_path="/weights",
        loss=self.loss,
        dataloader=self.dataloader,
        shape=shape or self.SHAPE,
        precision=cr.Precision(dtype="bfloat16", optimizer="AdamW"),
        parallel=parallel or cr.Parallel(),
        num_tokens_per_train_step=cr.AUTO_STEP_TOKENS,
        checkpoint_interval=5,
        seed=11,
    )

  def test_training_schedule_and_fixed_sections(self):
    config = self._config()
    training = config.training
    self.assertEqual(
        (training.steps, training.max_context_length, training.max_norm),
        (7, 512, 0.5),
    )
    self.assertEqual(training.num_tokens_per_microbatch_per_dp_rank, 1024)
    self.assertEqual(training.num_tokens_per_train_step, -1)
    self.assertEqual(training.dtype, "bfloat16")
    self.assertEqual(
        (training.mixed_precision_param, training.mixed_precision_reduce),
        ("bfloat16", "float32"),
    )
    schedule = config.lr_scheduler
    self.assertEqual(
        (schedule.warmup_steps, schedule.decay_type, schedule.min_lr_factor),
        (2, "linear", 0.5),
    )
    group = config.optimizer.param_groups[0]
    self.assertEqual(group.optimizer_name, "AdamW")
    self.assertEqual(
        group.optimizer_kwargs,
        {"lr": 3e-4, "betas": (0.8, 0.9), "eps": 1e-8, "weight_decay": 0.1},
    )
    self.assertIs(config.loss, self.loss)
    self.assertIs(config.dataloader, self.dataloader)
    self.assertEqual(config.hf_assets_path, "/weights")
    self.assertEqual(config.metrics.log_freq, 1)
    self.assertEqual(config.debug.seed, 11)
    self.assertEqual(config.comm.train_timeout_seconds, 1800)
    self.assertEqual(cr.TRAIN_COMM_TIMEOUT_SECONDS, 1800)
    self.assertFalse(config.validator.enable)
    self.assertFalse(config.loss_kernel.use_loss_kernel)
    self.assertFalse(config.tpu_config.enable_amp)

  def test_parallelism(self):
    parallel = cr.Parallel(
        dp_shard=4,
        tensor_parallel=2,
        fsdp_reshard_after_forward="never",
    )
    mesh = self._config(parallel).parallelism
    self.assertEqual(
        (
            mesh.data_parallel_shard_degree,
            mesh.tensor_parallel_degree,
            mesh.expert_parallel_degree,
        ),
        (4, 2, 1),
    )
    self.assertEqual(
        (
            mesh.data_parallel_replicate_degree,
            mesh.context_parallel_degree,
            mesh.pipeline_parallel_degree,
        ),
        (1, 1, 1),
    )
    self.assertTrue(mesh.use_simple_fsdp)
    self.assertEqual(mesh.fsdp_reshard_after_forward, "never")

  def test_checkpoint_switches(self):
    checkpoint = self._config().checkpoint
    self.assertTrue(checkpoint.enable)
    self.assertEqual(checkpoint.interval, 5)
    self.assertEqual(checkpoint.async_mode, "disabled")
    self.assertTrue(checkpoint.initial_load_in_hf)
    self.assertTrue(checkpoint.initial_load_model_only)
    self.assertFalse(checkpoint.last_save_in_hf)
    self.assertFalse(checkpoint.last_save_model_only)

  def test_activation_checkpoint_names(self):
    from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC

    def policy(name):
      parallel = cr.Parallel(activation_checkpoint=name)
      return self._config(parallel).activation_checkpoint

    self.assertIsNone(policy("none"))
    for name, kind in (
        ("selective", SelectiveAC.Config),
        ("full", FullAC.Config),
    ):
      with self.subTest(name):
        self.assertIs(type(policy(name)), kind)
        self.assertFalse(policy(name).preserve_rng_state)
    with self.assertRaises(KeyError):
      self._config(cr.Parallel(activation_checkpoint="region"))

  def test_compile_and_eager_mode(self):
    plain = self._config(cr.Parallel(compile_grad_norm=True))
    self.assertFalse(plain.compile.enable)
    self.assertEqual(plain.compile.components, ["model", "loss"])
    self.assertEqual(
        (plain.compile.backend, plain.compile.mode), ("tpu", "layer")
    )
    self.assertTrue(plain.compile.compile_grad_norm)
    self.assertEqual(plain.tpu_config.eager_mode, "")
    compiled = self._config(cr.Parallel(compile_model=True))
    self.assertTrue(compiled.compile.enable)
    self.assertEqual(compiled.compile.components, ["model"])
    self.assertFalse(compiled.compile.compile_grad_norm)
    self.assertEqual(compiled.tpu_config.eager_mode, "DEFER_AND_FUSE")
    chosen = self._config(
        cr.Parallel(compile_model=True, eager_mode="DEFER_NEVER")
    )
    self.assertEqual(chosen.tpu_config.eager_mode, "DEFER_NEVER")
    eager_only = self._config(cr.Parallel(eager_mode="INTERNAL_DEFER_ALL"))
    self.assertEqual(eager_only.tpu_config.eager_mode, "INTERNAL_DEFER_ALL")

  def test_chunks_must_divide_the_microbatch(self):
    shape = cr.TrainShape(seq_len=512, rows_per_microbatch=2, num_chunks=3)
    with self.assertRaises(ValueError) as caught:
      self._config(shape=shape)
    self.assertIn("3", str(caught.exception))
    self.assertIn("1024", str(caught.exception))

  def test_sft_recipes_derive_the_step_tokens(self):
    config = cpu_recipes.sft_debug_smoke()
    self.assertEqual(config.training.num_tokens_per_train_step, -1)
    self.assertEqual(cr.AUTO_STEP_TOKENS, -1)


if __name__ == "__main__":
  unittest.main()
