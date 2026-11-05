# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks of the Muse Spark trainer configs and the shared TPU7x settings.

The configs apply the TPU MoE converters, whose kernel modules import
``torch_tpu``; on a host without it, put its CPU import stand-in
``torchtitan/experiments/tpu/tests/cpu_stubs`` first on ``PYTHONPATH`` (see
``test_muse_spark_cpu.py``).

Run: ``python -m pytest -q torchtitan/experiments/tpu/muse_spark/tests/test_config_registry.py``.
"""

import dataclasses
import unittest


def _tpu_moe_kernels_importable() -> bool:
  try:
    # pylint: disable-next=g-import-not-at-top
    from torchtitan.experiments.tpu.muse_spark import tpu_moe_converters

    tpu_moe_converters()
    return True
  except (ImportError, RuntimeError):
    return False


def _field_names(config_cls) -> set[str]:
  return {f.name for f in dataclasses.fields(config_cls)}


@unittest.skipUnless(_tpu_moe_kernels_importable(), "needs torch_tpu or its CPU stubs")
class MuseSparkTrainerConfigTest(unittest.TestCase):

  def test_configs_build_and_parse(self):
    # pylint: disable-next=g-import-not-at-top
    from torchtitan.config import ConfigManager
    # pylint: disable-next=g-import-not-at-top
    from torchtitan.experiments.tpu.muse_spark import config_registry

    for name in config_registry.__all__:
      cfg = getattr(config_registry, name)()
      self.assertEqual(cfg.model_spec.name, "muse_spark", name)
      parsed = ConfigManager().parse_args([
          "--module", "torchtitan.experiments.tpu.muse_spark",
          "--config", name,
          "--training.steps", "3",
      ])
      self.assertEqual(parsed.model_spec.name, "muse_spark", name)
      self.assertEqual(parsed.training.steps, 3, name)

  def test_816b_2l_uses_the_tpu7x_settings(self):
    # pylint: disable-next=g-import-not-at-top
    from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts
    # pylint: disable-next=g-import-not-at-top
    from torchtitan.experiments.tpu.muse_spark import config_registry
    # pylint: disable-next=g-import-not-at-top
    from torchtitan.experiments.tpu.muse_spark import tpu7x

    cfg = config_registry.muse_spark_816b_2l()
    # The gradient-norm compile switch lives on the trainer compile config.
    self.assertTrue(cfg.compile.compile_grad_norm)
    self.assertFalse(cfg.compile.enable)
    self.assertEqual(cfg.tpu_config.eager_mode, "DEFER_AND_FUSE")
    for flag, value in tpu7x.LIBTPU_FLAGS.items():
      self.assertEqual(cfg.libtpu_init_flags[flag], value)
    self.assertEqual(cfg.libtpu_init_flags["xla_tpu_dvfs_p_state"], "7")
    self.assertEqual(cfg.libtpu_init_flags["xla_tpu_scoped_vmem_limit_kib"], "65472")

    splash = cfg.splash_attention_kernel
    self.assertTrue(splash.use_tokamax_splash_attention_kernel)
    self.assertEqual((splash.sa_block_q, splash.sa_block_kv), (512, 1024))
    self.assertEqual(splash.sa_num_stacked_q_heads, 4)
    if {"sa_use_base2_exp", "sa_max_logit_const"} <= _field_names(type(splash)):
      self.assertFalse(splash.sa_use_base2_exp)
      self.assertEqual(splash.sa_max_logit_const, 0.0)

    experts = [
        e for _, e, _, _ in cfg.model_spec.model.traverse(TpuGroupedExperts.Config)
    ]
    self.assertEqual(len(experts), 2)  # one per layer
    has_gmm_v2 = "use_tokamax_gmm_v2" in _field_names(TpuGroupedExperts.Config)
    for e in experts:
      self.assertTrue(e.use_pallas_grouped_mm)
      self.assertFalse(e.use_pallas_paired_grouped_mm)
      self.assertTrue(e.use_kernel_native_weight_layout)
      self.assertEqual(
          (e.pallas_grouped_mm_tile_m, e.pallas_grouped_mm_tile_k,
           e.pallas_grouped_mm_tile_n),
          tpu7x.GMM_TILES,
      )
      if has_gmm_v2:
        self.assertTrue(e.use_tokamax_gmm_v2)

  def test_tpu7x_logs_the_settings_it_cannot_apply(self):
    # pylint: disable-next=g-import-not-at-top
    from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts
    # pylint: disable-next=g-import-not-at-top
    from torchtitan.experiments.tpu.muse_spark import config_registry
    # pylint: disable-next=g-import-not-at-top
    from torchtitan.experiments.tpu.muse_spark import tpu7x
    # pylint: disable-next=g-import-not-at-top
    from torchtitan.experiments.tpu.tpu_job_config import SplashAttentionKernelConfig
    # pylint: disable-next=g-import-not-at-top
    from torchtitan.tools.logging import logger

    missing_splash = not {"sa_use_base2_exp", "sa_max_logit_const"} <= (
        _field_names(SplashAttentionKernelConfig)
    )
    missing_gmm_v2 = "use_tokamax_gmm_v2" not in _field_names(
        TpuGroupedExperts.Config
    )
    cfg = config_registry.muse_spark_debugmodel()
    if not (missing_splash or missing_gmm_v2):
      with self.assertNoLogs(logger, "WARNING"):
        tpu7x.apply(cfg)
      return
    with self.assertLogs(logger, "WARNING") as logs:
      tpu7x.apply(cfg)
    text = "\n".join(logs.output)
    self.assertEqual("SplashAttentionKernelConfig has no" in text, missing_splash)
    self.assertEqual("no Tokamax GMM-v2" in text, missing_gmm_v2)

  def test_tpu7x_without_gmm_v2_keeps_megablox(self):
    # pylint: disable-next=g-import-not-at-top
    from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts
    # pylint: disable-next=g-import-not-at-top
    from torchtitan.experiments.tpu.muse_spark import config_registry
    # pylint: disable-next=g-import-not-at-top
    from torchtitan.experiments.tpu.muse_spark import tpu7x

    cfg = config_registry.muse_spark_debugmodel()
    tpu7x.tile_expert_matmuls(cfg, gmm_v2=False)
    for _, e, _, _ in cfg.model_spec.model.traverse(TpuGroupedExperts.Config):
      self.assertTrue(e.use_pallas_grouped_mm)
      self.assertEqual(e.pallas_grouped_mm_tgmm_tiling, tpu7x.GMM_TILES)
      self.assertFalse(getattr(e, "use_tokamax_gmm_v2", False))


if __name__ == "__main__":
  unittest.main()
