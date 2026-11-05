# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TPU trainer configs for the Muse Spark Open text decoder.

Smoke / bring-up configs for ``--module torchtitan.experiments.tpu.muse_spark``:

* ``muse_spark_debugmodel``: 4 layers, 8 experts, random init, the packed
  ``c4_test`` set; runs on any slice.
* ``muse_spark_tiny_hf``: the public ``trl-internal-testing`` tiny Muse
  Spark checkpoint loaded through the HF checkpoint path
  (``--hf_assets_path=<dir>``); checks loading end to end.
* ``muse_spark_816b_2l``: the first two layers of the released model at real
  dimensions (8192 hidden, 256 experts of 4096x4096, 202048 vocabulary), BF16
  parameters and optimizer state, SimpleFSDP resharding after forward, full
  activation checkpointing and the shared TPU7x settings of ``tpu7x.py``
  (Tokamax Splash attention, tiled Pallas grouped matmuls, libtpu flags). The
  toy configs above keep the reference attention path. Point ``hf_assets_path``
  at the checkpoint (``--hf_assets_path``) and enable
  ``--checkpoint.initial_load_in_hf`` to start from the released weights
  (the adapter reads only layers 0-1).

Fine-tuning recipes build on these model specs.
"""

from torchtitan.components.checkpointer.dcp import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer.optimizer import default_adamw
from torchtitan.components.validate import Validator
from torchtitan.config.configs import CommConfig
from torchtitan.config.configs import ParallelismConfig
from torchtitan.config.configs import TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.tpu.datasets import c4_test_dataloader
from torchtitan.experiments.tpu.tpu_job_config import TPUConfig
from torchtitan.experiments.tpu.tpu_job_config import TpuTrainerCompileConfig
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig

from . import model_registry
from . import tpu7x

__all__ = ["muse_spark_816b_2l", "muse_spark_debugmodel", "muse_spark_tiny_hf"]


def _use_tpu_moe_kernels(cfg: TPUTrainerConfig) -> None:
  """Pallas grouped matmuls for the experts, as in the TPU Qwen3.5 MoE recipes.

  ``torch._grouped_mm``'s dWeight is unreliable on TPU at large row counts
  (google-pytorch/torch_tpu#2668), so training uses the Pallas kernels.
  """
  # pylint: disable-next=g-import-not-at-top
  from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts

  for _, experts, _, _ in cfg.model_spec.model.traverse(TpuGroupedExperts.Config):
    experts.use_pallas_grouped_mm = True
    experts.use_pallas_paired_grouped_mm = True
    experts.pallas_grouped_mm_tile_m = 512
    experts.pallas_grouped_mm_tgmm_tiling = (512, 0, 0)
    experts.use_kernel_native_weight_layout = True


def muse_spark_debugmodel() -> TPUTrainerConfig:
  """4-layer, 8-expert Muse Spark with random weights (bring-up)."""
  return TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./tests/assets/tokenizer",
      metrics=MetricsProcessor.Config(log_freq=1),
      model_spec=model_registry("debugmodel", seq_len=256),
      optimizer=default_adamw(lr=8e-4, implementation="foreach"),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=2,
          decay_ratio=0.8,
          decay_type="linear",
          min_lr_factor=0.0,
      ),
      training=TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=8 * 256,
          max_context_length=256,
          steps=10,
      ),
      dataloader=c4_test_dataloader(),
      parallelism=ParallelismConfig(use_simple_fsdp=True),
      checkpoint=CheckpointManager.Config(interval=10, last_save_model_only=False),
      activation_checkpoint=SelectiveAC.Config(preserve_rng_state=False),
      compile=TpuTrainerCompileConfig(enable=False, backend="tpu"),
      validator=Validator.Config(enable=False),
      tpu_config=TPUConfig(enable_amp=False),
  )


def muse_spark_tiny_hf() -> TPUTrainerConfig:
  """The public tiny checkpoint, loaded from HF safetensors at step 0."""
  cfg = muse_spark_debugmodel()
  cfg.model_spec = model_registry("tiny-hf", seq_len=256)
  cfg.hf_assets_path = "./assets/hf/tiny-MuseSparkForConditionalGeneration"
  cfg.checkpoint = CheckpointManager.Config(
      enable=True,
      initial_load_in_hf=True,
      initial_load_model_only=True,
      interval=1000,
  )
  return cfg


def muse_spark_816b_2l() -> TPUTrainerConfig:
  """First two released layers at real dimensions (zero-shot bring-up)."""
  cfg = TPUTrainerConfig(
      loss=CrossEntropyLoss.Config(),
      hf_assets_path="./assets/hf/Muse-Spark-Open-816B-A42B",
      metrics=MetricsProcessor.Config(log_freq=1),
      model_spec=model_registry("816B-A42B-2L", seq_len=8192),
      optimizer=default_adamw(lr=1e-5, implementation="foreach"),
      lr_scheduler=LRSchedulersContainer.Config(
          warmup_steps=2,
          decay_ratio=0.8,
          decay_type="cosine",
          min_lr_factor=0.1,
      ),
      training=TrainingConfig(
          num_tokens_per_microbatch_per_dp_rank=8192,
          max_context_length=8192,
          steps=5,
          dtype="bfloat16",
          mixed_precision_param="bfloat16",
          mixed_precision_reduce="float32",
      ),
      dataloader=c4_test_dataloader(),
      parallelism=ParallelismConfig(
          expert_parallel_degree=1,
          fsdp_reshard_after_forward="always",
          use_simple_fsdp=True,
      ),
      comm=CommConfig(train_timeout_seconds=900),
      activation_checkpoint=FullAC.Config(preserve_rng_state=False),
      compile=TpuTrainerCompileConfig(
          enable=False, backend="tpu", compile_grad_norm=True
      ),
      validator=Validator.Config(enable=False),
      tpu_config=TPUConfig(enable_amp=False, eager_mode="DEFER_AND_FUSE"),
  )
  _use_tpu_moe_kernels(cfg)
  # Real dimensions: Tokamax Splash attention, K/N-tiled unpaired grouped matmuls (the paired
  # kernel above overflows the 64 MiB scoped VMEM with 4096 x 4096 experts) and libtpu flags.
  tpu7x.apply(cfg)
  return cfg
