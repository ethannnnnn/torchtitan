# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parallelization for the Muse Spark Open text decoder on TPU.

Mirrors the TPU MoE lanes (``tpu/qwen3_5``, ``tpu/muse_glimmer``):

  AC -> per-block compile -> SimpleFSDP (or FSDP2).

Expert parallelism 1 (default): experts are FSDP-sharded like every other
weight, as in the Qwen3.5-35B-A3B TPU recipes: each rank all-gathers a layer's
experts before the grouped matmuls and reduce-scatters their gradients
afterwards. With 256 experts, SimpleFSDP shards the expert dimension for FSDP
degrees up to 256 and the hidden dimension beyond.

Expert parallelism > 1 (SimpleFSDP only) comes from ``expert_parallel.py``,
imported only when it is enabled: the expert weights are sharded on the
expert dimension over ``ep`` and FSDP-sharded over ``efsdp``, and the
post-expert norm shared by all experts stays a dense parameter whose gradient
is reduced over every rank.

Not supported yet (rejected explicitly): tensor, context and pipeline
parallelism.
"""

import dataclasses

import torch
from torch import nn

import torchtitan.config
import torchtitan.distributed
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder
from torchtitan.experiments.graph_trainer.common_utils import apply_simple_fsdp
from torchtitan.experiments.tpu.muse_glimmer.infra.parallelize import configure_muse_glimmer_simple_fsdp_shard_dims
from torchtitan.experiments.tpu.muse_spark.model import expert_parallel_module
from torchtitan.experiments.tpu.spmd_utils import apply_spmd_safe
from torchtitan.models.common.moe import MoE
from torchtitan.tools.logging import logger

__all__ = ["parallelize_muse_spark", "validate_parallel_dims"]


def validate_parallel_dims(
    model: nn.Module, parallel_dims: torchtitan.distributed.ParallelDims
) -> None:
  """Rejects parallelisms the Muse Spark TPU lane does not support yet."""
  unsupported = {
      "tensor": parallel_dims.tp_enabled,
      "context": parallel_dims.cp_enabled,
      "pipeline": parallel_dims.pp_enabled,
  }
  if bad := [name for name, enabled in unsupported.items() if enabled]:
    raise NotImplementedError(
        f"Muse Spark on TPU supports FSDP/HSDP and EP only; {', '.join(bad)}"
        " parallelism is not supported yet."
    )
  if getattr(model, "vision_encoder", None) is not None:
    raise NotImplementedError("Muse Spark on TPU is text-only.")


def parallelize_muse_spark(
    model: nn.Module,
    *,
    parallel_dims: torchtitan.distributed.ParallelDims,
    training: torchtitan.config.TrainingConfig,
    parallelism: torchtitan.config.ParallelismConfig,
    compile_config: torchtitan.config.CompileConfig,
    ac_config: "torchtitan.distributed.activation_checkpoint.ActivationCheckpointingConfig | None",
    dump_folder: str,
    skip_dp: bool = False,
    tpu_compile_mode: str = "layer",
    **kwargs,
) -> nn.Module:
  """Parallelizes a Muse Spark text model for TPU (see the module docstring).

  Args:
    model: The ``MuseSparkModel`` (may be on the meta device).
    parallel_dims: Parallel dimensions for this run.
    training: Training config (mixed-precision dtypes, CPU offload).
    parallelism: ``use_simple_fsdp`` selects SimpleFSDP instead of FSDP2.
    compile_config: ``torch.compile`` config.
    ac_config: Activation checkpointing config, or None.
    dump_folder: Dump folder for AC debugging artifacts.
    skip_dp: Skip data parallelism (e.g. for a generator).
    tpu_compile_mode: ``layer``, ``fullgraph`` or ``aot_fx_trace``.
    **kwargs: Ignored; accepted for signature compatibility with train.py.

  Returns:
    The parallelized model.
  """
  del kwargs  # Unused.
  validate_parallel_dims(model, parallel_dims)
  if parallel_dims.ep_enabled:
    if not parallelism.use_simple_fsdp and not skip_dp:
      raise NotImplementedError(
          "Muse Spark expert parallelism is implemented with SimpleFSDP"
          " (parallelism.use_simple_fsdp=True)."
      )
    # Shards the expert weights on the expert dim over `ep` (MoE sharding
    # configs from `configure_expert_parallel`) and wires the dispatchers.
    model.parallelize(parallel_dims)

  effective_compile_config = compile_config
  if compile_config.enable and hasattr(model, "layers"):
    # As in tpu/qwen3_5: torch_tpu `spmd_safe` for compiled non-MoE blocks.
    for layer in model.layers.values():
      if not any(isinstance(m, MoE) for m in layer.modules()):
        apply_spmd_safe(layer)
  if tpu_compile_mode in ("fullgraph", "aot_fx_trace") and compile_config.enable:
    effective_compile_config = dataclasses.replace(compile_config, enable=False)

  if ac_config is not None:
    ac_config.build(dump_folder=dump_folder).apply(model)

  if (
      effective_compile_config.enable
      and "model" in effective_compile_config.components
  ):
    apply_compile(
        model,
        compile_config=effective_compile_config,
        parallel_dims=parallel_dims,
    )

  if not skip_dp and (
      parallel_dims.dp_shard_enabled or parallel_dims.dp_replicate_enabled
  ):
    if parallelism.use_simple_fsdp:
      # The 202048-row vocabulary is uneven for FSDP degrees such as 128;
      # same embedding FQN as Muse Glimmer (``tok_embeddings.embedding``).
      configure_muse_glimmer_simple_fsdp_shard_dims(model, parallel_dims)
      if parallel_dims.ep_enabled:
        wrapped = expert_parallel_module().apply_expert_data_parallel(
            model, parallel_dims, training
        )
        logger.info(
            "Expert parallelism: SimpleFSDP over efsdp for %d expert banks",
            wrapped,
        )
      model = apply_simple_fsdp(
          model, parallel_dims=parallel_dims, training=training
      )
    else:
      dp_mesh_names = (
          ["dp_replicate", "fsdp"]
          if parallel_dims.dp_replicate_enabled
          else ["fsdp"]
      )
      apply_fsdp_to_decoder(
          model,
          parallel_dims.get_mesh(dp_mesh_names),
          param_dtype=torchtitan.config.TORCH_DTYPE_MAP[
              training.mixed_precision_param
          ],
          reduce_dtype=torchtitan.config.TORCH_DTYPE_MAP[
              training.mixed_precision_reduce
          ],
          pp_enabled=parallel_dims.pp_enabled,
          cpu_offload=training.enable_cpu_offload,
          reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
          dp_mesh_dims=None,
          enable_symm_mem=parallelism.enable_fsdp_symm_mem,
      )
    logger.info("Applied data parallelism to Muse Spark")

  if tpu_compile_mode == "fullgraph" and compile_config.enable:
    apply_spmd_safe(model)
    model = torch.compile(model, fullgraph=True, backend="tpu")
  return model
