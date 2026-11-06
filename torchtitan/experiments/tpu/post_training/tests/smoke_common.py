# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared helpers for the post-training CPU smokes."""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import shutil
import socket

import torch
from safetensors.torch import save_file

# Copied into the debug checkpoint. The chat template stays in the tokenizer
# directory, from which the SFT loader reads it (``--dataloader.hf-tokenizer-path``):
# TorchTitan's own tokenizer compiles any ``chat_template.jinja`` of the assets
# directory at start-up, and its Jinja environment does not know the
# ``{% generation %}`` blocks of templates that mark assistant tokens.
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "special_tokens_map.json",
)

# Debug-checkpoint keys written in float32 instead of bfloat16: the decay logs
# and gated norms of Qwen3.5's GatedDeltaNet layers, which the released
# checkpoints keep in float32 as well.
FLOAT32_KEY_SUFFIXES = (".linear_attn.A_log", ".linear_attn.norm.weight")


# Interface the free-port probe binds to (port 0: the OS picks the port).
_LOOPBACK_HOST = "127.0.0.1"


def free_port() -> int:
  """A TCP port on the loopback interface that nothing uses right now (the
  single-rank environment's ``MASTER_PORT``)."""
  import contextlib

  with contextlib.closing(
      socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  ) as probe:
    probe.bind((_LOOPBACK_HOST, 0))
    _, port = probe.getsockname()
  return port


def has_muse_spark() -> bool:
  """True when the Muse Spark TPU package is importable."""
  return (
      importlib.util.find_spec("torchtitan.experiments.tpu.muse_spark")
      is not None
  )


LORA_CHECKPOINT_MODULE = (
    "torchtitan.experiments.tpu.checkpointing.lora_checkpoint"
)


def has_lora_checkpoint() -> bool:
  """True when the adapter-only checkpoint support (``sft.lora_save=adapter``)
  of ``checkpointing/lora_checkpoint.py`` is present. Only a missing module or
  ``checkpointing`` package reads as absent: a ``checkpointing`` package that
  fails to import raises, so the checks that need it do not skip silently."""
  try:
    return importlib.util.find_spec(LORA_CHECKPOINT_MODULE) is not None
  except ModuleNotFoundError as exc:
    if exc.name == LORA_CHECKPOINT_MODULE.rpartition(".")[0]:
      return False
    raise


def debug_model_spec(seq_len: int = 512):
  """The TPU Qwen3.5 debug model of the CPU recipes (no LoRA)."""
  from torchtitan.experiments.tpu.post_training.config_registry import resolve_model_spec
  from torchtitan.experiments.tpu.post_training.tests.cpu_recipes import DEBUG

  return resolve_model_spec(
      DEBUG.name, DEBUG.flavor, seq_len, **dict(DEBUG.kwargs)
  )


# The debug checkpoint's only shard. Its name carries the shard numbering that
# TorchTitan's Hugging Face state-dict adapters look for.
DEBUG_SHARD_FILE = "model-00001-of-00001.safetensors"


def _debug_hf_weights() -> dict[str, torch.Tensor]:
  """The debug model's initial weights under Hugging Face keys, as stored."""
  debug_spec = debug_model_spec()
  # Seeding inside a forked generator scope fixes the weights and hands the
  # caller's random stream back exactly as it was.
  with torch.random.fork_rng(devices=[]):
    torch.manual_seed(0)
    network = debug_spec.model.build()
    network.init_states(buffer_device=None)
  hf_state = debug_spec.state_dict_adapter(debug_spec.model, None).to_hf(
      network.state_dict()
  )
  wide = {key for key in hf_state if key.endswith(FLOAT32_KEY_SUFFIXES)}
  return {
      key: weight.detach().to(
          torch.float32 if key in wide else torch.bfloat16,
          memory_format=torch.contiguous_format,
          copy=True,
      )
      for key, weight in hf_state.items()
  }


def _write_shard_with_index(
    folder: str | os.PathLike, tensors: dict[str, torch.Tensor]
) -> None:
  """Save ``tensors`` as one safetensors shard plus its weight-map index."""
  save_file(
      tensors, os.path.join(folder, DEBUG_SHARD_FILE), metadata={"format": "pt"}
  )
  weight_map = dict.fromkeys(tensors, DEBUG_SHARD_FILE)
  with open(os.path.join(folder, "model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)


def make_debug_checkpoint(
    destination: str | os.PathLike, tokenizer_source: str | os.PathLike
) -> dict[str, torch.Tensor]:
  """Write the debug model as a Hugging Face checkpoint; return its tensors.

  ``destination`` (created if needed) receives one safetensors shard plus its
  weight-map index, together with copies of whichever ``TOKENIZER_FILES``
  exist in ``tokenizer_source``. Every call writes the same weights, and the
  caller's torch random state is left as it was.
  """
  os.makedirs(destination, exist_ok=True)
  tensors = _debug_hf_weights()
  _write_shard_with_index(destination, tensors)
  sources = (os.path.join(tokenizer_source, name) for name in TOKENIZER_FILES)
  for source in filter(os.path.isfile, sources):
    target = os.path.join(destination, os.path.basename(source))
    shutil.copyfile(source, target)
  return tensors


def debug_base_state(hf: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
  """The model-keyed weights of a debug checkpoint written above."""
  spec = debug_model_spec()
  return spec.state_dict_adapter(spec.model, None).from_hf(dict(hf))


# torch.distributed env:// settings of a one-process group on this host.
_SINGLE_RANK = (
    ("RANK", "0"),
    ("LOCAL_RANK", "0"),
    ("WORLD_SIZE", "1"),
    ("LOCAL_WORLD_SIZE", "1"),
    ("MASTER_ADDR", "127.0.0.1"),
)


def single_rank_env() -> None:
  """Point torch.distributed at a fresh one-process group (env:// method).

  Overwrites the variables of ``_SINGLE_RANK`` and ``MASTER_PORT``, a free
  port picked anew on each call, and touches nothing else.
  """
  for name, value in (*_SINGLE_RANK, ("MASTER_PORT", str(free_port()))):
    os.environ[name] = value


# Check lines are flushed as they are printed, so they keep their place among
# the trainer's log lines.
_print_flushed = functools.partial(print, flush=True)


def report(checks: dict[str, bool], tag: str) -> int:
  """Print each check's outcome, then a verdict line; return the exit code.

  Check lines read ``[<tag>] OK   <name>`` or ``[<tag>] FAIL <name>``, in the
  order of ``checks``; the verdict line is ``[<tag>] PASS`` when no check
  failed and ``[<tag>] FAIL`` otherwise. PASS returns 0 and FAIL returns 1.
  """
  failed = [name for name in checks if not checks[name]]
  for name in checks:
    marker = "FAIL" if name in failed else "OK"
    _print_flushed(f"[{tag}] {marker:<4} {name}")
  _print_flushed(f"[{tag}] {'FAIL' if failed else 'PASS'}")
  return int(bool(failed))
