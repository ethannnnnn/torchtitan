# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared helpers for the post-training CPU smokes."""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import shutil
import socket

import torch

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

# Kept in float32 in the debug checkpoint, as in released Qwen3.5 checkpoints.
FLOAT32_KEY_SUFFIXES = (".linear_attn.A_log", ".linear_attn.norm.weight")


def free_port() -> int:
  """An unused local TCP port for the single-rank process group."""
  with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    return sock.getsockname()[1]


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


# The single shard of the debug checkpoint. TorchTitan's state-dict adapters
# read a shard number from every file name in the index, so it has one.
DEBUG_SHARD = "model-00001-of-00001.safetensors"


def _debug_dtype(hf_key: str) -> torch.dtype:
  if hf_key.endswith(FLOAT32_KEY_SUFFIXES):
    return torch.float32
  return torch.bfloat16


def _write_single_shard(
    out: pathlib.Path, tensors: dict[str, torch.Tensor]
) -> None:
  """``DEBUG_SHARD`` plus a Hugging Face index mapping every key to it."""
  from safetensors.torch import save_file

  save_file(tensors, str(out / DEBUG_SHARD), metadata={"format": "pt"})
  index = {"metadata": {}, "weight_map": dict.fromkeys(tensors, DEBUG_SHARD)}
  index_path = out / "model.safetensors.index.json"
  index_path.write_text(json.dumps(index, indent=2))


def make_debug_checkpoint(
    out: pathlib.Path, tokenizer_dir: pathlib.Path
) -> dict[str, torch.Tensor]:
  """Random weights of the debug model, saved as a Hugging Face checkpoint.

  Seeding torch with 0 right before the model is built and initialized makes
  the weights the same in every call and process. They are stored under
  Hugging Face names, in bfloat16 except for the float32 families of
  ``FLOAT32_KEY_SUFFIXES``, with the files of ``TOKENIZER_FILES`` that exist
  in ``tokenizer_dir`` copied alongside. Returns the tensors as written.
  """
  os.makedirs(out, exist_ok=True)
  spec = debug_model_spec()
  torch.manual_seed(0)
  model = spec.model.build()
  model.init_states(buffer_device=None)
  to_hf = spec.state_dict_adapter(spec.model, None).to_hf
  written = {
      key: tensor.detach().clone().contiguous().to(_debug_dtype(key))
      for key, tensor in to_hf(model.state_dict()).items()
  }
  _write_single_shard(out, written)
  found = [n for n in TOKENIZER_FILES if (tokenizer_dir / n).is_file()]
  for name in found:
    shutil.copyfile(tokenizer_dir / name, out / name)
  return written


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


def report(checks: dict[str, bool], tag: str) -> int:
  """Print one ``OK`` or ``FAIL`` line per check, then the overall verdict.

  Lines are flushed one by one, since they interleave with trainer logs.
  Returns the process exit status: 0 if every check holds (no checks count
  as holding), 1 otherwise.
  """
  for name, ok in checks.items():
    marker = "OK" if ok else "FAIL"
    print(f"[{tag}] {marker:<4} {name}", flush=True)
  passed = all(checks.values())
  verdict = "PASS" if passed else "FAIL"
  print(f"[{tag}] {verdict}", flush=True)
  return 0 if passed else 1
