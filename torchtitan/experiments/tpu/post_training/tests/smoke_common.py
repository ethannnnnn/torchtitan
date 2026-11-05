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
FLOAT32_SUFFIXES = (".linear_attn.A_log", ".linear_attn.norm.weight")


def free_port() -> int:
    """An unused local TCP port for the single-rank process group."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def has_muse_spark() -> bool:
    """True when the Muse Spark TPU package is importable."""
    return importlib.util.find_spec("torchtitan.experiments.tpu.muse_spark") is not None


LORA_CHECKPOINT_MODULE = "torchtitan.experiments.tpu.checkpointing.lora_checkpoint"


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

    return resolve_model_spec(DEBUG.name, DEBUG.flavor, seq_len, **dict(DEBUG.kwargs))


def make_debug_checkpoint(out: pathlib.Path, tokenizer_dir: pathlib.Path) -> dict[str, torch.Tensor]:
    """Random TPU Qwen3.5 debug-model weights in the Hugging Face layout of
    its state-dict adapter, plus the tokenizer files of ``tokenizer_dir`` (a
    Muse Spark tokenizer fits the 248320-row vocabulary) except its chat
    template (see ``TOKENIZER_FILES``)."""
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    spec = debug_model_spec()
    torch.manual_seed(0)
    model = spec.model.build()
    model.init_states(buffer_device=None)
    adapter = spec.state_dict_adapter(spec.model, None)
    hf = {k: v.detach().clone().contiguous() for k, v in adapter.to_hf(model.state_dict()).items()}
    hf = {k: (v.float() if k.endswith(FLOAT32_SUFFIXES) else v.to(torch.bfloat16))
          for k, v in hf.items()}
    shard = "model.safetensors-00001-of-00001.safetensors"
    save_file(hf, str(out / shard))
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {k: shard for k in hf}})
    )
    for name in TOKENIZER_FILES:
        if (tokenizer_dir / name).exists():
            shutil.copy(tokenizer_dir / name, out / name)
    return hf


def debug_base_state(hf: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The model-keyed weights of a debug checkpoint written above."""
    spec = debug_model_spec()
    return spec.state_dict_adapter(spec.model, None).from_hf(dict(hf))


def single_rank_env() -> None:
    """torch.distributed environment for one local rank."""
    os.environ.update(
        RANK="0", LOCAL_RANK="0", WORLD_SIZE="1", LOCAL_WORLD_SIZE="1",
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(free_port()),
    )


def report(checks: dict[str, bool], tag: str) -> int:
    for name, ok in checks.items():
        print(f"[{tag}] {'OK  ' if ok else 'FAIL'} {name}", flush=True)
    passed = all(checks.values())
    print(f"[{tag}] PASS" if passed else f"[{tag}] FAIL", flush=True)
    return 0 if passed else 1
