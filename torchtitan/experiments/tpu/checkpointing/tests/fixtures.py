# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared fixtures for the model-agnostic checkpointing tests (CPU, seconds).

The model is TorchTitan's Llama 3 ``debugmodel`` (SDPA attention, so its
backward runs on CPU) with LoRA applied by TorchTitan's ``LoRAConverter``; its
Hugging Face base checkpoint is written through the model's own
``Llama3StateDictAdapter`` (which renames keys, splits the fused QKV
projection and permutes the q/k rows for RoPE, and skips adapter keys).
"""

from __future__ import annotations

import json
import os
import tempfile

import torch
import torch.distributed as dist
import torch.nn.functional as F

SEQ = 32
BATCH = 2
VOCAB = 2048  # Llama 3 debugmodel
LORA_RANK = 4
LORA_ALPHA = 8.0


def ensure_process_group() -> None:
    """A one-rank gloo group (file store, so no TCP port is needed)."""
    if not dist.is_initialized():
        store = tempfile.NamedTemporaryFile(prefix="ckpt-test-pg-", delete=False)
        store.close()
        os.remove(store.name)
        dist.init_process_group("gloo", init_method=f"file://{store.name}", rank=0, world_size=1)


def build_llama3(*, lora: bool, seed: int, targets: tuple[str, ...] = ("wqkv", "wo")):
    """(model, model_spec) of the Llama 3 debugmodel in float32, optionally with LoRA."""
    from torchtitan.components.lora import LoRAConverter
    from torchtitan.models.llama3 import model_registry

    converters = (
        [LoRAConverter.Config(rank=LORA_RANK, alpha=LORA_ALPHA, target_modules=list(targets))]
        if lora
        else None
    )
    spec = model_registry("debugmodel", attn_backend="sdpa", seq_len=SEQ, converters=converters)
    torch.manual_seed(seed)
    model = spec.model.build()
    model.init_states()
    model.float()
    return model, spec


def write_hf_checkpoint(directory: str, state_dict: dict[str, torch.Tensor], spec, num_files: int = 1) -> None:
    """Write ``state_dict`` (model keys) as a Hugging Face safetensors checkpoint."""
    from safetensors.torch import save_file

    hf = spec.state_dict_adapter(spec.model, None).to_hf(dict(state_dict))
    hf = {k: v.detach().contiguous().clone() for k, v in hf.items()}
    os.makedirs(directory, exist_ok=True)
    keys = sorted(hf)
    if num_files == 1:
        save_file(hf, os.path.join(directory, "model.safetensors"), metadata={"format": "pt"})
    else:
        weight_map = {}
        for i in range(num_files):
            name = f"model-{i + 1:05d}-of-{num_files:05d}.safetensors"
            part = {k: hf[k] for j, k in enumerate(keys) if j % num_files == i}
            save_file(part, os.path.join(directory, name), metadata={"format": "pt"})
            weight_map.update(dict.fromkeys(part, name))
        total = sum(t.numel() * t.element_size() for t in hf.values())
        with open(os.path.join(directory, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)
    with open(os.path.join(directory, "config.json"), "w") as f:
        json.dump({"model_type": "llama", "note": "test checkpoint"}, f)


class Counter:
    """Minimal Stateful object for the dataloader / train state."""

    def __init__(self):
        self.step = 0

    def state_dict(self):
        return {"step": torch.tensor(self.step)}

    def load_state_dict(self, sd):
        self.step = int(sd["step"])


def build_manager(model, spec, hf_dir: str, folder: str, interval: int):
    """(checkpointer, optimizers, lr_schedulers, train_state) around TorchTitan's CheckpointManager."""
    from torchtitan.components.checkpointer.dcp import CheckpointManager
    from torchtitan.components.optimizer.lr_scheduler import LRSchedulersContainer
    from torchtitan.components.optimizer.optimizer import OptimizersContainer, ParamGroupConfig

    optimizers = OptimizersContainer.Config(
        param_groups=[ParamGroupConfig(pattern=".*", optimizer_name="AdamW",
                                       optimizer_kwargs={"lr": 3e-3, "weight_decay": 0.01})],
        implementation="for-loop",
    ).build(model_parts=[model])
    lr_schedulers = LRSchedulersContainer.Config(warmup_steps=2).build(optimizers=optimizers, training_steps=8)
    train_state = Counter()
    checkpointer = CheckpointManager(
        CheckpointManager.Config(enable=True, folder="checkpoint", interval=interval,
                                 initial_load_in_hf=True, last_save_model_only=False, keep_latest_k=0),
        dataloader=Counter(), model_parts=[model], optimizers=optimizers, lr_schedulers=lr_schedulers,
        states={"train_state": train_state}, sd_adapter=spec.state_dict_adapter(spec.model, hf_dir),
        base_folder=folder,
    )
    return checkpointer, optimizers, lr_schedulers, train_state


def batch(step: int, rank: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(1000 + 10 * step + rank)
    tokens = torch.randint(0, VOCAB, (BATCH, SEQ), generator=g)
    labels = torch.randint(0, VOCAB, (BATCH, SEQ), generator=g)
    return tokens, labels


def forward(model, tokens: torch.Tensor) -> torch.Tensor:
    b, s = tokens.shape
    return model(tokens.reshape(-1), positions=torch.arange(s).repeat(b))


def train(model, optimizers, lr_schedulers, train_state, checkpointer, until: int, rank: int = 0) -> list[float]:
    """Train to step ``until``, saving through the checkpointer; returns the losses."""
    losses = []
    while train_state.step < until:
        train_state.step += 1
        tokens, labels = batch(train_state.step, rank)
        optimizers.zero_grad()
        loss = F.cross_entropy(forward(model, tokens).float(), labels.reshape(-1))
        loss.backward()
        optimizers.step()
        lr_schedulers.step()
        losses.append(loss.item())
        checkpointer.save(curr_step=train_state.step)
    return losses


def dir_bytes(path: str) -> int:
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs)


def restore_checkpointer_dcp() -> None:
    """Undo ``tune_checkpoint_manager``'s process-wide patch of the checkpointer's ``dcp``."""
    import torch.distributed.checkpoint as dcp

    import torchtitan.components.checkpointer.dcp as checkpointer_module

    checkpointer_module.dcp = dcp
