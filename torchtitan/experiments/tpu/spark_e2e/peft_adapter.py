# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Hugging Face PEFT copies of the Muse Spark attention adapters. ASCII only.

The Muse Spark LoRA recipes adapt four projections in every decoder layer.
For layer 0, TorchTitan holds the query, key and value projections at
``layers.0.attention.qkv_linear.wq``, ``.wk`` and ``.wv``.
The output projection of the same layer is ``layers.0.attention.wo``.

Exporting collects these adapters from a model that may be sharded.
They are renamed after the modules of the released Hugging Face checkpoint.
The result is written as a PEFT adapter directory.
Hugging Face PEFT can load that directory.
vLLM can serve it as a LoRA adapter.
``checkpointing/hf_export.py --peft`` merges it into the base weights.
Each matrix in it is stored in float32, whatever dtype the training used.
A tensor name starts with the PEFT root and ends with the matrix; for example
``base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight``
is the A matrix of the query projection of layer 0.

The released checkpoint also contains a vision tower with projections named
like the attention ones.
For that reason the config's ``target_modules`` lists every exported module
path in full and matches nothing else.

Importing writes such a directory back into the live LoRA parameters.

Materializing sharded matrices issues collectives.
Every rank has to call it at the same point and with the same adapter keys.
The keys are visited in sorted order, so all ranks issue matching collectives.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch.distributed.tensor import distribute_tensor, DTensor

from torchtitan.components.checkpointer.utils import canonical_fqn

__all__ = [
    "LoraHyperparams",
    "PeftAdapterError",
    "export_spark_adapter",
    "load_spark_adapter",
    "lora_hyperparams_of",
    "materialize_lora_matrices",
    "pair_lora_matrices",
    "spark_hf_module_name",
    "to_peft_layout",
    "write_spark_adapter_dir",
]

_PEFT_ROOT = "base_model.model."
_HF_DECODER_LAYERS = "model.language_model.layers"
# Adapted projection inside one TorchTitan decoder layer -> its name in the
# released checkpoint's attention block.
_HF_PROJECTION_OF = {
    "attention.qkv_linear.wq": "q_proj",
    "attention.qkv_linear.wk": "k_proj",
    "attention.qkv_linear.wv": "v_proj",
    "attention.wo": "o_proj",
}
_TT_LAYER_PATH = re.compile(r"layers\.(?P<index>[0-9]+)\.(?P<inner>.+)")
_ADAPTER_KEY = re.compile(r"(?P<module>.+)\.lora_(?P<letter>[abAB])\.weight")
_PEFT_TENSOR = re.compile(re.escape(_PEFT_ROOT) + r"(?P<module>.+)\.lora_[AB]\.weight")
# Name segments added by activation checkpointing, torch.compile and FSDP1
# wrappers; they are not part of a module's path.
_WRAPPER_SEGMENTS = ("_checkpoint_wrapped_module", "_orig_mod", "_fsdp_wrapped_module")
_TENSOR_FILE = "adapter_model.safetensors"
_CONFIG_FILE = "adapter_config.json"
_INFO_FILE = "export_info.json"


class PeftAdapterError(RuntimeError):
    """An adapter that cannot be exported, written or imported as requested."""


@dataclass(frozen=True)
class LoraHyperparams:
    """Rank and alpha shared by all adapters of a model.

    ``scale`` is the factor PEFT applies to the adapter output: alpha divided
    by the rank, or by its square root with rank-stabilized LoRA.
    """

    rank: int
    alpha: float
    use_rslora: bool = False
    scale: float = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise PeftAdapterError(f"LoRA rank must be an int of at least 1, not {self.rank!r}")
        alpha = float(self.alpha)
        if not (math.isfinite(alpha) and alpha > 0):
            raise PeftAdapterError(f"LoRA alpha must be a finite number above 0, not {self.alpha!r}")
        divisor = math.sqrt(self.rank) if self.use_rslora else self.rank
        object.__setattr__(self, "scale", alpha / divisor)


def _is_adapted(module: torch.nn.Module) -> bool:
    return all(isinstance(getattr(module, part, None), torch.nn.Module) for part in ("lora_a", "lora_b"))


def _rank_alpha(path: str, module: torch.nn.Module) -> tuple[int, float]:
    if hasattr(module, "_lora_scaling"):
        factor = module._lora_scaling
    else:
        factor = getattr(module, "lora_scaling", None)
    if factor is None:
        raise PeftAdapterError(f"adapted module {path!r} carries no LoRA scaling factor")
    rank = int(module.lora_a.weight.shape[0])
    return rank, float(factor) * rank


def lora_hyperparams_of(model: torch.nn.Module) -> LoraHyperparams:
    """Read rank and alpha off the model's adapters, which must all agree."""
    found = [_rank_alpha(path, module) for path, module in model.named_modules() if _is_adapted(module)]
    if not found:
        raise PeftAdapterError("the model has no LoRA adapter")
    distinct = sorted(set(found))
    if len(distinct) > 1:
        raise PeftAdapterError(f"the adapters disagree on (rank, alpha): {distinct}")
    rank, alpha = distinct[0]
    return LoraHyperparams(rank=rank, alpha=alpha)


def _ab_letters(module: str, letters: set[str]) -> tuple[str, str]:
    if letters == {"a", "b"} or letters == {"A", "B"}:
        first, second = sorted(letters)
        return first, second
    raise PeftAdapterError(
        f"adapter {module!r} needs one A and one B matrix spelled the same way, found lora_{sorted(letters)}")


def pair_lora_matrices(keys: Iterable[str]) -> list[tuple[str, str, str]]:
    """``(module, A key, B key)`` for every adapter among ``keys``, by module path.

    Keys that do not name a LoRA matrix are skipped. Both TorchTitan's lower-case
    and PEFT's upper-case matrix letters are understood, but one module may not mix them.
    """
    letters_by_module: dict[str, set[str]] = defaultdict(set)
    for key in keys:
        match = _ADAPTER_KEY.fullmatch(key)
        if match is not None:
            letters_by_module[match["module"]].add(match["letter"])
    triples = []
    for module in sorted(letters_by_module):
        a, b = _ab_letters(module, letters_by_module[module])
        triples.append((module, f"{module}.lora_{a}.weight", f"{module}.lora_{b}.weight"))
    return triples


def spark_hf_module_name(module: str) -> str:
    """The released checkpoint's name for one adapted TorchTitan projection.

    ``layers.3.attention.wo`` becomes ``model.language_model.layers.3.self_attn.o_proj``.
    Any other path is refused, so that no tensor is stored under a name a
    loader would apply to the wrong module.
    """
    match = _TT_LAYER_PATH.fullmatch(module)
    projection = None if match is None else _HF_PROJECTION_OF.get(match["inner"])
    if projection is None:
        raise PeftAdapterError(f"{module!r} is not one of the adapted Muse Spark attention projections")
    return f"{_HF_DECODER_LAYERS}.{match['index']}.self_attn.{projection}"


def _gathered(value: Any) -> Any:
    if isinstance(value, DTensor):
        return value.full_tensor()
    return value


def materialize_lora_matrices(
    state: Mapping[str, Any], *, keep: bool, dtype: torch.dtype = torch.float32
) -> dict[str, torch.Tensor]:
    """Unsharded CPU copies of the adapter matrices of ``state`` (collective).

    Every rank must call this with the same keys. Matrices are visited module by
    module in sorted order, A before B, and a DTensor is gathered in full. Only
    a rank with ``keep`` stores the copies (detached, contiguous, in ``dtype``);
    the others drop each gathered matrix before the next one and get ``{}``.
    Entries other than adapter matrices are never touched.
    """
    triples = pair_lora_matrices(state)
    kept: dict[str, torch.Tensor] = {}
    for _, a_key, b_key in triples:
        for key in (a_key, b_key):
            whole = _gathered(state[key])
            if keep:
                kept[key] = whole.detach().to(device="cpu", dtype=dtype, copy=True).contiguous()
            del whole
    return kept


def _check_layout_entry(module: str, hf_module: str, a: Any, b: Any) -> None:
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dim() == 2 and b.dim() == 2):
        raise PeftAdapterError(f"{module}: the A and B matrices must both be 2-D tensors")
    if b.shape[1] != a.shape[0]:
        raise PeftAdapterError(
            f"{module}: B has {b.shape[1]} columns but A has {a.shape[0]} rows, so they do not chain")
    if not hf_module or "" in hf_module.split("."):
        raise PeftAdapterError(f"{module}: unusable Hugging Face module path {hf_module!r}")


def to_peft_layout(
    matrices: Mapping[str, torch.Tensor],
    hf_module_name: Callable[[str], str],
    *,
    hyperparams: LoraHyperparams | None = None,
) -> dict[str, torch.Tensor]:
    """Rename adapter matrices to PEFT tensor names; the tensors themselves are reused.

    ``hf_module_name`` turns a TorchTitan module path into a Hugging Face one.
    The result lists modules in sorted order, each A matrix before its B matrix.
    """
    plan = [(module, hf_module_name(module), matrices[a_key], matrices[b_key])
            for module, a_key, b_key in pair_lora_matrices(matrices)]
    if not plan:
        raise PeftAdapterError("there is no complete LoRA A/B pair to rename")
    for entry in plan:
        _check_layout_entry(*entry)
    clashes = sorted(path for path, uses in Counter(hf for _, hf, _, _ in plan).items() if uses > 1)
    if clashes:
        raise PeftAdapterError(f"more than one module would be stored as {clashes}")
    ranks = sorted({int(a.shape[0]) for _, _, a, _ in plan})
    if len(ranks) > 1:
        raise PeftAdapterError(f"the adapters do not share one rank: {ranks}")
    if hyperparams is not None and ranks[0] != hyperparams.rank:
        raise PeftAdapterError(f"the adapters have rank {ranks[0]}, the hyperparameters say {hyperparams.rank}")
    return {f"{_PEFT_ROOT}{hf}.lora_{letter}.weight": matrix
            for _, hf, a, b in plan for letter, matrix in (("A", a), ("B", b))}


def export_spark_adapter(model_parts: Iterable[torch.nn.Module], *, keep: bool) -> dict[str, torch.Tensor]:
    """The model's adapters as float32 PEFT tensors on a ``keep`` rank, ``{}`` elsewhere (collective)."""
    state: dict[str, Any] = {}
    for part in model_parts:
        state.update(part.state_dict())
    matrices = materialize_lora_matrices(state, keep=keep, dtype=torch.float32)
    if not keep:
        return {}
    return to_peft_layout(matrices, spark_hf_module_name)


def write_spark_adapter_dir(
    directory: str | os.PathLike[str],
    peft_tensors: Mapping[str, torch.Tensor],
    hyperparams: LoraHyperparams,
    *,
    base_model_path: str = "",
    export_info: Mapping[str, Any] | None = None,
) -> pathlib.Path:
    """Write a PEFT adapter directory (tensors, config, optional export info) and return its path."""
    modules = set()
    for name in peft_tensors:
        match = _PEFT_TENSOR.fullmatch(name)
        if match is None:
            raise PeftAdapterError(f"{name!r} is not a PEFT LoRA tensor name")
        modules.add(match["module"])
    os.makedirs(directory, exist_ok=True)
    save_file({name: tensor.contiguous() for name, tensor in peft_tensors.items()},
              os.path.join(directory, _TENSOR_FILE))
    config = dict(
        peft_type="LORA",
        task_type="CAUSAL_LM",
        r=hyperparams.rank,
        lora_alpha=hyperparams.alpha,
        use_rslora=bool(hyperparams.use_rslora),
        lora_dropout=0.0,
        bias="none",
        fan_in_fan_out=False,
        inference_mode=True,
        target_modules="^(?:" + "|".join(sorted(re.escape(path) for path in modules)) + ")$",
        base_model_name_or_path=str(base_model_path),
    )
    with open(os.path.join(directory, _CONFIG_FILE), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    if export_info:
        with open(os.path.join(directory, _INFO_FILE), "w", encoding="utf-8") as handle:
            json.dump(export_info, handle, indent=2, default=str)
    return pathlib.Path(directory)


def load_spark_adapter(model: torch.nn.Module, adapter_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Copy a PEFT adapter directory into the model's LoRA parameters.

    Every rank reads the whole tensor file. A DTensor parameter receives its own
    shard of the stored matrix through ``distribute_tensor``, so with sharded
    parameters every rank has to call this. Values are cast to each parameter's
    dtype; tensors in the file that no parameter asks for are ignored.
    """
    stored = load_file(os.path.join(adapter_dir, _TENSOR_FILE))
    live = {canonical_fqn(name, _WRAPPER_SEGMENTS): param for name, param in model.named_parameters()}
    copied = 0
    with torch.no_grad():
        for module, a_key, b_key in pair_lora_matrices(live):
            hf_module = spark_hf_module_name(module)
            for key, letter in ((a_key, "A"), (b_key, "B")):
                name = f"{_PEFT_ROOT}{hf_module}.lora_{letter}.weight"
                if name not in stored:
                    raise PeftAdapterError(f"{_TENSOR_FILE} in {adapter_dir} has no tensor {name!r}")
                target = live[key]
                incoming = stored[name].to(dtype=target.dtype)
                if isinstance(target, DTensor):
                    own_shard = distribute_tensor(incoming, target.device_mesh, target.placements)
                    target.to_local().copy_(own_shard.to_local())
                else:
                    target.copy_(incoming)
                copied += 1
    return {"adapter_dir": str(adapter_dir), "tensors_copied": copied}
