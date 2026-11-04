# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""PEFT layout of Muse Spark attention adapters: export and import. ASCII only.

A LoRA run of the Muse Spark port trains rank-r adapters on the attention
projections. TorchTitan's ``LoRALinear`` (``torchtitan.components.lora``)
registers them as ``<module>.lora_a`` (``[r, in]``) and ``<module>.lora_b``
(``[out, r]``) with ``scaling = alpha / r``. This module writes them in the
PEFT layout that Hugging Face PEFT, vLLM and ``checkpointing/hf_export.py
--peft`` read::

    <dir>/adapter_model.safetensors   base_model.model.<hf module>.lora_A.weight  [r, in]
                                      base_model.model.<hf module>.lora_B.weight  [out, r]
    <dir>/adapter_config.json         r, lora_alpha, target_modules, ...
    <dir>/export_info.json            step, tensor and parameter counts, seconds

``<hf module>`` is the Hugging Face module path of the adapted projection
(``model.language_model.layers.3.self_attn.q_proj``); the port keeps q/k/v
under ``attention.qkv_linear`` and the output projection at
``attention.wo`` (``SPARK_TT_LORA_MODULE_PATTERNS``). Tensors are float32.
``target_modules`` is an anchored regex over the exact module paths, so a
loader cannot attach the adapter to same-named layers elsewhere (for example
a vision tower's ``q_proj`` in the multimodal checkpoint).

The gather is a collective: every rank walks the adapter keys in sorted order
and calls ``full_tensor()`` on each sharded parameter; only the writing rank
keeps CPU copies. Base weights are never gathered.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

PEFT_PREFIX = "base_model.model."

# (lora A suffix, lora B suffix) naming conventions, tried in order.
_LORA_SUFFIXES = (
    ("lora_a.weight", "lora_b.weight"),  # torchtitan.components.lora.LoRALinear
    ("lora_A.weight", "lora_B.weight"),  # PEFT
)

# Muse Spark TorchTitan module paths -> Hugging Face module paths of the
# attention projections.
SPARK_TT_LORA_MODULE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"layers\.(\d+)\.attention\.qkv_linear\.wq", r"model.language_model.layers.\1.self_attn.q_proj"),
    (r"layers\.(\d+)\.attention\.qkv_linear\.wk", r"model.language_model.layers.\1.self_attn.k_proj"),
    (r"layers\.(\d+)\.attention\.qkv_linear\.wv", r"model.language_model.layers.\1.self_attn.v_proj"),
    (r"layers\.(\d+)\.attention\.wo", r"model.language_model.layers.\1.self_attn.o_proj"),
)

# Module wrappers whose names appear in ``named_parameters()`` but not in
# ``state_dict()`` keys: activation checkpointing, torch.compile, FSDP1.
_WRAPPER_SEGMENTS = ("_checkpoint_wrapped_module.", "_orig_mod.", "_fsdp_wrapped_module.")


class AdapterError(RuntimeError):
    """The adapter tensors do not form a valid PEFT LoRA adapter."""


@dataclass(frozen=True)
class AdapterSpec:
    """LoRA hyper-parameters a loader needs to rebuild ``scaling``."""

    rank: int
    alpha: float
    use_rslora: bool = False

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank <= 0:
            raise AdapterError(f"LoRA rank must be a positive int, got {self.rank!r}")
        if not math.isfinite(float(self.alpha)) or float(self.alpha) <= 0:
            raise AdapterError(f"LoRA alpha must be positive, got {self.alpha!r}")

    @property
    def scaling(self) -> float:
        denominator = math.sqrt(self.rank) if self.use_rslora else self.rank
        return float(self.alpha) / denominator


def canonical_fqn(name: str) -> str:
    """Parameter name without activation-checkpoint / compile / FSDP wrapper segments
    (``named_parameters`` keeps them; ``state_dict`` hooks strip them)."""
    for segment in _WRAPPER_SEGMENTS:
        name = name.replace(segment, "")
    return name


def lora_module_keys(keys: Iterable[str]) -> list[tuple[str, str, str]]:
    """``(module, a_key, b_key)`` for every adapted module, sorted by module.

    Raises when an adapter has only one of its two matrices.
    """
    keyset = set(keys)
    found: dict[str, tuple[str, str]] = {}
    for key in keyset:
        for a_suffix, b_suffix in _LORA_SUFFIXES:
            if key.endswith("." + a_suffix):
                module = key[: -len(a_suffix) - 1]
                b_key = f"{module}.{b_suffix}"
                if b_key not in keyset:
                    raise AdapterError(f"adapter {module} has {a_suffix} but no {b_suffix}")
                if module in found:
                    raise AdapterError(f"adapter {module} appears under two naming styles")
                found[module] = (key, b_key)
            elif key.endswith("." + b_suffix):
                module = key[: -len(b_suffix) - 1]
                if f"{module}.{a_suffix}" not in keyset:
                    raise AdapterError(f"adapter {module} has {b_suffix} but no {a_suffix}")
    return [(module, *found[module]) for module in sorted(found)]


def regex_module_map(patterns: Iterable[tuple[str, str]]) -> Callable[[str], str]:
    """Trainer module path -> HF module path from ``(regex, replacement)`` pairs.

    The first pattern that ``re.fullmatch``-es wins; an unmatched path raises,
    so a renamed model can never export an adapter under a wrong name.
    """
    compiled = [(re.compile(pattern), replacement) for pattern, replacement in patterns]

    def mapping(module: str) -> str:
        for pattern, replacement in compiled:
            if pattern.fullmatch(module):
                return pattern.sub(replacement, module)
        raise AdapterError(f"no HF module mapping for trainer module {module!r}")

    return mapping


def spark_module_map() -> Callable[[str], str]:
    return regex_module_map(SPARK_TT_LORA_MODULE_PATTERNS)


def adapter_spec_from_model(model: torch.nn.Module) -> AdapterSpec:
    """Rank and alpha from the model's LoRA modules (one value for all).

    Reads TorchTitan's ``LoRALinear`` (``lora_a`` / ``lora_b`` submodules and
    ``_lora_scaling``) or any module exposing ``lora_a``, ``lora_b`` and
    ``lora_scaling``.
    """
    specs = set()
    for name, module in model.named_modules():
        lora_a = getattr(module, "lora_a", None)
        lora_b = getattr(module, "lora_b", None)
        if not isinstance(lora_a, torch.nn.Module) or not isinstance(lora_b, torch.nn.Module):
            continue
        scaling = getattr(module, "_lora_scaling", getattr(module, "lora_scaling", None))
        if scaling is None:
            raise AdapterError(f"LoRA module {name} has no scaling attribute")
        rank = int(lora_a.weight.shape[0])
        specs.add((rank, float(scaling) * rank))
    if not specs:
        raise AdapterError("model has no LoRA modules")
    if len(specs) != 1:
        raise AdapterError(f"LoRA modules disagree on (rank, alpha): {sorted(specs)}")
    (rank, alpha), = specs
    return AdapterSpec(rank=rank, alpha=alpha)


def gather_lora_tensors(
    state: Mapping[str, Any],
    *,
    keep: bool,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Gather only the adapter tensors of a (possibly sharded) state dict.

    A collective: every rank must call it with the same keys. ``keep`` ranks
    return CPU copies in ``dtype``; the others return {}.
    """
    from torch.distributed.tensor import DTensor

    gathered: dict[str, torch.Tensor] = {}
    for _module, a_key, b_key in lora_module_keys(state.keys()):
        for key in (a_key, b_key):
            value = state[key]
            full = value.full_tensor() if isinstance(value, DTensor) else value
            if keep:
                gathered[key] = full.detach().to("cpu", dtype).contiguous()
            del full
    return gathered


def peft_adapter_tensors(
    lora: Mapping[str, torch.Tensor],
    module_map: Callable[[str], str],
    *,
    spec: AdapterSpec | None = None,
) -> dict[str, torch.Tensor]:
    """Rename gathered adapter tensors to the PEFT layout and validate shapes.

    Every adapter must be ``A [r, in]`` and ``B [out, r]`` with one common
    ``r`` (equal to ``spec.rank`` when given); HF names must be unique.
    """
    out: dict[str, torch.Tensor] = {}
    ranks = set()
    for module, a_key, b_key in lora_module_keys(lora.keys()):
        a, b = lora[a_key], lora[b_key]
        if a.dim() != 2 or b.dim() != 2 or b.shape[1] != a.shape[0]:
            raise AdapterError(
                f"adapter {module}: A {tuple(a.shape)} and B {tuple(b.shape)} are not [r, in] / [out, r]")
        ranks.add(int(a.shape[0]))
        hf = module_map(module)
        if not hf or hf.startswith(".") or hf.endswith(".") or ".." in hf:
            raise AdapterError(f"module map produced an invalid HF path {hf!r} for {module}")
        a_name = f"{PEFT_PREFIX}{hf}.lora_A.weight"
        b_name = f"{PEFT_PREFIX}{hf}.lora_B.weight"
        if a_name in out:
            raise AdapterError(f"two trainer modules map to HF module {hf}")
        out[a_name] = a
        out[b_name] = b
    if not out:
        raise AdapterError("no adapter tensors to export")
    if len(ranks) != 1:
        raise AdapterError(f"adapters disagree on rank: {sorted(ranks)}")
    if spec is not None and ranks != {spec.rank}:
        raise AdapterError(f"adapter rank {ranks} differs from spec rank {spec.rank}")
    return out


def target_module_names(tensors: Mapping[str, torch.Tensor]) -> list[str]:
    """Sorted HF module paths present in a PEFT-layout tensor dict."""
    names = set()
    for name in tensors:
        if not name.startswith(PEFT_PREFIX) or not name.endswith((".lora_A.weight", ".lora_B.weight")):
            raise AdapterError(f"not a PEFT adapter tensor name: {name!r}")
        names.add(name[len(PEFT_PREFIX): -len(".lora_A.weight")])
    return sorted(names)


def peft_adapter_config(
    spec: AdapterSpec,
    tensors: Mapping[str, torch.Tensor],
    *,
    base_model_name_or_path: str = "",
) -> dict[str, Any]:
    """``adapter_config.json`` content matching the tensors (PEFT schema)."""
    modules = target_module_names(tensors)
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": spec.rank,
        "lora_alpha": spec.alpha,
        "use_rslora": spec.use_rslora,
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "target_modules": "^(?:" + "|".join(re.escape(m) for m in modules) + ")$",
        "base_model_name_or_path": base_model_name_or_path,
    }


def gather_peft_adapter(model_parts, *, keep: bool) -> dict[str, torch.Tensor]:
    """PEFT-layout adapter tensors (float32, CPU) on ``keep`` ranks; collective."""
    state = {k: v for m in model_parts for k, v in m.state_dict().items()}
    gathered = gather_lora_tensors(state, keep=keep, dtype=torch.float32)
    if not keep:
        return {}
    return peft_adapter_tensors(gathered, spark_module_map())


def save_peft_dir(out_dir: Path, tensors: dict[str, torch.Tensor], spec: AdapterSpec,
                  *, base: str = "", extra: dict[str, Any] | None = None) -> Path:
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(out_dir / "adapter_model.safetensors"))
    config = peft_adapter_config(spec, tensors, base_model_name_or_path=base)
    (out_dir / "adapter_config.json").write_text(json.dumps(config, indent=2))
    if extra:
        (out_dir / "export_info.json").write_text(json.dumps(extra, indent=2, default=str))
    return out_dir


def load_peft_into_model(model: torch.nn.Module, adapter_dir: str | Path) -> dict[str, Any]:
    """Copy a PEFT adapter (``save_peft_dir`` layout) into the live LoRA params.

    Works with sharded (DTensor) parameters: each rank reads the full tensor
    and keeps its own shard via ``distribute_tensor`` placements.
    """
    from safetensors.torch import load_file
    from torch.distributed.tensor import DTensor, distribute_tensor

    tensors = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
    mapping = spark_module_map()
    params = {canonical_fqn(k): v for k, v in model.named_parameters()}
    loaded = 0
    for module, a_key, b_key in lora_module_keys(params.keys()):
        hf = mapping(module)
        for key, suffix in ((a_key, "lora_A"), (b_key, "lora_B")):
            src = tensors[f"{PEFT_PREFIX}{hf}.{suffix}.weight"]
            param = params[key]
            with torch.no_grad():
                if isinstance(param, DTensor):
                    local = distribute_tensor(
                        src.to(param.dtype), param.device_mesh, param.placements
                    ).to_local()
                    param.to_local().copy_(local)
                else:
                    param.copy_(src.to(param.dtype))
            loaded += 1
    return {"adapter_dir": str(adapter_dir), "tensors": loaded}
