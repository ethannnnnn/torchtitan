# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LoRA checkpoints: adapter-only resume state and PEFT adapter export/import.

A LoRA run only changes its adapters, so writing the full model at every
checkpoint (hundreds of GB to TB for a large MoE) is wasted I/O. This module
makes a TorchTitan ``CheckpointManager`` save and restore only the trainable
parameters (the ``lora_a`` / ``lora_b`` weights of TorchTitan's
``LoRALinear``) plus the optimizer, LR scheduler, dataloader and train state,
and restores the frozen base weights from the original Hugging Face
checkpoint on every (re)start:

    enable_adapter_only_checkpoints(checkpointer, model_parts)

* save: DCP of the trainable parameters + optimizer state (the optimizer
  only holds trainable parameters) + the small states. For rank-16 adapters
  on the q/k/v/o projections of a 62-layer, 8192-wide model this is about
  51M parameters: ~0.1 GB in bf16, ~0.6 GB with fp32 AdamW moments.
* load: base weights from Hugging Face safetensors through the model's state
  dict adapter (it must skip adapter keys, e.g. a LoRA-aware wrapper), then
  the newest ``step-N`` adapter checkpoint if there is one.

For serving and evaluation, ``save_peft_adapter`` writes a standard PEFT
LoRA adapter (``adapter_model.safetensors`` + ``adapter_config.json``, keys
``base_model.model.<HF module>.lora_A.weight``) from sharded parameters: every
rank joins the gather, rank 0 writes. Module names come from the model's state
dict adapter, and ``lora_B`` rows follow any row permutation it applies to the
base weight (e.g. RoPE); adapters on weights that do not map one-to-one (e.g. a
fused ``wqkv``), or whose columns or values the adapter changes, are refused.
Engines that load LoRA adapters take it directly; ``hf_export --peft`` merges
it into the original safetensors layout for engines that need merged weights.
``load_peft_adapter`` initializes the model's adapters from such a directory
(e.g. SFT adapters as the start of RL).
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
import time
from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn

# MODEL: TorchTitan's key for the model state in CheckpointManager.states.
from torchtitan.components.checkpointer.base import MODEL, ModelWrapper
from torchtitan.components.checkpointer.utils import canonical_fqn

logger = logging.getLogger(__name__)

__all__ = [
    "LORA_A",
    "LORA_B",
    "is_lora_key",
    "TrainableModelWrapper",
    "enable_adapter_only_checkpoints",
    "gather_lora_state_dict",
    "tt_lora_to_peft",
    "peft_to_tt_lora",
    "save_peft_adapter",
    "read_peft_adapter",
    "load_peft_adapter",
]

LORA_A = "lora_a.weight"
LORA_B = "lora_b.weight"
_PEFT_KEY = re.compile(r"^(?:base_model\.model\.)?(.+)\.lora_([AB])(?:\.[^.]+)?\.weight$")


def is_lora_key(key: str) -> bool:
    return key.endswith("." + LORA_A) or key.endswith("." + LORA_B)


# ---------------------------------------------------------------------------
# Adapter-only DCP checkpoints
# ---------------------------------------------------------------------------


def _trainable_names(model_parts: list[nn.Module]) -> set[str]:
    """``state_dict()`` keys of the trainable parameters.

    ``named_parameters()`` keeps the activation-checkpoint wrapper's segment
    (``layers.0._checkpoint_wrapped_module.attention...``), which the wrapper's
    state-dict hook drops; TorchTitan's ``canonical_fqn`` (also behind the
    optimizer's parameter names) strips exactly that segment.
    """
    return {
        canonical_fqn(n) for m in model_parts for n, p in m.named_parameters() if p.requires_grad
    }


class TrainableModelWrapper(ModelWrapper):
    """TorchTitan ``ModelWrapper`` restricted to the trainable parameters."""

    def _get_state_dict(self) -> dict[str, Any]:
        keep = _trainable_names(self.model)
        full = super()._get_state_dict()
        missing = keep - set(full)
        if missing:
            raise KeyError(f"trainable parameters missing from state_dict(): {sorted(missing)[:5]}")
        return {k: v for k, v in full.items() if k in keep}


def enable_adapter_only_checkpoints(checkpointer: Any, model_parts: list[nn.Module]) -> None:
    """Switch a TorchTitan ``CheckpointManager`` to adapter-only checkpoints.

    Call after the manager is built and before ``checkpointer.load()``. The
    manager must be able to load the base model from Hugging Face
    safetensors (``sd_adapter`` with ``hf_assets_path``, or
    ``initial_load_path`` + ``initial_load_in_hf``). ``last_save_in_hf``
    must be off: export adapters with ``save_peft_adapter`` instead.

    After ``load()`` the manager carries ``adapter_resume_step``: the step of
    the adapter checkpoint that was restored, or ``None`` on a fresh start
    (base weights only), and ``adapter_load_seconds`` ({"base": s,
    "adapter": s}).
    """
    if not getattr(checkpointer, "enable", False):
        return
    if getattr(checkpointer, "last_save_in_hf", False):
        raise ValueError("adapter-only checkpoints cannot be saved in HF format; use save_peft_adapter")
    if checkpointer.sd_adapter is None:
        raise ValueError("adapter-only checkpoints need the model's sd_adapter to load the base")
    trainable = _trainable_names(model_parts)
    if not trainable:
        raise ValueError("no trainable parameters; nothing to checkpoint")
    checkpointer.states[MODEL] = TrainableModelWrapper(model_parts)
    checkpointer.adapter_resume_step = None
    checkpointer.adapter_load_seconds = {}

    @torch.no_grad()
    def load(step: int = -1) -> bool:
        storage = checkpointer._storage
        base_id = checkpointer.initial_load_path or checkpointer.sd_adapter.hf_assets_path
        if not base_id:
            raise ValueError("no Hugging Face base checkpoint (initial_load_path / hf_assets_path)")
        checkpointer.adapter_resume_step = None
        begin = time.monotonic()
        # A full view of the model only for this load: ModelWrapper caches the
        # whole state dict (including tensors that state-dict hooks create,
        # e.g. re-laid-out expert weights), so it must not outlive the load.
        full = ModelWrapper(model_parts)
        checkpointer._load_checkpoint(
            {MODEL: full}, base_id, from_hf=True,
            from_quantized=bool(getattr(checkpointer, "initial_load_in_hf_quantized", False)),
        )
        del full
        gc.collect()
        checkpointer.adapter_load_seconds = {"base": time.monotonic() - begin}
        logger.info("[ckpt] base weights loaded from %s in %.1f s", base_id,
                    checkpointer.adapter_load_seconds["base"])
        if not storage.isdir(checkpointer.folder):
            if step != -1:
                raise FileNotFoundError(f"step {step} requested but {checkpointer.folder} does not exist")
            logger.info("[ckpt] no adapter checkpoint folder %s: fresh start", checkpointer.folder)
            return True
        load_step = checkpointer._find_load_step() if step == -1 else step
        if load_step == -1:
            logger.info("[ckpt] no complete adapter checkpoint in %s: fresh start", checkpointer.folder)
            return True
        checkpoint_id = checkpointer._create_checkpoint_id(load_step)
        if not storage.isdir(checkpoint_id):
            raise FileNotFoundError(f"adapter checkpoint step {load_step} not found at {checkpoint_id}")
        begin = time.monotonic()
        checkpointer._load_checkpoint(
            checkpointer._states_to_load(model_only=load_step == 0),
            checkpoint_id, from_hf=False, from_quantized=False,
        )
        checkpointer.adapter_load_seconds["adapter"] = time.monotonic() - begin
        checkpointer.adapter_resume_step = load_step
        logger.info("[ckpt] resumed adapter-only state of step %d from %s in %.1f s",
                    load_step, checkpoint_id, checkpointer.adapter_load_seconds["adapter"])
        return True

    checkpointer.load = load


# ---------------------------------------------------------------------------
# PEFT adapters
# ---------------------------------------------------------------------------


def gather_lora_state_dict(model_parts: nn.Module | list[nn.Module], *, to_cpu: bool = True) -> dict[str, torch.Tensor]:
    """Full (unsharded) LoRA tensors keyed by their ``state_dict()`` key (``canonical_fqn``).

    Collective when parameters are DTensors: every rank must call it.
    """
    from torch.distributed.tensor import DTensor

    parts = [model_parts] if isinstance(model_parts, nn.Module) else list(model_parts)
    out: dict[str, torch.Tensor] = {}
    for m in parts:
        for name, p in m.named_parameters():
            if not is_lora_key(name):
                continue
            t = p.detach()
            if isinstance(t, DTensor):
                t = t.full_tensor()
            out[canonical_fqn(name)] = t.to("cpu") if to_cpu else t
    return out


def _index_probe(shape: tuple[int, ...], dim: int, base: int, digit: int) -> torch.Tensor:
    """Every element holds digit ``digit`` (base ``base``) of its index along ``dim``.

    Constant along the other dim with stride 0, so an adapter that hands the
    tensor through allocates nothing.
    """
    index = torch.arange(shape[dim]) // base**digit % base
    return index.to(torch.float32).unsqueeze(1 - dim).expand(shape)


def _source_index(
    run: Any, shape: tuple[int, ...], dim: int, base: int, first: torch.Tensor | None = None
) -> torch.Tensor | None:
    """For each output row (``dim=0``) or column (``dim=1``), the input one it holds.

    ``run`` maps a probe through the adapter; ``first`` is its output for the
    first probe if already computed. The index is read one base-``base`` digit
    per probe (``base`` keeps every digit exact in the adapter's output dtype).
    ``None`` if an output row (column) mixes input rows (columns) or holds
    values that no probe holds (e.g. scaled ones).
    """
    n = shape[dim]
    index = torch.zeros(n, dtype=torch.int64)
    digit = 0
    while digit == 0 or base**digit < n:
        out = first if digit == 0 and first is not None else run(_index_probe(shape, dim, base, digit))
        out = out.movedim(dim, 0)
        line = out[:, :1]
        if not torch.equal(out, line.expand_as(out)):
            return None
        values = line[:, 0].to(torch.float64)
        if not torch.equal(values, values.round()) or values.min() < 0 or values.max() >= base:
            return None
        index += values.to(torch.int64) * base**digit
        digit += 1
    return index


def _hf_module_for(
    adapter: Any, weight_key: str, shape: tuple[int, ...]
) -> tuple[str, torch.Tensor | None]:
    """HF module path of a TorchTitan weight and the row order of its HF layout.

    Returns ``(module, rows)``: ``rows`` is ``None`` when the adapter only
    renames the weight (and possibly casts it), else the permutation with
    ``hf_weight[i] == weight[rows[i]]`` (e.g. a RoPE row permutation of q/k
    projections), which ``lora_B`` must follow since ``P (B A) == (P B) A``.
    Any other re-layout (several output tensors, a changed shape, moved or
    mixed columns, changed values) raises ``NotImplementedError``.
    """
    to_hf = getattr(adapter, "to_hf_base", adapter.to_hf)
    mapped = to_hf({weight_key: torch.empty(shape, dtype=torch.float32, device="meta")})
    if len(mapped) != 1:
        raise NotImplementedError(f"{weight_key} does not map one-to-one ({sorted(mapped)})")
    (hf_key, value), = mapped.items()
    if not hf_key.endswith(".weight") or tuple(value.shape) != tuple(shape):
        raise NotImplementedError(f"{weight_key} -> {hf_key} is not a plain rename")
    if not value.dtype.is_floating_point:
        raise NotImplementedError(f"{weight_key} -> {hf_key} changes the values (dtype {value.dtype})")
    module = hf_key[: -len(".weight")]

    def run(probe: torch.Tensor) -> torch.Tensor:
        try:
            (out,) = to_hf({weight_key: probe}).values()
        except RuntimeError:  # e.g. the adapter views its input: retry with a contiguous probe
            (out,) = to_hf({weight_key: probe.contiguous()}).values()
        if tuple(out.shape) != tuple(shape):
            raise NotImplementedError(f"{weight_key} -> {hf_key} changes the shape")
        return out

    # Value probes hold the row (column) index of each element, one digit per
    # probe in a base whose digits the output dtype holds exactly (256 for bf16).
    base = int(2 / torch.finfo(value.dtype).eps)
    probe = _index_probe(shape, 0, base, 0)
    first = run(probe)
    if first is probe:
        return module, None  # handed through unchanged: a plain rename
    out_features, in_features = shape
    rows = _source_index(run, shape, 0, base, first)
    cols = _source_index(run, shape, 1, base)
    if (
        rows is None
        or cols is None
        or not torch.equal(cols, torch.arange(in_features))
        or not torch.equal(torch.sort(rows).values, torch.arange(out_features))
    ):
        raise NotImplementedError(f"{weight_key} -> {hf_key} is neither a rename nor a row permutation")
    return module, None if torch.equal(rows, torch.arange(out_features)) else rows


def tt_lora_to_peft(lora: dict[str, torch.Tensor], adapter: Any) -> dict[str, torch.Tensor]:
    """``<tt module>.lora_a.weight`` -> ``base_model.model.<hf module>.lora_A.weight``.

    Module names come from the model's state-dict adapter; ``lora_B`` rows
    follow any row permutation the adapter applies to the base weight.
    """
    out: dict[str, torch.Tensor] = {}
    prefixes = sorted(k[: -len(LORA_A) - 1] for k in lora if k.endswith("." + LORA_A))
    for prefix in prefixes:
        a, b = lora[f"{prefix}.{LORA_A}"], lora[f"{prefix}.{LORA_B}"]
        module, rows = _hf_module_for(adapter, f"{prefix}.weight", (b.shape[0], a.shape[1]))
        out[f"base_model.model.{module}.lora_A.weight"] = a
        out[f"base_model.model.{module}.lora_B.weight"] = b if rows is None else b[rows]
    if len(out) != len(lora):
        raise ValueError("unpaired LoRA tensors")
    return out


def peft_to_tt_lora(
    peft: dict[str, torch.Tensor], adapter: Any, model_lora_shapes: dict[str, tuple[int, ...]]
) -> dict[str, torch.Tensor]:
    """Inverse of ``tt_lora_to_peft`` for a model whose LoRA tensors have the given shapes."""
    pairs: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for k, v in peft.items():
        m = _PEFT_KEY.match(k)
        if m is None:
            raise ValueError(f"unexpected adapter tensor {k}")
        pairs[m.group(1)][m.group(2)] = v
    out: dict[str, torch.Tensor] = {}
    used: set[str] = set()
    prefixes = sorted(k[: -len(LORA_A) - 1] for k in model_lora_shapes if k.endswith("." + LORA_A))
    for prefix in prefixes:
        a_shape = model_lora_shapes[f"{prefix}.{LORA_A}"]
        b_shape = model_lora_shapes[f"{prefix}.{LORA_B}"]
        module, rows = _hf_module_for(adapter, f"{prefix}.weight", (b_shape[0], a_shape[1]))
        if module not in pairs:
            continue
        ab = pairs[module]
        if tuple(ab["A"].shape) != tuple(a_shape) or tuple(ab["B"].shape) != tuple(b_shape):
            raise ValueError(f"{module}: adapter shapes {tuple(ab['A'].shape)}/{tuple(ab['B'].shape)} "
                             f"!= model {tuple(a_shape)}/{tuple(b_shape)}")
        out[f"{prefix}.{LORA_A}"] = ab["A"]
        if rows is None:
            out[f"{prefix}.{LORA_B}"] = ab["B"]
        else:  # undo the HF row order: model row rows[i] is HF row i
            inverse = torch.empty_like(rows)
            inverse[rows] = torch.arange(rows.numel())
            out[f"{prefix}.{LORA_B}"] = ab["B"][inverse]
        used.add(module)
    unused = sorted(set(pairs) - used)
    if unused:
        raise KeyError(f"adapter modules with no LoRA slot in the model: {unused[:5]}")
    return out


def save_peft_adapter(
    model_parts: nn.Module | list[nn.Module],
    adapter: Any,
    out_dir: str,
    *,
    rank: int,
    alpha: float,
    base_model_name_or_path: str = "",
    dtype: torch.dtype | None = torch.bfloat16,
    storage_options: dict | None = None,
) -> dict[str, Any] | None:
    """Gather the adapters (all ranks) and write a PEFT LoRA adapter (rank 0).

    ``out_dir`` may be local, a FUSE mount, or an fsspec URL. Returns a small
    report on the writing rank, ``None`` elsewhere.
    """
    import torch.distributed as dist

    lora = gather_lora_state_dict(model_parts)
    if dist.is_initialized() and dist.get_rank() != 0:
        return None
    peft = tt_lora_to_peft(lora, adapter)
    if dtype is not None:
        peft = {k: v.to(dtype) for k, v in peft.items()}
    return write_peft_dir(peft, out_dir, rank=rank, alpha=alpha,
                          base_model_name_or_path=base_model_name_or_path,
                          storage_options=storage_options)


def write_peft_dir(
    peft: dict[str, torch.Tensor],
    out_dir: str,
    *,
    rank: int,
    alpha: float,
    base_model_name_or_path: str = "",
    storage_options: dict | None = None,
) -> dict[str, Any]:
    from safetensors.torch import save as st_save

    modules = sorted({_PEFT_KEY.match(k).group(1).rsplit(".", 1)[-1] for k in peft})
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": int(rank),
        "lora_alpha": float(alpha),
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": modules,
        "base_model_name_or_path": base_model_name_or_path,
        "fan_in_fan_out": False,
        "use_rslora": False,
        "inference_mode": True,
    }
    blob = st_save({k: v.detach().contiguous().cpu() for k, v in sorted(peft.items())},
                   metadata={"format": "pt"})
    _write(os.path.join(out_dir, "adapter_model.safetensors") if "://" not in out_dir
           else out_dir.rstrip("/") + "/adapter_model.safetensors", blob, storage_options)
    _write(os.path.join(out_dir, "adapter_config.json") if "://" not in out_dir
           else out_dir.rstrip("/") + "/adapter_config.json",
           (json.dumps(config, indent=2) + "\n").encode(), storage_options)
    return {"out_dir": out_dir, "tensors": len(peft), "bytes": len(blob),
            "params": sum(v.numel() for v in peft.values()), "target_modules": modules}


def _write(path: str, data: bytes, storage_options: dict | None) -> None:
    if "://" in path and not path.startswith("file://"):
        import fsspec

        with fsspec.open(path, "wb", **(storage_options or {})) as f:
            f.write(data)
        return
    path = path[len("file://"):] if path.startswith("file://") else path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".partial"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def read_peft_adapter(adapter_dir: str, storage_options: dict | None = None) -> tuple[dict, dict[str, torch.Tensor]]:
    """(adapter_config, tensors) of a PEFT LoRA adapter directory."""
    from safetensors.torch import load as st_load

    def read(name: str) -> bytes:
        path = adapter_dir.rstrip("/") + "/" + name
        if "://" in path and not path.startswith("file://"):
            import fsspec

            fs, p = fsspec.core.url_to_fs(path, **(storage_options or {}))
            with fs.open(p, "rb", cache_type="readahead") as f:
                return f.read()
        with open(path[len("file://"):] if path.startswith("file://") else path, "rb") as f:
            return f.read()

    config = json.loads(read("adapter_config.json"))
    return config, st_load(read("adapter_model.safetensors"))


@torch.no_grad()
def load_peft_adapter(model_parts: nn.Module | list[nn.Module], adapter: Any, adapter_dir: str,
                      storage_options: dict | None = None) -> int:
    """Copy a PEFT adapter into the model's (possibly sharded) LoRA parameters.

    Every rank reads the (small) adapter and copies its own shard. Returns the
    number of parameters set.
    """
    from torch.distributed.tensor import DTensor, distribute_tensor

    parts = [model_parts] if isinstance(model_parts, nn.Module) else list(model_parts)
    _, peft = read_peft_adapter(adapter_dir, storage_options)
    params = {canonical_fqn(n): p for m in parts for n, p in m.named_parameters() if is_lora_key(n)}
    tt = peft_to_tt_lora(peft, adapter, {n: tuple(p.shape) for n, p in params.items()})
    missing = sorted(set(params) - set(tt))
    if missing:
        raise KeyError(f"adapter does not cover LoRA parameters {missing[:5]}")
    for name, value in tt.items():
        p = params[name]
        if isinstance(p, DTensor):
            shard = distribute_tensor(value.to(p.dtype), p.device_mesh, p.placements)
            p.to_local().copy_(shard.to_local())
        else:
            if tuple(p.shape) != tuple(value.shape):
                raise ValueError(f"{name}: adapter shape {tuple(value.shape)} != {tuple(p.shape)}")
            p.copy_(value.to(p.dtype))
    return len(tt)
