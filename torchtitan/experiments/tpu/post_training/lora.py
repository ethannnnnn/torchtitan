# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LoRA on attention projections, on top of TorchTitan's ``LoRAConverter``.

TorchTitan already ships LoRA (``torchtitan.components.lora``): a model-config
converter that swaps target ``Linear`` configs for a ``LoRALinear`` subclass
(frozen base weight, trainable ``lora_a`` [r, in] kaiming-initialized and
``lora_b`` [out, r] zero-initialized, output ``base(x) + alpha / r *
lora_b(lora_a(x))``) and freezes every other module's parameters. The
optimizer container only collects ``requires_grad`` parameters and the TPU
gradient-norm helper skips parameters without gradients, so frozen base
weights cost no optimizer state. This module adds what the post-training
recipes need around it:

* ``AttentionLoRAConverter`` matches targets by full module path (default:
  the q / k / v / o projections under an attention module, in the TorchTitan
  and Hugging Face naming styles) instead of the last path segment only, so a
  projection that happens to share a name elsewhere is never adapted.
* ``apply_lora`` converts a ``ModelSpec`` in place and wraps its state-dict
  adapter: the Hugging Face load path skips the adapter tensors (the base
  checkpoint has none; ``ModelWrapper`` loads non-strictly, so they keep their
  initialization), and ``to_hf_for_export`` -- used by the RL weight publisher
  -- folds ``W + alpha / r * B @ A`` into the base weights so the rollout
  engine receives ordinary merged weights.
* ``merge_lora_state_dict`` / ``split_lora_state_dict`` operate on gathered
  state dicts; ``peft_adapter_state_dict`` renames adapter tensors to the
  PEFT layout (``base_model.model.<hf module>.lora_A.weight``) and lays out
  lora_B rows through the model's own adapter (renames and any RoPE row
  permutation) for engines that load adapters directly.
* A fused ``wqkv`` projection (``FusedQKVLinear``, saved as separate ``wq`` /
  ``wk`` / ``wv`` by its state-dict hooks) keeps one adapter during training;
  its merge and PEFT export split lora_B rows with the module's head layout
  (``FusedLayout``) into q / k / v parts that share lora_A.

LoRA parameters are ordinary parameters of the converted model, so the
model's own parallelize function shards them like any other weight.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable

import torch

from torchtitan.components.lora import LoRAConverter, _get_lora_cls, _make_frozen_config
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger

LORA_A = "lora_a.weight"
LORA_B = "lora_b.weight"

# Logical projection -> module names in TorchTitan / Hugging Face naming.
PROJECTION_NAMES: dict[str, tuple[str, ...]] = {
    "q": ("wq", "q_proj"),
    "k": ("wk", "wkv", "k_proj"),
    "v": ("wv", "wkv", "v_proj"),
    "o": ("wo", "o_proj"),
    "qkv": ("wqkv", "qkv_proj"),
}


def target_regex(projections: tuple[str, ...] = ("q", "k", "v", "o")) -> str:
  """Full-path regex for the given logical attention projections."""
  names: list[str] = []
  for projection in projections:
    if projection not in PROJECTION_NAMES:
      raise ValueError(
          f"unknown projection {projection!r}; use {sorted(PROJECTION_NAMES)}"
      )
    names.extend(PROJECTION_NAMES[projection])
  if {"q", "k", "v"} <= set(projections):
    names.extend(PROJECTION_NAMES["qkv"])
  alternatives = "|".join(sorted(set(names)))
  return (
      rf"(?:^|.*\.)(?:attention|self_attn|attn)(?:\.[^.]+)*\.(?:{alternatives})"
  )


@dataclass(frozen=True)
class FusedLayout:
  """Head layout of a fused ``wqkv`` projection (``FusedQKVLinear``).

  The fused output dimension is ``[n_kv_heads, heads_per_kv + 2, head_dim]``
  (the query heads of each KV group, then its key and its value); the
  module's state-dict hooks save it as separate ``wq`` / ``wk`` / ``wv``.
  """

  n_heads: int
  n_kv_heads: int
  head_dim: int

  def split_rows(
      self, tensor: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rows of a fused-output tensor as (q, k, v), like the save hook."""
    hpk = self.n_heads // self.n_kv_heads
    tail = tuple(tensor.shape[1:])
    w = tensor.reshape(self.n_kv_heads, hpk + 2, self.head_dim, *tail)
    return (
        w[:, :hpk].reshape(-1, *tail),
        w[:, hpk].reshape(-1, *tail),
        w[:, hpk + 1].reshape(-1, *tail),
    )


class AttentionLoRAConverter(LoRAConverter):
  """``LoRAConverter`` that selects targets by a full-path regex."""

  @dataclass(kw_only=True, slots=True)
  class Config(LoRAConverter.Config):
    rank: int = 16
    alpha: float = 32.0
    target_fqn_regex: str = field(default_factory=target_regex)
    """re.fullmatch against each Linear config's dotted path."""

  def __init__(self, config: Config, **kwargs: Any) -> None:
    super().__init__(config, **kwargs)
    self.pattern = re.compile(config.target_fqn_regex)
    self.matched: list[str] = []
    self.fused_layouts: dict[str, FusedLayout] = {}

  def convert(self, model_config: Module.Config) -> Module.Config:
    converted_root = model_config
    configs = list(model_config.traverse(Module.Config, recurse=True))
    for fqn, cfg, parent, attr in reversed(configs):
      if isinstance(cfg, Linear.Config) and self.pattern.fullmatch(fqn):
        assert cfg._owner is not None
        new_cfg = _get_lora_cls(cfg._owner).Config(
            **{f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init},
            rank=self.rank,
            alpha=self.alpha,
        )
        self.matched.append(fqn)
        if attr == "wqkv" and all(
            hasattr(parent, name)
            for name in ("n_heads", "n_kv_heads", "head_dim")
        ):
          self.fused_layouts[fqn] = FusedLayout(
              parent.n_heads, parent.n_kv_heads, parent.head_dim
          )
      else:
        new_cfg = _make_frozen_config(cfg)
      if parent is None:
        converted_root = new_cfg
      elif isinstance(parent, list):
        parent[attr] = new_cfg
      else:
        setattr(parent, attr, new_cfg)
    if not self.matched:
      raise ValueError(
          f"LoRA target regex {self.pattern.pattern!r} matched no Linear config"
      )
    logger.info(
        "LoRA rank=%d alpha=%s on %d projections (e.g. %s)",
        self.rank,
        self.alpha,
        len(self.matched),
        sorted(self.matched)[:4],
    )
    return converted_root


# ---------------------------------------------------------------------------
# State dicts
# ---------------------------------------------------------------------------


def is_lora_key(key: str) -> bool:
  return key.endswith("." + LORA_A) or key.endswith("." + LORA_B)


def split_lora_state_dict(
    state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
  base = {k: v for k, v in state.items() if not is_lora_key(k)}
  lora = {k: v for k, v in state.items() if is_lora_key(k)}
  return base, lora


def lora_prefixes(state: dict[str, Any]) -> list[str]:
  """Module paths that carry an adapter (keys ``<prefix>.lora_a.weight``)."""
  return sorted(
      k[: -len(LORA_A) - 1] for k in state if k.endswith("." + LORA_A)
  )


def merge_lora_state_dict(
    state: dict[str, Any],
    scaling: float,
    fused_layouts: dict[str, FusedLayout] | None = None,
) -> dict[str, Any]:
  """Base weights with ``W + scaling * B @ A`` folded in; adapter keys dropped.

  A fused ``wqkv`` adapter (listed in ``fused_layouts``) is split into the
  ``wq`` / ``wk`` / ``wv`` weights its module saves. Computed in float32 and
  cast back to each base weight's dtype; works on meta tensors too.
  """
  fused_layouts = fused_layouts or {}
  base, lora = split_lora_state_dict(state)
  merged = dict(base)
  for prefix in lora_prefixes(state):
    a = lora[f"{prefix}.{LORA_A}"]
    b = lora.get(f"{prefix}.{LORA_B}")
    if b is None:
      raise KeyError(f"adapter {prefix} lacks lora_b")
    delta = torch.matmul(b.to(torch.float32), a.to(torch.float32))
    if prefix in fused_layouts:
      parent = prefix[: -len("wqkv")]
      parts = zip(("wq", "wk", "wv"), fused_layouts[prefix].split_rows(delta))
    else:
      parts = [(None, delta)]
    for name, part in parts:
      weight_key = (
          f"{prefix}.weight" if name is None else f"{parent}{name}.weight"
      )
      if weight_key not in base:
        raise KeyError(f"adapter {prefix} has no base weight {weight_key}")
      w = base[weight_key]
      if tuple(part.shape) != tuple(w.shape):
        raise ValueError(
            f"adapter delta {tuple(part.shape)} does not match {weight_key} {tuple(w.shape)}"
        )
      merged[weight_key] = (w.to(torch.float32) + scaling * part).to(w.dtype)
  return merged


def lora_state_dict_adapter(
    inner_cls: type,
    scaling: float,
    fused_layouts: dict[str, FusedLayout] | None = None,
) -> type:
  """A subclass of ``inner_cls`` that is LoRA aware (see module doc)."""

  class LoRAStateDictAdapter(inner_cls):  # type: ignore[valid-type, misc]
    lora_scaling = scaling
    lora_fused_layouts = dict(fused_layouts or {})

    def __init__(self, *args: Any, **kwargs: Any) -> None:
      super().__init__(*args, **kwargs)
      # Strict adapters check that a checkpoint covers every model key
      # (built from the converted config); a base checkpoint never holds
      # adapter tensors.
      shapes = getattr(self, "expected_model_shapes", None)
      if isinstance(shapes, dict):
        self.expected_model_shapes = {
            k: v for k, v in shapes.items() if not is_lora_key(k)
        }

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
      base, _ = split_lora_state_dict(state_dict)
      return super().to_hf(base)

    def to_hf_for_export(self, state_dict: dict[str, Any]) -> dict[str, Any]:
      merged = merge_lora_state_dict(
          state_dict, self.lora_scaling, self.lora_fused_layouts
      )
      parent = super()
      if hasattr(parent, "to_hf_for_export"):
        return parent.to_hf_for_export(merged)
      return parent.to_hf(merged)

    def to_hf_base(self, state_dict: dict[str, Any]) -> dict[str, Any]:
      return super().to_hf(state_dict)

  LoRAStateDictAdapter.__name__ = f"LoRA{inner_cls.__name__}"
  LoRAStateDictAdapter.__qualname__ = LoRAStateDictAdapter.__name__
  return LoRAStateDictAdapter


@dataclass(frozen=True)
class LoRAOptions:
  rank: int = 16
  alpha: float = 32.0
  projections: tuple[str, ...] = ("q", "k", "v", "o")
  target_fqn_regex: str | None = None
  """Overrides ``projections`` when set."""

  @property
  def scaling(self) -> float:
    return self.alpha / self.rank


@dataclass
class AppliedLoRA:
  matched: list[str]
  """Adapted module paths (config paths, e.g. layers.0.attention.wq)."""
  fused_layouts: dict[str, FusedLayout]
  scaling: float


def apply_lora(model_spec: Any, options: LoRAOptions) -> AppliedLoRA:
  """Convert ``model_spec.model`` and wrap its adapter in place.

  Raises when nothing matched.
  """
  converter = AttentionLoRAConverter.Config(
      rank=options.rank,
      alpha=options.alpha,
      target_fqn_regex=options.target_fqn_regex
      or target_regex(options.projections),
  ).build()
  model_spec.model = converter.convert(model_spec.model)
  if model_spec.state_dict_adapter is not None:
    model_spec.state_dict_adapter = lora_state_dict_adapter(
        model_spec.state_dict_adapter, options.scaling, converter.fused_layouts
    )
  return AppliedLoRA(
      list(converter.matched), dict(converter.fused_layouts), options.scaling
  )


def lora_parameter_report(model: torch.nn.Module) -> dict[str, int]:
  """Counts of trainable / frozen parameters (numel), for logs and tests."""
  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
  adapters = sum(
      p.numel()
      for n, p in model.named_parameters()
      if n.endswith(("lora_a.weight", "lora_b.weight"))
  )
  return {"trainable": trainable, "frozen": frozen, "adapter": adapters}


# ---------------------------------------------------------------------------
# PEFT layout
# ---------------------------------------------------------------------------


def _map_rows(
    adapter: Any, weight_key: str, rows: torch.Tensor, in_features: int
) -> tuple[str, torch.Tensor]:
  """HF module path of a TorchTitan weight and ``rows`` laid out as HF rows.

  ``rows`` (lora_b, [out, r]) is placed in the first r columns of a
  weight-shaped zero tensor and mapped by the model's own adapter, so it
  receives whatever row re-layout the weight receives (renames, RoPE
  permutations). Raises NotImplementedError when the adapter does not map
  the key alone to one HF weight of the same shape, or mixes columns.
  """
  to_hf = getattr(adapter, "to_hf_base", adapter.to_hf)
  r = rows.shape[1]
  probe = torch.zeros(rows.shape[0], in_features, dtype=torch.float32)
  probe[:, :r] = rows.detach().to(torch.float32)
  try:
    mapped = to_hf({weight_key: probe})
  except Exception as exc:  # noqa: BLE001 - adapters may need the full dict
    raise NotImplementedError(
        f"adapter could not map {weight_key} alone ({exc}); pass hf_module_map"
    ) from exc
  if len(mapped) != 1:
    raise NotImplementedError(
        f"{weight_key} maps to {sorted(mapped)}; pass hf_module_map"
    )
  ((hf_key, tensor),) = mapped.items()
  if not hf_key.endswith(".weight") or tuple(tensor.shape) != tuple(
      probe.shape
  ):
    raise NotImplementedError(
        f"{weight_key} -> {hf_key} changes the weight shape"
    )
  if bool(tensor[:, r:].abs().sum() != 0):
    raise NotImplementedError(f"{weight_key} -> {hf_key} mixes columns")
  return hf_key[: -len(".weight")], tensor[:, :r].to(rows.dtype)


def peft_adapter_state_dict(
    state: dict[str, Any],
    *,
    adapter: Any | None = None,
    hf_module_map: Callable[[str], str] | None = None,
    fused_layouts: dict[str, FusedLayout] | None = None,
) -> dict[str, torch.Tensor]:
  """Adapter tensors in the PEFT layout (``base_model.model.<module>.lora_A``).

  With an ``adapter``, module names and lora_b row layouts come from the
  model's own state-dict adapter; a fused ``wqkv`` adapter becomes three
  adapters (q, k, v) that share lora_A. ``hf_module_map`` renames without
  re-laying rows (only for adapters that rename keys and nothing else).
  """
  if adapter is not None and fused_layouts is None:
    fused_layouts = getattr(adapter, "lora_fused_layouts", None)
  fused_layouts = fused_layouts or {}
  out: dict[str, torch.Tensor] = {}
  for prefix in lora_prefixes(state):
    a = state[f"{prefix}.{LORA_A}"]
    b = state[f"{prefix}.{LORA_B}"]
    if prefix in fused_layouts:
      parent = prefix[: -len("wqkv")]
      parts = list(
          zip(
              (f"{parent}wq", f"{parent}wk", f"{parent}wv"),
              fused_layouts[prefix].split_rows(b),
          )
      )
    else:
      parts = [(prefix, b)]
    for module_prefix, rows in parts:
      if hf_module_map is not None:
        module, hf_rows = hf_module_map(module_prefix), rows
      elif adapter is not None:
        module, hf_rows = _map_rows(
            adapter, f"{module_prefix}.weight", rows, int(a.shape[1])
        )
      else:
        raise ValueError("pass an adapter or hf_module_map")
      out[f"base_model.model.{module}.lora_A.weight"] = a
      out[f"base_model.model.{module}.lora_B.weight"] = hf_rows.contiguous()
  return out


def save_peft_adapter(
    out_dir: str | Path,
    tensors: dict[str, torch.Tensor],
    *,
    options: LoRAOptions,
    base_model_name_or_path: str = "",
) -> Path:
  """``adapter_model.safetensors`` plus ``adapter_config.json``."""
  from safetensors.torch import save_file

  out = Path(out_dir)
  out.mkdir(parents=True, exist_ok=True)
  # clone(): fused q / k / v adapters share one lora_A tensor.
  save_file(
      {k: v.detach().cpu().clone().contiguous() for k, v in tensors.items()},
      str(out / "adapter_model.safetensors"),
  )
  modules = sorted({k.split(".")[-3] for k in tensors})
  (out / "adapter_config.json").write_text(
      json.dumps(
          {
              "peft_type": "LORA",
              "task_type": "CAUSAL_LM",
              "r": options.rank,
              "lora_alpha": options.alpha,
              "lora_dropout": 0.0,
              "bias": "none",
              "target_modules": modules,
              "base_model_name_or_path": base_model_name_or_path,
              "fan_in_fan_out": False,
              "inference_mode": True,
          },
          indent=2,
      )
  )
  return out
