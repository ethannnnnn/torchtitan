# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hugging Face checkpoint mapping for the Muse Spark Open text decoder.

The public checkpoint (``MuseSparkForConditionalGeneration``) stores the text
decoder under ``model.language_model.*`` plus ``lm_head.weight``, next to the
vision and audio towers, which the text-only model does not build. The
mapping is one-to-one except for the routed experts: the checkpoint packs
``gate_up_proj`` as ``[E, 2F, D]`` (gate rows first), while TorchTitan keeps
``w1_EFD`` (gate) and ``w3_EFD`` (up) separately. Both sides use the
rotate-half RoPE convention, so Q/K weights are not permuted.

The adapter is strict about the text decoder:

* With ``hf_assets_path``, the checkpoint index must contain every text
  tensor of the model's layers (deeper layers of a truncated model and the
  vision/audio towers are ignored).
* ``to_hf`` must map every model key; ``from_hf`` rejects unknown keys and
  must produce exactly the model's keys and shapes.

LoRA adapter weights (``*.lora_a.weight`` / ``*.lora_b.weight``) are not part
of the checkpoint: they are skipped by ``to_hf`` and not expected from
``from_hf`` (the checkpointer loads the base weights non-strictly).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import torch

from torch.distributed.checkpoint import HuggingFaceStorageReader

from torchtitan.protocols.state_dict_adapter import StateDictAdapter
from torchtitan.tools.logging import logger

from .model import expert_parallel_module
from .model import MuseSparkModel

__all__ = ["MuseSparkStateDictAdapter", "hf_text_keys_for_config"]

TEXT_PREFIX = "model.language_model"
# Checkpoint families the text-only model has no parameters for.
EXCLUDED_HF_PREFIXES = (
    "model.vision_tower.",
    "model.vision_adapter.",
    "model.vision_projection.",
    "model.audio_tower.",
)
_EXPERTS = "moe.routed_experts.inner_experts"
_LAYER_KEY = re.compile(re.escape(TEXT_PREFIX) + r"\.layers\.(\d+)\.")
# LoRA adapters (``torchtitan.components.lora``) have no checkpoint tensors.
_LORA_KEY = re.compile(r"\.lora_[ab]\.weight$")


def _layer_map(layer_id: int, layer_config: Any) -> dict[str, str]:
  """HF key -> TT key for one layer (excluding the packed experts)."""
  hf = f"{TEXT_PREFIX}.layers.{layer_id}"
  tt = f"layers.{layer_id}"
  mapping = {
      f"{hf}.input_layernorm.weight": f"{tt}.attention_norm.weight",
      f"{hf}.self_attn.q_proj.weight": f"{tt}.attention.qkv_linear.wq.weight",
      f"{hf}.self_attn.k_proj.weight": f"{tt}.attention.qkv_linear.wk.weight",
      f"{hf}.self_attn.v_proj.weight": f"{tt}.attention.qkv_linear.wv.weight",
      f"{hf}.self_attn.o_proj.weight": f"{tt}.attention.wo.weight",
      f"{hf}.self_attn.gate_proj.weight": f"{tt}.attention.output_gate.weight",
      f"{hf}.post_attention_residual_gate.gate": f"{tt}.attention_residual.gate",
      f"{hf}.pre_feedforward_layernorm.weight": f"{tt}.ffn_norm.weight",
      f"{hf}.post_feedforward_layernorm.weight": f"{tt}.post_ffn_norm.weight",
      f"{hf}.post_feedforward_residual_gate.gate": f"{tt}.ffn_residual.gate",
  }
  if layer_config.post_attention_norm.elementwise_affine:
    mapping[f"{hf}.post_attention_layernorm.weight"] = (
        f"{tt}.post_attention_norm.weight"
    )
  if layer_config.moe is None:
    mapping.update({
        f"{hf}.mlp.gate_proj.weight": f"{tt}.feed_forward.w1.weight",
        f"{hf}.mlp.up_proj.weight": f"{tt}.feed_forward.w3.weight",
        f"{hf}.mlp.down_proj.weight": f"{tt}.feed_forward.w2.weight",
    })
    return mapping
  mapping.update({
      f"{hf}.mlp.gate.weight": f"{tt}.moe.router.gate.weight",
      f"{hf}.mlp.gate.e_score_correction_bias": f"{tt}.moe.expert_bias_E",
      f"{hf}.mlp.pre_expert_proj.weight": f"{tt}.moe.pooler.weight",
      f"{hf}.mlp.pre_expert_norm.weight": f"{tt}.moe.pre_expert_norm.weight",
      f"{hf}.mlp.experts.down_proj": f"{tt}.{_EXPERTS}.w2_EDF",
      f"{hf}.mlp.experts.post_expert_norm.weight": (
          f"{tt}.moe.routed_experts.output_norm.weight"
      ),
      f"{hf}.mlp.post_expert_proj.weight": f"{tt}.moe.unpooler.weight",
  })
  return mapping


def _gate_up_key(layer_id: int) -> str:
  return f"{TEXT_PREFIX}.layers.{layer_id}.mlp.experts.gate_up_proj"


def hf_text_keys_for_config(model_config: MuseSparkModel.Config) -> list[str]:
  """Every checkpoint key the text model of ``model_config`` loads."""
  adapter_map = _build_from_hf_map(model_config)
  keys = list(adapter_map)
  for layer_id, layer in enumerate(model_config.layers):
    if layer.moe is not None:
      keys.append(_gate_up_key(layer_id))
  return sorted(keys)


def _build_from_hf_map(model_config: MuseSparkModel.Config) -> dict[str, str]:
  mapping = {
      f"{TEXT_PREFIX}.embed_tokens.weight": "tok_embeddings.embedding.weight",
      f"{TEXT_PREFIX}.norm.weight": "norm.weight",
      "lm_head.weight": "lm_head.weight",
  }
  for layer_id, layer in enumerate(model_config.layers):
    mapping.update(_layer_map(layer_id, layer))
  return mapping


_EXPERT_HF_KEY = re.compile(r".*\.mlp\.experts\.(gate_up_proj|down_proj)$")


def expert_slot_to_logical(num_experts: int, ep_degree: int, placement: str) -> list[int]:
  """``logical[p]``: logical expert stored in physical expert row ``p``."""
  if placement == "contiguous" or ep_degree <= 1:
    return list(range(num_experts))
  if placement != "interleaved":
    raise ValueError(f"unknown expert placement {placement!r}")
  local = num_experts // ep_degree
  return [(p % local) * ep_degree + p // local for p in range(num_experts)]


_EXPERT_LAYER = re.compile(r".*\.layers\.(\d+)\.mlp\.experts\.")


class ExpertPermutingHFStorageReader(HuggingFaceStorageReader):
  """HF safetensors reader for a non-contiguous expert placement.

  The model stores routed-expert rows in physical-slot order (see
  ``expert_parallel.configure_expert_parallel(placement="interleaved")``),
  so physical row ``p`` of every ``experts.gate_up_proj`` / ``down_proj``
  tensor is read from logical row ``slot_to_logical[p]`` of the checkpoint
  (each expert row is contiguous in the file). Other tensors are read as by
  the stock reader.
  """

  def __init__(self, path: str, slot_to_logical, **kwargs):
    """``slot_to_logical``: one map for every layer, or ``{layer_id: map}``."""
    super().__init__(path, **kwargs)
    self.slot_to_logical = slot_to_logical

  def _map(self, fqn: str) -> list[int]:
    if isinstance(self.slot_to_logical, dict):
      return self.slot_to_logical[str(int(_EXPERT_LAYER.match(fqn).group(1)))]
    return self.slot_to_logical

  def read_data(self, plan, planner):
    from safetensors import safe_open  # pylint: disable=g-import-not-at-top
    from torch.futures import Future  # pylint: disable=g-import-not-at-top

    per_file: dict[str, list] = {}
    for req in plan.items:
      per_file.setdefault(
          self.storage_data[req.storage_index].relative_path, []
      ).append(req)
    for file_name, reqs in per_file.items():
      with safe_open(filename=file_name, framework="pt") as f:
        for req in reqs:
          offs, lens = list(req.storage_offsets), list(req.lengths)
          view = f.get_slice(req.storage_index.fqn)
          rest = tuple(slice(o, o + n) for o, n in zip(offs[1:], lens[1:]))
          if _EXPERT_HF_KEY.match(req.storage_index.fqn):
            s2l = self._map(req.storage_index.fqn)
            rows = [
                view[(slice(l, l + 1),) + rest]
                for l in (s2l[p] for p in range(offs[0], offs[0] + lens[0]))
            ]
            tensor = torch.cat(rows) if rows else view[(slice(0, 0),) + rest]
          else:
            tensor = view[(slice(offs[0], offs[0] + lens[0]),) + rest]
          target = planner.resolve_tensor(req).detach()
          if target.size() != tensor.size():
            raise AssertionError(
                f"{req.storage_index}: size {target.size()} vs {tensor.size()}"
            )
          target.copy_(tensor)
          planner.commit_tensor(req, target)
    fut: Future = Future()
    fut.set_result(None)
    return fut


class MuseSparkStateDictAdapter(StateDictAdapter):
  """Strict HF <-> TorchTitan mapping for the text decoder (module docstring)."""

  def __init__(
      self, model_config: MuseSparkModel.Config, hf_assets_path: str | None
  ):
    super().__init__(model_config, hf_assets_path)
    moe = next((l.moe for l in model_config.layers if l.moe is not None), None)
    placement = getattr(model_config, "ep_expert_placement", "contiguous")
    degree = getattr(model_config, "ep_placement_degree", 1)
    if placement.startswith("file:"):
      self.expert_slot_to_logical = (
          expert_parallel_module().load_expert_placement(placement, degree)
      )
    else:
      self.expert_slot_to_logical = expert_slot_to_logical(
          moe.num_experts if moe is not None else 0, degree, placement
      )
    self.num_layers = len(model_config.layers)
    self.from_hf_map = _build_from_hf_map(model_config)
    self.to_hf_map = {tt: hf for hf, tt in self.from_hf_map.items()}
    self.moe_layers = [
        i for i, layer in enumerate(model_config.layers) if layer.moe is not None
    ]
    self.expected_hf_keys = set(hf_text_keys_for_config(model_config))
    with torch.device("meta"):
      model = model_config.build()
    self.expected_model_shapes = {
        k: tuple(v.shape)
        for k, v in model.state_dict().items()
        if not _LORA_KEY.search(k)
    }
    del model
    expected_tt = set(self.from_hf_map.values()) | {
        f"layers.{i}.{_EXPERTS}.{w}" for i in self.moe_layers for w in ("w1_EFD", "w3_EFD")
    }
    if expected_tt != set(self.expected_model_shapes):
      raise ValueError(
          "MuseSparkStateDictAdapter does not cover the model: missing"
          f" {sorted(set(self.expected_model_shapes) - expected_tt)[:10]},"
          f" unknown {sorted(expected_tt - set(self.expected_model_shapes))[:10]}"
      )
    if hf_assets_path:
      self._check_index(hf_assets_path)

  def _check_index(self, hf_assets_path: str) -> None:
    index_path = os.path.join(hf_assets_path, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
      # The trainer builds the adapter for tokenizer-only asset folders too;
      # a single-file or absent checkpoint is checked in from_hf instead.
      logger.info(
          "Muse Spark: no checkpoint index in %s; keys are checked on load",
          hf_assets_path,
      )
      return
    with open(index_path) as f:
      weight_map = json.load(f)["weight_map"]
    missing = sorted(self.expected_hf_keys - set(weight_map))
    unknown = sorted(
        k for k in weight_map if self._classify_hf_key(k) == "unknown"
    )
    if missing or unknown:
      raise ValueError(
          f"Muse Spark checkpoint {hf_assets_path} does not match the text"
          f" model: missing {missing[:10]}, unknown {unknown[:10]}"
      )

  def _classify_hf_key(self, hf_key: str) -> str:
    """``expected`` | ``excluded`` (towers, deeper layers) | ``unknown``."""
    if hf_key in self.expected_hf_keys:
      return "expected"
    if hf_key.startswith(EXCLUDED_HF_PREFIXES):
      return "excluded"
    match = _LAYER_KEY.match(hf_key)
    if match is not None and int(match.group(1)) >= self.num_layers:
      return "excluded"
    return "unknown"

  def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
    hf_state_dict: dict[str, Any] = {}
    w1: dict[int, Any] = {}
    w3: dict[int, Any] = {}
    unmapped = []
    for key, value in state_dict.items():
      if key in self.to_hf_map:
        hf_state_dict[self.to_hf_map[key]] = value
        continue
      if _LORA_KEY.search(key):
        continue
      match = re.fullmatch(rf"layers\.(\d+)\.{re.escape(_EXPERTS)}\.(w1_EFD|w3_EFD)", key)
      if match is None:
        unmapped.append(key)
        continue
      (w1 if match.group(2) == "w1_EFD" else w3)[int(match.group(1))] = value
    if unmapped:
      raise ValueError(f"MuseSparkStateDictAdapter.to_hf: unmapped keys {unmapped[:10]}")
    if set(w1) != set(w3):
      raise ValueError("MuseSparkStateDictAdapter.to_hf: unpaired w1/w3 experts")
    for layer_id in sorted(w1):
      hf_state_dict[_gate_up_key(layer_id)] = torch.cat(
          [w1[layer_id], w3[layer_id]], dim=1
      )
    return hf_state_dict

  def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
    state_dict: dict[str, Any] = {}
    unknown = []
    for hf_key, value in hf_state_dict.items():
      kind = self._classify_hf_key(hf_key)
      if kind == "excluded":
        continue
      if kind == "unknown":
        unknown.append(hf_key)
        continue
      if hf_key in self.from_hf_map:
        state_dict[self.from_hf_map[hf_key]] = value
        continue
      layer_id = int(_LAYER_KEY.match(hf_key).group(1))
      gate, up = value.chunk(2, dim=1)
      state_dict[f"layers.{layer_id}.{_EXPERTS}.w1_EFD"] = gate
      state_dict[f"layers.{layer_id}.{_EXPERTS}.w3_EFD"] = up
    if unknown:
      raise ValueError(
          f"MuseSparkStateDictAdapter.from_hf: unexpected checkpoint keys {unknown[:10]}"
      )
    missing = sorted(set(self.expected_model_shapes) - set(state_dict))
    mismatched = [
        f"{k}: {tuple(v.shape)} vs {self.expected_model_shapes[k]}"
        for k, v in state_dict.items()
        if tuple(v.shape) != self.expected_model_shapes[k]
    ]
    if missing or mismatched:
      raise ValueError(
          "MuseSparkStateDictAdapter.from_hf: the checkpoint does not cover the"
          f" model: missing {missing[:10]}, shape mismatches {mismatched[:10]}"
      )
    return state_dict

  @property
  def permutes_experts(self) -> bool:
    """True when expert rows are stored in a non-contiguous placement's slot
    order. HF loading then reads each physical row from its logical row; a
    TorchTitan checkpoint saved from such a model keeps the slot order (an HF
    export must apply ``expert_slot_to_logical`` to the expert dim)."""
    maps = self.expert_slot_to_logical
    if isinstance(maps, dict):
      return any(m != sorted(m) for m in maps.values())
    return maps != sorted(maps)

  def get_hf_storage_reader(self, path, from_quantized=False):
    if self.permutes_experts:
      return ExpertPermutingHFStorageReader(path, self.expert_slot_to_logical)
    return super().get_hf_storage_reader(path, from_quantized)
