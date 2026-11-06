# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Muse Spark text decoder state dicts and the public Hugging Face checkpoint.

Three pieces live here. ``MuseSparkStateDictAdapter`` translates between the
public ``MuseSparkForConditionalGeneration`` checkpoint and the state dict of
TorchTitan's Muse Spark text decoder. ``ExpertPermutingHFStorageReader`` reads
that checkpoint into a decoder whose routed-expert rows sit in a
non-contiguous placement, and ``expert_slot_to_logical`` builds the slot map
of such a placement: for each physical expert row, the logical expert in it.

What a load reads: the decoder tensors under ``model.language_model.`` and
the output head ``lm_head.weight``. The vision tower, vision adapter, vision
projection and audio tower stay unread in the checkpoint, and so does every
decoder layer a truncated flavor leaves out (layer index at or above its
layer count).

How names and tensors convert:

* Most decoder tensors only change their name.
* Each MoE layer of the checkpoint packs the gate and up projections of all
  routed experts into ``mlp.experts.gate_up_proj``, shaped ``[E, 2F, D]``.
  Rows ``0..F-1`` along dim 1 become ``w1_EFD`` (gate) and rows ``F..2F-1``
  become ``w3_EFD`` (up); ``mlp.experts.down_proj`` maps to ``w2_EDF`` as is.
* Q and K projection rows keep their order, with no per-head reshuffle: the
  checkpoint and this decoder both rotate the two halves of every head
  dimension (rotate-half RoPE), so their rotary pairs already coincide.

What triggers a ``ValueError``:

* Constructing the adapter for a checkpoint folder that has an index file
  (``model.safetensors.index.json``) holds that file against the decoder.
  A decoder tensor that the index file lacks, or a listed name that is not
  a decoder tensor, not a tower tensor and not inside a layer past the
  layer count, aborts construction. On a folder without an index file that
  comparison moves to ``from_hf``. Construction also aborts if the name
  mapping and the decoder built from the config disagree about the
  TorchTitan tensor names.
* In ``from_hf``, a name of that unplaceable kind aborts the load before
  any tensor is converted; after conversion, a decoder tensor that was not
  produced, or whose shape differs from the built decoder's, aborts it.
* ``to_hf`` aborts on a state-dict key it has no checkpoint name for, and on
  an expert gate tensor that comes without its up tensor, or the reverse.

LoRA: the adapter matrices that ``torchtitan.components.lora`` adds have no
tensors in a base checkpoint. Conversion leaves them out in both directions
and the shape table omits them. That is safe because a LoRA model loads its
base weights non-strictly, so its adapters keep their initial values.

Placement: when the routed-expert rows use a non-contiguous placement,
``get_hf_storage_reader`` hands out the permuting reader.
"""

from __future__ import annotations

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

_SPARK_DECODER_ROOT = "model.language_model"
_SPARK_VISION_AUDIO_TOWERS = (
    "model.vision_tower.",
    "model.vision_adapter.",
    "model.vision_projection.",
    "model.audio_tower.",
)
_EXPERTS = "moe.routed_experts.inner_experts"
_LAYER_KEY = re.compile(re.escape(_SPARK_DECODER_ROOT) + r"\.layers\.(\d+)\.")
# LoRA adapters (``torchtitan.components.lora``) have no checkpoint tensors.
_LORA_KEY = re.compile(r"\.lora_[ab]\.weight$")


def _layer_map(layer_id: int, layer_config: Any) -> dict[str, str]:
  """HF key -> TT key for one layer (excluding the packed experts)."""
  hf = f"{_SPARK_DECODER_ROOT}.layers.{layer_id}"
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
  return f"{_SPARK_DECODER_ROOT}.layers.{layer_id}.mlp.experts.gate_up_proj"


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
      f"{_SPARK_DECODER_ROOT}.embed_tokens.weight": "tok_embeddings.embedding.weight",
      f"{_SPARK_DECODER_ROOT}.norm.weight": "norm.weight",
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
  """Muse Spark text-decoder state dict <-> HF checkpoint; see the module doc."""

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
    self._decoder_depth = len(model_config.layers)
    self.moe_layers = [
        idx for idx, block in enumerate(model_config.layers) if block.moe is not None
    ]
    self.from_hf_map = _build_from_hf_map(model_config)
    self.to_hf_map = {tt_key: hf_key for hf_key, tt_key in self.from_hf_map.items()}
    self._decoder_hf_names = frozenset(hf_text_keys_for_config(model_config))

    # One shape per tensor of the decoder, LoRA matrices aside, taken from a
    # build on the meta device (no parameter memory). from_hf looks the table
    # up on every call, so a subclass or wrapper may replace it.
    with torch.device("meta"):
      model = model_config.build()
    self.expected_model_shapes = {}
    for tt_key, entry in model.state_dict().items():
      if _LORA_KEY.search(tt_key) is None:
        self.expected_model_shapes[tt_key] = tuple(entry.shape)
    del model

    mapped = set(self.from_hf_map.values())
    for idx in self.moe_layers:
      mapped.update(
          f"layers.{idx}.{_EXPERTS}.{half}" for half in ("w1_EFD", "w3_EFD")
      )
    model_keys = set(self.expected_model_shapes)
    if mapped != model_keys:
      raise ValueError(
          f"{type(self).__name__} construction: the HF name mapping and the"
          " decoder built from model_config list different TorchTitan tensors."
          f" Only the mapping yields {self._count_and_first_ten(mapped - model_keys)};"
          " only the built decoder has"
          f" {self._count_and_first_ten(model_keys - mapped)}"
      )
    self._compare_index_with_decoder()

  @staticmethod
  def _count_and_first_ten(entries) -> str:
    """``"<count> [first ten in sorted order]"``, for the error messages."""
    ordered = sorted(entries)
    return f"{len(ordered)} {ordered[:10]}"

  def _compare_index_with_decoder(self) -> None:
    """Holds the names in the folder's index file against the text decoder."""
    folder = self.hf_assets_path
    if not folder:
      return
    checkpoint_keys = self.fqn_to_index_mapping
    if checkpoint_keys is None:
      logger.info(
          f"{type(self).__name__}: {folder} holds no model.safetensors.index.json;"
          " the text decoder's tensor names get compared with the checkpoint"
          " once from_hf runs"
      )
      return
    missing = self._decoder_hf_names.difference(checkpoint_keys)
    unknown = [
        n for n in checkpoint_keys if self._classify_hf_key(n) == "unknown"
    ]
    if missing or unknown:
      index_file = os.path.join(folder, "model.safetensors.index.json")
      raise ValueError(
          f"{type(self).__name__} construction: {index_file} does not fit the"
          f" {self._decoder_depth}-layer text decoder. Decoder tensors the"
          f" index file omits: {self._count_and_first_ten(missing)}. Listed"
          " names belonging to neither the decoder, its vision and audio"
          " towers nor a layer past the layer count:"
          f" {self._count_and_first_ten(unknown)}"
      )

  def _classify_hf_key(self, hf_name: str) -> str:
    """Label of one checkpoint name, shared by from_hf and the index check.

    ``"expected"``: the text decoder loads it. ``"excluded"``: a tower tensor,
    or a tensor of a layer at or past the layer count; it stays in the file.
    ``"unknown"``: anything else; both checks reject it.
    """
    if hf_name in self._decoder_hf_names:
      return "expected"
    if hf_name.startswith(_SPARK_VISION_AUDIO_TOWERS):
      return "excluded"
    found = _LAYER_KEY.match(hf_name)
    if found is not None and int(found.group(1)) >= self._decoder_depth:
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
    where = f"{type(self).__name__}.from_hf"
    labels = {name: self._classify_hf_key(name) for name in hf_state_dict}
    unknown = [name for name, label in labels.items() if label == "unknown"]
    if unknown:
      raise ValueError(
          f"{where}: checkpoint names outside the text decoder, its vision and"
          " audio towers and the layers past its layer count:"
          f" {self._count_and_first_ten(unknown)}"
      )

    state: dict[str, Any] = {}
    for name, label in labels.items():
      if label != "expected":
        continue
      if name in self.from_hf_map:
        state[self.from_hf_map[name]] = hf_state_dict[name]
        continue
      # Every other decoder name is a packed [E, 2F, D] expert tensor; views of
      # the first and second half of dim 1 are w1_EFD (gate) and w3_EFD (up).
      packed = hf_state_dict[name]
      stem = f"layers.{_LAYER_KEY.match(name).group(1)}.{_EXPERTS}"
      mid = packed.shape[1] // 2
      state[f"{stem}.w1_EFD"] = packed.narrow(1, 0, mid)
      state[f"{stem}.w3_EFD"] = packed.narrow(1, mid, packed.shape[1] - mid)

    table = self.expected_model_shapes
    missing = [n for n in table if n not in state]
    extra = [n for n in state if n not in table]
    mismatches = [
        f"{n} {tuple(state[n].shape)} vs {tuple(table[n])}"
        for n in state
        if n in table and tuple(state[n].shape) != tuple(table[n])
    ]
    if missing or extra or mismatches:
      raise ValueError(
          f"{where}: the converted decoder tensors do not fit"
          " expected_model_shapes. Table entries never produced:"
          f" {self._count_and_first_ten(missing)}. Shapes produced vs recorded:"
          f" {self._count_and_first_ten(mismatches)}. Produced names the table"
          f" does not list: {self._count_and_first_ten(extra)}"
      )
    return state

  @property
  def permutes_experts(self) -> bool:
    """Whether some stored expert row holds a logical expert other than its own.

    ``expert_slot_to_logical`` is one slot map for every layer or a dict of
    per-layer maps keyed by the layer index as a string; it is read on each
    call. When this is True, a Hugging Face load goes through
    ``ExpertPermutingHFStorageReader``, which fills stored row ``p`` of every
    routed-expert tensor from the checkpoint row of the logical expert the
    map names for ``p``. TorchTitan checkpoints of such a model keep the
    stored row order, so exporting one to Hugging Face has to invert the
    permutation along the expert dimension with the same map.
    """
    layout = self.expert_slot_to_logical
    for order in layout.values() if isinstance(layout, dict) else (layout,):
      if any(int(logical) != row for row, logical in enumerate(order)):
        return True
    return False

  def get_hf_storage_reader(self, path, from_quantized=False):
    if self.permutes_experts:
      return ExpertPermutingHFStorageReader(path, self.expert_slot_to_logical)
    return super().get_hf_storage_reader(path, from_quantized)
