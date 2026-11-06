# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Packed chat SFT microbatches in which only assistant tokens are trained.

Data: records that hold a conversation under ``messages`` (system, user,
assistant and tool turns), stored as JSONL lines or parquet rows. Each
conversation is rendered by ``chat_template.render_chat`` with the template
that ships with the model; the assistant tokens are the only targets.

A microbatch consists of 1-D tensors of length T, filled row after row:

* ``input``, int64 -- token ids; padded slots hold the pad id.
* ``labels``, int64 -- the following token wherever it is trained, otherwise
  ``IGNORE_INDEX``: outside assistant targets, on the last token of every
  document and on all padding.
* ``positions``, int64 -- counts 0, 1, 2, ... within a document and starts
  over at the first token of the next one.
* ``padding_mask``, bool -- true on exactly the padded slots.
* ``num_valid_tokens``, a Python int -- how many targets the microbatch
  trains.

T equals rows x ``max_context_length``, i.e.
``training.num_tokens_per_microbatch_per_dp_rank``, and no document crosses
from one row into the next. Attention kernels that find document starts
where ``positions`` returns to 0 (flex, splash, varlen) thus keep every
document separate. Padding at the end of a row carries on counting from the
final position of the row's last document rather than restarting at 0, so it
never opens a document of its own. The trainer divides the summed loss by the
number of trained targets across all ranks.

Packing considers a window of ``lookahead_documents`` documents, places the
longest first into the first row with space (first fit), with at most
``max_documents_per_row`` per row; documents left without space wait for the
next microbatch. Conversations longer than a row are either dropped
(``overlong="drop"``) or cut to the row length (``"truncate"``), in which case
the assistant tokens that remain are trained.

Every epoch draws a single shuffled order seeded with ``shuffle_seed +
epoch``; data-parallel rank r takes positions r, r + n, r + 2n, ... of it for
n ranks, so no conversation is seen by two ranks. ``load_state_dict`` brings a
loader to the saved position, after which it produces the very microbatches
the saved loader would have produced next. ``batch_log`` receives an entry for
each microbatch served -- a fingerprint plus the indices of the
conversations packed into it -- and the trainer clears it after each step.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from torchtitan.components.data.collators import TrainerBatch
from torchtitan.components.data.loader import BaseDataLoader
from torchtitan.components.loss import IGNORE_INDEX

from . import chat_template as ct

logger = logging.getLogger(__name__)

OVERLONG_POLICIES = ("drop", "truncate")


def _read_records(file: Path) -> Iterator[dict[str, Any]]:
  """The rows of a parquet file (read with pyarrow); for any other file, the
  JSON object on each non-blank line (UTF-8)."""
  if file.suffix == ".parquet":
    import pyarrow.parquet as pq

    yield from pq.read_table(str(file)).to_pylist()
    return
  with file.open(encoding="utf-8") as handle:
    for line in handle:
      if line.strip():
        yield json.loads(line)


def _decoded_messages(record: dict[str, Any]) -> Any:
  """A record's ``messages`` value, parsed first when it is JSON text."""
  value = record.get("messages")
  return json.loads(value) if isinstance(value, str) else value


def load_conversations(
    path: str | Path, *, limit: int | None = None
) -> list[list[dict[str, Any]]]:
  """``messages`` lists from JSONL / parquet files (or a directory of them)."""
  source = Path(path)
  files = (
      sorted(
          p
          for p in source.iterdir()
          if p.suffix in (".jsonl", ".json", ".parquet")
      )
      if source.is_dir()
      else [source]
  )
  if not files or not all(f.is_file() for f in files):
    raise FileNotFoundError(f"no SFT records at {source}")
  conversations: list[list[dict[str, Any]]] = []
  for file in files:
    for messages in map(_decoded_messages, _read_records(file)):
      if not isinstance(messages, list) or not messages:
        raise ValueError(f"SFT record {len(conversations)} has no messages")
      if not any(m.get("role") == "assistant" for m in messages):
        raise ValueError(
            f"SFT record {len(conversations)} has no assistant turn"
        )
      conversations.append([dict(m) for m in messages])
      if limit is not None and len(conversations) >= limit:
        return conversations
  if not conversations:
    raise ValueError(f"{source} holds no SFT records")
  return conversations


@dataclass
class SFTDocument:
  """One tokenized conversation ready to pack."""

  index: int
  input_ids: list[int]
  labels: list[int]
  """labels[t] = input_ids[t + 1] where that token is trained, else IGNORE."""

  @property
  def length(self) -> int:
    return len(self.input_ids)

  @property
  def num_valid(self) -> int:
    return sum(1 for label in self.labels if label != IGNORE_INDEX)


def document_from_rendered(
    index: int, rendered: ct.RenderedChat, *, max_length: int, overlong: str
) -> SFTDocument | None:
  """Shifted labels for one conversation; None when it must be skipped."""
  ids, mask = rendered.input_ids, rendered.assistant_mask
  if len(ids) > max_length:
    if overlong == "drop":
      return None
    ids, mask = ids[:max_length], mask[:max_length]
  labels = [IGNORE_INDEX] * len(ids)
  for t in range(len(ids) - 1):
    if mask[t + 1]:
      labels[t] = ids[t + 1]
  document = SFTDocument(index, list(ids), labels)
  return document if document.num_valid > 0 else None


def pack_rows(
    documents: list[SFTDocument],
    *,
    rows: int,
    width: int,
    max_docs_per_row: int,
) -> tuple[list[list[SFTDocument]], list[SFTDocument]]:
  """First-fit decreasing: place whole documents into ``rows`` rows.

  Documents are taken longest first, equal lengths in input order. Each goes
  into the lowest-numbered row that still has a free document slot (at most
  ``max_docs_per_row``) and at least its length in unused width. Returns the
  rows, each in placement order, and the documents that fit nowhere, in input
  order. Nothing is split or truncated and ``documents`` is not modified. A
  document no longer than ``width`` always fits the empty first row, so every
  call places at least one document when the longest one fits.
  """
  room = [width] * rows
  placed = [[] for _row in range(rows)]
  taken: set[int] = set()
  longest_first = sorted(
      enumerate(documents), key=lambda item: item[1].length, reverse=True
  )
  for arrival, doc in longest_first:
    target = next(
        (
            r
            for r in range(rows)
            if len(placed[r]) < max_docs_per_row and doc.length <= room[r]
        ),
        None,
    )
    if target is None:
      continue
    placed[target].append(doc)
    room[target] -= doc.length
    taken.add(arrival)
  waiting = [
      doc for arrival, doc in enumerate(documents) if arrival not in taken
  ]
  return placed, waiting


def build_microbatch(
    placed: list[list[SFTDocument]], *, width: int, pad_id: int
) -> dict[str, Any]:
  """One microbatch from ``placed``, which lists the documents of each row.

  Row ``r`` covers slots ``[r * width, (r + 1) * width)``. Its documents come
  first, in list order, each with positions ``0 .. n - 1``; the slots left
  over are padding, with ``pad_id`` inputs, ``IGNORE_INDEX`` labels and a true
  ``padding_mask``. Padding positions go on from the final position of the
  row's last document (an empty row gets ``0 .. width - 1``), so they never
  start a document. ``num_valid_tokens`` counts the labels that differ from
  ``IGNORE_INDEX``.
  """
  token_ids: list[int] = []
  targets: list[int] = []
  position_ids: list[int] = []
  is_pad: list[bool] = []
  for row in placed:
    used = sum(doc.length for doc in row)
    if used > width:
      raise ValueError(
          f"a packed row holds {used} tokens; rows are {width} wide"
      )
    resume_at = 0
    for doc in row:
      token_ids += doc.input_ids
      targets += doc.labels
      position_ids += range(doc.length)
      resume_at = doc.length
    gap = width - used
    token_ids += [pad_id] * gap
    targets += [IGNORE_INDEX] * gap
    position_ids += range(resume_at, resume_at + gap)
    is_pad += [False] * used + [True] * gap
  labels = torch.tensor(targets, dtype=torch.int64)
  return {
      "input": torch.tensor(token_ids, dtype=torch.int64),
      "labels": labels,
      "positions": torch.tensor(position_ids, dtype=torch.int64),
      "padding_mask": torch.tensor(is_pad, dtype=torch.bool),
      "num_valid_tokens": int((labels != IGNORE_INDEX).sum()),
  }


def batch_fingerprint(batch: dict[str, Any]) -> str:
  """SHA-256, as lowercase hex, of the ``input``, ``labels`` and
  ``positions`` bytes taken in that order."""
  raw = b"".join(
      batch[name].detach().cpu().contiguous().numpy().tobytes()
      for name in ("input", "labels", "positions")
  )
  return hashlib.sha256(raw).hexdigest()


@dataclass
class SFTStats:
  epoch: int = 0
  microbatches: int = 0
  documents: int = 0
  tokens: int = 0
  valid_tokens: int = 0
  dropped: int = 0
  truncated: int = 0
  last: dict[str, int] = field(default_factory=dict)


def _tokenizer_dir(tokenizer: Any) -> str:
  """Where the trainer's own tokenizer was loaded from; '' if unknown."""
  try:
    return tokenizer.tokenizer_path or ""
  except AttributeError:  # no tokenizer (None), or one without a path
    return ""


def _chat_template_tokenizer(configured: str, tokenizer: Any) -> Any:
  """The Hugging Face tokenizer whose chat template renders the records.

  ``configured`` (``hf_tokenizer_path``) wins over the folder of the trainer's
  own tokenizer. Raises ``ValueError`` when neither names a source.
  """
  source = configured or _tokenizer_dir(tokenizer)
  if not source:
    raise ValueError(
        "ChatSFTDataLoader has no chat template to render with: set "
        "--dataloader.hf-tokenizer-path, or --hf_assets_path to reuse the "
        "trainer's tokenizer folder"
    )
  return ct.load_hf_tokenizer(source)


class ChatSFTDataLoader(BaseDataLoader):
  """Assistant-only, document-packed chat SFT batches."""

  @dataclass(kw_only=True, slots=True)
  class Config(BaseDataLoader.Config):
    data_path: str = ""
    hf_tokenizer_path: str = ""
    """Folder or hub id of the Hugging Face tokenizer whose chat template
      renders the records. Empty: the folder of the tokenizer the trainer
      loaded itself, i.e. the one given by ``--hf_assets_path``."""
    reasoning_effort: str = ""
    current_date: str = ""
    mask_mode: str = "auto"
    train_on_generation_prompt: bool = False
    max_documents_per_row: int = 16
    overlong: str = "drop"
    shuffle_seed: int | None = 42
    lookahead_documents: int = 64
    pad_token_id: int = -1
    """-1 = the tokenizer's pad id (else its eos id)."""
    max_examples: int = 0
    microbatches_per_step: int = 1
    """Gradient-accumulation microbatches per rank per optimizer step
        (read by SFTTPUTrainer; the global batch scales with the DP degree)."""

  def __init__(
      self,
      config: Config,
      *,
      dp_world_size: int,
      dp_rank: int,
      tokenizer: Any,
      max_context_length: int,
      num_tokens_per_batch: int,
      **kwargs: Any,
  ) -> None:
    del kwargs
    if config.overlong not in OVERLONG_POLICIES:
      raise ValueError(f"overlong must be one of {OVERLONG_POLICIES}")
    if config.max_documents_per_row < 1 or config.lookahead_documents < 1:
      raise ValueError(
          "max_documents_per_row and lookahead_documents must be >= 1"
      )
    if config.microbatches_per_step < 1:
      raise ValueError("microbatches_per_step must be >= 1")
    self.config = config
    self._width = int(max_context_length)
    self._rows, leftover = divmod(int(num_tokens_per_batch), self._width)
    if leftover:
      raise ValueError(
          "num_tokens_per_batch={} does not split into {}-token rows".format(
              num_tokens_per_batch, self._width
          )
      )
    self.max_num_documents = self._rows * config.max_documents_per_row
    self._dp_world_size, self._dp_rank = int(dp_world_size), int(dp_rank)
    self._hf_tokenizer = _chat_template_tokenizer(
        config.hf_tokenizer_path, tokenizer
    )
    self._options = ct.ChatTemplateOptions(
        reasoning_effort=config.reasoning_effort or None,
        current_date=config.current_date or None,
    )
    if config.pad_token_id >= 0:
      self._pad_id = int(config.pad_token_id)
    else:
      pad = getattr(self._hf_tokenizer, "pad_token_id", None)
      self._pad_id = int(
          pad if pad is not None else self._hf_tokenizer.eos_token_id
      )
    self._conversations = load_conversations(
        config.data_path, limit=config.max_examples or None
    )
    if len(self._conversations) < self._dp_world_size:
      raise ValueError(
          f"{len(self._conversations)} conversations cannot shard over "
          f"{self._dp_world_size} data-parallel ranks"
      )
    self._epoch = 0
    self._cursor = 0
    self._pending: list[SFTDocument] = []
    self._shard = self._epoch_shard(0)
    self.stats = SFTStats()
    self.batch_log: list[dict[str, Any]] = []
    if dp_rank == 0:
      logger.info(
          "[sft] %d conversations, %d ranks, %d rows x %d tokens per microbatch, "
          "<= %d documents per row, template options %s, mask %s, overlong %s",
          len(self._conversations),
          self._dp_world_size,
          self._rows,
          self._width,
          config.max_documents_per_row,
          self._options,
          config.mask_mode,
          config.overlong,
      )

  def _epoch_shard(self, epoch: int) -> list[int]:
    """The conversation indices this rank visits in ``epoch``."""
    order = list(range(len(self._conversations)))
    seed = self.config.shuffle_seed
    if seed is not None:
      random.Random(int(seed) + epoch).shuffle(order)
    # Equal shard sizes keep ranks in step across epochs.
    usable = len(order) - len(order) % self._dp_world_size
    return order[self._dp_rank : usable : self._dp_world_size]

  def tokenize(self, index: int) -> SFTDocument | None:
    rendered = ct.render_chat(
        self._hf_tokenizer,
        self._conversations[index],
        self._options,
        mask_mode=self.config.mask_mode,
        train_on_generation_prompt=self.config.train_on_generation_prompt,
    )
    if len(rendered.input_ids) > self._width:
      if self.config.overlong == "drop":
        self.stats.dropped += 1
        return None
      self.stats.truncated += 1
    doc = document_from_rendered(
        index, rendered, max_length=self._width, overlong=self.config.overlong
    )
    if doc is None and len(rendered.input_ids) <= self._width:
      self.stats.dropped += 1  # no assistant token to train
    return doc

  def _next_document(self) -> SFTDocument:
    while True:
      if self._cursor >= len(self._shard):
        self._epoch += 1
        self._cursor = 0
        self._shard = self._epoch_shard(self._epoch)
        self.stats.epoch = self._epoch
      index = self._shard[self._cursor]
      self._cursor += 1
      doc = self.tokenize(index)
      if doc is not None:
        return doc

  def next_microbatch(self) -> dict[str, Any]:
    while len(self._pending) < self.config.lookahead_documents:
      self._pending.append(self._next_document())
    placed, self._pending = pack_rows(
        self._pending,
        rows=self._rows,
        width=self._width,
        max_docs_per_row=self.config.max_documents_per_row,
    )
    batch = build_microbatch(placed, width=self._width, pad_id=self._pad_id)
    docs = sum(len(r) for r in placed)
    tokens = sum(d.length for r in placed for d in r)
    self.stats.microbatches += 1
    self.stats.documents += docs
    self.stats.tokens += tokens
    self.stats.valid_tokens += batch["num_valid_tokens"]
    self.stats.last = {
        "documents": docs,
        "tokens": tokens,
        "valid_tokens": batch["num_valid_tokens"],
        "padded_tokens": self._rows * self._width,
    }
    self.batch_log.append(
        {
            "microbatch": self.stats.microbatches,
            "sha256": batch_fingerprint(batch),
            "docs": [doc.index for row in placed for doc in row],
        }
    )
    if len(self.batch_log) > 4096:  # bounded when nobody drains it
      del self.batch_log[:-4096]
    return batch

  def accumulation_steps(self) -> int:
    """Microbatches that make up one optimizer step on this rank: the
    configured ``microbatches_per_step``, fixed for the loader's lifetime."""
    return int(self.config.microbatches_per_step)

  # -- iteration and resumable position ---------------------------------
  def __iter__(self) -> Iterator[TrainerBatch]:
    """Microbatches without end; all iterators share this loader's position."""
    return iter(self.next_microbatch, None)

  def state_dict(self) -> dict[str, Any]:
    rank_key = f"dp_rank_{self._dp_rank}"
    position = dict(
        epoch=self._epoch,
        cursor=self._cursor,
        pending=[doc.index for doc in self._pending],
    )
    return {
        "version": 1,
        "dp_world_size": self._dp_world_size,
        rank_key: position,
    }

  def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    """Continue from a ``state_dict()``; an empty state changes nothing.

    The state must be version 1, come from as many data-parallel ranks and
    hold this rank's entry; otherwise ``ValueError``, with the position left
    as it was. Pending documents are tokenized again in their saved order.
    """
    if not state_dict:
      return
    rank_key = "dp_rank_{}".format(self._dp_rank)
    complaint = None
    if state_dict.get("version") != 1:
      complaint = "unsupported ChatSFTDataLoader state version {!r}".format(
          state_dict.get("version")
      )
    elif state_dict.get("dp_world_size") != self._dp_world_size:
      complaint = (
          "the loader state comes from {} data-parallel ranks, this run has "
          "{}".format(state_dict.get("dp_world_size"), self._dp_world_size)
      )
    elif rank_key not in state_dict:
      complaint = "the loader state has no {} entry".format(rank_key)
    if complaint is not None:
      raise ValueError(complaint)
    position = state_dict[rank_key]
    self._epoch = int(position["epoch"])
    self._cursor = int(position["cursor"])
    self._shard = self._epoch_shard(self._epoch)
    self._pending = [
        doc
        for doc in map(self.tokenize, map(int, position["pending"]))
        if doc is not None
    ]
    self.stats.epoch = self._epoch
