# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Packed chat SFT dataloader.

Each record holds a conversation (``messages``: system / user / assistant /
tool turns; JSONL or parquet). ``ChatSFTDataLoader`` renders it with the
checkpoint's chat template (``chat_template.render_chat``), trains only the
assistant tokens, and packs conversations into the native fixed-shape batch:

    input            [T] int64  token ids, pad-filled
    labels           [T] int64  next-token targets; IGNORE_INDEX off the
                                assistant tokens, at document ends and on pad
    positions        [T] int64  restart at 0 at every document start
    padding_mask     [T] bool   True on padding
    num_valid_tokens int        supervised targets in this microbatch

``T = rows * max_context_length`` (``training.num_tokens_per_microbatch_per_
dp_rank``). A document never crosses a row; positions reset per document, so
models that derive document masks from position resets (flex / splash /
varlen attention) never attend across documents. The tail pad continues the
last document's positions, as in the native RL packer, so it creates no
phantom document. The trainer sums ``num_valid_tokens`` over microbatches and
data-parallel ranks and divides the summed cross-entropy by that global count.

Packing is first-fit decreasing over a lookahead window of documents, at most
``max_documents_per_row`` documents per row; documents that do not fit wait
for the next microbatch. Conversations longer than a row are dropped
(``overlong="drop"``) or cut to the row (``"truncate"``; only the retained
assistant tokens train). Data-parallel ranks read disjoint strided shards of
one shuffled epoch order; epochs reshuffle with ``shuffle_seed + epoch``. The
state dict holds the epoch, the shard cursor and the pending documents, so a
resumed run serves exactly the batches it would have served.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch

from torchtitan.components.data.loader import BaseDataLoader
from torchtitan.components.loss import IGNORE_INDEX

from . import chat_template as ct

logger = logging.getLogger(__name__)

OVERLONG_POLICIES = ("drop", "truncate")


def load_conversations(path: str | Path, *, limit: int | None = None) -> list[list[dict[str, Any]]]:
    """``messages`` lists from JSONL / parquet files (or a directory of them)."""
    resolved = Path(path)
    files = (
        sorted(p for p in resolved.iterdir() if p.suffix in (".jsonl", ".json", ".parquet"))
        if resolved.is_dir() else [resolved]
    )
    if not files or not all(f.is_file() for f in files):
        raise FileNotFoundError(f"no SFT records at {resolved}")
    conversations: list[list[dict[str, Any]]] = []
    for file in files:
        if file.suffix == ".parquet":
            import pyarrow.parquet as pq

            records = pq.read_table(file).to_pylist()
        else:
            records = [
                json.loads(line) for line in file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        for record in records:
            messages = record.get("messages")
            if isinstance(messages, str):
                messages = json.loads(messages)
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"SFT record {len(conversations)} has no messages")
            if not any(m.get("role") == "assistant" for m in messages):
                raise ValueError(f"SFT record {len(conversations)} has no assistant turn")
            conversations.append([dict(m) for m in messages])
            if limit is not None and len(conversations) >= limit:
                return conversations
    if not conversations:
        raise ValueError(f"{resolved} holds no SFT records")
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
    documents: list[SFTDocument], *, rows: int, width: int, max_docs_per_row: int
) -> tuple[list[list[SFTDocument]], list[SFTDocument]]:
    """First-fit decreasing placement. Returns (per-row documents, leftovers).

    The first document always lands (it fits an empty row), so every call
    makes progress; leftovers keep their arrival order.
    """
    order = sorted(range(len(documents)), key=lambda i: (-documents[i].length, i))
    used = [0] * rows
    placed: list[list[SFTDocument]] = [[] for _ in range(rows)]
    left: set[int] = set()
    for i in order:
        doc = documents[i]
        for row in range(rows):
            if len(placed[row]) < max_docs_per_row and used[row] + doc.length <= width:
                placed[row].append(doc)
                used[row] += doc.length
                break
        else:
            left.add(i)
    return placed, [documents[i] for i in range(len(documents)) if i in left]


def build_microbatch(
    placed: list[list[SFTDocument]], *, width: int, pad_id: int
) -> dict[str, Any]:
    rows = len(placed)
    ids = torch.full((rows, width), pad_id, dtype=torch.int64)
    labels = torch.full((rows, width), IGNORE_INDEX, dtype=torch.int64)
    positions = torch.zeros((rows, width), dtype=torch.int64)
    padding = torch.ones((rows, width), dtype=torch.bool)
    for row, docs in enumerate(placed):
        fill = 0
        for doc in docs:
            n = doc.length
            ids[row, fill : fill + n] = torch.tensor(doc.input_ids, dtype=torch.int64)
            labels[row, fill : fill + n] = torch.tensor(doc.labels, dtype=torch.int64)
            positions[row, fill : fill + n] = torch.arange(n, dtype=torch.int64)
            padding[row, fill : fill + n] = False
            fill += n
        if fill == 0:
            positions[row] = torch.arange(width, dtype=torch.int64)
        elif fill < width:
            positions[row, fill:] = positions[row, fill - 1] + 1 + torch.arange(width - fill)
    return {
        "input": ids.reshape(-1),
        "labels": labels.reshape(-1),
        "positions": positions.reshape(-1),
        "padding_mask": padding.reshape(-1),
        "num_valid_tokens": int((labels != IGNORE_INDEX).sum()),
    }


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


class ChatSFTDataLoader(BaseDataLoader):
    """Assistant-only, document-packed chat SFT batches."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        data_path: str = ""
        hf_tokenizer_path: str = ""
        """Chat-template source; empty = the trainer tokenizer's path."""
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
            raise ValueError("max_documents_per_row and lookahead_documents must be >= 1")
        if config.microbatches_per_step < 1:
            raise ValueError("microbatches_per_step must be >= 1")
        width = int(max_context_length)
        if num_tokens_per_batch % width:
            raise ValueError(
                f"num_tokens_per_microbatch_per_dp_rank {num_tokens_per_batch} must be "
                f"a multiple of max_context_length {width}"
            )
        self.config = config
        self._width = width
        self._rows = num_tokens_per_batch // width
        self.max_num_documents = self._rows * config.max_documents_per_row
        self._dp_world_size = int(dp_world_size)
        self._dp_rank = int(dp_rank)
        path = config.hf_tokenizer_path or getattr(tokenizer, "tokenizer_path", "")
        if not path:
            raise ValueError("ChatSFTDataLoader needs hf_tokenizer_path or --hf_assets_path")
        self._hf_tokenizer = ct.load_hf_tokenizer(path)
        self._options = ct.ChatTemplateOptions(
            reasoning_effort=config.reasoning_effort or None,
            current_date=config.current_date or None,
        )
        if config.pad_token_id >= 0:
            self._pad_id = int(config.pad_token_id)
        else:
            pad = getattr(self._hf_tokenizer, "pad_token_id", None)
            self._pad_id = int(pad if pad is not None else self._hf_tokenizer.eos_token_id)
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
        if dp_rank == 0:
            logger.info(
                "[sft] %d conversations, %d ranks, %d rows x %d tokens per microbatch, "
                "<= %d documents per row, template options %s, mask %s, overlong %s",
                len(self._conversations), self._dp_world_size, self._rows, width,
                config.max_documents_per_row, self._options, config.mask_mode,
                config.overlong,
            )

    # -- data order -------------------------------------------------------
    def _epoch_shard(self, epoch: int) -> list[int]:
        order = list(range(len(self._conversations)))
        if self.config.shuffle_seed is not None:
            random.Random(int(self.config.shuffle_seed) + epoch).shuffle(order)
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
        return batch

    def microbatches_for_step(self) -> int:
        """Accumulation count for the next optimizer step (fixed per recipe)."""
        return int(self.config.microbatches_per_step)

    # -- BaseDataLoader -----------------------------------------------------
    def __iter__(self) -> Iterator[dict[str, Any]]:
        while True:
            yield self.next_microbatch()

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "dp_world_size": self._dp_world_size,
            f"dp_rank_{self._dp_rank}": {
                "epoch": self._epoch,
                "cursor": self._cursor,
                "pending": [doc.index for doc in self._pending],
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return
        if state_dict.get("version") != 1:
            raise ValueError("unsupported ChatSFTDataLoader state version")
        if state_dict.get("dp_world_size") != self._dp_world_size:
            raise ValueError("cannot resume after changing the data-parallel degree")
        mine = state_dict[f"dp_rank_{self._dp_rank}"]
        self._epoch = int(mine["epoch"])
        self._cursor = int(mine["cursor"])
        self._shard = self._epoch_shard(self._epoch)
        self._pending = [doc for doc in (self.tokenize(i) for i in mine["pending"]) if doc]
        self.stats.epoch = self._epoch
