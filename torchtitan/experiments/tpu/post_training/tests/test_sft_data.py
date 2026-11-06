# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests: chat-template masks, SFT packing and the SFT dataloader.

Tokenizer tests need ``SPARK_TEST_ASSETS``: a directory with a Muse Spark
tokenizer, chat template and generation config (for example a snapshot of the
public tiny test model trl-internal-testing/tiny-MuseSparkForConditionalGeneration).
"""

from __future__ import annotations

import json
import os
import tempfile
import types
import unittest
from pathlib import Path

import torch

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.experiments.tpu.post_training import chat_template as ct
from torchtitan.experiments.tpu.post_training import sft_data

ASSETS = os.environ.get("SPARK_TEST_ASSETS", "")

CONVERSATIONS = [
    [
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": "What is 2 + 2?"},
        {
            "role": "assistant",
            "content": "4",
            "reasoning_content": "Simple sum.",
        },
    ],
    [
        {"role": "user", "content": "Name a color."},
        {"role": "assistant", "content": "Blue."},
        {"role": "user", "content": "Another one?"},
        {"role": "assistant", "content": "Green."},
    ],
    [
        {"role": "user", "content": "Say hi " * 30},
        {"role": "assistant", "content": "hi " * 20},
    ],
]


def _write_jsonl(rows, path):
  with open(path, "w") as handle:
    for row in rows:
      handle.write(json.dumps(row) + "\n")


class PackingTest(unittest.TestCase):
  """Tokenizer-free checks of the packer and the batch layout."""

  def _doc(self, index, length, trained):
    ids = list(range(100 + index * 10, 100 + index * 10 + length))
    mask = [i >= length - trained for i in range(length)]
    rendered = ct.RenderedChat(ids, mask)
    return sft_data.document_from_rendered(
        index, rendered, max_length=16, overlong="drop"
    )

  def test_labels_shift_and_skip(self):
    doc = self._doc(0, 6, 3)
    self.assertEqual(
        doc.labels, [IGNORE_INDEX, IGNORE_INDEX, 103, 104, 105, IGNORE_INDEX]
    )
    self.assertEqual(doc.num_valid, 3)
    self.assertIsNone(self._doc(1, 5, 0))  # nothing to train
    long_rendered = ct.RenderedChat(list(range(20)), [True] * 20)
    self.assertIsNone(
        sft_data.document_from_rendered(
            2, long_rendered, max_length=16, overlong="drop"
        )
    )
    cut = sft_data.document_from_rendered(
        2, long_rendered, max_length=16, overlong="truncate"
    )
    self.assertEqual(cut.length, 16)
    self.assertEqual(cut.num_valid, 15)

  def test_first_fit_decreasing_and_layout(self):
    docs = [self._doc(i, n, 2) for i, n in enumerate([9, 7, 6, 5, 4, 3])]
    placed, left = sft_data.pack_rows(
        docs, rows=2, width=16, max_docs_per_row=2
    )
    self.assertEqual(
        [[d.length for d in row] for row in placed], [[9, 7], [6, 5]]
    )
    self.assertEqual([d.length for d in left], [4, 3])  # arrival order kept
    batch = sft_data.build_microbatch(placed, width=16, pad_id=0)
    positions = batch["positions"].view(2, 16)
    self.assertEqual(positions[0].tolist(), list(range(9)) + list(range(7)))
    self.assertEqual(
        positions[1, :11].tolist(), list(range(6)) + list(range(5))
    )
    self.assertEqual(
        positions[1, 11:].tolist(), list(range(5, 10))
    )  # pad continues
    self.assertEqual(int(batch["padding_mask"].sum()), 5)
    self.assertEqual(batch["num_valid_tokens"], 8)
    labels = batch["labels"].view(2, 16)
    self.assertTrue((labels[1, 11:] == IGNORE_INDEX).all())
    # A document's last position never predicts the next document's first token.
    self.assertEqual(int(labels[0, 8]), IGNORE_INDEX)
    empty = sft_data.build_microbatch([[], []], width=4, pad_id=7)
    self.assertEqual(empty["positions"].tolist(), [0, 1, 2, 3, 0, 1, 2, 3])
    self.assertEqual(empty["num_valid_tokens"], 0)

  def test_batch_fingerprint(self):
    docs = [self._doc(i, n, 2) for i, n in enumerate([9, 7])]
    placed, _ = sft_data.pack_rows(docs, rows=1, width=16, max_docs_per_row=2)
    a = sft_data.build_microbatch(placed, width=16, pad_id=0)
    b = sft_data.build_microbatch(placed, width=16, pad_id=0)
    self.assertEqual(
        sft_data.batch_fingerprint(a), sft_data.batch_fingerprint(b)
    )
    b["labels"][7] = IGNORE_INDEX  # one supervised target fewer
    self.assertNotEqual(
        sft_data.batch_fingerprint(a), sft_data.batch_fingerprint(b)
    )

  def test_progress_guarantee(self):
    docs = [self._doc(i, 16, 2) for i in range(3)]
    placed, left = sft_data.pack_rows(
        docs, rows=1, width=16, max_docs_per_row=4
    )
    self.assertEqual(sum(len(r) for r in placed), 1)
    self.assertEqual(len(left), 2)


class LoaderArgumentTest(unittest.TestCase):
  """Constructor checks that fail before any record or tokenizer is read."""

  def _loader(self, *, tokens=1024, template_source="", tokenizer=None):
    config = sft_data.ChatSFTDataLoader.Config(
        data_path="/nonexistent/sft.jsonl", hf_tokenizer_path=template_source
    )
    return sft_data.ChatSFTDataLoader(
        config,
        dp_world_size=1,
        dp_rank=0,
        tokenizer=tokenizer,
        max_context_length=512,
        num_tokens_per_batch=tokens,
    )

  def test_token_budget_must_fill_whole_rows(self):
    with self.assertRaisesRegex(ValueError, "1000") as caught:
      self._loader(tokens=1000, template_source="/unused")
    self.assertIn("512", str(caught.exception))

  def test_chat_template_source_required(self):
    # The data path does not exist, so a ValueError (not FileNotFoundError)
    # shows that the check runs before any data is read.
    for tokenizer in (None, object(), types.SimpleNamespace(tokenizer_path="")):
      with self.subTest(tokenizer=tokenizer):
        with self.assertRaisesRegex(ValueError, "hf-tokenizer-path"):
          self._loader(tokenizer=tokenizer)


class MicrobatchHookTest(unittest.TestCase):
  """The per-step microbatch count that SFTTPUTrainer reads from its loader."""

  def test_protocol_needs_the_method(self):
    from torchtitan.experiments.tpu.post_training.sft_train import (
        MicrobatchesPerStep,
    )

    class FixedCount:

      def num_microbatches_per_step(self) -> int:
        return 3

    self.assertIsInstance(FixedCount(), MicrobatchesPerStep)
    self.assertNotIsInstance(object(), MicrobatchesPerStep)


class _WordTokenizer:
  """Recording stand-in for a Hugging Face tokenizer: one token per word.

  The template writes ``<role> content`` for each message and ``<assistant>``
  as the generation prompt. With ``sections`` its source carries
  ``{% generation %}`` blocks, and ``apply_chat_template(tokenize=True)``
  returns a mask over each assistant turn, header included. With ``bare``,
  ``__call__`` returns an id tuple instead of a dict.
  """

  eos_token_id = 2
  all_special_tokens = ["<eos>"]

  def __init__(self, *, sections=False, bare=False):
    reply = "{{ m.role }} {{ m.content }}"
    if sections:
      reply = "{% generation %}" + reply + "{% endgeneration %}"
    self.chat_template = (
        "{% for m in messages %}{% if m.role == 'assistant' %}" + reply
        + "{% else %}{{ m.role }} {{ m.content }}{% endif %}{% endfor %}"
        "{{ current_date }}"
    )
    self.bare = bare
    self.calls = []
    self.vocab = {}

  def _ids(self, words):
    return [self.vocab.setdefault(word, 10 + len(self.vocab)) for word in words]

  def _turns(self, conversation, add_generation_prompt):
    turns = [
        (m["role"] == "assistant", ["<%s>" % m["role"], *m["content"].split()])
        for m in conversation
    ]
    if add_generation_prompt:
      turns.append((False, ["<assistant>"]))
    return turns

  def apply_chat_template(self, conversation, **kwargs):
    self.calls.append(("apply_chat_template", conversation, kwargs))
    turns = self._turns(conversation, kwargs.get("add_generation_prompt"))
    if not kwargs.get("tokenize"):
      return "".join(" ".join(words) + " " for _, words in turns)
    ids, mask = [], []
    for trained, words in turns:
      ids += self._ids(words)
      mask += [trained] * len(words)
    return {"input_ids": ids, "assistant_masks": mask}

  def __call__(self, text, **kwargs):
    self.calls.append(("__call__", text, kwargs))
    ids = self._ids(text.split())
    return tuple(ids) if self.bare else {"input_ids": ids}

  def decode(self, ids, **kwargs):
    self.calls.append(("decode", ids, kwargs))
    words = {value: word for word, value in self.vocab.items()}
    return " ".join(words[i] for i in ids)


class ChatTemplateCallTest(unittest.TestCase):
  """The exact Hugging Face calls of ``chat_template``, on ``_WordTokenizer``."""

  CHAT = (
      types.MappingProxyType({"role": "system", "content": "Be brief."}),
      {"role": "user", "content": "Two plus two?"},
      {"role": "assistant", "content": "Four.", "reasoning_content": "Sum."},
      {"role": "user", "content": "And three?"},
      {"role": "assistant", "content": "Five."},
  )
  OPTIONS = ct.ChatTemplateOptions()

  @staticmethod
  def _trained_words(tok, rendered):
    words = {value: word for word, value in tok.vocab.items()}
    pairs = zip(rendered.input_ids, rendered.assistant_mask)
    return [words[token] for token, keep in pairs if keep]

  @staticmethod
  def _template_calls(tok):
    return [call for call in tok.calls if call[0] == "apply_chat_template"]

  def test_text_render(self):
    tok = _WordTokenizer()
    text = ct.render_text(
        tok, self.CHAT[:2], self.OPTIONS, add_generation_prompt=True
    )
    self.assertEqual(text, "<system> Be brief. <user> Two plus two? <assistant> ")
    ((name, conversation, kwargs),) = tok.calls
    self.assertEqual(name, "apply_chat_template")
    self.assertEqual(
        kwargs,
        {"add_generation_prompt": True, "tokenize": False, "current_date": None},
    )
    self.assertIs(type(conversation), list)
    self.assertEqual([type(m) for m in conversation], [dict, dict])
    self.assertEqual(conversation, [dict(m) for m in self.CHAT[:2]])
    self.assertIsNot(conversation[1], self.CHAT[1])
    thinking = self.OPTIONS.with_extra(enable_thinking=False)
    ct.render_text(tok, self.CHAT, thinking, add_generation_prompt=False)
    self.assertEqual(
        tok.calls[-1][2],
        {
            "add_generation_prompt": False,
            "tokenize": False,
            "current_date": None,
            "enable_thinking": False,
        },
    )
    clash = self.OPTIONS.with_extra(tokenize=True)
    with self.assertRaises(TypeError):
      ct.render_text(tok, self.CHAT, clash, add_generation_prompt=False)

  def test_encoding(self):
    for bare in (False, True):
      with self.subTest(bare=bare):
        tok = _WordTokenizer(bare=bare)
        ids = ct.encode_rendered(tok, "<user> hi <assistant> ")
        self.assertEqual(ids, [10, 11, 12])
        self.assertTrue(all(type(token) is int for token in ids))
        self.assertEqual(
            tok.calls,
            [("__call__", "<user> hi <assistant> ", {"add_special_tokens": False})],
        )

  def test_prompt_ids(self):
    tok = _WordTokenizer()
    ids = ct.render_prompt_ids(tok, list(self.CHAT[:2]), self.OPTIONS)
    self.assertEqual(
        [call[0] for call in tok.calls], ["apply_chat_template", "__call__"]
    )
    self.assertIs(tok.calls[0][2]["add_generation_prompt"], True)
    self.assertEqual(len(ids), 8)  # <system> Be brief. <user> ... <assistant>
    self.assertEqual(ids[-1], tok.vocab["<assistant>"])

  def test_generation_sections(self):
    tok = _WordTokenizer(sections=True)
    chat = ct.render_chat(tok, self.CHAT, self.OPTIONS)  # "auto" finds sections
    masked = [c for c in self._template_calls(tok) if c[2].get("tokenize")]
    ((_, conversation, kwargs),) = masked
    self.assertEqual(
        kwargs,
        {
            "tokenize": True,
            "return_dict": True,
            "return_assistant_tokens_mask": True,
            "current_date": None,
        },
    )
    self.assertEqual(conversation, [dict(m) for m in self.CHAT])
    self.assertEqual(self._trained_words(tok, chat), ["Four.", "Five."])
    headed = ct.render_chat(
        tok, self.CHAT, self.OPTIONS, train_on_generation_prompt=True
    )
    self.assertEqual(headed.input_ids, chat.input_ids)
    self.assertEqual(
        self._trained_words(tok, headed),
        ["<assistant>", "Four.", "<assistant>", "Five."],
    )
    for train in (False, True):
      with self.subTest(train_on_generation_prompt=train):
        renders = [
            ct.render_chat(
                tok,
                self.CHAT,
                self.OPTIONS,
                mask_mode=mode,
                train_on_generation_prompt=train,
            )
            for mode in ("prefix", "generation")
        ]
        self.assertEqual(renders[0].input_ids, renders[1].input_ids)
        self.assertEqual(renders[0].assistant_mask, renders[1].assistant_mask)

  def test_auto_without_sections_renders_prefixes(self):
    tok = _WordTokenizer()
    chat = ct.render_chat(tok, list(self.CHAT), self.OPTIONS)
    self.assertEqual(self._trained_words(tok, chat), ["Four.", "Five."])
    # The whole conversation first, then per reply the prefix that ends with
    # the generation prompt and the one that ends with the reply.
    self.assertEqual(
        [
            (len(conversation), kwargs["add_generation_prompt"])
            for _, conversation, kwargs in self._template_calls(tok)
        ],
        [(5, False), (2, True), (3, False), (4, True), (5, False)],
    )

  def test_errors(self):
    tok = _WordTokenizer()
    with self.assertRaises(ValueError):
      ct.render_chat(tok, self.CHAT, self.OPTIONS, mask_mode="assistant")
    self.assertEqual(tok.calls, [])  # refused before any tokenizer call
    effort = ct.ChatTemplateOptions(reasoning_effort="low")
    with self.assertRaises(ValueError):  # this template has no such switch
      ct.render_text(tok, self.CHAT, effort, add_generation_prompt=True)

    class Drifting(_WordTokenizer):
      """Renders that end with the generation prompt spell it differently."""

      def apply_chat_template(self, conversation, **kwargs):
        text = super().apply_chat_template(conversation, **kwargs)
        if kwargs.get("add_generation_prompt"):
          return text.replace("<assistant>", "<reply>")
        return text

    with self.assertRaises(ValueError):
      ct.render_chat(Drifting(), self.CHAT, self.OPTIONS, mask_mode="prefix")

    class ShortMask(_WordTokenizer):
      """The assistant mask comes back one entry short."""

      def apply_chat_template(self, conversation, **kwargs):
        out = super().apply_chat_template(conversation, **kwargs)
        if kwargs.get("tokenize"):
          out["assistant_masks"] = out["assistant_masks"][:-1]
        return out

    with self.assertRaises(ValueError):
      ct.render_chat(ShortMask(sections=True), self.CHAT, self.OPTIONS)

  def test_with_extra(self):
    base = ct.ChatTemplateOptions(
        reasoning_effort="low", extra=(("b", 1), ("a", 2))
    )
    merged = base.with_extra(c=3, a=4)
    self.assertEqual(merged.extra, (("a", 4), ("b", 1), ("c", 3)))
    self.assertEqual(merged.reasoning_effort, "low")
    self.assertEqual(base.extra, (("b", 1), ("a", 2)))  # unchanged
    self.assertEqual(base.with_extra().extra, (("a", 2), ("b", 1)))

  def test_stop_token_ids(self):
    tok = _WordTokenizer()
    with tempfile.TemporaryDirectory() as tmpdir:
      config = Path(tmpdir, "generation_config.json")
      for value, want in ((7, [7]), ([5, "6"], [5, 6]), ([], [2]), ("x", [2])):
        with self.subTest(eos_token_id=value):
          config.write_text(json.dumps({"eos_token_id": value}))
          self.assertEqual(ct.stop_token_ids(tmpdir, tok), want)
      config.write_text("{}")
      self.assertEqual(ct.stop_token_ids(Path(tmpdir), tok), [2])
      missing = os.path.join(tmpdir, "absent")
      self.assertEqual(ct.stop_token_ids(missing, tok), [2])
    self.assertEqual(ct.stop_token_ids(None, tok), [2])
    tok.eos_token_id = None
    with self.assertRaises(ValueError):
      ct.stop_token_ids(None, tok)

  def test_decode_final_answer(self):
    tok = _WordTokenizer()
    ids = tok._ids(["plain", "answer", "<eos>"])
    parsed = ct.decode_final_answer(tok, tuple(ids))
    self.assertEqual(tok.calls, [("decode", ids, {"skip_special_tokens": False})])
    self.assertIs(type(tok.calls[0][1]), list)
    self.assertEqual((parsed.final_text, parsed.structured), ("plain answer", False))


@unittest.skipUnless(ASSETS, "SPARK_TEST_ASSETS is not set")
class ChatTemplateTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.tok = ct.load_hf_tokenizer(ASSETS)

  def test_kwargs_and_determinism(self):
    opts = ct.ChatTemplateOptions(reasoning_effort="low")
    kwargs = ct.template_kwargs(self.tok, opts)
    self.assertEqual(kwargs, {"current_date": None, "reasoning_effort": "low"})
    text = ct.render_text(
        self.tok, CONVERSATIONS[0][:2], opts, add_generation_prompt=True
    )
    self.assertIn("Reasoning strength: 8.", text)
    self.assertNotIn("Current date", text)
    high = ct.render_text(
        self.tok,
        CONVERSATIONS[0][:2],
        ct.ChatTemplateOptions(reasoning_effort="high"),
        add_generation_prompt=True,
    )
    self.assertIn("Reasoning strength: 256.", high)
    dated = ct.render_text(
        self.tok,
        CONVERSATIONS[0][:2],
        ct.ChatTemplateOptions(current_date="2026-01-31"),
        add_generation_prompt=True,
    )
    self.assertIn("Current date: 2026-01-31.", dated)
    self.assertTrue(text.endswith("<|start|>assistant"))
    with self.assertRaises(Exception):
      ct.render_text(
          self.tok,
          CONVERSATIONS[0][:2],
          ct.ChatTemplateOptions(reasoning_effort="extreme"),
          add_generation_prompt=True,
      )

  def test_assistant_mask_modes_agree(self):
    opts = ct.ChatTemplateOptions(reasoning_effort="medium")
    for conversation in CONVERSATIONS:
      gen = ct.render_chat(self.tok, conversation, opts, mask_mode="generation")
      pre = ct.render_chat(self.tok, conversation, opts, mask_mode="prefix")
      self.assertEqual(gen.input_ids, pre.input_ids)
      self.assertEqual(gen.assistant_mask, pre.assistant_mask)
      auto = ct.render_chat(self.tok, conversation, opts)
      self.assertEqual(auto.assistant_mask, gen.assistant_mask)

  def test_mask_content(self):
    opts = ct.ChatTemplateOptions()
    rendered = ct.render_chat(self.tok, CONVERSATIONS[0], opts)
    trained = self.tok.decode(
        [t for t, m in zip(rendered.input_ids, rendered.assistant_mask) if m],
        skip_special_tokens=False,
    )
    self.assertEqual(
        trained,
        " to=self<|message|>Simple sum.<|eom|><|start|>assistant to=user<|message|>4<|eot|>",
    )
    with_prompt = ct.render_chat(
        self.tok, CONVERSATIONS[0], opts, train_on_generation_prompt=True
    )
    self.assertEqual(
        with_prompt.num_assistant_tokens, rendered.num_assistant_tokens + 2
    )
    # Multi-turn: both assistant turns train, the user turns do not.
    multi = ct.render_chat(self.tok, CONVERSATIONS[1], opts)
    text = self.tok.decode(
        [t for t, m in zip(multi.input_ids, multi.assistant_mask) if m],
        skip_special_tokens=False,
    )
    self.assertIn("Blue.", text)
    self.assertIn("Green.", text)
    self.assertNotIn("Another one?", text)

  def test_prompt_ids_prefix_of_conversation(self):
    opts = ct.ChatTemplateOptions(reasoning_effort="low")
    prompt = ct.render_prompt_ids(self.tok, CONVERSATIONS[0][:2], opts)
    full = ct.render_chat(self.tok, CONVERSATIONS[0], opts)
    self.assertEqual(full.input_ids[: len(prompt)], prompt)
    self.assertFalse(any(full.assistant_mask[: len(prompt)]))
    self.assertTrue(full.assistant_mask[len(prompt)])

  def test_stop_ids_and_decode(self):
    stop = ct.stop_token_ids(ASSETS, self.tok)
    self.assertEqual(
        stop, self.tok.convert_tokens_to_ids(["<|end_of_text|>", "<|eot|>"])
    )
    self.assertNotIn(self.tok.convert_tokens_to_ids("<|eom|>"), stop)
    reply = '{"answer": 0.4}'
    ids = self.tok(
        " to=self<|message|>hmm<|eom|><|start|>assistant to=user<|message|>"
        + reply
        + "<|eot|>",
        add_special_tokens=False,
    )["input_ids"]
    parsed = ct.decode_final_answer(self.tok, ids)
    self.assertEqual(parsed.final_text, reply)


@unittest.skipUnless(ASSETS, "SPARK_TEST_ASSETS is not set")
class ChatSFTDataLoaderTest(unittest.TestCase):

  def _loader(self, path, rank=0, world=1, **overrides):
    config = sft_data.ChatSFTDataLoader.Config(
        data_path=path,
        hf_tokenizer_path=ASSETS,
        reasoning_effort="low",
        lookahead_documents=4,
        max_documents_per_row=3,
        **overrides,
    )
    return sft_data.ChatSFTDataLoader(
        config,
        dp_world_size=world,
        dp_rank=rank,
        tokenizer=None,
        max_context_length=512,
        num_tokens_per_batch=1024,
    )

  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    rows = [{"messages": CONVERSATIONS[i % 3]} for i in range(12)]
    rows[5]["messages"] = [
        {"role": "user", "content": "x " * 700},
        {"role": "assistant", "content": "too long"},
    ]
    self.path = str(Path(self.tmp.name) / "sft.jsonl")
    _write_jsonl(rows, self.path)

  def tearDown(self):
    self.tmp.cleanup()

  def test_batches(self):
    loader = self._loader(self.path)
    batch = next(iter(loader))
    self.assertEqual(batch["input"].shape, (1024,))
    self.assertEqual(
        batch["num_valid_tokens"], int((batch["labels"] != IGNORE_INDEX).sum())
    )
    self.assertGreater(batch["num_valid_tokens"], 0)
    starts = (batch["positions"].view(2, 512) == 0).sum(dim=1)
    self.assertTrue((starts <= 3).all())
    self.assertEqual(loader.max_num_documents, 6)
    for _ in range(8):
      next(iter(loader))
    self.assertGreaterEqual(loader.stats.dropped, 1)  # the overlong record
    self.assertGreaterEqual(loader.stats.epoch, 1)

  def test_dp_shards_are_disjoint(self):
    a, b = self._loader(self.path, 0, 2), self._loader(self.path, 1, 2)
    self.assertFalse(set(a._shard) & set(b._shard))
    self.assertEqual(len(a._shard), len(b._shard))

  def test_batch_log(self):
    loader = self._loader(self.path)
    it = iter(loader)
    batches = [next(it) for _ in range(3)]
    self.assertEqual([e["microbatch"] for e in loader.batch_log], [1, 2, 3])
    for entry, batch in zip(loader.batch_log, batches):
      self.assertEqual(entry["sha256"], sft_data.batch_fingerprint(batch))
      # One document per position reset outside the padding.
      starts = int(((batch["positions"] == 0) & ~batch["padding_mask"]).sum())
      self.assertEqual(len(entry["docs"]), starts)

  def test_resume_reproduces_batches(self):
    loader = self._loader(self.path)
    it = iter(loader)
    for _ in range(3):
      next(it)
    state = loader.state_dict()
    expected = [next(it) for _ in range(4)]
    resumed = self._loader(self.path)
    resumed.load_state_dict(state)
    rit = iter(resumed)
    for want in expected:
      got = next(rit)
      for key in ("input", "labels", "positions", "padding_mask"):
        self.assertTrue(torch.equal(want[key], got[key]), key)
      self.assertEqual(want["num_valid_tokens"], got["num_valid_tokens"])

  def test_state_dict_schema(self):
    loader = self._loader(self.path, rank=1, world=2)
    next(iter(loader))
    state = loader.state_dict()
    self.assertEqual(set(state), {"version", "dp_world_size", "dp_rank_1"})
    self.assertEqual((state["version"], state["dp_world_size"]), (1, 2))
    position = state["dp_rank_1"]
    self.assertEqual(set(position), {"epoch", "cursor", "pending"})
    self.assertIsInstance(position["epoch"], int)
    self.assertIsInstance(position["cursor"], int)
    self.assertIsInstance(position["pending"], list)
    self.assertTrue(all(isinstance(i, int) for i in position["pending"]))

  def test_load_empty_state_changes_nothing(self):
    loader = self._loader(self.path)
    next(iter(loader))
    before = loader.state_dict()
    loader.load_state_dict({})
    self.assertEqual(loader.state_dict(), before)

  def test_load_state_guards(self):
    loader = self._loader(self.path)
    state = loader.state_dict()
    with self.assertRaises(ValueError):
      loader.load_state_dict(dict(state, version=2))
    with self.assertRaises(ValueError):
      loader.load_state_dict(dict(state, dp_world_size=2))

  def test_microbatch_hook(self):
    from torchtitan.experiments.tpu.post_training.sft_train import (
        MicrobatchesPerStep,
    )

    loader = self._loader(self.path, microbatches_per_step=3)
    self.assertIsInstance(loader, MicrobatchesPerStep)
    self.assertEqual(loader.num_microbatches_per_step(), 3)


if __name__ == "__main__":
  unittest.main()
