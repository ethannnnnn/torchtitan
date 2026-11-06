# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Chat rendering for SFT on top of Hugging Face ``apply_chat_template``.

Conversations and prompts are rendered by the chat template that comes with
the model's tokenizer; for Muse Spark that is the public template published
together with the open weights. Around the plain Hugging Face call this
module adds three things:

1. Repeatable text. Templates that print the current day when
   ``current_date`` is undefined always receive ``current_date`` from
   ``ChatTemplateOptions``, so a render never depends on the clock; ``None``
   leaves the date line out.
2. Optional template switches such as ``reasoning_effort``, handed only to a
   template whose source refers to them.
3. A mask over the assistant's tokens, built in one of ``MASK_MODES``.
   ``"generation"`` reads the template's ``{% generation %}`` sections with
   ``return_assistant_tokens_mask``. ``"prefix"`` renders the conversation
   up to every assistant turn and requires each of those renders to match
   the start of the full render token for token. ``"auto"`` takes
   ``"generation"`` when the template has such sections, else ``"prefix"``.
   The generation prompt that opens an assistant turn (``<|start|>assistant``
   for Muse Spark) stays untrained unless ``train_on_generation_prompt``.

Sampled replies of templates with message headers
(``<|start|>assistant to=RECIPIENT<|message|>BODY``, ended by ``<|eot|>`` or
``<|eom|>``) are cut into assistant messages by ``split_response``, and
``decode_final_answer`` picks the one meant for the user. Text without
headers falls back to the decoded string minus special tokens and
``<think>`` blocks.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

MASK_MODES = ("auto", "generation", "prefix")


@dataclass(frozen=True)
class ChatTemplateOptions:
  """Keyword arguments for ``apply_chat_template``."""

  reasoning_effort: str | None = None
  """Passed as ``reasoning_effort`` when the template references it (Muse
    Spark: low, medium, high or xhigh). None leaves the template default."""

  current_date: str | None = None
  """Always passed as ``current_date`` when the template references it, so
    renders never depend on the wall clock. None omits the date line."""

  extra: tuple[tuple[str, Any], ...] = ()
  """Further template keyword arguments, e.g. (("enable_thinking", False),)."""

  def with_extra(self, **kwargs: Any) -> ChatTemplateOptions:
    """A copy with ``kwargs`` merged into ``extra`` (they win on equal keys),
    the pairs sorted by key."""
    # dict() keeps the last value of a repeated key, and kwargs come last.
    return dataclasses.replace(
        self, extra=tuple(sorted(dict([*self.extra, *kwargs.items()]).items()))
    )


def load_hf_tokenizer(location: str | os.PathLike[str]) -> Any:
  """``transformers.AutoTokenizer`` loaded from a local folder or a hub id;
  its ``chat_template`` is what this module renders with. ``transformers`` is
  imported on the first call only."""
  import transformers

  return transformers.AutoTokenizer.from_pretrained(os.fspath(location))


def template_source(tokenizer: Any) -> str:
  template = getattr(tokenizer, "chat_template", None)
  if isinstance(template, dict):  # named templates: use the default one
    template = template.get("default") or next(iter(template.values()), None)
  if not template:
    raise ValueError("tokenizer has no chat template")
  return str(template)


def template_references(tokenizer: Any, name: str) -> bool:
  """True when the chat template mentions the variable ``name``."""
  return (
      re.search(rf"\b{re.escape(name)}\b", template_source(tokenizer))
      is not None
  )


def template_kwargs(
    tokenizer: Any, options: ChatTemplateOptions
) -> dict[str, Any]:
  """The keyword arguments this template accepts, from ``options``."""
  kwargs: dict[str, Any] = {}
  if template_references(tokenizer, "current_date"):
    kwargs["current_date"] = options.current_date
  if options.reasoning_effort is not None:
    if not template_references(tokenizer, "reasoning_effort"):
      raise ValueError(
          "reasoning_effort was set but the chat template does not use it"
      )
    kwargs["reasoning_effort"] = options.reasoning_effort
  kwargs.update(dict(options.extra))
  return kwargs


# A conversation: chat messages, each a mapping with at least ``"role"``.
Conversation = Sequence[Mapping[str, Any]]


def render_text(
    tokenizer: Any,
    messages: Conversation,
    options: ChatTemplateOptions,
    *,
    add_generation_prompt: bool,
) -> str:
  """The template's output for plain ``dict`` copies of ``messages``."""
  return tokenizer.apply_chat_template(
      [*map(dict, messages)],
      add_generation_prompt=add_generation_prompt,
      tokenize=False,
      **template_kwargs(tokenizer, options),
  )


def encode_rendered(tokenizer: Any, text: str) -> list[int]:
  """Token ids of rendered text. The template wrote the BOS token already,
  so the tokenizer adds no special tokens."""
  encoded = tokenizer(text, add_special_tokens=False)
  is_mapping = hasattr(encoded, "keys")
  return [
      int(token) for token in (encoded["input_ids"] if is_mapping else encoded)
  ]


def render_prompt_ids(tokenizer: Any, messages: Conversation,
                      options: ChatTemplateOptions) -> list[int]:
  """Ids of ``messages`` followed by the generation prompt, where sampling
  starts."""
  return encode_rendered(
      tokenizer,
      render_text(tokenizer, messages, options, add_generation_prompt=True),
  )


def generation_prompt_text(tokenizer: Any, options: ChatTemplateOptions) -> str:
  """Text the template appends for ``add_generation_prompt`` after a user turn."""
  probe = [{"role": "user", "content": "x"}]
  with_prompt = render_text(
      tokenizer, probe, options, add_generation_prompt=True
  )
  without = render_text(tokenizer, probe, options, add_generation_prompt=False)
  if not with_prompt.startswith(without):
    raise ValueError("generation prompt is not a suffix of the rendered turn")
  return with_prompt[len(without) :]


@dataclass
class RenderedChat:
  """One tokenized conversation with its assistant-token mask."""

  input_ids: list[int]
  assistant_mask: list[bool]

  @property
  def num_assistant_tokens(self) -> int:
    return sum(self.assistant_mask)


def _true_runs(mask: Sequence[bool]) -> list[tuple[int, int]]:
  """``(start, stop)`` of every maximal stretch of true entries (stop
  exclusive), in order."""
  padded = [False, *map(bool, mask), False]
  edges = [i for i in range(len(padded) - 1) if padded[i] != padded[i + 1]]
  return list(zip(edges[::2], edges[1::2]))


def _mask_from_generation_blocks(
    tokenizer: Any, messages: Conversation, options: ChatTemplateOptions
) -> RenderedChat:
  """The template's own assistant mask: its ``{% generation %}`` sections,
  which begin with the generation prompt."""
  encoded = tokenizer.apply_chat_template(
      [*map(dict, messages)],
      tokenize=True,
      return_dict=True,
      return_assistant_tokens_mask=True,
      **template_kwargs(tokenizer, options),
  )
  ids = list(map(int, encoded["input_ids"]))
  mask = list(map(bool, encoded["assistant_masks"]))
  if len(mask) != len(ids):
    raise ValueError(
        "the template returned {} assistant-mask entries for {} tokens".format(
            len(mask), len(ids)
        )
    )
  return RenderedChat(ids, mask)


def _mask_from_prefixes(
    tokenizer: Any, messages: Conversation, options: ChatTemplateOptions
) -> RenderedChat:
  """Assistant mask found by rendering the conversation up to each reply:
  a reply is what the render that includes it adds to the render that ends
  with the generation prompt."""

  def encoded_head(count: int, open_reply: bool) -> list[int]:
    return encode_rendered(
        tokenizer,
        render_text(
            tokenizer,
            messages[:count],
            options,
            add_generation_prompt=open_reply,
        ),
    )

  whole = encoded_head(len(messages), False)
  replies = [
      index
      for index, message in enumerate(messages)
      if message.get("role") == "assistant"
  ]
  mask = [False] * len(whole)
  for index in replies:
    opened, closed = encoded_head(index, True), encoded_head(index + 1, False)
    if whole[: len(opened)] != opened or whole[: len(closed)] != closed:
      raise ValueError(
          "message {}: the renders up to this assistant reply are not token "
          "prefixes of the whole conversation; use a template with "
          '{{% generation %}} blocks and mask_mode="generation"'.format(index)
      )
    mask[len(opened) : len(closed)] = [True] * (len(closed) - len(opened))
  return RenderedChat(whole, mask)


# A ``{% generation %}`` tag, with or without whitespace control and spaces.
_GENERATION_TAG = re.compile(r"\{%-?\s*generation\s*-?%\}")

# The mask builder of each resolved mode, and whether the stretches it marks
# include the generation prompt: generation sections start with it, while a
# prefix render already ends with it, before the reply.
_MASK_BUILDERS = {
    "generation": (_mask_from_generation_blocks, True),
    "prefix": (_mask_from_prefixes, False),
}


def _resolved_mask_mode(tokenizer: Any, requested: str) -> str:
  """``requested``, with ``"auto"`` settled by the template source."""
  if requested == "auto":
    if _GENERATION_TAG.search(template_source(tokenizer)):
      return "generation"
    return "prefix"
  if requested in MASK_MODES:
    return requested
  raise ValueError(
      "mask_mode {!r} is not supported; pick one of {}".format(
          requested, MASK_MODES
      )
  )


def _untrain_headers(chat: RenderedChat, header: list[int]) -> None:
  """Turn the mask off over ``header`` wherever a marked run begins with it,
  never past the end of that run."""
  size = len(header)
  for start, stop in _true_runs(chat.assistant_mask):
    if chat.input_ids[start : start + size] == header:
      end = min(start + size, stop)
      chat.assistant_mask[start:end] = [False] * (end - start)


def _train_headers(chat: RenderedChat, header: list[int]) -> None:
  """Set the mask on the header just before each trained stretch."""
  size = len(header)
  for start, _ in _true_runs(chat.assistant_mask):
    first = start - size
    if first >= 0 and chat.input_ids[first:start] == header:
      chat.assistant_mask[first:start] = [True] * size


def render_chat(
    tokenizer: Any,
    messages: Conversation,
    options: ChatTemplateOptions,
    *,
    mask_mode: str = "auto",
    train_on_generation_prompt: bool = False,
) -> RenderedChat:
  """Ids of the whole conversation and the mask of its assistant tokens.

  ``mask_mode`` is one of ``MASK_MODES``. The generation prompt in front of
  each reply joins the mask only with ``train_on_generation_prompt``.
  """
  mode = _resolved_mask_mode(tokenizer, mask_mode)
  build, header_inside = _MASK_BUILDERS[mode]
  chat = build(tokenizer, messages, options)
  if header_inside == bool(train_on_generation_prompt):
    return chat
  header = encode_rendered(
      tokenizer, generation_prompt_text(tokenizer, options)
  )
  adjust = _train_headers if train_on_generation_prompt else _untrain_headers
  if header:
    adjust(chat, header)
  return chat


def stop_token_ids(
    model_dir: str | os.PathLike[str] | None, tokenizer: Any
) -> list[int]:
  """Ids that end a sampled reply.

  ``eos_token_id`` of ``<model_dir>/generation_config.json`` when that file
  exists and holds an int or a non-empty list there; otherwise the
  tokenizer's ``eos_token_id``.
  """
  if model_dir is not None:
    settings = Path(model_dir) / "generation_config.json"
    if settings.is_file():
      eos = json.loads(settings.read_text(encoding="utf-8")).get("eos_token_id")
      if isinstance(eos, int):
        return [eos]
      if isinstance(eos, list) and eos:
        return list(map(int, eos))
  fallback = getattr(tokenizer, "eos_token_id", None)
  if fallback is None:
    raise ValueError(
        "cannot tell where replies end: no eos_token_id in "
        "generation_config.json or on the tokenizer"
    )
  return [int(fallback)]


# ---------------------------------------------------------------------------
# Sampled responses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResponseMarkers:
  """Delimiters of a header-style assistant response (Muse Spark defaults)."""

  start: str = "<|start|>"
  message: str = "<|message|>"
  ends: tuple[str, ...] = ("<|eot|>", "<|eom|>", "<|end_of_text|>")
  role: str = "assistant"
  final_recipients: tuple[str, ...] = ("user",)


@dataclass
class AssistantMessage:
  recipient: str
  body: str
  closed: bool


@dataclass
class ParsedResponse:
  messages: list[AssistantMessage] = field(default_factory=list)
  final_text: str = ""
  structured: bool = False
  """True when the completion used the header markers."""


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def split_response(
    text: str,
    *,
    special_tokens: Sequence[str] = (),
    markers: ResponseMarkers = ResponseMarkers(),
    opened: bool = True,
) -> ParsedResponse:
  """Assistant messages of a completion decoded with special tokens kept.

  ``opened``: the prompt already ended with ``<start><role>`` (the generation
  prompt), so the completion begins with the first header's tail.
  """
  if markers.message not in text:
    plain = text
    for token in special_tokens:
      plain = plain.replace(token, "")
    plain = _THINK.sub("", plain).strip()
    return ParsedResponse(final_text=plain, structured=False)
  head = markers.start + markers.role if opened else ""
  chunks = (head + text).split(markers.start)
  messages: list[AssistantMessage] = []
  for chunk in chunks:
    if markers.message not in chunk:
      continue
    header, body = chunk.split(markers.message, 1)
    header = header.strip()
    if header.startswith(markers.role):
      header = header[len(markers.role) :].strip()
    recipient = header[3:].strip() if header.startswith("to=") else header
    closed = False
    cut = len(body)
    for end in markers.ends:
      position = body.find(end)
      if position != -1 and position < cut:
        cut, closed = position, True
    messages.append(AssistantMessage(recipient, body[:cut], closed))
  final = ""
  for message in reversed(messages):
    if message.recipient in markers.final_recipients:
      final = message.body
      break
  else:
    if messages and messages[-1].recipient not in ("self",):
      final = messages[-1].body
  return ParsedResponse(
      messages=messages, final_text=final.strip(), structured=True
  )


def decode_final_answer(
    tokenizer: Any,
    completion_ids: Sequence[int],
    markers: ResponseMarkers = ResponseMarkers(),
) -> ParsedResponse:
  """``split_response`` of sampled ids decoded with their special tokens
  (the prompt is taken to end with the generation prompt)."""
  text = tokenizer.decode(
      list(map(int, completion_ids)), skip_special_tokens=False
  )
  specials = list(getattr(tokenizer, "all_special_tokens", None) or [])
  return split_response(text, special_tokens=specials, markers=markers)
