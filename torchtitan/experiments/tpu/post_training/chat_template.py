# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Chat-template rendering for SFT and RL.

Every prompt and every SFT conversation is rendered by the checkpoint's own
Hugging Face chat template (for Muse Spark: the public template shipped with
the open weights). This module adds the three things the trainers need on top
of ``apply_chat_template``:

* Deterministic rendering. Templates that stamp today's date when
  ``current_date`` is undefined (the Muse Spark template does) would make
  prompts change from day to day; ``ChatTemplateOptions`` always passes
  ``current_date`` explicitly (``None`` omits the date line).
* Template keyword arguments such as ``reasoning_effort`` are passed only when
  the template references them, so one options object serves templates with
  and without them.
* An assistant-only token mask. ``"generation"`` uses the template's
  ``{% generation %}`` blocks (``return_assistant_tokens_mask``); ``"prefix"``
  renders each assistant turn's prompt and checks that it is a token prefix of
  the full conversation; ``"auto"`` picks ``"generation"`` when the template has
  the blocks. By default the generation prompt that opens an assistant turn
  (``<|start|>assistant`` for Muse Spark) is not trained: at inference the
  prompt supplies it, and RL rollouts start right after it.

The response helpers split a sampled completion into the assistant messages of
a header-style template (``<|start|>assistant to=RECIPIENT<|message|>BODY``
closed by ``<|eot|>`` / ``<|eom|>``) and return the final user-facing message.
Completions without those markers fall back to the decoded text with special
tokens and ``<think>`` blocks removed.
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

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

    def with_extra(self, **kwargs: Any) -> "ChatTemplateOptions":
        merged = dict(self.extra)
        merged.update(kwargs)
        return dataclasses.replace(self, extra=tuple(sorted(merged.items())))


def load_hf_tokenizer(path: str | Path) -> Any:
    """The checkpoint's Hugging Face tokenizer, with its chat template."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path))


def template_source(tokenizer: Any) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, dict):  # named templates: use the default one
        template = template.get("default") or next(iter(template.values()), None)
    if not template:
        raise ValueError("tokenizer has no chat template")
    return str(template)


def template_references(tokenizer: Any, name: str) -> bool:
    """True when the chat template mentions the variable ``name``."""
    return re.search(rf"\b{re.escape(name)}\b", template_source(tokenizer)) is not None


def template_kwargs(tokenizer: Any, options: ChatTemplateOptions) -> dict[str, Any]:
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


def _as_ids(value: Any) -> list[int]:
    if isinstance(value, dict) or hasattr(value, "keys"):
        value = value["input_ids"]
    return [int(t) for t in value]


def render_text(
    tokenizer: Any,
    messages: Sequence[dict[str, Any]],
    options: ChatTemplateOptions,
    *,
    add_generation_prompt: bool,
) -> str:
    return tokenizer.apply_chat_template(
        [dict(m) for m in messages],
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **template_kwargs(tokenizer, options),
    )


def encode_rendered(tokenizer: Any, text: str) -> list[int]:
    """Token ids of rendered template text (the template emits BOS itself)."""
    return _as_ids(tokenizer(text, add_special_tokens=False))


def render_prompt_ids(
    tokenizer: Any,
    messages: Sequence[dict[str, Any]],
    options: ChatTemplateOptions,
) -> list[int]:
    """Prompt ids ending with the generation prompt (the RL rollout prompt)."""
    return encode_rendered(
        tokenizer, render_text(tokenizer, messages, options, add_generation_prompt=True)
    )


def generation_prompt_text(tokenizer: Any, options: ChatTemplateOptions) -> str:
    """Text the template appends for ``add_generation_prompt`` after a user turn."""
    probe = [{"role": "user", "content": "x"}]
    with_prompt = render_text(tokenizer, probe, options, add_generation_prompt=True)
    without = render_text(tokenizer, probe, options, add_generation_prompt=False)
    if not with_prompt.startswith(without):
        raise ValueError("generation prompt is not a suffix of the rendered turn")
    return with_prompt[len(without):]


@dataclass
class RenderedChat:
    """One tokenized conversation with its assistant-token mask."""

    input_ids: list[int]
    assistant_mask: list[bool]

    @property
    def num_assistant_tokens(self) -> int:
        return sum(self.assistant_mask)


def _spans(mask: Sequence[bool]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans


def _generation_mask(tokenizer: Any, messages, options) -> RenderedChat:
    enc = tokenizer.apply_chat_template(
        [dict(m) for m in messages],
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        **template_kwargs(tokenizer, options),
    )
    ids = [int(t) for t in enc["input_ids"]]
    mask = [bool(v) for v in enc["assistant_masks"]]
    if len(ids) != len(mask):
        raise ValueError("assistant mask length differs from the token count")
    return RenderedChat(ids, mask)


def _prefix_mask(tokenizer: Any, messages, options) -> RenderedChat:
    full = encode_rendered(
        tokenizer, render_text(tokenizer, messages, options, add_generation_prompt=False)
    )
    mask = [False] * len(full)
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        start = encode_rendered(
            tokenizer,
            render_text(tokenizer, messages[:index], options, add_generation_prompt=True),
        )
        end = encode_rendered(
            tokenizer,
            render_text(tokenizer, messages[: index + 1], options, add_generation_prompt=False),
        )
        if full[: len(start)] != start or full[: len(end)] != end:
            raise ValueError(
                f"assistant turn {index} is not a token prefix of the conversation; "
                "use mask_mode='generation' with a template that has generation blocks"
            )
        for position in range(len(start), len(end)):
            mask[position] = True
    return RenderedChat(full, mask)


def render_chat(
    tokenizer: Any,
    messages: Sequence[dict[str, Any]],
    options: ChatTemplateOptions,
    *,
    mask_mode: str = "auto",
    train_on_generation_prompt: bool = False,
) -> RenderedChat:
    """Tokenize a conversation and mark the tokens an SFT loss should train."""
    if mask_mode not in MASK_MODES:
        raise ValueError(f"mask_mode {mask_mode!r} must be one of {MASK_MODES}")
    if mask_mode == "auto":
        source = template_source(tokenizer)
        mask_mode = (
            "generation" if re.search(r"{%-?\s*generation\s*-?%}", source) else "prefix"
        )
    if mask_mode == "generation":
        rendered = _generation_mask(tokenizer, messages, options)
        if not train_on_generation_prompt:
            prompt = encode_rendered(tokenizer, generation_prompt_text(tokenizer, options))
            for start, end in _spans(rendered.assistant_mask):
                if prompt and rendered.input_ids[start : start + len(prompt)] == prompt:
                    for position in range(start, min(start + len(prompt), end)):
                        rendered.assistant_mask[position] = False
        return rendered
    rendered = _prefix_mask(tokenizer, messages, options)
    if train_on_generation_prompt:
        prompt = encode_rendered(tokenizer, generation_prompt_text(tokenizer, options))
        for start, _ in _spans(rendered.assistant_mask):
            begin = start - len(prompt)
            if begin >= 0 and rendered.input_ids[begin:start] == prompt:
                for position in range(begin, start):
                    rendered.assistant_mask[position] = True
    return rendered


def stop_token_ids(model_dir: str | Path | None, tokenizer: Any) -> list[int]:
    """End-of-turn ids: ``generation_config.json`` eos ids, else the tokenizer eos."""
    if model_dir is not None:
        path = Path(model_dir) / "generation_config.json"
        if path.is_file():
            eos = json.loads(path.read_text()).get("eos_token_id")
            if isinstance(eos, int):
                return [eos]
            if isinstance(eos, list) and eos:
                return [int(t) for t in eos]
    if getattr(tokenizer, "eos_token_id", None) is None:
        raise ValueError("no eos token id in generation_config.json or the tokenizer")
    return [int(tokenizer.eos_token_id)]


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
            header = header[len(markers.role):].strip()
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
    return ParsedResponse(messages=messages, final_text=final.strip(), structured=True)


def decode_final_answer(
    tokenizer: Any,
    token_ids: Sequence[int],
    markers: ResponseMarkers = ResponseMarkers(),
) -> ParsedResponse:
    """Decode sampled ids with special tokens kept and split the messages."""
    text = tokenizer.decode([int(t) for t in token_ids], skip_special_tokens=False)
    specials = list(getattr(tokenizer, "all_special_tokens", []) or [])
    return split_response(text, special_tokens=specials, markers=markers)
