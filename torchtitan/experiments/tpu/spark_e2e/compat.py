# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Compatibility shims for running Muse Spark through the TorchTitan trainer. ASCII only.

* The Muse Spark chat template uses Transformers' ``{% generation %}`` block
  (assistant-token tracking). TorchTitan's ``HuggingFaceTokenizer`` compiles
  the template with a plain Jinja environment and fails on the unknown tag
  while the trainer builds its tokenizer, although the post-training data
  loaders render prompts through the Hugging Face tokenizer. ``install()``
  teaches that environment the tag (rendered as a plain block, as in
  Transformers when no assistant mask is requested). It changes nothing when
  the tokenizer already accepts the template.
"""

from __future__ import annotations

import jinja2
import jinja2.ext
import jinja2.sandbox
from jinja2.ext import Extension

_INSTALLED = False


class GenerationBlock(Extension):
    """``{% generation %} ... {% endgeneration %}`` -> the block's body."""

    tags = {"generation"}

    def parse(self, parser):  # noqa: D102
        next(parser.stream)
        return parser.parse_statements(("name:endgeneration",), drop_needle=True)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from torchtitan.components import tokenizer as tt_tokenizer

    cls = tt_tokenizer.HuggingFaceTokenizer
    original = cls.set_chat_template

    def set_chat_template(self, template: str) -> None:
        try:
            original(self, template)
            return
        except jinja2.exceptions.TemplateSyntaxError:
            pass
        import json

        def raise_exception(message):
            raise jinja2.exceptions.TemplateError(message)

        def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
            return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                              separators=separators, sort_keys=sort_keys)

        def strftime_now(fmt):
            from datetime import datetime

            return datetime.now().strftime(fmt)

        env = jinja2.sandbox.ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True,
            extensions=[jinja2.ext.loopcontrols, GenerationBlock],
        )
        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = strftime_now
        env.filters["tojson"] = tojson
        self._chat_template = env.from_string(template)

    cls.set_chat_template = set_chat_template
    _INSTALLED = True
