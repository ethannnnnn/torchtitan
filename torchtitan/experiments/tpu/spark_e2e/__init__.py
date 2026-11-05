# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""End-to-end LoRA SFT of Muse Spark Open on TPU. ASCII only.

Wires the pieces of the Muse Spark TPU lane into one runnable SFT path:

* ``recipes``: LoRA SFT recipes per stage (CPU tiny checkpoint, the first two
  released layers at real dimensions, the full model) on top of
  ``post_training.config_registry``.
* ``lora_publish``: the SFT trainer those recipes run: adapter-only
  checkpoints that resume after a restart, PEFT adapter exports at every
  checkpoint and at the last step, and an optional initial adapter.
* ``peft_adapter``: the PEFT layout of the attention adapters (export and
  import).
* ``tpu_knobs``: TPU7x trainer settings and the fast Hugging Face reader.
* ``build_trunc``: truncated (first N layers) checkpoints for small slices.
* ``train``: the entry point
  (``-m torchtitan.experiments.tpu.spark_e2e.train``).

It builds on the Muse Spark model package (``muse_spark``: model, Hugging
Face mapping, TPU7x settings, expert placement), the chat SFT layer
(``post_training``) and the LoRA checkpoint helpers (``checkpointing``).
"""
