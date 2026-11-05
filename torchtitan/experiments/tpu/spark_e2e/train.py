# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Entry point for Muse Spark LoRA SFT (TPU or CPU). ASCII only.

Runs the TPU trainer startup with ``SparkSFTTrainer`` (``lora_publish``):

    torchrun ... -m torchtitan.experiments.tpu.spark_e2e.train \\
        --module torchtitan.experiments.tpu.spark_e2e.recipes --config s1_sft_trunc2 ...
"""

from __future__ import annotations

from typing import Any

from torchtitan.experiments.tpu import gmain
from torchtitan.experiments.tpu import train as native_train

from . import compat
from .lora_publish import SparkSFTTrainer

compat.install()


def start_trainer(train_config: Any) -> None:
    original = native_train.TPUTrainer
    native_train.TPUTrainer = SparkSFTTrainer  # type: ignore[assignment]
    try:
        native_train.start_trainer(train_config)
    finally:
        native_train.TPUTrainer = original  # type: ignore[assignment]


if __name__ == "__main__":
    gmain.handle_main(start_trainer)
