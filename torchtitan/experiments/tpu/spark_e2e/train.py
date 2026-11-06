# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Program behind the LoRA SFT stages of ``spark_e2e``, on TPU or on CPU. ASCII only.

Upstream's TPU launcher does all the work. The only thing that differs is the
trainer class it constructs: ``lora_publish.SparkSFTTrainer``, swapped in with
the ``tpu_trainer_class`` helper of ``post_training.sft_train``. Importing this
module also installs the chat-template shim from ``compat``. The stage recipes
are in ``spark_e2e/recipes.py``, and ``muse_spark/docs/REPRODUCE.md`` lists
complete commands.
"""

from __future__ import annotations

from torchtitan.experiments.tpu import gmain
from torchtitan.experiments.tpu.post_training.sft_train import tpu_trainer_class
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig
from torchtitan.experiments.tpu.train import start_trainer as launch_tpu_training

from . import compat
from .lora_publish import SparkSFTTrainer

compat.install()


def start_trainer(train_config: TPUTrainerConfig) -> None:
    """Upstream's TPU training launch, building ``SparkSFTTrainer`` as the trainer."""
    with tpu_trainer_class(SparkSFTTrainer):
        launch_tpu_training(train_config)


if __name__ == "__main__":
    gmain.handle_main(start_trainer)
