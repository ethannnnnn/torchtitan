# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Import-only stand-in for ``torch_tpu`` in the post-training CPU tests.

TorchTitan's TPU trainer modules import TorchTPU kernels and runtime helpers at
module scope. The CPU test runner puts the parent directory first on
``PYTHONPATH`` when TorchTPU is not installed, so that those imports succeed.
Nothing here registers a device or a backend: the tests run on CPU, and every
function that would need a TPU raises when it is called.
"""


def unavailable(name: str):
    """A function that raises: ``name`` needs TorchTPU."""

    def call(*args, **kwargs):
        raise RuntimeError(f"{name} needs torch_tpu; this is the CPU test stand-in")

    call.__name__ = name.rsplit(".", 1)[-1]
    return call
