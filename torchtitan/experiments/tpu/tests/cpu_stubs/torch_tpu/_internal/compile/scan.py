# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU import stand-in for ``torch_tpu._internal.compile.scan``.

``torchtitan.experiments.tpu.scan_backward_patch`` annotates a scan's step
body with this key so that TorchTPU's compiler runs the loop in reverse. Only
TorchTPU honours it; nothing on CPU reads it.
"""

SCAN_REVERSE_ANNOTATION = "torch_tpu_scan_reverse"
