# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Optimizer options for full-parameter post-training.

``PostTrainingOptimizersContainer`` is TorchTitan's ``OptimizersContainer``
with two extra optimizer names backed by torchao (BSD-3-Clause, already in the
trainer image):

* ``"AdamW8bit"``: AdamW whose two moment estimates are stored block-wise
  quantized to 8 bits (dynamic quantization map, block size 256; Dettmers et
  al., arXiv:2110.02861). Tensors smaller than 4096 elements keep full-size
  states. With bf16 parameters, ``bf16_stochastic_round`` rounds updates
  stochastically.
* ``"AdamWBf16"``: AdamW with states in the parameter dtype and the update
  computed in float32, then stochastically rounded into bfloat16 parameters
  (pair it with ``training.dtype="bfloat16"``).

torchao compiles its per-parameter update with the default Inductor backend.
``lowbit_compile_backend`` selects ``""`` (eager; works on every device),
``"tpu"`` (the TorchTPU dynamo backend) or ``"inductor"``. The update math is
torchao's ``single_param_adam`` unchanged.

Memory per parameter for Adam-type training (weights + gradients + two
moments): float32 everything 16 bytes; bf16 weights / gradients / moments
8 bytes; bf16 weights and gradients with 8-bit moments 6 bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch.optim import Optimizer

from torchtitan.components.optimizer.optimizer import OptimizersContainer, ParamGroupConfig

LOWBIT_OPTIMIZERS = ("AdamW8bit", "AdamWBf16")
COMPILE_BACKENDS = ("", "tpu", "inductor")


def _lowbit_class(base_cls: type, backend: str) -> type:
    """``base_cls`` with its step running ``single_param_adam`` on ``backend``."""
    from torchao.optim.adam import single_param_adam

    update: Callable[..., Any] = (
        single_param_adam
        if not backend
        else torch.compile(single_param_adam, backend=backend, fullgraph=True, dynamic=False)
    )

    class LowBitAdamW(base_cls):  # type: ignore[valid-type, misc]
        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for group in self.param_groups:
                lr = group["lr"]
                if not isinstance(lr, torch.Tensor):
                    raise RuntimeError("lr must stay a tensor; update it with fill_()")
                beta1, beta2 = group["betas"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if p.grad.is_sparse:
                        raise RuntimeError("sparse gradients are not supported")
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = torch.tensor(0.0)
                        state["exp_avg"] = self._new_buffer(p, True)
                        state["exp_avg_sq"] = self._new_buffer(p, False)
                        if group["amsgrad"]:
                            state["max_exp_avg_sq"] = self._new_buffer(p, False)
                    state["step"] += 1
                    update(
                        p.detach(),
                        p.grad,
                        state["step"],
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state.get("max_exp_avg_sq", None),
                        lr,
                        beta1,
                        beta2,
                        group["weight_decay"],
                        group["eps"],
                        self.is_adamw,
                        self.bf16_stochastic_round and p.dtype is torch.bfloat16,
                    )
            return loss

    LowBitAdamW.__name__ = f"{base_cls.__name__}Step"
    LowBitAdamW.__qualname__ = LowBitAdamW.__name__
    return LowBitAdamW


def lowbit_factory(name: str, backend: str = "") -> Callable[..., Optimizer]:
    if backend not in COMPILE_BACKENDS:
        raise ValueError(f"lowbit_compile_backend must be one of {COMPILE_BACKENDS}")
    from torchao.optim import AdamW8bit, _AdamW

    if name == "AdamW8bit":
        cls = _lowbit_class(AdamW8bit, backend)

        def build(params, **kwargs):
            kwargs.pop("fused", None)
            kwargs.pop("foreach", None)
            return cls(params, **kwargs)

        return build
    if name == "AdamWBf16":
        cls = _lowbit_class(_AdamW, backend)

        def build_bf16(params, **kwargs):
            kwargs.pop("fused", None)
            kwargs.pop("foreach", None)
            kwargs.setdefault("bf16_stochastic_round", True)
            return cls(params, **kwargs)

        return build_bf16
    raise NotImplementedError(f"unknown low-bit optimizer {name}")


class PostTrainingOptimizersContainer(OptimizersContainer):
    """``OptimizersContainer`` that also resolves the torchao AdamW variants."""

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        lowbit_compile_backend: str = ""
        """"" eager, "tpu" or "inductor" for the AdamW8bit / AdamWBf16 update."""

    def __init__(self, config: Config, *, model_parts: list[torch.nn.Module]) -> None:
        self._lowbit_backend = config.lowbit_compile_backend
        if self._lowbit_backend not in COMPILE_BACKENDS:
            raise ValueError(f"lowbit_compile_backend must be one of {COMPILE_BACKENDS}")
        super().__init__(config, model_parts=model_parts)

    def _resolve_optimizer_factory(self, name: str) -> Callable[..., Optimizer]:  # type: ignore[override]
        if name in LOWBIT_OPTIMIZERS:
            return lowbit_factory(name, self._lowbit_backend)
        return OptimizersContainer._resolve_optimizer_factory(name)


def adamw_config(
    *,
    lr: float,
    name: str = "AdamW",
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    implementation: str = "foreach",
    lowbit_compile_backend: str = "",
) -> OptimizersContainer.Config:
    """One catch-all AdamW group: ``name`` is AdamW, AdamW8bit or AdamWBf16."""
    if name not in ("AdamW",) + LOWBIT_OPTIMIZERS:
        raise ValueError(f"optimizer name {name!r} must be AdamW or one of {LOWBIT_OPTIMIZERS}")
    group = ParamGroupConfig(
        pattern=r".*",
        optimizer_name=name,
        optimizer_kwargs={"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay},
    )
    if name == "AdamW":
        return OptimizersContainer.Config(param_groups=[group], implementation=implementation)
    return PostTrainingOptimizersContainer.Config(
        param_groups=[group],
        implementation="for-loop",
        lowbit_compile_backend=lowbit_compile_backend,
    )
