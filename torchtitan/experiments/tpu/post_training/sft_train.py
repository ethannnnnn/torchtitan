# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Entry point for chat SFT on TPU.

Runs the upstream TPU trainer startup with ``SFTTPUTrainer``:

* ``TPUTrainer.__init__`` replaces any configured loss with plain
  cross-entropy (or the Pallas fused kernel when selected). After
  ``Trainer.__init__`` has already wired a configured ``ChunkedLossWrapper``
  (``model._skip_lm_head = True``), that replacement would feed hidden states
  to a logits loss, so ``SFTTPUTrainer`` restores the configured chunked loss
  and its lm_head binding. The loss is the summed cross-entropy of the
  supervised tokens divided by the step's global supervised-token count.
* The accumulation count per optimizer step comes from the dataloader
  (``ChatSFTDataLoader.Config.microbatches_per_step``).
* Each step logs (and rank 0 writes to ``<dump_folder>/sft-records-rank0.json``)
  the packed documents, tokens, supervised tokens and this rank's loss share;
  the trainable / frozen parameter counts are logged once (LoRA runs).
* LoRA runs save only the adapter (``sft.lora_save="adapter"``, default): at
  every checkpoint step TorchTitan's DCP save writes the adapter-only resume
  state (adapter tensors, their optimizer state, the LR schedule, the
  dataloader position and the step counter; never the frozen base) to
  ``<dump_folder>/checkpoint/step-N`` (``checkpointing/lora_checkpoint.py``),
  then all ranks gather the adapter tensors (a collective over the sorted
  adapter keys) and rank 0 writes them in the PEFT layout, named through the
  model's state-dict adapter, to ``<dump_folder>/lora_adapter/step-N``. Every
  start reads the base from the Hugging Face checkpoint and then restores the
  newest complete ``step-N``, so relaunching with the same dump folder resumes
  a killed run. ``"peft"`` writes only the PEFT adapter (no resume state),
  ``"full"`` keeps TorchTitan's DCP save of the whole model, ``"none"`` saves
  nothing.
* On TPU every rank leaves each checkpoint save together
  (``checkpointing/flush.py``; ``SFT_CHECKPOINT_FLUSH=1`` forces it on any
  device, ``0`` turns it off).
* Held-out loss (``sft.val_data_path``, e.g. the dataset's ``sft/val.jsonl``):
  before the first step, every ``sft.val_every`` steps and after the last
  step, a forward-only pass over every held-out record (packed like the
  training data, each data-parallel rank a strided share) gives the summed
  cross-entropy of the assistant tokens divided by their count. It is logged
  as ``validate step: N loss: X`` and ``[sft] validation ...`` and written by
  rank 0 to ``<dump_folder>/sft-val-rank0.json``.
* Resume checks: with ``SFT_RECORDS_ALL_RANKS=1`` every rank appends one JSON
  line per step to ``<dump_folder>/sft-records/rank-NNNNN.jsonl`` (its loss
  share, the gradient norm, the learning rate, and a SHA-256 fingerprint plus
  the document ids of every microbatch it trained on), so two runs can be
  compared batch by batch (``SFT_RECORDS_DIR`` moves them, e.g. to local disk
  when the dump folder is an object-store mount). ``SFT_FAULT_KILL_STEP=N``
  sends SIGKILL to every rank inside the forward/backward of step N (after
  that step's batches were drawn): a hard crash for fault-tolerance tests.

Launch with torchrun, module torchtitan.experiments.tpu.post_training.sft_train,
``--module torchtitan.experiments.tpu.post_training.config_registry`` and an
SFT recipe; see README.md.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import signal
import time
from pathlib import Path
from typing import Any

import torch

import torch.distributed as dist
from torch.distributed.tensor import DTensor

from torchtitan.components.checkpointer.base import ModelWrapper
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.tpu import gmain
from torchtitan.experiments.tpu import train as native_train
from torchtitan.experiments.tpu.train import TPUTrainer

from . import lora as lora_mod
from .lora import lora_parameter_report
from .sft_config import LORA_SAVE_MODES

logger = logging.getLogger(__name__)


class SFTTPUTrainer(TPUTrainer):
  """TPU trainer for supervised fine-tuning with a chunked loss."""

  def __init__(self, job_config: Any) -> None:
    super().__init__(job_config)
    config = self.config
    if isinstance(config.loss, ChunkedLossWrapper.Config):
      rebuilt = config.loss.build(compile_config=config.compile)
      assert len(self.model_parts) == 1, "SFT path expects one model part"
      lm_head = getattr(self.model_parts[0], "lm_head", None)
      assert lm_head is not None, "ChunkedLossWrapper needs model.lm_head"
      rebuilt.set_lm_head(lm_head)
      self.model_parts[0]._skip_lm_head = True
      self.loss_fn = rebuilt
      logger.info(
          "[sft] loss restored to ChunkedLossWrapper(%d chunks)",
          rebuilt.num_chunks,
      )
    self.parameter_report = lora_parameter_report(self.model_parts[0])
    logger.info("[sft] parameters: %s", self.parameter_report)
    self.sft_records: list[dict[str, Any]] = []
    self._step_loss: torch.Tensor | None = None
    self._rank = int(os.environ.get("RANK", "0"))
    self.lora_exports: list[str] = []
    self._install_lora_save(config)
    # TorchTPU: every rank leaves each checkpoint save together, so a crash
    # right after a save cannot strand ranks 1..N-1 in DCP's last collective
    # (checkpointing/flush.py). SFT_CHECKPOINT_FLUSH=1 forces it on any
    # device, 0 turns it off.
    flush = os.environ.get("SFT_CHECKPOINT_FLUSH", "")
    if flush == "1" or (
        flush != "0" and getattr(self.device, "type", "") == "tpu"
    ):
      try:
        from torchtitan.experiments.tpu.checkpointing.flush import flush_after_saves
      except ImportError as exc:
        raise ImportError(
            "every rank leaves a checkpoint save together through "
            "torchtitan.experiments.tpu.checkpointing.flush; "
            "SFT_CHECKPOINT_FLUSH=0 turns this off"
        ) from exc

      flush_after_saves(self.checkpointer, self.device)
    self._last_grad_norm: torch.Tensor | None = None
    self._records_all_ranks = (
        os.environ.get("SFT_RECORDS_ALL_RANKS", "0") == "1"
    )
    self._fault_kill_step = int(os.environ.get("SFT_FAULT_KILL_STEP", "0") or 0)
    if self._fault_kill_step and self._rank == 0:
      logger.warning(
          "[sft] fault injection: every rank is killed (SIGKILL) at step %d",
          self._fault_kill_step,
      )
    self._setup_validation(config)

  # -- LoRA adapter saves ----------------------------------------------
  def _lora_modules(self) -> list[tuple[str, torch.nn.Module]]:
    return [
        (name, module)
        for name, module in self.model_parts[0].named_modules()
        if isinstance(getattr(module, "lora_a", None), torch.nn.Module)
        and isinstance(getattr(module, "lora_b", None), torch.nn.Module)
    ]

  def _install_lora_save(self, config: Any) -> None:
    sft = getattr(config, "sft", None)
    mode = getattr(sft, "lora_save", "full")
    if mode not in LORA_SAVE_MODES:
      raise ValueError(
          f"sft.lora_save={mode!r}; expected one of {LORA_SAVE_MODES}"
      )
    modules = self._lora_modules()
    if not modules or mode == "full":
      return
    _, first = modules[0]
    rank = int(first.lora_a.weight.shape[0])
    self._lora_options = lora_mod.LoRAOptions(
        rank=rank, alpha=float(first._lora_scaling) * rank
    )
    checkpointer = self.checkpointer
    resumable = mode == "adapter" and getattr(checkpointer, "enable", False)
    if resumable:
      try:
        from torchtitan.experiments.tpu.checkpointing.lora_checkpoint import (
            enable_adapter_only_checkpoints,
        )
      except ImportError as exc:
        raise ImportError(
            "sft.lora_save='adapter' writes the adapter-only resume state through "
            "torchtitan.experiments.tpu.checkpointing.lora_checkpoint; without it use "
            "sft.lora_save='peft' (PEFT adapter only)"
        ) from exc

      # Adapter-only DCP resume state; loads read the HF base first.
      enable_adapter_only_checkpoints(checkpointer, self.model_parts)
    dcp_save = checkpointer.save

    def _save(curr_step: int, last_step: bool = False) -> bool:
      if mode == "none" or not checkpointer._should_save(curr_step, last_step):
        return False
      if resumable:
        dcp_save(curr_step, last_step=last_step)
      self.export_lora_adapter(curr_step)
      return True

    checkpointer.save = _save
    logger.info(
        "[sft] LoRA run: checkpoint saves write the adapter only (sft.lora_save=%s%s)",
        mode,
        ", with adapter-only resume state" if resumable else "",
    )

  def export_lora_adapter(self, step: int) -> str | None:
    """Gather the adapter on every rank; rank 0 writes the PEFT files."""
    state = ModelWrapper(self.model_parts).state_dict()
    gathered: dict[str, torch.Tensor] = {}
    for key in sorted(k for k in state if lora_mod.is_lora_key(k)):
      value = state[key]
      full = value.full_tensor() if isinstance(value, DTensor) else value
      if self._rank == 0:
        gathered[key] = full.detach().to("cpu", torch.float32)
      del full
    out = None
    if self._rank == 0:
      tensors = lora_mod.peft_adapter_state_dict(
          gathered, adapter=self.checkpointer.sd_adapter
      )
      target = Path(self.config.dump_folder) / "lora_adapter" / f"step-{step}"
      out = str(
          lora_mod.save_peft_adapter(
              target,
              tensors,
              options=self._lora_options,
              base_model_name_or_path=str(self.config.hf_assets_path),
          )
      )
      self.lora_exports.append(out)
      logger.info(
          "[sft] step=%d LoRA adapter (%d tensors) -> %s",
          step,
          len(tensors),
          out,
      )
    if dist.is_initialized():
      dist.barrier()
    return out

  # -- held-out loss ------------------------------------------------------
  def _setup_validation(self, config: Any) -> None:
    sft = getattr(config, "sft", None)
    self._val_path = str(getattr(sft, "val_data_path", "") or "")
    self._val_every = int(getattr(sft, "val_every", 0) or 0)
    self._val_cap = int(getattr(sft, "val_max_microbatches", 0) or 0)
    self._val_batches: list[dict[str, Any]] | None = None
    self._val_info: dict[str, Any] = {}
    self._val_started = False
    self.val_records: list[dict[str, Any]] = []
    if self._val_path:
      from .sft_data import ChatSFTDataLoader

      if not isinstance(self.dataloader, ChatSFTDataLoader):
        raise ValueError(
            "sft.val_data_path needs the ChatSFTDataLoader training data"
        )
      logger.info(
          "[sft] validation on %s every %s steps",
          self._val_path,
          self._val_every or "(first and last only)",
      )

  def _dp_coords(self) -> tuple[int, int]:
    if self.parallel_dims.dp_enabled:
      mesh = self.parallel_dims.get_mesh("batch")
      return mesh.size(), mesh.get_local_rank()
    return 1, 0

  def _build_val_microbatches(
      self,
  ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """This rank's packed held-out microbatches, padded to a common count.

    Every rank tokenizes all held-out records and packs every rank's
    strided share (records ``r, r + n, ...``), so the per-rank microbatch
    counts are known on every rank without a collective. Ranks with fewer
    microbatches append all-padding ones (no supervised token), so every
    rank runs the same forwards and collectives.
    """
    from .sft_data import ChatSFTDataLoader, build_microbatch, pack_rows

    dp_size, dp_rank = self._dp_coords()
    cfg = dataclasses.replace(
        self.config.dataloader,
        data_path=self._val_path,
        max_examples=0,
        shuffle_seed=None,
    )
    loader = ChatSFTDataLoader(
        cfg,
        dp_world_size=1,
        dp_rank=0,
        tokenizer=self.tokenizer,
        max_context_length=self.config.training.max_context_length,
        num_tokens_per_batch=self.config.training.num_tokens_per_microbatch_per_dp_rank,
    )
    rows, width = loader._rows, loader._width
    num_records = len(loader._conversations)
    docs = [loader.tokenize(i) for i in range(num_records)]
    counts, mine = [], []
    for rank in range(dp_size):
      pending = [d for d in docs[rank::dp_size] if d is not None]
      packed = []
      while pending:
        placed, pending = pack_rows(
            pending,
            rows=rows,
            width=width,
            max_docs_per_row=cfg.max_documents_per_row,
        )
        packed.append(placed)
      counts.append(len(packed))
      if rank == dp_rank:
        mine = packed
    steps = max(counts) if counts else 0
    if self._val_cap:
      steps = min(steps, self._val_cap)
    empty = [[] for _ in range(rows)]
    batches = [
        build_microbatch(p, width=width, pad_id=loader._pad_id)
        for p in mine[:steps]
    ]
    batches += [
        build_microbatch(empty, width=width, pad_id=loader._pad_id)
        for _ in range(steps - len(batches))
    ]
    kept = [d for d in docs if d is not None]
    info = {
        "path": self._val_path,
        "records": num_records,
        "documents": len(kept),
        "dropped": num_records - len(kept),
        "supervised_tokens": int(sum(d.num_valid for d in kept)),
        "tokens": int(sum(d.length for d in kept)),
        "microbatches_per_rank": steps,
        "max_microbatches_needed": max(counts) if counts else 0,
        "complete": not self._val_cap or max(counts or [0]) <= self._val_cap,
    }
    return batches, info

  @torch.no_grad()
  def run_validation(self, step: int) -> dict[str, Any] | None:
    """Held-out loss of the current weights (collective; every rank calls it)."""
    if not self._val_path:
      return None
    from torchtitan.experiments.tpu.unflatten_shim import set_unflatten_seq_len

    t0 = time.perf_counter()
    if self._val_batches is None:
      self._val_batches, self._val_info = self._build_val_microbatches()
      logger.info("[sft] validation set: %s", self._val_info)
    model = self.model_parts[0]
    was_training = model.training
    model.eval()
    seq_len = self.config.training.max_context_length
    loss_sum = torch.zeros((), dtype=torch.float32, device=self.device)
    valid = 0
    try:
      with set_unflatten_seq_len(seq_len):
        for batch in self._val_batches:
          valid += int(batch["num_valid_tokens"])
          input_dict = {
              k: v.to(self.device)
              for k, v in batch.items()
              if isinstance(v, torch.Tensor)
          }
          inputs, labels, extra_kwargs = model.preprocess_inputs(
              input_dict,
              parallel_dims=self.parallel_dims,
              parallelism=self.config.parallelism,
              max_num_documents=self.dataloader.max_num_documents,
              max_context_length=seq_len,
          )
          with self.train_context():
            pred = model(inputs, **extra_kwargs)
            loss_kwargs = {}
            if "positions" in extra_kwargs:
              loss_kwargs["positions"] = extra_kwargs["positions"]
            # No global token count: the loss comes back as a sum.
            loss, _ = self.loss_fn(pred, labels, None, **loss_kwargs)
            del pred
          loss_sum = loss_sum + loss.detach().float()
    finally:
      if was_training:
        model.train()
    totals = torch.stack(
        [loss_sum, torch.tensor(float(valid), device=self.device)]
    )
    if self.parallel_dims.dp_enabled:
      totals = dist_utils.dist_sum_tensor(
          totals, self.parallel_dims.get_mesh("batch")
      )
    total_loss, total_valid = (float(x) for x in totals.cpu().tolist())
    value = total_loss / max(total_valid, 1.0)
    record = {
        "step": step,
        "loss": value,
        "supervised_tokens": int(total_valid),
        "microbatches_per_rank": len(self._val_batches),
        "seconds": round(time.perf_counter() - t0, 3),
    }
    self.val_records.append(record)
    logger.info(
        "[sft] validation step=%d loss=%.5f supervised_tokens=%d seconds=%.1f",
        step,
        value,
        int(total_valid),
        record["seconds"],
    )
    metrics = getattr(self, "metrics_processor", None)
    if metrics is not None and hasattr(metrics, "log_validation"):
      # Also resets the step timer, so the next step's time excludes this pass.
      metrics.log_validation(loss=value, step=step)
    if self._rank == 0 and self.config.dump_folder:
      path = Path(self.config.dump_folder) / "sft-val-rank0.json"
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_text(
          json.dumps(
              {"set": self._val_info, "evals": self.val_records}, indent=2
          )
      )
    return record

  def _maybe_validate(self, *, before_step: bool) -> None:
    if not self._val_path:
      return
    if before_step:
      if not self._val_started:
        self._val_started = True
        self.run_validation(self.step - 1)
      return
    last = self.step == self.config.training.steps
    if last or (self._val_every and self.step % self._val_every == 0):
      self.run_validation(self.step)

  def _clip_grad_norm(self) -> torch.Tensor:
    grad_norm = super()._clip_grad_norm()
    self._last_grad_norm = grad_norm
    return grad_norm

  def forward_backward_step(self, *, input_dict, global_valid_tokens):
    if self._fault_kill_step and self.step == self._fault_kill_step:
      # Test hook (SFT_FAULT_KILL_STEP): a hard crash on every rank after
      # this step's batches were drawn and before any state is saved.
      logger.warning(
          "[sft] fault injection: SIGKILL at step %d (rank %d)",
          self.step,
          self._rank,
      )
      for handler in logging.getLogger().handlers:
        handler.flush()
      os.kill(os.getpid(), signal.SIGKILL)
    loss = super().forward_backward_step(
        input_dict=input_dict, global_valid_tokens=global_valid_tokens
    )
    detached = loss.detach()
    self._step_loss = (
        detached if self._step_loss is None else self._step_loss + detached
    )
    return loss

  def train_step(self, data_iterator):
    self._maybe_validate(before_step=True)
    if hasattr(self.dataloader, "microbatches_for_step"):
      self.gradient_accumulation_steps = self.dataloader.microbatches_for_step()
    before = getattr(self.dataloader, "stats", None)
    snapshot = (
        (before.documents, before.tokens, before.valid_tokens)
        if before
        else (0, 0, 0)
    )
    batch_log = getattr(self.dataloader, "batch_log", None)
    log_start = len(batch_log) if batch_log is not None else 0
    lr_used = (
        self.lr_schedulers.get_metrics() if self._records_all_ranks else {}
    )
    self._step_loss = None
    self._last_grad_norm = None
    result = super().train_step(data_iterator)
    stats = getattr(self.dataloader, "stats", None)
    record: dict[str, Any] = {"step": self.step}
    if self._step_loss is not None:
      # This rank's share of the step's global mean loss (the shares of
      # all data-parallel ranks sum to it). The trainer already reads the
      # loss back every log step, so this adds no device sync there.
      record["loss_local"] = float(self._step_loss)
    if stats is not None:
      record.update(
          epoch=stats.epoch,
          documents=stats.documents - snapshot[0],
          tokens=stats.tokens - snapshot[1],
          valid_tokens_local=stats.valid_tokens - snapshot[2],
          dropped_total=stats.dropped,
          truncated_total=stats.truncated,
      )
    self.sft_records.append(record)
    logger.info("[sft] step=%d %s", self.step, record)
    if self._records_all_ranks:
      self._append_rank_record(
          record,
          batch_log[log_start:] if batch_log is not None else [],
          lr_used,
      )
    if batch_log is not None:
      del batch_log[:]
    if self._rank == 0 and self.config.dump_folder:
      path = Path(self.config.dump_folder) / "sft-records-rank0.json"
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_text(
          json.dumps(
              {"parameters": self.parameter_report, "steps": self.sft_records},
              indent=2,
          )
      )
    self._maybe_validate(before_step=False)
    return result

  def _append_rank_record(
      self,
      record: dict[str, Any],
      batches: list[dict[str, Any]],
      lr_used: dict[str, float],
  ) -> None:
    line = dict(record, rank=self._rank, batches=batches, lr=lr_used)
    if self._last_grad_norm is not None:
      line["grad_norm"] = float(self._last_grad_norm.item())
    root = Path(
        os.environ.get("SFT_RECORDS_DIR")
        or Path(self.config.dump_folder or ".") / "sft-records"
    )
    root.mkdir(parents=True, exist_ok=True)
    with open(
        root / f"rank-{self._rank:05d}.jsonl", "a", encoding="utf-8"
    ) as handle:
      handle.write(json.dumps(line, sort_keys=True) + "\n")


def start_trainer(train_config: Any) -> None:
  original = native_train.TPUTrainer
  native_train.TPUTrainer = SFTTPUTrainer  # type: ignore[assignment]
  try:
    native_train.start_trainer(train_config)
  finally:
    native_train.TPUTrainer = original  # type: ignore[assignment]


if __name__ == "__main__":
  gmain.handle_main(start_trainer)
