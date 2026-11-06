# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Chat supervised fine-tuning on TPU: ``SFTTPUTrainer`` and its entry point.

``start_trainer`` runs upstream's TPU launcher,
``torchtitan.experiments.tpu.train.start_trainer``, so that it constructs
``SFTTPUTrainer`` in place of ``TPUTrainer`` (``tpu_trainer_class``).

Loss
  A job whose loss is configured as ``ChunkedLossWrapper`` gets that loss back
  once ``TPUTrainer.__init__`` has run. Upstream ``Trainer`` builds the wrapper,
  hands it the model's ``lm_head`` and makes the model stop short of that
  projection; ``TPUTrainer`` then replaces the wrapper by cross-entropy over
  logits (or by its fused kernel), which would be fed hidden states. Each
  token's cross-entropy is summed over the supervised targets, and the sum is
  divided by how many supervised targets the optimizer step holds over all
  microbatches and data-parallel ranks.

Microbatches
  An optimizer step consumes ``ChatSFTDataLoader.Config.microbatches_per_step``
  microbatches per rank.

Records
  Every step logs its documents, tokens, supervised targets and the share of
  the loss computed on this rank. Rank 0 also writes them to
  ``<dump_folder>/sft-records-rank0.json``. Trainable and frozen parameter
  counts are reported once, at construction.

LoRA saves, ``sft.lora_save``
  ``"adapter"`` (default) writes two things at a checkpoint step: DCP resume
  state in ``<dump_folder>/checkpoint/step-N`` (step counter, data position,
  LR schedule, LoRA tensors and their optimizer state; the frozen base is
  never included) and a PEFT copy in ``<dump_folder>/lora_adapter/step-N``.
  ``"peft"`` writes the PEFT copy only, ``"full"`` uses TorchTitan's DCP save
  of the whole model, and ``"none"`` writes nothing. Whatever the mode, a start
  loads the frozen base from the released Hugging Face weights and then
  restores the newest complete ``step-N`` over it.

Checkpoint flush, ``SFT_CHECKPOINT_FLUSH``
  On TPU, all ranks finish each checkpoint save in lockstep
  (``checkpointing/flush.py``). ``1`` enables this on any device; ``0``
  disables it.

Held-out loss, ``sft.val_data_path`` and ``sft.val_every``
  Evaluated once ahead of step 1, at each multiple of ``val_every`` and once
  more after the final step, with forward passes only. Held-out records are
  packed like training data; with n data-parallel ranks, rank r evaluates
  records r, r + n, r + 2n and so on. The figure is the cross-entropy summed
  over assistant targets, divided by their number. It appears in upstream's
  ``validate step: N loss: X`` line and in a ``[sft] validation ...`` line,
  and rank 0 keeps the history in ``<dump_folder>/sft-val-rank0.json``.

Resume checks
  ``SFT_RECORDS_ALL_RANKS=1`` appends one line per step to each rank's
  ``<dump_folder>/sft-records/rank-NNNNN.jsonl``; ``SFT_RECORDS_DIR`` moves
  those files. ``SFT_FAULT_KILL_STEP=N`` makes every rank SIGKILL itself while
  running step N. After a restart, the two rank-0 JSON files describe the new
  process only, whereas the per-rank JSONL files keep accumulating.

Launch
  torchrun runs this module as its program, with ``--module`` naming this
  package's recipe registry (``config_registry``) and ``--config`` one of
  its recipes; README.md gives a complete command.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import os
import signal
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch

import torch.distributed as dist
from torch.distributed.tensor import DTensor

from torchtitan.components.checkpointer.base import ModelWrapper
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.config.configurable import Configurable
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.tpu import gmain
import torchtitan.experiments.tpu.train as tpu_entry
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig
from torchtitan.experiments.tpu.train import TPUTrainer

from . import lora as lora_mod
from .lora import lora_parameter_report
from .sft_config import validated_lora_save

logger = logging.getLogger(__name__)


@runtime_checkable
class MicrobatchesPerStep(Protocol):
  """A data loader that fixes how many microbatches form one optimizer step.

  ``SFTTPUTrainer.train_step`` asks before every step and accumulates that
  many microbatches.
  """

  def accumulation_steps(self) -> int:
    ...


@dataclasses.dataclass(frozen=True)
class _StepStart:
  """Loader counters, batch-log length and learning rates seen before a step.

  ``None`` marks what the loader does not have.
  """

  documents: int | None = None
  tokens: int | None = None
  valid_tokens: int | None = None
  log_length: int | None = None
  learning_rates: dict[str, float] = dataclasses.field(default_factory=dict)


class SFTTPUTrainer(TPUTrainer):
  """TPU trainer for supervised fine-tuning with a chunked loss."""

  def __init__(self, job_config: Configurable.Config):
    super().__init__(job_config)
    self._reinstall_chunked_loss(job_config)
    self.parameter_report = lora_parameter_report(self.model_parts[0])
    logger.info("[sft] parameters: %s", self.parameter_report)
    rank = int(os.getenv("RANK", "0"))
    self._rank = rank
    self.sft_records = []
    self._step_loss: torch.Tensor | None = None
    self.lora_exports: list[str] = []
    self._install_lora_save(job_config)
    # On TPU, all ranks finish each checkpoint save in lockstep; see
    # checkpointing/flush.py. It must wrap the save that the LoRA hook above
    # installed, hence its place after that hook.
    flush = os.environ.get("SFT_CHECKPOINT_FLUSH", "")
    if flush == "1" or (
        flush != "0" and getattr(self.device, "type", "") == "tpu"
    ):
      try:
        from torchtitan.experiments.tpu.checkpointing.flush import flush_after_saves
      except ImportError as exc:
        raise ImportError(
            "SFT_CHECKPOINT_FLUSH asks for the lockstep checkpoint exit, but "
            "torchtitan.experiments.tpu.checkpointing.flush failed to import; "
            "run with SFT_CHECKPOINT_FLUSH=0 to go without it"
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
    self._setup_validation(job_config)

  # -- loss -------------------------------------------------------------
  def _reinstall_chunked_loss(self, job_config: Any) -> None:
    """Give a job configured with ``ChunkedLossWrapper`` that loss back.

    ``TPUTrainer.__init__`` replaces it with a loss over logits; this builds
    it again the way upstream ``Trainer`` does, binds the single model part's
    ``lm_head`` to it and lets the model stop before that projection.
    """
    if not isinstance(job_config.loss, ChunkedLossWrapper.Config):
      return
    heads = [getattr(part, "lm_head", None) for part in self.model_parts]
    if len(heads) != 1 or heads[0] is None:
      raise ValueError(
          "a ChunkedLossWrapper loss needs exactly one model part with an "
          "lm_head; found {} part(s), lm_head present: {}".format(
              len(heads), [head is not None for head in heads]
          )
      )
    (part,), (head,) = self.model_parts, heads
    chunked = job_config.loss.build(compile_config=job_config.compile)
    chunked.set_lm_head(head)
    part._skip_lm_head = True
    self.loss_fn = chunked
    logger.info(
        "[sft] chunked loss restored: %d chunks through the model's lm_head",
        job_config.loss.num_chunks,
    )

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
    mode = validated_lora_save(mode)
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
    """Write this step's LoRA adapter as PEFT files. Collective.

    All ranks must call it with the same ``step``: DTensor adapter weights are
    materialized with ``full_tensor()``, key by key in sorted order. Rank 0
    keeps float32 CPU copies, renames them to PEFT keys through the
    checkpointer's state-dict adapter and writes
    ``<dump_folder>/lora_adapter/step-<step>``; the other ranks drop every
    materialized tensor before the next one. All ranks then meet at a
    barrier, so none returns before the files exist.

    Returns the directory on rank 0 and None elsewhere.
    """
    model_state = ModelWrapper(self.model_parts).state_dict()
    is_writer = self._rank == 0
    cpu_adapter: dict[str, torch.Tensor] = {}
    for key in sorted(filter(lora_mod.is_lora_key, model_state)):
      shard = model_state[key]
      whole = shard.full_tensor() if isinstance(shard, DTensor) else shard
      if is_writer:
        cpu_adapter[key] = whole.detach().to(
            device="cpu", dtype=torch.float32, copy=True
        )
      del whole
    written: str | None = None
    if is_writer:
      peft_tensors = lora_mod.peft_adapter_state_dict(
          cpu_adapter, adapter=getattr(self.checkpointer, "sd_adapter", None)
      )
      out_dir = Path(self.config.dump_folder) / "lora_adapter" / f"step-{step}"
      released_weights = str(self.config.hf_assets_path)
      lora_options = self._lora_options
      saved_dir = lora_mod.save_peft_adapter(
          out_dir,
          peft_tensors,
          options=lora_options,
          base_model_name_or_path=released_weights,
      )
      written = str(saved_dir)
      self.lora_exports.append(written)
      logger.info(
          "[sft] step %d: PEFT adapter, %d tensors, saved under %s",
          step,
          len(peft_tensors),
          written,
      )
    if dist.is_initialized():
      dist.barrier()
    return written

  # -- held-out loss ------------------------------------------------------
  def _setup_validation(self, config: Any) -> None:
    sft = getattr(config, "sft", None)
    self._val_path = str(getattr(sft, "val_data_path", "") or "")
    self._val_every = int(getattr(sft, "val_every", 0) or 0)
    self._val_cap = int(getattr(sft, "val_max_microbatches", 0) or 0)
    self._val_batches: list[dict[str, Any]] | None = None
    self._val_info: dict[str, Any] = {}
    self._val_started = False
    self.val_records = []
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
    evaluation = {
        "step": step,
        "loss": value,
        "supervised_tokens": int(total_valid),
        "microbatches_per_rank": len(self._val_batches),
        "seconds": round(time.perf_counter() - t0, 3),
    }
    self.val_records.append(evaluation)
    logger.info(
        "[sft] validation step=%d loss=%.5f supervised_tokens=%d seconds=%.1f",
        step,
        value,
        int(total_valid),
        evaluation["seconds"],
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
    return evaluation

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

  def forward_backward_step(
      self,
      *,
      input_dict: dict[str, Any] | list[dict[str, Any]],
      global_valid_tokens: torch.Tensor,
  ) -> torch.Tensor:
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
        input_dict=input_dict,
        global_valid_tokens=global_valid_tokens,
    )
    detached = loss.detach()
    self._step_loss = (
        detached if self._step_loss is None else self._step_loss + detached
    )
    return loss

  # -- per-step bookkeeping ----------------------------------------------
  def train_step(self, data_iterator: Iterator[dict[str, Any]]):
    self._maybe_validate(before_step=True)
    loader = self.dataloader
    if isinstance(loader, MicrobatchesPerStep):
      self.gradient_accumulation_steps = loader.accumulation_steps()
    start = self._note_step_start()
    self._step_loss = None
    self._last_grad_norm = None
    result = super().train_step(data_iterator)
    this_step = self._step_record(start)
    self.sft_records.append(this_step)
    logger.info("[sft] step %d: %s", self.step, this_step)
    drawn = self._take_step_batches(start)
    self._append_rank_line(this_step, drawn, start.learning_rates)
    self._rewrite_rank0_records()
    self._maybe_validate(before_step=False)
    return result

  def _note_step_start(self) -> _StepStart:
    stats = getattr(self.dataloader, "stats", None)
    log = getattr(self.dataloader, "batch_log", None)
    counters = (
        {}
        if stats is None
        else dict(
            documents=stats.documents,
            tokens=stats.tokens,
            valid_tokens=stats.valid_tokens,
        )
    )
    rates = self.lr_schedulers.get_metrics() if self._records_all_ranks else {}
    return _StepStart(
        log_length=None if log is None else len(log),
        learning_rates=rates,
        **counters,
    )

  def _step_record(self, start: _StepStart) -> dict[str, Any]:
    """This step's entry of ``sft_records``; loader counts are step deltas."""
    record: dict[str, Any] = {"step": self.step}
    if self._step_loss is not None:
      record["loss_local"] = float(self._step_loss)
    stats = getattr(self.dataloader, "stats", None)
    if stats is not None:
      record.update(
          epoch=stats.epoch,
          documents=stats.documents - (start.documents or 0),
          tokens=stats.tokens - (start.tokens or 0),
          valid_tokens_local=stats.valid_tokens - (start.valid_tokens or 0),
          dropped_total=stats.dropped,
          truncated_total=stats.truncated,
      )
    return record

  def _take_step_batches(self, start: _StepStart) -> list[dict[str, Any]]:
    """The batch-log entries this step added; the loader's log is emptied."""
    log = getattr(self.dataloader, "batch_log", None)
    if log is None:
      return []
    drawn = log[start.log_length or 0 :]
    log.clear()
    return drawn

  def _append_rank_line(
      self,
      record: dict[str, Any],
      batches: list[dict[str, Any]],
      learning_rates: dict[str, float],
  ) -> None:
    """With ``SFT_RECORDS_ALL_RANKS=1``, add this step's line to the rank's
    JSONL file (in ``$SFT_RECORDS_DIR``, else ``<dump_folder>/sft-records``)."""
    if not self._records_all_ranks:
      return
    root = os.environ.get("SFT_RECORDS_DIR") or os.path.join(
        self.config.dump_folder or ".", "sft-records"
    )
    os.makedirs(root, exist_ok=True)
    extra: dict[str, Any] = {
        "rank": self._rank,
        "batches": batches,
        "lr": learning_rates,
    }
    if self._last_grad_norm is not None:
      extra["grad_norm"] = float(self._last_grad_norm)
    line = json.dumps({**record, **extra}, sort_keys=True)
    with open(os.path.join(root, f"rank-{self._rank:05d}.jsonl"), "a") as out:
      out.write(line + "\n")

  def _rewrite_rank0_records(self) -> None:
    """Rank 0 keeps ``<dump_folder>/sft-records-rank0.json`` up to date."""
    folder = self.config.dump_folder
    if self._rank != 0 or not folder:
      return
    os.makedirs(folder, exist_ok=True)
    summary = {"parameters": self.parameter_report, "steps": self.sft_records}
    with open(os.path.join(folder, "sft-records-rank0.json"), "w") as out:
      json.dump(summary, out, indent=2)


@contextlib.contextmanager
def tpu_trainer_class(trainer_cls: type[TPUTrainer]) -> Iterator[None]:
  """Have upstream's TPU launcher build ``trainer_cls`` inside the block.

  ``torchtitan.experiments.tpu.train.start_trainer`` constructs whatever
  class that module's global ``TPUTrainer`` names at the moment it runs. The
  global is rebound for the duration of the block and handed back to
  upstream's class on every way out (return, exception, KeyboardInterrupt),
  so other code in the process keeps seeing ``TPUTrainer``.
  """
  module_globals = vars(tpu_entry)
  upstream_cls = module_globals["TPUTrainer"]
  module_globals["TPUTrainer"] = trainer_cls
  try:
    yield
  finally:
    module_globals["TPUTrainer"] = upstream_cls


def start_trainer(train_config: TPUTrainerConfig) -> None:
  """Upstream's TPU training launch with ``SFTTPUTrainer`` as the trainer."""
  with tpu_trainer_class(SFTTPUTrainer):
    tpu_entry.start_trainer(train_config)


if __name__ == "__main__":
  gmain.handle_main(start_trainer)
