"""Trainer that lets the turbo engine/backend own the training mechanics.

:class:`TurboTrainerMixin` reroutes the pieces the FSDP2 backend must own, and
is mixed ahead of swift's trainer base for any stage:

* ``_wrap_model``  — the model is already FSDP2-sharded; don't let accelerate
  re-wrap it.
* ``create_optimizer`` — build via the engine (FSDP2-aware); ``create_scheduler``
  keeps HF's default around the engine optimizer.
* ``compute_loss``  — ``engine.backend.compute_loss`` (with SP loss re-scale).
* ``_prepare_inputs`` — bridge swift-packed inputs to turbo's forward contract
  (see :mod:`torchturbo.adapters.msswift.inputs_bridging`).
* ``save_model`` — engine writes DCP weights; swift still records metadata.
* ``get_train_dataloader`` — engine SP-aware sampler when ``sp_size > 1``.

:func:`make_turbo_seq2seq_trainer` combines it with swift's
:class:`Seq2SeqTrainer` for the SFT stage. A future RLHF stage would combine it
with ``DPOTrainer`` etc. and only override the RLHF-specific forward
(``concatenated_forward`` / reference log-probs). Everything else (logging,
callbacks, the loop, grad accumulation) stays swift/HF.

The mixin is module-level and imports swift only lazily inside method bodies,
so this module stays swift-free until a concrete trainer is actually built.
"""

from __future__ import annotations

import os
from functools import partial
from typing import Any

from ...logging import get_logger
from . import engine_registry

logger = get_logger(__name__)


def apply_ulysses_sp_patch(engine: Any) -> bool:
    """Redirect global flash-attention to the Ulysses SP kernel when ``sp>1``.

    swift runs its own training loop, so turbo's engine hooks never fire — the
    SP attention patch is applied eagerly from the trainer init instead.
    Idempotent; a no-op when SP is off.  Returns whether the patch was applied.
    """
    try:
        sp_info = engine.get_sp_info()
    except Exception:
        return False
    sp_group = sp_info.get("sp_group")
    sp_size = sp_info.get("sp_size", 1)
    if not sp_group or sp_size <= 1:
        return False

    import transformers.integrations.flash_attention as flash_integration
    import transformers.modeling_flash_attention_utils as flash_utils

    from torchturbo.models.modules.attention import ulysses

    patched = partial(ulysses.ulysses_flash_attention_forward, sp_group=sp_group, sp_size=sp_size)
    flash_utils._flash_attention_forward = patched
    if hasattr(flash_integration, "_flash_attention_forward"):
        flash_integration._flash_attention_forward = patched
    logger.info("Ulysses SP flash-attention patch applied (sp_size=%d)", sp_size)
    return True


def install_clip_override(trainer, engine) -> None:
    """Route ``accelerator.clip_grad_norm_`` through the FSDP2 backend.

    HF's default clip can't reduce a DTensor grad norm (``.item()`` would
    raise), so bind an instance-level override that defers to the backend.
    No-op when the backend has no ``clip_grad_norm_``.
    """
    backend = getattr(engine, "backend", None)
    if backend is None or not hasattr(backend, "clip_grad_norm_"):
        return

    def _clip(parameters, max_norm, *args, **kwargs):
        norm = backend.clip_grad_norm_(parameters, max_norm, *args, **kwargs)
        try:
            return float(norm)
        except (TypeError, ValueError):
            return norm

    trainer.accelerator.clip_grad_norm_ = _clip


def install_accelerator_overrides(trainer) -> None:
    """Stop accelerate from re-wrapping the FSDP2 model / hijacking backward.

    HF ``Trainer._prepare_for_training`` calls
    ``self.accelerator.prepare(self.model, self.optimizer)``. With turbo having
    forced ``fsdp=""`` (the engine owns FSDP2 sharding), accelerate's
    ``prepare_model`` falls back to DDP-wrapping the model and broadcasting its
    FSDP2 DTensor params — which crashes on
    ``found no DeviceMesh from dtensor args for c10d.broadcast_``.

    Mirrors the llamafactory adapter's ``AcceleratorPatch``, but instance-scoped
    to this trainer's accelerator (consistent with
    :func:`install_clip_override`):

    * ``prepare_model`` -> passthrough: skip DDP/FSDP wrapping (the engine
      already sharded the model) but still track it in ``accelerator._models``
      so HF/accelerate bookkeeping (``unwrap_model`` / ``get_state_dict``) works.
      The optimizer is still accelerated (grad accumulation / scheduling) via
      ``prepare_optimizer``.
    * ``backward`` -> plain ``loss.backward()`` — FSDP2's own backward hooks do
      the gradient reduction, instead of accelerate's dispatch on DTensor grads.

    ``clip_grad_norm_`` is handled separately by :func:`install_clip_override`.
    """

    def _passthrough_prepare_model(model, device_placement=None, evaluation_mode=False):
        # accelerate's own prepare_model appends to _models for bookkeeping; keep
        # that so HF/accelerate state queries still find the model, but skip ALL
        # wrapping (the model is already FSDP2-sharded by the turbo engine).
        trainer.accelerator._models.append(model)
        return model

    def _plain_backward(loss, **kwargs):
        loss.backward()

    trainer.accelerator.prepare_model = _passthrough_prepare_model
    trainer.accelerator.backward = _plain_backward


class TurboTrainerMixin:
    """Engine-backed overrides shared by every turbo takeover trainer.

    Mixed AHEAD of the swift trainer base (e.g.
    ``type("TurboSeq2SeqTrainer", (TurboTrainerMixin, Seq2SeqTrainer), {})``)
    so these methods shadow swift's. The base retains swift/HF logging,
    callbacks, the train loop, and grad accumulation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        engine = engine_registry.get_engine_or_none()
        if engine is None:
            return
        apply_ulysses_sp_patch(engine)
        install_clip_override(self, engine)
        install_accelerator_overrides(self)

    def train(self, *args, **kwargs):
        """Wrap swift's training loop with TorchTurbo profiling when
        ``--tt_profiling`` is set. Mirrors the llamafactory adapter.

        swift owns the training loop (turbo engine hooks never fire — see
        :func:`apply_ulysses_sp_patch`), so the profiler can't be driven by
        engine hooks; instead attach an HF ``TrainerCallback`` whose
        ``on_step_end`` calls ``engine.step_profiling()`` once per optimizer
        step. ``start_profiling`` / ``stop_profiling`` bracket the whole loop;
        traces land under ``{output_dir}/profiling/``.
        """
        from ...engine.args import get_torchturbo_args

        engine = engine_registry.get_engine_or_none()
        tt_args = get_torchturbo_args()
        profiling_enabled = (
            engine is not None and tt_args is not None and tt_args.profiling
        )
        if profiling_enabled:
            self.add_callback(self._make_torchturbo_profiling_callback())
            engine.start_profiling(self.args.output_dir)
        try:
            return super().train(*args, **kwargs)
        finally:
            if profiling_enabled:
                engine.stop_profiling()

    def _make_torchturbo_profiling_callback(self):
        """HF ``TrainerCallback`` that advances TorchTurbo profiling per step.

        Built in a closure, not at module scope — it is held only by the
        trainer (never pickled through swift's ``dataset.map``), matching the
        llamafactory adapter's callback factory.
        """
        from transformers import TrainerCallback

        engine = engine_registry.get_engine_or_none()

        class TorchTurboProfilingCallback(TrainerCallback):
            """Bridge swift/HF step boundaries to TorchTurbo profiling."""

            def on_step_end(self, args, state, control, **kwargs):
                if engine is not None:
                    engine.step_profiling()
                return control

        return TorchTurboProfilingCallback()

    def _wrap_model(self, model, training=True, dataloader=None, **kwargs):
        # The model is already FSDP2-sharded; return it as-is.
        self.model_wrapped = model
        return model

    def create_optimizer(self, model=None):
        engine = engine_registry.get_engine()
        opt_res = self.get_optimizer_cls_and_kwargs(self.args)
        # transformers' get_optimizer_cls_and_kwargs returns either
        # (optimizer_cls, optimizer_kwargs) [2-tuple] or
        # (<optimizer_name>, optimizer_cls, optimizer_kwargs) [3-tuple]
        # across versions; handle both so the adapter isn't pinned to one.
        if len(opt_res) == 3:
            _, optimizer_cls, optimizer_kwargs = opt_res
        else:
            optimizer_cls, optimizer_kwargs = opt_res
        self.optimizer = engine.prepare_optimizer(optimizer_cls, **optimizer_kwargs)
        self._optimizer_ori = self.optimizer
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        from swift.trainers.mixin import SwiftMixin

        if optimizer is None:
            optimizer = getattr(self, "_optimizer_ori", None) or self.optimizer
        # Skip swift's OptimizerCallback scheduler; use HF's default.
        self.lr_scheduler = super(SwiftMixin, self).create_scheduler(num_training_steps, optimizer)
        return self.lr_scheduler

    def _prepare_inputs(self, inputs):
        from transformers.trainer import Trainer as HfTrainer

        inputs = HfTrainer._prepare_inputs(self, inputs)
        # Bridge swift-packed inputs into turbo's forward contract:
        # integer-attention-mask rebuild, loss_scale fold, fixed-length pad
        # (text + mrope rows + cu_seq_lens + [4,b,s] position_ids). See
        # inputs_bridging.py.
        from .inputs_bridging import bridge_swift_inputs

        engine = engine_registry.get_engine_or_none()
        return bridge_swift_inputs(inputs, engine)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        engine = engine_registry.get_engine_or_none()
        if engine is None:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        for key in ("compute_loss_func", "loss_scale", "logits_to_keep"):
            inputs.pop(key, None)

        loss, outputs = engine.backend.compute_loss(model, inputs, return_outputs=True)

        # SP gathers loss to full length; scale back when HF gives no
        # num_items_in_batch.
        if num_items_in_batch is None and engine.get_sp_info().get("sp_size", 1) > 1:
            loss = loss * engine.get_sp_info()["sp_size"]

        return (loss, outputs) if return_outputs else loss

    def _save_model(self, output_dir=None, state_dict=None):
        # Weights are written by save_model() below; keep HF/swift from
        # materializing a DTensor state_dict.
        return None

    def save_model(self, output_dir=None, _internal_call=False):
        engine = engine_registry.get_engine_or_none()
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        if engine is not None:
            os.makedirs(output_dir, exist_ok=True)
            engine.save_model(self.model, output_dir)
        # Let swift persist training_args.bin / args.json / tokenizer;
        # _save_model is a no-op above, so weights aren't double-written.
        super().save_model(output_dir, _internal_call=_internal_call)

    def get_train_dataloader(self, skip_batches=0):
        engine = engine_registry.get_engine_or_none()
        sp_size = engine.get_sp_info().get("sp_size", 1) if engine is not None else 1
        if engine is not None and sp_size > 1 and self.train_dataset is not None:
            # turbo's SP sampler (GatherAndInnerLoopSPDistributedSampler) yields
            # sp_size× more samples per rank than a standard DistributedSampler:
            # every SP group of sp_size ranks jointly consumes sp_size samples
            # per iteration block, so each rank iterates over sp_size samples
            # per logical SP step (sp_size micro-batches feed one all-gather
            # / scatter sequence step). Without compensation the HF step count
            # (len(dataloader) // gradient_accumulation_steps) is inflated by
            # sp_size — e.g. sp=2 doubled steps 1019 -> 2040 vs the megatron
            # path. Scale gradient_accumulation_steps by sp_size ONCE so the
            # effective global batch size / step count match the non-SP path.
            # Mirrors the llamafactory adapter's get_train_dataloader.
            args = self.args
            if not getattr(args, "_torchturbo_sp_adjusted", False):
                orig = args.gradient_accumulation_steps
                args.gradient_accumulation_steps = orig * sp_size
                args._torchturbo_sp_adjusted = True
                logger.info(
                    "SP=%d: gradient_accumulation_steps %d -> %d (step count "
                    "compensated)",
                    sp_size, orig, args.gradient_accumulation_steps,
                )
            return engine.create_dataloader(
                self.train_dataset,
                batch_size=self._train_batch_size,
                shuffle=self.args.train_dataloader_shuffle,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                drop_last=True,
            )
        return super().get_train_dataloader(skip_batches=skip_batches)


def make_turbo_seq2seq_trainer():
    """Build (lazily) the SFT trainer: ``TurboTrainerMixin`` + ``Seq2SeqTrainer``.

    A future RLHF stage's trainer is the same mixin ahead of e.g.
    ``DPOTrainer`` — only the RLHF-specific forward is overridden on that class.
    """
    from swift.trainers.seq2seq_trainer import Seq2SeqTrainer

    return type(
        "TurboSeq2SeqTrainer",
        (TurboTrainerMixin, Seq2SeqTrainer),
        {"__doc__": "Seq2SeqTrainer with the turbo engine/backend driving training."},
    )


_cls = None


def get_turbo_seq2seq_trainer():
    """Return (and cache) the ``TurboSeq2SeqTrainer`` class."""
    global _cls
    if _cls is None:
        _cls = make_turbo_seq2seq_trainer()
    return _cls