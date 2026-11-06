"""Trainer that lets the turbo engine/backend own the training mechanics.

:class:`TurboTrainerMixin` reroutes the pieces the FSDP2 backend must own, and
is mixed ahead of swift's trainer base for any stage:

* ``_wrap_model``  -- the model is already FSDP2-sharded; don't let accelerate
  re-wrap it.
* ``create_optimizer`` -- build via the engine (FSDP2-aware); ``create_scheduler``
  uses the engine for engine-managed optimizers and HF's default otherwise.
* ``compute_loss``  -- ``engine.compute_loss`` (with SP loss re-scale).
* ``_get_data_collator`` - the taken-over plugin's collator, else
  ``SwiftInputCollator`` over swift's template collator.
* ``save_model`` -- engine writes DCP checkpoint state and HF exports; swift
  still records metadata.
* ``get_train_dataloader`` -- engine SP-aware sampler for all ``sp_size``.

:func:`make_turbo_seq2seq_trainer` combines it with swift's
:class:`Seq2SeqTrainer` for the SFT stage. :func:`make_turbo_dpo_trainer`
combines :class:`TurboDPOTrainerMixin` with swift's ``DPOTrainer`` and
overrides only the DPO forward (``concatenated_forward``) on flattened
batches. Everything else (logging, callbacks, the loop, grad accumulation)
stays swift/HF.

The mixin is module-level and imports swift only lazily inside method bodies,
so this module stays swift-free until a concrete trainer is actually built.
"""

from __future__ import annotations

import os
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ...engine import CPUOffloadMode
from ...engine.distributed import assert_finite
from ...engine.checkpoint import CheckpointFormat
from ...engine.checkpoint.layout import validate_output_dir
from ...logging import get_logger
from . import engine_registry

logger = get_logger(__name__)


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


def install_accelerator_overrides(trainer, engine) -> None:
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
    * ``prepare_optimizer`` -> retain Accelerate's wrapper while leaving
      optimizer-state placement to the TorchTurbo backend.
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

    original_prepare_optimizer = trainer.accelerator.prepare_optimizer

    def _prepare_optimizer(optimizer, device_placement=None):
        if optimizer is engine.optimizer:
            device_placement = False
        return original_prepare_optimizer(
            optimizer,
            device_placement=device_placement,
        )

    trainer.accelerator.prepare_model = _passthrough_prepare_model
    trainer.accelerator.prepare_optimizer = _prepare_optimizer
    trainer.accelerator.backward = _plain_backward


class TurboTrainerMixin:
    """Engine-backed overrides shared by every turbo takeover trainer.

    Mixed AHEAD of the swift trainer base (e.g.
    ``type("TurboSeq2SeqTrainer", (TurboTrainerMixin, Seq2SeqTrainer), {})``)
    so these methods shadow swift's. The base retains swift/HF logging,
    callbacks, the train loop, and grad accumulation.
    """

    # Checkpoint resume hooks — route to the engine.
    def _is_final_save(self) -> bool:
        """Swift saves its final weights through the internal checkpoint path.

        The final save is the checkpoint of the last step, or the one taken
        when training stops early, for example by ``EarlyStoppingCallback``.
        """
        reached_max_steps = (
            self.state.max_steps > 0 and self.state.global_step >= self.state.max_steps
        )
        return reached_max_steps or self.control.should_training_stop

    def _save_optimizer_and_scheduler(self, output_dir: str) -> None:
        engine = engine_registry.get_engine()
        engine.save_optimizer_and_scheduler(
            self.model,
            engine.optimizer,
            self.lr_scheduler,
            output_dir,
        )

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        engine = engine_registry.get_engine()
        engine.load_from_checkpoint(self.model, resume_from_checkpoint)

    def _load_optimizer_and_scheduler(self, checkpoint):
        if checkpoint:
            engine = engine_registry.get_engine()
            engine.load_optimizer_and_scheduler(
                self.model,
                engine.optimizer,
                self.lr_scheduler,
                checkpoint,
            )

    def __init__(self, *args, **kwargs):
        # Routed plugin collator handed over by the pipeline; pop before super so
        # the swift/HF base never sees it.
        self._plugin_collator = kwargs.pop('plugin_collator', None)
        super().__init__(*args, **kwargs)
        engine = engine_registry.get_engine_or_none()
        if engine is None:
            return

        # Resolve GA here (not in get_train_dataloader) so the HF/swift
        # "Running training" summary shows the SP-compensated value instead
        # of the raw script value.
        self.args.gradient_accumulation_steps = (
            engine.resolve_gradient_accumulation_steps(
                self.args.gradient_accumulation_steps
            )
        )

        install_clip_override(self, engine)
        install_accelerator_overrides(self, engine)

        offload = engine.config.backend_config.cpu_offload
        if offload.enabled and offload.mode is CPUOffloadMode.FULL:
            import torch
            from ...runtime import get_device_name

            def _offload_device(model):
                try:
                    first = next(model.parameters())
                except StopIteration:
                    return torch.device(get_device_name())
                # FSDP2-managed DTensor offloaded to CPU -> report compute device.
                if hasattr(first, "device_mesh") and first.device.type == "cpu":
                    return torch.device(get_device_name())
                return first.device

            model = self.model
            if model is not None:
                setattr(type(model), "device", _offload_device(model))

    def _move_model_to_device(self, model, _device):
        # turbo engine manages param placement, so skip the move.
        return model

    def train(self, *args, **kwargs):
        """Wrap swift's training loop with TorchTurbo profiling when
        ``--tt_profiling`` is set. Mirrors the llamafactory adapter.

        swift owns the training loop, so the profiler is driven by an HF
        ``TrainerCallback`` whose
        ``on_step_end`` calls ``engine.step_profiling()`` once per optimizer
        step. ``start_profiling`` / ``stop_profiling`` bracket the whole loop;
        traces land under ``{output_dir}/profiling/``.
        """
        from ...engine.args import get_torchturbo_args

        engine = engine_registry.get_engine_or_none()
        if engine is not None:
            # Fail before training instead of at the first checkpoint save.
            validate_output_dir(self.args.output_dir, engine.model_name_or_path)
        tt_args = get_torchturbo_args()
        profiling_enabled = (
            engine is not None and tt_args is not None and tt_args.profiling
        )
        memory_snapshot_steps = (
            tt_args.memory_snapshot_steps
            if engine is not None and tt_args is not None
            else None
        )
        if profiling_enabled or memory_snapshot_steps is not None:
            self.add_callback(self._make_torchturbo_profiling_callback())
            engine.start_profiling(
                self.args.output_dir,
                profiling_enabled,
                memory_snapshot_steps,
            )
        try:
            return super().train(*args, **kwargs)
        finally:
            if profiling_enabled or memory_snapshot_steps is not None:
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
        if self.optimizer is not None:
            return self.optimizer

        engine = engine_registry.get_engine()
        if engine.config.optimizer.name is not None:
            self.optimizer = engine.prepare_optimizer()
            return self.optimizer

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
        optimizer_kwargs = dict(optimizer_kwargs)
        optimizer_kwargs["weight_decay"] = self.args.weight_decay
        self.optimizer = engine.prepare_optimizer(optimizer_cls, **optimizer_kwargs)
        return self.optimizer

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optimizer | None = None
    ) -> LRScheduler:
        from swift.trainers.mixin import SwiftMixin

        if optimizer is None:
            optimizer = self.optimizer
        engine = engine_registry.get_engine()
        if engine.config.optimizer.name is not None:
            # Engine-managed optimizers take their LRs from engine config, so
            # the engine also builds their LR schedule.
            if self.lr_scheduler is None:
                self.lr_scheduler = engine.prepare_lr_scheduler(
                    self.args.lr_scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                    scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
                )
                self._created_lr_scheduler = True
            return self.lr_scheduler

        # Skip swift's OptimizerCallback scheduler; use HF's default.
        self.lr_scheduler = super(SwiftMixin, self).create_scheduler(num_training_steps, optimizer)
        return self.lr_scheduler

    def _get_data_collator(self, args, template):
        """Serve the routed plugin's collator (handed over at construction); else
        wrap swift's template collator with ``SwiftInputCollator``."""
        if self._plugin_collator is not None:
            return self._plugin_collator
        from .inputs_bridging import SwiftInputCollator

        return SwiftInputCollator(super()._get_data_collator(args, template), template)

    def _prepare_inputs(self, inputs):
        from transformers.trainer import Trainer as HfTrainer

        return HfTrainer._prepare_inputs(self, inputs)

    def _report_engine_metrics(self, logs) -> None:
        engine = engine_registry.get_engine_or_none()
        if engine is not None:
            engine.report_metrics(logs, self.accelerator.gather)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        engine = engine_registry.get_engine_or_none()
        if engine is None:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        for key in ("compute_loss_func", "loss_scale", "logits_to_keep"):
            inputs.pop(key, None)

        sp_size = engine.get_sp_info()["sp_size"]
        loss, outputs = engine.compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
            gradient_accumulation_steps=(
                self.current_gradient_accumulation_steps if model.training else 1
            ),
        )

        # SP gathers loss to full length; scale back when HF gives no
        # num_items_in_batch.
        if num_items_in_batch is None and sp_size > 1:
            loss = loss * sp_size

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, *args, **kwargs):
        """Check loss and collect engine metrics after backward."""
        result = super().training_step(model, inputs, *args, **kwargs)
        engine = engine_registry.get_engine_or_none()
        if engine is not None:
            assert_finite(result, "loss")
            engine.collect_metrics(model)
        return result

    def log(self, logs, *args, **kwargs):
        self._report_engine_metrics(logs)
        return super().log(logs, *args, **kwargs)

    def _save_model(self, output_dir=None, state_dict=None):
        # Weights are written by save_model() below; keep HF/swift from
        # materializing a DTensor state_dict.
        return None

    def save_model(self, output_dir=None, _internal_call=False):
        engine = engine_registry.get_engine()
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Checkpoints from the HF machinery (_internal_call=True) store
        # native model state; swift's last-step checkpoint also exports HF weights.
        # Explicit saves and model-only checkpoints are exports.
        if _internal_call and not self.args.save_only_model:
            engine.save_model(
                self.model, output_dir, final=self._is_final_save()
            )
        else:
            engine.save_model(self.model, output_dir, format=CheckpointFormat.HF)
        # Let swift persist training_args.bin / args.json / tokenizer;
        # _save_model is a no-op above, so weights aren't double-written.
        super().save_model(output_dir, _internal_call=_internal_call)

    def get_train_dataloader(self, skip_batches=0):
        engine = engine_registry.get_engine_or_none()
        if engine is None or self.train_dataset is None:
            return super().get_train_dataloader(skip_batches=skip_batches)

        # GA was already resolved in __init__; just build the dataloader.
        return engine.create_dataloader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            shuffle=self.args.train_dataloader_shuffle,
            seed=self.args.data_seed,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=True,
        )


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


class _MTPAuxLossAutoScaler(torch.autograd.Function):
    """Add the MTP gradient to the graph, as ``MoEAuxLossAutoScaler`` does."""

    @staticmethod
    def forward(
        ctx, output: torch.Tensor, mtp_loss: torch.Tensor, loss_scale: float
    ) -> torch.Tensor:
        ctx.save_for_backward(mtp_loss)
        ctx.loss_scale = loss_scale
        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        (mtp_loss,) = ctx.saved_tensors
        return grad_output, torch.ones_like(mtp_loss) * ctx.loss_scale, None


class TurboDPOTrainerMixin(TurboTrainerMixin):
    """Add Swift DPO wiring while retaining TorchTurbo engine ownership.

    :class:`~.inputs_bridging.SwiftInputCollator` flattens every DPO batch
    into one row with cu boundaries, packed or not, so one
    ``concatenated_forward`` serves all sequence-parallel sizes.
    """

    def __init__(self, model=None, ref_model=None, *_args, **kwargs):
        super().__init__(model, ref_model=None, *_args, **kwargs)
        # The engine's routers attach the auxiliary gradients and report their
        # metrics. Swift must not also read or add the model's HF aux_loss.
        self.aux_loss_enabled = False
        if ref_model is not None:
            self.ref_model = ref_model
            logger.info_rank0(
                "TurboDPOTrainer: restored engine-sharded reference model "
                "(accelerate re-wrap skipped)."
            )

    def _prepare_inputs(self, inputs):
        """Use Swift's preference-input preparation, not the SFT override."""
        return super(TurboTrainerMixin, self)._prepare_inputs(inputs)

    def get_train_dataloader(self, *args, **kwargs):
        """Use Swift's preference sampler at sp=1 and Turbo's sampler for SP."""
        engine = engine_registry.get_engine_or_none()
        if engine is None or engine.get_sp_info().get("sp_size", 1) <= 1:
            return super(TurboTrainerMixin, self).get_train_dataloader(*args, **kwargs)
        return super().get_train_dataloader(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Resolve directly to Swift's DPO loss implementation.
        return super(TurboTrainerMixin, self).compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    def concatenated_forward(self, model, batch, is_ref_model: bool = False):
        """Compute Swift's DPO forward outputs from one flattened row.

        The row holds ``2 * N`` segments delimited by ``cu_seq_lens_q``: the
        ``N`` chosen samples, then their rejected counterparts. The model
        receives the global boundaries. Under sequence parallelism each rank
        runs its token shard and the per-segment sums are all-reduced over
        the SP group.

        Returns:
            The outputs Swift's ``get_batch_loss_metrics`` reads: chosen and
            rejected log-probs, mean logits and the policy NLL loss. The engine
            owns router auxiliary gradients and metrics.
        """
        from torch.distributed.nn.functional import all_reduce

        from ...models.modules.attention.ulysses import slice_input_tensor
        from ...utils.flash_attn_utils import cu_seqlens_to_segment_ids

        engine = engine_registry.get_engine()
        sp_info = engine.get_sp_info()
        sp_size = sp_info["sp_size"]

        inputs = dict(batch)
        # Swift's DPO loss reads neither field; keep them out of the model call.
        for key in ("text_position_ids", "loss_scale"):
            inputs.pop(key, None)
        inputs = model.on_prepare_inputs(inputs)
        # Pad the token fields, labels included, to a multiple of the SP size.
        inputs = engine.backend.prepare_inputs(model, inputs)
        labels = inputs.pop("labels")

        # MFU and MoE runtime statistics read the whole row.
        if model.training:
            engine._record_micro(inputs)
            gradient_accumulation_steps = self.current_gradient_accumulation_steps
        else:
            gradient_accumulation_steps = 1
        engine._prepare_moe_runtime(model, inputs, gradient_accumulation_steps)

        cu_seqlens = inputs["cu_seq_lens_q"]
        num_segments = cu_seqlens.numel() - 1
        num_examples = num_segments // 2
        segment_ids = cu_seqlens_to_segment_ids(cu_seqlens, labels.size(-1))
        # The reference pass reaches the model's MTP block too, so shadow the
        # LM head for both; only the policy pass turns MTP into a loss.
        mtp_enabled = (
            engine.config.mtp.enabled
            and model.config.num_nextn_predict_layers > 0
        )
        shadow_lm_head = mtp_enabled and model.training
        run_mtp = shadow_lm_head and not is_ref_model
        # MTP stages shift this same row further, so build their labels first.
        mtp_labels: list[torch.Tensor] = []
        mtp_normalizers: list[torch.Tensor] = []
        if run_mtp:
            mtp_labels, mtp_normalizers = self._mtp_stage_labels(
                labels,
                segment_ids,
                num_examples,
                model.config.num_nextn_predict_layers,
            )
        labels = self._shift_packed_labels(labels, segment_ids, self.label_pad_token_id)
        loss_mask = labels != self.label_pad_token_id

        # Weight each token's log-prob in its segment sum. The weights depend
        # on whole segments, so compute them before SP sharding.
        segment_index = (segment_ids - 1).clamp_min(0)
        token_weights = torch.ones_like(labels, dtype=torch.float32)
        completion_lengths = self._sum_by_segment(
            loss_mask.float(), segment_ids, num_segments
        )
        loss_types = self.loss_type if isinstance(self.loss_type, list) else [self.loss_type]
        if "ipo" in loss_types:
            # IPO averages the log-probs over each completion.
            token_weights = token_weights / completion_lengths[segment_index].clamp_min(1)
        ld_alpha = self.args.ld_alpha
        if ld_alpha is not None and not is_ref_model:
            # LD-DPO scales the completion tokens past the shorter completion
            # of each pair by ld_alpha.
            completion_starts = completion_lengths.cumsum(0) - completion_lengths
            completion_positions = (
                loss_mask.cumsum(-1) - 1 - completion_starts[segment_index]
            )
            public_lengths = torch.minimum(
                completion_lengths[:num_examples], completion_lengths[num_examples:]
            ).repeat(2)
            is_public = completion_positions < public_lengths[segment_index]
            token_weights = torch.where(
                is_public, token_weights, token_weights * ld_alpha
            )

        if sp_size > 1:
            # The model slices its own inputs at its SP boundary, so only the
            # loss-side tensors are sharded here to the same local tokens.
            shard = partial(
                slice_input_tensor,
                dim=-1,
                group=sp_info["sp_group"],
                rank=sp_info["sp_rank"],
            )
            labels = shard(labels)
            segment_ids = shard(segment_ids)
            token_weights = shard(token_weights)
            # MTP hidden states cover the same local tokens as the backbone's.
            mtp_labels = [shard(stage_labels) for stage_labels in mtp_labels]

        if self.get_use_logits_to_keep():
            # Project only the positions that predict a target token.
            keep = labels[0] != self.label_pad_token_id
            inputs["logits_to_keep"] = keep
            labels = labels[:, keep]
            segment_ids = segment_ids[:, keep]
            token_weights = token_weights[:, keep]

        if shadow_lm_head:
            outputs = self._forward_with_shadowed_lm_head(model, inputs)
            # The head was bypassed, so project the kept positions here instead.
            (
                lm_head_weight,
                main_hidden_states,
                loss_anchor,
            ) = engine.backend._prepare_chunk_loss_lm_head(model, outputs.logits)
            main_logits = F.linear(
                main_hidden_states.to(lm_head_weight.dtype), lm_head_weight
            )
        else:
            outputs = model(**inputs, use_cache=False)
            main_logits = outputs.logits
        if getattr(outputs, "dspark_loss_fn", None) is not None:
            raise ValueError(
                "DeepSeek V4 DSpark training is not implemented for Swift DPO; "
                "disable MTP/DSpark or use an explicit DSpark DPO objective."
            )

        per_token_logps, mean_logits, loss_mask = self.get_per_token_logps(
            main_logits, labels, label_pad_token_id=self.label_pad_token_id
        )
        if run_mtp:
            if not outputs.mtp_logits:
                raise RuntimeError(
                    "MTP is enabled but the model returned no MTP outputs. MTP "
                    "needs 'input_ids' in the forward kwargs, which Swift drops "
                    "unless the template preserves them; use a turbo-registered "
                    "template such as qwen3_5 or qwen3_8."
                )
            mtp_loss, stage_losses = self._mtp_auxiliary_loss(
                engine,
                outputs.mtp_logits,
                mtp_labels,
                mtp_normalizers,
                lm_head_weight,
                sp_size,
                sp_info["sp_group"],
            )
            # Always zero; it only carries the autograd edge FSDP needs.
            mtp_loss = mtp_loss + loss_anchor.to(mtp_loss.dtype)
            # The kernel applied the factor; only the accumulation divisor is left.
            per_token_logps = _MTPAuxLossAutoScaler.apply(
                per_token_logps, mtp_loss, 1.0 / gradient_accumulation_steps
            )
            self._record_mtp_metrics(
                engine, stage_losses, engine.config.mtp.loss_scaling_factor
            )
        segment_logps = self._sum_by_segment(
            per_token_logps * token_weights, segment_ids, num_segments
        )
        is_chosen = loss_mask & (segment_ids <= num_examples)
        is_rejected = loss_mask & (segment_ids > num_examples)
        # Reduce the segment sums and the metric sums in one collective.
        stats = torch.cat((
            segment_logps,
            torch.stack((
                (mean_logits * is_chosen).sum(),
                (mean_logits * is_rejected).sum(),
                (per_token_logps * is_chosen).sum(),
                is_chosen.sum().float(),
                is_rejected.sum().float(),
            )),
        ))
        if sp_size > 1:
            stats = all_reduce(stats, group=sp_info["sp_group"])
        (
            chosen_logits_sum,
            rejected_logits_sum,
            chosen_logps_sum,
            num_chosen_tokens,
            num_rejected_tokens,
        ) = stats[num_segments:]

        num_chosen_tokens = num_chosen_tokens.clamp_min(1)
        output = {
            "chosen_logps": stats[:num_examples],
            "rejected_logps": stats[num_examples:num_segments],
            "mean_chosen_logits": chosen_logits_sum / num_chosen_tokens,
            "mean_rejected_logits": rejected_logits_sum / num_rejected_tokens.clamp_min(1),
        }
        if not is_ref_model:
            output["nll_loss"] = -chosen_logps_sum / num_chosen_tokens
        return output

    @staticmethod
    def _sum_by_segment(
        values: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
    ) -> torch.Tensor:
        """Sum ``[1, seq]`` values per segment, dropping ID 0 (tail padding)."""
        sums = values.new_zeros(num_segments + 1).index_add(0, segment_ids[0], values[0])
        return sums[1:]

    @staticmethod
    def _shift_packed_labels(
        labels: torch.Tensor,
        segment_ids: torch.Tensor,
        pad_token_id: int,
        shift: int = 1,
    ) -> torch.Tensor:
        """Shift labels ``shift`` positions ahead, only inside each segment."""
        next_labels = F.pad(labels, (0, shift), value=pad_token_id)[..., shift:]
        segment = segment_ids.to(torch.long)
        same_segment = segment == F.pad(segment, (0, shift), value=0)[..., shift:]
        valid = segment.ne(0) & same_segment
        return torch.where(valid, next_labels, torch.full_like(labels, pad_token_id))

    def _mtp_stage_labels(
        self,
        labels: torch.Tensor,
        segment_ids: torch.Tensor,
        num_examples: int,
        num_stages: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Build one target row and one normalizer per stage, chosen rows only."""
        # Supervising the rejected segments would pull against the preference
        # objective. Normalizers span the whole row so every rank agrees.
        is_rejected = segment_ids > num_examples
        stage_labels: list[torch.Tensor] = []
        normalizers: list[torch.Tensor] = []
        for stage in range(num_stages):
            shifted = self._shift_packed_labels(
                labels, segment_ids, self.label_pad_token_id, shift=stage + 2
            )
            shifted = shifted.masked_fill(is_rejected, self.label_pad_token_id)
            stage_labels.append(shifted)
            supervised = shifted != self.label_pad_token_id
            normalizers.append(supervised.sum().clamp_min(1))
        return stage_labels, normalizers

    @staticmethod
    def _forward_with_shadowed_lm_head(model, inputs: dict) -> Any:
        """Run the forward with the LM head shadowed by an identity module."""
        # Leaving the head in place costs one [tokens, vocab] tensor per MTP
        # stage; with it shadowed, logits and mtp_logits hold hidden states.
        # The shadow sits on the parent attribute so FSDP registration stays.
        lm_head = model.get_lm_head()
        if lm_head.bias is not None:
            raise NotImplementedError(
                "Swift DPO with MTP does not support an lm_head bias"
            )
        module_name = next(
            name for name, module in model.named_modules() if module is lm_head
        )
        parent_name, _, attr_name = module_name.rpartition(".")
        parent = model.get_submodule(parent_name)

        object.__setattr__(parent, attr_name, torch.nn.Identity())
        try:
            return model(**inputs, use_cache=False)
        finally:
            object.__delattr__(parent, attr_name)

    def _mtp_auxiliary_loss(
        self,
        engine: Any,
        mtp_hidden_states: list[torch.Tensor],
        stage_labels: list[torch.Tensor],
        normalizers: list[torch.Tensor],
        lm_head_weight: torch.Tensor,
        sp_size: int,
        sp_group: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the scaled total and the raw per-stage MTP cross-entropy."""
        from torch.distributed.nn.functional import all_reduce

        from ...models.modules.loss import chunk_loss

        hidden_states = [
            hidden_state.to(lm_head_weight.dtype).contiguous()
            for hidden_state in mtp_hidden_states
        ]
        # Every stage carries the full factor; Megatron divides by stage count.
        loss_scale = engine.config.mtp.loss_scaling_factor
        # The streams span the whole row, not the kept positions, so this path
        # always chunks; only chunk_size comes from loss_chunk. The factor goes
        # to the kernel so a zero detaches the streams.
        total_loss, stage_losses = chunk_loss(
            hidden_states,
            stage_labels,
            lm_head=lm_head_weight,
            normalizers=normalizers,
            loss_scales=[loss_scale] * len(hidden_states),
            chunk_size=engine.config.loss_chunk.chunk_size,
        )
        if sp_size > 1:
            # Reduce the objective and its per-stage breakdown in one collective.
            reduced = all_reduce(
                torch.cat((total_loss.reshape(1), stage_losses)), group=sp_group
            )
            total_loss = reduced[0]
            stage_losses = reduced[1:]
        return total_loss, stage_losses

    @staticmethod
    def _record_mtp_metrics(
        engine: Any, stage_losses: torch.Tensor, loss_scale: float
    ) -> None:
        """Report mtp_loss/*; the DPO path never runs the engine's reporting."""
        stages = stage_losses.detach().unbind()
        metrics = {f"mtp_loss/{index}": value for index, value in enumerate(stages)}
        metrics["mtp_loss/avg"] = sum(stages) / len(stages)
        metrics["mtp_loss/scaled"] = stage_losses.detach().sum() * loss_scale
        engine.metric_tracker.record(metrics)

    def get_per_token_logps(self, logits, labels, label_pad_token_id=-100, reduction="mean"):
        # Use LF's full-vocabulary log-softmax while preserving Swift's return contract.
        if logits.shape[:-1] != labels.shape:
            raise ValueError(
                "DPO forward requires logits batch and sequence dimensions to match "
                f"labels, but got logits {tuple(logits.shape[:-1])} and labels "
                f"{tuple(labels.shape)}. For multimodal models, templates must expand "
                "visual placeholder tokens before the model forward."
            )
        logits = logits.float()
        loss_mask = labels != label_pad_token_id
        safe_labels = labels.masked_fill(~loss_mask, 0)
        if reduction == "sum":
            mean_logits = logits.sum(dim=-1)
        else:
            mean_logits = logits.mean(dim=-1)
        per_token_logps = torch.gather(
            logits.log_softmax(-1),
            dim=-1,
            index=safe_labels.unsqueeze(-1),
        ).squeeze(-1)
        return (
            per_token_logps * loss_mask,
            mean_logits * loss_mask,
            loss_mask,
        )


_dpo_cls = None


def make_turbo_dpo_trainer():
    """Build the lazy TorchTurbo DPO trainer class."""
    from swift.rlhf_trainers import DPOTrainer

    return type(
        "TurboDPOTrainer",
        (TurboDPOTrainerMixin, DPOTrainer),
        {"__doc__": "swift DPOTrainer with the turbo engine/backend driving training."},
    )


def get_turbo_dpo_trainer():
    """Return (and cache) the ``TurboDPOTrainer`` class."""
    global _dpo_cls
    if _dpo_cls is None:
        _dpo_cls = make_turbo_dpo_trainer()
    return _dpo_cls
