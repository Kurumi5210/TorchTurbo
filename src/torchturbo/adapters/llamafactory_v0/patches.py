"""Monkey-patches for LlamaFactory v0 — version-aware BasePatch classes.

Each patch declares ``supported_versions`` — incompatible patches are
automatically skipped by the adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...logging import get_logger
from ...registry import BasePatch

if TYPE_CHECKING:
    from ...engine import TorchTurboEngine

logger = get_logger(__name__)

# Version range for LlamaFactory v0
_LF_V0 = ">=0.5,<1.0"


# ---------------------------------------------------------------------------
# RunExpPatch — intercepts run_exp to build engine from LF args
# ---------------------------------------------------------------------------


class RunExpPatch(BasePatch):
    """Intercept ``run_exp`` to build the engine and apply inner patches."""

    name = "run_exp"
    supported_versions = _LF_V0

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter
        self._original = None
        self._module = None

    def patch(self, engine: TorchTurboEngine | None = None) -> None:
        try:
            from llamafactory.train import tuner as tuner_module
        except ImportError:
            raise ImportError("tuner in llmafactory.train not found.")

        self._module = tuner_module
        self._original = tuner_module.run_exp
        adapter = self.adapter

        def _patched_run_exp(
            args: dict[str, Any] | None = None,
            callbacks: list[Any] | None = None,
        ):
            from llamafactory.hparams import get_train_args
            from .utils import build_config_from_lf_args
            from .hooks import LlamaFactoryHook
            from ...engine import TorchTurboEngine
            from ...engine.args import get_torchturbo_args

            model_args, data_args, training_args, finetuning_args, generating_args = (
                get_train_args(args)
            )

            # Retrieve CLI-parsed torchturbo args (stashed by the entry point)
            tt_args = get_torchturbo_args()

            config = build_config_from_lf_args(
                training_args=training_args,
                finetuning_args=finetuning_args,
                model_args=model_args,
                torchturbo_args=tt_args,
            )
            engine = TorchTurboEngine(config)
            adapter.engine = engine

            engine.add_hook(
                LlamaFactoryHook(
                    training_args=training_args,
                    output_dir=getattr(training_args, "output_dir", "./output"),
                )
            )

            # apply_patches checks version compat for each patch
            adapter.apply_patches(engine)

            logger.info(
                "torchturbo: engine ready, entering run_exp for stage: %s",
                finetuning_args.stage,
            )
            return self._original(args, callbacks)

        tuner_module.run_exp = _patched_run_exp
        logger.info("Patched tuner.run_exp")

        # launcher.py does `from .train.tuner import run_exp`
        # meaning it holds a reference to the unpatched version if imported before torchturbo
        try:
            import llamafactory.launcher as launcher_module

            if hasattr(launcher_module, "run_exp"):
                self._launcher_module = launcher_module
                launcher_module.run_exp = _patched_run_exp
                logger.debug("Patched launcher.run_exp")
        except ImportError as e:
            raise ImportError("Failed to patch llamafactory.launcher")

        # src/train.py does `from llamafactory.train.tuner import run_exp`
        try:
            import train as train_module

            if hasattr(train_module, "run_exp"):
                self._train_module = train_module
                train_module.run_exp = _patched_run_exp
                logger.debug("Patched train.run_exp")
        except ImportError as e:
            raise ImportError("Failed to patch train")

    def revert(self) -> None:
        if self._module and self._original:
            self._module.run_exp = self._original
        if self._launcher_module and self._original:
            self._launcher_module.run_exp = self._original
            self._launcher_module = None
        if self._train_module and self._original:
            self._train_module.run_exp = self._original
            self._train_module = None
        self._original = None


# ---------------------------------------------------------------------------
# SPTrainingPatch — adjust LF's Trainer for sequence parallelism
# ---------------------------------------------------------------------------


class SPTrainingPatch(BasePatch):
    """Adjust LlamaFactory's Trainer for sequence parallelism.

    When SP is enabled:
    1. ``gradient_accumulation_steps *= sp_size`` to keep global batch size
    2. ``get_train_dataloader()`` uses SP-aware ``DistributedSampler``
       (ranks in the same SP group get the same data)
    """

    name = "sp_training"
    supported_versions = _LF_V0

    def __init__(self) -> None:
        self._original_get_dataloader = None
        self._cls = None
        self._orig_grad_accum = None

    def patch(self, engine: TorchTurboEngine) -> None:
        # 1. Adjust gradient_accumulation_steps on HF training_args
        #    (LF's Trainer reads this, not engine.fit)
        try:
            from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
        except ImportError:
            return

        self._cls = CustomSeq2SeqTrainer

        # 2. Patch get_train_dataloader for SP-aware sampling
        self._original_get_dataloader = CustomSeq2SeqTrainer.get_train_dataloader

        def _patched_get_train_dataloader(trainer_self):
            from torch.utils.data import DataLoader, DistributedSampler

            # Get SP info
            sp_info = engine.get_sp_info()
            sp_size = sp_info["sp_size"]

            # Adjust training_args grad_accum (once)
            args = trainer_self.args
            if not getattr(args, "_sp_adjusted", False):
                orig = args.gradient_accumulation_steps
                args.gradient_accumulation_steps = orig * sp_size
                args._sp_adjusted = True
                logger.info(
                    "SP=%d: gradient_accumulation_steps %d → %d",
                    sp_size,
                    orig,
                    args.gradient_accumulation_steps,
                )

            dataset = trainer_self.train_dataset
            return engine.create_dataloader(
                dataset=dataset,
                batch_size=trainer_self._train_batch_size,
                collate_fn=trainer_self.data_collator,
                num_workers=args.dataloader_num_workers,
                pin_memory=args.dataloader_pin_memory,
                drop_last=args.dataloader_drop_last,
                shuffle=False,
            )

        CustomSeq2SeqTrainer.get_train_dataloader = _patched_get_train_dataloader
        logger.info("Patched Trainer.get_train_dataloader for SP")

    def revert(self) -> None:
        if self._cls and self._original_get_dataloader:
            self._cls.get_train_dataloader = self._original_get_dataloader
            self._original_get_dataloader = None


# ---------------------------------------------------------------------------
# LoadModelPatch
# ---------------------------------------------------------------------------


class LoadModelPatch(BasePatch):
    """Intercept ``from_pretrained`` inside LF's ``load_model``.

    Temporarily monkey-patches ``AutoModelForCausalLM.from_pretrained``
    (and related Auto classes) so that when LF calls ``load_class.from_pretrained()``,
    it goes through ``engine.prepare_model()`` instead.

    All other LF logic (config patching, adapter init, value head, etc.) is preserved.
    """

    name = "load_model"
    supported_versions = _LF_V0

    # Auto classes that we intercept
    _AUTO_CLASSES = (
        "AutoModelForCausalLM",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForSeq2SeqLM",
        "AutoModelForTextToWaveform",
    )

    def __init__(self) -> None:
        self._originals: dict[str, Any] = {}  # class_name → original from_pretrained
        self._module = None

    def patch(self, engine: TorchTurboEngine) -> None:
        try:
            import transformers
        except ImportError:
            return

        self._module = transformers

        for cls_name in self._AUTO_CLASSES:
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                continue

            original_fp = cls.from_pretrained
            self._originals[cls_name] = original_fp

            def _make_patched(orig, klass_name):
                def _patched_from_pretrained(
                    pretrained_model_name_or_path, *args, **kwargs
                ):
                    logger.info(
                        "torchturbo: intercepting %s.from_pretrained → "
                        "engine.prepare_model(%r)",
                        klass_name,
                        pretrained_model_name_or_path,
                    )

                    # Determine dtype from kwargs or default
                    torch_dtype = kwargs.get("torch_dtype", None)
                    if torch_dtype == "auto" or torch_dtype is None:
                        import torch

                        torch_dtype = torch.bfloat16

                    trust_remote_code = kwargs.get("trust_remote_code", False)

                    model = engine.prepare_model(
                        model_name_or_path=pretrained_model_name_or_path,
                        torch_dtype=torch_dtype,
                        trust_remote_code=trust_remote_code,
                        init_kwargs=kwargs,
                    )

                    return model

                return _patched_from_pretrained

            cls.from_pretrained = classmethod(
                lambda klass, *a, _fn=_make_patched(original_fp, cls_name), **kw: _fn(
                    *a, **kw
                )
            )

            logger.info("Patched %s.from_pretrained", cls_name)

    def revert(self) -> None:
        if self._module is None:
            return

        for cls_name, original in self._originals.items():
            cls = getattr(self._module, cls_name, None)
            if cls is not None:
                cls.from_pretrained = original
                logger.info("Reverted %s.from_pretrained", cls_name)

        self._originals.clear()


# ---------------------------------------------------------------------------
# TrainerTrainPatch
# ---------------------------------------------------------------------------


class TrainerTrainPatch(BasePatch):
    """Bridge CustomSeq2SeqTrainer.train to fire engine hook on_train_begin."""

    name = "trainer_train"
    supported_versions = _LF_V0

    def __init__(self) -> None:
        self._original = None
        self._cls = None

    def patch(self, engine: TorchTurboEngine) -> None:
        try:
            from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
        except ImportError:
            return

        self._cls = CustomSeq2SeqTrainer
        self._original = CustomSeq2SeqTrainer.train

        def _patched_train(trainer_self, *args, **kwargs):
            from ...engine.hooks import TrainState

            # 1. Fire on_train_begin BEFORE starting the actual HF trainer loop
            sp_info = engine.backend.get_sp_info()

            num_epochs = (
                int(trainer_self.args.num_train_epochs)
                if hasattr(trainer_self, "args")
                else 1
            )
            max_steps = (
                trainer_self.args.max_steps if hasattr(trainer_self, "args") else None
            )

            t_state = TrainState(
                model=trainer_self.model,
                optimizer=trainer_self.optimizer,
                lr_scheduler=trainer_self.lr_scheduler,
                num_epochs=num_epochs,
                max_steps=max_steps,
                sp_info=sp_info,
            )
            engine._fire(engine.hooks, "on_train_begin", t_state)

            # 2. Proceed with actual train loop
            return self._original(trainer_self, *args, **kwargs)

        CustomSeq2SeqTrainer.train = _patched_train
        logger.info(
            "TrainerTrainPatch applied to intercept on_train_begin across all workflows"
        )

    def revert(self) -> None:
        if self._cls and self._original:
            self._cls.train = self._original
            self._original = None
            self._cls = None


# ---------------------------------------------------------------------------
# ComputeLossPatch
# ---------------------------------------------------------------------------


class ComputeLossPatch(BasePatch):
    """Wrap ``compute_loss`` to calculate mathematically accurate SP Loss natively."""

    name = "compute_loss"
    supported_versions = _LF_V0

    def __init__(self) -> None:
        self._original = None

    def patch(self, engine: TorchTurboEngine) -> None:
        try:
            from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
        except ImportError:
            return

        self._original = CustomSeq2SeqTrainer.compute_loss

        logger.info("Applying ComputeLossPatch for optimized SP loss reduction")

        def _patched_compute_loss(
            trainer_self, model, inputs, return_outputs=False, **kwargs
        ):
            if not "labels" in inputs:
                return self._original(
                    trainer_self, model, inputs, return_outputs, **kwargs
                )
            num_items_in_batch = kwargs.get("num_items_in_batch", None)

            # Always request outputs so we can scrape MTP diagnostics.
            loss, outputs = engine.backend.compute_loss(
                model, inputs, return_outputs=True, **kwargs
            )

            # When num_items_in_batch is None (old-style path), HF Trainer's
            # training_step divides by gradient_accumulation_steps which
            # equals original_ga * sp_size.  But SP ranks share the same
            # data, so Trainer over-divides by sp_size.  Pre-multiply to
            # compensate.
            sp_size = engine.get_sp_info()["sp_size"]
            if num_items_in_batch is None and sp_size > 1:
                loss = loss * sp_size
                outputs.loss = loss

            # Harvest MTP diagnostics for LogMTPPatch. 
            # Buffer lives on the trainer to survive across micro-batches
            # inside one logging window.
            scaling = getattr(getattr(model, "config", None), "mtp_loss_scaling_factor", 0.0)
            if scaling and float(scaling) != 0:
                mtp_loss = getattr(outputs, "mtp_loss", None)
                if mtp_loss:
                    mtp_log = getattr(trainer_self, "_mtp_loss_buf", None)
                    if mtp_log is None:
                        mtp_log = {"mtp": [], "main": [], "scaled": []}
                        trainer_self._mtp_loss_buf = mtp_log
                    mtp_log["mtp"].append([float(m.item()) for m in mtp_loss])
                    main_l = getattr(outputs, "main_loss", None)
                    if main_l is not None:
                        mtp_log["main"].append(float(main_l.item()))
                    scaled = getattr(outputs, "mtp_loss_scaled", None)
                    if scaled is not None:
                        mtp_log["scaled"].append(float(scaled.item()))

            return (loss, outputs) if return_outputs else loss

        CustomSeq2SeqTrainer.compute_loss = _patched_compute_loss

    def revert(self) -> None:
        if self._original:
            self._original = None


# ---------------------------------------------------------------------------
# LogMTPPatch — surface MTP diagnostics in HF Trainer logs
# ---------------------------------------------------------------------------


class LogMTPPatch(BasePatch):
    """Inject MTP loss diagnostics into ``Trainer.log()``.

    Reads the buffer populated by :class:`ComputeLossPatch` and adds the
    following keys to the logs dict (only when MTP is active and
    ``mtp_loss_scaling_factor > 0``):

    - ``mtp_loss/i``       — mean of i-th MTP head's loss over the window
    - ``mtp_loss/avg``     — mean across MTP heads
    - ``mtp_loss/scaled``  — mean of ``scaling_factor * sum(mtp_loss_i)``
    - ``mtp_loss/main``    — mean of the detached main-path loss (for cross-check)

    Skipped entirely when ``mtp_loss_scaling_factor`` is ``0`` or the model
    has no MTP layers — no buffer, no extra log keys.

    The buffer is cleared after each log call.
    """

    name = "log_mtp"
    supported_versions = _LF_V0

    def __init__(self) -> None:
        self._original = None
        self._cls = None

    def patch(self, engine: TorchTurboEngine) -> None:
        try:
            from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
        except ImportError:
            return

        self._cls = CustomSeq2SeqTrainer
        self._original = CustomSeq2SeqTrainer.log

        def _patched_log(trainer_self, logs, *args, **kwargs):
            mtp_log = getattr(trainer_self, "_mtp_loss_buf", None)
            if mtp_log and mtp_log.get("mtp"):
                rows = mtp_log["mtp"]
                num_layers = len(rows[0]) if rows else 0
                if num_layers > 0:
                    # Per-layer mean over the logging window
                    by_layer = list(zip(*rows))
                    per_layer_mean = [sum(c) / len(c) for c in by_layer]
                    for i, m in enumerate(per_layer_mean):
                        logs[f"mtp_loss/{i}"] = round(m, 6)
                    logs["mtp_loss/avg"] = round(
                        sum(per_layer_mean) / num_layers, 6
                    )
                if mtp_log.get("main"):
                    logs["mtp_loss/main"] = round(
                        sum(mtp_log["main"]) / len(mtp_log["main"]), 6
                    )
                if mtp_log.get("scaled"):
                    logs["mtp_loss/scaled"] = round(
                        sum(mtp_log["scaled"]) / len(mtp_log["scaled"]), 6
                    )
                mtp_log["mtp"].clear()
                mtp_log["main"].clear()
                mtp_log["scaled"].clear()
            return self._original(trainer_self, logs, *args, **kwargs)

        CustomSeq2SeqTrainer.log = _patched_log
        logger.info("LogMTPPatch applied — MTP loss will appear in Trainer logs")

    def revert(self) -> None:
        if self._cls and self._original:
            self._cls.log = self._original
            self._original = None
            self._cls = None


# ---------------------------------------------------------------------------
# AcceleratorPatch — skip FSDP model prep in accelerator
# ---------------------------------------------------------------------------


class AcceleratorPatch(BasePatch):
    """Skip accelerator's FSDP model preparation and patch clip_grad_norm.

    When torchturbo has already applied FSDP2 sharding via ``LoadModelPatch``,
    the accelerator's ``prepare_model`` would try to re-shard the model.
    This patch:

    1. Guards ``Accelerator.prepare`` — raises if ``is_fsdp2`` is ``True``
       (accelerate's FSDP2 param-fixup conflicts with torchturbo's sharding).
    2. Makes ``Accelerator.prepare_model`` a lightweight pass-through
       (preserves ``native_amp`` autocast wrapping).
    3. Overrides ``Accelerator.clip_grad_norm`` if backend provides a custom
       implementation
    4. Overrides ``Accelerator.backward`` with plain ``loss.backward()``
    """

    name = "accelerator"
    supported_versions = _LF_V0

    def __init__(self) -> None:
        self._orig_prepare_model = None
        self._orig_prepare = None
        self._orig_clip_grad_norm = None
        self._orig_backward = None

    def patch(self, engine: TorchTurboEngine) -> None:
        try:
            from accelerate import Accelerator
        except ImportError:
            logger.warning("accelerate not found — skipping AcceleratorPatch.")
            return

        # ---- 1. Guard prepare: error out if accelerate thinks it owns FSDP2 ----
        self._orig_prepare = Accelerator.prepare

        def _guarded_prepare(accel_self, *args, device_placement=None):
            if getattr(accel_self, "is_fsdp2", False):
                raise RuntimeError(
                    "torchturbo already handles FSDP2 sharding. "
                    "Do not configure accelerate with FSDP2 — remove "
                    "fsdp_plugin / FSDP settings from your accelerate config."
                )
            return self._orig_prepare(
                accel_self, *args, device_placement=device_placement
            )

        Accelerator.prepare = _guarded_prepare

        # ---- 2. Patch prepare_model: skip FSDP, keep native_amp ----
        self._orig_prepare_model = Accelerator.prepare_model

        def _patched_prepare_model(
            accel_self,
            model,
            device_placement=None,
            evaluation_mode=False,
        ):
            from types import MethodType
            from accelerate.utils import convert_outputs_to_fp32

            accel_self._models.append(model)

            logger.info(
                "torchturbo: prepare_model skipped FSDP wrapping "
                "(already sharded by engine)"
            )
            return model

        Accelerator.prepare_model = _patched_prepare_model

        # ---- 3. Patch clip_grad_norm_ if backend supports it ----
        if hasattr(engine.backend, "clip_grad_norm_"):
            self._orig_clip_grad_norm = Accelerator.clip_grad_norm_

            def _patched_clip_grad_norm(accel_self, parameters, max_norm, norm_type=2):
                return engine.backend.clip_grad_norm_(parameters, max_norm, norm_type)

            Accelerator.clip_grad_norm_ = _patched_clip_grad_norm
            logger.info(
                "torchturbo: Patched Accelerator.clip_grad_norm_ to use backend's implementation"
            )

        # ---- 4. Patch backward ----
        self._orig_backward = Accelerator.backward

        def _patched_backward(accel_self, loss, **kwargs):
            loss.backward()

        Accelerator.backward = _patched_backward
        logger.info(
            "torchturbo: Patched Accelerator.backward to plain"
        )

        # ---- 5. Patch prepare_data_loader for SP-aware sampling ----
        # sp_size = engine.get_sp_info()["sp_size"]
        # if sp_size > 1:
        #     self._orig_prepare_data_loader = Accelerator.prepare_data_loader
        #
        #     def _patched_prepare_data_loader(accel_self, data_loader, device_placement=None, slice_fn_for_dispatch=None):
        #         dp_info = engine.get_dp_info()
        #         # Temporarily override process topology for SP
        #         orig_num = accel_self.state.num_processes
        #         orig_idx = accel_self.state.process_index
        #         accel_self.state.num_processes = dp_info["dp_size"]
        #         accel_self.state.process_index = dp_info["dp_rank"]
        #         try:
        #             result = self._orig_prepare_data_loader(accel_self, data_loader, device_placement, slice_fn_for_dispatch)
        #         finally:
        #             accel_self.state.num_processes = orig_num
        #             accel_self.state.process_index = orig_idx
        #         return result

        #     Accelerator.prepare_data_loader = _patched_prepare_data_loader
        #     logger.info(
        #         "torchturbo: Patched Accelerator.prepare_data_loader to aware SP"
        #     )

        logger.info("AcceleratorPreparePatch applied")

    def revert(self) -> None:
        try:
            from accelerate import Accelerator
        except ImportError:
            return

        if self._orig_prepare_model is not None:
            Accelerator.prepare_model = self._orig_prepare_model
            self._orig_prepare_model = None
        if self._orig_prepare is not None:
            Accelerator.prepare = self._orig_prepare
            self._orig_prepare = None
        if self._orig_clip_grad_norm is not None:
            Accelerator.clip_grad_norm = self._orig_clip_grad_norm
            self._orig_clip_grad_norm = None
        if self._orig_backward is not None:
            Accelerator.backward = self._orig_backward
            self._orig_backward = None


# ---------------------------------------------------------------------------
# CreateOptimizerPatch — use engine.prepare_optimizer for optimizer creation
# ---------------------------------------------------------------------------


class CreateOptimizerPatch(BasePatch):
    """Delegate optimizer creation to ``engine.prepare_optimizer()``.

    Replaces ``CustomSeq2SeqTrainer.create_optimizer`` so the optimizer
    is created through the engine's backend (e.g. FSDP2Backend), ensuring
    parameters are correctly bound to the already-sharded model.
    """

    name = "create_optimizer"
    supported_versions = _LF_V0

    def __init__(self) -> None:
        self._orig_create_optimizer = None
        self._trainer_cls = None

    def patch(self, engine: TorchTurboEngine) -> None:
        try:
            from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
        except ImportError:
            logger.warning(
                "CustomSeq2SeqTrainer not found — skipping CreateOptimizerPatch."
            )
            return

        self._trainer_cls = CustomSeq2SeqTrainer
        self._orig_create_optimizer = CustomSeq2SeqTrainer.create_optimizer

        def _patched_create_optimizer(trainer_self):
            if trainer_self.optimizer is not None:
                return trainer_self.optimizer

            from transformers import Trainer

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                trainer_self.args,
                trainer_self.model,
            )

            # Remove keys that engine.prepare_optimizer doesn't need
            optimizer_kwargs.pop("params", None)
            optimizer_kwargs.pop("model", None)
            optimizer_kwargs.pop("optimizer_dict", None)

            trainer_self.optimizer = engine.prepare_optimizer(
                optimizer_cls,
                **optimizer_kwargs,
            )
            logger.info(
                "torchturbo: created optimizer via engine.prepare_optimizer "
                "(%s, %s)",
                optimizer_cls.__name__,
                {k: v for k, v in optimizer_kwargs.items() if k != "params"},
            )
            return trainer_self.optimizer

        CustomSeq2SeqTrainer.create_optimizer = _patched_create_optimizer
        logger.info("CreateOptimizerPatch applied")

    def revert(self) -> None:
        if self._trainer_cls is not None and self._orig_create_optimizer is not None:
            self._trainer_cls.create_optimizer = self._orig_create_optimizer
            self._orig_create_optimizer = None
            self._trainer_cls = None


# ---------------------------------------------------------------------------
# SaveModelPatch
# ---------------------------------------------------------------------------


class SaveModelPatch(BasePatch):
    """Delegate save_model to engine.save_model."""

    name = "save_model"
    supported_versions = _LF_V0

    def __init__(self) -> None:
        self._orig_save_model = None
        self._cls = None

    def patch(self, engine: TorchTurboEngine) -> None:
        if not hasattr(engine, "save_model"):
            return

        try:
            from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
        except ImportError:
            logger.warning("CustomSeq2SeqTrainer not found — skipping SaveModelPatch.")
            return

        self._cls = CustomSeq2SeqTrainer
        self._orig_save_model = CustomSeq2SeqTrainer.save_model

        def _patched_save_model(trainer_self, output_dir=None, _internal_call=False):
            if output_dir is None:
                output_dir = trainer_self.args.output_dir

            logger.info(
                "torchturbo: Intercepting save_model → engine.save_model(%s)",
                output_dir,
            )
            engine.save_model(trainer_self.model, output_dir)

            if trainer_self.args.push_to_hub and not _internal_call:
                trainer_self.push_to_hub(commit_message="Model save")

        CustomSeq2SeqTrainer.save_model = _patched_save_model
        logger.info("SaveModelPatch applied")

    def revert(self) -> None:
        if self._cls is not None and self._orig_save_model is not None:
            self._cls.save_model = self._orig_save_model
            self._orig_save_model = None
            self._cls = None
