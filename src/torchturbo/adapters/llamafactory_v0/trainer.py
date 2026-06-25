"""TorchTurbo trainer wrappers for LlamaFactory v0."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...logging import get_logger


logger = get_logger(__name__)


class TorchTurboTrainerBase:
    """Shared trainer behavior injected above LlamaFactory trainer classes.

    ``make_torchturbo_trainer`` creates a temporary class with this base
    first in the MRO, followed by the concrete LlamaFactory trainer.  Native
    LlamaFactory behavior remains available through ``super()``.
    """

    engine: Any = None
    stage: str | None = None
    loss_mode: str = "native"
    torchturbo_wrapped: bool = False
    base_trainer_cls: type | None = None
    custom_optimizer_options: tuple[str, ...] = (
        "use_galore",
        "use_apollo",
        "use_badam",
        "use_adam_mini",
        "use_muon",
    )

    def __init__(self, *args, **kwargs):
        self._raise_if_unsupported_trainer_options(
            kwargs.get("finetuning_args", args[0] if args else None)
        )
        self._mm_processor = kwargs.get("processor", None)
        super().__init__(*args, **kwargs)
        self._maybe_upgrade_collator_for_mllm_pt()

    def _maybe_upgrade_collator_for_mllm_pt(self) -> None:
        """Swap to MultiModalDataCollatorForSeq2Seq for multimodal PT datasets."""
        if self.stage != "pt":
            return

        dataset = self.train_dataset
        if dataset is None or not hasattr(dataset, "features"):
            return

        if not any(k in dataset.features for k in ("images", "videos", "audios")):
            return

        from transformers import DataCollatorForLanguageModeling
        if not isinstance(self.data_collator, DataCollatorForLanguageModeling):
            return

        from llamafactory.data.collator import MultiModalDataCollatorForSeq2Seq
        from llamafactory.extras.constants import IGNORE_INDEX

        # The existing data_collator was constructed with a tokenizer in run_pt.
        # MultiModalDataCollatorForSeq2Seq needs a template — retrieve it from
        # the engine where RunExpPatch stashed the template name.
        engine = self._engine()
        template_name = getattr(engine, "_lf_template_name", None)
        if not template_name:
            logger.warning("No template name on engine; skipping MLLM PT collator upgrade.")
            return

        from llamafactory.data.template import TEMPLATES
        template = TEMPLATES.get(template_name)
        if template is None:
            logger.warning(
                "Template %r not in registry; skipping MLLM PT collator upgrade.",
                template_name,
            )
            return

        self.data_collator = MultiModalDataCollatorForSeq2Seq(
            template=template,
            model=self.model,
            pad_to_multiple_of=8,
            label_pad_token_id=IGNORE_INDEX,
            tokenizer=self.tokenizer,
            processor=self._mm_processor,
        )
        logger.info("Upgraded PT collator to MultiModalDataCollatorForSeq2Seq")

    def _raise_if_unsupported_trainer_options(self, finetuning_args: Any) -> None:
        unsupported = self._get_unsupported_trainer_options(finetuning_args)
        if unsupported:
            raise NotImplementedError(
                "TorchTurbo LlamaFactory trainer does not support "
                f"{', '.join(unsupported)} yet."
            )

    def _get_unsupported_trainer_options(self, finetuning_args: Any) -> list[str]:
        return (
            self._get_enabled_custom_optimizer_options(finetuning_args)
            + self._get_enabled_custom_loss_options(finetuning_args)
        )

    def _get_enabled_custom_optimizer_options(self, finetuning_args: Any) -> list[str]:
        enabled: list[str] = []
        for name in self.custom_optimizer_options:
            if bool(getattr(finetuning_args, name, None)):
                enabled.append(name)
        return enabled

    def _get_enabled_custom_loss_options(self, finetuning_args: Any) -> list[str]:
        if self.stage == "sft" and getattr(finetuning_args, "use_dft_loss", False):
            return ["use_dft_loss"]
        return []

    def _engine(self) -> Any:
        engine = getattr(self, "engine", None)
        if engine is None:
            raise RuntimeError("TorchTurbo trainer is missing its engine binding.")
        return engine

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        engine = self._engine()

        from transformers import Trainer

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args,
            self.model,
        )
        optimizer_kwargs.pop("params", None)
        optimizer_kwargs.pop("model", None)
        optimizer_kwargs.pop("optimizer_dict", None)

        self.optimizer = engine.prepare_optimizer(optimizer_cls, **optimizer_kwargs)
        logger.info(
            "torchturbo: created optimizer via engine.prepare_optimizer "
            "(%s, %s)",
            optimizer_cls.__name__,
            {k: v for k, v in optimizer_kwargs.items() if k != "params"},
        )
        return self.optimizer

    def save_model(self, output_dir=None, _internal_call=False):
        engine = self._engine()
        if output_dir is None:
            output_dir = self.args.output_dir

        save_path = str(Path(output_dir) / "huggingface")
        logger.info("torchturbo: saving model via engine.save_model(%s)", save_path)
        engine.save_model(self.model, save_path)

        if getattr(self.args, "push_to_hub", False) and not _internal_call:
            self.push_to_hub(commit_message="Model save")

    def get_train_dataloader(self):
        if self.loss_mode != "causal_lm":
            return super().get_train_dataloader()

        engine = self._engine()
        args = self.args
        sp_size = engine.get_sp_info()["sp_size"]
        if sp_size <= 1:
            return super().get_train_dataloader()

        if not getattr(args, "_torchturbo_sp_adjusted", False):
            orig = args.gradient_accumulation_steps
            args.gradient_accumulation_steps = orig * sp_size
            args._torchturbo_sp_adjusted = True
            logger.info(
                "SP=%d: gradient_accumulation_steps %d -> %d",
                sp_size,
                orig,
                args.gradient_accumulation_steps,
            )

        finetuning_args = getattr(self, "finetuning_args", None)
        disable_shuffling = bool(
            getattr(finetuning_args, "disable_shuffling", False)
        )

        return engine.create_dataloader(
            dataset=self.train_dataset,
            batch_size=self._train_batch_size,
            collate_fn=self.data_collator,
            num_workers=args.dataloader_num_workers,
            pin_memory=args.dataloader_pin_memory,
            drop_last=args.dataloader_drop_last,
            shuffle=not disable_shuffling,
        )

    def train(self, *args, **kwargs):
        from ...engine.hooks import TrainState
        from ...engine.args import get_torchturbo_args
        from ...engine.profiling import start_profiling, stop_profiling

        engine = self._engine()
        sp_info = engine.backend.get_sp_info()
        train_state = TrainState(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            num_epochs=int(getattr(self.args, "num_train_epochs", 1)),
            max_steps=getattr(self.args, "max_steps", None),
            sp_info=sp_info,
        )
        engine._fire(engine.hooks, "on_train_begin", train_state)

        tt_args = get_torchturbo_args()
        if tt_args is not None and tt_args.profiling:
            start_profiling(self.args.output_dir)

        result = super().train(*args, **kwargs)

        if tt_args is not None and tt_args.profiling:
            stop_profiling()

        return result

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.loss_mode == "causal_lm":
            return self._compute_causal_lm_loss(
                model, inputs, return_outputs=return_outputs, **kwargs
            )

        if self.stage == "dpo" and self._engine().get_sp_info()["sp_size"] > 1:
            raise NotImplementedError(
                "TorchTurbo DPO support currently keeps LlamaFactory's native "
                "preference loss and does not support sequence parallelism yet."
            )

        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            **kwargs,
        )

    def log(self, logs, *args, **kwargs):
        mtp_log = getattr(self, "_mtp_loss_buf", None)
        if mtp_log and mtp_log.get("mtp"):
            rows = mtp_log["mtp"]
            num_layers = len(rows[0]) if rows else 0
            if num_layers > 0:
                by_layer = list(zip(*rows))
                per_layer_mean = [sum(c) / len(c) for c in by_layer]
                for i, mean_value in enumerate(per_layer_mean):
                    logs[f"mtp_loss/{i}"] = round(mean_value, 6)
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

        return super().log(logs, *args, **kwargs)

    def _compute_causal_lm_loss(self, model, inputs, return_outputs=False, **kwargs):
        if "labels" not in inputs:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                **kwargs,
            )

        engine = self._engine()
        num_items_in_batch = kwargs.get("num_items_in_batch", None)

        loss, outputs = engine.backend.compute_loss(
            model,
            inputs,
            return_outputs=True,
            **kwargs,
        )

        sp_size = engine.get_sp_info()["sp_size"]
        if num_items_in_batch is None and sp_size > 1:
            loss = loss * sp_size
            outputs.loss = loss

        self._record_mtp_outputs(outputs)

        from ...engine.profiling import step_profiling
        step_profiling()

        return (loss, outputs) if return_outputs else loss

    def _record_mtp_outputs(self, outputs) -> None:
        mtp_loss = getattr(outputs, "mtp_loss", None)
        if not mtp_loss:
            return

        mtp_log = getattr(self, "_mtp_loss_buf", None)
        if mtp_log is None:
            mtp_log = {"mtp": [], "main": [], "scaled": []}
            self._mtp_loss_buf = mtp_log

        mtp_log["mtp"].append([float(m.item()) for m in mtp_loss])
        main_loss = getattr(outputs, "main_loss", None)
        if main_loss is not None:
            mtp_log["main"].append(float(main_loss.item()))
        scaled_loss = getattr(outputs, "mtp_loss_scaled", None)
        if scaled_loss is not None:
            mtp_log["scaled"].append(float(scaled_loss.item()))


def make_torchturbo_trainer(
    base_cls: type,
    *,
    engine: Any,
    stage: str,
    loss_mode: str,
) -> type:
    """Create a stage/engine-bound trainer subclass."""

    base_cls = unwrap_torchturbo_trainer_class(base_cls)
    trainer_engine = engine
    trainer_stage = stage
    trainer_loss_mode = loss_mode

    class TorchTurboTrainer(TorchTurboTrainerBase, base_cls):
        engine = trainer_engine
        stage = trainer_stage
        loss_mode = trainer_loss_mode
        torchturbo_wrapped = True
        base_trainer_cls = base_cls

    name = f"TorchTurbo{stage.upper()}{base_cls.__name__}"
    TorchTurboTrainer.__name__ = name
    TorchTurboTrainer.__qualname__ = name
    TorchTurboTrainer.__module__ = base_cls.__module__

    return TorchTurboTrainer


def unwrap_torchturbo_trainer_class(cls: type) -> type:
    """Return the original LlamaFactory trainer behind a TT wrapper."""

    if not getattr(cls, "torchturbo_wrapped", False):
        return cls

    base_cls = getattr(cls, "base_trainer_cls", None)
    if isinstance(base_cls, type):
        return base_cls

    mro = getattr(cls, "__mro__", ())
    if len(mro) > 2:
        return mro[2]

    return cls

