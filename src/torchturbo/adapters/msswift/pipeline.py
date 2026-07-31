"""Turbo-backed swift training pipelines.

:class:`TurboEnginePipelineMixin` owns the pieces every turbo takeover stage
shares: building + stashing the turbo engine after args are parsed but before
the model loads, and bypassing swift's ``TunerMixin.prepare_model`` (LoRA /
freeze are owned by the engine). :class:`TurboSwiftSft` mixes it ahead of
swift's :class:`SwiftSft`. A future RLHF stage would mix it ahead of
:class:`SwiftRLHF` and reuse both methods unchanged.

swift's ``sft_main`` closes over the module-global ``SwiftSft``; the
``--external_plugins`` entry (:mod:`plugin`) / the ``run_swift.py`` launcher
swaps in :class:`TurboSwiftSft` (driven by the stage specs in :mod:`stages`) so
the native ``swift sft`` CLI runs the turbo pipeline unchanged.

The turbo pipeline classes are **module-level** on purpose (not built inside a
factory closure). Their base chain (``SwiftSft`` -> ``SwiftPipeline(ABC, ...)``)
uses ``ABCMeta``; a closure-local class would be pickled *by value* by
``dill`` (``save_type``) in swift's multiprocess ``dataset.map`` — the object
swift sends to worker processes — tripping on ``_abc._abc_data``. Module-level
lets ``dill`` pickle it *by reference* (just ``(module, __qualname__)``). See
``args.py`` for the same reasoning on ``TurboSftArguments``.

The swift imports below are heavy; this module is imported only via the
msswift plugin (turbo-swift code path), so they stay off plain
``import torchturbo``.
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

from swift.pipelines.base import SwiftPipeline
from swift.pipelines.train.sft import SwiftSft

from .args import TurboSftArguments
from . import engine_registry
from .utils import build_config_from_swift_args, build_lora_config_from_swift_args
from ...logging import LogLevel, get_logger, logging_configure

logger = get_logger(__name__)


class TurboEnginePipelineMixin:
    """Engine construction + tuner bypass shared by every turbo takeover stage.

    Mixed in AHEAD of the swift pipeline base (e.g.
    ``class TurboSwiftSft(TurboEnginePipelineMixin, SwiftSft)`` so its methods
    shadow swift's). Provides:

    * :meth:`_build_turbo_engine` — build the :class:`TorchTurboEngine` from the
      parsed args, apply LoRA, stash it (per stage) in :mod:`engine_registry`
      for the model loader. Runs BEFORE the model loads.
    * :meth:`prepare_model` — short-circuit swift's ``TunerMixin.prepare_model``
      (LoRA / freeze are applied inside the engine).
    """

    def _build_turbo_engine(self) -> None:
        """Build the engine and stash it for :class:`TurboModelLoader`.

        Runs before ``_prepare_model_tokenizer`` — at this point ``self.model``
        does not exist yet, so anything needing the built model belongs after
        ``_prepare_template``.
        """
        from ...engine import TorchTurboEngine
        from ...engine.args import get_torchturbo_args

        tt_args = get_torchturbo_args()
        logging_configure(tt_args.log_level if tt_args is not None else LogLevel.INFO)

        config = build_config_from_swift_args(self.args)
        engine = TorchTurboEngine(config)
        if getattr(self.args, "tuner_type", "full") == "lora":
            engine.peft_config = build_lora_config_from_swift_args(self.args)
        engine_registry.set_engine(engine, stage=getattr(self.args, "rlhf_type", None) or "sft")
        # Stash the fixed packing length from the pipeline's FULL args: swift
        # hands the trainer a stripped args view without packing_length /
        # max_length, so the trainer can't read it from self.args. The trainer
        # pads each batch to this length so turbo's constant-shape packing path
        # runs unchanged (Triton/FA shape cache; no unpad roundtrip).
        engine_registry.set_packing_length(
            getattr(self.args, "packing_length", None) or getattr(self.args, "max_length", None)
        )
        logger.info_rank0(
            "TurboSwiftSft: engine ready (backend=%s, mp=%s)",
            config.backend.value, config.mixed_precision.value,
        )

    def prepare_model(self, args=None, model=None, template=None, train_dataset=None, **kwargs):
        # The engine already built + sharded + applied freeze/LoRA inside
        # TurboModelLoader; skip swift's TunerMixin.prepare_model.
        return model if model is not None else self.model


class TurboSwiftSft(TurboEnginePipelineMixin, SwiftSft):  # type: ignore[misc]
    """SwiftSft that builds the turbo engine up front.

    Module-level (not a factory closure) so ``dill`` pickles it by reference in
    swift's multiprocess ``dataset.map`` — a closure-local class pickles by
    value and trips on the ABCMeta internals (``_abc._abc_data``) inherited via
    ``SwiftPipeline(ABC)``.
    """

    args_class = TurboSftArguments

    def __init__(self, args: Optional[Union[List[str], Any]] = None) -> None:
        # Parse args without loading the model (skip SwiftSft.__init__).
        SwiftPipeline.__init__(self, args)
        self.train_msg: dict = {}

        self._build_turbo_engine()

        # swift's normal prepare sequence; SP is off on the args (forced by
        # TurboSwiftArgsMixin), so swift's own ``sequence_parallel.prepare`` is
        # skipped.
        self._prepare_model_tokenizer()
        self._prepare_template()
        self._prepare_flash_ckpt()


def get_turbo_swift_sft():
    """Return the module-level ``TurboSwiftSft`` (kept for API compat)."""
    return TurboSwiftSft
