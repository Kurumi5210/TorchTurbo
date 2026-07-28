"""Per-stage wiring for the ms-swift adapter — data-driven, extendable.

A *stage* is one ms-swift training pipeline (SFT today; DPO/KTO/ORPO/GRPO via
``SwiftRLHF`` later). Each stage is described by a :class:`SwiftStageSpec`
that ties together the three swift surfaces turbo overrides for that stage:

* the **pipeline module global** to swap (``swift.pipelines.train.sft.SwiftSft``
  → turbo's :class:`TurboSwiftSft`) — driven from :mod:`plugin`;
* the **trainer dotted paths** to register into
  ``TrainerFactory.TRAINER_MAPPING`` — driven from :mod:`model_registry`.

Adding a stage is therefore local: define its turbo pipeline / trainer /
args classes, then ``register_stage(...)`` here — :mod:`plugin` and
:mod:`model_registry` pick it up without further edits.

The ``turbo_pipeline_factory`` is a **closure that imports lazily**, so
importing this module never pulls in swift (keeps ``import torchturbo`` swift-
free). The args class is NOT on the spec — it lives on the turbo pipeline class
as ``args_class`` (e.g. ``TurboSwiftSft.args_class = TurboSftArguments``), which
swift reads automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple


@dataclass(frozen=True)
class SwiftStageSpec:
    """Wiring for one ms-swift training stage (SFT, DPO, ...)."""

    name: str
    #: swift module holding the pipeline class global to swap, e.g.
    #: ``"swift.pipelines.train.sft"``.
    pipeline_module: str
    #: attribute name on that module, e.g. ``"SwiftSft"``.
    pipeline_attr: str
    #: zero-arg callable returning the turbo pipeline class to swap in. Imports
    #: lazily so this module stays swift-free at import time.
    turbo_pipeline_factory: Callable[[], type]
    #: ``TrainerFactory.TRAINER_MAPPING`` entries this stage contributes:
    #: ``{key: dotted_path_of_turbo_trainer}`` (e.g. ``{"causal_lm": "..."}``,
    #: or for RLHF ``{"dpo": "...", "kto": "..."}``).
    turbo_trainers: Dict[str, str]


TURBO_SWIFT_STAGES: List[SwiftStageSpec] = []


def register_stage(spec: SwiftStageSpec) -> None:
    """Append a stage spec (idempotent on ``spec.name``)."""
    if any(s.name == spec.name for s in TURBO_SWIFT_STAGES):
        return
    TURBO_SWIFT_STAGES.append(spec)


def iter_stages() -> Tuple[SwiftStageSpec, ...]:
    """Snapshot of registered stages (tuple so callers can iterate safely)."""
    return tuple(TURBO_SWIFT_STAGES)


def _sft_pipeline_factory() -> type:
    """Lazy: build/return the turbo SFT pipeline class (imports swift here)."""
    from .pipeline import get_turbo_swift_sft

    return get_turbo_swift_sft()


#: The SFT stage. ``causal_lm`` is swift's task_type for causal-LM SFT; the
#: trainer is the engine-backed :class:`TurboSeq2SeqTrainer`.
SFT_STAGE = SwiftStageSpec(
    name="sft",
    pipeline_module="swift.pipelines.train.sft",
    pipeline_attr="SwiftSft",
    turbo_pipeline_factory=_sft_pipeline_factory,
    turbo_trainers={
        "causal_lm": "torchturbo.adapters.msswift.trainer_factory.TurboSeq2SeqTrainer",
    },
)

register_stage(SFT_STAGE)