"""Process-level slot for the active turbo engine during a swift run.

The pipeline (:class:`TurboSwiftSft`) builds the engine before swift loads the
model; swift's model factory then calls :class:`TurboModelLoader.get_model`,
which has no other way to reach that engine through swift's call chain.  This
module bridges the two.  Bounded to a single training process.
"""

from __future__ import annotations

from typing import Optional


_engine = None
_stage: Optional[str] = None  # 'sft' | 'pt' | 'dpo' | ... (pipeline class turbo_stage)


def set_engine(engine, stage: Optional[str] = None) -> None:
    global _engine
    _engine = engine
    if stage is not None:
        global _stage
        _stage = stage


def get_engine():
    if _engine is None:
        raise RuntimeError(
            "No active turbo engine -- a turbo pipeline (TurboSwiftSft / "
            "TurboSwiftPretrain) must build one before swift loads the model."
        )
    return _engine


def get_engine_or_none():
    return _engine


def get_stage() -> Optional[str]:
    return _stage


def clear() -> None:
    global _engine, _stage
    _engine = None
    _stage = None