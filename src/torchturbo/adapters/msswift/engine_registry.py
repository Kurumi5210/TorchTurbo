"""Process-level slot for the active turbo engine during a swift run.

The pipeline (:class:`TurboSwiftSft`) builds the engine before swift loads the
model; swift's model factory then calls :class:`TurboModelLoader.get_model`,
which has no other way to reach that engine through swift's call chain.  This
module bridges the two.  Bounded to a single training process.
"""

from __future__ import annotations

from typing import Optional


_engine = None
_stage: Optional[str] = None  # 'sft' | 'dpo' | ...
# Configured packing length (the fixed length turbo's packing path wants),
# stashed from the pipeline's FULL args. swift hands the trainer a stripped args
# view without swift's data/template fields (packing_length / max_length), so
# the trainer reads turbo's fixed packing length from here.
_packing_length: Optional[int] = None


def set_engine(engine, stage: Optional[str] = None) -> None:
    global _engine
    _engine = engine
    if stage is not None:
        global _stage
        _stage = stage


def get_engine():
    if _engine is None:
        raise RuntimeError(
            "No active turbo engine — TurboSwiftSft must build one before "
            "swift loads the model."
        )
    return _engine


def get_engine_or_none():
    return _engine


def get_stage() -> Optional[str]:
    return _stage


def set_packing_length(length: Optional[int]) -> None:
    """Stash the configured packing length from the pipeline's full args.

    swift passes the trainer a stripped args view (no ``packing_length`` /
    ``max_length``), so the trainer reads turbo's fixed packing length from
    here.
    """
    global _packing_length
    _packing_length = length


def get_packing_length() -> Optional[int]:
    return _packing_length


def clear() -> None:
    global _engine, _stage, _packing_length
    _engine = None
    _stage = None
    _packing_length = None