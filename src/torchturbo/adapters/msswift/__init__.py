"""torchturbo ↔ ms-swift adapter — native registration, no swift monkey-patching.

Auto-activates on ``import torchturbo`` when ms-swift is installed.  Activation
populates swift's own extension registries (``MODEL_MAPPING`` per turbo model
with :class:`TurboModelLoader`; ``TrainerFactory.TRAINER_MAPPING`` per task
type with the turbo trainer).  A SFT run is then driven by swift's native
pipeline via the ``--external_plugins`` entry (``plugin.py``), which swaps in
:class:`TurboSwiftSft` so the engine owns model construction + training.
"""

from __future__ import annotations

from ...logging import get_logger
from ...registry import BaseAdapter, register_adapter
from .compat import get_swift_version, is_swift_supported

logger = get_logger(__name__)

__all__ = ["MSSwiftAdapter"]


@register_adapter("msswift", priority=20)
class MSSwiftAdapter(BaseAdapter):
    """Wires torchturbo into ms-swift through swift's native extension points."""

    framework_name = "msswift"

    def __init__(self) -> None:
        super().__init__()
        self._active = False

    def get_framework_version(self) -> str:
        return get_swift_version()

    def detect(self) -> bool:
        """Activate only when this process intends to run ms-swift.
        """
        return is_swift_supported()

    def activate(self) -> None:
        """Register turbo models and trainers into swift's registries."""
        if self._active:
            return
        from .model_registry import register_turbo_models_to_swift, register_turbo_trainers_to_swift

        register_turbo_models_to_swift()
        register_turbo_trainers_to_swift()

        # Install the VLM dataset adapter: it auto-detects multimodal datasets at
        # construction time (presence of an ``images``/``videos`` column) and only
        # then pads media-less samples with fake media, so every rank drives the
        # media tower (FSDP+expert-parallel needs every rank to post the
        # per-layer media-tower all-gathers). Pure-text datasets stay untouched.
        # No CLI flag — the signal comes from the dataset swift just built. The
        # current path pads with an image for qwen3.5-moe; future multimodal
        # towers reuse the same pattern (see vlm_dataset_adapter).
        from .vlm_dataset_adapter import install_vlm_dataset_adapter

        install_vlm_dataset_adapter()

        self._active = True
        logger.info_rank0("ms-swift adapter activated (turbo models/trainers registered)")

    def deactivate(self) -> None:
        from . import engine_registry
        from .vlm_dataset_adapter import uninstall_vlm_dataset_adapter

        uninstall_vlm_dataset_adapter()
        engine_registry.clear()
        self._active = False
        logger.info_rank0("ms-swift adapter deactivated")
