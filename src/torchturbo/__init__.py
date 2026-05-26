"""torchturbo — one-line import for optimized distributed training.

Usage::

    import torchturbo  # auto-detects and activates the right adapter

That's it.  If LlamaFactory v0 is active, torchturbo will automatically
patch ``run_sft`` to use the engine with optimal FSDP settings.

Extra engine config via env vars::

    TORCH_TURBO_COMM_OVERLAP=1 TORCH_TURBO_SELECTIVE_AC_LAYERS=0,4,8 \\
        llamafactory-cli train config.yaml
"""

from __future__ import annotations

from .logging import get_logger
from .detector import FrameworkDetector
from .engine import (
    BaseBackend,
    EngineConfig,
    FSDP2Backend,
    FSDPConfig,
    MixedPrecisionPolicy,
    TorchTurboEngine,
    get_available_backends,
    register_backend,
)
from .registry import (
    BaseAdapter,
    BasePatch,
    get_adapter,
    get_all_adapters,
    register_adapter,
)


__version__ = "0.1.0"
__all__ = [
    "BaseAdapter",
    "BasePatch",
    "BaseBackend",
    "EngineConfig",
    "FSDP2Backend",
    "FrameworkDetector",
    "FSDPConfig",
    "MixedPrecisionPolicy",
    "TorchTurboEngine",
    "detect",
    "detect_all",
    "get_available_backends",
    "register_adapter",
    "register_backend",
]

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Auto-discovery and activation
# ---------------------------------------------------------------------------


def _discover_builtin_adapters() -> None:
    """Import built-in adapter packages so they auto-register."""
    import importlib
    import pkgutil

    from . import adapters as adapters_pkg

    for _importer, module_name, _is_pkg in pkgutil.walk_packages(
        adapters_pkg.__path__,
        prefix=adapters_pkg.__name__ + ".",
    ):
        try:
            importlib.import_module(module_name)
        except Exception as e:
            logger.warning("Failed to import adapter %s", module_name, exc_info=True)


def detect() -> str | None:
    """Detect which training framework is active."""
    return FrameworkDetector.detect()


def detect_all() -> list[str]:
    """Detect all active training frameworks."""
    return FrameworkDetector.detect_all()


def _auto_activate() -> None:
    """Auto-detect and activate the first matching adapter.

    Called once at import time.  This is what makes the one-line import work.
    Set ``TORCH_TURBO_DISABLE=1`` to skip auto-activation.
    """
    import os

    if os.environ.get("TORCH_TURBO_DISABLE", "").lower() in ("1", "true", "yes"):
        logger.info("torchturbo disabled via TORCH_TURBO_DISABLE env var.")
        return

    framework = detect()
    if framework is None:
        logger.debug("No framework detected.")
        return

    adapter = get_adapter(framework)
    if adapter is None:
        return

    logger.info("Auto-activating adapter for %r", framework)
    adapter.activate()


# Run on import
_discover_builtin_adapters()
_auto_activate()

# Register built-in models
from . import models as _models_pkg  # noqa: F401 - trggers _register_builtins()
