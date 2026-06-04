"""torchturbo engine package.

Structure::

    engine/
    ├── __init__.py          # this file — public exports
    ├── args.py              # TorchTurboArgs CLI definitions
    ├── config.py            # EngineConfig + common configs
    ├── core.py              # TorchTurboEngine facade
    ├── hooks.py             # TrainState, TrainHook
    ├── backend_registry.py  # register_backend / get_backend
    └── backends/
        ├── __init__.py      # imports all built-in backends
        ├── base.py          # BaseBackend ABC
        └── fsdp2/
            ├── __init__.py
            ├── config.py    # FSDPConfig, ShardingStrategy
            └── backend.py   # FSDP2Backend
"""

# Import backends first so they auto-register
from . import backends  # noqa: F401
from .args import TorchTurboArgs, get_torchturbo_args, parse_torchturbo_args, set_torchturbo_args
from .backend_registry import get_available_backends, get_backend, register_backend
from .backends.fsdp2 import FSDP2Backend, FSDPConfig
from .backends.base import BaseBackend
from .config import (
    ActivationCheckpointingConfig,
    CheckpointConfig,
    CommConfig,
    DistributedBackend,
    EngineConfig,
    MixedPrecisionPolicy,
)
from .core import TorchTurboEngine
from . import peft  # noqa: F401
from .hooks import TrainHook, TrainState


__all__ = [
    # CLI args
    "TorchTurboArgs",
    "get_torchturbo_args",
    "parse_torchturbo_args",
    "set_torchturbo_args",
    # Common config
    "ActivationCheckpointingConfig",
    "CheckpointConfig",
    "CommConfig",
    "DistributedBackend",
    "EngineConfig",
    "MixedPrecisionPolicy",
    # Engine
    "BaseBackend",
    "FSDP2Backend",
    "FSDPConfig",
    # Training loop
    "TrainHook",
    "TrainState",
    # Registry
    "get_available_backends",
    "get_backend",
    "register_backend",
]
