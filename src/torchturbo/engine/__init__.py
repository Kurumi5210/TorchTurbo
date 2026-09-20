"""torchturbo engine package.

Structure::

    engine/
    ├── __init__.py          # this file — public exports
    ├── args.py              # TorchTurboArgs CLI definitions
    ├── config.py            # EngineConfig + common configs
    ├── core.py              # TorchTurboEngine facade
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
from .args import (
    TorchTurboArgs,
    get_torchturbo_args,
    parse_torchturbo_args,
    set_torchturbo_args,
    string_to_bool,
)
from .backend_registry import get_available_backends, get_backend, register_backend
from .backends.fsdp2 import (
    CPUOffloadConfig,
    CPUOffloadMode,
    FSDP2Backend,
    FSDPConfig,
)
from .backends.base import BaseBackend
from .config import (
    ActivationCheckpointingConfig,
    CheckpointConfig,
    CommConfig,
    DistributedBackend,
    EngineConfig,
    FreezePolicy,
    LossChunkConfig,
    MixedPrecisionPolicy,
    MoEConfig,
    MTPConfig,
    OpsConfig,
    ParameterGroup,
    build_from_args,
)
from .core import TorchTurboEngine
from .distributed import init_process_group_if_needed
from .metrics import MetricTracker
from .plugins import (
    DatasetPlugin,
    PackingPlugin,
    Plugin,
    PluginConsumer,
    PluginKind,
    PluginManager,
    register_plugin,
)


__all__ = [
    # Shared EngineConfig builder (lf + msswift + ...)
    "build_from_args",
    # CLI args
    "TorchTurboArgs",
    "get_torchturbo_args",
    "parse_torchturbo_args",
    "set_torchturbo_args",
    "string_to_bool",
    # Common config
    "ActivationCheckpointingConfig",
    "CheckpointConfig",
    "CommConfig",
    "CPUOffloadConfig",
    "CPUOffloadMode",
    "DistributedBackend",
    "EngineConfig",
    "FreezePolicy",
    "LossChunkConfig",
    "MixedPrecisionPolicy",
    "MetricTracker",
    "MoEConfig",
    "MTPConfig",
    "OpsConfig",
    "ParameterGroup",
    # Engine
    "BaseBackend",
    "FSDP2Backend",
    "FSDPConfig",
    # Registry
    "get_available_backends",
    "get_backend",
    "register_backend",
    # Distributed
    "init_process_group_if_needed",
    # Plugins
    "DatasetPlugin",
    "PackingPlugin",
    "Plugin",
    "PluginConsumer",
    "PluginKind",
    "PluginManager",
    "register_plugin",
]
