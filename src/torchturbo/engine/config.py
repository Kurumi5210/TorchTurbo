"""Structured configuration dataclasses for torchturbo engine.

This module contains **common** (cross-backend) configuration.  Backend-specific
configs live in their own packages, e.g. ``backends.fsdp2.config.FSDPConfig``.

Usage::

    from torchturbo.engine.config import EngineConfig
    from torchturbo.parallel import ParallelPlan, FSDPParallelPlan

    config = EngineConfig(
        mixed_precision="bf16",
    )
    engine = TorchTurboEngine(config)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Env var overrides — applies to ALL adapters and direct usage
# ---------------------------------------------------------------------------

_ENV_MAP: dict[str, str] = {
    "TORCH_TURBO_BACKEND": "backend",
    "TORCH_TURBO_MIXED_PRECISION": "mixed_precision",
    "TORCH_TURBO_SELECTIVE_AC_LAYERS": "activation_checkpointing.selective_layers",
    "TORCH_TURBO_RESHARD_AFTER_FWD": "backend_config.reshard_after_forward",
    "TORCH_TURBO_FORWARD_PREFETCH": "backend_config.forward_prefetch",
    "TORCH_TURBO_CPU_OFFLOAD": "backend_config.cpu_offload",
    "TORCH_TURBO_SP_SIZE": "parallel",
    "TORCH_TURBO_EP_SIZE": "parallel",
}


def _set_field(obj: Any, field_name: str, value: str) -> None:
    """Set a field on a config object, coercing the string value to the right type."""
    if field_name == "selective_layers":
        setattr(obj, field_name, [int(x.strip()) for x in value.split(",")])
    elif value.lower() in ("1", "true", "yes"):
        setattr(obj, field_name, True)
    elif value.lower() in ("0", "false", "no"):
        setattr(obj, field_name, False)
    else:
        setattr(obj, field_name, value)


# ---------------------------------------------------------------------------
# Enums (common)
# ---------------------------------------------------------------------------


class DistributedBackend(str, Enum):
    """Supported distributed backends."""

    FSDP2 = "fsdp2"
    # Future:
    # MEGATRON = "megatron"


# ---------------------------------------------------------------------------
# Shared config classes — re-exported for backward compatibility
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Training optimizations (common)
# ---------------------------------------------------------------------------


class MixedPrecisionPolicy(str, Enum):
    """Mixed precision dtype policy."""

    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"
    FP8 = "fp8"


@dataclass
class MixedPrecisionConfig:
    """Mixed precision configuration."""

    policy: str = "bf16"
    param_dtype: str | None = None
    reduce_dtype: str | None = None


@dataclass
class ActivationCheckpointingConfig:
    """Activation checkpointing (gradient checkpointing) configuration.

    Example::

        ActivationCheckpointingConfig(enabled=True, selective_layers=[0, 4, 8])
    """

    enabled: bool = True
    selective_layers: list[int] | None = None
    use_reentrant: bool = False
    cpu_offload: bool = False


# ---------------------------------------------------------------------------
# Communication optimization (common)
# ---------------------------------------------------------------------------


@dataclass
class CommConfig:
    """Communication optimization settings.

    Attributes
    ----------
    overlap_comm_compute:
        Overlap allgather / reduce-scatter with compute.
    bucket_cap_mb:
        Size of allreduce buckets (MB) for gradient sync.
    reduce_dtype:
        Dtype for gradient reduction.  ``None`` = same as param dtype.
    """

    overlap_comm_compute: bool = False
    bucket_cap_mb: float = 25.0
    reduce_dtype: MixedPrecisionPolicy | None = None


# ---------------------------------------------------------------------------
# Checkpoint (common)
# ---------------------------------------------------------------------------


@dataclass
class CheckpointConfig:
    """Model checkpoint loading / saving settings.

    Attributes
    ----------
    dcp_path:
        Path to a distributed checkpoint (DCP) for efficient sharded load.
    hf_model_path:
        HuggingFace model path (local or repo id) for weight loading.
    save_format:
        Format for saving: ``"dcp"`` (distributed) or ``"hf"`` (HuggingFace).
    """

    dcp_path: str | None = None
    hf_model_path: str | None = None
    save_format: str = "dcp"


# ---------------------------------------------------------------------------
# Top-level engine config
# ---------------------------------------------------------------------------


@dataclass
class EngineConfig:
    """Top-level configuration for :class:`TorchTurboEngine`.

    Backend-specific settings are stored in ``backend_config`` as an ``Any``
    type — each backend defines its own config dataclass (e.g.
    ``FSDPConfig`` from ``backends.fsdp2.config``).

    Example::

        from torchturbo.engine.backends.fsdp2.config import FSDPConfig

        config = EngineConfig(
            backend=DistributedBackend.FSDP2,
            mixed_precision=MixedPrecisionPolicy.BF16,
            backend_config=FSDPConfig(reshard_after_forward=True),
            activation_checkpointing=ActivationCheckpointingConfig(
                selective_layers=[0, 4, 8, 12],
            ),
            comm=CommConfig(overlap_comm_compute=True),
        )
    """

    backend: DistributedBackend = DistributedBackend.FSDP2
    """Distributed backend."""

    parallel_plan: Any = None
    """Layer-specific parallel plan (``ParallelPlan``).
    Maps FQN patterns to mesh-dim placements for EP, SP, etc."""

    mixed_precision: MixedPrecisionPolicy = MixedPrecisionPolicy.BF16
    """Global mixed precision policy."""

    backend_config: Any = None
    """Backend-specific config (e.g. ``FSDPConfig``, ``MegatronConfig``).
    If ``None``, the backend uses its own defaults."""

    activation_checkpointing: ActivationCheckpointingConfig = field(
        default_factory=ActivationCheckpointingConfig
    )
    """Activation checkpointing settings."""

    comm: CommConfig = field(default_factory=CommConfig)
    """Communication optimization settings."""

    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    """Checkpoint loading / saving settings."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Catch-all for user-defined settings."""

    def __post_init__(self) -> None:
        """Auto-create default backend_config, then apply env var overrides."""
        if self.backend_config is None:
            self.backend_config = self._default_backend_config()
        self.apply_env_overrides()

    def _set_top_level(self, field_name: str, value: str) -> None:
        """Set a top-level config field from a string value."""
        if field_name == "backend":
            self.backend = DistributedBackend(value)
        elif field_name == "mixed_precision":
            self.mixed_precision = MixedPrecisionPolicy(value)
        elif field_name == "parallel" and "=" in value:
            # Handles TORCH_TURBO_SP_SIZE, etc manually in apply_env_overrides
            pass

    def apply_env_overrides(self) -> None:
        """Apply ``TORCH_TURBO_*`` env vars to this config."""
        for env_key, config_path in _ENV_MAP.items():
            value = os.environ.get(env_key)
            if value is None:
                continue

            if env_key in ("TORCH_TURBO_SP_SIZE", "TORCH_TURBO_EP_SIZE"):
                if self.parallel_plan is None:
                    from torchturbo.parallel import FSDPParallelPlan

                    self.parallel_plan = FSDPParallelPlan()
                attr_name = env_key.replace("TORCH_TURBO_", "").lower()
                setattr(self.parallel_plan, attr_name, int(value))
                continue

            parts = config_path.split(".")
            if len(parts) == 1:
                self._set_top_level(parts[0], value)
            else:
                obj = getattr(self, parts[0], None)
                if obj is not None:
                    _set_field(obj, parts[1], value)

    def _default_backend_config(self) -> Any:
        """Create the default config for the selected backend."""
        if self.backend == DistributedBackend.FSDP2:
            from .backends.fsdp2.config import FSDPConfig

            return FSDPConfig()
        return {}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EngineConfig:
        """Create an ``EngineConfig`` from a flat or nested dict.

        Supports both flat keys (``{"backend": "fsdp2", "dp_size": 4}``) and
        nested keys (``{"parallel": {"dp_size": 4}}``).
        """
        backend = DistributedBackend(d.get("backend", "fsdp2"))
        mp = MixedPrecisionPolicy(d.get("mixed_precision", "bf16"))

        # Parallel Plan (parse sp, ep, tp sizes)
        parallel_d = d.get("parallel", {})
        parallel_plan = None

        sp_size = parallel_d.get("sp_size", d.get("sp_size", 1))
        ep_size = parallel_d.get("ep_size", d.get("ep_size", 1))

        if sp_size > 1 or ep_size > 1 or parallel_d:
            from torchturbo.parallel import FSDPParallelPlan

            parallel_plan = FSDPParallelPlan(
                sp_size=sp_size,
                ep_size=ep_size,
            )

        # Backend-specific config
        backend_config = d.get("backend_config", None)
        if backend_config is None and backend == DistributedBackend.FSDP2:
            from .backends.fsdp2.config import FSDPConfig, ShardingStrategy

            fsdp_d = d.get("fsdp", {})
            if not fsdp_d:
                fsdp_d = {
                    k: d[k]
                    for k in (
                        "sharding_strategy",
                        "reshard_after_forward",
                        "forward_prefetch",
                        "cpu_offload",
                        "pin_memory",
                    )
                    if k in d
                }
            if "sharding_strategy" in fsdp_d and isinstance(
                fsdp_d["sharding_strategy"], str
            ):
                fsdp_d["sharding_strategy"] = ShardingStrategy(
                    fsdp_d["sharding_strategy"]
                )
            backend_config = FSDPConfig(**fsdp_d) if fsdp_d else FSDPConfig()

        # Activation checkpointing
        ac_d = d.get("activation_checkpointing", {})
        if not ac_d:
            ac_d = {}
            if "selective_ac_layers" in d:
                ac_d["selective_layers"] = d["selective_ac_layers"]
            if "use_reentrant" in d:
                ac_d["use_reentrant"] = d["use_reentrant"]
        ac = ActivationCheckpointingConfig(**ac_d)

        # Communication
        comm_d = d.get("comm", {})
        if not comm_d:
            if "comm_overlap" in d:
                comm_d["overlap_comm_compute"] = d["comm_overlap"]
        comm = CommConfig(**comm_d)

        # Checkpoint
        ckpt_d = d.get("checkpoint", {})
        if not ckpt_d:
            ckpt_d = {
                k: d[k] for k in ("dcp_path", "hf_model_path", "save_format") if k in d
            }
        ckpt = CheckpointConfig(**ckpt_d)

        # Extra
        known_keys = {
            "backend",
            "mixed_precision",
            "parallel",
            "fsdp",
            "backend_config",
            "activation_checkpointing",
            "comm",
            "checkpoint",
            "dp_size",
            "tp_size",
            "ep_size",
            "pp_size",
            "sp_size",
            "sharding_strategy",
            "reshard_after_forward",
            "forward_prefetch",
            "backward_prefetch",
            "cpu_offload",
            "pin_memory",
            "sync_module_states",
            "selective_ac_layers",
            "use_reentrant",
            "comm_overlap",
            "dcp_path",
            "hf_model_path",
            "save_format",
        }
        extra = {k: v for k, v in d.items() if k not in known_keys}

        return cls(
            backend=backend,
            parallel_plan=parallel_plan,
            mixed_precision=mp,
            backend_config=backend_config,
            activation_checkpointing=ac,
            comm=comm,
            checkpoint=ckpt,
            extra=extra,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize back to a dict (for logging, etc.)."""
        from dataclasses import asdict

        return asdict(self)
