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

import dataclasses
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from omegaconf import OmegaConf

from ..logging import get_logger
from ..ops.base import OpPolicy
from .args import TorchTurboArgs, get_arg_specs, string_to_bool

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums (common)
# ---------------------------------------------------------------------------


class DistributedBackend(str, Enum):
    """Supported distributed backends."""

    FSDP2 = "fsdp2"
    # Future:
    # MEGATRON = "megatron"


class ParameterGroup(str, Enum):
    """Semantic model parameter groups understood by engine policies."""

    ALL = "all"
    VISION_TOWER = "vision_tower"
    MULTI_MODAL_PROJECTOR = "multi_modal_projector"
    LANGUAGE_MODEL = "language_model"


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
class OpsConfig:
    """Op dispatch configuration."""

    policy: OpPolicy = OpPolicy.AUTO

    def __post_init__(self) -> None:
        if not isinstance(self.policy, OpPolicy):
            raise TypeError("OpsConfig.policy must be an OpPolicy.")


@dataclass
class ActivationCheckpointingConfig:
    """Activation checkpointing (gradient checkpointing) configuration.

    Example::

        ActivationCheckpointingConfig(enabled=True)
    """

    enabled: bool = True
    use_reentrant: bool = False


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


@dataclass
class MTPConfig:
    """Multi-Token-Prediction configuration."""

    enabled: bool = False
    loss_scaling_factor: float = 0.1

    def __post_init__(self) -> None:
        if isinstance(self.enabled, str):
            self.enabled = string_to_bool(self.enabled)
        else:
            self.enabled = bool(self.enabled)
        self.loss_scaling_factor = float(self.loss_scaling_factor)


@dataclass
class MoEConfig:
    """MoE training configuration."""

    aux_loss_coeff: float = 0.0


@dataclass
class DiffusionConfig:
    """Configuration for block-diffusion (BD) training.

    Consumed by ``FSDP2Backend.compute_diffusion_loss`` (CAP weight) and by
    the engine's BD dataloader (per-sample noise range). Gated by the model's
    ``is_diffusion_model`` class attribute — these fields are inert for
    non-diffusion models.
    """

    cap_lambda: float = 0.0
    """CAP/DPARALLEL confidence-loss weight (entropy on correct tokens).
    0 = disabled, behavior unchanged. Recommended: 2.0."""

    bd_mask_low: float = 0.3
    """Per-sample noise ratio sigma lower bound (``sigma ~ U[low, high]``)."""

    bd_mask_high: float = 0.8
    """Per-sample noise ratio sigma upper bound."""

    bd_block_size: int = 32
    """Block size (``BLOCK_SIZE``) for the block-diagonal attention mask; the
    unit ``cutoff_len`` is ceil-aligned to. Must be a positive integer (default
    ``32``, dFactory/LLaDA2 default). Only affects the BD dataset/loss path."""

    def __post_init__(self) -> None:
        self.cap_lambda = float(self.cap_lambda)
        self.bd_mask_low = float(self.bd_mask_low)
        self.bd_mask_high = float(self.bd_mask_high)
        self.bd_block_size = int(self.bd_block_size)
        if self.bd_block_size < 1:
            raise ValueError(
                f"DiffusionConfig.bd_block_size must be a positive integer, "
                f"got {self.bd_block_size}."
            )


@dataclass(frozen=True)
class FreezePolicy:
    """Engine config for freezing semantic model parameter groups."""

    parameter_groups: tuple[ParameterGroup, ...] = ()

    @property
    def enabled(self) -> bool:
        return bool(self.parameter_groups)

    @classmethod
    def from_list(
        cls,
        groups: Iterable[ParameterGroup | str] | ParameterGroup | str | None,
    ) -> FreezePolicy:
        if groups is None:
            return cls()
        if isinstance(groups, (ParameterGroup, str)):
            groups = (groups,)
        normalized = []
        seen = set()
        for group in groups:
            parameter_group = cls._coerce_group(group)
            if parameter_group in seen:
                continue
            normalized.append(parameter_group)
            seen.add(parameter_group)
        return cls(tuple(normalized))

    @staticmethod
    def _coerce_group(group: ParameterGroup | str) -> ParameterGroup:
        if isinstance(group, ParameterGroup):
            return group
        return ParameterGroup(group)


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
            activation_checkpointing=ActivationCheckpointingConfig(enabled=True),
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

    ops: OpsConfig = field(default_factory=OpsConfig)
    """Op dispatch settings."""

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

    mtp: MTPConfig = field(default_factory=MTPConfig)
    """Multi-Token-Prediction settings."""

    moe: MoEConfig = field(default_factory=MoEConfig)
    """MoE training settings."""

    freeze_policy: FreezePolicy = field(default_factory=FreezePolicy)
    """Policy for freezing semantic model parameter groups."""

    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    """Block-diffusion training settings (only active for ``is_diffusion_model`` models)."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Catch-all for user-defined settings."""

    def __post_init__(self) -> None:
        """Create missing nested configuration objects."""
        if self.backend_config is None:
            self.backend_config = self._default_backend_config()
        if self.ops is None:
            self.ops = OpsConfig()

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

        if "op_policy" in d:
            raise ValueError("op_policy is no longer supported; use ops.policy.")

        # Parallel Plan (parse sp, ep, tp sizes)
        parallel_d = d.get("parallel", {})

        sp_size = parallel_d.get("sp_size", d.get("sp_size", 1))
        ep_size = parallel_d.get("ep_size", d.get("ep_size", 1))

        from torchturbo.parallel import FSDPParallelPlan

        parallel_plan = FSDPParallelPlan(
            sp_size=sp_size,
            ep_size=ep_size,
        )

        # Backend-specific config
        backend_config = d.get("backend_config", None)
        if isinstance(backend_config, dict) and backend == DistributedBackend.FSDP2:
            from .backends.fsdp2.config import FSDPConfig

            backend_config = FSDPConfig(**backend_config)

        if backend_config is None and backend == DistributedBackend.FSDP2:
            from .backends.fsdp2.config import FSDPConfig

            fsdp_d = d.get("fsdp", {})
            if not fsdp_d:
                fsdp_d = {
                    k: d[k]
                    for k in (
                        "sharding_strategy",
                        "reshard_after_forward",
                        "prefetch_strategy",
                        "prefetch_direction",
                        "forward_prefetch_depth",
                        "backward_prefetch_depth",
                        "cpu_offload",
                        "pin_memory",
                    )
                    if k in d
                }
            # FSDPConfig.__post_init__ coerces str values to the right enums.
            backend_config = FSDPConfig(**fsdp_d) if fsdp_d else FSDPConfig()

        # Activation checkpointing
        ac_d = d.get("activation_checkpointing", {})
        ac = ActivationCheckpointingConfig(**ac_d)

        # Communication
        comm_d = d.get("comm", {})
        if not comm_d:
            if "comm_overlap" in d:
                comm_d["overlap_comm_compute"] = d["comm_overlap"]
        comm = CommConfig(**comm_d)

        # Ops
        ops_d = d.get("ops", {})
        if isinstance(ops_d, OpsConfig):
            ops = ops_d
        else:
            if ops_d is None:
                ops_d = {}
            if not isinstance(ops_d, dict):
                raise TypeError("ops must be a dict or OpsConfig.")
            ops_d = dict(ops_d)
            if "policy" in ops_d:
                ops_d["policy"] = OpPolicy(ops_d["policy"])
            ops = OpsConfig(**ops_d) if ops_d else OpsConfig()

        # Checkpoint
        ckpt_d = d.get("checkpoint", {})
        if not ckpt_d:
            ckpt_d = {
                k: d[k] for k in ("dcp_path", "hf_model_path", "save_format") if k in d
            }
        ckpt = CheckpointConfig(**ckpt_d)

        # MTP
        mtp_config = cls._parse_mtp_config(d)

        # MoE
        moe_config = cls._parse_moe_config(d.get("moe", {}))

        # Freeze policy
        freeze_policy = cls._parse_freeze_policy(d.get("freeze_policy", {}))

        # Extra
        known_keys = {
            "backend",
            "mixed_precision",
            "parallel",
            "fsdp",
            "backend_config",
            "activation_checkpointing",
            "comm",
            "ops",
            "checkpoint",
            "dp_size",
            "tp_size",
            "ep_size",
            "pp_size",
            "sp_size",
            "sharding_strategy",
            "reshard_after_forward",
            "prefetch_strategy",
            "prefetch_direction",
            "forward_prefetch_depth",
            "backward_prefetch_depth",
            "cpu_offload",
            "pin_memory",
            "sync_module_states",
            "comm_overlap",
            "dcp_path",
            "hf_model_path",
            "save_format",
            "mtp",
            "moe",
            "enable_mtp",
            "mtp_loss_scaling_factor",
            "freeze_policy",
        }
        extra = {k: v for k, v in d.items() if k not in known_keys}

        return cls(
            backend=backend,
            parallel_plan=parallel_plan,
            mixed_precision=mp,
            ops=ops,
            backend_config=backend_config,
            activation_checkpointing=ac,
            comm=comm,
            checkpoint=ckpt,
            mtp=mtp_config,
            moe=moe_config,
            freeze_policy=freeze_policy,
            extra=extra,
        )

    @staticmethod
    def _parse_mtp_config(d: dict[str, Any]) -> MTPConfig:
        mtp_d = d.get("mtp", {})
        if isinstance(mtp_d, MTPConfig):
            return mtp_d
        if mtp_d is None:
            mtp_d = {}
        if not isinstance(mtp_d, dict):
            raise TypeError("mtp must be a dict or MTPConfig.")

        mtp_d = dict(mtp_d)
        if "enable_mtp" in d:
            mtp_d["enabled"] = d["enable_mtp"]
        if "mtp_loss_scaling_factor" in d:
            mtp_d["loss_scaling_factor"] = d["mtp_loss_scaling_factor"]
        if mtp_d.get("loss_scaling_factor") is None:
            mtp_d["loss_scaling_factor"] = 0.1
        return MTPConfig(**mtp_d)

    @staticmethod
    def _parse_moe_config(value: Any) -> MoEConfig:
        if isinstance(value, MoEConfig):
            return value
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise TypeError("moe must be a dict or MoEConfig.")
        return MoEConfig(**value)

    @staticmethod
    def _parse_freeze_policy(value: Any) -> FreezePolicy:
        if isinstance(value, FreezePolicy):
            return value
        if value is None or value == {}:
            return FreezePolicy()
        if not isinstance(value, dict):
            raise TypeError("freeze_policy must be a dict or FreezePolicy.")

        allowed_keys = {"parameter_groups"}
        unknown_keys = set(value) - allowed_keys
        if unknown_keys:
            keys = ", ".join(sorted(unknown_keys))
            raise ValueError(f"Unknown freeze_policy field(s): {keys}.")
        if "parameter_groups" not in value:
            raise ValueError("freeze_policy requires 'parameter_groups'.")

        return FreezePolicy.from_list(value["parameter_groups"])

    def to_dict(self) -> dict[str, Any]:
        """Serialize back to a dict (for logging, etc.)."""

        data = {}
        for field_info in dataclasses.fields(self):
            value = getattr(self, field_info.name)
            if dataclasses.is_dataclass(value):
                value = dataclasses.asdict(value)
            data[field_info.name] = value
        return data


# ---------------------------------------------------------------------------
# EngineConfig builder — single mapping from ``TorchTurboArgs`` → EngineConfig
# ---------------------------------------------------------------------------


def build_from_args(
    torchturbo_args: TorchTurboArgs | None = None,
) -> EngineConfig:
    """Assemble an :class:`EngineConfig` purely from a :class:`TorchTurboArgs`.

    Every slice of the engine config is reconstructed from the parsed
    ``--tt_*`` flags (the single source of truth for turbo-owned settings).
    Both adapters (llamafactory_v0, msswift) — and future ones — fold any
    adapter-resolved bits (mixed-precision seed from the adapter's bf16/fp16,
    checkpoint paths from the adapter's model/output_dir, CPU-offload
    reconciliation, gradient-checkpointing takeover) onto the
    :class:`TorchTurboArgs` they pass in, so this function needs only that one
    argument. Adding a new ``--tt_*`` flag touches only ``engine/args.py`` +
    this builder, and every adapter picks it up.

    The final config is built once from a merged dict with four layers:
    ``TorchTurboArgs`` < ``TORCH_TURBO_*`` env < YAML/JSON file < explicit
    ``--tt_*`` CLI flags.

    ``FSDPConfig`` / ``FSDPParallelPlan`` are imported lazily (mirroring
    :meth:`EngineConfig.__post_init__`) to avoid an import cycle: they live in
    ``torchturbo.engine.backends.fsdp2.config`` / ``torchturbo.parallel``,
    which import nothing from this module.

    Parameters
    ----------
    torchturbo_args:
        Parsed ``--tt_*`` args (the single source of truth for turbo flags).
        ``None`` is treated as all-defaults.
    """
    tt = torchturbo_args or TorchTurboArgs()
    arg_specs = get_arg_specs()

    # Layer 1: TorchTurboArgs (adapter-seeded values are the base layer).
    fields = OmegaConf.create()
    for arg_spec in arg_specs:
        if arg_spec.args_name is None or arg_spec.config_path is None:
            continue
        value = getattr(tt, arg_spec.args_name)
        if value is not None:
            OmegaConf.update(fields, arg_spec.config_path, value, force_add=True)
    # Defaults without a --tt_* flag (tt values above win via merge).
    fields = OmegaConf.merge(
        OmegaConf.create(
            {
                # --- Backend ---
                "backend": "fsdp2",
                # --- Mixed precision (adapter may have seeded tt.mixed_precision) ---
                "mixed_precision": "fp32",
                # --- Parallel plan (sp / ep) ---
                "parallel_plan": {"sp_size": 1, "ep_size": 1},
                # --- Backend config (prefetch from tt; cpu_offload resolved by adapter) ---
                "backend_config": {"reshard_after_forward": True},
            }
        ),
        fields,
    )

    # Layer 2: TORCH_TURBO_* environment variables — applies to ALL adapters
    # and direct usage.
    for arg_spec in arg_specs:
        if arg_spec.env_name is None or arg_spec.config_path is None:
            continue
        value = os.environ.get(arg_spec.env_name)
        if value is None:
            continue
        if arg_spec.type is not None:
            value = arg_spec.type(value)
        OmegaConf.update(fields, arg_spec.config_path, value, force_add=True)

    # Layer 3: YAML/JSON config file.
    config_from_file = OmegaConf.create()
    if tt.config_path is not None:
        config_from_file = OmegaConf.load(tt.config_path)
        if "engine_config" in config_from_file:
            config_from_file = config_from_file["engine_config"]

    # Layer 4: explicitly supplied --tt_* CLI flags.
    config_from_cli = OmegaConf.create()
    for destination, value in tt.config_overrides.items():
        for arg_spec in arg_specs:
            if (
                arg_spec.cli_name is None
                or arg_spec.cli_name.lstrip("-") != destination
                or arg_spec.config_path is None
            ):
                continue
            OmegaConf.update(
                config_from_cli, arg_spec.config_path, value, force_add=True
            )
            break

    # Merge all layers into one dict, then build the config once.
    fields = OmegaConf.to_container(
        OmegaConf.merge(fields, config_from_file, config_from_cli), resolve=True
    )

    # Parallel plan (parse FQN -> FSDPPlanSpec mappings from the file).
    from torchturbo.parallel import FSDPParallelPlan, FSDPPlanSpec

    parallel_d = fields.get("parallel_plan") or {}
    spec_fields = [f.name for f in dataclasses.fields(FSDPPlanSpec)]
    parallel_plan = FSDPParallelPlan(
        plan={
            fqn: FSDPPlanSpec(**{f: spec_d.get(f) for f in spec_fields})
            for fqn, spec_d in (parallel_d.get("plan") or {}).items()
        },
        sp_size=parallel_d.get("sp_size", 1),
        ep_size=parallel_d.get("ep_size", 1),
    )

    ac_d = dict(fields.get("activation_checkpointing") or {})
    if "enabled" in ac_d:
        ac_d["enabled"] = string_to_bool(ac_d["enabled"])
    if "use_reentrant" in ac_d:
        ac_d["use_reentrant"] = string_to_bool(ac_d["use_reentrant"])
    comm_d = dict(fields.get("comm") or {})
    if "overlap_comm_compute" in comm_d:
        comm_d["overlap_comm_compute"] = string_to_bool(
            comm_d["overlap_comm_compute"]
        )

    config = EngineConfig(
        backend=DistributedBackend(fields["backend"]),
        mixed_precision=MixedPrecisionPolicy(fields["mixed_precision"]),
        parallel_plan=parallel_plan,
        ops=_build_ops_config(fields),
        backend_config=_build_fsdp_config(fields),
        activation_checkpointing=ActivationCheckpointingConfig(**ac_d),
        comm=CommConfig(**comm_d),
        checkpoint=_build_checkpoint_config(fields),
        mtp=_build_mtp_config(fields),
        moe=_build_moe_config(fields),
        freeze_policy=_build_freeze_policy(fields),
        diffusion=_build_diffusion_config(fields),
        extra={},
    )

    logger.info_rank0(
        "Built EngineConfig from TorchTurboArgs: backend=%s, mp=%s, sp=%d, ep=%d",
        config.backend.value,
        config.mixed_precision.value,
        config.parallel_plan.sp_size,
        config.parallel_plan.ep_size,
    )
    return config


def _build_fsdp_config(fields: Mapping[str, Any]) -> Any:
    """FSDPConfig from the merged config dict."""
    from torchturbo.engine.backends.fsdp2.config import FSDPConfig, ShardingStrategy

    fsdp_d = fields.get("backend_config") or {}
    return FSDPConfig(
        sharding_strategy=fsdp_d.get(
            "sharding_strategy", ShardingStrategy.FULL_SHARD
        ),
        reshard_after_forward=string_to_bool(
            fsdp_d.get("reshard_after_forward", True)
        ),
        cpu_offload=string_to_bool(fsdp_d.get("cpu_offload", False)),
        pin_memory=string_to_bool(fsdp_d.get("pin_memory", True)),
        prefetch_strategy=fsdp_d.get("prefetch_strategy", "auto"),
        prefetch_direction=fsdp_d.get("prefetch_direction", "forward_backward"),
        forward_prefetch_depth=fsdp_d.get("forward_prefetch_depth", 1),
        backward_prefetch_depth=fsdp_d.get("backward_prefetch_depth", 1),
    )


def _build_checkpoint_config(fields: Mapping[str, Any]) -> CheckpointConfig:
    """CheckpointConfig from the merged config dict."""
    ckpt_d = fields.get("checkpoint") or {}
    return CheckpointConfig(
        hf_model_path=ckpt_d.get("hf_model_path"),
        dcp_path=ckpt_d.get("dcp_path") or None,
        save_format=ckpt_d.get("save_format", "dcp"),
    )


def _build_ops_config(fields: Mapping[str, Any]) -> OpsConfig:
    """OpsConfig from the merged config dict."""
    policy = (fields.get("ops") or {}).get("policy")
    if policy is None:
        return OpsConfig()
    return OpsConfig(policy=OpPolicy(str(policy).lower()))


def _build_mtp_config(fields: Mapping[str, Any]) -> MTPConfig:
    """MTPConfig from the merged config dict (nested or flat keys)."""
    mtp_d = dict(fields.get("mtp") or {})
    if "enable_mtp" in fields:
        mtp_d["enabled"] = fields["enable_mtp"]
    if "mtp_loss_scaling_factor" in fields:
        mtp_d["loss_scaling_factor"] = fields["mtp_loss_scaling_factor"]
    loss_scaling_factor = mtp_d.get("loss_scaling_factor")
    return MTPConfig(
        enabled=string_to_bool(mtp_d.get("enabled", False)),
        loss_scaling_factor=float(
            loss_scaling_factor if loss_scaling_factor is not None else 0.1
        ),
    )


def _build_moe_config(fields: Mapping[str, Any]) -> MoEConfig:
    """MoEConfig from the merged config dict."""
    moe_d = fields.get("moe") or {}
    return MoEConfig(aux_loss_coeff=float(moe_d.get("aux_loss_coeff", 0.0)))


def _build_freeze_policy(fields: Mapping[str, Any]) -> FreezePolicy:
    """FreezePolicy from the merged config dict."""
    freeze_d = fields.get("freeze_policy") or {}
    groups = [
        name
        for name, enabled in (
            ("vision_tower", string_to_bool(freeze_d.get("vision_tower", False))),
            (
                "multi_modal_projector",
                string_to_bool(freeze_d.get("multi_modal_projector", False)),
            ),
            ("language_model", string_to_bool(freeze_d.get("language_model", False))),
        )
        if enabled
    ]
    return FreezePolicy.from_list(groups)


def _build_diffusion_config(fields: Mapping[str, Any]) -> DiffusionConfig:
    """DiffusionConfig from the merged config dict."""
    diffusion_d = fields.get("diffusion") or {}
    return DiffusionConfig(
        cap_lambda=float(diffusion_d.get("cap_lambda") or 0.0),
        bd_mask_low=float(diffusion_d.get("bd_mask_low", 0.3)),
        bd_mask_high=float(diffusion_d.get("bd_mask_high", 0.8)),
        bd_block_size=int(diffusion_d.get("bd_block_size", 32)),
    )
