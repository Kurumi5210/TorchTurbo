"""Utility functions for integrating LlamaFactory v0 with torchturbo."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...engine import EngineConfig
from ...engine.args import TorchTurboArgs
from ...engine.config import (
    ActivationCheckpointingConfig,
    CheckpointConfig,
    CommConfig,
    DistributedBackend,
    MixedPrecisionPolicy,
)
from ...engine.backends.fsdp2.config import FSDPConfig, ShardingStrategy
from ...engine.hooks import TrainState
from ...logging import get_logger
from torchturbo.parallel import FSDPParallelPlan


logger = get_logger(__name__)


def build_config_from_lf_args(
    training_args: Any = None,
    finetuning_args: Any = None,
    model_args: Any = None,
    torchturbo_args: TorchTurboArgs | None = None,
    overrides: dict[str, Any] | None = None,
) -> EngineConfig:
    """Build an ``EngineConfig`` from LlamaFactory args.

    Priority (lowest → highest): defaults → LF args mapping → ``torchturbo_args``
    → env vars(via ``EngineConfig.__post_init__``)  → ``overrides`` dict.

    Parameters
    ----------
    training_args:
        LlamaFactory's ``TrainingArguments``.
    finetuning_args:
        LlamaFactory's ``FinetuningArguments``.
    model_args:
        LlamaFactory's ``ModelArguments``.
    torchturbo_args:
        Parsed ``--tt_*`` CLI arguments.  Applied directly during
        config construction so they override LF-derived values.
    overrides:
        Optional dict of overrides (highest priority, applied after
        construction).

    Returns
    -------
    A fully configured ``EngineConfig``.
    """
    config = _build_config_from_args(
        training_args, finetuning_args, model_args, torchturbo_args
    )
    # Note: env vars are already applied by EngineConfig.__post_init__

    # Apply explicit overrides (highest priority)
    if overrides:
        override_config = EngineConfig.from_dict(overrides)
        for field_name in (
            "parallel",
            "backend_config",
            "activation_checkpointing",
            "comm",
            "checkpoint",
        ):
            override_val = getattr(override_config, field_name)
            if override_val is not None:
                setattr(config, field_name, override_val)
        if overrides.get("backend"):
            config.backend = override_config.backend
        if overrides.get("mixed_precision"):
            config.mixed_precision = override_config.mixed_precision

    logger.info(
        "Built EngineConfig from LlamaFactory args: backend=%s, mp=%s",
        config.backend.value,
        config.mixed_precision.value,
    )
    return config


def _build_config_from_args(
    training_args: Any,
    finetuning_args: Any,
    model_args: Any,
    torchturbo_args: TorchTurboArgs | None = None,
) -> EngineConfig:
    """Map LlamaFactory args and torchturbo CLI args to EngineConfig."""

    tt = torchturbo_args  # shorthand

    # --- Backend ---
    backend = DistributedBackend.FSDP2
    if tt is not None and tt.backend is not None:
        backend = DistributedBackend(tt.backend)

    # --- Mixed precision ---
    mixed_precision = MixedPrecisionPolicy.FP32
    if training_args is not None:
        if getattr(training_args, "bf16", False):
            mixed_precision = MixedPrecisionPolicy.BF16
        elif getattr(training_args, "fp16", False):
            mixed_precision = MixedPrecisionPolicy.FP16
        if getattr(training_args, "fp8", False):
            mixed_precision = MixedPrecisionPolicy.FP8
    if tt is not None and tt.mixed_precision is not None:
        mixed_precision = MixedPrecisionPolicy(tt.mixed_precision)

    # --- FSDP config (from HF's fsdp / fsdp_config) ---
    fsdp_config = _build_fsdp_config(training_args)

    # --- Activation checkpointing ---
    ac_enabled = False
    if training_args is not None:
        ac_enabled = getattr(training_args, "gradient_checkpointing", False)

    ac_config = ActivationCheckpointingConfig(
        enabled=ac_enabled,
        use_reentrant=False,  # FSDP2 requires non-reentrant
    )

    # --- Communication ---
    comm_config = CommConfig(
        overlap_comm_compute=False,  # default off, user can override
    )

    # --- Checkpoint paths ---
    checkpoint_config = CheckpointConfig(
        hf_model_path=(
            getattr(model_args, "model_name_or_path", None) if model_args else None
        ),
    )
    if training_args is not None:
        output_dir = getattr(training_args, "output_dir", None)
        if output_dir:
            checkpoint_config.dcp_path = output_dir

    # --- Parallel plan ---
    sp_size = tt.sp_size if (tt is not None and tt.sp_size is not None) else 1
    ep_size = tt.ep_size if (tt is not None and tt.ep_size is not None) else 1
    parallel_plan = FSDPParallelPlan(sp_size=sp_size, ep_size=ep_size)

    # --- Extra: carry through MTP scaling so backend / patches can read it ---
    extra: dict[str, Any] = {}
    if tt is not None:
        extra["enable_mtp"] = bool(tt.enable_mtp)
        if tt.mtp_loss_scaling_factor is not None:
            extra["mtp_loss_scaling_factor"] = float(tt.mtp_loss_scaling_factor)

    return EngineConfig(
        backend=backend,
        parallel_plan=parallel_plan,
        mixed_precision=mixed_precision,
        backend_config=fsdp_config,
        activation_checkpointing=ac_config,
        comm=comm_config,
        checkpoint=checkpoint_config,
        extra=extra,
    )


def _build_fsdp_config(training_args: Any) -> FSDPConfig:
    """Extract FSDP settings from HF training_args.fsdp + fsdp_config."""
    if training_args is None:
        return FSDPConfig()

    fsdp_options = getattr(training_args, "fsdp", [])
    fsdp_config_dict = getattr(training_args, "fsdp_config", {}) or {}

    # Sharding strategy
    sharding = ShardingStrategy.FULL_SHARD

    # CPU offload
    cpu_offload = False
    if isinstance(fsdp_options, (list, tuple)):
        cpu_offload = "offload" in fsdp_options

    # Forward/backward prefetch from fsdp_config dict
    forward_prefetch = fsdp_config_dict.get("forward_prefetch", False)

    return FSDPConfig(
        sharding_strategy=sharding,
        reshard_after_forward=True,
        forward_prefetch=forward_prefetch,
        cpu_offload=cpu_offload,
        pin_memory=not cpu_offload,
    )


def compute_sft_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    state: TrainState,
) -> torch.Tensor:
    """Standard SFT (supervised fine-tuning) loss.

    Computes cross-entropy on the shifted labels, compatible with
    HuggingFace-style models.
    """
    device = next(model.parameters()).device
    model_inputs = {
        k: v.to(device, non_blocking=True) for k, v in batch.items() if k != "labels"
    }
    labels = batch["labels"].to(device, non_blocking=True)

    outputs = model(**model_inputs)
    logits = outputs.logits.float()

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss
