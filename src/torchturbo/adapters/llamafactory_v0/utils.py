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
    _raise_if_unsupported_finetuning_type(finetuning_args)

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


def _raise_if_unsupported_finetuning_type(finetuning_args: Any) -> None:
    if getattr(finetuning_args, "finetuning_type", None) == "oft":
        raise NotImplementedError(
            "TorchTurbo LlamaFactory adapter does not support "
            "finetuning_type=oft yet."
        )


def _get_unsupported_lora_options(
    finetuning_args: Any,
    model_args: Any | None = None,
) -> list[str]:
    if getattr(finetuning_args, "finetuning_type", None) != "lora":
        return []

    checks = {
        "adapter_name_or_path": getattr(model_args, "adapter_name_or_path", None)
        is not None,
        "additional_target": getattr(finetuning_args, "additional_target", None)
        is not None,
        "loraplus_lr_ratio": getattr(finetuning_args, "loraplus_lr_ratio", None)
        is not None,
        "pissa_init": bool(getattr(finetuning_args, "pissa_init", False)),
        "pissa_convert": bool(getattr(finetuning_args, "pissa_convert", False)),
        "create_new_adapter": bool(getattr(finetuning_args, "create_new_adapter", False)),
        "use_llama_pro": bool(getattr(finetuning_args, "use_llama_pro", False)),
    }
    return [name for name, enabled in checks.items() if enabled]


def build_lora_config_from_lf_args(
    finetuning_args: Any,
    model_args: Any | None = None,
) -> Any | None:
    """Build a ``peft.LoraConfig`` from LlamaFactory's ``FinetuningArguments``.

    Maps supported LlamaFactory LoRA fields (``lora_rank``, ``lora_alpha``,
    ``lora_target``, ``lora_dropout``, ``use_rslora``, ``use_dora``) to a
    :class:`peft.LoraConfig` and registers the
    :class:`~torchturbo.models.modules.moe.experts.GroupedMLP` custom module
    handler so that MoE expert layers are included in LoRA.

    Call :func:`strip_lora_args` before re-entering LlamaFactory so that
    LlamaFactory does not apply LoRA itself.  TorchTurbo passes this config
    into ``engine.prepare_model`` and owns the LoRA lifecycle from there.

    Returns ``None`` if PEFT is not installed.
    """
    unsupported = _get_unsupported_lora_options(finetuning_args, model_args)
    if unsupported:
        raise NotImplementedError(
            "TorchTurbo LlamaFactory adapter does not support "
            f"LoRA options {', '.join(unsupported)} yet."
        )

    try:
        from peft import LoraConfig, TaskType
    except ImportError:
        logger.warning("peft not installed — cannot build LoraConfig from LF args.")
        return None

    # Map LlamaFactory finetuning args to LoraConfig fields.
    # LlamaFactory uses: finetuning_type="lora", lora_rank, lora_alpha,
    # lora_target, lora_dropout, etc.
    lora_rank = getattr(finetuning_args, "lora_rank", 8)
    lora_alpha = getattr(finetuning_args, "lora_alpha", lora_rank * 2)
    lora_target = getattr(finetuning_args, "lora_target", "all-linear")
    lora_dropout = getattr(finetuning_args, "lora_dropout", 0.0)

    # Optional LoRA init / scaling flags from LlamaFactory.
    use_rslora = getattr(finetuning_args, "use_rslora", False)
    use_dora = getattr(finetuning_args, "use_dora", False)

    # Handle target_modules: LlamaFactory uses comma-separated string or list.
    # "all" in LlamaFactory means "all linear modules" — same as PEFT's "all-linear".
    if isinstance(lora_target, str):
        if lora_target in ("all", "all-linear"):
            target_modules = "all-linear"
        else:
            target_modules = [m.strip() for m in lora_target.split(",") if m.strip()]
    elif isinstance(lora_target, (list, set)):
        # Handle "all" / "all-linear" appearing as list elements — they are
        # supersets of any specific target, so just use the string form.
        normalized = {str(m).strip().lower() for m in lora_target}
        if normalized & {"all", "all-linear"}:
            target_modules = "all-linear"
        else:
            target_modules = list(lora_target)
    else:
        target_modules = "all-linear"

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        init_lora_weights=True,
        use_rslora=use_rslora,
        use_dora=use_dora,
        task_type=TaskType.CAUSAL_LM,
    )

    return lora_config


# LlamaFactory LoRA CLI flags that must be stripped from ``remaining_argv``
# when torchturbo manages LoRA.  HF's ``HfArgumentParser`` accepts both
# ``--flag value`` and ``--flag=value`` forms; boolean flags (``--use_rslora``)
# are consumed without a value.
_LORA_CLI_FLAGS: list[str] = [
    # LoRA hyper-parameters (each takes a value)
    "lora_rank",
    "lora_alpha",
    "lora_target",
    "lora_dropout",
    "additional_target",
    "loraplus_lr_ratio",
    "loraplus_lr_embedding",
    "pissa_iter",
    # Boolean flags (no value consumed)
    "use_rslora",
    "use_dora",
    "pissa_init",
    "pissa_convert",
    "create_new_adapter",
]


def strip_lora_args(args: Any) -> Any:
    """Strip LoRA-related flags from LlamaFactory CLI args.

    When torchturbo manages the LoRA lifecycle, LlamaFactory must not
    apply LoRA itself.  This function removes all ``--lora_*``,
    ``--use_rslora``, ``--use_dora``, ``--pissa_*``, etc. flags from the
    arg list and changes ``--finetuning_type lora`` to
    ``--finetuning_type full``.

    Handles ``list[str]`` (CLI argv), ``dict`` (pre-parsed), and ``None``.
    Returns a **new** object; the input is never mutated.
    """
    if args is None:
        return None

    # --- dict form ---
    if isinstance(args, dict):
        stripped = dict(args)
        if stripped.get("finetuning_type") == "lora":
            stripped["finetuning_type"] = "full"
        for flag in _LORA_CLI_FLAGS:
            stripped.pop(flag, None)
        return stripped

    # --- list[str] form (CLI argv) ---
    if isinstance(args, (list, tuple)):
        bool_flags = {
            "use_rslora",
            "use_dora",
            "pissa_init",
            "pissa_convert",
            "create_new_adapter",
        }
        val_flags = set(_LORA_CLI_FLAGS) - bool_flags
        out: list[str] = []
        i = 0
        while i < len(args):
            tok = args[i]

            # --finetuning_type=lora  →  --finetuning_type=full
            if tok.startswith("--finetuning_type=lora"):
                out.append("--finetuning_type=full")
                i += 1
                continue

            # --finetuning_type lora  (separate tokens)
            if tok == "--finetuning_type" and i + 1 < len(args) and args[i + 1] == "lora":
                out.append("--finetuning_type")
                out.append("full")
                i += 2
                continue

            # --flag=value  form (single token with =)
            if tok.startswith("--") and "=" in tok:
                key_eq = tok.split("=", 1)[0][2:]  # strip leading --
                if key_eq in val_flags or key_eq in bool_flags:
                    i += 1  # skip --flag=value
                    continue

            # --flag value  (separate tokens) or --bool-flag (no value)
            if tok.startswith("--"):
                key = tok[2:]  # strip leading --
                if key in val_flags and i + 1 < len(args):
                    i += 2  # skip --flag and value
                    continue
                if key in bool_flags:
                    i += 1  # skip --bool-flag
                    continue

            out.append(tok)
            i += 1

        logger.info(
            "torchturbo: stripped LoRA CLI flags and set finetuning_type=full "
            "(LoRA managed by torchturbo engine)"
        )
        return out

    # Unknown type — return as-is
    return args


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
