"""Tests for LlamaFactory adapter config mapping."""

import os
import sys
import types

import pytest

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

if "grouped_gemm" not in sys.modules:  # pragma: no cover - env shim
    _gg = types.ModuleType("grouped_gemm")
    _gg.ops = types.SimpleNamespace()
    sys.modules["grouped_gemm"] = _gg


@pytest.fixture(autouse=True)
def _reset_cli(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pytest"])


def test_build_config_cli_prefetch_takes_effect():
    """--tt_prefetch_strategy / --tt_prefetch_direction must reach backend_config."""
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs
    from torchturbo.engine.backends.fsdp2.config import (
        FSDPPrefetchDirection,
        FSDPPrefetchStrategy,
    )

    training_args = types.SimpleNamespace(
        fsdp=[], fsdp_config={}, gradient_checkpointing=True
    )
    tt = TorchTurboArgs(
        prefetch_strategy="expert_with_layer", prefetch_direction="forward"
    )
    cfg, _ = build_config_from_lf_args(training_args=training_args, torchturbo_args=tt)

    assert cfg.backend_config.prefetch_strategy == FSDPPrefetchStrategy.EXPERT_WITH_LAYER
    assert cfg.backend_config.prefetch_direction == FSDPPrefetchDirection.FORWARD


def test_build_config_maps_freeze_args_to_engine_policy():
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs
    from torchturbo.engine.config import ParameterGroup

    training_args = types.SimpleNamespace(
        fsdp=[], fsdp_config={}, gradient_checkpointing=True
    )
    tt = TorchTurboArgs(
        freeze_vision_tower=True,
        freeze_language_model=True,
    )

    cfg, _ = build_config_from_lf_args(training_args=training_args, torchturbo_args=tt)

    assert cfg.freeze_policy.parameter_groups == (
        ParameterGroup.VISION_TOWER,
        ParameterGroup.LANGUAGE_MODEL,
    )


def test_build_config_maps_mtp_args_to_engine_config():
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs

    training_args = types.SimpleNamespace(
        fsdp=[], fsdp_config={}, gradient_checkpointing=True
    )
    tt = TorchTurboArgs(enable_mtp=True, mtp_loss_scaling_factor=0.3)

    cfg, _ = build_config_from_lf_args(training_args=training_args, torchturbo_args=tt)

    assert cfg.mtp.enabled is True
    assert cfg.mtp.loss_scaling_factor == 0.3


def test_build_config_maps_op_policy_to_engine_config():
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs
    from torchturbo.ops.base import OpPolicy

    training_args = types.SimpleNamespace(
        fsdp=[], fsdp_config={}, gradient_checkpointing=True
    )
    tt = TorchTurboArgs(op_policy="native")

    cfg, _ = build_config_from_lf_args(training_args=training_args, torchturbo_args=tt)

    assert cfg.ops.policy is OpPolicy.NATIVE


def test_build_config_cli_prefetch_ignores_lf_yaml_prefetch():
    """TorchTurbo prefetch config does not read LF fsdp_config prefetch fields."""
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs
    from torchturbo.engine.backends.fsdp2.config import FSDPPrefetchStrategy

    training_args = types.SimpleNamespace(
        fsdp=[],
        fsdp_config={"prefetch_strategy": "expert_lookahead"},
        gradient_checkpointing=True,
    )
    tt = TorchTurboArgs()  # defaults: auto / forward_backward
    cfg, _ = build_config_from_lf_args(training_args=training_args, torchturbo_args=tt)

    assert cfg.backend_config.prefetch_strategy == FSDPPrefetchStrategy.AUTO


def test_build_config_maps_lf_fsdp_without_offload_to_disabled_cpu_offload():
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs

    training_args = types.SimpleNamespace(
        fsdp=["full_shard"], fsdp_config={}, gradient_checkpointing=True
    )

    cfg, _ = build_config_from_lf_args(
        training_args=training_args, torchturbo_args=TorchTurboArgs()
    )

    assert cfg.backend_config.cpu_offload is False
    assert cfg.backend_config.pin_memory is True


def test_build_config_maps_lf_fsdp_offload_to_enabled_cpu_offload():
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs

    training_args = types.SimpleNamespace(
        fsdp=["full_shard", "offload"], fsdp_config={}, gradient_checkpointing=True
    )

    cfg, _ = build_config_from_lf_args(
        training_args=training_args, torchturbo_args=TorchTurboArgs()
    )

    assert cfg.backend_config.cpu_offload is True
    assert cfg.backend_config.pin_memory is True


def test_build_config_cpu_offload_torchturbo_cli_cannot_disable_lf_offload(
    monkeypatch,
):
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs

    training_args = types.SimpleNamespace(
        fsdp=["full_shard", "offload"], fsdp_config={}, gradient_checkpointing=True
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_lf.py", "--tt_cpu_offload=false"],
    )

    with pytest.raises(ValueError, match="tt_cpu_offload must not disable"):
        build_config_from_lf_args(
            training_args=training_args,
            torchturbo_args=TorchTurboArgs(cpu_offload=False),
        )


def test_build_config_cpu_offload_env_cannot_disable_lf_offload(monkeypatch):
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs

    monkeypatch.setenv("TORCH_TURBO_CPU_OFFLOAD", "false")
    training_args = types.SimpleNamespace(
        fsdp=["full_shard", "offload"], fsdp_config={}, gradient_checkpointing=True
    )

    with pytest.raises(ValueError, match="tt_cpu_offload must not disable"):
        build_config_from_lf_args(
            training_args=training_args,
            torchturbo_args=TorchTurboArgs(),
        )


def test_build_config_cpu_offload_env_overrides_lf(monkeypatch):
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs

    monkeypatch.setenv("TORCH_TURBO_CPU_OFFLOAD", "true")
    training_args = types.SimpleNamespace(
        fsdp=["full_shard"], fsdp_config={}, gradient_checkpointing=True
    )

    cfg, _ = build_config_from_lf_args(
        training_args=training_args,
        torchturbo_args=TorchTurboArgs(),
    )

    assert cfg.backend_config.cpu_offload is True


def test_build_config_cpu_offload_cli_overrides_lf(monkeypatch):
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs

    monkeypatch.setattr(sys, "argv", ["run_lf.py", "--tt_cpu_offload"])
    training_args = types.SimpleNamespace(
        fsdp=["full_shard"], fsdp_config={}, gradient_checkpointing=True
    )

    cfg, _ = build_config_from_lf_args(
        training_args=training_args,
        torchturbo_args=TorchTurboArgs(cpu_offload=True),
    )

    assert cfg.backend_config.cpu_offload is True


def test_engine_activation_checkpointing_defaults_to_enabled():
    from torchturbo.engine.config import EngineConfig

    cfg = EngineConfig.from_dict({"backend": "fsdp2"})

    assert cfg.activation_checkpointing.enabled is True


def test_lf_gradient_checkpointing_is_taken_over_by_engine():
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs

    training_args = types.SimpleNamespace(
        fsdp=[], fsdp_config={}, gradient_checkpointing=True
    )
    model_args = types.SimpleNamespace(
        disable_gradient_checkpointing=False, use_reentrant_gc=True
    )

    cfg, _ = build_config_from_lf_args(
        training_args=training_args,
        model_args=model_args,
        torchturbo_args=TorchTurboArgs(gradient_checkpointing=True),
    )

    assert cfg.activation_checkpointing.enabled is True
    assert training_args.gradient_checkpointing is False
    assert model_args.disable_gradient_checkpointing is True
    assert model_args.use_reentrant_gc is True


def test_lf_gradient_checkpointing_uses_lf_value_when_tt_arg_is_default():
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs

    training_args = types.SimpleNamespace(
        fsdp=[], fsdp_config={}, gradient_checkpointing=False
    )
    model_args = types.SimpleNamespace(disable_gradient_checkpointing=True)

    cfg, _ = build_config_from_lf_args(
        training_args=training_args,
        model_args=model_args,
        torchturbo_args=TorchTurboArgs(),
    )

    assert cfg.activation_checkpointing.enabled is False


def test_lf_disable_gradient_checkpointing_does_not_enable_engine_ac():
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs

    training_args = types.SimpleNamespace(
        fsdp=[], fsdp_config={}, gradient_checkpointing=False
    )
    model_args = types.SimpleNamespace(disable_gradient_checkpointing=False)

    cfg, _ = build_config_from_lf_args(
        training_args=training_args,
        model_args=model_args,
        torchturbo_args=TorchTurboArgs(),
    )

    assert cfg.activation_checkpointing.enabled is False
    assert model_args.disable_gradient_checkpointing is True


@pytest.mark.parametrize("adapter_gc, tt_gc", [(True, False), (False, True)])
def test_lf_gradient_checkpointing_torchturbo_cli_mismatch_raises(
    adapter_gc, tt_gc, monkeypatch
):
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs

    training_args = types.SimpleNamespace(
        fsdp=[], fsdp_config={}, gradient_checkpointing=adapter_gc
    )
    model_args = types.SimpleNamespace(disable_gradient_checkpointing=True)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_lf.py", "--tt_gradient_checkpointing", str(tt_gc).lower()],
    )

    with pytest.raises(ValueError, match="tt_gradient_checkpointing must match"):
        build_config_from_lf_args(
            training_args=training_args,
            model_args=model_args,
            torchturbo_args=TorchTurboArgs(gradient_checkpointing=tt_gc),
        )

    assert training_args.gradient_checkpointing is adapter_gc
    assert model_args.disable_gradient_checkpointing is True


@pytest.mark.parametrize("adapter_gc, tt_gc", [(True, True), (False, False)])
def test_lf_gradient_checkpointing_torchturbo_cli_match_takes_effect(
    adapter_gc, tt_gc, monkeypatch
):
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs

    training_args = types.SimpleNamespace(
        fsdp=[], fsdp_config={}, gradient_checkpointing=adapter_gc
    )
    model_args = types.SimpleNamespace(disable_gradient_checkpointing=True)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_lf.py", f"--tt_gradient_checkpointing={str(tt_gc).lower()}"],
    )

    cfg, _ = build_config_from_lf_args(
        training_args=training_args,
        model_args=model_args,
        torchturbo_args=TorchTurboArgs(gradient_checkpointing=tt_gc),
    )

    assert cfg.activation_checkpointing.enabled is adapter_gc


def test_lf_gradient_checkpointing_env_mismatch_raises(monkeypatch):
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs

    monkeypatch.setenv("TORCH_TURBO_GRADIENT_CHECKPOINTING", "false")
    training_args = types.SimpleNamespace(
        fsdp=[], fsdp_config={}, gradient_checkpointing=True
    )
    model_args = types.SimpleNamespace(disable_gradient_checkpointing=True)

    with pytest.raises(ValueError, match="tt_gradient_checkpointing must match"):
        build_config_from_lf_args(
            training_args=training_args,
            model_args=model_args,
            torchturbo_args=TorchTurboArgs(),
        )

    assert training_args.gradient_checkpointing is True
    assert model_args.disable_gradient_checkpointing is True


def test_lf_gradient_checkpointing_env_match_takes_effect(monkeypatch):
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs

    monkeypatch.setenv("TORCH_TURBO_GRADIENT_CHECKPOINTING", "false")
    training_args = types.SimpleNamespace(
        fsdp=[], fsdp_config={}, gradient_checkpointing=False
    )
    model_args = types.SimpleNamespace(disable_gradient_checkpointing=True)

    cfg, _ = build_config_from_lf_args(
        training_args=training_args,
        model_args=model_args,
        torchturbo_args=TorchTurboArgs(),
    )

    assert cfg.activation_checkpointing.enabled is False


def test_lf_gradient_checkpointing_raw_args_are_disabled():
    from torchturbo.adapters.llamafactory_v0.utils import build_config_from_lf_args
    from torchturbo.engine.args import TorchTurboArgs

    training_args = types.SimpleNamespace(
        fsdp=[], fsdp_config={}, gradient_checkpointing=False
    )
    model_args = types.SimpleNamespace(disable_gradient_checkpointing=True)

    cfg, dict_args = build_config_from_lf_args(
        training_args=training_args,
        model_args=model_args,
        torchturbo_args=TorchTurboArgs(),
        raw_args={
            "gradient_checkpointing": True,
            "disable_gradient_checkpointing": False,
        },
    )
    assert cfg.activation_checkpointing.enabled is False
    assert dict_args["gradient_checkpointing"] is False
    assert dict_args["disable_gradient_checkpointing"] is True

    _, list_args = build_config_from_lf_args(
        training_args=training_args,
        model_args=model_args,
        torchturbo_args=TorchTurboArgs(),
        raw_args=[
            "--gradient_checkpointing=true",
            "--use_reentrant_gc",
            "true",
            "--foo",
            "bar",
        ],
    )
    assert "--gradient_checkpointing=true" not in list_args
    assert list_args[-1] == "--disable_gradient_checkpointing"
    assert "--use_reentrant_gc" in list_args
