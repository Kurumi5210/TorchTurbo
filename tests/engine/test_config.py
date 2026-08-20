"""Tests for TorchTurbo engine config mapping."""

import json
import os

import pytest

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")


def test_engine_config_env_injects_gradient_checkpointing(monkeypatch):
    from torchturbo.engine.config import EngineConfig

    monkeypatch.setenv("TORCH_TURBO_GRADIENT_CHECKPOINTING", "f")

    cfg = EngineConfig.from_dict({"backend": "fsdp2"})

    assert cfg.activation_checkpointing.enabled is False


def test_engine_config_env_injects_op_policy(monkeypatch):
    from torchturbo.engine.config import EngineConfig
    from torchturbo.ops.base import OpPolicy

    monkeypatch.setenv("TORCH_TURBO_OP_POLICY", "native")

    cfg = EngineConfig.from_dict({"backend": "fsdp2"})

    assert cfg.ops.policy is OpPolicy.NATIVE


def test_engine_config_from_dict_rejects_flat_op_policy():
    from torchturbo.engine.config import EngineConfig

    with pytest.raises(ValueError, match="op_policy is no longer supported"):
        EngineConfig.from_dict(
            {
                "backend": "fsdp2",
                "op_policy": "native",
            }
        )


@pytest.mark.parametrize("value", [False, "false", "f", "0"])
def test_engine_config_normalizes_disabled_use_deterministic_algorithms(value):
    from torchturbo.engine.config import EngineConfig

    direct = EngineConfig(use_deterministic_algorithms=value)
    from_dict = EngineConfig.from_dict({"use_deterministic_algorithms": value})

    assert direct.use_deterministic_algorithms is False
    assert from_dict.use_deterministic_algorithms is False


@pytest.mark.parametrize("value", [True, "true", "t", "1"])
def test_engine_config_normalizes_enabled_use_deterministic_algorithms(value):
    from torchturbo.engine.config import EngineConfig

    direct = EngineConfig(use_deterministic_algorithms=value)
    from_dict = EngineConfig.from_dict({"use_deterministic_algorithms": value})

    assert direct.use_deterministic_algorithms is True
    assert from_dict.use_deterministic_algorithms is True


def test_fsdp_cpu_offload_defaults_to_disabled():
    from torchturbo.engine.config import EngineConfig

    cfg = EngineConfig.from_dict({"backend": "fsdp2"})

    assert cfg.backend_config.cpu_offload is False


def test_engine_config_to_dict_excludes_runtime_state():
    from torchturbo.engine.config import EngineConfig

    data = EngineConfig().to_dict()

    assert "device" not in data
    assert "device_runtime" not in data


def test_engine_config_from_dict_parses_moe_config():
    from torchturbo.engine.config import EngineConfig

    cfg = EngineConfig.from_dict({"moe": {"aux_loss_coeff": 0.01}})

    assert cfg.moe.aux_loss_coeff == 0.01


def test_engine_config_env_enables_cpu_offload(monkeypatch):
    from torchturbo.engine.config import EngineConfig

    monkeypatch.setenv("TORCH_TURBO_CPU_OFFLOAD", "1")

    cfg = EngineConfig.from_dict({"backend": "fsdp2"})

    assert cfg.backend_config.cpu_offload is True


def test_engine_config_from_dict_parses_freeze_policy():
    from torchturbo.engine.config import EngineConfig
    from torchturbo.engine.config import ParameterGroup

    cfg = EngineConfig.from_dict(
        {
            "backend": "fsdp2",
            "freeze_policy": {
                "parameter_groups": ["vision_tower", "language_model"],
            },
        }
    )

    assert cfg.freeze_policy.parameter_groups == (
        ParameterGroup.VISION_TOWER,
        ParameterGroup.LANGUAGE_MODEL,
    )


def test_engine_config_from_dict_parses_ops_policy():
    from torchturbo.engine.config import EngineConfig
    from torchturbo.engine.config import OpsConfig
    from torchturbo.ops.base import OpPolicy

    with pytest.raises(TypeError, match="OpsConfig.policy must be an OpPolicy"):
        OpsConfig(policy="custom")

    from_string = EngineConfig.from_dict(
        {
            "backend": "fsdp2",
            "ops": {"policy": "custom"},
        }
    )

    from_enum = EngineConfig.from_dict(
        {
            "backend": "fsdp2",
            "ops": {"policy": OpPolicy.CUSTOM},
        }
    )

    assert from_string.ops.policy is OpPolicy.CUSTOM
    assert from_enum.ops.policy is OpPolicy.CUSTOM


def test_engine_config_from_dict_parses_per_op_overrides():
    from torchturbo.engine.config import EngineConfig
    from torchturbo.ops.base import OpPolicy

    cfg = EngineConfig.from_dict(
        {
            "backend": "fsdp2",
            "ops": {
                "policy": "auto",
                "overrides": {
                    "rms_norm": "torch_rms_norm",
                    "grouped_matmul": "grouped_gemm",
                },
            },
        }
    )

    assert cfg.ops.policy is OpPolicy.AUTO
    assert cfg.ops.overrides == {
        "rms_norm": "torch_rms_norm",
        "grouped_matmul": "grouped_gemm",
    }


@pytest.mark.parametrize(
    "overrides",
    [None, [], {"rms_norm": 1}, {1: "torch_rms_norm"}],
)
def test_ops_config_rejects_invalid_overrides(overrides):
    from torchturbo.engine.config import OpsConfig

    with pytest.raises(TypeError, match="OpsConfig.overrides"):
        OpsConfig(overrides=overrides)


def test_ops_config_copies_overrides():
    from torchturbo.engine.config import OpsConfig

    overrides = {"rms_norm": "torch_rms_norm"}
    config = OpsConfig(overrides=overrides)
    overrides["rms_norm"] = "other"

    assert config.overrides == {"rms_norm": "torch_rms_norm"}


@pytest.mark.parametrize("overrides", [None, []])
def test_build_ops_config_rejects_falsey_invalid_overrides(overrides):
    from torchturbo.engine.config import _build_ops_config

    with pytest.raises(TypeError, match="OpsConfig.overrides"):
        _build_ops_config({"ops": {"overrides": overrides}})


def test_engine_config_json_round_trip_restores_ops_policy():
    from torchturbo.engine.config import EngineConfig
    from torchturbo.engine.config import OpsConfig
    from torchturbo.ops.base import OpPolicy

    original = EngineConfig(ops=OpsConfig(policy=OpPolicy.NATIVE))
    serialized = json.loads(json.dumps(original.to_dict()))

    restored = EngineConfig.from_dict(serialized)

    assert restored.ops.policy is OpPolicy.NATIVE


def test_engine_config_json_round_trip_restores_per_op_overrides():
    from torchturbo.engine.config import EngineConfig, OpsConfig

    original = EngineConfig(
        ops=OpsConfig(
            overrides={
                "rms_norm": "torch_rms_norm",
                "swiglu": "torch_swiglu",
            }
        )
    )
    serialized = json.loads(json.dumps(original.to_dict()))
    restored = EngineConfig.from_dict(serialized)

    assert restored.ops.overrides == original.ops.overrides


def test_freeze_policy_direct_constructor_accepts_parameter_groups():
    from torchturbo.engine.config import FreezePolicy
    from torchturbo.engine.config import ParameterGroup

    policy = FreezePolicy((ParameterGroup.LANGUAGE_MODEL,))

    assert policy.parameter_groups == (ParameterGroup.LANGUAGE_MODEL,)


def test_freeze_policy_from_list_parses_string_groups():
    from torchturbo.engine.config import FreezePolicy
    from torchturbo.engine.config import ParameterGroup

    policy = FreezePolicy.from_list(["language_model", "language_model"])

    assert policy.parameter_groups == (ParameterGroup.LANGUAGE_MODEL,)


def test_engine_config_from_dict_rejects_unknown_freeze_policy_fields():
    from torchturbo.engine.config import EngineConfig

    with pytest.raises(ValueError, match="Unknown freeze_policy field"):
        EngineConfig.from_dict(
            {
                "backend": "fsdp2",
                "freeze_policy": {"groups": ["vision_tower"]},
            }
        )


def test_engine_config_from_dict_parses_mtp_config():
    from torchturbo.engine.config import EngineConfig

    cfg = EngineConfig.from_dict(
        {
            "backend": "fsdp2",
            "mtp": {"enabled": True, "loss_scaling_factor": 0.25},
        }
    )

    assert cfg.mtp.enabled is True
    assert cfg.mtp.loss_scaling_factor == 0.25


def test_engine_config_from_dict_parses_flat_mtp_fields():
    from torchturbo.engine.config import EngineConfig

    cfg = EngineConfig.from_dict(
        {
            "backend": "fsdp2",
            "enable_mtp": "true",
            "mtp_loss_scaling_factor": None,
        }
    )

    assert cfg.mtp.enabled is True
    assert cfg.mtp.loss_scaling_factor == 0.1
