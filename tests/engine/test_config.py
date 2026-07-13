"""Tests for TorchTurbo engine config mapping."""

import os

import pytest

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")


def test_engine_config_env_injects_gradient_checkpointing(monkeypatch):
    from torchturbo.engine.config import EngineConfig

    monkeypatch.setenv("TORCH_TURBO_GRADIENT_CHECKPOINTING", "f")

    cfg = EngineConfig.from_dict({"backend": "fsdp2"})

    assert cfg.activation_checkpointing.enabled is False


def test_fsdp_cpu_offload_defaults_to_disabled():
    from torchturbo.engine.config import EngineConfig

    cfg = EngineConfig.from_dict({"backend": "fsdp2"})

    assert cfg.backend_config.cpu_offload is False


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
