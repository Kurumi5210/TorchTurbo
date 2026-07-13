"""Tests for TorchTurbo engine config mapping."""

import os

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
