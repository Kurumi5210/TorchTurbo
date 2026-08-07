from types import SimpleNamespace

import torch

from torchturbo.engine.config import EngineConfig, MTPConfig
from torchturbo.models.base import TurboModel, TurboVLMModel


def test_init_context_strips_mtp_without_restoring_model_structure_config():
    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        num_nextn_predict_layers=2,
    )
    engine_config = EngineConfig(mtp=MTPConfig(enabled=False))

    with TurboModel.init_context(model_config, engine_config):
        assert model_config.dtype == torch.float32
        assert model_config.num_nextn_predict_layers == 0

    assert model_config.dtype == torch.bfloat16
    assert model_config.num_nextn_predict_layers == 0


def test_init_context_retains_mtp_when_enabled():
    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        num_nextn_predict_layers=2,
    )
    engine_config = EngineConfig(mtp=MTPConfig(enabled=True))

    with TurboModel.init_context(model_config, engine_config):
        assert model_config.num_nextn_predict_layers == 2

    assert model_config.num_nextn_predict_layers == 2


def test_vlm_init_context_forwards_engine_config_and_restores_nested_dtype():
    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        text_config=SimpleNamespace(dtype=torch.float16),
        vision_config=SimpleNamespace(dtype=torch.float16),
        num_nextn_predict_layers=1,
    )
    engine_config = EngineConfig(mtp=MTPConfig(enabled=False))

    with TurboVLMModel.init_context(model_config, engine_config):
        assert model_config.dtype == torch.float32
        assert model_config.text_config.dtype == torch.float32
        assert model_config.vision_config.dtype == torch.float32
        assert model_config.num_nextn_predict_layers == 0

    assert model_config.dtype == torch.bfloat16
    assert model_config.text_config.dtype == torch.float16
    assert model_config.vision_config.dtype == torch.float16
    assert model_config.num_nextn_predict_layers == 0
