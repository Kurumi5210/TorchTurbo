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


def test_init_context_forces_flash_attention_2():
    model_config = SimpleNamespace(dtype=torch.bfloat16)
    engine_config = EngineConfig()

    with TurboModel.init_context(model_config, engine_config):
        assert model_config._attn_implementation == "flash_attention_2"


def test_vlm_init_context_propagates_flash_attention_2_to_subconfigs():
    from torchturbo.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeConfig,
    )

    model_config = Qwen3_5MoeConfig()
    model_config._attn_implementation = "sdpa"
    model_config.text_config._attn_implementation = "sdpa"
    model_config.vision_config._attn_implementation = "sdpa"

    with TurboVLMModel.init_context(model_config, EngineConfig()):
        assert model_config._attn_implementation == "flash_attention_2"
        assert model_config.text_config._attn_implementation == "flash_attention_2"
        assert model_config.vision_config._attn_implementation == "flash_attention_2"


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
