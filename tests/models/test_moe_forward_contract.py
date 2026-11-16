"""Model initialization must preserve the shared MoE output contract."""

import copy
import importlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from torchturbo.models.deepseek_v4 import modeling_deepseek_v4 as dsv4
from torchturbo.models.deepseek_v4.model import TurboDeepseekV4ForCausalLM
from torchturbo.models.glm_moe_dsa import modeling_glm_moe_dsa as glm
from torchturbo.models.glm_moe_dsa.model import TurboGlmMoeDsaForCausalLM
from torchturbo.models.kimi_k25 import model as kimi_model
from torchturbo.models.kimi_k25 import modeling_deepseek as kimi
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.module import ModuleSpec
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.moe_layer import MoELayer, MoESubmodules
from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.runtime.cpu import CpuRuntime

from .deepseek_v4.test_mtp import _make_mtp_model
from .deepseek_v4.test_packed_inputs import (
    _make_model,
    _parallelize,
    restore_attention_forwards,  # noqa: F401
)


@pytest.fixture(autouse=True)
def cpu_dispatch(monkeypatch):
    runtime = CpuRuntime()
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    monkeypatch.setattr(
        importlib.import_module("torchturbo.runtime.runtime"), "_SELECTED_RUNTIME", runtime,
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(kimi_model, "is_npu_available", lambda: False)
    monkeypatch.setattr(kimi, "seed_all", lambda: None)
    for norm in (glm.GlmMoeDsaRMSNorm, kimi.DeepseekV3RMSNorm, dsv4.DeepseekV4RMSNorm):
        monkeypatch.setattr(norm, "forward", norm.forward)
    get_or_create_ops(device_runtime=runtime, default_policy=OpPolicy.NATIVE)


def _make_moe(hidden_size=4, *, hash_routing=False):
    config = MoEConfig(
        hidden_size=hidden_size, moe_ffn_hidden_size=3, num_moe_experts=2,
        gated_linear_unit=True, activation_func=F.silu,
        moe_router_topk=1 if hash_routing else 2,
        moe_router_score_function="sigmoid", moe_router_dtype="fp32",
        num_hash_layers=int(hash_routing), vocab_size=16,
    )
    layer = MoELayer(
        config, None, layer_number=0,
        submodules=MoESubmodules(experts=ModuleSpec(module=GroupedMLP)),
    )
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.normal_(std=0.3)
        if hash_routing:
            layer.router.tid2eid.copy_(torch.arange(16).remainder(2).unsqueeze(1))
    return layer


class _TensorOutputMoE(nn.Module):
    def __init__(self, moe):
        super().__init__()
        self.moe = moe

    def forward(self, hidden_states, input_ids=None):
        return self.moe(hidden_states, input_ids)[0]


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("families", [
    ("glm", "glm"),
    ("kimi", "kimi"),
    ("deepseek", "deepseek"),
    ("glm", "kimi", "deepseek", "glm"),
    ("deepseek", "kimi", "glm", "deepseek"),
])
def test_repeated_and_mixed_initialization_preserves_moe_forward(batch_size, families):
    torch.manual_seed(37)
    layer = _make_moe(hash_routing=True)
    inputs = torch.randn(batch_size, 3, 4)
    input_ids = torch.arange(batch_size * 3).view(batch_size, 3)
    expected_output, expected_logits = layer(inputs, input_ids)
    original_forward = MoELayer.forward

    for family in families:
        if family == "glm":
            TurboGlmMoeDsaForCausalLM._pre_init_patches(SimpleNamespace())
        elif family == "kimi":
            kimi_model.TurboKimiK25ForConditionalGeneration._pre_init_patches()
        else:
            TurboDeepseekV4ForCausalLM._pre_init_patches(SimpleNamespace())

    assert MoELayer.forward is original_forward
    output, logits = layer(inputs, input_ids)
    assert output.shape == inputs.shape
    assert logits.shape == (batch_size * 3, 2)
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("tensor_output", [False, True])
@pytest.mark.parametrize("decoder_cls,num_attention_outputs", [
    (glm.GlmMoeDsaDecoderLayer, 4),
    (kimi.DeepseekV3DecoderLayer, 3),
])
def test_decoder_preserves_moe_outputs_and_gradients(
    batch_size, tensor_output, decoder_cls, num_attention_outputs,
):
    torch.manual_seed(41)
    moe = _make_moe()
    reference = copy.deepcopy(moe)
    decoder = SimpleNamespace(
        input_layernorm=nn.Identity(),
        post_attention_layernorm=nn.Identity(),
        self_attn=lambda hidden_states, **kwargs: (
            torch.zeros_like(hidden_states), *([None] * (num_attention_outputs - 1))
        ),
        mlp=_TensorOutputMoE(moe) if tensor_output else moe,
    )
    inputs = torch.randn(batch_size, 3, 4, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()

    actual = decoder_cls.forward(decoder, inputs)[0]
    expected = reference_inputs + reference(reference_inputs)[0]
    assert actual.shape == inputs.shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(inputs.grad, reference_inputs.grad)
    for actual_parameter, expected_parameter in zip(
        moe.parameters(), reference.parameters(), strict=True,
    ):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_deepseek_decoder_preserves_hash_routing_outputs_and_gradients(batch_size):
    model = _make_model(1)
    model.layers[0].ffn = _make_moe(hidden_size=8, hash_routing=True)
    _parallelize(model)
    decoder = model.layers[0]
    reference = copy.deepcopy(decoder)
    reference.ffn = _TensorOutputMoE(reference.ffn)
    input_ids = torch.arange(batch_size * 3).view(batch_size, 3)
    position_ids = torch.arange(3).unsqueeze(0)
    inputs = torch.randn(batch_size, 3, 2, 8, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()

    actual = decoder(inputs, input_ids=input_ids, position_ids=position_ids)[0]
    expected = reference(reference_inputs, input_ids=input_ids, position_ids=position_ids)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(inputs.grad, reference_inputs.grad)
    for actual_parameter, expected_parameter in zip(
        decoder.parameters(), reference.parameters(), strict=True,
    ):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("block_size", [0, 2], ids=["mtp", "dspark"])
def test_deepseek_mtp_preserves_moe_outputs_and_gradients(batch_size, block_size):
    model = _make_mtp_model(block_size=block_size, num_stages=1)
    model.mtp[0].ffn = _make_moe(hidden_size=8)
    reference = copy.deepcopy(model)
    reference.mtp[0].ffn = _TensorOutputMoE(reference.mtp[0].ffn)
    input_ids = torch.arange(1, batch_size * 5 + 1).view(batch_size, 5)

    actual = model(input_ids=input_ids)
    expected = reference(input_ids=input_ids)
    if block_size:
        actual_loss = actual.dspark_loss_fn(input_ids)
        expected_loss = expected.dspark_loss_fn(input_ids)
    else:
        torch.testing.assert_close(actual.mtp_logits[0], expected.mtp_logits[0], rtol=0, atol=0)
        actual_loss = actual.mtp_logits[0].square().sum()
        expected_loss = expected.mtp_logits[0].square().sum()
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    actual_loss.backward()
    expected_loss.backward()
    assert model.mtp[0].ffn.experts.weight1.grad.abs().sum() > 0
    for actual_parameter, expected_parameter in zip(
        model.parameters(), reference.parameters(), strict=True,
    ):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)
