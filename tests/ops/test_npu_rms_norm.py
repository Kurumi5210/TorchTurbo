from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from torchturbo.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Head,
    DeepseekV4RMSNorm,
)
from torchturbo.ops.providers.npu.rms_norm import NpuRMSNorm


def _install_fused_norm(monkeypatch, implementation):
    module = ModuleType("torch_npu")
    module.npu_rms_norm = implementation
    monkeypatch.setitem(sys.modules, "torch_npu", module)


@pytest.mark.parametrize("shape", [(0, 8), (2, 0, 8)])
@pytest.mark.parametrize(
    ("input_dtype", "weight_dtype"),
    [
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.bfloat16),
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.float32),
        (torch.float16, torch.float32),
    ],
)
def test_empty_rms_norm_preserves_dtype_and_zero_gradients(
    monkeypatch, shape, input_dtype, weight_dtype
):
    fused_norm = Mock(side_effect=AssertionError("fused RMSNorm received empty input"))
    _install_fused_norm(monkeypatch, fused_norm)
    hidden = torch.empty(shape, dtype=input_dtype, requires_grad=True)
    weight = torch.ones(shape[-1], dtype=weight_dtype, requires_grad=True)

    output = NpuRMSNorm()(hidden, weight, 1e-6)

    assert output.shape == hidden.shape
    assert output.dtype == hidden.dtype
    hidden_grad, weight_grad = torch.autograd.grad(output.sum(), (hidden, weight))
    torch.testing.assert_close(hidden_grad, torch.zeros_like(hidden), rtol=0, atol=0)
    torch.testing.assert_close(weight_grad, torch.zeros_like(weight), rtol=0, atol=0)
    fused_norm.assert_not_called()


def test_nonempty_rms_norm_keeps_fused_kernel_result(monkeypatch):
    hidden = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    weight = torch.ones(8, dtype=torch.float32)
    expected = torch.randn_like(hidden)
    fused_norm = Mock(return_value=(expected, torch.ones(2, 3, 1)))
    _install_fused_norm(monkeypatch, fused_norm)

    actual = NpuRMSNorm()(hidden, weight, 1e-5)

    assert actual is expected
    fused_norm.assert_called_once_with(hidden, weight, 1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_empty_deepseek_head_keeps_all_parameter_gradients(monkeypatch, dtype):
    def reject_empty_gradient(gradient):
        if gradient.numel() == 0:
            raise RuntimeError("RmsNormGrad does not support empty dy")
        return gradient

    def fused_norm(values, weight, eps):
        values_fp32 = values.float()
        inverse = torch.rsqrt(values_fp32.square().mean(-1, keepdim=True) + eps)
        output = (values_fp32 * inverse * weight).to(values.dtype)
        # Match CANN's failure boundary: forward accepts empty input, backward
        # rejects empty dy. Full-head-then-selection sends nonempty zero dy.
        output.register_hook(reject_empty_gradient)
        return output, inverse

    _install_fused_norm(monkeypatch, fused_norm)
    config = SimpleNamespace(vocab_size=7, hidden_size=8, rms_norm_eps=1e-6, hc_eps=1e-5)
    torch.manual_seed(42)
    head = DeepseekV4Head(config)
    nn.init.normal_(head.weight)
    norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    monkeypatch.setattr(
        norm,
        "forward",
        lambda values: NpuRMSNorm()(values, norm.weight, config.rms_norm_eps),
    )
    hidden = torch.randn(1, 4, 2, config.hidden_size, dtype=dtype, requires_grad=True)
    hc_fn = nn.Parameter(torch.randn(2, 2 * config.hidden_size))
    hc_scale = nn.Parameter(torch.randn(1))
    hc_base = nn.Parameter(torch.randn(2))
    parameters = (hidden, hc_fn, hc_scale, hc_base, norm.weight, head.weight)
    selector = torch.zeros(hidden.shape[1], dtype=torch.bool)

    reference = head(hidden, hc_fn, hc_scale, hc_base, norm)[:, selector]
    reference_grads = torch.autograd.grad(reference.sum(), parameters)
    selected = head(hidden[:, selector], hc_fn, hc_scale, hc_base, norm)
    selected_grads = torch.autograd.grad(selected.sum(), parameters)

    assert selected.shape == (1, 0, config.vocab_size)
    torch.testing.assert_close(selected, reference, rtol=0, atol=0)
    for actual, expected, parameter in zip(selected_grads, reference_grads, parameters):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual, torch.zeros_like(parameter), rtol=0, atol=0)
