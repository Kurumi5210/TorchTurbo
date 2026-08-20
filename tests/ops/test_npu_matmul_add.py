from __future__ import annotations

import sys
from types import ModuleType

import torch

from torchturbo.ops.providers.npu.matmul_add import (
    MindspeedMatmulAdd,
    TorchNpuMatmulAdd,
)


def _install_fake_mindspeed(monkeypatch, fused_op) -> None:
    mindspeed = ModuleType("mindspeed")
    mindspeed.__path__ = []
    mindspeed_ops = ModuleType("mindspeed.ops")
    mindspeed_ops.__path__ = []
    matmul_add = ModuleType("mindspeed.ops.npu_matmul_add")
    matmul_add.npu_matmul_add_fp32 = fused_op
    monkeypatch.setitem(sys.modules, "mindspeed", mindspeed)
    monkeypatch.setitem(sys.modules, "mindspeed.ops", mindspeed_ops)
    monkeypatch.setitem(sys.modules, "mindspeed.ops.npu_matmul_add", matmul_add)


def test_mindspeed_impl_has_highest_priority():
    assert (
        MindspeedMatmulAdd.op_spec().priority
        > TorchNpuMatmulAdd.op_spec().priority
    )


def test_mindspeed_impl_calls_fused_op(monkeypatch):
    calls = []

    def fused_op(input, grad_output, grad_weight):
        calls.append((input, grad_output, grad_weight))
        grad_weight.addmm_(grad_output.t(), input)

    _install_fake_mindspeed(monkeypatch, fused_op)
    lhs = torch.randn(3, 5)
    rhs = torch.randn(3, 4)
    accumulator = torch.zeros(5, 4)

    impl = MindspeedMatmulAdd()
    assert impl.available()
    impl(accumulator, lhs, rhs, transpose_lhs=True)

    assert len(calls) == 1
    assert calls[0][0] is rhs
    assert calls[0][1] is lhs
    assert calls[0][2] is accumulator
    assert torch.allclose(accumulator, lhs.t().matmul(rhs))


def test_torch_npu_fallback_uses_in_place_addmm():
    lhs = torch.randn(3, 5)
    rhs = torch.randn(3, 4)
    accumulator = torch.zeros(5, 4)

    TorchNpuMatmulAdd()(
        accumulator,
        lhs,
        rhs,
        transpose_lhs=True,
    )

    assert torch.allclose(accumulator, lhs.t().matmul(rhs))


def test_torch_npu_impl_detects_torch_npu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))

    assert TorchNpuMatmulAdd().available()
