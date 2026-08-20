"""Explicit Qwen3.5 dense contract for backend-owned chunked LM loss."""

from types import SimpleNamespace

import torch
from torch import nn

from torchturbo.engine.utils import validate_signature
from torchturbo.models.qwen3_5.model import TurboQwen3_5ForConditionalGeneration
from torchturbo.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ModelOutputWithPast


class _Backbone(nn.Module):
    def forward(self, input_ids, **kwargs):
        hidden = torch.randn(input_ids.shape[0], input_ids.shape[1], 3)
        return Qwen3_5ModelOutputWithPast(last_hidden_state=hidden)


class _FailingHead(nn.Module):
    def forward(self, hidden_states):  # pragma: no cover - failure path
        raise AssertionError("lm_head must not run when skip_lm_head=True")


def test_qwen_dense_declares_explicit_chunked_loss_contract():
    validate_signature(
        TurboQwen3_5ForConditionalGeneration.forward,
        skip_lm_head=False,
    )


def test_skip_lm_head_returns_hidden_states_without_projection():
    model = object.__new__(TurboQwen3_5ForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(return_dict=True)
    model.model = _Backbone()
    model.lm_head = _FailingHead()
    model.train()

    outputs = model(
        input_ids=torch.ones(1, 4, dtype=torch.long),
        skip_lm_head=True,
    )

    assert getattr(outputs, "logits", None) is None
    assert outputs.last_hidden_state.shape == (1, 4, 3)
