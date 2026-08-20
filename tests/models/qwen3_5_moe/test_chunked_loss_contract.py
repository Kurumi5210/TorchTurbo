"""Explicit Qwen3.5 contract for backend-owned chunked LM loss."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from torchturbo.engine.utils import validate_signature
from torchturbo.models.qwen3_5_moe.model import (
    TurboQwen3_5MoeForConditionalGeneration,
)
from torchturbo.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeModelOutputWithPast,
)


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = SimpleNamespace(embed_tokens=nn.Embedding(8, 3))

    def forward(self, input_ids, **kwargs):
        hidden = torch.randn(input_ids.shape[0], input_ids.shape[1], 3)
        return Qwen3_5MoeModelOutputWithPast(last_hidden_state=hidden)


class _MTP(nn.Module):
    def forward(self, *, hidden_states, **kwargs):
        return [hidden_states * 0.5]


class _FailingHead(nn.Module):
    def forward(self, hidden_states):  # pragma: no cover - failure path
        raise AssertionError("lm_head must not run when skip_lm_head=True")


def test_qwen_declares_explicit_chunked_loss_contract():
    validate_signature(
        TurboQwen3_5MoeForConditionalGeneration.forward,
        skip_lm_head=False,
    )


def test_skip_lm_head_returns_main_and_mtp_hidden_states_without_projection():
    model = object.__new__(TurboQwen3_5MoeForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        return_dict=True,
        text_config=SimpleNamespace(pad_token_id=0),
    )
    model.model = _Backbone()
    model.mtp = _MTP()
    model.lm_head = _FailingHead()
    model.train()

    outputs = model(
        input_ids=torch.ones(1, 4, dtype=torch.long),
        skip_lm_head=True,
    )

    assert getattr(outputs, "logits", None) is None
    assert outputs.last_hidden_state.shape == (1, 4, 3)
    assert len(outputs.mtp_hidden_states) == 1
    assert outputs.mtp_hidden_states[0].shape == (1, 4, 3)
