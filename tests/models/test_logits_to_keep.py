"""Causal LM forwards project only the positions selected by logits_to_keep."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    MoeModelOutputWithPast,
)

from torchturbo.models.bailing_hybrid.model import TurboBailingMoeV3ForCausalLM
from torchturbo.models.bailing_hybrid.modeling_bailing_moe_v3 import (
    MoeV3ModelOutputWithPast,
)
from torchturbo.models.glm_moe_dsa.model import TurboGlmMoeDsaForCausalLM
from torchturbo.models.kimi_k25.model import TurboKimiK25ForConditionalGeneration
from torchturbo.models.kimi_k25.modeling_deepseek import DeepseekV3ForCausalLM
from torchturbo.models.llada2_moe.model import TurboLLaDA2MoeForCausalLM

from .deepseek_v4.test_chunked_loss_contract import _make_model as _make_deepseek_v4

_HIDDEN_SIZE = 3
_VOCAB_SIZE = 7


class _Backbone(nn.Module):
    def forward(self, inputs_embeds, **kwargs):
        return BaseModelOutputWithPast(last_hidden_state=inputs_embeds.sin())


class _MoeBackbone(nn.Module):
    def forward(self, inputs_embeds, **kwargs):
        return MoeModelOutputWithPast(last_hidden_state=inputs_embeds.sin())


class _BailingBackbone(nn.Module):
    def forward(self, inputs_embeds, **kwargs):
        hidden = inputs_embeds.sin()
        return MoeV3ModelOutputWithPast(
            last_hidden_state=hidden,
            mtp_hidden_states=[hidden * 0.5],
        )


def _inputs_embeds():
    return {"inputs_embeds": torch.randn(2, 5, _HIDDEN_SIZE)}


def _bailing_hybrid(num_mtp=0):
    model = object.__new__(TurboBailingMoeV3ForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        output_attentions=False,
        output_hidden_states=False,
        output_router_logits=False,
        use_return_dict=True,
        vocab_size=_VOCAB_SIZE,
    )
    model.model = _BailingBackbone()
    model.lm_head = nn.Linear(_HIDDEN_SIZE, _VOCAB_SIZE, bias=False)
    model.num_nextn_predict_layers = num_mtp
    model.mtp_loss_scaling_factor = 0.1
    return model, _inputs_embeds()


def _deepseek_v4():
    return _make_deepseek_v4(), {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}


def _glm_moe_dsa():
    model = object.__new__(TurboGlmMoeDsaForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(use_return_dict=True)
    model.model = _Backbone()
    model.lm_head = nn.Linear(_HIDDEN_SIZE, _VOCAB_SIZE, bias=False)
    return model, _inputs_embeds()


def _kimi_k25():
    config = SimpleNamespace(
        output_attentions=False,
        output_hidden_states=False,
        use_return_dict=True,
    )
    language_model = object.__new__(DeepseekV3ForCausalLM)
    nn.Module.__init__(language_model)
    language_model.config = config
    language_model.model = _Backbone()
    language_model.lm_head = nn.Linear(_HIDDEN_SIZE, _VOCAB_SIZE, bias=False)

    model = object.__new__(TurboKimiK25ForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = config
    model.vision_tower = nn.Identity()
    model.language_model = language_model
    return model, _inputs_embeds()


def _llada2_moe():
    model = object.__new__(TurboLLaDA2MoeForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        output_attentions=False,
        output_hidden_states=False,
        output_router_logits=False,
        use_return_dict=True,
    )
    model.model = _MoeBackbone()
    model.lm_head = nn.Linear(_HIDDEN_SIZE, _VOCAB_SIZE, bias=False)
    return model, _inputs_embeds()


@pytest.mark.parametrize(
    "selector",
    [
        0,
        2,
        torch.tensor([False, True, False, True, True]),
        torch.tensor([4, 1, 3]),
        # A sequence-parallel shard may hold no target position.
        torch.zeros(5, dtype=torch.bool),
    ],
)
@pytest.mark.parametrize(
    "make_model",
    [_bailing_hybrid, _deepseek_v4, _glm_moe_dsa, _kimi_k25, _llada2_moe],
)
def test_selected_logits_match_full_projection(make_model, selector):
    torch.manual_seed(0)
    model, inputs = make_model()
    indices = slice(-selector, None) if isinstance(selector, int) else selector
    expected_logits = model(**inputs).logits[:, indices]

    projected_lengths = []
    hook = model.get_lm_head().register_forward_hook(
        lambda module, args, output: projected_lengths.append(output.shape[1])
    )
    logits = model(**inputs, logits_to_keep=selector).logits
    hook.remove()

    torch.testing.assert_close(logits, expected_logits)
    # The head projects the selected positions only.
    assert projected_lengths == [expected_logits.shape[1]]


def test_bailing_hybrid_mtp_logits_follow_the_selection():
    torch.manual_seed(0)
    model, inputs = _bailing_hybrid(num_mtp=1)
    selector = torch.tensor([False, True, False, True, True])
    expected = model(**inputs)

    outputs = model(**inputs, logits_to_keep=selector)

    torch.testing.assert_close(outputs.logits, expected.logits[:, selector])
    torch.testing.assert_close(
        outputs.mtp_logits[0], expected.mtp_logits[0][:, selector]
    )
