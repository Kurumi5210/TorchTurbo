"""Early validation of the explicit chunked-loss model contract."""

import pytest
import torch.nn as nn

from torchturbo.engine.utils import validate_signature


class _SupportedModel(nn.Module):
    def forward(self, inputs, *, skip_lm_head=False):
        return inputs


class _MissingModel(nn.Module):
    def forward(self, inputs, **kwargs):
        return inputs


class _RequiredFlagModel(nn.Module):
    def forward(self, inputs, skip_lm_head):
        return inputs


class _PositionalFlagModel(nn.Module):
    def forward(self, skip_lm_head=False, /):
        return skip_lm_head


def test_accepts_explicit_optional_keyword_parameter():
    validate_signature(_SupportedModel.forward, skip_lm_head=False)


@pytest.mark.parametrize(
    "model_cls",
    [_MissingModel, _RequiredFlagModel, _PositionalFlagModel],
)
def test_rejects_models_without_exact_explicit_contract(model_cls):
    with pytest.raises(TypeError, match="skip_lm_head"):
        validate_signature(model_cls.forward, skip_lm_head=False)
