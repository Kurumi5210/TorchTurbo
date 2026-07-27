from types import SimpleNamespace

import torch
from torch import nn

from torchturbo.models.bailing_hybrid.model import TurboBailingMoeV3ForCausalLM


class _FakeRotaryEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        with torch.device("meta"):
            self.register_buffer(
                "inv_freq",
                torch.empty(2),
                persistent=False,
            )
        self.attention_scaling = 1.0
        self.config = object()
        self.rope_init_fn = lambda config, device: (
            torch.ones(2, device=device),
            1.0,
        )


class _MaterializedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = SimpleNamespace(rotary_emb=_FakeRotaryEmbedding())
        self.param = nn.Parameter(torch.empty(1))


def test_on_materialized_rebuilds_meta_rotary_buffer():
    model = _MaterializedModel()

    TurboBailingMoeV3ForCausalLM.on_materialized(model)

    inv_freq = model.model.rotary_emb.inv_freq
    assert inv_freq.is_meta is False
    assert inv_freq.device == torch.device("cpu")
