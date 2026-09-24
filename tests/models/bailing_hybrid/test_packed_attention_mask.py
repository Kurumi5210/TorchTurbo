import sys
from types import SimpleNamespace

import pytest
import torch

import torchturbo.ops as turbo_ops
from torchturbo.models.bailing_hybrid import modeling_bailing_moe_v3 as modeling
from torchturbo.models.bailing_hybrid.model import TurboBailingMoeV3ForCausalLM


class _CapturedKDA(Exception):
    pass


@pytest.mark.parametrize(
    ("mask", "expected_cu_seqlens"),
    [
        ([1, 1, 2, 2], [0, 2, 4]),
        ([1, 1, 2, 2, 0], [0, 2, 4, 5]),
    ],
)
def test_model_preserves_packed_boundaries_for_kda(monkeypatch, mask, expected_cu_seqlens):
    captured = []

    def capture_conv(**kwargs):
        cu_seqlens = kwargs["cu_seqlens"]
        captured.append(None if cu_seqlens is None else cu_seqlens.tolist())
        return kwargs["x"], None

    def capture_kda(**kwargs):
        cu_seqlens = kwargs["cu_seqlens"]
        captured.append(None if cu_seqlens is None else cu_seqlens.tolist())
        raise _CapturedKDA

    monkeypatch.setattr(turbo_ops, "ops", SimpleNamespace(causal_conv1d=capture_conv))
    monkeypatch.setitem(
        sys.modules, "fla.modules.fused_norm_gate", SimpleNamespace(rms_norm_gated=None)
    )
    monkeypatch.setitem(sys.modules, "fla.ops.kda", SimpleNamespace(chunk_kda=capture_kda))
    attention_type = modeling.BailingMoeV3KimiDeltaAttention
    monkeypatch.setattr(attention_type, "forward", attention_type.forward)
    TurboBailingMoeV3ForCausalLM.patch_kda_forward(None, 1, 0)

    def identity(x):
        return x

    conv = SimpleNamespace(weight=torch.ones(2, 1, 2), bias=None, activation="silu")
    attention = SimpleNamespace(
        mode="chunk",
        q_proj=identity,
        k_proj=identity,
        v_proj=identity,
        q_conv1d=conv,
        k_conv1d=conv,
        v_conv1d=conv,
        no_kda_lora=True,
        f_proj=identity,
        b_proj=lambda x: x[..., :1],
        head_k_dim=2,
        head_dim=2,
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(2),
        safe_gate=True,
        lower_bound=-5,
    )
    decoder = SimpleNamespace(
        input_layernorm=identity,
        attention_layer_type="linear_attention",
        attention=lambda **kwargs: attention_type.forward(attention, **kwargs),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            output_attentions=False,
            output_hidden_states=False,
            output_router_logits=False,
            use_cache=False,
            use_return_dict=False,
            layer_group_size=2,
        ),
        gradient_checkpointing=False,
        training=True,
        _use_flash_attention_2=True,
        _use_sdpa=False,
        num_nextn_predict_layers=0,
        rotary_emb=lambda hidden, positions: None,
        layers=[
            lambda *args, **kwargs: modeling.BailingMoeV3DecoderLayer.forward(
                decoder, *args, **kwargs
            )
        ],
    )
    positions = [0, 1, 0, 1] + [0] * (len(mask) - 4)

    with pytest.raises(_CapturedKDA):
        modeling.BailingMoeV3Model.forward(
            model,
            inputs_embeds=torch.ones(1, len(mask), 2),
            attention_mask=torch.tensor([mask]),
            position_ids=torch.tensor([positions]),
        )

    # The three causal convolutions and KDA must all reset at sample boundaries.
    assert captured == [expected_cu_seqlens] * 4
