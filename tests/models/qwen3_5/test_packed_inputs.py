import importlib
from types import SimpleNamespace

import pytest
import torch

import torchturbo.ops as turbo_ops
from torchturbo.models.qwen3_5 import modeling_qwen3_5 as dense
from torchturbo.models.qwen3_5.model import TurboQwen3_5ForConditionalGeneration
from torchturbo.models.qwen3_5_moe import modeling_qwen3_5_moe as moe
from torchturbo.models.qwen3_5_moe.model import TurboQwen3_5MoeForConditionalGeneration
from torchturbo.ops.dispatcher import OpDispatcher


@pytest.mark.parametrize("text_model", [dense.Qwen3_5TextModel, moe.Qwen3_5MoeTextModel])
def test_text_model_preserves_generation_padding_mask(text_model):
    captured = []

    class Decoder:
        layer_type = "full_attention"

        def __call__(self, hidden_states, **kwargs):
            captured.append(kwargs["attention_mask"])
            return hidden_states

    model = SimpleNamespace(
        config=SimpleNamespace(use_cache=False, num_hidden_layers=1),
        gradient_checkpointing=False,
        training=False,
        layers=[Decoder()],
        rotary_emb=lambda *args: None,
        norm=lambda x: x,
    )
    mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])

    text_model.forward(model, inputs_embeds=torch.ones(2, 4, 4), attention_mask=mask)

    assert captured == [mask]


def _attention():
    return SimpleNamespace(
        num_k_heads=2,
        num_v_heads=2,
        head_k_dim=2,
        head_v_dim=2,
        key_dim=4,
        value_dim=4,
        conv_kernel_size=2,
        conv1d=SimpleNamespace(weight=torch.ones(12, 1, 2), bias=None),
        activation="silu",
        in_proj_qkv=lambda x: x.repeat(1, 1, 3),
        in_proj_z=lambda x: x,
        in_proj_b=lambda x: x[..., :2],
        in_proj_a=lambda x: x[..., :2],
        A_log=torch.zeros(2),
        dt_bias=torch.zeros(2),
        norm=lambda x, z: x,
        out_proj=lambda x: x,
        _sp_a2a_stream=None,
    )


@pytest.mark.parametrize("family", ["dense", "moe"])
@pytest.mark.parametrize("sp_size", [1, 2])
def test_gated_delta_ops_cover_padding_and_share_metadata(monkeypatch, family, sp_size):
    captured = []

    def dispatch(op_name, *args, **kwargs):
        packed = kwargs["packed_params"]
        assert kwargs.get("cu_seqlens") is None
        assert packed.cu_seqlens_list[-1] == args[0].size(1)
        captured.append(packed)
        if op_name == "causal_conv1d":
            return args[0], None
        assert op_name == "gated_delta_rule"
        return args[2], None

    # Keep public op normalization real; replace only the unavailable kernels.
    dispatcher = OpDispatcher.__new__(OpDispatcher)
    monkeypatch.setattr(dispatcher, "_dispatch", dispatch)
    monkeypatch.setattr(turbo_ops, "ops", dispatcher)
    attention = _attention()
    if family == "dense":
        attention_cls = dense.Qwen3_5GatedDeltaNet
        wrapper = TurboQwen3_5ForConditionalGeneration
    else:
        attention_cls = moe.Qwen3_5MoeGatedDeltaNet
        wrapper = TurboQwen3_5MoeForConditionalGeneration

    if sp_size > 1:
        ulysses = importlib.import_module("torchturbo.models.modules.attention.ulysses.ulysses")

        def gather_sequence(x, seq_dim, head_dim, **kwargs):
            return torch.cat([x, x], dim=seq_dim).narrow(head_dim, 0, x.size(head_dim) // 2)

        def gather_heads(x, seq_dim, head_dim, **kwargs):
            return torch.cat([x, x], dim=head_dim).narrow(seq_dim, 0, x.size(seq_dim) // 2)

        monkeypatch.setattr(ulysses, "gather_seq_scatter_heads", gather_sequence)
        monkeypatch.setattr(ulysses, "gather_heads_scatter_seq", gather_heads)
        runtime = SimpleNamespace(current_stream=lambda: SimpleNamespace(wait_stream=lambda stream: None))
        dense_wrapper = importlib.import_module("torchturbo.models.qwen3_5.model")
        monkeypatch.setattr(dense_wrapper, "get_device_runtime", lambda: runtime)
        monkeypatch.setattr(attention_cls, "forward", attention_cls.forward)
        wrapper.patch_gated_delta_net_for_sp(attention_cls, None, sp_size, 0)

    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    output = attention_cls.forward(attention, torch.ones(1, 6 // sp_size, 4), cu_seq_lens=cu)

    assert output.shape == (1, 6 // sp_size, 4)
    assert captured[0] is captured[1]
    assert captured[0].cu_seqlens_list == [0, 2, 5, 6]
    assert cu.tolist() == [0, 2, 5]


def _rope_index_model():
    config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=1),
        image_token_id=99,
        video_token_id=98,
    )
    model = SimpleNamespace(config=config)
    model.get_vision_position_ids = lambda *args, **kwargs: moe.Qwen3_5MoeModel.get_vision_position_ids(model, *args, **kwargs)
    model.get_rope_index = lambda *args, **kwargs: moe.Qwen3_5MoeModel.get_rope_index(model, *args, **kwargs)
    TurboQwen3_5MoeForConditionalGeneration._post_init_patches(SimpleNamespace(model=model, config=config))
    return model


def test_moe_rope_index_restarts_positions_per_mask_block():
    position_ids, _ = _rope_index_model().get_rope_index(
        input_ids=torch.tensor([[10, 99, 99, 11, 12, 99, 99, 13, 0]]),
        image_grid_thw=torch.tensor([[1, 1, 2], [1, 1, 2]]),
        attention_mask=torch.tensor([[1, 1, 1, 1, 2, 2, 2, 2, 0]]),
    )

    expected = torch.tensor([
        [[0, 1, 1, 3, 0, 1, 1, 3, 0]],
        [[0, 1, 1, 3, 0, 1, 1, 3, 0]],
        [[0, 1, 2, 3, 0, 1, 2, 3, 0]],
    ])
    torch.testing.assert_close(position_ids, expected)


def test_moe_rope_index_aligns_truncated_images_from_collator_masks():
    # The LF collator passes a binary float mask; truncation can keep only a
    # prefix of an image's placeholder tokens.
    with pytest.warns(UserWarning, match="placeholder_len=2 != grid_len=4"):
        position_ids, _ = _rope_index_model().get_rope_index(
            input_ids=torch.tensor([[10, 11, 99, 99]]),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
            attention_mask=torch.ones(1, 4),
        )

    expected = torch.tensor([[[0, 1, 2, 2]], [[0, 1, 2, 2]], [[0, 1, 2, 3]]])
    torch.testing.assert_close(position_ids, expected)


def test_moe_mtp_attention_receives_prefix_boundaries():
    captured = {}

    def attention(hidden_states, **kwargs):
        captured.update(kwargs)
        return hidden_states, None

    layer = SimpleNamespace(
        input_layernorm=lambda x: x,
        self_attn=attention,
        post_attention_layernorm=lambda x: x,
        mlp=lambda x: x,
    )
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    moe.Qwen3_5MoeMTPLayer.forward(
        layer,
        torch.ones(1, 6, 4),
        cu_seq_lens_q=cu,
        cu_seq_lens_k=cu,
        max_length_q=3,
        max_length_k=3,
    )

    assert captured["cu_seq_lens_q"] is cu
    assert captured["cu_seq_lens_k"] is cu
    assert captured["max_length_q"] == captured["max_length_k"] == 3
