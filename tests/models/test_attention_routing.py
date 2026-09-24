from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.mark.parametrize(
    ("model_name", "class_prefix"),
    [("qwen3_5", "Qwen3_5"), ("qwen3_5_moe", "Qwen3_5Moe")],
)
@pytest.mark.parametrize("packed", [False, True])
def test_vision_uses_local_attention_and_preserves_metadata(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    class_prefix: str,
    packed: bool,
) -> None:
    from transformers.integrations import flash_attention

    from torchturbo.models.modules.attention import interface

    modeling = importlib.import_module(f"torchturbo.models.{model_name}.modeling_{model_name}")
    model_module = importlib.import_module(f"torchturbo.models.{model_name}.model")
    attention_type = vars(modeling)[f"{class_prefix}VisionAttention"]
    model_type = vars(model_module)[f"Turbo{class_prefix}ForConditionalGeneration"]
    origin_attr = f"_torchturbo_{model_name}_vision_origin_attention_interface"
    monkeypatch.setattr(attention_type, origin_attr, None, raising=False)
    monkeypatch.setattr(
        attention_type, "attention_interface", attention_type.__dict__["attention_interface"]
    )

    query = torch.randn(1, 2, 5, 3)
    output = query.transpose(1, 2)
    local_attention = Mock(return_value=output)
    global_attention = Mock(side_effect=AssertionError("Vision entered the global SP patch."))
    monkeypatch.setattr(interface, "attention_forward", local_attention)
    monkeypatch.setattr(flash_attention, "_flash_attention_forward", global_attention)
    model_type.patch_vision_attention_interface()

    attention_mask = None if packed else torch.ones(1, 5, dtype=torch.long)
    kwargs = {"deterministic": True}
    if packed:
        cumulative_lengths = torch.tensor([0, 2, 5], dtype=torch.int32)
        kwargs.update(
            scaling=0.5,
            dropout=0.15,
            cu_seq_lens_q=cumulative_lengths,
            cu_seq_lens_k=cumulative_lengths,
            max_length_q=3,
            max_length_k=3,
            is_causal=False,
        )

    actual, weights = attention_type.attention_interface(
        SimpleNamespace(), query, query, query, attention_mask, **kwargs
    )

    assert actual is output
    assert weights is None
    global_attention.assert_not_called()
    local_attention.assert_called_once()
    call = local_attention.call_args
    for tensor in call.args[:3]:
        torch.testing.assert_close(tensor, query.transpose(1, 2))
    assert call.args[3] is attention_mask
    expected_kwargs = {
        "query_length": 5,
        "is_causal": False,
        "dropout": 0.15 if packed else 0.0,
        "position_ids": None,
        "softmax_scale": 0.5 if packed else None,
        "deterministic": True,
    }
    if packed:
        expected_kwargs.update(
            cu_seq_lens_q=cumulative_lengths,
            cu_seq_lens_k=cumulative_lengths,
            max_length_q=3,
            max_length_k=3,
        )
    assert call.kwargs == expected_kwargs


@pytest.mark.parametrize("sp_size", [1, 2])
def test_swift_uses_the_shared_sp_patch(monkeypatch: pytest.MonkeyPatch, sp_size: int) -> None:
    from torchturbo.adapters.msswift.trainer import apply_ulysses_sp_patch
    from torchturbo.models.modules.attention import ulysses

    sp_info = {"sp_group": object(), "sp_size": sp_size}
    shared_patch = Mock()
    monkeypatch.setattr(ulysses, "patch_flash_attention_forward_for_sp", shared_patch)
    engine = SimpleNamespace(get_sp_info=lambda: sp_info)

    assert apply_ulysses_sp_patch(engine) is (sp_size > 1)
    if sp_size > 1:
        shared_patch.assert_called_once_with(sp_info)
    else:
        shared_patch.assert_not_called()
