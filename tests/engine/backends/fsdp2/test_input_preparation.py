"""FSDP2-owned Ulysses input alignment tests."""

from types import SimpleNamespace

import pytest
import torch

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend


def _backend(sp_size: int) -> FSDP2Backend:
    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend.get_sp_info = lambda: {
        "sp_size": sp_size,
        "sp_group": None,
        "sp_rank": 0,
    }
    return backend


def _model(pad_token_id: int = 99):
    return SimpleNamespace(config=SimpleNamespace(pad_token_id=pad_token_id))


def test_prepare_inputs_aligns_all_sequence_fields_and_packed_metadata():
    seq_len = 5
    position_ids = torch.zeros(4, 1, seq_len, dtype=torch.long)
    position_ids[0, 0] = torch.tensor([0, 1, 0, 1, 2])
    batch = {
        "input_ids": torch.arange(seq_len).reshape(1, -1),
        "labels": torch.arange(10, 10 + seq_len).reshape(1, -1),
        "position_ids": position_ids,
        "text_position_ids": torch.tensor([[0, 1, 0, 1, 2]]),
        "token_type_ids": torch.ones(1, seq_len, dtype=torch.long),
        "mm_token_type_ids": torch.ones(1, seq_len, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 2, 2, 2]]),
        "cu_seq_lens_q": torch.tensor([0, 2, 5], dtype=torch.int32),
        "cu_seq_lens_k": torch.tensor([0, 2, 5], dtype=torch.int32),
        "max_length_q": 3,
        "max_length_k": 3,
    }

    result = _backend(sp_size=3).prepare_inputs(_model(), batch)

    assert result["input_ids"].shape == (1, 6)
    assert result["labels"].shape == (1, 6)
    assert result["position_ids"].shape == (4, 1, 6)
    assert result["text_position_ids"].shape == (1, 6)
    assert result["token_type_ids"].shape == (1, 6)
    assert result["mm_token_type_ids"].shape == (1, 6)
    assert result["input_ids"][0, -1].item() == 99
    assert result["labels"][0, -1].item() == -100
    assert result["attention_mask"].tolist() == [[1, 1, 2, 2, 2, 0]]
    assert result["token_type_ids"][0, -1].item() == 0
    assert result["mm_token_type_ids"][0, -1].item() == 0
    assert result["cu_seq_lens_q"].tolist() == [0, 2, 6]
    assert result["cu_seq_lens_k"].tolist() == [0, 2, 6]
    assert result["max_length_q"] == 4
    assert result["max_length_k"] == 4


def test_prepare_inputs_creates_padding_mask_when_missing():
    result = _backend(sp_size=4).prepare_inputs(
        _model(),
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "labels": torch.tensor([[1, 2, 3]]),
        },
    )

    assert result["attention_mask"].tolist() == [[1, 1, 1, 0]]


def test_prepare_inputs_rejects_misaligned_2d_mask():
    with pytest.raises(ValueError, match="attention_mask must be aligned"):
        _backend(sp_size=4).prepare_inputs(
            _model(),
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "labels": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.ones(1, 2),
            },
        )
