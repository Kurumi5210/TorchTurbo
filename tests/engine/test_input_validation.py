"""Checks for the engine's packed SFT input contract."""

import pytest
import torch

from torchturbo.engine.utils import assert_packed_inputs


def _batch():
    return {
        "input_ids": torch.tensor([[11, 12, 21, 22, 23, 99]]),
        "labels": torch.tensor([[-100, 12, -100, 22, 23, -100]]),
        "position_ids": torch.tensor([[0, 1, 0, 1, 2, 0]]),
        "cu_seq_lens_q": torch.tensor([0, 2, 5], dtype=torch.int32),
        "cu_seq_lens_k": torch.tensor([0, 2, 5], dtype=torch.int32),
        "max_length_q": 3,
        "max_length_k": 3,
    }


def test_valid_prefix_with_masked_tail():
    assert_packed_inputs(_batch())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("attention_mask", torch.ones(1, 6), "without attention_mask"),
        ("input_ids", torch.ones(2, 3), "shape"),
        ("position_ids", torch.zeros(1, 5), "align"),
        ("position_ids", torch.tensor([[0, 1, 2, 0, 1, 0]]), "resets"),
        ("labels", torch.zeros(2, 3), "labels"),
        ("cu_seq_lens_k", torch.tensor([0, 3, 5], dtype=torch.int32), "identical"),
        ("cu_seq_lens_q", torch.tensor([0, 2, 5]), "int32"),
        ("max_length_q", 2, "longest"),
        ("max_length_k", torch.tensor(3), "as an int"),
    ],
)
def test_rejects_inconsistent_contract(field, value, message):
    batch = _batch()
    batch[field] = value
    with pytest.raises(ValueError, match=message):
        assert_packed_inputs(batch)


@pytest.mark.parametrize("boundaries", [[1, 2, 5], [0, 2, 2], [0, 2, 7]])
def test_rejects_invalid_sample_boundaries(boundaries):
    batch = _batch()
    batch["cu_seq_lens_q"] = batch["cu_seq_lens_k"] = torch.tensor(boundaries, dtype=torch.int32)
    with pytest.raises(ValueError, match="start at zero and increase"):
        assert_packed_inputs(batch)


def test_mrope_coordinates_remain_independent_of_text_boundaries():
    batch = _batch()
    batch["text_position_ids"] = batch["position_ids"]
    positions = torch.tensor([[[0, 0, 1, 1, 1, 0]]]).expand(3, 1, 6).clone()
    batch["position_ids"] = positions

    assert_packed_inputs(batch)

    assert batch["position_ids"] is positions
