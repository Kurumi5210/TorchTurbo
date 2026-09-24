"""CPU tests for Swift collation and the packed cu/max input contract."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

import torch  # noqa: E402

from torchturbo.adapters.msswift.inputs_bridging import (  # noqa: E402
    SwiftInputCollator,
    bake_loss_scale_into_labels,
    bridge_packed_inputs,
    bridge_swift_inputs,
)


def _t(values):
    return torch.tensor(values)


def _packed_inputs(lengths):
    positions = [position for length in lengths for position in range(length)]
    return {
        "input_ids": torch.arange(len(positions)).unsqueeze(0),
        "position_ids": _t([positions]),
    }


@pytest.mark.parametrize("lengths,expected", [([3, 2], [0, 3, 5]), ([1], [0, 1]), ([2, 2, 3], [0, 2, 4, 7])])
def test_sample_lengths_produce_cu_max(lengths, expected):
    inputs = _packed_inputs(lengths)
    positions = inputs["position_ids"]
    out = bridge_packed_inputs(inputs, sequence_lengths=lengths)
    assert "attention_mask" not in out
    assert out["cu_seq_lens_q"].tolist() == expected
    assert out["cu_seq_lens_k"].tolist() == expected
    assert out["cu_seq_lens_q"].dtype == torch.int32
    assert out["max_length_q"] == out["max_length_k"] == max(lengths)
    assert out["position_ids"] is positions


def test_preserves_explicit_cu_and_mrope_coordinates():
    cu = _t([0, 3, 5]).int()
    positions = _t([[[0, 1, 1, 0, 1]], [[0, 1, 2, 0, 1]], [[0, 1, 1, 0, 1]]])
    inputs = {
        "input_ids": _t([[10, 11, 12, 13, 14]]),
        "position_ids": positions,
        "text_position_ids": _t([[0, 1, 2, 0, 1]]),
        "cu_seq_lens_q": cu,
        "cu_seq_lens_k": cu,
        "max_length_q": _t(3),
        "max_length_k": _t(3),
    }
    out = bridge_packed_inputs(inputs, sequence_lengths=[3, 2])
    assert out["cu_seq_lens_q"] is cu
    assert out["cu_seq_lens_k"] is cu
    assert out["position_ids"] is positions
    assert type(out["max_length_q"]) is int
    assert "attention_mask" not in out


def test_longlora_tail_padding_is_not_counted_as_samples():
    inputs = {
        "input_ids": _t([[10, 11, 12, 13, 14, 0, 0]]),
        "position_ids": _t([[0, 1, 0, 1, 2, 0, 0]]),
        "cu_seq_lens_q": _t([0, 2, 5, 6, 7]).int(),
        "cu_seq_lens_k": _t([0, 2, 5, 6, 7]).int(),
        "max_length_q": _t(3),
        "max_length_k": _t(3),
    }
    out = bridge_packed_inputs(inputs, sequence_lengths=[2, 3])
    assert out["input_ids"].shape == (1, 7)
    assert out["cu_seq_lens_q"].tolist() == [0, 2, 5]
    assert "attention_mask" not in out


def test_packed_left_padding_moves_to_tail_with_all_token_fields():
    inputs = {
        "input_ids": _t([[0, 0, 10, 11, 12, 13, 14]]),
        "labels": _t([[-100, -100, 10, 11, 12, 13, 14]]),
        "position_ids": _t([[[0, 0, 0, 1, 0, 1, 2]]] * 3),
        "text_position_ids": _t([[0, 0, 0, 1, 0, 1, 2]]),
        "loss_scale": _t([[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]]),
        "mm_token_type_ids": _t([[0, 0, 0, 1, 0, 0, 0]]),
        "cu_seq_lens_q": _t([0, 1, 2, 4, 7]).int(),
        "cu_seq_lens_k": _t([0, 1, 2, 4, 7]).int(),
        "max_length_q": _t(3),
        "max_length_k": _t(3),
        "pixel_values": torch.randn(4, 6),
    }
    pixels = inputs["pixel_values"]
    out = bridge_swift_inputs(inputs, sequence_lengths=[2, 3], padding_side="left")
    assert out["input_ids"].tolist() == [[10, 11, 12, 13, 14, 0, 0]]
    assert out["labels"].tolist() == [[10, -100, 12, 13, 14, -100, -100]]
    assert out["text_position_ids"].tolist() == [[0, 1, 0, 1, 2, 0, 0]]
    assert out["position_ids"][0].tolist() == out["text_position_ids"].tolist()
    assert out["mm_token_type_ids"].tolist() == [[0, 1, 0, 0, 0, 0, 0]]
    assert out["cu_seq_lens_q"].tolist() == [0, 2, 5]
    assert out["pixel_values"] is pixels


def test_nonpacked_left_padding_is_unchanged():
    inputs = {
        "input_ids": _t([[0, 0, 10, 11], [0, 12, 13, 14]]),
        "position_ids": _t([[0, 0, 0, 1], [0, 0, 1, 2]]),
        "attention_mask": _t([[0, 0, 1, 1], [0, 1, 1, 1]]),
    }
    original = dict(inputs)
    out = bridge_swift_inputs(inputs)
    assert set(out) == set(original)
    assert all(out[key] is value for key, value in original.items())


def test_rejects_native_cu_with_different_sample_boundaries():
    inputs = _packed_inputs([2, 3])
    inputs["cu_seq_lens_q"] = _t([0, 3, 5]).int()
    with pytest.raises(ValueError, match="original Swift sample boundaries"):
        bridge_packed_inputs(inputs, sequence_lengths=[2, 3])


def test_rejects_position_resets_inconsistent_with_samples():
    inputs = _packed_inputs([3, 2])
    with pytest.raises(ValueError, match="position resets"):
        bridge_packed_inputs(inputs, sequence_lengths=[2, 3])


def test_rejects_incorrect_native_maximum_length():
    inputs = _packed_inputs([2, 3])
    inputs["max_length_q"] = _t(5)
    with pytest.raises(ValueError, match="longest packed sample"):
        bridge_packed_inputs(inputs, sequence_lengths=[2, 3])


@pytest.mark.parametrize("nested", [False, True])
def test_collator_captures_lengths_before_native_batch_mutation(nested):
    samples = [{"input_ids": [10, 11]}, {"input_ids": [12, 13, 14]}]
    batch = [[samples[0]], [samples[1]]] if nested else samples
    template = SimpleNamespace(padding_free=True, padding_side="right", is_training=True)

    def native_collator(rows):
        rows[:] = [{"input_ids": [10, 11, 12, 13, 14]}]
        return {
            "input_ids": _t([[10, 11, 12, 13, 14]]),
            "position_ids": _t([[0, 1, 0, 1, 2]]),
            "labels": _t([[10, 11, 12, 13, 14]]),
            "loss_scale": _t([[1.0, 0.0, 1.0, 1.0, 1.0]]),
        }

    out = SwiftInputCollator(native_collator, template)(batch)
    assert out["cu_seq_lens_q"].tolist() == [0, 2, 5]
    assert out["labels"].tolist() == [[10, -100, 12, 13, 14]]
    assert "attention_mask" not in out
    assert "loss_scale" not in out
    assert out["cu_seq_lens_q"].device.type == "cpu"


def test_nonpacked_collator_preserves_binary_mask():
    inputs = {
        "input_ids": _t([[0, 0, 10, 11]]),
        "position_ids": _t([[0, 0, 0, 1]]),
        "attention_mask": _t([[0, 0, 1, 1]]),
    }
    template = SimpleNamespace(padding_free=False, padding_side="left", is_training=True)
    out = SwiftInputCollator(lambda batch: inputs, template)([{"input_ids": [10, 11]}])
    assert out is inputs
    assert out["attention_mask"].tolist() == [[0, 0, 1, 1]]
    assert "cu_seq_lens_q" not in out


def test_loss_scale_zeros_masked_into_labels():
    labels = _t([[10, 11, 12, 13, 14]])
    loss_scale = _t([[1.0, 0.0, 1.0, 0.0, 1.0]])  # ignore tokens at idx 1,3
    inputs = {"labels": labels.clone(), "loss_scale": loss_scale}
    out = bake_loss_scale_into_labels(inputs)
    assert out["labels"].tolist() == [[10, -100, 12, -100, 14]]
    assert "loss_scale" not in out  # removed after folding


def test_loss_scale_all_positive_passthrough_and_drops_loss_scale():
    labels = _t([[10, 11, 12]])
    inputs = {"labels": labels.clone(), "loss_scale": _t([[1.0, 1.0, 1.0]])}
    out = bake_loss_scale_into_labels(inputs)
    assert out["labels"].tolist() == [[10, 11, 12]]
    assert "loss_scale" not in out


def test_loss_scale_absent_noop():
    inputs = {"labels": _t([[1, 2, 3]])}
    out = bake_loss_scale_into_labels(inputs)
    assert out is inputs
    assert out["labels"].tolist() == [[1, 2, 3]]


def test_loss_scale_no_labels_noop():
    inputs = {"loss_scale": _t([[1.0, 0.0]])}
    out = bake_loss_scale_into_labels(inputs)
    assert out is inputs  # nothing to fold into


def test_loss_scale_non_binary_warns_but_still_masks_zeros(caplog):
    labels = _t([[10, 11, 12, 13]])
    loss_scale = _t([[1.0, 0.5, 0.0, 0.3]])  # values in (0,1) -> non-binary
    inputs = {"labels": labels.clone(), "loss_scale": loss_scale}
    out = bake_loss_scale_into_labels(inputs)
    # zeros still masked; non-binary fractional weights dropped (warned).
    assert out["labels"].tolist() == [[10, 11, -100, 13]]
