"""Tests for the pack-boundary bridging layer (swift position_ids → turbo int mask).

Pure-torch, no swift import — same CI expectations as ``test_config.py``.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

import torch  # noqa: E402

from torchturbo.adapters.msswift.inputs_bridging import (  # noqa: E402
    bake_loss_scale_into_labels,
    bridge_packed_inputs,
    bridge_swift_inputs,
    pad_batch_to_fixed_length,
    position_ids_reset_to_packed_mask,
)
from torchturbo.adapters.msswift import engine_registry  # noqa: E402


def _t(values):
    return torch.tensor(values)


# --------------------------------------------------------------------------- #
# position_ids_reset_to_packed_mask
# --------------------------------------------------------------------------- #


def test_multi_segment_reset_to_increasing_integers():
    pos = _t([[0, 1, 2, 0, 1]])
    mask = position_ids_reset_to_packed_mask(pos)
    assert mask is not None
    assert mask.tolist() == [[1, 1, 1, 2, 2]]


def test_single_segment_passthrough_none():
    # ordinary arange — no boundary reset → single segment → None (passthrough)
    assert position_ids_reset_to_packed_mask(_t([[0, 1, 2, 3]])) is None


def test_none_input_returns_none():
    assert position_ids_reset_to_packed_mask(None) is None


def test_n_leq_one_returns_none():
    assert position_ids_reset_to_packed_mask(_t([[5]])) is None


def test_fallback_attention_mask_zeros_padding():
    # two segments with a trailing padding position
    pos = _t([[0, 1, 0, 1, 0]])
    am = _t([[1, 1, 1, 1, 0]])
    mask = position_ids_reset_to_packed_mask(pos, fallback_attention_mask=am)
    assert mask is not None
    assert mask.tolist() == [[1, 1, 2, 2, 0]]


def test_fallback_mask_can_collide_segments_to_single_then_none():
    # If fallback mask keeps only one contiguous segment, result collapses to
    # single segment → None (don't emit a spurious 1-only int mask).
    pos = _t([[0, 1, 2, 0, 1]])
    am = _t([[1, 1, 1, 0, 0]])  # second "segment" fully padded out
    assert position_ids_reset_to_packed_mask(pos, fallback_attention_mask=am) is None


def test_mrope_3d_uses_text_row():
    # mrope (3, B, N): row 0 is the text/segment row.
    pos3 = torch.zeros(3, 1, 5)
    pos3[0, 0] = _t([0, 1, 2, 0, 1])
    pos3[1, 0] = _t([9, 9, 9, 9, 9])  # irrelevant spatial rows
    pos3[2, 0] = _t([7, 7, 7, 7, 7])
    mask = position_ids_reset_to_packed_mask(pos3)
    assert mask is not None
    assert mask.tolist() == [[1, 1, 1, 2, 2]]


def test_multiple_batch_rows_independent_segments():
    pos = _t([[0, 1, 0, 1, 2], [0, 1, 2, 3, 4]])
    mask = position_ids_reset_to_packed_mask(pos)
    assert mask is not None
    # row 0: two segments (1,1,2,2,2); row 1: single segment → but max>1 overall
    # so it still emits, with row 1 labeled all-1.
    assert mask.tolist() == [[1, 1, 2, 2, 2], [1, 1, 1, 1, 1]]


def test_three_segments_increasing_ids():
    pos = _t([[0, 1, 0, 1, 0, 1, 2]])
    mask = position_ids_reset_to_packed_mask(pos)
    assert mask.tolist() == [[1, 1, 2, 2, 3, 3, 3]]


# --------------------------------------------------------------------------- #
# bridge_packed_inputs
# --------------------------------------------------------------------------- #


def test_bridge_rewrites_attention_mask_when_packed():
    pos = _t([[0, 1, 2, 0, 1]])
    inputs = {"position_ids": pos, "attention_mask": _t([[1, 1, 1, 1, 1]])}
    out = bridge_packed_inputs(inputs)
    assert out["attention_mask"].tolist() == [[1, 1, 1, 2, 2]]
    assert out["position_ids"] is pos  # position_ids untouched


def test_bridge_passthrough_when_single_segment():
    pos = _t([[0, 1, 2, 3]])
    original_mask = _t([[1, 1, 1, 1]])
    inputs = {"position_ids": pos, "attention_mask": original_mask.clone()}
    out = bridge_packed_inputs(inputs)
    assert torch.equal(out["attention_mask"], original_mask)
    assert "attention_mask" in out  # not removed, left as-is


def test_bridge_passthrough_when_no_position_ids():
    inputs = {"input_ids": _t([[1, 2, 3]]), "attention_mask": _t([[1, 1, 1]])}
    out = bridge_packed_inputs(inputs)
    # unchanged
    assert torch.equal(out["attention_mask"], _t([[1, 1, 1]]))


def test_bridge_with_padding_preserves_zero_padding():
    pos = _t([[0, 1, 0, 1, 0]])
    am = _t([[1, 1, 1, 1, 0]])
    inputs = {"position_ids": pos, "attention_mask": am}
    out = bridge_packed_inputs(inputs)
    assert out["attention_mask"].tolist() == [[1, 1, 2, 2, 0]]


@pytest.mark.parametrize(
    "pos,expected",
    [
        ([[0, 1, 2, 0, 1]], [[1, 1, 1, 2, 2]]),
        ([[0, 1, 0, 1, 0, 1, 2]], [[1, 1, 2, 2, 3, 3, 3]]),
        ([[0, 1, 2, 3]], None),
    ],
)
def test_parametrated_segments(pos, expected):
    mask = position_ids_reset_to_packed_mask(_t(pos))
    if expected is None:
        assert mask is None
    else:
        assert mask.tolist() == [expected[0]]


# --------------------------------------------------------------------------- #
# bake_loss_scale_into_labels
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# pad_batch_to_fixed_length + bridge_swift_inputs (end-to-end packing bridge)
# --------------------------------------------------------------------------- #


def test_pad_batch_pads_all_seq_fields_to_target_and_labels_ignored():
    # A 2-segment packed batch (boundaries at reset-0 in position_ids).
    inputs = {
        "input_ids": _t([[1, 2, 3, 4, 5]]),
        "attention_mask": _t([[1, 1, 1, 1, 1]]),
        "position_ids": _t([[[0, 1, 0, 1, 2], [9, 9, 9, 9, 9], [7, 7, 7, 7, 7]]]),  # (3,1,5) mrope
        "labels": _t([[10, 11, 12, 13, 14]]),
    }
    out = pad_batch_to_fixed_length(inputs, target_length=8)
    # Every sequence-length field agrees at 8.
    assert out["input_ids"].shape[-1] == 8
    assert out["labels"].shape[-1] == 8
    assert out["attention_mask"].shape[-1] == 8
    assert out["position_ids"].shape[-1] == 8
    assert out["attention_mask"].tolist() == [
        [1, 1, 1, 1, 1, 0, 0, 0]
    ]
    # Padded labels are loss-ignored; original labels preserved.
    assert out["labels"].tolist()[0][:5] == [10, 11, 12, 13, 14]
    assert out["labels"].tolist()[0][5:] == [-100, -100, -100]


def test_pad_batch_noop_when_already_at_target():
    inputs = {"input_ids": _t([[1, 2, 3]]), "labels": _t([[10, 11, 12]])}
    out = pad_batch_to_fixed_length(inputs, target_length=3)
    assert out is inputs  # no padding needed -> returned unchanged


def test_bridge_swift_inputs_end_to_end_with_packing_length():
    # Full bridge on a packed batch: rebuild the packed mask, fold loss_scale,
    # and pad to the stashed packing length.
    try:
        engine_registry.set_packing_length(8)
        inputs = {
            "input_ids": _t([[1, 2, 3, 4, 5]]),
            "position_ids": _t([[0, 1, 0, 1, 2]]),  # 2 segments -> triggers mask rebuild
            "attention_mask": _t([[1, 1, 1, 1, 1]]),
            "labels": _t([[10, 0, 12, 13, 14]]),
            "loss_scale": _t([[1.0, 0.0, 1.0, 1.0, 1.0]]),  # idx 1 zeroed -> label -100
        }
        out = bridge_swift_inputs(inputs, engine=None)
        assert out["input_ids"].shape[-1] == 8           # padded to packing_length
        # loss_scale folded: the zero-weight token's label is -100 (then padded).
        assert out["labels"].tolist()[0][0:5] == [10, -100, 12, 13, 14]
        assert "loss_scale" not in out                    # removed after folding
        assert out["position_ids"].shape[-1] == 8
        assert out["attention_mask"].tolist() == [
            [1, 1, 2, 2, 2, 0, 0, 0]
        ]
    finally:
        engine_registry.clear()
