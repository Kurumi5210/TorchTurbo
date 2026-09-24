"""Normalize Swift batches on CPU before the trainer moves them to the device."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import accumulate
from typing import Any

import torch

from ...engine.utils import assert_packed_inputs
from ...logging import get_logger

logger = get_logger(__name__)

_IGNORE_INDEX = -100


@dataclass
class SwiftInputCollator:
    """Keep Swift collation and normalize packed boundaries before device transfer."""

    collator: Callable[[list], dict[str, Any]]
    template: Any

    def __call__(self, batch: list) -> dict[str, Any]:
        sequence_lengths = None
        if self.template.padding_free:
            # PackingDataset groups samples into lists; the native collator
            # subsequently flattens these groups and mutates the batch.
            samples = batch
            if isinstance(batch[0], list):
                samples = [sample for group in batch for sample in group]
            sequence_lengths = [len(sample["input_ids"]) for sample in samples]
        padding_side = "left"
        if self.template.is_training:
            padding_side = self.template.padding_side
        inputs = self.collator(batch)
        return bridge_swift_inputs(
            inputs,
            sequence_lengths=sequence_lengths,
            padding_side=padding_side,
        )


def bridge_packed_inputs(
    inputs: dict[str, Any],
    *,
    sequence_lengths: list[int],
    padding_side: str = "right",
) -> dict[str, Any]:
    """Express native Swift packing with cu/max and preserve RoPE coordinates.

    Original sample lengths distinguish actual tokens from optional LongLoRA
    padding, whose zero position IDs otherwise resemble one-token samples.
    Physical padding is retained at the tail for the engine to mask.
    """
    input_ids = inputs["input_ids"]
    if input_ids.device.type != "cpu":
        raise ValueError("Packed Swift inputs must be normalized on CPU before device transfer.")
    num_tokens = input_ids.shape[-1]
    num_valid_tokens = sum(sequence_lengths)
    num_padding = num_tokens - num_valid_tokens
    if num_padding < 0 or not sequence_lengths or min(sequence_lengths) <= 0:
        raise ValueError("Packed Swift sample lengths must be positive and fit the collated input.")

    # The engine contract uses a valid prefix followed by optional tail padding.
    if num_padding and padding_side == "left":
        for key in (
            "input_ids",
            "labels",
            "loss_scale",
            "position_ids",
            "text_position_ids",
            "token_type_ids",
            "mm_token_type_ids",
        ):
            if key in inputs:
                inputs[key] = inputs[key].roll(-num_padding, dims=-1)

    expected_cu = torch.tensor([0, *accumulate(sequence_lengths)], dtype=torch.int32)
    for key in ("cu_seq_lens_q", "cu_seq_lens_k"):
        cu = inputs.get(key)
        if cu is None:
            cu = expected_cu
        else:
            if cu.device.type != "cpu":
                raise ValueError("Packed Swift cumulative lengths must be normalized on CPU.")
            if num_padding:
                offset = num_padding if padding_side == "left" else 0
                cu = cu[(cu >= offset) & (cu <= offset + num_valid_tokens)] - offset
            if not torch.equal(cu, expected_cu):
                raise ValueError(f"{key} does not match the original Swift sample boundaries.")
        inputs[key] = cu

    max_length = max(sequence_lengths)
    for key in ("max_length_q", "max_length_k"):
        value = inputs.get(key, max_length)
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                raise ValueError("Packed Swift maximum lengths must be normalized on CPU.")
            value = int(value)
        inputs[key] = value
    inputs.pop("attention_mask", None)
    assert_packed_inputs(inputs)
    return inputs


def bake_loss_scale_into_labels(inputs: dict) -> dict:
    """Fold swift's per-token ``loss_scale`` (binary) into ``labels`` as -100.

    turbo's shifted CE does not read ``loss_scale``; binary weights (e.g.
    ``--loss_scale ignore_empty_think``) zeroing think tokens are baked into
    ``labels`` so turbo skips exactly those tokens. Non-binary weights in
    (0,1) are dropped with a warning. No-op when ``loss_scale`` is absent or
    all-positive.
    """
    loss_scale = inputs.get("loss_scale")
    if loss_scale is None:
        return inputs
    labels = inputs.get("labels")
    if labels is None:
        return inputs

    ls = loss_scale
    if ls.dim() > labels.dim():  # defensive: align ranks
        ls = ls.reshape(labels.shape)
    mask = ls <= 0
    if not bool(mask.any()):
        inputs.pop("loss_scale", None)
        return inputs

    non_binary = bool(mask.any()) and bool(((ls > 0) & (ls < 1)).any())
    if non_binary:
        logger.warning_rank0(
            "loss_scale has non-binary values in (0,1); turbo drops them "
            "(only 0/positive binary weights are folded into labels). "
            "Use --loss_scale ignore_empty_think."
        )

    labels = labels.clone()
    labels[mask] = _IGNORE_INDEX
    inputs["labels"] = labels
    inputs.pop("loss_scale", None)
    return inputs


def bridge_swift_inputs(
    inputs: dict[str, Any],
    *,
    sequence_lengths: list[int] | None = None,
    padding_side: str = "right",
) -> dict[str, Any]:
    """Normalize packing when enabled and fold Swift's per-token loss scales."""
    if sequence_lengths is not None:
        inputs = bridge_packed_inputs(
            inputs,
            sequence_lengths=sequence_lengths,
            padding_side=padding_side,
        )
    return bake_loss_scale_into_labels(inputs)
