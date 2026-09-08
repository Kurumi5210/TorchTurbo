"""Bridge swift-packed inputs into turbo's VLM forward contract.

swift's padding-free packing + qwen3.5 mrope lay out a batch differently from
what turbo's model forward consumes. This module rewrites a swift batch dict
(from the trainer's ``_prepare_inputs``) so turbo's full-attn / GDN / mrope
paths run on correct inputs:

1. **integer attention_mask from reset position_ids** — swift encodes packed
   sub-sequence boundaries as ``position_ids`` resets to 0; turbo models key
   their block-diagonal varlen path (``_get_cu_seqlens``) off an integer
   attention_mask (1,2,3... per sub-sequence, 0 = padding). Rebuild it here.

2. **fold swift per-token loss_scale into labels** — turbo's shifted CE does not
   read ``loss_scale``; binary weights (e.g. ``--loss_scale ignore_empty_think``)
   are baked into ``labels`` as ``-100``.

3. **model-specific input layout** — Swift's split position rows are left intact
   here. Model-specific layout normalization runs in the swift template's
   pre_forward_hook, not in this bridge.
   before its forward pass. Batches keep Swift's native variable length; backend
   ``prepare_inputs`` owns any SP alignment padding.
"""

from __future__ import annotations

from typing import Optional

import torch

from ...logging import get_logger

logger = get_logger(__name__)

_IGNORE_INDEX = -100


def position_ids_reset_to_packed_mask(
    position_ids: Optional[torch.Tensor],
    fallback_attention_mask: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Rebuild an integer packed ``attention_mask`` from reset ``position_ids``.

    ``position_ids`` is ``(B, N)`` or mrope ``(3, B, N)`` / ``(4, B, N)`` — the
    text row (row 0) carries the segment structure. A new packed sub-sequence
    starts at a position RESET TO 0 (swift's padding-free packing restarts
    position_ids at 0 on each member). NB: do NOT use "non-strictly-increasing"
    (``<=``): mrope assigns long runs of identical positions to visual tokens
    (e.g. ``[0,1,2,3,4,4,4,...]``), and ``<=`` would count every such plateau
    as a segment — fabricating 1e4+ phantom segments.

    Returns ``None`` for a single-segment batch so the caller leaves the mask as
    swift supplied it.
    """
    if position_ids is None:
        return None
    # mrope/4-row: text row (row 0) carries the segment structure.
    text = position_ids[0] if position_ids.dim() == 3 else position_ids
    if text.dim() != 2:
        return None
    bsz, n = text.shape
    if n <= 1:
        return None

    start = torch.zeros(bsz, n, dtype=torch.bool, device=text.device)
    start[:, 0] = True
    start[:, 1:] = text[:, 1:] == 0
    packed = start.cumsum(dim=-1).to(torch.int64)  # 1,1,1,2,2,...

    if fallback_attention_mask is not None:
        packed = packed * fallback_attention_mask.to(torch.int64)

    if int(packed.max().item()) <= 1:
        return None
    return packed


def bridge_packed_inputs(inputs: dict) -> dict:
    """Rebuild the integer attention_mask from reset position_ids, if packed."""
    pos = inputs.get("position_ids")
    if pos is None:
        return inputs
    packed_mask = position_ids_reset_to_packed_mask(pos, inputs.get("attention_mask"))
    if packed_mask is None:
        return inputs
    inputs["attention_mask"] = packed_mask
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


def bridge_swift_inputs(inputs: dict) -> dict:
    """Rebuild the packed attention mask, then fold loss_scale (per batch)."""
    inputs = bridge_packed_inputs(inputs)
    return bake_loss_scale_into_labels(inputs)
