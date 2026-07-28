"""Bridge swift-packed inputs into turbo's VLM forward contract.

swift's padding-free packing + qwen3.5 mrope lay out a batch differently from
what turbo's model forward consumes. This module rewrites a swift batch dict
(from the trainer's ``_prepare_inputs``) so turbo's full-attn / GDN / mrope
paths run on correct, fixed-length inputs:

1. **integer attention_mask from reset position_ids** — swift encodes packed
   sub-sequence boundaries as ``position_ids`` resets to 0; turbo models key
   their block-diagonal varlen path (``_get_cu_seqlens``) off an integer
   attention_mask (1,2,3... per sub-sequence, 0 = padding). Rebuild it here.

2. **fold swift per-token loss_scale into labels** — turbo's shifted CE does not
   read ``loss_scale``; binary weights (e.g. ``--loss_scale ignore_empty_think``)
   are baked into ``labels`` as ``-100``.

3. **pad the batch to a FIXED length** — turbo's packing path wants a constant
   shape (Triton / flash-attn shape-cache; no unpad roundtrip), while swift
   padding-free packing emits variable lengths. Every sequence-length field is
   padded to ``packing_length`` (rounded up to a multiple of ``sp_size``): the
   padding is folded into the LAST packed sub-sequence (its tail), so
   ``cu_seqlens`` gains no toy segment and ``labels=-100`` keeps it out of loss.

4. **[4, bsz, seq] position_ids** — turbo qwen3.5 expects ``row 0 = text
   positions`` (per-block reset) + ``rows 1-3 = 3D mrope``. swift collator stores
   them split (``position_ids=[3,b,s]`` mrope + ``text_position_ids=[1,b,s]``
   text). Once both are padded to the same length, prepend the text row.

5. **attention_mask = None before the model** — turbo full-attn (flash) / GDN
   consume attention_mask as 0/1 or None; turbos block-diagonals come from
   ``position_ids`` (per-block reset) like the lf adapter, NOT from the integer
   mask. Dropping the integer mask here avoids flash-attn treating integers as
   additive bias → softmax overflow → forward NaN.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

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


def resolve_fixed_packing_length(engine: Any) -> Optional[int]:
    """Fixed sequence length to pad every batch to, for turbo's packing path.

    The target is the configured ``packing_length`` (falling back to
    ``max_length``) — stashed in :mod:`engine_registry` by ``TurboSwiftSft``
    from its FULL args, because swift hands the trainer a stripped args view
    that lacks ``packing_length``/``max_length``. Rounded UP to a multiple of
    ``sp_size`` so turbo's ``ulysses_pad`` also stays a no-op
    (``pad_size == 0``). ``None`` when no length is configured.
    """
    from . import engine_registry
    length = engine_registry.get_packing_length()
    if not length:
        return None
    try:
        sp_size = engine.get_sp_info().get("sp_size", 1) or 1
    except Exception:
        sp_size = 1
    length = int(length)
    if sp_size <= 1:
        return length
    return ((length + sp_size - 1) // sp_size) * sp_size


def pad_batch_to_fixed_length(inputs: dict, target_length: int) -> dict:
    """Pad the batch to a FIXED length, folding padding into the last segment tail.

    Every sequence-length field is padded so they agree at ``target_length``:
    ``input_ids`` (0), ``attention_mask`` (last segment id, no new block),
    ``position_ids`` (each row continues from its OWN last position — rows
    differ, text ends higher than mrope), ``text_position_ids`` (continue its
    last text position), ``labels`` (-100, loss-ignored). Packed-seq params
    (``cu_seq_lens_q/k``, ``max_length_q/k``) are re-derived from the padded
    ``text_position_ids`` so ``cu_seq_lens_q[-1] == target_length``.

    No-op when the batch is already at (or beyond) ``target_length``.
    """
    if not target_length or target_length <= 0:
        return inputs
    input_ids = inputs.get("input_ids")
    if input_ids is None:
        return inputs
    seq = int(input_ids.size(-1))
    pad = target_length - seq
    if pad <= 0:
        return inputs
    device = input_ids.device

    inputs["input_ids"] = F.pad(input_ids, (0, pad), value=0)

    # attention_mask: keep the last segment id for the padding (segment count
    # unchanged) so cu_seqlens adds no toy segment.
    am = inputs.get("attention_mask")
    last_seg_id = None
    if am is not None:
        last_seg_id = int(am.max().item())
        pad_am = torch.full(am.shape[:-1] + (pad,), last_seg_id, dtype=am.dtype, device=am.device)
        inputs["attention_mask"] = torch.cat((am, pad_am), dim=-1)

    # position_ids (mrope rows): each row continues from its OWN last position;
    # rows differ (text ends higher than mrope rows) so must not share one tail.
    pos = inputs.get("position_ids")
    if pos is not None:
        starts = pos[..., -1].unsqueeze(-1) + 1  # (R, B, 1)
        offsets = torch.arange(pad, device=pos.device, dtype=pos.dtype).reshape((1,) * (pos.dim() - 1) + (pad,))
        pad_pos = (starts + offsets).to(pos.dtype)
        inputs["position_ids"] = torch.cat((pos, pad_pos), dim=-1)

    # text_position_ids: continue the text row's last position (drives
    # cu_seq_lens_q; MUST match the padded length).
    tpos = inputs.get("text_position_ids")
    if tpos is not None:
        last_t = int(tpos[..., -1].max().item())
        tpad = torch.arange(last_t + 1, last_t + 1 + pad, device=tpos.device, dtype=tpos.dtype)
        tpad = tpad.reshape((1,) * (tpos.dim() - 1) + (pad,)).expand(tpos.shape[:-1] + (pad,))
        inputs["text_position_ids"] = torch.cat((tpos, tpad), dim=-1)

    labels = inputs.get("labels")
    if labels is not None:
        inputs["labels"] = F.pad(labels, (0, pad), value=_IGNORE_INDEX)

    ls = inputs.get("loss_scale")
    if ls is not None and torch.is_tensor(ls):
        inputs["loss_scale"] = F.pad(ls.float(), (0, pad), value=0.0)

    # Re-derive cu_seq_lens_q/k + max_length_q/k from the padded text row so
    # [-1] == target_length and the boundaries stay consistent with the padded
    # attention_mask cu_seqlens (same reset-0 segmentation source).
    tpos2 = inputs.get("text_position_ids")
    if tpos2 is not None:
        try:
            from swift.utils import get_packed_seq_params as _gpc
            row = tpos2[0] if tpos2.dim() == 3 else tpos2
            inputs.update(_gpc(row.reshape(1, -1) if row.dim() == 1 else row))
        except Exception:
            logger.debug_rank0("pad: recompute cu_seq_lens failed", exc_info=True)

    # Position_ids row 0 = text: prepend the (now-padded) text row so the model
    # sees [4, bsz, seq]. mrope rows stay 1-3.
    tpos3 = inputs.get("text_position_ids")
    pos3 = inputs.get("position_ids")
    if (
        pos3 is not None and tpos3 is not None
        and pos3.dim() == 3 and pos3.shape[0] == 3
        and pos3.shape[-1] == tpos3.shape[-1]
    ):
        inputs["position_ids"] = torch.cat(
            [tpos3.to(pos3.device).reshape(1, *pos3.shape[1:]), pos3], dim=0
        )

    return inputs

def bridge_swift_inputs(inputs: dict, engine: Any) -> dict:
    """Full swift → turbo input bridge for one batch (called from _prepare_inputs).

    Order matters: rebuild integer mask (drives nothing downstream once dropped,
    but kept long enough for any cu_seqlens re-derivation parity) → fold
    loss_scale → pad to fixed length (text + mrope rows + cu_seq_lens + [4,s]
    position_ids) → drop the integer mask so the model never sees it as a bias.
    """
    inputs = bridge_packed_inputs(inputs)
    inputs = bake_loss_scale_into_labels(inputs)
    target = resolve_fixed_packing_length(engine)
    if target is not None:
        inputs = pad_batch_to_fixed_length(inputs, target)
    return inputs
