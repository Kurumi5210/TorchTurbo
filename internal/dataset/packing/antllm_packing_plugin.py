"""Antllm-aligned packing plugin (internal)."""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

from torchturbo.engine.plugins import PackingPlugin, PluginKind, register_plugin

# ``(sample_index, token_length)`` -- the shape swift's ``calculate_matched_group`` consumes (weight at index 1).
SeqItem = Tuple[int, int]


@register_plugin("antllm-packing", PluginKind.PACKING)
class AntllmPackingPlugin(PackingPlugin):
    """Antllm sequential first-fit packing + ``max_pack_sample_num`` cap."""

    def __init__(self, config: str | None = None) -> None:
        super().__init__(config)
        self.max_pack_sample_num: Optional[int] = int(env) if (env := os.environ.get("MAX_PACK_SAMPLE_NUM", "")).strip() else 300

    def pack_groups(
        self,
        sequences: Sequence[SeqItem],
        packing_length: int,
        *,
        is_finished: bool = True,
    ) -> Tuple[List[List[SeqItem]], List[SeqItem]]:
        """Order-preserving greedy first-fit; returns ``(packs, leftover)``."""
        cap = self.max_pack_sample_num
        packs: List[List[SeqItem]] = []
        cur: List[SeqItem] = []
        cur_len = 0
        cur_count = 0

        def _flush() -> None:
            nonlocal cur, cur_len, cur_count
            if cur:
                packs.append(cur)
            cur, cur_len, cur_count = [], 0, 0

        for idx, length in sequences:
            # antllm ``pack_skip_line``: drop a conversation longer than max_length (flush the open pack first).
            if length > packing_length:
                _flush()
                continue
            if cur and (
                cur_len + length > packing_length
                or (cap is not None and cur_count >= cap)
            ):
                _flush()
            cur.append((idx, length))
            cur_len += length
            cur_count += 1

        if is_finished:
            _flush()
            return packs, []
        return packs, cur

    def knapsack(self, numbers: list[int], capacity: int) -> list[list[int]]:
        """Sequential first-fit (order-preserving, capped); oversize drop is the caller's job."""
        cap = self.max_pack_sample_num
        knapsacks: list[list[int]] = []
        cur: list[int] = []
        cur_len = 0
        for length in numbers:
            if cur and (
                cur_len + length > capacity
                or (cap is not None and len(cur) >= cap)
            ):
                knapsacks.append(cur)
                cur, cur_len = [], 0
            cur.append(length)
            cur_len += length
        if cur:
            knapsacks.append(cur)
        return knapsacks


__all__ = ["AntllmPackingPlugin"]