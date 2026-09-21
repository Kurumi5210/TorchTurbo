"""Reusable packed-sequence metadata prepared outside operator providers."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import torch


class ValidatedOffsets(list[int]):
    """Host offsets validated while building :class:`PackedParams`."""


class ValidatedChunkLists(dict[str, list[int]]):
    """Host chunk indices derived from :class:`ValidatedOffsets`."""


_GDN_CHUNK_SIZES = (16, 32, 64, 128, 608 * 2)


@dataclass(frozen=True)
class PackedParams:
    """Device and host metadata shared by packed causal-conv and GDN calls.

    Construction performs the device-to-host synchronization once. The caller
    owns caching and lifetime; operator providers only consume these values.
    """

    cu_seqlens: torch.Tensor
    cu_seqlens_list: ValidatedOffsets
    chunk_indices: dict[str, torch.Tensor]
    chunk_indices_list: ValidatedChunkLists


def build_packed_params(
    cu_seqlens: torch.Tensor,
    chunk_sizes: tuple[int, ...] = _GDN_CHUNK_SIZES,
) -> PackedParams:
    """Build all metadata shared by causal-conv and GDN for a packed batch."""
    if cu_seqlens.ndim != 1:
        raise ValueError(
            f"cu_seqlens must be rank-1, got shape={tuple(cu_seqlens.shape)}"
        )
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"cu_seqlens must be int32/int64, got dtype={cu_seqlens.dtype}"
        )

    offsets = ValidatedOffsets(
        int(item) for item in cu_seqlens.detach().cpu().tolist()
    )
    chunk_tensors: dict[str, torch.Tensor] = {}
    chunk_lists = ValidatedChunkLists()
    for chunk_size in dict.fromkeys(chunk_sizes):
        pairs: list[list[int]] = []
        flattened: list[int] = []
        for sequence_index, (left, right) in enumerate(pairwise(offsets)):
            count = (right - left + chunk_size - 1) // chunk_size
            for chunk_index in range(count):
                pairs.append([sequence_index, chunk_index])
                flattened.extend((sequence_index, chunk_index))
        key = str(chunk_size)
        chunk_tensors[key] = torch.tensor(
            pairs,
            dtype=torch.int64,
            device=cu_seqlens.device,
        ).reshape(-1, 2)
        chunk_lists[key] = flattened

    return PackedParams(
        cu_seqlens=cu_seqlens.to(dtype=torch.int64).contiguous(),
        cu_seqlens_list=offsets,
        chunk_indices=chunk_tensors,
        chunk_indices_list=chunk_lists,
    )


__all__ = [
    "PackedParams",
    "ValidatedChunkLists",
    "ValidatedOffsets",
    "build_packed_params",
]
