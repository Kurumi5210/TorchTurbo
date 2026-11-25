import torch
import triton
import triton.language as tl


@triton.jit
def _pack_kernel(
    ids, weights, dense, mask, ordered, rows, experts: tl.constexpr, block: tl.constexpr
):
    row = tl.program_id(0) * block + tl.arange(0, block)
    offset = row[:, None] * 8 + tl.arange(0, 8)[None, :]
    choices = tl.load(ids + offset, row[:, None] < rows, 0).to(tl.int32)
    values = tl.load(weights + offset, row[:, None] < rows, 0)
    ranks = tl.sum((choices[:, :, None] > choices[:, None, :]).to(tl.int32), axis=2)
    destinations = row[:, None] * experts + choices
    tl.store(dense + destinations, values, row[:, None] < rows)
    tl.store(mask + destinations, 1, row[:, None] < rows)
    tl.store(
        ordered + row[:, None] * 8 + ranks, choices.to(tl.int64), row[:, None] < rows
    )


def pack(
    weights: torch.Tensor, ids: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = ids.shape[0]
    dense = weights.new_zeros((rows, num_experts))
    mask = torch.zeros_like(dense, dtype=torch.bool)
    ordered = torch.empty((rows, 8), dtype=ids.dtype, device=ids.device)
    if rows:
        _pack_kernel[(triton.cdiv(rows, 8),)](
            ids.contiguous(),
            weights.contiguous(),
            dense,
            mask,
            ordered,
            rows,
            num_experts,
            8,
        )
    return dense, mask, ordered
