"""Verify arbitrary packed segment lengths are preserved."""

import pytest

torch = pytest.importorskip("torch")

from torchturbo.models.deepseek_v4.npu import (
    _get_li_query_tnd_data,
    _get_sfa_compressed_kv_tnd_data,
    _get_sfa_query_kv_tnd_data,
)
from torchturbo.models.deepseek_v4.packing_utils import (
    _segment_spans,
    compress_packed_kv,
    packed_compress_kv_ids_and_positions,
)


def test_compression_keeps_unaligned_segment_boundaries():
    mask = torch.tensor([[1] * 127 + [2] * 129], dtype=torch.int32)
    positions = torch.tensor([list(range(127)) + list(range(129))])

    block_ids, block_positions = packed_compress_kv_ids_and_positions(
        mask, 128, positions
    )

    assert block_ids.tolist() == [[2]]
    assert block_positions.tolist() == [[0]]


def test_compression_metadata_handles_padding_and_multiple_rows():
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2],
            [0, 0, 3, 3, 3, 3, 3, 3, 0, 4, 4, 4],
        ],
        dtype=torch.long,
    )
    positions = torch.arange(mask.size(1)).unsqueeze(0).expand_as(mask)

    block_ids, block_positions = packed_compress_kv_ids_and_positions(
        mask, 4, positions
    )

    assert block_ids.tolist() == [[1, 2], [3, 0]]
    assert block_positions.tolist() == [[0, 7], [2, 0]]


def _reference_compress_packed_kv(kv, score, ape, mask, ratio, head_dim, overlap):
    rows = []
    for batch_idx, row_mask in enumerate(mask):
        segments = []
        starts, ends, _ = _segment_spans(row_mask)
        for start, end in zip(starts.tolist(), ends.tolist()):
            cutoff = start + ((end - start) // ratio) * ratio
            if cutoff == start:
                continue
            segment_kv = kv[batch_idx : batch_idx + 1, start:cutoff].unflatten(
                1, (-1, ratio)
            )
            segment_score = (
                score[batch_idx : batch_idx + 1, start:cutoff].unflatten(
                    1, (-1, ratio)
                )
                + ape
            )
            if overlap:
                num_blocks = segment_kv.size(1)
                overlap_kv = segment_kv.new_zeros(
                    (1, num_blocks, 2 * ratio, head_dim)
                )
                overlap_score = segment_score.new_full(
                    (1, num_blocks, 2 * ratio, head_dim), float("-inf")
                )
                overlap_kv[:, :, ratio:] = segment_kv[:, :, :, head_dim:]
                overlap_kv[:, 1:, :ratio] = segment_kv[:, :-1, :, :head_dim]
                overlap_score[:, :, ratio:] = segment_score[:, :, :, head_dim:]
                overlap_score[:, 1:, :ratio] = segment_score[
                    :, :-1, :, :head_dim
                ]
                segment_kv, segment_score = overlap_kv, overlap_score
            segments.append(
                (segment_kv * segment_score.softmax(dim=2)).sum(dim=2).squeeze(0)
            )
        rows.append(
            torch.cat(segments, dim=0)
            if segments
            else kv.new_empty((0, head_dim))
        )

    max_blocks = max(row.size(0) for row in rows)
    output = kv.new_zeros((kv.size(0), max_blocks, head_dim))
    for batch_idx, row in enumerate(rows):
        output[batch_idx, : row.size(0)] = row
    return output


@pytest.mark.parametrize("ratio,overlap", [(4, True), (128, False)])
def test_vectorized_compressor_matches_segment_loop(ratio, overlap):
    if ratio == 4:
        rows = [
            [1] * 7 + [0] * 2 + [2] * 9 + [0] * 2,
            [0] + [3] * 12 + [0] * 2 + [4] * 5,
        ]
    else:
        rows = [
            [1] * 127 + [2] * 129 + [0] * 8,
            [0] * 3 + [3] * 260 + [0],
        ]
    mask = torch.tensor(rows, dtype=torch.long)
    positions = torch.arange(mask.size(1)).unsqueeze(0).expand_as(mask)
    head_dim = 4
    projected_dim = head_dim * (2 if overlap else 1)
    generator = torch.Generator().manual_seed(ratio)
    kv = torch.randn(
        mask.size(0),
        mask.size(1),
        projected_dim,
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
    )
    score = torch.randn(
        kv.shape, generator=generator, dtype=torch.float64, requires_grad=True
    )
    ape = torch.randn(
        ratio,
        projected_dim,
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
    )

    actual, _ = compress_packed_kv(
        kv, score, ape, mask, positions, ratio, head_dim, overlap
    )
    expected = _reference_compress_packed_kv(
        kv, score, ape, mask, ratio, head_dim, overlap
    )
    torch.testing.assert_close(actual, expected)

    grad_output = torch.randn(actual.shape, generator=generator, dtype=actual.dtype)
    expected_grads = torch.autograd.grad(
        expected, (kv, score, ape), grad_output, retain_graph=True
    )
    actual_grads = torch.autograd.grad(actual, (kv, score, ape), grad_output)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_li_query_metadata_ignores_zero_id_spans():
    mask = torch.tensor([[0, 1, 1, 0, 2, 2, 2]], dtype=torch.long)

    indices, cu_seqlens = _get_li_query_tnd_data(mask)

    assert indices.tolist() == [1, 2, 4, 5, 6]
    assert cu_seqlens.tolist() == [0, 2, 5]


@pytest.mark.parametrize(
    ("sp_rank", "expected_q", "expected_kv"),
    [
        (0, [0, 127, 128], [0, 127, 128]),
        (1, [0, 128], [0, 129]),
    ],
)
def test_sfa_tnd_boundaries_cross_sp_shards(sp_rank, expected_q, expected_kv):
    mask = torch.tensor([[1] * 127 + [2] * 129], dtype=torch.int32)
    query = torch.empty(1, 128, 1, 8)
    kv = torch.empty(1, 256, 8)

    _, _, cu_seqlens_q, cu_seqlens_kv, _ = _get_sfa_query_kv_tnd_data(
        query, kv, mask, sp_rank
    )

    assert cu_seqlens_q.tolist() == expected_q
    assert cu_seqlens_kv.tolist() == expected_kv


@pytest.mark.parametrize(
    ("sp_rank", "expected_cu_seqlens", "expected_residuals"),
    [
        (0, [0, 0, 0], [127, 1]),
        (1, [0, 1], [1]),
    ],
)
def test_sfa_preserves_partial_compression_blocks(
    sp_rank, expected_cu_seqlens, expected_residuals
):
    mask = torch.tensor([[1] * 127 + [2] * 129], dtype=torch.int32)
    local_mask = mask[:, sp_rank * 128 : (sp_rank + 1) * 128]
    kv_compress = torch.empty(1, 1, 8)
    topk = torch.empty(1, 128, 0, dtype=torch.long)

    _, cu_seqlens, _, residuals = _get_sfa_compressed_kv_tnd_data(
        kv_compress,
        topk,
        mask,
        local_mask,
        sp_rank * 128,
        256,
        128,
    )

    assert cu_seqlens.tolist() == expected_cu_seqlens
    assert residuals.tolist() == expected_residuals
