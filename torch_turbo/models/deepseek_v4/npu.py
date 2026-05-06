import torch

def npu_sparse_attn_shared_kv(
    query,
    ori_kv,
    cmp_kv,
    cmp_sparse_indices,
    sinks,
    softmax_scale,
    cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
):
    cu_seq_lens_q = cu_seq_lens_ori_kv = cu_seq_lens_cmp_kv = None  # not support TND
    ori_sparse_indices = None  # ori kv use band mode
    batch_size, max_seq_len_q, num_heads_q, head_dim = query.size()
    num_heads_kv = 1
    max_seq_len_kv = ori_kv.size(1)
    topk = 0 if cmp_ratio != 4 else cmp_sparse_indices.size(-1)
    layout_q = layout_kv = "BSND"
    query = query.contiguous()  # [B, S, N, D] --> [B, S, N, D]
    ori_kv = ori_kv.unsqueeze(2).contiguous()  # [B, S, D] --> [B, S, 1, D]
    cmp_kv = (
        cmp_kv if cmp_kv is None else cmp_kv.unsqueeze(2).contiguous()
    )  # [B, S, D] --> [B, S, 1, D]
    if cmp_ratio != 4:
        cmp_sparse_indices = None
    else:
        cmp_sparse_indices = cmp_sparse_indices.unsqueeze(2).contiguous()
    from mindspeed.ops.npu_sparse_attn_shared_kv import SparseAttnSharedKV

    output = SparseAttnSharedKV.apply(
        query,
        ori_kv,
        cmp_kv,
        cu_seq_lens_q,
        cu_seq_lens_ori_kv,
        cu_seq_lens_cmp_kv,
        ori_sparse_indices,
        cmp_sparse_indices,
        sinks,
        softmax_scale,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        num_heads_q,
        num_heads_kv,
        head_dim,
        batch_size,
        max_seq_len_q,
        max_seq_len_kv,
        topk,
        layout_q,
        layout_kv,
    )
    return output.contiguous()


def npu_sfa(
    self, query_states, kv_states, attn_sink, kv_compress, compress_topk_idxs
):
    if compress_topk_idxs is not None:
        if compress_topk_idxs.dtype != torch.int32:
            compress_topk_idxs = compress_topk_idxs.to(torch.int32)

    output = npu_sparse_attn_shared_kv(
        query=query_states,
        ori_kv=kv_states,
        cmp_kv=kv_compress,
        cmp_sparse_indices=compress_topk_idxs,
        sinks=attn_sink.float(),
        softmax_scale=self.softmax_scale,
        cmp_ratio=self.compress_ratio,
    )
    return output


def npu_li(
    self,
    q_indexer: torch.Tensor,
    k_indexer: torch.Tensor,
    weights: torch.Tensor,
    seqlen: int,
    offset: int,
):
    q_indexer = q_indexer.to(torch.bfloat16)
    k_indexer = k_indexer.to(torch.bfloat16).unsqueeze(2)
    weights = weights.to(torch.bfloat16)
    import mindspeed.ops.npu_lightning_indexer as mindspeed_li

    compress_topk_idxs, index_score = mindspeed_li.npu_lightning_indexer(
        q_indexer,
        k_indexer,
        weights,
        sparse_count=self.index_topk,
        sparse_mode=3,
        cmp_ratio=self.ratio,
    )

    compress_topk_idxs = compress_topk_idxs.squeeze(2)
    index_score = index_score.squeeze(2)
    if offset != 0:
        compress_topk_idxs = torch.where(
            compress_topk_idxs == -1, compress_topk_idxs, compress_topk_idxs + offset
        )

    return compress_topk_idxs, index_score
