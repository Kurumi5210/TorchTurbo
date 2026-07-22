import functools
from typing import Optional, Tuple

from contextlib import nullcontext

import torch
from torch import nn
import torch.nn.functional as F

import torch_npu
from torchturbo.utils.functools import cache
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .modeling_glm_moe_dsa import apply_rotary_pos_emb, eager_attention_forward
from torchturbo.utils.flash_attn_utils import (
    packing_mask_to_causal_mask,
    get_seqlens_in_batch,
)
from torchturbo.models.modules.attention.ulysses import (
    all_gather_sp,
)


# ============================================================================
# DSA Indexer Loss AutoScaler
# ============================================================================
class DSAIndexerLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that triggers the backward pass and scales the grad for DSA indexer loss.

    This is used to apply a gradient scale to the DSA indexer loss during training.
    The scale is shared via the class attribute `main_loss_backward_scale` which is
    typically set to match the main training loss scale.
    """

    main_loss_backward_scale: torch.Tensor = None

    @staticmethod
    def forward(ctx, output: torch.Tensor, loss: torch.Tensor):
        ctx.save_for_backward(loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (loss,) = ctx.saved_tensors
        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = torch.tensor(
                1.0, device=loss.device
            )
        dsa_indexer_loss_backward_scale = (
            DSAIndexerLossAutoScaler.main_loss_backward_scale
        )
        scaled_dsa_indexer_loss_grad = (
            torch.ones_like(loss) * dsa_indexer_loss_backward_scale
        )
        return grad_output, scaled_dsa_indexer_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = scale
        else:
            DSAIndexerLossAutoScaler.main_loss_backward_scale.copy_(scale)


# ============================================================================
# Fused Lightning Indexer KL Loss
# ============================================================================
class LILossTrain(torch.autograd.Function):
    """
    A custom autograd function that computes kl loss in sparse lightning indexer.
    This interface implements the backward functionality of npu_lightning_indexer and integrates the loss computation.
    The npu_lightning_indexer selects the top-k pairs between queries and keys in attention that exhibit the strongest
    intrinsic correlations, storing them in sparse_indices. This reduces the computational cost of attention in
    long-sequence scenarios and improves training performance.
    """

    @staticmethod
    def forward(
        ctx,
        query,
        key,
        query_index,
        key_index,
        weights,
        sparse_indices,
        softmax_max,
        softmax_sum,
        scale_value=1,
        query_rope=None,
        key_rope=None,
        actual_seq_qlen=None,
        actual_seq_klen=None,
        layout="TND",
        sparse_mode=3,
    ):
        d_query_index, d_key_index, d_weights, loss = (
            torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
                query,
                key,
                query_index,
                key_index,
                weights,
                sparse_indices,
                softmax_max,
                softmax_sum,
                scale_value,  # positional argument
                query_rope=query_rope,
                key_rope=key_rope,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_klen=actual_seq_klen,
                layout=layout,
                sparse_mode=sparse_mode,
            )
        )
        ctx.save_for_backward(d_query_index, d_key_index, d_weights)
        return loss[0]

    @staticmethod
    def backward(ctx, *grad_output) -> Tuple:
        d_query_index, d_key_index, d_weights = ctx.saved_tensors
        grad_scale = grad_output[0]
        if torch.ne(grad_scale, torch.tensor(1.0, device=grad_scale.device)):
            d_query_index = d_query_index * grad_scale
            d_key_index = d_key_index * grad_scale
            d_weights = d_weights * grad_scale
        res_list = [None] * 12
        return None, None, d_query_index, d_key_index, d_weights, *res_list


def fused_sparse_lightning_indexer_kl_loss(
    query,
    key,
    query_index,
    key_index,
    weights,
    sparse_indices,
    softmax_max,
    softmax_sum,
    scale_value=1,
    *,
    query_rope=None,
    key_rope=None,
    actual_seq_qlen=None,
    actual_seq_klen=None,
    layout="TND",
    sparse_mode=3,
):
    """NPU Sparse Lightning Indexer KL Divergence Loss Function.
    Args:
        query: [T, H, D] in TND layout - Attention query (bfloat16/float16)
        key: [T, 1, D] in TND layout - Attention key (KV_N=1 for MLA absorb) (bfloat16/float16)
        query_index: [T, N_index, D] query for lightning indexer in TND layout (bfloat16/float16)
        key_index: [T, 1, D] key for lightning indexer in TND layout (bfloat16/float16)
        weights: [T, N_index] weights for lightning indexer in TND layout (bfloat16/float16)
        sparse_indices: [T, 1, topk] selected sparse indices in TND layout (int32)
        softmax_max: [N2, T, G] from sparse flash attention in TND layout (N2=1, G=H) (float32)
        softmax_sum: [N2, T, G] from sparse flash attention in TND layout (N2=1, G=H) (float32)
        scale_value: attention scale (float)
        query_rope: [T, H, D_rope] query RoPE part in TND layout (bfloat16/float16)
        key_rope: [T, 1, D_rope] key RoPE part in TND layout (bfloat16/float16)
        actual_seq_qlen: list[int64] - cumulative sequence lengths for query (for packing)
        actual_seq_klen: list[int64] - cumulative sequence lengths for key (for packing)
        layout: input layout format (TND or BSND)
        sparse_mode: sparse computation mode (3 = causal/rightDownCausal)
    Returns:
        loss: KL divergence loss scaled by total_tokens
    NPU API Constraints (npu_sparse_lightning_indexer_grad_kl_loss):
        - query shape N (num_heads) must be 64 or 128
        - topK must be 2048 (integer multiple of 1024, < 8192)
        - sparse_indices dtype: int32
        - softmax_max/softmax_sum dtype: float32
        - actual_seq_qlen/klen dtype: int64
    """
    # Get dimensions for scaling
    if layout == "TND":
        total_tokens = query.shape[0]
    else:
        total_tokens = query.shape[0] * query.shape[1]
    loss = LILossTrain.apply(
        query,
        key,
        query_index,
        key_index,
        weights,
        sparse_indices,
        softmax_max,
        softmax_sum,
        scale_value,
        query_rope,
        key_rope,
        actual_seq_qlen,
        actual_seq_klen,
        layout,
        sparse_mode,
    )
    return loss / total_tokens


def fix_invalid_topk_indices(
    topk_indices: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Fix invalid indices in topk_indices for packing mode in naive torch indexer implementation.
    i.e. indexer_forward_torch

    For positions where topk selected invalid indices (cross-sequence or causal violation),
    replace them with the first token position of the corresponding sub-sequence.

    Args:
        topk_indices: [batch_size, seq_len, topk] - Selected top-k indices
        attention_mask: [batch_size, seq_len] - Packing attention mask with values 1, 2, 3...

    Returns:
        topk_indices: [seq_len, 1, topk] - Fixed top-k indices with invalid positions replaced (TND layout)
    """
    batch_size, seq_len, topk = topk_indices.shape
    dtype = topk_indices.dtype

    # Get sequence start positions
    seqlens = get_seqlens_in_batch(attention_mask)  # (num_sequences,)
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0), (1, 0))  # (num_sequences + 1,)

    # Create a mapping from each position to its sequence start position
    # seq_start_positions[i] = start position of the sequence that position i belongs to
    seq_start_positions = torch.zeros(
        seq_len, dtype=torch.long, device=topk_indices.device
    )
    for seq_idx in range(len(seqlens)):
        start = cu_seqlens[seq_idx].item()
        end = cu_seqlens[seq_idx + 1].item()
        seq_start_positions[start:end] = start

    # Check validity of each selected index
    # For each query position, the selected key index must be:
    # 1. In the same sub-sequence (same sequence_id)
    # 2. Not after the query position (causal)

    # Get sequence_id for each position
    sequence_ids = attention_mask[0].long()  # (total_len,)

    # Expand for comparison: topk_indices shape [batch, seq_len, topk]
    query_seq_ids = (
        sequence_ids.unsqueeze(0).unsqueeze(-1).expand_as(topk_indices)
    )  # [1, seq_len, topk]

    # Gather sequence_ids for the selected key positions
    selected_seq_ids = torch.gather(
        sequence_ids.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, topk),
        dim=1,
        index=topk_indices,
    )  # [batch, seq_len, topk]

    # Check if selected indices are valid:
    # 1. Same sequence as query
    is_same_seq = selected_seq_ids == query_seq_ids

    # 2. Causal: selected_index <= query_position (within same sequence)
    query_positions = (
        torch.arange(seq_len, device=topk_indices.device).unsqueeze(0).unsqueeze(-1)
    )
    is_causal = topk_indices <= query_positions
    is_valid = topk_indices >= 0

    # Valid if both conditions are met
    valid_mask = is_same_seq & is_causal & is_valid

    # Replace invalid indices with the start position of the corresponding sub-sequence
    seq_start_positions_expanded = (
        seq_start_positions.unsqueeze(0).unsqueeze(-1).expand_as(topk_indices)
    )
    topk_indices = torch.where(valid_mask, topk_indices, seq_start_positions_expanded)

    return topk_indices.to(dtype).view(seq_len, 1, -1)


def fix_npu_topk_indices(
    topk_indices: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Fix and convert NPU lightning indexer output to global indices.

    NPU lightning indexer returns:
    1. -1 for invalid positions (cross-sequence or causal violation)
    2. Sub-sequence relative positions (not global positions)

    This function:
    1. Replaces -1 with the first token position of each sub-sequence (local position 0)
    2. Converts sub-sequence relative positions to global positions

    Args:
        topk_indices: [T, 1, topk] - NPU output with relative positions and -1 for invalid
        attention_mask: [1, total_tokens] - Packing attention mask with values 1, 2, 3...

    Returns:
        topk_indices: [T, 1, topk] - Global indices with invalid positions fixed
    """
    total_tokens, _, topk = topk_indices.shape
    device = topk_indices.device
    dtype = topk_indices.dtype

    # Get sequence lengths and cumulative lengths
    seqlens = get_seqlens_in_batch(attention_mask)  # (num_sequences,)
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0), (1, 0))  # (num_sequences + 1,)
    num_seqs = len(seqlens)

    # Create a mapping from each global position to its sub-sequence start position
    # seq_start_positions[i] = start position of the sequence that position i belongs to
    seq_start_positions = torch.zeros(total_tokens, dtype=torch.long, device=device)
    for seq_idx in range(num_seqs):
        start = cu_seqlens[seq_idx].item()
        end = cu_seqlens[seq_idx + 1].item()
        seq_start_positions[start:end] = start

    # Expand seq_start_positions to match topk_indices shape [T, 1, topk]
    seq_start_expanded = (
        seq_start_positions.unsqueeze(-1).unsqueeze(-1).expand_as(topk_indices)
    )

    # Create mask for valid indices (not -1)
    valid_mask = topk_indices >= 0

    # Convert relative positions to global positions
    # global_position = seq_start + relative_position
    global_indices = torch.where(
        valid_mask, seq_start_expanded + topk_indices, topk_indices
    )

    # For invalid positions (-1), replace with the corresponding sub-sequence start position
    # This ensures each query at least attends to the first token of its sub-sequence
    fixed_indices = torch.where(valid_mask, global_indices, seq_start_expanded)

    return fixed_indices.to(dtype)


def create_index_mask(
    topk_indices: torch.Tensor,
    dtype=torch.bfloat16,
) -> torch.Tensor:
    """
    Create sparse attention index mask from topk indices for packing mode.

    The output mask can be added to attention scores to implement sparse attention:
    - Selected positions (from topk_indices): 0.0
    - Other positions: -inf

    Args:
        topk_indices: [T, 1, topk] - Selected top-k indices (TND layout, N=1 for MLA)

    Returns:
        index_mask: [1, 1, T, T] - Sparse attention mask
    """
    seq_length, _, _ = topk_indices.shape
    total_len = seq_length  # For packing mode, total_len == seq_length
    device = topk_indices.device

    # Create full mask with -inf
    index_mask = torch.full(
        (1, seq_length, total_len),
        float("-inf"),
        device=device,
        dtype=dtype,
    )
    # Set selected positions to 0.0
    # topk_indices is [T, 1, topk], need to squeeze the N dimension for scatter
    index_mask.scatter_(-1, topk_indices.squeeze(1).unsqueeze(0), 0.0)
    # Add head dimension: [batch, seq, kv] -> [batch, 1, seq, kv]
    index_mask = index_mask.unsqueeze(1)

    return index_mask


def indexer_forward_torch(
    self,
    hidden_states: torch.Tensor,
    q_resid: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    use_cache: bool = False,
) -> torch.Tensor:
    """
    Original eager implementation for reference.
    """
    batch_size, seq_len, _ = hidden_states.shape
    cos, sin = position_embeddings

    q = self.wq_b(q_resid)
    q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
    q_pe, q_nope = torch.split(
        q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
    )
    q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=2)
    q = torch.cat([q_pe, q_nope], dim=-1)

    k = self.k_norm(self.wk(hidden_states))
    k_pe, k_nope = torch.split(
        k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
    )
    k_pe = apply_rotary_pos_emb(k_pe.unsqueeze(2), cos, sin, unsqueeze_dim=2).squeeze(2)
    k = torch.cat([k_pe, k_nope], dim=-1)

    if seq_len > 1:
        self._cached_keys = None

    if use_cache:
        if self._cached_keys is not None:
            k_cached = torch.cat([self._cached_keys, k], dim=1)
        else:
            k_cached = k
        self._cached_keys = k_cached
    else:
        k_cached = k

    weights = self.weights_proj(hidden_states).float() * (self.n_heads**-0.5)

    scores = (
        torch.einsum("bshd,btd->bsht", q.float(), k_cached.float()) * self.softmax_scale
    )
    scores = F.relu(scores)
    index_scores = torch.einsum("bsht,bsh->bst", scores, weights)

    if attention_mask is not None:
        # Convert packing mask to causal mask and apply to index_scores
        # packing_mask_to_causal_mask returns (total_len, total_len) bool mask
        causal_mask = packing_mask_to_causal_mask(attention_mask)
        # Convert to additive mask: True -> 0, False -> -inf
        additive_mask = torch.where(causal_mask, 0.0, float("-inf"))
        index_scores = index_scores + additive_mask.to(index_scores.dtype)

    total_len = index_scores.shape[-1]
    topk = min(self.index_topk, total_len)
    topk_indices = index_scores.topk(topk, dim=-1).indices

    return topk_indices.to(torch.int32)


def indexer_forward_npu(
    self,
    hidden_states: torch.Tensor,
    q_resid: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    use_cache: bool = False,
) -> torch.Tensor:
    """
    NPU fused implementation using torch_npu.npu_lightning_indexer with TND layout.

    This is a wrapper that calls forward_with_index and forward_with_scores.
    For more granular control, use those functions directly.

    Args:
        self: GlmMoeDsaIndexer instance
        hidden_states: [batch_size, seq_len, hidden_size] - Input hidden states (BSND format)
        q_resid: [batch_size, seq_len, q_lora_rank] - Low-rank query residual
        position_embeddings: Tuple of (cos, sin) for rotary embeddings
        attention_mask: [1, total_tokens] - Packing attention mask with values 1, 2, 3...
        use_cache: Whether to use KV caching for generation

    Returns:
        topk_indices: [T, 1, topk] - Top-k indices for sparse attention (TND layout)
    """
    # Step 1: Get query, key, weights from forward_with_index
    query_index, key_index, weights = indexer_forward_with_index(
        self, hidden_states, q_resid, position_embeddings, use_cache
    )

    # Step 2: Get topk indices from forward_with_scores
    topk_indices, _ = indexer_forward_with_scores(
        self, query_index, key_index, weights, attention_mask
    )

    return topk_indices


def indexer_forward_with_index(
    self,
    hidden_states: torch.Tensor,
    q_resid: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    use_cache: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    First phase of NPU lightning indexer: compute query, key, and weights.

    This computes the query, key projections and weights for the lightning indexer.
    The outputs can be reused for both topk selection and KL loss computation.

    With SP: each rank computes on local sequence chunk in parallel (no communication).

    Args:
        self: GlmMoeDsaIndexer instance
        hidden_states: [batch_size, seq_len/sp_size, hidden_size] - Input hidden states (BSND format, local with SP)
        q_resid: [batch_size, seq_len/sp_size, q_lora_rank] - Low-rank query residual (local with SP)
        position_embeddings: Tuple of (cos, sin) for rotary embeddings (local with SP)
        use_cache: Must be False (training only, no KV cache)

    Returns:
        query_index: [B, S/sp_size, H, D] - Query for lightning indexer in BSND layout (local with SP)
        key_index: [B, S/sp_size, 1, D] - Key for lightning indexer in BSND layout (local with SP)
        weights: [B, S/sp_size, H] - Weights for lightning indexer in BSND layout (local with SP)
    """
    # Training only - no KV cache support
    assert (
        not use_cache
    ), "indexer_forward_with_index does not support use_cache (training only)"

    batch_size, seq_len, _ = hidden_states.shape
    cos, sin = position_embeddings

    # Project low-rank query to full multi-head query
    q = self.wq_b(q_resid)
    q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)  # [B, S, N, D]

    # Apply rotary positional embedding to the RoPE part of the query
    q_pe, q_nope = torch.split(
        q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
    )
    q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=2)
    q = torch.cat([q_pe, q_nope], dim=-1)

    # Project and normalize keys
    k = self.k_norm(self.wk(hidden_states))
    k_pe, k_nope = torch.split(
        k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
    )
    k_pe = apply_rotary_pos_emb(k_pe.unsqueeze(2), cos, sin, unsqueeze_dim=2).squeeze(2)
    k = torch.cat([k_pe, k_nope], dim=-1)

    # Compute weights with scaling
    weights = self.weights_proj(hidden_states)
    weights = weights * (self.n_heads**-0.5)
    weights = weights * self.softmax_scale

    # Convert to bfloat16 for NPU operator
    q_bf16 = q.to(torch.bfloat16)
    k_bf16 = k.to(torch.bfloat16)
    weights_bf16 = weights.to(torch.bfloat16)

    # Ensure key has head dimension: [B, S, head_dim] -> [B, S, 1, head_dim]
    if k_bf16.dim() == 3:
        k_bf16 = k_bf16.unsqueeze(2)

    return q_bf16, k_bf16, weights_bf16


@cache
def get_sp_local_cu_seqlens(
    attention_mask, bs, local_seq_size, device, sp_group, sp_size
):
    """Get sequence parallel local cumulative sequence lengths for TND layout.

    Computes the cumulative sequence lengths for query and key sequences
    in the local SP rank, and the corresponding global positions in the KV sequence.

    Args:
        attention_mask: [1, total_tokens] - Packing attention mask with values 1, 2, 3...

    Returns:
        Tuple of (mode, q_cu_seqlens, kv_cu_seqlens, kv_start, kv_end, pad) where:
            mode: "TND" or "BSND" - the layout for NPU ops
            q_cu_seqlens: [num_sequences] - cumulative sequence lengths for query (local, int32)
            kv_cu_seqlens: [num_sequences] - cumulative sequence lengths for key (global, int32)
            kv_start: int - global start position of KV for this rank's sequences
            kv_end: int - global end position of KV for this rank's sequences
            pad: int - padding in the local sequence
    """
    sp_rank = dist.get_rank(group=sp_group)
    if attention_mask is None or (bs != 1 and torch.max(attention_mask) <= 1):
        kv_end = local_seq_size * (sp_rank + 1)
        q_seqlens = torch.cumsum(
            torch.full((bs,), local_seq_size, dtype=torch.int32, device=device), dim=0
        ).to(torch.int32)
        kv_cu_seqlens = torch.cumsum(
            torch.full((bs,), kv_end, dtype=torch.int32, device=device), dim=0
        ).to(torch.int32)
        return ("BSND", q_seqlens, kv_cu_seqlens, 0, kv_end)

    bs, seq_len = attention_mask.shape
    assert bs == 1, "must be"

    # assume padding belongs to the last sequence
    max_value = torch.max(attention_mask)
    attention_mask = attention_mask.clone()
    attention_mask[attention_mask == 0] = max_value

    if sp_size > 1:
        chunk_size = seq_len // sp_size
        assert seq_len % sp_size == 0, "must be"
        start = sp_rank * chunk_size
        end = (sp_rank + 1) * chunk_size
        local_attention_mask = attention_mask[:, start:end]
        attention_mask = attention_mask[:, 0:end]
    else:
        sp_rank = 0
        start = 0
        end = seq_len
        local_attention_mask = attention_mask

    local_seqlens = get_seqlens_in_batch(local_attention_mask)
    q_cu_seqlens = torch.cumsum(local_seqlens, dim=0).to(torch.int32)
    if len(q_cu_seqlens) == 0:
        fake_len = local_attention_mask.size(1)
        q_cu_seqlens = F.pad(q_cu_seqlens, (1, 0))
        q_cu_seqlens[0] = fake_len
        return ("TND", q_cu_seqlens, q_cu_seqlens, 0, fake_len)

    full_seqlens = get_seqlens_in_batch(attention_mask)  # (num_sequences,)
    full_cu_seqlens = F.pad(
        torch.cumsum(full_seqlens, dim=0), (1, 0)
    )  # (num_sequences+1,)

    # deal with kv
    unique_ids = torch.unique(
        local_attention_mask[local_attention_mask != 0], sorted=True
    )
    assert len(unique_ids) > 0, "must be"

    kv_seqlens_list = []
    for seq_id in unique_ids:
        seq_idx = seq_id - 1
        kv_seqlens_list.append(full_seqlens[seq_idx].item())

    kv_cu_seqlens = torch.cumsum(
        torch.tensor(kv_seqlens_list, dtype=torch.int32, device=attention_mask.device),
        dim=0,
    ).to(torch.int32)

    first_seq_id = unique_ids[0] - 1
    kv_start = full_cu_seqlens[first_seq_id].item()
    kv_end = end

    return ("TND", q_cu_seqlens, kv_cu_seqlens, kv_start, kv_end)


def indexer_forward_with_scores(
    self,
    query_index: torch.Tensor,
    key_index: torch.Tensor,
    weights: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    sp_group: ProcessGroup,
    sp_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Second phase of NPU lightning indexer: compute topk indices and scores.

    This uses the precomputed query, key, weights from forward_with_index
    to compute the sparse attention indices.

    Args:
        self: GlmMoeDsaIndexer instance
        query_index: [B, S/sp, H, D] or [T/sp, H, D] - Query in BSND or TND layout
        key_index: [B, T, 1, D] or [T, 1, D] - Key in BSND or TND layout (global after all-gather)
        weights: [B, S/sp, H] or [T/sp, H] - Weights in BSND or TND layout
        attention_mask: [1, total_tokens] - Packing attention mask with values 1, 2, 3...

    Returns:
        topk_indices: [T_local, 1, topk] - Selected sparse indices (relative positions, to be fixed to global, TND layout)
                      [B, T_local, 1, topk] - BSND layout
        topk_score: [T_local, 1, topk] - Scores for selected indices (TND layout)
                    [B, T_local, 1, topk] - BSND layout
    """
    topk = self.index_topk

    # Get cumulative sequence lengths for TND layout
    mode, q_cu_seqlens, kv_cu_seqlens, kv_start, kv_end = get_sp_local_cu_seqlens(
        attention_mask,
        query_index.size(0),
        query_index.size(1),
        query_index.device,
        sp_group,
        sp_size,
    )

    # deal with inputs
    if mode == "TND":
        query_index = query_index.view(-1, query_index.size(-2), query_index.size(-1))
        key_index = key_index.view(-1, 1, key_index.size(-1))[kv_start:kv_end]
        weights = weights.view(-1, weights.size(-1))
    else:
        query_index = query_index.view(-1, query_index.size(-2), query_index.size(-1))
        key_index = key_index[:, kv_start:kv_end].contiguous()
        key_index = key_index.view(-1, 1, key_index.size(-1))
        weights = weights.view(-1, weights.size(-1))

    topk_indices, topk_score = torch_npu.npu_lightning_indexer(
        query_index,
        key_index,
        weights,
        actual_seq_lengths_query=q_cu_seqlens,
        actual_seq_lengths_key=kv_cu_seqlens,
        layout_query="TND",
        layout_key="TND",
        sparse_count=topk,
        sparse_mode=3,  # rightDownCausal mode
        return_value=True,
    )

    return topk_indices, topk_score


def generate_sparse_mask(
    topk_indices: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    seq_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Generate sparse attention mask from topk indices.

    Args:
        topk_indices: [T, 1, topk] - Fixed top-k indices (global positions, TND layout)
        attention_mask: Packing attention mask (not used for sparse mode)
        seq_length: Sequence length
        dtype: Data type for the mask
        device: Device for the mask

    Returns:
        None (sparse mask is handled internally by npu_sparse_flash_attention)
    """
    # For npu_sparse_flash_attention, we pass topk_indices directly
    # No need to generate explicit mask
    return None


# def dsa_attn_forward(
#     self,
#     hidden_states: torch.Tensor,
#     position_embeddings: Tuple[torch.Tensor, torch.Tensor],
#     attention_mask: Optional[torch.Tensor],
#     past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
#     use_cache: bool = False,
#     output_attentions: bool = False,
#     **kwargs,
# ):
#     """
#     DSA attention forward.
#     """
#     batch_size, seq_length = hidden_states.shape[:-1]
#     cos, sin = position_embeddings
#
#     if self.q_lora_rank is None:
#         query_states = self.q_proj(hidden_states)
#         q_resid = None
#     else:
#         q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
#         query_states = self.q_b_proj(q_resid)
#
#     query_states = query_states.view(batch_size, seq_length, -1, self.qk_head_dim).transpose(1, 2)
#     q_nope, q_pe = torch.split(query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
#     q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=1)
#
#     compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
#     k_compressed, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
#     k_compressed = self.kv_a_layernorm(k_compressed)
#
#     kv_expanded = self.kv_b_proj(k_compressed)
#     kv_expanded = kv_expanded.view(batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)
#     k_nope, value_states = torch.split(kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
#     k_nope = k_nope.transpose(1, 2)
#     value_states = value_states.transpose(1, 2)
#
#     k_pe = k_pe.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
#     k_pe = apply_rotary_pos_emb(k_pe, cos, sin, unsqueeze_dim=1)
#     k_pe = k_pe.expand(-1, k_nope.shape[1], -1, -1)
#
#     query_states = torch.cat([q_nope, q_pe], dim=-1)
#     key_states = torch.cat([k_nope, k_pe], dim=-1)
#
#     if past_key_value is not None:
#         past_key, past_value = past_key_value
#         key_states = torch.cat([past_key, key_states], dim=2)
#         value_states = torch.cat([past_value, value_states], dim=2)
#
#     present_key_value = (key_states, value_states) if use_cache else None
#
#     # =================================================================
#     # Indexer: generate sparse mask
#     # =================================================================
#     topk_indices = indexer_forward_npu(
#         self.indexer,
#         hidden_states.detach(),
#         q_resid.detach(),
#         position_embeddings,
#         attention_mask,
#         use_cache=use_cache,
#     )
#
#     # Convert to sparse mask
#     indices_fixed = fix_npu_topk_indices(topk_indices, attention_mask)
#     combined_mask = create_index_mask(indices_fixed, dtype=query_states.dtype)
#     # =================================================================
#
#
#     attn_output, attn_weights = eager_attention_forward(
#         self,
#         query_states,
#         key_states,
#         value_states,
#         combined_mask,
#         dropout=0.0 if not self.training else self.attention_dropout,
#         scaling=self.scaling,
#         **kwargs,
#     )
#
#     attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
#     attn_output = self.o_proj(attn_output)
#
#     if not output_attentions:
#         attn_weights = None
#
#     return attn_output, attn_weights, present_key_value


def _npu_prepare_qkv_for_mla_absorb(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Prepare QKV tensors for NPU fused DSA attention with MLA absorb mode.

    This function performs:
    1. Query projection and RoPE application
    2. MLA Absorb: absorb W_UK into W_UQ (q_nope projection)
    3. Key-Value compression (MLA absorb mode: no up-projection)

    Args:
        self: GlmMoeDsaAttention module
        hidden_states: [batch_size, seq_len, hidden_size]
        position_embeddings: Tuple of (cos, sin) for rotary embeddings

    Returns:
        q_nope_absorbed: [B, S, H, kv_lora_rank] - Query with absorbed W_UK
        q_pe: [B, S, H, qk_rope_head_dim] - Query RoPE component
        k_lat: [B, S, 1, kv_lora_rank] - Compressed key (no up-projection)
        v_lat: [B, S, 1, kv_lora_rank] - Compressed value (no up-projection)
        k_pe: [B, S, 1, qk_rope_head_dim] - Key RoPE component
        q_resid: [B, S, q_lora_rank] or None - Query residual for indexer
        kv_b_weight: [H, qk_nope + v_head, kv_lora_rank] - KV projection weight for output projection
    """
    batch_size, seq_length = hidden_states.shape[:-1]
    cos, sin = position_embeddings

    # =====================================================================
    # 1. Query projection and RoPE application
    # =====================================================================
    if self.q_lora_rank is None:
        query_states = self.q_proj(hidden_states)
        q_resid = None
    else:
        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        query_states = self.q_b_proj(q_resid)

    # Reshape to [batch_size, seq_len, num_heads, qk_head_dim]
    query_states = query_states.view(
        batch_size, seq_length, self.num_heads, self.qk_head_dim
    )
    q_nope, q_pe = torch.split(
        query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
    )

    # Apply RoPE to q_pe: [B, S, H, D]
    q_pe = apply_rotary_pos_emb(
        q_pe.transpose(1, 2), cos, sin, unsqueeze_dim=1
    ).transpose(1, 2)

    # =====================================================================
    # 2. MLA Absorb: absorb W_UK into W_UQ (q_nope projection)
    #    attn_score = (q * W_UQ) * (k * W_UK)^T
    #              = (q * W_UQ * W_UK^T) * k^T
    # =====================================================================
    # Get kv_b weight and reshape to [H, qk_nope + v_head, kv_lora_rank]
    kv_b_weight = self.kv_b_proj.weight.view(self.num_heads, -1, self.kv_lora_rank)

    # q_nope: [B, S, H, qk_nope] @ [H, qk_nope, kv_lora] -> [B, S, H, kv_lora]
    q_nope_absorbed = torch.einsum(
        "bshd,hdc->bshc", q_nope, kv_b_weight[:, : self.qk_nope_head_dim]
    )
    q_nope_absorbed = q_nope_absorbed.to(hidden_states.dtype)

    # =====================================================================
    # 3. Key-Value compression (MLA absorb mode: no up-projection)
    # =====================================================================
    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    k_compressed, k_pe = torch.split(
        compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )
    k_compressed = self.kv_a_layernorm(k_compressed)

    # Use compressed KV directly as k_lat and v_lat (no up-projection)
    k_lat = k_compressed.unsqueeze(2)  # [B, S, 1, kv_lora_rank]
    v_lat = k_compressed.unsqueeze(2)  # [B, S, 1, kv_lora_rank]

    # Apply RoPE to k_pe: [B, S, 1, qk_rope_head_dim]
    k_pe = k_pe.unsqueeze(2)  # [B, S, 1, qk_rope_head_dim]
    k_pe = apply_rotary_pos_emb(
        k_pe.transpose(1, 2), cos, sin, unsqueeze_dim=1
    ).transpose(1, 2)

    return q_nope_absorbed, q_pe, k_lat, v_lat, k_pe, q_resid, kv_b_weight


def _npu_compute_sparse_indices(
    self,
    hidden_states: torch.Tensor,
    q_resid: Optional[torch.Tensor],
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    sp_enabled: bool,
    sp_group: ProcessGroup,
    sp_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute sparse indices using npu_lightning_indexer.

    SP strategy:
        Available when sequence parallel size is 2^n. The sp rank should
        satisfy the kv-key sequence requirement of npu_lightning_indexer.

    Args:
        self: GlmMoeDsaAttention module
        hidden_states: [batch_size, seq_len/sp_size, hidden_size] (local with SP)
        q_resid: [batch_size, seq_len/sp_size, q_lora_rank] or None (local with SP)
        position_embeddings: Tuple of (cos, sin)
        attention_mask: Packing attention mask (full sequence, not split in SP)
        sp_enabled: Whether sequence parallelism is enabled

    Returns:
        topk_indices: [T, 1, topk] - Selected sparse indices in TND layout
        query_index: [B, S/sp_size, H, D] - Query for lightning indexer in BSND
        key_index: [B, S/sp_size, 1, D] - Key for lightning indexer in BSND
        weights_indexer: [B, S/sp_size, H] - Weights for lightning indexer in BSND
    """
    # Phase 1: Get query, key, weights for indexer (LOCAL sequence, SP parallel)
    query_index, key_index, weights_indexer = indexer_forward_with_index(
        self.indexer,
        hidden_states.detach(),
        q_resid.detach() if q_resid is not None else None,
        position_embeddings,
    )

    # Phase 1.5: All-gather full key sequence (if SP enabled)
    if sp_enabled:
        # All-gather key_index: [B, N/sp, 1, D] -> [B, N, 1, D]
        key_index = all_gather_sp(key_index, dim=1, sp_group=sp_group, sp_size=sp_size)

    # Phase 2: Get topk indices using npu_lightning_indexer
    topk_indices, _ = indexer_forward_with_scores(
        self.indexer,
        query_index,
        key_index,
        weights_indexer,
        attention_mask,
        sp_group,
        sp_size,
    )

    return topk_indices, query_index, key_index, weights_indexer


def npu_fused_dsa_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False,
    output_attentions: bool = False,
    sp_group: ProcessGroup = None,
    sp_size: int = None,
    prev_topk_indices: Optional[torch.Tensor] = None,
    **kwargs,
):
    """
    NPU fused DSA attention forward with MLA absorb mode (training only).

    This implementation:
    1. Uses MLA absorb pattern - absorb W_UK into q_nope projection
    2. Uses compressed KV directly (no up-projection) for k_lat and v_lat
    3. Uses npu_lightning_indexer for sparse index selection
    4. Uses npu_sparse_flash_attention for core attention
    5. Uses fused_sparse_lightning_indexer_kl_loss for training loss
    6. Supports Ulysses sequence parallelism (SP)

    Ulysses SP for MLA (all-to-all communication is done via all_gather_sp, not all-to-all):
    - KV (head=1): all-gather sequence before attention -> [B, seq/sp, 1, D] -> [B, seq, 1, D]
    - Q (H heads): no communication before attention (heads are parallel)

    Args:
        hidden_states: [batch_size, seq_len/sp_size, hidden_size] (with SP)
        position_embeddings: Tuple of (cos, sin) for rotary embeddings
        attention_mask: Packing attention mask [1, seq_len] with values 1, 2, 3... (full sequence, not split)
        past_key_value: Must be None (training only, no KV cache)
        use_cache: Must be False (training only, no KV cache)
        output_attentions: Must be False (not supported in sparse mode)

    Returns:
        attn_output: [batch_size, seq_len/sp_size, hidden_size] (with SP)
        attn_weights: None (output_attentions not supported)
        present_key_value: None (training only, no KV cache)
        topk_indices: sparse indices for index-share relay (reused by shared layers)
    """
    # Training only - no KV cache support
    assert (
        past_key_value is None
    ), "npu_fused_dsa_attn_forward does not support past_key_value (training only)"
    assert (
        not use_cache
    ), "npu_fused_dsa_attn_forward does not support use_cache (training only)"
    assert (
        not output_attentions
    ), "npu_fused_dsa_attn_forward does not support output_attentions"

    assert sp_group, "sp_group should not be none"
    assert sp_size, "sp_size should not be none"
    sp_enabled = sp_size > 1

    # =====================================================================
    # 1-3. Prepare QKV tensors with MLA absorb
    # =====================================================================
    batch_size = hidden_states.size(0)
    q_nope_absorbed, q_pe, k_lat, v_lat, k_pe, q_resid, kv_b_weight = (
        _npu_prepare_qkv_for_mla_absorb(self, hidden_states, position_embeddings)
    )

    # =====================================================================
    # 4. Compute sparse indices using npu_lightning_indexer
    #    index share: full 层自己跑 indexer 产出 top-k；shared 层复用上一个 full 层的 top-k。
    # =====================================================================
    if self.indexer is not None:
        topk_indices, query_index, key_index, weights_indexer = _npu_compute_sparse_indices(
            self,
            hidden_states,
            q_resid,
            position_embeddings,
            attention_mask,
            sp_enabled,
            sp_group,
            sp_size,
        )
    else:
        if prev_topk_indices is None:
            raise ValueError(
                "Shared DSA layers require top-k indices from a previous full indexer layer."
            )
        topk_indices = prev_topk_indices
        query_index = key_index = weights_indexer = None

    # =====================================================================
    # 5. SP communication before attention
    #    KV: all-gather sequence to get full context (since head=1, can't use all-to-all)
    #    Q: heads are already parallel (no communication needed before attention)
    # =====================================================================
    if sp_enabled:
        # KV: all-gather sequence to get full context for attention
        # k_lat: [B, seq/sp, 1, kv_lora] -> [B, seq, 1, kv_lora]
        k_lat = all_gather_sp(k_lat, dim=1, sp_group=sp_group, sp_size=sp_size)
        # v_lat: [B, seq/sp, 1, kv_lora] -> [B, seq, 1, kv_lora]
        v_lat = all_gather_sp(v_lat, dim=1, sp_group=sp_group, sp_size=sp_size)
        # k_pe: [B, seq/sp, 1, qk_rope] -> [B, seq, 1, qk_rope]
        k_pe = all_gather_sp(k_pe, dim=1, sp_group=sp_group, sp_size=sp_size)

    # =====================================================================
    # 6. Core attention - sparse attention with npu_sparse_flash_attention (TND layout)
    # =====================================================================
    # TND layout: T = total_tokens, N = num_heads, D = head_dim

    # Get cumulative sequence lengths for TND layout
    mode, q_cu_seqlens, kv_cu_seqlens, kv_start, kv_end = get_sp_local_cu_seqlens(
        attention_mask,
        q_nope_absorbed.size(0),
        q_nope_absorbed.size(1),
        q_nope_absorbed.device,
        sp_group,
        sp_size,
    )
    # kv_start and kv_end can be used to slice the full KV tensor to get relevant portion
    # For now, the full KV is all-gathered, so these are not immediately used
    scale_value = self.scaling

    num_heads = q_nope_absorbed.shape[2]
    if mode == "TND":
        query = q_nope_absorbed.view(-1, num_heads, self.kv_lora_rank)
        query_rope = q_pe.view(-1, num_heads, self.qk_rope_head_dim)
        key = k_lat.view(-1, 1, self.kv_lora_rank)[kv_start:kv_end]
        value = v_lat.view(-1, 1, self.kv_lora_rank)[kv_start:kv_end]
        key_rope = k_pe.view(-1, 1, self.qk_rope_head_dim)[kv_start:kv_end]
        sparse_indices = topk_indices.view(-1, 1, topk_indices.size(-1))
    else:
        # BSND to TND
        query = q_nope_absorbed.contiguous().view(-1, num_heads, self.kv_lora_rank)
        query_rope = q_pe.contiguous().view(-1, num_heads, self.qk_rope_head_dim)
        key = k_lat[:, kv_start:kv_end].contiguous().view(-1, 1, self.kv_lora_rank)
        value = v_lat[:, kv_start:kv_end].contiguous().view(-1, 1, self.kv_lora_rank)
        key_rope = (
            k_pe[:, kv_start:kv_end].contiguous().view(-1, 1, self.qk_rope_head_dim)
        )
        sparse_indices = topk_indices.view(-1, 1, topk_indices.size(-1))

    # Call npu_sparse_flash_attention with TND layout
    attn_output, softmax_max, softmax_sum, *_ = torch_npu.npu_sparse_flash_attention(
        query,
        key,
        value,
        sparse_indices,
        scale_value,  # positional argument
        block_table=None,
        actual_seq_lengths_query=q_cu_seqlens,
        actual_seq_lengths_kv=kv_cu_seqlens,
        query_rope=query_rope,
        key_rope=key_rope,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="TND",
        sparse_mode=3,
        attention_mode=2,
        return_softmax_lse=True,
    )

    # =====================================================================
    # 7. Compute DSA indexer loss BEFORE post-attention SP
    #    (NPU API requires H=64 or 128, so compute loss with full heads)
    # =====================================================================
    loss = None
    indexer_loss_coeff = getattr(self.config, "dsa_indexer_loss_coeff", 0.01)
    if self.indexer is not None and self.training and torch.is_grad_enabled() and indexer_loss_coeff > 0:
        if mode == "TND":
            query_index_for_loss = query_index.view(
                -1, query_index.size(-2), query_index.size(-1)
            )
            weights_indexer_for_loss = weights_indexer.view(
                -1, weights_indexer.size(-1)
            )
            key_index_for_loss = key_index.view(-1, 1, key_index.size(-1))[
                kv_start:kv_end
            ]
        else:
            query_index_for_loss = query_index.view(
                -1, query_index.size(-2), query_index.size(-1)
            )
            weights_indexer_for_loss = weights_indexer.view(
                -1, weights_indexer.size(-1)
            )
            key_index = key_index[:, kv_start:kv_end].contiguous()
            key_index_for_loss = key_index.view(-1, 1, key_index.size(-1))

        softmax_max_for_loss = softmax_max
        softmax_sum_for_loss = softmax_sum
        query_for_loss = query
        query_rope_for_loss = query_rope
        key_for_loss = key
        key_rope_for_loss = key_rope
        # Compute loss
        loss = fused_sparse_lightning_indexer_kl_loss(
            query_for_loss,
            key_for_loss,
            query_index_for_loss,
            key_index_for_loss,
            weights_indexer_for_loss,
            topk_indices,
            softmax_max_for_loss,
            softmax_sum_for_loss,
            scale_value=scale_value,
            query_rope=query_rope_for_loss,
            key_rope=key_rope_for_loss,
            actual_seq_qlen=q_cu_seqlens,
            actual_seq_klen=kv_cu_seqlens,
            layout="TND",
            sparse_mode=3,
        )
        loss = loss * indexer_loss_coeff

    attn_output = attn_output.view(batch_size, -1, self.num_heads, self.kv_lora_rank)

    # =====================================================================
    # 8. MLA Absorb: absorb W_UV into W_O (output projection)
    #    attn_out = attn_weight * (v * W_UV) * W_O
    #             = (attn_weight * v * W_UV) * W_O
    # =====================================================================
    # kv_b_weight is already [H, qk_nope + v_head, kv_lora_rank]
    # Get W_UV: [H, v_head, kv_lora] and permute to [H, kv_lora, v_head]
    w_uv = kv_b_weight[:, -self.v_head_dim :]  # [H, v_head, kv_lora_rank]
    # Sparse mode: attn_output is [B, S, H, kv_lora_rank] (after SP restore)
    # Project: [B, S, H, kv_lora] @ [H, kv_lora, v_head]^T -> [B, S, H, kv_lora]
    attn_output = torch.einsum("bshd,hvd->bshv", attn_output, w_uv).contiguous()
    attn_output = attn_output.to(hidden_states.dtype)
    # Reshape to [B, S, H * v_head_dim]
    attn_output = attn_output.view(batch_size, -1, self.num_heads * self.v_head_dim)

    # Output projection
    attn_output = self.o_proj(attn_output)

    # =====================================================================
    # 9. Apply DSA indexer loss (if computed)
    # =====================================================================
    if loss is not None:
        attn_output = DSAIndexerLossAutoScaler.apply(attn_output, loss)

    return attn_output, None, None, topk_indices
