from typing import Optional, Tuple

from contextlib import nullcontext

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import torch.distributed as dist

import torch_npu

from .modeling_glm_moe_dsa import apply_rotary_pos_emb, eager_attention_forward
from torch_turbo.utils.flash_attn_utils import packing_mask_to_causal_mask, get_seqlens_in_batch


# ============================================================================
# MLA Absorb Mode - DSA Indexer Loss AutoScaler
# ============================================================================

class DSAIndexerLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that triggers the backward pass and scales the grad for DSA indexer loss."""

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
        dsa_indexer_loss_backward_scale = DSAIndexerLossAutoScaler.main_loss_backward_scale
        scaled_dsa_indexer_loss_grad = torch.ones_like(loss) * dsa_indexer_loss_backward_scale
        return grad_output, scaled_dsa_indexer_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = scale
        else:
            DSAIndexerLossAutoScaler.main_loss_backward_scale.copy_(scale)


# ============================================================================
# MLA Absorb Mode - Fused Lightning Indexer KL Loss
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
            layout='TND',
            sparse_mode=3,
    ):
        d_query_index, d_key_index, d_weights, loss = torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
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
        layout='TND',
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
    # =====================================================================
    # NPU API Constraints Validation
    # =====================================================================
    num_heads = query.shape[1]  # N dimension in TND layout
    topk = sparse_indices.shape[-1]

    # Constraint: query shape N must be 64 or 128
    if num_heads not in (64, 128):
        raise ValueError(
            f"npu_sparse_lightning_indexer_grad_kl_loss requires query N (num_heads) "
            f"to be 64 or 128, but got {num_heads}. "
            f"Please config num_attention_heads=64 or 128 in model config."
        )

    # Constraint: topK must be 2048 (integer multiple of 1024, < 8192)
    if topk != 2048:
        raise ValueError(
            f"npu_sparse_lightning_indexer_grad_kl_loss requires topK=2048, "
            f"but got topK={topk}. Please config index_topk=2048 in model config."
        )

    # Get dimensions for scaling
    total_tokens = query.shape[0]

    loss = LILossTrain.apply(
        query, key, query_index, key_index, weights,
        sparse_indices, softmax_max, softmax_sum,
        scale_value, query_rope, key_rope,
        actual_seq_qlen, actual_seq_klen, layout, sparse_mode,
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
        batch_size: Batch size
        seq_len: Sequence length
        total_len: Total key length
        topk: Number of top-k indices

    Returns:
        Fixed topk_indices with invalid positions replaced
    """
    batch_size, seq_len, topk = topk_indices.shape
    dtype = topk_indices.dtype

    # Get sequence start positions
    seqlens = get_seqlens_in_batch(attention_mask)  # (num_sequences,)
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0), (1, 0))  # (num_sequences + 1,)

    # Create a mapping from each position to its sequence start position
    # seq_start_positions[i] = start position of the sequence that position i belongs to
    seq_start_positions = torch.zeros(seq_len, dtype=torch.long, device=topk_indices.device)
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
    query_seq_ids = sequence_ids.unsqueeze(0).unsqueeze(-1).expand_as(topk_indices)  # [1, seq_len, topk]

    # Gather sequence_ids for the selected key positions
    selected_seq_ids = torch.gather(
        sequence_ids.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, topk),
        dim=1,
        index=topk_indices
    )  # [batch, seq_len, topk]

    # Check if selected indices are valid:
    # 1. Same sequence as query
    is_same_seq = (selected_seq_ids == query_seq_ids)

    # 2. Causal: selected_index <= query_position (within same sequence)
    query_positions = torch.arange(seq_len, device=topk_indices.device).unsqueeze(0).unsqueeze(-1)
    is_causal = (topk_indices <= query_positions)
    is_valid = topk_indices >= 0

    # Valid if both conditions are met
    valid_mask = is_same_seq & is_causal & is_valid

    # Replace invalid indices with the start position of the corresponding sub-sequence
    seq_start_positions_expanded = seq_start_positions.unsqueeze(0).unsqueeze(-1).expand_as(topk_indices)
    topk_indices = torch.where(valid_mask, topk_indices, seq_start_positions_expanded)

    return topk_indices.to(dtype)


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
        topk_indices: [1, total_tokens, topk] - NPU output with relative positions and -1 for invalid
        attention_mask: [1, total_tokens] - Packing attention mask with values 1, 2, 3...

    Returns:
        topk_indices: [1, total_tokens, topk] - Global indices with invalid positions fixed
    """
    _, total_tokens, topk = topk_indices.shape
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

    # Expand seq_start_positions to match topk_indices shape
    seq_start_expanded = seq_start_positions.unsqueeze(0).unsqueeze(-1).expand_as(topk_indices)

    # Create mask for valid indices (not -1)
    valid_mask = topk_indices >= 0

    # Convert relative positions to global positions
    # global_position = seq_start + relative_position
    global_indices = torch.where(valid_mask, seq_start_expanded + topk_indices, topk_indices)

    # For invalid positions (-1), replace with the corresponding sub-sequence start position
    # This ensures each query at least attends to the first token of its sub-sequence
    fixed_indices = torch.where(valid_mask, global_indices, seq_start_expanded)

    return fixed_indices.to(dtype)

def create_index_mask(
    topk_indices: torch.Tensor,
    dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Create sparse attention index mask from topk indices for packing mode.

    The output mask can be added to attention scores to implement sparse attention:
    - Selected positions (from topk_indices): 0.0
    - Other positions: -inf

    Args:
        topk_indices: [1, seq_length, topk] - Selected top-k indices (batch_size=1 for packing)

    Returns:
        index_mask: [1, 1, seq_length, total_len] - Sparse attention mask
    """
    batch_size, seq_length, _ = topk_indices.shape
    total_len = topk_indices.shape[1]  # For packing mode, total_len == seq_length
    device = topk_indices.device

    # Create full mask with -inf
    index_mask = torch.full(
        (batch_size, seq_length, total_len),
        float("-inf"),
        device=device,
        dtype=dtype,
    )
    # Set selected positions to 0.0
    index_mask.scatter_(-1, topk_indices, 0.0)
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
    q_pe, q_nope = torch.split(q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
    q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=2)
    q = torch.cat([q_pe, q_nope], dim=-1)

    k = self.k_norm(self.wk(hidden_states))
    k_pe, k_nope = torch.split(k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
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

    weights = self.weights_proj(hidden_states).float() * (self.n_heads ** -0.5)

    scores = torch.einsum("bshd,btd->bsht", q.float(), k_cached.float()) * self.softmax_scale
    scores = F.relu(scores)
    index_scores = torch.einsum("bsht,bsh->bst", scores, weights)

    if attention_mask is not None:
        # Convert packing mask to causal mask and apply to index_scores
        # packing_mask_to_causal_mask returns (total_len, total_len) bool mask
        causal_mask = packing_mask_to_causal_mask(attention_mask)
        # Convert to additive mask: True -> 0, False -> -inf
        additive_mask = torch.where(causal_mask, 0.0, float('-inf'))
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
        topk_indices: [1, total_tokens, topk] - Top-k indices for sparse attention
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

    Args:
        self: GlmMoeDsaIndexer instance
        hidden_states: [batch_size, seq_len, hidden_size] - Input hidden states (BSND format)
        q_resid: [batch_size, seq_len, q_lora_rank] - Low-rank query residual
        position_embeddings: Tuple of (cos, sin) for rotary embeddings
        use_cache: Must be False (training only, no KV cache)

    Returns:
        query_index: [T, N, D] - Query for lightning indexer in TND layout
        key_index: [T, 1, D] - Key for lightning indexer in TND layout
        weights: [T, N] - Weights for lightning indexer in TND layout
    """
    # Training only - no KV cache support
    assert not use_cache, "indexer_forward_with_index does not support use_cache (training only)"

    batch_size, seq_len, _ = hidden_states.shape
    cos, sin = position_embeddings

    # Project low-rank query to full multi-head query
    q = self.wq_b(q_resid)
    q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)  # [B, S, N, D]

    # Apply rotary positional embedding to the RoPE part of the query
    q_pe, q_nope = torch.split(q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
    q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=2)
    q = torch.cat([q_pe, q_nope], dim=-1)

    # Project and normalize keys
    k = self.k_norm(self.wk(hidden_states))
    k_pe, k_nope = torch.split(k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
    k_pe = apply_rotary_pos_emb(k_pe.unsqueeze(2), cos, sin, unsqueeze_dim=2).squeeze(2)
    k = torch.cat([k_pe, k_nope], dim=-1)

    # Compute weights with scaling
    weights = self.weights_proj(hidden_states)
    weights = weights * (self.n_heads ** -0.5)
    weights = weights * self.softmax_scale

    # Convert to bfloat16 for NPU operator
    q_bf16 = q.to(torch.bfloat16)
    k_bf16 = k.to(torch.bfloat16)
    weights_bf16 = weights.to(torch.bfloat16)

    # Ensure key has head dimension: [B, S, head_dim] -> [B, S, 1, head_dim]
    if k_bf16.dim() == 3:
        k_bf16 = k_bf16.unsqueeze(2)

    # Store total_len for forward_with_scores
    total_len = k_bf16.shape[1]
    self._total_key_len = total_len

    # Convert from BSND to TND: [B, S, N, D] -> [T, N, D]
    query_index = q_bf16.view(1, seq_len, self.n_heads, self.head_dim).squeeze(0)  # [T, N, D]
    key_index = k_bf16.view(1, total_len, 1, self.head_dim).squeeze(0)  # [T, 1, D]
    weights_tnd = weights_bf16.view(1, seq_len, self.n_heads).squeeze(0)  # [T, N]

    return query_index, key_index, weights_tnd

def get_cu_seqlens_npu(attention_mask):
    assert attention_mask is not None, "must be"
    if hasattr(attention_mask, "cu_seqlens"):
        return attention_mask.cu_seqlens

    seqlens = get_seqlens_in_batch(attention_mask)  # (num_sequences,)
    cu_seqlens = torch.cumsum(seqlens, dim=0).to(torch.int32)  # (num_sequences)
    attention_mask.cu_seqlens = cu_seqlens
    return cu_seqlens

def indexer_forward_with_scores(
    self,
    query_index: torch.Tensor,
    key_index: torch.Tensor,
    weights: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Second phase of NPU lightning indexer: compute topk indices and scores.

    This uses the precomputed query, key, weights from forward_with_index
    to compute the sparse attention indices.

    Args:
        self: GlmMoeDsaIndexer instance
        query_index: [T, N, D] - Query from forward_with_index in TND layout
        key_index: [T, 1, D] - Key from forward_with_index in TND layout
        weights: [T, N] - Weights from forward_with_index in TND layout
        attention_mask: [1, total_tokens] - Packing attention mask with values 1, 2, 3...

    Returns:
        topk_indices: [1, total_tokens, topk] - Selected sparse indices (relative positions, unfixed)
        topk_score: [1, total_tokens, topk] - Scores for selected indices
    """
    topk = min(self.index_topk, query_index.shape[0])

    # Get cumulative sequence lengths for TND layout
    cu_seqlens = get_cu_seqlens_npu(attention_mask)

    # NPU lightning indexer with TND layout
    topk_indices, topk_score = torch_npu.npu_lightning_indexer(
        query_index,
        key_index,
        weights,
        actual_seq_lengths_query=cu_seqlens,
        actual_seq_lengths_key=cu_seqlens,
        layout_query='TND',
        layout_key='TND',
        sparse_count=topk,
        sparse_mode=3,  # rightDownCausal mode
        return_value=True,
    )

    # Reshape: [T, 1, topk] -> [1, T, topk]
    topk_indices = topk_indices.squeeze(1).unsqueeze(0)
    topk_score = topk_score.squeeze(1).unsqueeze(0)

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
        topk_indices: [1, seq_len, topk] - Fixed top-k indices (global positions)
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


def dsa_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False,
    output_attentions: bool = False,
    skip_indexer: bool = False,  # If True, skip indexer and use pure causal mask
    **kwargs,
):
    """
    DSA attention forward with optional indexer skip for debugging.

    Args:
        skip_indexer: If True, skip indexer and use pure causal mask for attention.
                      This is useful for comparing MLA naive vs MLA absorb without indexer interference.
    """
    batch_size, seq_length = hidden_states.shape[:-1]
    cos, sin = position_embeddings

    if self.q_lora_rank is None:
        query_states = self.q_proj(hidden_states)
        q_resid = None
    else:
        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        query_states = self.q_b_proj(q_resid)

    query_states = query_states.view(batch_size, seq_length, -1, self.qk_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=1)

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    k_compressed, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_compressed = self.kv_a_layernorm(k_compressed)

    kv_expanded = self.kv_b_proj(k_compressed)
    kv_expanded = kv_expanded.view(batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)
    k_nope, value_states = torch.split(kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    k_nope = k_nope.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    k_pe = k_pe.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
    k_pe = apply_rotary_pos_emb(k_pe, cos, sin, unsqueeze_dim=1)
    k_pe = k_pe.expand(-1, k_nope.shape[1], -1, -1)

    query_states = torch.cat([q_nope, q_pe], dim=-1)
    key_states = torch.cat([k_nope, k_pe], dim=-1)

    if past_key_value is not None:
        past_key, past_value = past_key_value
        key_states = torch.cat([past_key, key_states], dim=2)
        value_states = torch.cat([past_value, value_states], dim=2)

    present_key_value = (key_states, value_states) if use_cache else None

    # =================================================================
    # Indexer or pure causal mask
    # =================================================================
    if skip_indexer:
        # Use pure causal mask for debugging MLA vs MLA absorb
        topk_indices = None
        # Create causal mask from packing mask
        causal_mask = packing_mask_to_causal_mask(attention_mask)
        combined_mask = torch.zeros(batch_size, 1, seq_length, seq_length, device=hidden_states.device, dtype=query_states.dtype)
        combined_mask = combined_mask.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    else:
        # Use indexer to generate sparse mask
        topk_indices = indexer_forward_npu(
            self.indexer,
            hidden_states.detach(),
            q_resid.detach(),
            position_embeddings,
            attention_mask,
            use_cache=use_cache,
        )

        # Convert to sparse mask
        indices_fixed = fix_npu_topk_indices(topk_indices, attention_mask)
        combined_mask = create_index_mask(indices_fixed, dtype=query_states.dtype)
    # =================================================================


    attn_output, attn_weights = eager_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        combined_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, present_key_value


def npu_fused_dsa_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False,
    output_attentions: bool = False,
    indexer_loss_coeff: float = 1.0,
    use_fused_loss: bool = True,
    skip_indexer: bool = False,  # If True, skip indexer and use dense attention for debugging
):
    """
    NPU fused DSA attention forward with MLA absorb mode (training only).

    This implementation:
    1. Uses MLA absorb pattern - absorb W_UK into q_nope projection
    2. Uses compressed KV directly (no up-projection) for k_lat and v_lat
    3. Uses npu_lightning_indexer for sparse index selection
    4. Uses npu_sparse_flash_attention for core attention
    5. Uses fused_sparse_lightning_indexer_kl_loss for training loss

    Args:
        hidden_states: [batch_size, seq_len, hidden_size]
        position_embeddings: Tuple of (cos, sin) for rotary embeddings
        attention_mask: Packing attention mask [1, seq_len] with values 1, 2, 3...
        past_key_value: Must be None (training only, no KV cache)
        use_cache: Must be False (training only, no KV cache)
        output_attentions: Must be False (not supported in sparse mode)
        indexer_loss_coeff: Loss coefficient for DSA indexer
        use_fused_loss: Whether to use fused lightning indexer KL loss
        skip_indexer: If True, skip indexer and use dense attention. For debugging MLA vs MLA absorb.

    Returns:
        attn_output: [batch_size, seq_len, hidden_size]
    """
    # Training only - no KV cache support
    assert past_key_value is None, "npu_fused_dsa_attn_forward does not support past_key_value (training only)"
    assert not use_cache, "npu_fused_dsa_attn_forward does not support use_cache (training only)"
    assert not output_attentions, "npu_fused_dsa_attn_forward does not support output_attentions"
    assert attention_mask is not None, "must be, only support packing mode"
    assert torch.max(attention_mask) >= 0, "must be, only support packing mode"

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
    query_states = query_states.view(batch_size, seq_length, self.num_heads, self.qk_head_dim)
    q_nope, q_pe = torch.split(query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

    # Apply RoPE to q_pe: [B, S, H, D]
    q_pe = apply_rotary_pos_emb(q_pe.transpose(1, 2), cos, sin, unsqueeze_dim=1).transpose(1, 2)

    # =====================================================================
    # 2. MLA Absorb: absorb W_UK into W_UQ (q_nope projection)
    #    attn_score = (q * W_UQ) * (k * W_UK)^T
    #              = (q * W_UQ * W_UK^T) * k^T
    # =====================================================================
    # Get kv_b weight and reshape to [H, qk_nope + v_head, kv_lora_rank]
    kv_b_weight = self.kv_b_proj.weight.view(self.num_heads, -1, self.kv_lora_rank)

    # q_nope: [B, S, H, qk_nope] @ [H, qk_nope, kv_lora] -> [B, S, H, kv_lora]
    q_nope_absorbed = torch.einsum('bshd,hdc->bshc', q_nope, kv_b_weight[:, :self.qk_nope_head_dim])
    q_nope_absorbed = q_nope_absorbed.to(hidden_states.dtype)

    # =====================================================================
    # 3. Key-Value compression (MLA absorb mode: no up-projection)
    # =====================================================================
    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    k_compressed, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_compressed = self.kv_a_layernorm(k_compressed)

    # Use compressed KV directly as k_lat and v_lat (no up-projection)
    k_lat = k_compressed.unsqueeze(2)  # [B, S, 1, kv_lora_rank]
    v_lat = k_compressed.unsqueeze(2)  # [B, S, 1, kv_lora_rank]

    # Apply RoPE to k_pe: [B, S, 1, qk_rope_head_dim]
    k_pe = k_pe.unsqueeze(2)  # [B, S, 1, qk_rope_head_dim]
    k_pe = apply_rotary_pos_emb(k_pe.transpose(1, 2), cos, sin, unsqueeze_dim=1).transpose(1, 2)

    # =====================================================================
    # 4. Compute sparse indices using npu_lightning_indexer
    #    Split into forward_with_index and forward_with_scores for reuse
    #    Detach inputs to isolate indexer branch from main attention branch (MindSpeed-LLM pattern)
    # =====================================================================
    if skip_indexer:
        # Skip indexer for debugging MLA absorb vs MLA naive
        topk_indices = None
        query_index, key_index, weights_indexer = None, None, None
    else:
        # Phase 1: Get query, key, weights for indexer (used for both topk and KL loss)
        # Detach inputs to create independent computation graph for indexer
        query_index, key_index, weights_indexer = indexer_forward_with_index(
            self.indexer,
            hidden_states.detach() if use_fused_loss else hidden_states,
            q_resid.detach() if use_fused_loss else q_resid,
            position_embeddings,
        )

        # Phase 2: Get topk indices using npu_lightning_indexer
        # Use no_grad for forward_with_scores when using fused loss (MindSpeed-LLM pattern)
        dsa_indexer_context = torch.no_grad() if use_fused_loss else nullcontext()
        with dsa_indexer_context:
            topk_indices, _ = indexer_forward_with_scores(
                self.indexer,
                query_index,
                key_index,
                weights_indexer,
                attention_mask,
            )

    # =====================================================================
    # 5. Core attention
    # =====================================================================
    if skip_indexer:
        # Use MLA absorb score calculation directly (MQA decode style)
        # Reference: MLA forward MQA decode branch
        # q_nope_absorbed: [B, S, H, kv_lora_rank] (已吸收W_UK)
        # k_lat: [B, S, 1, kv_lora_rank] (compressed KV, 不扩展heads)
        # q_pe: [B, S, H, qk_rope]
        # k_pe: [B, S, 1, qk_rope]
        # v_lat: [B, S, 1, kv_lora_rank]

        # scores = (q_nope_absorbed @ k_lat.T + q_pe @ k_pe.T) * scale
        # [B, S, H, kv_lora] @ [B, T, 1, kv_lora].T -> [B, S, H, T]
        scores_nope = torch.einsum('bshc,btc->bsht', q_nope_absorbed, k_lat.squeeze(2))
        scores_rope = torch.einsum('bshr,btr->bsht', q_pe.squeeze(2), k_pe.squeeze(2))
        scores = (scores_nope + scores_rope) * self.scaling

        # Create causal mask from packing mask and apply to scores
        causal_mask = packing_mask_to_causal_mask(attention_mask)
        scores = scores.masked_fill(~causal_mask.unsqueeze(1), float('-inf'))

        # softmax
        attn_weights = scores.softmax(dim=-1)

        # output = attn_weights @ v_lat
        # [B, S, H, T] @ [B, T, 1, kv_lora] -> [B, S, H, kv_lora]
        attn_output = torch.einsum('bsht,btc->bshc', attn_weights, v_lat.squeeze(2))

        softmax_max, softmax_sum = None, None
    else:
        # Use sparse attention with npu_sparse_flash_attention (TND layout)
        # TND layout: T = total_tokens, N = num_heads, D = head_dim

        # Get cumulative sequence lengths for TND layout
        cu_seqlens = get_cu_seqlens_npu(attention_mask)
        scale_value = self.scaling

        # Convert from BSND to TND layout
        query = q_nope_absorbed.view(-1, self.num_heads, self.kv_lora_rank).to(torch.bfloat16)
        query_rope = q_pe.view(-1, self.num_heads, self.qk_rope_head_dim).to(torch.bfloat16)

        key = k_lat.view(-1, 1, self.kv_lora_rank).to(torch.bfloat16)
        value = v_lat.view(-1, 1, self.kv_lora_rank).to(torch.bfloat16)
        key_rope = k_pe.view(-1, 1, self.qk_rope_head_dim).to(torch.bfloat16)

        sparse_indices = topk_indices.view(-1, 1, self.indexer.index_topk).to(torch.int32)

        # Call npu_sparse_flash_attention with TND layout
        attn_output, softmax_max, softmax_sum, *_ = torch_npu.npu_sparse_flash_attention(
            query, key, value,
            sparse_indices,
            scale_value,  # positional argument
            block_table=None,
            actual_seq_lengths_query=cu_seqlens,
            actual_seq_lengths_kv=cu_seqlens,
            query_rope=query_rope,
            key_rope=key_rope,
            sparse_block_size=1,
            layout_query='TND',
            layout_kv='TND',
            sparse_mode=3,
            attention_mode=2,
            return_softmax_lse=True,
        )

    # =====================================================================
    # 6. MLA Absorb: absorb W_UV into W_O (output projection)
    #    attn_out = attn_weight * (v * W_UV) * W_O
    #             = (attn_weight * v * W_UV) * W_O
    # =====================================================================
    # kv_b_weight is already [H, qk_nope + v_head, kv_lora_rank]
    # Get W_UV: [H, v_head, kv_lora] and permute to [H, kv_lora, v_head]
    w_uv = kv_b_weight[:, -self.v_head_dim:]  # [H, v_head, kv_lora_rank]
    w_uv_t = w_uv.permute(0, 2, 1).contiguous()  # [H, kv_lora_rank, v_head_dim]

    if skip_indexer:
        # Dense mode: attn_output is [B, S, H, kv_lora_rank]
        # Project: [B, S, H, kv_lora] @ [H, kv_lora, v_head] -> [B, S, H, v_head]
        attn_output = torch.einsum('bshd,hdv->bshv', attn_output, w_uv_t).contiguous()
        # Reshape to [B, S, H * v_head_dim]
        attn_output = attn_output.view(batch_size, seq_length, self.num_heads * self.v_head_dim)
    else:
        # Sparse mode: attn_output is [T, H, kv_lora_rank] in TND layout
        # Project: [T, H, kv_lora] @ [H, kv_lora, v_head] -> [T, H, v_head]
        attn_output = torch.einsum('thd,hdv->thv', attn_output, w_uv_t).contiguous()
        attn_output = attn_output.to(hidden_states.dtype)
        # Convert back from TND [T, H, v_head_dim] to [B, S, H * v_head_dim]
        attn_output = attn_output.view(batch_size, seq_length, self.num_heads * self.v_head_dim)

    # Output projection
    attn_output = self.o_proj(attn_output)

    # =====================================================================
    # 7. Compute DSA indexer loss (if training with fused loss and not skip_indexer)
    # =====================================================================
    if self.training and torch.is_grad_enabled() and use_fused_loss and not skip_indexer:
        # query_index, key_index, weights_indexer are already in TND layout:
        # query_index: [T, N, D]
        # key_index: [T, 1, D]
        # weights_indexer: [T, N]

        # Convert query and key from BSND to TND for loss computation
        query_for_loss = q_nope_absorbed.view(-1, self.num_heads, self.kv_lora_rank).detach().to(torch.bfloat16)
        key_for_loss = k_lat.view(-1, 1, self.kv_lora_rank).detach().to(torch.bfloat16)

        # query_rope and key_rope for loss: convert to TND
        query_rope_for_loss = q_pe.view(-1, self.num_heads, self.qk_rope_head_dim).detach().to(torch.bfloat16)
        key_rope_for_loss = k_pe.view(-1, 1, self.qk_rope_head_dim).detach().to(torch.bfloat16)

        # Compute loss using TND layout
        loss = fused_sparse_lightning_indexer_kl_loss(
            query_for_loss,        # [T, H, kv_lora_rank] bfloat16
            key_for_loss,          # [T, 1, kv_lora_rank] bfloat16
            query_index,           # [T, N, D] bfloat16
            key_index,             # [T, 1, D] bfloat16
            weights_indexer,       # [T, N] bfloat16
            topk_indices,          # [T, 1, topk] int32
            softmax_max,           # [1, T, H] float32
            softmax_sum,           # [1, T, H] float32
            scale_value=scale_value,
            query_rope=query_rope_for_loss,  # [T, H, qk_rope] bfloat16
            key_rope=key_rope_for_loss,      # [T, 1, qk_rope] bfloat16
            actual_seq_qlen=cu_seqlens,      # list[int64]
            actual_seq_klen=cu_seqlens,      # list[int64]
            layout='TND',
            sparse_mode=3,
        )
        loss = loss * indexer_loss_coeff

        # Apply loss scaling through autograd function
        attn_output = DSAIndexerLossAutoScaler.apply(attn_output, loss)

    return attn_output, None, None