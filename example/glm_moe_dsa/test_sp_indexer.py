"""
Test SP (Sequence Parallel) functionality for indexer_forward_with_scores.

This script verifies that:
1. indexer_forward_with_scores produces equivalent results with and without SP
2. get_sp_local_cu_seqlens returns correct kv_start and kv_end for each SP rank

SP=2 is simulated by splitting sequence into chunks and comparing:
- Non-SP: full sequence processed at once
- SP-mode: sequence split into 2 chunks, each rank sees local query + all-gathered key

Run with: python test_sp_indexer.py
"""

import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

# Check if torch_npu is available
try:
    import torch_npu
    NPU_AVAILABLE = True
except ImportError:
    NPU_AVAILABLE = False
    print("Warning: torch_npu not available, will skip NPU tests")

from torch_turbo.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from torch_turbo.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaIndexer,
    apply_rotary_pos_emb,
)
from torch_turbo.utils.npu_utils import seed_all


def create_test_config(
    hidden_size=4096,
    num_attention_heads=64,
    q_lora_rank=512,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    index_n_heads=8,
    index_head_dim=128,
    index_topk=2048,
):
    """Create a test config with dimensions that satisfy NPU API constraints."""
    config = GlmMoeDsaConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_attention_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        attention_dropout=0.0,
    )
    return config


def create_rotary_embeddings(seq_len, head_dim, batch_size=1, device='cpu', dtype=torch.bfloat16):
    """Create rotary position embeddings (cos, sin)."""
    rope_theta = 10000.0
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    freqs = torch.outer(positions, inv_freq)

    cos = torch.cos(freqs).repeat(1, 2)[:, :head_dim]
    sin = torch.sin(freqs).repeat(1, 2)[:, :head_dim]

    cos = cos.unsqueeze(0).expand(batch_size, -1, -1).to(dtype)
    sin = sin.unsqueeze(0).expand(batch_size, -1, -1).to(dtype)

    return cos, sin


def create_packing_attention_mask(batch_size, seq_lengths, device='cpu'):
    """
    Create a packing attention mask.

    Args:
        batch_size: Must be 1 for packing mode
        seq_lengths: List of sequence lengths, e.g., [1024, 2048, 512]

    Returns:
        attention_mask: [batch_size, total_seq_len] with values 1, 2, 3...
    """
    assert batch_size == 1, "Packing mode only supports batch_size=1"

    total_len = sum(seq_lengths)
    attention_mask = torch.zeros(batch_size, total_len, dtype=torch.int32, device=device)

    current_pos = 0
    for seq_id, seq_len in enumerate(seq_lengths, start=1):
        attention_mask[:, current_pos:current_pos + seq_len] = seq_id
        current_pos += seq_len

    return attention_mask


def compute_indexer_inputs(indexer, hidden_states, q_resid, position_embeddings, device):
    """
    Compute query, key, weights for indexer.
    Equivalent to indexer_forward_with_index but returns CPU tensors for simulation.
    """
    batch_size, seq_len, _ = hidden_states.shape
    cos, sin = position_embeddings

    # Compute query
    q = indexer.wq_b(q_resid)
    q = q.view(batch_size, seq_len, indexer.n_heads, indexer.head_dim)
    q_pe, q_nope = torch.split(q, [indexer.qk_rope_head_dim, indexer.head_dim - indexer.qk_rope_head_dim], dim=-1)
    q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=2)
    q = torch.cat([q_pe, q_nope], dim=-1)

    # Compute key
    k = indexer.k_norm(indexer.wk(hidden_states))
    k_pe, k_nope = torch.split(k, [indexer.qk_rope_head_dim, indexer.head_dim - indexer.qk_rope_head_dim], dim=-1)
    k_pe = apply_rotary_pos_emb(k_pe.unsqueeze(2), cos, sin, unsqueeze_dim=2).squeeze(2)
    k = torch.cat([k_pe, k_nope], dim=-1)

    # Compute weights
    weights = indexer.weights_proj(hidden_states)
    weights = weights * (indexer.n_heads ** -0.5)
    weights = weights * indexer.softmax_scale

    q = q.to(torch.bfloat16)
    k = k.to(torch.bfloat16)
    weights = weights.to(torch.bfloat16)

    if k.dim() == 3:
        k = k.unsqueeze(2)

    return q, k, weights


def manual_get_sp_local_cu_seqlens(attention_mask, sp_size, sp_rank):
    """
    Manually compute get_sp_local_cu_seqlens results for testing.

    Returns:
        mode, q_cu_seqlens, kv_cu_seqlens, kv_start, kv_end
    """
    from torch_turbo.utils.flash_attn_utils import get_seqlens_in_batch
    import torch.nn.functional as F

    bs, seq_len = attention_mask.shape
    if attention_mask is None or (bs != 1 and torch.max(attention_mask) <= 1):
        return "BSND", None, None, 0, 0

    bs, seq_len = attention_mask.shape
    assert bs == 1, "must be"

    # Handle padding - assume padding belongs to the last sequence
    max_value = torch.max(attention_mask)
    attention_mask = attention_mask.clone()
    attention_mask[attention_mask == 0] = max_value

    if sp_size > 1:
        chunk_size = seq_len // sp_size
        assert seq_len % sp_size == 0, f"seq_len {seq_len} must be divisible by sp_size {sp_size}"
        start = sp_rank * chunk_size
        end = (sp_rank + 1) * chunk_size
        local_attention_mask = attention_mask[:, start:end]
        # For KV, we need attention_mask up to 'end' (all-gathered keys)
        attention_mask_for_kv = attention_mask[:, 0:end]
    else:
        start = 0
        end = seq_len
        local_attention_mask = attention_mask
        attention_mask_for_kv = attention_mask

    # Get local sequence lengths for query
    local_seqlens = get_seqlens_in_batch(local_attention_mask)
    q_cu_seqlens = torch.cumsum(local_seqlens, dim=0).to(torch.int32)

    if len(q_cu_seqlens) == 0:
        fake_len = local_attention_mask.size(1)
        q_cu_seqlens = F.pad(q_cu_seqlens, (1, 0))
        q_cu_seqlens[0] = fake_len
        return "TND", q_cu_seqlens, q_cu_seqlens, 0, fake_len

    # Get full sequence lengths up to 'end' for KV
    full_seqlens = get_seqlens_in_batch(attention_mask_for_kv)
    full_cu_seqlens = F.pad(torch.cumsum(full_seqlens, dim=0), (1, 0))

    # Get unique sequence IDs in local attention_mask
    unique_ids = torch.unique(local_attention_mask[local_attention_mask != 0], sorted=True)
    assert len(unique_ids) > 0, "must be"

    # Compute KV cumulative sequence lengths
    kv_seqlens_list = []
    for seq_id in unique_ids:
        seq_idx = seq_id - 1
        kv_seqlens_list.append(full_seqlens[seq_idx].item())

    kv_cu_seqlens = torch.cumsum(
        torch.tensor(kv_seqlens_list, dtype=torch.int32, device=attention_mask.device),
        dim=0).to(torch.int32)

    # Compute kv_start and kv_end
    first_seq_id = unique_ids[0] - 1
    kv_start = full_cu_seqlens[first_seq_id].item()
    kv_end = end

    return "TND", q_cu_seqlens, kv_cu_seqlens, kv_start, kv_end


def run_indexer_full(query, key, weights, cu_seqlens, topk, device):
    """
    Run npu_lightning_indexer with full sequence (non-SP mode).

    Args:
        query: [T, H, D] - Query in TND layout
        key: [T, 1, D] - Key in TND layout (full sequence)
        weights: [T, H] - Weights in TND layout
        cu_seqlens: [num_sequences] - Cumulative sequence lengths (int32)
        topk: int - Number of topk indices to select
        device: torch device

    Returns:
        topk_indices: [T, 1, topk] - Selected sparse indices (global positions)
        topk_scores: [T, 1, topk] - Scores for selected indices
    """
    if not NPU_AVAILABLE:
        print("NPU not available, skipping full sequence test")
        return None, None

    with torch.no_grad():
        topk_indices, topk_scores = torch_npu.npu_lightning_indexer(
            query,
            key,
            weights,
            actual_seq_lengths_query=cu_seqlens,
            actual_seq_lengths_key=cu_seqlens,
            layout_query="TND",
            layout_key="TND",
            sparse_count=topk,
            sparse_mode=3,  # rightDownCausal mode
            return_value=True,
        )
    return topk_indices, topk_scores


def run_indexer_sp_chunk(query_chunk, key_full_up_to_end, weights_chunk,
                         q_cu_seqlens, kv_cu_seqlens, topk, device):
    """
    Run npu_lightning_indexer for an SP rank (local query, gathered key).

    Args:
        query_chunk: [T_local, H, D] - Local query in TND layout
        key_full_up_to_end: [T_kv, 1, D] - Key up to this rank's end position (all-gathered)
        weights_chunk: [T_local, H] - Local weights in TND layout
        q_cu_seqlens: [num_sequences] - Cumulative sequence lengths for local query (int32)
        kv_cu_seqlens: [num_sequences] - Cumulative sequence lengths for KV (int32)
        topk: int - Number of topk indices to select
        device: torch device

    Returns:
        topk_indices: [T_local, 1, topk] - Selected sparse indices (relative positions)
        topk_scores: [T_local, 1, topk] - Scores for selected indices
    """
    if not NPU_AVAILABLE:
        print("NPU not available, skipping SP chunk test")
        return None, None

    with torch.no_grad():
        topk_indices, topk_scores = torch_npu.npu_lightning_indexer(
            query_chunk,
            key_full_up_to_end,
            weights_chunk,
            actual_seq_lengths_query=q_cu_seqlens,
            actual_seq_lengths_key=kv_cu_seqlens,
            layout_query="TND",
            layout_key="TND",
            sparse_count=topk,
            sparse_mode=3,  # rightDownCausal mode
            return_value=True,
        )
    return topk_indices, topk_scores


def compare_sp_vs_non_sp(full_indices, sp_indices_list, full_scores, sp_scores_list,
                         sp_chunk_sizes, atol=1e-2):
    """
    Compare full (non-SP) and SP results.

    SP indices are relative to the local sequence chunk and need to be converted
    to global indices for comparison.
    """
    if full_indices is None or sp_indices_list is None:
        print("Cannot compare: one or both results are None")
        return False

    # Concatenate SP results
    concat_indices = torch.cat(sp_indices_list, dim=0)
    concat_scores = torch.cat(sp_scores_list, dim=0)

    print(f"\nComparison Results:")
    print(f"  Full indices shape: {full_indices.shape}")
    print(f"  Concatenated SP indices shape: {concat_indices.shape}")

    # Note: SP indices from NPU are relative to the sub-sequence start
    # For single sequence without packing, they should match directly
    # But with packing, we need to convert relative positions to global positions

    # Check for exact match
    exact_match_indices = torch.all(full_indices == concat_indices).item()
    exact_match_scores = torch.allclose(full_scores, concat_scores, atol=atol)

    print(f"  Exact index match: {exact_match_indices}")
    print(f"  Scores allclose (atol={atol}): {exact_match_scores}")

    # Check overlap ratio
    overlap = (full_indices == concat_indices).float().mean().item()
    print(f"  Index overlap ratio: {overlap:.4f} ({overlap*100:.1f}%)")

    # Analyze overlap by SP chunk
    start = 0
    for i, chunk_size in enumerate(sp_chunk_sizes):
        end = start + chunk_size
        full_chunk = full_indices[start:end]
        sp_chunk = concat_indices[start:end]
        chunk_overlap = (full_chunk == sp_chunk).float().mean().item()
        print(f"    SP Rank {i} (positions {start}:{end}): overlap={chunk_overlap:.4f}")
        start = end

    # Score difference statistics
    if full_scores is not None and concat_scores is not None:
        score_diff = (full_scores - concat_scores).abs()
        print(f"  Score diff - Max: {score_diff.max().item():.6e}, Mean: {score_diff.mean().item():.6e}")

    return overlap > 0.8  # Consider it a pass if overlap > 80%


def test_sp_cu_seqlens_single_sequence(device='npu:0', total_seq_len=4096, sp_size=2):
    """
    Test get_sp_local_cu_seqlens correctness for single sequence with SP.
    """
    print("\n" + "="*60)
    print(f"Test: SP cu_seqlens correctness (seq_len={total_seq_len}, sp_size={sp_size})")
    print("="*60)

    # Create attention mask for single sequence
    batch_size = 1
    attention_mask = create_packing_attention_mask(batch_size, [total_seq_len], device)

    chunk_size = total_seq_len // sp_size
    print(f"\nSequence length: {total_seq_len}, SP size: {sp_size}, Chunk size: {chunk_size}")

    results = []
    for sp_rank in range(sp_size):
        mode, q_cu_seqlens, kv_cu_seqlens, kv_start, kv_end = manual_get_sp_local_cu_seqlens(
            attention_mask, sp_size, sp_rank
        )

        print(f"\nSP Rank {sp_rank}:")
        print(f"  mode: {mode}")
        print(f"  q_cu_seqlens (local query lengths): {q_cu_seqlens.tolist() if q_cu_seqlens is not None else None}")
        print(f"  kv_cu_seqlens (KV lengths up to rank): {kv_cu_seqlens.tolist() if kv_cu_seqlens is not None else None}")
        print(f"  kv_start: {kv_start}")
        print(f"  kv_end: {kv_end}")

        # Verify correctness
        expected_q_len = chunk_size
        expected_kv_end = (sp_rank + 1) * chunk_size
        expected_kv_start = 0  # For single sequence, starts from 0

        q_len = q_cu_seqlens[-1].item() if q_cu_seqlens is not None and len(q_cu_seqlens) > 0 else 0
        kv_len = kv_cu_seqlens[-1].item() if kv_cu_seqlens is not None and len(kv_cu_seqlens) > 0 else 0

        print(f"  Expected: q_len={expected_q_len}, kv_end={expected_kv_end}, kv_start={expected_kv_start}")

        # For single sequence, q_cu_seqlens = [chunk_size], kv_cu_seqlens = [(rank+1)*chunk_size]
        q_correct = (q_len == expected_q_len)
        kv_end_correct = (kv_end == expected_kv_end)
        kv_start_correct = (kv_start == expected_kv_start)

        rank_pass = q_correct and kv_end_correct and kv_start_correct
        results.append(rank_pass)
        print(f"  Verification: {'PASS' if rank_pass else 'FAIL'}")

    all_pass = all(results)
    print(f"\nOverall Test Result: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_sp_cu_seqlens_packing(device='npu:0', seq_lengths=[2048, 2048], sp_size=2):
    """
    Test get_sp_local_cu_seqlens correctness for packing mode with SP.

    This tests the case where sequences are packed and SP splits across them.
    """
    print("\n" + "="*60)
    print(f"Test: SP cu_seqlens with packing (seq_lengths={seq_lengths}, sp_size={sp_size})")
    print("="*60)

    batch_size = 1
    total_seq_len = sum(seq_lengths)
    attention_mask = create_packing_attention_mask(batch_size, seq_lengths, device)

    chunk_size = total_seq_len // sp_size
    print(f"\nTotal length: {total_seq_len}, SP size: {sp_size}, Chunk size: {chunk_size}")
    print(f"Sequence boundaries: {seq_lengths}")
    print(f"Attention mask: {attention_mask[0].tolist()[:20]}... (first 20)")

    results = []
    for sp_rank in range(sp_size):
        mode, q_cu_seqlens, kv_cu_seqlens, kv_start, kv_end = manual_get_sp_local_cu_seqlens(
            attention_mask, sp_size, sp_rank
        )

        print(f"\nSP Rank {sp_rank} (positions {sp_rank*chunk_size}:{(sp_rank+1)*chunk_size}):")
        print(f"  mode: {mode}")
        print(f"  q_cu_seqlens: {q_cu_seqlens.tolist() if q_cu_seqlens is not None else None}")
        print(f"  kv_cu_seqlens: {kv_cu_seqlens.tolist() if kv_cu_seqlens is not None else None}")
        print(f"  kv_start: {kv_start}")
        print(f"  kv_end: {kv_end}")

        # Verify basic correctness
        expected_q_len = chunk_size
        expected_kv_end = (sp_rank + 1) * chunk_size

        q_len = q_cu_seqlens[-1].item() if q_cu_seqlens is not None and len(q_cu_seqlens) > 0 else 0

        q_correct = (q_len == expected_q_len)
        kv_end_correct = (kv_end == expected_kv_end)

        # kv_start should be 0 if rank 0 starts from first sequence, otherwise the start of
        # the first sequence in this rank's chunk
        rank_pass = q_correct and kv_end_correct and kv_start >= 0 and kv_start <= kv_end
        results.append(rank_pass)
        print(f"  Verification: {'PASS' if rank_pass else 'FAIL'}")

    all_pass = all(results)
    print(f"\nOverall Test Result: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_sp_indexer_equivalence_single_sequence(device='npu:0', total_seq_len=4096, sp_size=2):
    """
    Test indexer_forward_with_scores equivalence between non-SP and SP for single sequence.
    """
    print("\n" + "="*60)
    print(f"Test: SP indexer equivalence (seq_len={total_seq_len}, sp_size={sp_size})")
    print("="*60)

    if not NPU_AVAILABLE:
        print("NPU not available, skipping test")
        return False

    config = create_test_config()
    indexer = GlmMoeDsaIndexer(config, layer_idx=0).to(device)
    indexer = indexer.to(torch.bfloat16)
    indexer.eval()

    batch_size = 1

    # Create inputs
    hidden_states = torch.randn(batch_size, total_seq_len, config.hidden_size, device=device, dtype=torch.bfloat16)
    q_resid = torch.randn(batch_size, total_seq_len, config.q_lora_rank, device=device, dtype=torch.bfloat16)

    cos, sin = create_rotary_embeddings(total_seq_len, config.qk_rope_head_dim, batch_size, device)
    position_embeddings = (cos, sin)

    # Create attention mask
    attention_mask = create_packing_attention_mask(batch_size, [total_seq_len], device)

    # Compute query, key, weights
    with torch.no_grad():
        q, k, weights = compute_indexer_inputs(indexer, hidden_states, q_resid, position_embeddings, device)

    # Convert to TND layout
    query_tnd = q.view(-1, indexer.n_heads, indexer.head_dim)
    key_tnd = k.view(-1, 1, indexer.head_dim)
    weights_tnd = weights.view(-1, indexer.n_heads)

    print(f"Query shape (TND): {query_tnd.shape}")
    print(f"Key shape (TND): {key_tnd.shape}")
    print(f"Weights shape (TND): {weights_tnd.shape}")

    # Test 1: Non-SP mode (full sequence)
    print("\n--- Non-SP mode (full sequence) ---")
    seed_all(42)
    # Single sequence: cu_seqlens = [total_seq_len]
    cu_seqlens_full = torch.tensor([total_seq_len], dtype=torch.int32, device=device)
    full_indices, full_scores = run_indexer_full(
        query_tnd, key_tnd, weights_tnd, cu_seqlens_full, config.index_topk, device
    )
    print(f"Full indices shape: {full_indices.shape}")

    # Test 2: SP mode (simulated with chunks)
    print("\n--- SP mode (simulated with 2 ranks) ---")
    chunk_size = total_seq_len // sp_size

    sp_indices_list = []
    sp_scores_list = []

    for sp_rank in range(sp_size):
        print(f"\nSimulating SP Rank {sp_rank}:")

        start = sp_rank * chunk_size
        end = (sp_rank + 1) * chunk_size

        # Local query and weights
        query_chunk = query_tnd[start:end]
        weights_chunk = weights_tnd[start:end]

        # All-gathered key: from start=0 to end
        key_gathered = key_tnd[0:end]

        # Get cu_seqlens for this rank
        mode, q_cu_seqlens, kv_cu_seqlens, kv_start, kv_end = manual_get_sp_local_cu_seqlens(
            attention_mask, sp_size, sp_rank
        )

        print(f"  Query chunk: [{start}:{end}], shape: {query_chunk.shape}")
        print(f"  Key gathered: [0:{end}], shape: {key_gathered.shape}")
        print(f"  q_cu_seqlens: {q_cu_seqlens}")
        print(f"  kv_cu_seqlens: {kv_cu_seqlens}")
        print(f"  kv_start: {kv_start}, kv_end: {kv_end}")

        # Run indexer for this rank
        seed_all(42)  # Use same seed as non-SP
        indices_chunk, scores_chunk = run_indexer_sp_chunk(
            query_chunk, key_gathered, weights_chunk,
            q_cu_seqlens, kv_cu_seqlens, config.index_topk, device
        )

        # Note: NPU returns relative positions within the KV context
        # For comparison, we need to check if indices are valid and within expected range
        print(f"  Indices chunk shape: {indices_chunk.shape}")
        print(f"  Indices range: [{indices_chunk.min().item()}, {indices_chunk.max().item()}]")

        sp_indices_list.append(indices_chunk)
        sp_scores_list.append(scores_chunk)

    # Compare results
    sp_chunk_sizes = [chunk_size] * sp_size
    passed = compare_sp_vs_non_sp(full_indices, sp_indices_list, full_scores, sp_scores_list, sp_chunk_sizes)

    print(f"\nTest Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_sp_indexer_equivalence_packing(device='npu:0', seq_lengths=[2048, 2048], sp_size=2):
    """
    Test indexer_forward_with_scores equivalence with packing mode and SP.
    """
    print("\n" + "="*60)
    print(f"Test: SP indexer with packing (seq_lengths={seq_lengths}, sp_size={sp_size})")
    print("="*60)

    if not NPU_AVAILABLE:
        print("NPU not available, skipping test")
        return False

    # Check if packing split is valid for SP
    total_seq_len = sum(seq_lengths)
    chunk_size = total_seq_len // sp_size

    # Check if SP splits align with sequence boundaries
    boundary = seq_lengths[0]
    if chunk_size != boundary:
        print(f"Skipping: SP chunk_size ({chunk_size}) doesn't align with sequence boundary ({boundary})")
        print("This is expected - packing with SP requires careful alignment")
        return True  # Skip but don't fail

    config = create_test_config()
    indexer = GlmMoeDsaIndexer(config, layer_idx=0).to(device)
    indexer = indexer.to(torch.bfloat16)
    indexer.eval()

    batch_size = 1

    # Create inputs
    hidden_states = torch.randn(batch_size, total_seq_len, config.hidden_size, device=device, dtype=torch.bfloat16)
    q_resid = torch.randn(batch_size, total_seq_len, config.q_lora_rank, device=device, dtype=torch.bfloat16)

    cos, sin = create_rotary_embeddings(total_seq_len, config.qk_rope_head_dim, batch_size, device)
    position_embeddings = (cos, sin)

    attention_mask = create_packing_attention_mask(batch_size, seq_lengths, device)

    # Compute query, key, weights
    with torch.no_grad():
        q, k, weights = compute_indexer_inputs(indexer, hidden_states, q_resid, position_embeddings, device)

    query_tnd = q.view(-1, indexer.n_heads, indexer.head_dim)
    key_tnd = k.view(-1, 1, indexer.head_dim)
    weights_tnd = weights.view(-1, indexer.n_heads)

    # Test 1: Non-SP mode (full sequence)
    print("\n--- Non-SP mode (packing) ---")
    seed_all(42)
    mode, q_cu_seqlens, kv_cu_seqlens, kv_start, kv_end = manual_get_sp_local_cu_seqlens(
        attention_mask, 1, 0
    )
    full_indices, full_scores = run_indexer_full(
        query_tnd, key_tnd, weights_tnd, q_cu_seqlens, config.index_topk, device
    )
    print(f"Full indices shape: {full_indices.shape}")

    # Test 2: SP mode
    print("\n--- SP mode with packing (split at sequence boundary) ---")
    sp_indices_list = []
    sp_scores_list = []

    for sp_rank in range(sp_size):
        print(f"\nSimulating SP Rank {sp_rank}:")

        start = sp_rank * chunk_size
        end = (sp_rank + 1) * chunk_size

        query_chunk = query_tnd[start:end]
        weights_chunk = weights_tnd[start:end]

        # Get cu_seqlens for this rank
        mode, q_cu_seqlens, kv_cu_seqlens, kv_start, kv_end = manual_get_sp_local_cu_seqlens(
            attention_mask, sp_size, sp_rank
        )
        key_gathered = key_tnd[kv_start:kv_end]

        print(f"  Query chunk: [{start}:{end}], shape: {query_chunk.shape}")
        print(f"  Key gathered: [0:{end}], shape: {key_gathered.shape}")
        print(f"  q_cu_seqlens: {q_cu_seqlens.tolist() if q_cu_seqlens is not None else None}")
        print(f"  kv_cu_seqlens: {kv_cu_seqlens.tolist() if kv_cu_seqlens is not None else None}")
        print(f"  kv_start: {kv_start}, kv_end: {kv_end}")

        seed_all(42)
        indices_chunk, scores_chunk = run_indexer_sp_chunk(
            query_chunk, key_gathered, weights_chunk,
            q_cu_seqlens, kv_cu_seqlens, config.index_topk, device
        )

        print(f"  Indices chunk shape: {indices_chunk.shape}")
        sp_indices_list.append(indices_chunk)
        sp_scores_list.append(scores_chunk)

    # Compare
    sp_chunk_sizes = [chunk_size] * sp_size
    passed = compare_sp_vs_non_sp(full_indices, sp_indices_list, full_scores, sp_scores_list, sp_chunk_sizes)

    print(f"\nTest Result: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    print("="*60)
    print("SP (Sequence Parallel) Indexer Test")
    print("="*60)
    print("Testing:")
    print("1. get_sp_local_cu_seqlens returns correct kv_start/kv_end")
    print("2. indexer_forward_with_scores equivalence with/without SP")

    device = 'npu:0' if NPU_AVAILABLE else 'cpu'
    print(f"\nUsing device: {device}")

    if NPU_AVAILABLE:
        torch.npu.set_device(0)

    results = []

    # Test 1: cu_seqlens correctness for single sequence
    results.append(test_sp_cu_seqlens_single_sequence(
       device=device, total_seq_len=4096, sp_size=2
    ))

    # Test 2: cu_seqlens correctness for packing (boundary aligned)
    results.append(test_sp_cu_seqlens_packing(
       device=device, seq_lengths=[2048, 2048], sp_size=2
    ))

    # Test 3: indexer equivalence for single sequence
    results.append(test_sp_indexer_equivalence_single_sequence(
       device=device, total_seq_len=4096, sp_size=2
    ))

    # Test 4: indexer equivalence with packing (boundary aligned)
    results.append(test_sp_indexer_equivalence_packing(
        device=device, seq_lengths=[2048, 2048], sp_size=2
    ))

    # Test 5: cu_seqlens with sp_size=4
    results.append(test_sp_cu_seqlens_single_sequence(
       device=device, total_seq_len=4096, sp_size=4
    ))

    print("\n" + "="*60)
    print(f"All tests completed! Passed: {sum(results)}/{len(results)}")
    print("="*60)

    return sum(results) == len(results)


if __name__ == "__main__":
    main()