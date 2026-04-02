"""
Test equivalence between dsa_attn_forward and npu_fused_dsa_attn_forward.

This script compares the two implementations:
- dsa_attn_forward: Naive implementation using eager attention + torch indexer
- npu_fused_dsa_attn_forward: NPU optimized using MLA absorb + sparse flash attention

Run with: python test_dsa_attn_equiv.py
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
    GlmMoeDsaAttention,
    GlmMoeDsaIndexer,
    apply_rotary_pos_emb,
)
from torch_turbo.models.glm_moe_dsa.npu import (
    dsa_attn_forward,
    npu_fused_dsa_attn_forward,
    indexer_forward_torch,
    indexer_forward_npu,
    fix_invalid_topk_indices,
    fix_npu_topk_indices,
    create_index_mask,
)
from torch_turbo.utils.flash_attn_utils import get_seqlens_in_batch
from torch_turbo.utils.npu_utils import seed_all


def create_test_config(
    hidden_size=4096,
    num_attention_heads=64,  # NPU requires 64 or 128 for kl_loss
    q_lora_rank=512,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    index_n_heads=8,
    index_head_dim=128,
    index_topk=2048,  # NPU requires topK=2048 for kl_loss
):
    """Create a test config with dimensions that satisfy NPU API constraints.

    NPU API Constraints (npu_sparse_lightning_indexer_grad_kl_loss):
        - query shape N (num_attention_heads) must be 64 or 128
        - topK (index_topk) must be 2048
    """
    config = GlmMoeDsaConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_attention_heads,  # GQA = 1 for simplicity
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

    # Create cos and sin with full head_dim
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


def run_dsa_attn_forward(module, hidden_states, position_embeddings, attention_mask, skip_indexer=False):
    """Run the naive dsa_attn_forward implementation."""
    with torch.no_grad():
        seed_all()
        output, _, _ = dsa_attn_forward(
            module,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_value=None,
            use_cache=False,
            output_attentions=False,
            skip_indexer=skip_indexer
        )
    return output


def run_npu_fused_dsa_attn_forward(module, hidden_states, position_embeddings, attention_mask, skip_indexer=False):
    """Run the NPU optimized npu_fused_dsa_attn_forward implementation."""
    with torch.no_grad():
        seed_all()
        output, _, _ = npu_fused_dsa_attn_forward(
            module,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_value=None,
            use_cache=False,
            output_attentions=False,
            indexer_loss_coeff=1.0,
            use_fused_loss=True,  # Enable fused loss for equivalence test
            skip_indexer=skip_indexer
        )
    return output


def compare_outputs(output_torch, output_npu, name="test", rtol=1e-2, atol=1e-2):
    """Compare two outputs and print results."""
    if output_torch is None or output_npu is None:
        print(f"[{name}] One or both outputs are None")
        return False

    if output_torch.shape != output_npu.shape:
        print(f"[{name}] Shape mismatch: torch={output_torch.shape}, npu={output_npu.shape}")
        return False

    # Check for NaN or Inf
    if torch.isnan(output_torch).any() or torch.isnan(output_npu).any():
        print(f"[{name}] NaN detected in outputs")
        return False
    if torch.isinf(output_torch).any() or torch.isinf(output_npu).any():
        print(f"[{name}] Inf detected in outputs")
        return False

    # Compute relative and absolute differences
    diff = (output_torch - output_npu).abs()
    rel_diff = diff / (output_torch.abs() + 1e-8)

    max_abs_diff = diff.max().item()
    max_rel_diff = rel_diff.max().item()
    mean_abs_diff = diff.mean().item()
    mean_rel_diff = rel_diff.mean().item()

    # Check if close
    is_close = torch.allclose(output_torch, output_npu, rtol=rtol, atol=atol)

    print(f"\n[{name}] Comparison Results:")
    print(f"  Shape: {output_torch.shape}")
    print(f"  Max absolute diff: {max_abs_diff:.6e}")
    print(f"  Mean absolute diff: {mean_abs_diff:.6e}")
    print(f"  Max relative diff: {max_rel_diff:.6e}")
    print(f"  Mean relative diff: {mean_rel_diff:.6e}")
    print(f"  Within tolerance (rtol={rtol}, atol={atol}): {is_close}")

    return is_close


def test_single_sequence(device='cpu', seq_len=2048):
    """Test with a single sequence (no packing).

    Note: seq_len should be >= index_topk (2048) for valid sparse attention.
    """
    print("\n" + "="*60)
    print(f"Test: Single Sequence (seq_len={seq_len})")
    print("="*60)

    # Use NPU-compatible config: num_heads=64, index_topk=2048
    config = create_test_config()
    module = GlmMoeDsaAttention(config, layer_idx=0).to(device)
    module = module.to(torch.bfloat16)
    module.eval()

    batch_size = 1
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device=device, dtype=torch.bfloat16)

    cos, sin = create_rotary_embeddings(seq_len, config.qk_rope_head_dim, batch_size, device)
    position_embeddings = (cos, sin)

    attention_mask = create_packing_attention_mask(batch_size, [seq_len], device)

    print("\n" + "="*60)
    print(f"Compare mla vs. mla_absorb (skip_indexer=True)")
    print("="*60)

    output_torch = run_dsa_attn_forward(module, hidden_states, position_embeddings, attention_mask, True)
    output_npu = run_npu_fused_dsa_attn_forward(module, hidden_states, position_embeddings, attention_mask, True)

    compare_outputs(output_torch, output_npu, "single_sequence mla vs. mla_absorb")

    print("\n" + "="*60)
    print(f"Compare mla vs. mla_absorb_sparse_attn (with indexer)")
    print("="*60)

    output_torch = run_dsa_attn_forward(module, hidden_states, position_embeddings, attention_mask)
    output_npu = run_npu_fused_dsa_attn_forward(module, hidden_states, position_embeddings, attention_mask)

    compare_outputs(output_torch, output_npu, "single_sequence mla vs. mla_absorb_sparse_attn")


def test_packing_mode(device='cpu', seq_lengths=[2048, 2048]):
    """Test with packing mode (multiple sequences packed together).

    Note: Total seq_len should be >= index_topk (2048) for valid sparse attention.
    NPU requires num_heads=64 or 128, index_topk=2048 for kl_loss.
    """
    print("\n" + "="*60)
    print(f"Test: Packing Mode (seq_lengths={seq_lengths})")
    print("="*60)

    # Use NPU-compatible config: num_heads=64, index_topk=2048
    config = create_test_config()
    module = GlmMoeDsaAttention(config, layer_idx=0).to(device)
    module = module.to(torch.bfloat16)
    module.eval()

    batch_size = 1
    total_seq_len = sum(seq_lengths)

    hidden_states = torch.randn(batch_size, total_seq_len, config.hidden_size, device=device, dtype=torch.bfloat16)

    cos, sin = create_rotary_embeddings(total_seq_len, config.qk_rope_head_dim, batch_size, device)
    position_embeddings = (cos, sin)

    attention_mask = create_packing_attention_mask(batch_size, seq_lengths, device)

    output_torch = run_dsa_attn_forward(module, hidden_states, position_embeddings, attention_mask)
    output_npu = run_npu_fused_dsa_attn_forward(module, hidden_states, position_embeddings, attention_mask)

    compare_outputs(output_torch, output_npu, "packing_mode")


def test_indexer_equivalence(device='cpu', seq_lengths=[2048, 2048]):
    """Test indexer equivalence separately.

    Note: This tests indexer only (not kl_loss), so constraints may be different.
    But using consistent config for simplicity.
    """
    print("\n" + "="*60)
    print(f"Test: Indexer Equivalence (seq_lengths={seq_lengths})")
    print("="*60)

    # Use NPU-compatible config for consistency
    config = create_test_config()
    indexer = GlmMoeDsaIndexer(config, layer_idx=0).to(device)
    indexer = indexer.to(torch.bfloat16)
    indexer.eval()

    batch_size = 1
    total_seq_len = sum(seq_lengths)

    hidden_states = torch.randn(batch_size, total_seq_len, config.hidden_size, device=device, dtype=torch.bfloat16)
    q_resid = torch.randn(batch_size, total_seq_len, config.q_lora_rank, device=device, dtype=torch.bfloat16)

    cos, sin = create_rotary_embeddings(total_seq_len, config.qk_rope_head_dim, batch_size, device)
    position_embeddings = (cos, sin)

    attention_mask = create_packing_attention_mask(batch_size, seq_lengths, device)

    # Torch indexer
    with torch.no_grad():
        indices_torch = indexer_forward_torch(
            indexer, hidden_states, q_resid, position_embeddings, attention_mask, use_cache=False
        )
        indices_torch = fix_invalid_topk_indices(indices_torch, attention_mask)
        torch_mask = create_index_mask(indices_torch)

    # NPU indexer (if available)
    if NPU_AVAILABLE:
        with torch.no_grad():
            indices_npu = indexer_forward_npu(
                indexer, hidden_states, q_resid, position_embeddings, attention_mask, use_cache=False
            )
            indices_npu = fix_npu_topk_indices(indices_npu, attention_mask)
            npu_mask = create_index_mask(indices_npu)

        # Compare index masks instead of raw indices (indices may differ but masks should be similar)
        mask_match = (torch_mask == npu_mask).float().mean().item()
        print(f"Index mask match ratio: {mask_match:.4f} ({mask_match*100:.1f}%)")
    else:
        print("Skipping NPU indexer test (torch_npu not available)")


def test_backward_equivalence(device='cpu', seq_len=2048):
    """Test backward pass equivalence between dsa_attn_forward and npu_fused_dsa_attn_forward.

    This tests gradient computation for the main attention path (excluding indexer)
    by using skip_indexer=True.

    Compares:
    - Gradient of hidden_states
    - Gradients of attention parameters (q_proj, kv_a_proj, kv_b_proj, o_proj, etc.)
    - Excludes indexer parameters since backward implementations differ
    """
    print("\n" + "="*60)
    print(f"Test: Backward Equivalence (seq_len={seq_len}, skip_indexer=True)")
    print("="*60)

    # Create two identical models
    config = create_test_config()

    # Model 1: naive implementation
    model_torch = GlmMoeDsaAttention(config, layer_idx=0).to(device)
    model_torch = model_torch.to(torch.bfloat16)

    # Model 2: NPU optimized implementation
    model_npu = GlmMoeDsaAttention(config, layer_idx=0).to(device)
    model_npu = model_npu.to(torch.bfloat16)

    # Copy weights to ensure identical models
    model_npu.load_state_dict(model_torch.state_dict())

    # Set to training mode for gradient computation
    model_torch.train()
    model_npu.train()

    batch_size = 1

    # Create identical inputs (requires_grad for input gradient comparison)
    hidden_states_torch = torch.randn(
        batch_size, seq_len, config.hidden_size,
        device=device, dtype=torch.bfloat16, requires_grad=True
    )
    hidden_states_npu = hidden_states_torch.clone().detach().requires_grad_(True)

    cos, sin = create_rotary_embeddings(seq_len, config.qk_rope_head_dim, batch_size, device)
    position_embeddings = (cos, sin)

    attention_mask = create_packing_attention_mask(batch_size, [seq_len], device)

    # Create identical loss targets
    loss_target = torch.randn(batch_size, seq_len, config.hidden_size, device=device, dtype=torch.bfloat16)

    # =========================================================================
    # Forward + Backward: Naive implementation
    # =========================================================================
    seed_all()
    output_torch, _, _ = dsa_attn_forward(
        model_torch,
        hidden_states_torch,
        position_embeddings,
        attention_mask,
    )

    loss_torch = ((output_torch - loss_target) ** 2).mean()
    loss_torch.backward()

    # =========================================================================
    # Forward + Backward: NPU optimized implementation
    # =========================================================================
    seed_all()
    output_npu, _, _ = npu_fused_dsa_attn_forward(
        model_npu,
        hidden_states_npu,
        position_embeddings,
        attention_mask,
    )

    loss_npu = ((output_npu - loss_target) ** 2).mean()
    loss_npu.backward()

    # =========================================================================
    # Forward output comparison
    # =========================================================================
    print("\n--- Forward Output Comparison ---")
    output_diff = (output_torch - output_npu).abs()
    max_diff = output_diff.max().item()
    mean_diff = output_diff.mean().item()
    print(f"Max absolute diff: {max_diff:.6e}")
    print(f"Mean absolute diff: {mean_diff:.6e}")
    forward_match = torch.allclose(output_torch, output_npu, rtol=1e-2, atol=1e-2)
    print(f"Forward outputs match (rtol=1e-2, atol=1e-2): {forward_match}")

    # =========================================================================
    # Compare gradients
    # =========================================================================
    print("\n--- Gradient Comparison ---")

    # 1. Compare hidden_states gradients
    grad_hidden_torch = hidden_states_torch.grad
    grad_hidden_npu = hidden_states_npu.grad

    if grad_hidden_torch is not None and grad_hidden_npu is not None:
        hidden_grad_diff = (grad_hidden_torch - grad_hidden_npu).abs()
        max_diff = hidden_grad_diff.max().item()
        mean_diff = hidden_grad_diff.mean().item()
        rel_diff = (hidden_grad_diff / (grad_hidden_torch.abs() + 1e-8)).max().item()

        print(f"\nhidden_states.grad comparison:")
        print(f"  Max absolute diff: {max_diff:.6e}")
        print(f"  Mean absolute diff: {mean_diff:.6e}")
        print(f"  Max relative diff: {rel_diff:.6e}")

        is_close = torch.allclose(grad_hidden_torch, grad_hidden_npu, rtol=1e-2, atol=1e-2)
        print(f"  Within tolerance (rtol=1e-2, atol=1e-2): {is_close}")
    else:
        print("\nhidden_states.grad: One or both are None")
        is_close = False

    # 2. Compare parameter gradients (exclude indexer)
    # Indexer parameter names typically contain: 'indexer'
    param_results = {}

    print("\nParameter gradient comparison (excluding indexer):")
    for name, param_torch in model_torch.named_parameters():
        if 'indexer' in name:
            continue  # Skip indexer parameters

        param_npu = dict(model_npu.named_parameters()).get(name)
        if param_npu is None:
            print(f"  {name}: Not found in model_npu")
            continue

        if param_torch.grad is None and param_npu.grad is None:
            print(f"  {name}: Both gradients are None (OK)")
            continue

        if param_torch.grad is None or param_npu.grad is None:
            print(f"  {name}: One gradient is None (MISMATCH)")
            param_results[name] = False
            continue

        grad_diff = (param_torch.grad - param_npu.grad).abs()
        max_diff = grad_diff.max().item()
        mean_diff = grad_diff.mean().item()

        # Check if gradients are close
        is_param_close = torch.allclose(param_torch.grad, param_npu.grad, rtol=1e-2, atol=1e-2)
        param_results[name] = is_param_close

        status = "✓" if is_param_close else "✗"
        print(f"  {status} {name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")

    # Summary
    all_passed = forward_match and is_close and all(param_results.values())
    print("\n--- Summary ---")
    print(f"Forward output: {'PASS' if forward_match else 'FAIL'}")
    print(f"hidden_states gradient: {'PASS' if is_close else 'FAIL'}")
    print(f"Parameter gradients: {'PASS' if all(param_results.values()) else 'FAIL'}")
    print(f"Overall: {'PASS' if all_passed else 'FAIL'}")

    return all_passed


def main():
    print("="*60)
    print("DSA Attention Equivalence Test")
    print("="*60)
    print("NPU Constraints: num_heads=64 or 128, index_topk=2048")

    device = 'npu:0' if NPU_AVAILABLE else 'cpu'
    print(f"Using device: {device}")

    if NPU_AVAILABLE:
        torch.npu.set_device(0)
        seed_all()
    else:
        torch.manual_seed(42)

    if NPU_AVAILABLE:
        test_indexer_equivalence(device=device, seq_lengths=[2048, 4096, 2048])
        test_backward_equivalence(device=device, seq_len=2048)

    # Run tests with seq_len >= index_topk (2048)
    test_single_sequence(device=device, seq_len=2048)
    test_packing_mode(device=device, seq_lengths=[2048, 4096, 2048])

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()