"""
Test equivalence between dsa_attn_forward and npu_fused_dsa_attn_forward.

This test compares:
- dsa_attn_forward: Naive MLA implementation using eager attention + torch indexer
- npu_fused_dsa_attn_forward: NPU optimized implementation using MLA absorb + sparse flash attention

Run with: pytest tests/glm_moe_dsa/test_dsa_attn_equiv.py -v
"""

import pytest
import torch

# Check if torch_npu is available
try:
    import torch_npu
    NPU_AVAILABLE = True
except ImportError:
    NPU_AVAILABLE = False

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
        attention_mask: [batch_size, total_seq_len] with values 1, 2, 3... for different sequences
    """
    assert batch_size == 1, "Packing mode only supports batch_size=1"

    total_len = sum(seq_lengths)
    attention_mask = torch.zeros(batch_size, total_len, dtype=torch.int32, device=device)

    current_pos = 0
    for seq_id, seq_len in enumerate(seq_lengths, start=1):
        attention_mask[:, current_pos:current_pos + seq_len] = seq_id
        current_pos += seq_len

    return attention_mask


@pytest.mark.skipif(not NPU_AVAILABLE, reason="torch_npu not available")
class TestDsaAttentionEquivalence:
    """Test equivalence between dsa_attn_forward and npu_fused_dsa_attn_forward."""

    @pytest.fixture
    def device(self):
        return 'npu:0'

    @pytest.fixture
    def seed(self):
        return 42

    @pytest.fixture
    def model(self, device):
        """Create a model instance for testing with NPU-compatible config."""
        config = create_test_config()  # Uses NPU-compatible defaults: num_heads=64, topk=2048
        module = GlmMoeDsaAttention(config, layer_idx=0).to(device)
        module = module.to(torch.bfloat16)
        module.eval()
        return module

    def _create_inputs(self, model, seq_len, device):
        """Create random inputs for testing."""
        batch_size = 1
        hidden_states = torch.randn(
            batch_size, seq_len, model.config.hidden_size,
            device=device, dtype=torch.bfloat16
        )

        cos, sin = create_rotary_embeddings(
            seq_len, model.config.qk_rope_head_dim,
            batch_size, device
        )
        position_embeddings = (cos, sin)

        # Create packing attention mask (single sequence)
        attention_mask = create_packing_attention_mask(batch_size, [seq_len], device)

        return hidden_states, position_embeddings, attention_mask

    def test_mla_vs_mla_absorb_single_sequence(self, model, device, seed):
        """Test MLA naive vs MLA absorb with single sequence.

        Note: seq_len should be >= index_topk (2048) for valid sparse attention.
        """
        torch.manual_seed(seed)
        seed_all()

        seq_len = 2048  # Must be >= index_topk (2048)
        hidden_states, position_embeddings, attention_mask = self._create_inputs(
            model, seq_len, device
        )

        # Run both implementations
        output_naive, _, _ = dsa_attn_forward(
            model,
            hidden_states,
            position_embeddings,
            attention_mask,
            skip_indexer=True,  # Skip indexer for pure MLA comparison
        )

        output_absorb, _, _ = npu_fused_dsa_attn_forward(
            model,
            hidden_states,
            position_embeddings,
            attention_mask,
            indexer_loss_coeff=1.0,
            use_fused_loss=False,
            skip_indexer=True,  # Skip indexer for pure MLA absorb comparison
        )

        # Compare outputs
        assert output_naive.shape == output_absorb.shape, \
            f"Shape mismatch: naive={output_naive.shape}, absorb={output_absorb.shape}"

        # Allow some numerical tolerance
        assert torch.allclose(output_naive, output_absorb, rtol=1e-2, atol=1e-2), \
            f"Output mismatch: max diff={(output_naive - output_absorb).abs().max()}"

    def test_mla_vs_mla_absorb_with_indexer(self, model, device, seed):
        """Test MLA naive vs MLA absorb with indexer.

        Note: seq_len should be >= index_topk (2048) for valid sparse attention.
        """
        torch.manual_seed(seed)
        seed_all()

        seq_len = 2048  # Must be >= index_topk (2048)
        hidden_states, position_embeddings, attention_mask = self._create_inputs(
            model, seq_len, device
        )

        # Run both implementations with indexer
        output_naive, _, _  = dsa_attn_forward(
            model,
            hidden_states,
            position_embeddings,
            attention_mask,
            skip_indexer=False,  # Enable indexer
        )

        output_absorb, _, _ = npu_fused_dsa_attn_forward(
            model,
            hidden_states,
            position_embeddings,
            attention_mask,
            indexer_loss_coeff=1.0,
            use_fused_loss=True,
            skip_indexer=False,  # Enable indexer
        )

        # Compare outputs
        assert output_naive.shape == output_absorb.shape, \
            f"Shape mismatch: naive={output_naive.shape}, absorb={output_absorb.shape}"

        # Allow some numerical tolerance (with indexer, match may be different due to sparse selection)
        diff = (output_naive - output_absorb).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"Max diff: {max_diff}, Mean diff: {mean_diff}")

        # Relaxed tolerance with indexer
        assert torch.allclose(output_naive, output_absorb, rtol=0.05, atol=0.01), \
            f"Output mismatch with indexer: max diff={max_diff}"


@pytest.mark.skipif(not NPU_AVAILABLE, reason="torch_npu not available")
class TestPackingMode:
    """Test attention implementations with packing mode (multiple sequences packed together).

    NPU Constraints: num_heads=64 or 128, index_topk=2048
    """

    @pytest.fixture
    def device(self):
        return 'npu:0'

    @pytest.fixture
    def model(self, device):
        """Create a model instance for testing with NPU-compatible config."""
        config = create_test_config()  # Uses NPU-compatible defaults: num_heads=64, topk=2048
        module = GlmMoeDsaAttention(config, layer_idx=0).to(device)
        module = module.to(torch.bfloat16)
        module.eval()
        return module

    def test_packing_single_sequence(self, model, device):
        """Test packing mode with single sequence (should match non-packing).

        Note: seq_len should be >= index_topk (2048) for valid sparse attention.
        """
        seq_len = 2048  # Must be >= index_topk (2048)
        batch_size = 1

        hidden_states = torch.randn(
            batch_size, seq_len, model.config.hidden_size,
            device=device, dtype=torch.bfloat16
        )

        cos, sin = create_rotary_embeddings(
            seq_len, model.config.qk_rope_head_dim,
            batch_size, device
        )
        position_embeddings = (cos, sin)

        # Create packing mask with single sequence
        attention_mask = create_packing_attention_mask(batch_size, [seq_len], device)

        output_naive, _, _  = dsa_attn_forward(
            model,
            hidden_states,
            position_embeddings,
            attention_mask,
            skip_indexer=False,
        )

        output_absorb, _, _  = npu_fused_dsa_attn_forward(
            model,
            hidden_states,
            position_embeddings,
            attention_mask,
            indexer_loss_coeff=1.0,
            use_fused_loss=True,
            skip_indexer=False,
        )

        assert torch.allclose(output_naive, output_absorb, rtol=1e-2, atol=1e-2), \
            f"Single sequence packing mismatch: max diff={(output_naive - output_absorb).abs().max()}"

    def test_packing_multiple_sequences(self, model, device):
        """Test packing mode with multiple sequences.

        Note: total_seq_len should be >= index_topk (2048) for valid sparse attention.
        """
        seq_lengths = [2048, 2048]  # Total = 4096, each >= index_topk
        batch_size = 1
        total_seq_len = sum(seq_lengths)

        hidden_states = torch.randn(
            batch_size, total_seq_len, model.config.hidden_size,
            device=device, dtype=torch.bfloat16
        )

        cos, sin = create_rotary_embeddings(
            total_seq_len, model.config.qk_rope_head_dim,
            batch_size, device
        )
        position_embeddings = (cos, sin)

        # Create packing mask
        attention_mask = create_packing_attention_mask(batch_size, seq_lengths, device)

        output_naive, _, _  = dsa_attn_forward(
            model,
            hidden_states,
            position_embeddings,
            attention_mask,
            skip_indexer=False,
        )

        output_absorb, _, _  = npu_fused_dsa_attn_forward(
            model,
            hidden_states,
            position_embeddings,
            attention_mask,
            indexer_loss_coeff=1.0,
            use_fused_loss=True,
            skip_indexer=False,
        )

        assert torch.allclose(output_naive, output_absorb, rtol=1e-2, atol=1e-2), \
            f"Multi-sequence packing mismatch: max diff={(output_naive - output_absorb).abs().max()}"

    @pytest.mark.parametrize(
        "seq_lengths",
        [
            [4096],  # Single long sequence
            [2048, 2048],  # Two equal sequences
            [2048, 3072],  # Multiple sequences with varying lengths
        ],
        ids=[
            "single_seq_4096",
            "two_seq_2048_2048",
            "multi_seq_2048_3072",
        ]
    )
    def test_packing_various_configs(self, device, seq_lengths):
        """Test packing mode with various configurations.

        Note: All tests use NPU-compatible config: num_heads=64, index_topk=2048.
        skip_indexer=True to avoid kl_loss constraints.
        """
        config = create_test_config()  # NPU-compatible: num_heads=64, index_topk=2048
        model = GlmMoeDsaAttention(config, layer_idx=0).to(device)
        model = model.to(torch.bfloat16)
        model.eval()

        batch_size = 1
        total_seq_len = sum(seq_lengths)

        hidden_states = torch.randn(
            batch_size, total_seq_len, config.hidden_size,
            device=device, dtype=torch.bfloat16
        )

        cos, sin = create_rotary_embeddings(
            total_seq_len, config.qk_rope_head_dim,
            batch_size, device
        )
        position_embeddings = (cos, sin)

        attention_mask = create_packing_attention_mask(batch_size, seq_lengths, device)

        # Run both implementations with skip_indexer=True to avoid kl_loss constraints
        seed_all()
        output_naive, _, _  = dsa_attn_forward(
            model,
            hidden_states,
            position_embeddings,
            attention_mask,
            skip_indexer=False,
        )

        seed_all()
        output_absorb, _, _  = npu_fused_dsa_attn_forward(
            model,
            hidden_states,
            position_embeddings,
            attention_mask,
            indexer_loss_coeff=1.0,
            use_fused_loss=True,
            skip_indexer=False,
        )

        # Check shape
        assert output_naive.shape == output_absorb.shape, \
            f"Shape mismatch: naive={output_naive.shape}, absorb={output_absorb.shape}"

        # Check values
        assert torch.allclose(output_naive, output_absorb, rtol=1e-2, atol=1e-2), \
            f"Output mismatch: max diff={(output_naive - output_absorb).abs().max()}"


@pytest.mark.skipif(not NPU_AVAILABLE, reason="torch_npu not available")
class TestIndexerEquivalence:
    """Test indexer equivalence separately.

    Note: Indexer tests don't use kl_loss, so they have fewer constraints.
    But using consistent config for simplicity.
    """

    @pytest.fixture
    def device(self):
        return 'npu:0'

    @pytest.fixture
    def indexer(self, device):
        """Create an indexer instance for testing with NPU-compatible config."""
        config = create_test_config()  # NPU-compatible: num_heads=64, index_topk=2048
        indexer = GlmMoeDsaIndexer(config, layer_idx=0).to(device)
        indexer = indexer.to(torch.bfloat16)
        indexer.eval()
        return indexer

    @pytest.mark.parametrize(
        "seq_lengths",
        [
            [2048],
            [2048, 2048],
            [2048, 4096, 2048],
        ],
        ids=[
            "single_seq_2048",
            "two_seq_2048_2048",
            "three_seq_2048_4096_2048",
        ]
    )
    def test_indexer_packing_mode(self, indexer, device, seq_lengths):
        """Test indexer with packing mode.

        Note: seq_lengths should be >= index_topk (2048) for valid results.
        """
        batch_size = 1
        total_seq_len = sum(seq_lengths)

        hidden_states = torch.randn(
            batch_size, total_seq_len, indexer.config.hidden_size,
            device=device, dtype=torch.bfloat16
        )
        q_resid = torch.randn(
            batch_size, total_seq_len, indexer.config.q_lora_rank,
            device=device, dtype=torch.bfloat16
        )

        cos, sin = create_rotary_embeddings(
            total_seq_len, indexer.config.qk_rope_head_dim,
            batch_size, device
        )
        position_embeddings = (cos, sin)

        attention_mask = create_packing_attention_mask(batch_size, seq_lengths, device)

        # Run torch indexer
        with torch.no_grad():
            indices_torch = indexer_forward_torch(
                indexer,
                hidden_states,
                q_resid,
                position_embeddings,
                attention_mask,
                use_cache=False
            )
            indices_torch = fix_invalid_topk_indices(indices_torch, attention_mask)
            torch_mask = create_index_mask(indices_torch)

        # Run NPU indexer
        with torch.no_grad():
            indices_npu = indexer_forward_npu(
                indexer,
                hidden_states,
                q_resid,
                position_embeddings,
                attention_mask,
                use_cache=False
            )
            indices_npu = fix_npu_topk_indices(indices_npu, attention_mask)
            npu_mask = create_index_mask(indices_npu)

        # Compare index masks
        mask_match = (torch_mask == npu_mask).float().mean().item()
        print(f"Index mask match ratio for {seq_lengths}: {mask_match:.4f}")

        # Allow some tolerance since different implementations may select different indices
        assert mask_match >= 0.95, f"Index mask match ratio {mask_match} below threshold 0.95"


@pytest.mark.skipif(not NPU_AVAILABLE, reason="torch_npu not available")
class TestBackwardEquivalence:
    """Test backward pass equivalence between dsa_attn_forward and npu_fused_dsa_attn_forward.

    This tests gradient computation for the main attention path (excluding indexer)
    by using skip_indexer=True.

    Compares:
    - Gradient of hidden_states
    - Gradients of attention parameters (q_proj, kv_a_proj, kv_b_proj, o_proj, etc.)
    - Excludes indexer parameters since backward implementations differ
    """

    @pytest.fixture
    def device(self):
        return 'npu:0'

    @pytest.fixture
    def seq_len(self):
        return 2048

    @pytest.fixture
    def model_torch(self, device):
        """Create a model for naive implementation."""
        config = create_test_config()
        model = GlmMoeDsaAttention(config, layer_idx=0).to(device)
        model = model.to(torch.bfloat16)
        model.train()
        return model

    @pytest.fixture
    def model_npu(self, device, model_torch):
        """Create a model for NPU implementation with identical weights."""
        config = create_test_config()
        model = GlmMoeDsaAttention(config, layer_idx=0).to(device)
        model = model.to(torch.bfloat16)
        model.load_state_dict(model_torch.state_dict())
        model.train()
        return model

    def test_hidden_states_gradient(self, model_torch, model_npu, seq_len, device):
        """Test hidden_states gradient equivalence."""
        batch_size = 1
        config = create_test_config()

        hidden_torch = torch.randn(
            batch_size, seq_len, config.hidden_size,
            device=device, dtype=torch.bfloat16, requires_grad=True
        )
        hidden_npu = hidden_torch.clone().detach().requires_grad_(True)

        position_embeddings = create_rotary_embeddings(
            seq_len, config.qk_rope_head_dim, batch_size, device
        )
        attention_mask = create_packing_attention_mask(batch_size, [seq_len], device)

        # Forward + backward: naive
        seed_all()
        output_torch, _, _ = dsa_attn_forward(
            model_torch, hidden_torch, position_embeddings, attention_mask,
            skip_indexer=False,
        )

        loss_target = torch.randn_like(output_torch)
        loss_torch = ((output_torch - loss_target) ** 2).mean()
        loss_torch.backward()

        # Forward + backward: NPU optimized
        seed_all()
        output_npu, _, _ = npu_fused_dsa_attn_forward(
            model_npu, hidden_npu, position_embeddings, attention_mask,
            skip_indexer=False, indexer_loss_coeff=0.0, use_fused_loss=True,
        )

        loss_npu = ((output_npu - loss_target) ** 2).mean()
        loss_npu.backward()

        # Compare hidden_states gradients
        assert torch.allclose(hidden_torch.grad, hidden_npu.grad, rtol=1e-2, atol=1e-2), \
            f"hidden_states grad mismatch: max diff={(hidden_torch.grad - hidden_npu.grad).abs().max()}"

    def test_parameter_gradients(self, model_torch, model_npu, seq_len, device):
        """Test parameter gradients equivalence (excluding indexer)."""
        batch_size = 1
        config = create_test_config()

        hidden_torch = torch.randn(
            batch_size, seq_len, config.hidden_size,
            device=device, dtype=torch.bfloat16, requires_grad=True
        )
        hidden_npu = hidden_torch.clone().detach().requires_grad_(True)

        position_embeddings = create_rotary_embeddings(
            seq_len, config.qk_rope_head_dim, batch_size, device
        )
        attention_mask = create_packing_attention_mask(batch_size, [seq_len], device)

        # Forward + backward: naive
        seed_all()
        output_torch, _, _ = dsa_attn_forward(
            model_torch, hidden_torch, position_embeddings, attention_mask,
            skip_indexer=False,
        )

        loss_target = torch.randn_like(output_torch)
        loss_torch = ((output_torch - loss_target) ** 2).mean()
        loss_torch.backward()

        # Forward + backward: NPU optimized
        seed_all()
        output_npu, _, _ = npu_fused_dsa_attn_forward(
            model_npu, hidden_npu, position_embeddings, attention_mask,
            skip_indexer=False, indexer_loss_coeff=0.0, use_fused_loss=True,
        )

        loss_npu = ((output_npu - loss_target) ** 2).mean()
        loss_npu.backward()

        # Compare parameter gradients (exclude indexer)
        param_results = {}
        for name, param_torch in model_torch.named_parameters():
            if 'indexer' in name:
                continue  # Skip indexer parameters

            param_npu = dict(model_npu.named_parameters()).get(name)
            if param_npu is None:
                continue

            if param_torch.grad is None and param_npu.grad is None:
                continue

            if param_torch.grad is None or param_npu.grad is None:
                param_results[name] = False
                continue

            is_close = torch.allclose(param_torch.grad, param_npu.grad, rtol=1e-2, atol=1e-2)
            param_results[name] = is_close

            if not is_close:
                max_diff = (param_torch.grad - param_npu.grad).abs().max().item()
                print(f"  {name}: max_diff={max_diff:.6e}")

        assert all(param_results.values()), \
            f"Parameter grad mismatch for: {[k for k, v in param_results.items() if not v]}"