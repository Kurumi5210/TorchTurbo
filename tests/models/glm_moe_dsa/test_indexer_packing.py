"""
Test indexer implementations with packing mask.

Tests equivalence between torch and NPU implementations of the lightning indexer
using packing attention masks (multiple sequences packed together).

Run with: pytest tests/models/glm_moe_dsa/test_indexer_packing.py -v
"""

import pytest
import torch
import torch.nn as nn

# Check if torch_npu is available
try:
    import torch_npu
    NPU_AVAILABLE = True
except ImportError:
    NPU_AVAILABLE = False

from torch_turbo.models.glm_moe_dsa.npu import (
    indexer_forward_torch,
    indexer_forward_npu,
    fix_invalid_topk_indices,
    fix_npu_topk_indices,
    create_index_mask,
)
from torch_turbo.utils.flash_attn_utils import packing_mask_to_causal_mask
from torch_turbo.utils.npu_utils import seed_all


class MockConfig:
    """Mock config for testing - matches GLM-MoE-DSA config."""
    def __init__(
        self,
        hidden_size=2048,
        index_n_heads=4,
        index_head_dim=128,  # NPU lightning indexer only supports head_dim=128
        qk_rope_head_dim=64,
        index_topk=2048,
        q_lora_rank=512,
    ):
        self.hidden_size = hidden_size
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.index_topk = index_topk
        self.q_lora_rank = q_lora_rank


class GlmMoeDsaIndexerForTest(nn.Module):
    """Test version of GlmMoeDsaIndexer."""

    def __init__(self, config: MockConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank

        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(self.hidden_size, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim ** -0.5

        self._cached_keys = None

    def forward_torch(self, hidden_states, q_resid, position_embeddings, attention_mask=None, use_cache=False):
        return indexer_forward_torch(self, hidden_states, q_resid, position_embeddings, attention_mask, use_cache)

    def forward_npu(self, hidden_states, q_resid, position_embeddings, attention_mask=None, use_cache=False):
        return indexer_forward_npu(self, hidden_states, q_resid, position_embeddings, attention_mask, use_cache)


def create_rotary_embeddings(seq_len, head_dim, batch_size=1, device='cpu', dtype=torch.bfloat16):
    """Create rotary position embeddings (cos, sin)."""
    dim = head_dim
    rope_theta = 10000.0

    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
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
class TestIndexerPackingEquivalence:
    """Test equivalence between torch and NPU indexer implementations with packing mask."""

    @pytest.fixture
    def device(self):
        return 'npu:0'

    @pytest.fixture
    def seed(self):
        return 42

    @pytest.fixture
    def model(self, device):
        """Create a model instance for testing."""
        config = MockConfig()
        model = GlmMoeDsaIndexerForTest(config).to(device)
        model = model.to(torch.bfloat16)
        return model

    def _run_comparison(
        self,
        model,
        seq_lengths,
        index_topk,
        device,
        seed,
        match_threshold=0.98,
    ):
        """
        Run comparison between torch and NPU implementations.

        Args:
            model: The indexer model
            seq_lengths: List of sequence lengths for packing
            index_topk: Number of top-k indices to select
            device: Device to run on
            seed: Random seed
            match_threshold: Minimum match ratio for test to pass

        Returns:
            bool: True if index_mask match ratio >= threshold
        """
        torch.manual_seed(seed)

        batch_size = 1  # Packing mode only supports bs=1
        total_seq_len = sum(seq_lengths)

        # Create random inputs
        hidden_states = torch.randn(batch_size, total_seq_len, model.hidden_size, device=device, dtype=torch.bfloat16)
        q_resid = torch.randn(batch_size, total_seq_len, model.q_lora_rank, device=device, dtype=torch.bfloat16)

        # Create rotary embeddings
        cos, sin = create_rotary_embeddings(total_seq_len, model.qk_rope_head_dim, batch_size, device)
        position_embeddings = (cos, sin)

        # Create packing attention mask
        attention_mask = create_packing_attention_mask(batch_size, seq_lengths, device)

        # Run torch implementation
        with torch.no_grad():
            model._cached_keys = None
            seed_all()
            indices_torch = model.forward_torch(
                hidden_states, q_resid, position_embeddings,
                attention_mask=attention_mask, use_cache=False
            )
            indices_torch = fix_invalid_topk_indices(indices_torch, attention_mask)
            torch_mask = create_index_mask(indices_torch)

        # Run NPU implementation
        with torch.no_grad():
            model._cached_keys = None
            seed_all()
            indices_npu = model.forward_npu(
                hidden_states, q_resid, position_embeddings,
                attention_mask=attention_mask, use_cache=False
            )
            indices_npu = fix_npu_topk_indices(indices_npu, attention_mask)
            npu_mask = create_index_mask(indices_npu)

        # Check shape match
        assert indices_torch.shape == indices_npu.shape, \
            f"Shape mismatch! Torch: {indices_torch.shape}, NPU: {indices_npu.shape}"

        # Calculate mask match ratio
        mask_match = (torch_mask == npu_mask).float().mean().item()

        return mask_match >= match_threshold, mask_match

    @pytest.mark.parametrize(
        "seq_lengths,index_topk",
        [
            ([4096], 2048),  # Single long sequence
            ([2048, 2048], 2048),  # Two equal sequences
            ([1024, 2048, 512], 2048),  # Multiple sequences with varying lengths
            ([256, 512, 256, 1024, 128], 256),  # Many short sequences
            ([4096, 4096, 4096], 2048),  # Multiple long sequences
        ],
        ids=[
            "single_seq_4096",
            "two_seq_2048_2048",
            "multi_seq_1024_2048_512",
            "many_short_seqs",
            "multi_long_seq_4096x3",
        ]
    )
    def test_packing_mask_equivalence(self, model, seq_lengths, index_topk, device, seed):
        """Test equivalence between torch and NPU implementations with packing mask."""
        # Update model's index_topk for this test
        model.index_topk = index_topk
        model.config.index_topk = index_topk

        passed, match_ratio = self._run_comparison(
            model, seq_lengths, index_topk, device, seed
        )

        print(f"\nSequence lengths: {seq_lengths}, topk: {index_topk}")
        print(f"Index mask match ratio: {match_ratio:.4f} ({match_ratio*100:.1f}%)")

        assert passed, f"Index mask match ratio {match_ratio:.4f} is below threshold 0.98"

    def test_single_sequence_4096_detailed(self, model, device, seed):
        """Detailed test for single sequence with per-position analysis."""
        seq_lengths = [4096]
        index_topk = 2048

        model.index_topk = index_topk
        model.config.index_topk = index_topk

        torch.manual_seed(seed)
        batch_size = 1
        total_seq_len = sum(seq_lengths)

        hidden_states = torch.randn(batch_size, total_seq_len, model.hidden_size, device=device, dtype=torch.bfloat16)
        q_resid = torch.randn(batch_size, total_seq_len, model.q_lora_rank, device=device, dtype=torch.bfloat16)
        cos, sin = create_rotary_embeddings(total_seq_len, model.qk_rope_head_dim, batch_size, device)
        position_embeddings = (cos, sin)
        attention_mask = create_packing_attention_mask(batch_size, seq_lengths, device)

        # Run both implementations
        with torch.no_grad():
            model._cached_keys = None
            seed_all()
            indices_torch = model.forward_torch(
                hidden_states, q_resid, position_embeddings,
                attention_mask=attention_mask, use_cache=False
            )
            indices_torch = fix_invalid_topk_indices(indices_torch, attention_mask)
            torch_mask = create_index_mask(indices_torch)

            model._cached_keys = None
            seed_all()
            indices_npu = model.forward_npu(
                hidden_states, q_resid, position_embeddings,
                attention_mask=attention_mask, use_cache=False
            )
            indices_npu = fix_npu_topk_indices(indices_npu, attention_mask)
            npu_mask = create_index_mask(indices_npu)

        # Verify shapes
        assert indices_torch.shape == indices_npu.shape

        # Calculate exact match ratio of indices
        indices_match = (indices_torch == indices_npu).float().mean().item()
        print(f"\nExact indices match ratio: {indices_match:.4f} ({indices_match*100:.1f}%)")

        # Calculate index_mask match ratio
        mask_match = (torch_mask == npu_mask).float().mean().item()
        print(f"Index mask match ratio: {mask_match:.4f} ({mask_match*100:.1f}%)")

        assert mask_match >= 0.98, f"Index mask match ratio {mask_match:.4f} below threshold"


@pytest.mark.skipif(not NPU_AVAILABLE, reason="torch_npu not available")
class TestIndexerEdgeCases:
    """Test edge cases for indexer implementations."""

    @pytest.fixture
    def device(self):
        return 'npu:0'

    @pytest.fixture
    def model(self, device):
        config = MockConfig()
        model = GlmMoeDsaIndexerForTest(config).to(device)
        model = model.to(torch.bfloat16)
        return model

    def test_short_single_sequence(self, model, device):
        """Test with a very short single sequence."""
        seq_lengths = [256]
        index_topk = 128

        model.index_topk = index_topk
        model.config.index_topk = index_topk

        torch.manual_seed(42)
        batch_size = 1
        total_seq_len = sum(seq_lengths)

        hidden_states = torch.randn(batch_size, total_seq_len, model.hidden_size, device=device, dtype=torch.bfloat16)
        q_resid = torch.randn(batch_size, total_seq_len, model.q_lora_rank, device=device, dtype=torch.bfloat16)
        cos, sin = create_rotary_embeddings(total_seq_len, model.qk_rope_head_dim, batch_size, device)
        position_embeddings = (cos, sin)
        attention_mask = create_packing_attention_mask(batch_size, seq_lengths, device)

        with torch.no_grad():
            model._cached_keys = None
            seed_all()
            indices_torch = model.forward_torch(
                hidden_states, q_resid, position_embeddings,
                attention_mask=attention_mask, use_cache=False
            )
            indices_torch = fix_invalid_topk_indices(indices_torch, attention_mask)
            torch_mask = create_index_mask(indices_torch)

            model._cached_keys = None
            seed_all()
            indices_npu = model.forward_npu(
                hidden_states, q_resid, position_embeddings,
                attention_mask=attention_mask, use_cache=False
            )
            indices_npu = fix_npu_topk_indices(indices_npu, attention_mask)
            npu_mask = create_index_mask(indices_npu)

        assert indices_torch.shape == indices_npu.shape
        mask_match = (torch_mask == npu_mask).float().mean().item()
        assert mask_match >= 0.98, f"Index mask match ratio {mask_match:.4f} below threshold"

    def test_topk_larger_than_sequence(self, model, device):
        """Test when topk is larger than sequence length."""
        seq_lengths = [512]
        index_topk = 1024  # Larger than sequence length

        model.index_topk = index_topk
        model.config.index_topk = index_topk

        torch.manual_seed(42)
        batch_size = 1
        total_seq_len = sum(seq_lengths)

        hidden_states = torch.randn(batch_size, total_seq_len, model.hidden_size, device=device, dtype=torch.bfloat16)
        q_resid = torch.randn(batch_size, total_seq_len, model.q_lora_rank, device=device, dtype=torch.bfloat16)
        cos, sin = create_rotary_embeddings(total_seq_len, model.qk_rope_head_dim, batch_size, device)
        position_embeddings = (cos, sin)
        attention_mask = create_packing_attention_mask(batch_size, seq_lengths, device)

        with torch.no_grad():
            model._cached_keys = None
            seed_all()
            indices_torch = model.forward_torch(
                hidden_states, q_resid, position_embeddings,
                attention_mask=attention_mask, use_cache=False
            )
            indices_torch = fix_invalid_topk_indices(indices_torch, attention_mask)

            model._cached_keys = None
            seed_all()
            indices_npu = model.forward_npu(
                hidden_states, q_resid, position_embeddings,
                attention_mask=attention_mask, use_cache=False
            )
            indices_npu = fix_npu_topk_indices(indices_npu, attention_mask)

        # The actual topk should be min(index_topk, seq_len)
        actual_topk = min(index_topk, total_seq_len)
        assert indices_torch.shape[-1] == actual_topk
        assert indices_npu.shape[-1] == actual_topk
