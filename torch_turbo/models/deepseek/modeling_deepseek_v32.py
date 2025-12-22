import math
from dataclasses import dataclass
from collections.abc import Callable
from typing import Optional, Union, Tuple, Literal
import json

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from scipy.linalg import hadamard

from transformers import (
    AutoConfig,
    PretrainedConfig,
    AutoModelForCausalLM
)

from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3ForCausalLM,
    DeepseekV3Attention,
    DeepseekV3RMSNorm,
    rotate_half,
    eager_attention_forward
)
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

class DeepseekV32Config(DeepseekV3Config):
    model_type = "deepseek_v32"
    
    def __init__(self, **kwargs):
        self.index_head_dim = kwargs.pop("index_head_dim", 128)
        self.index_n_heads = kwargs.pop("index_n_heads", 64)
        self.index_topk = kwargs.pop("index_topk", 2048)
        
        super().__init__(**kwargs)

AutoConfig.register("deepseek_v32", DeepseekV32Config)

class LayerNorm(nn.Module):
    """
    Layer Normalization.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        return F.layer_norm(x.float(), (self.dim,), self.weight.float(), self.bias.float(), self.eps).type_as(x)

def hadamard_transform_ref(x, scale=1.0):
    """
    Eager implementation of the Hadamard transform

    Args:
        x:(torch.Tensor): input tensor
    """
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2 ** log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = F.linear(x, torch.tensor(hadamard(dim_padded, dtype=float), dtype=x.dtype, device=x.device))
    out = out * scale

    return out[..., :dim].reshape(*x_shape)

def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    hidden_size = x.size(-1)
    return hadamard_transform_ref(x, scale=hidden_size ** -0.5)


def apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    dtype = q.dtype
    q = q.float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed.to(dtype)

def apply_rotary_pos_emb_interleave(q, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    dtype = q.dtype
    q = q.float().view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    # rope interleave implementation in transformers is wrong which need transpose
    q_embed = q_embed.view(b, h, s, 2, d // 2).transpose(4, 3).reshape(b, h, s, d)
    return q_embed.to(dtype)

class Indexer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim: int = config.hidden_size
        self.n_heads: int = config.index_n_heads
        self.n_local_heads = config.index_n_heads
        self.head_dim: int = config.index_head_dim
        self.rope_head_dim: int = config.qk_rope_head_dim
        self.index_topk: int = config.index_topk
        self.q_lora_rank: int = config.q_lora_rank
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.head_dim, bias=False)
        self.k_norm = LayerNorm(self.head_dim)
        # weights_proj in the checkpoint is stored in bf16, while the parameters here are stored in fp32 for convenient.
        self.weights_proj = nn.Linear(self.dim, self.n_heads, dtype=torch.float32, bias=False)
        self.softmax_scale = self.head_dim ** -0.5
    
    def _compute_index_scores(
        self, q: torch.Tensor, weights: torch.Tensor, k: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform index score using BF16 precision.
        Reference:
            https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/kernel.py#L254-L274
        This is a BF16 implementation of the `fp8_index` logic:
            1. Compute attention scores: q @ k^T;
            2. Apply ReLU activation;
            3. Weight by attention weights;
            4. Sum across attention heads.
        Args:
            q: BF16 [batch, seqlen_q, index_n_heads, index_head_dim], the query tensor.
            weights: BF16 [batch, seqlen_q, index_n_heads], the attention weights.
            k: BF16 [batch, seqlen_k, index_head_dim], the key tensor.
        Returns:
            index_scores: FP32 [batch, seqlen_q, seqlen_k], the index scores.
        """
        # Compute attention scores: q @ k^T
        # [batch, seqlen_q, index_n_heads, index_head_dim] @ [batch, seqlen_k, index_head_dim]^T
        #   -> [batch, seqlen_q, index_n_heads, seqlen_k]
        index_scores = torch.einsum('bshd,btd->bsht', q.float(), k.float())

        # Apply ReLU activation.
        index_scores = torch.relu(index_scores)

        # Weight each head by attention weights.
        # [batch, seqlen_q, index_n_heads, seqlen_k] * [batch, seqlen_q, index_n_heads, 1]
        #   -> [batch, seqlen_q, index_n_heads, seqlen_k]
        index_scores = index_scores * weights.unsqueeze(-1)

        # Sum across attention heads.
        # [batch, seqlen_q, index_n_heads, seqlen_k] -> [batch, seqlen_q, seqlen_k]
        index_scores = index_scores.sum(dim=2)

        # No transpose needed since it's already [batch, seqlen_q, seqlen_k]
        return index_scores

    def forward(self, x: torch.Tensor, qr: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: Optional[torch.Tensor]):
        x = x.detach()
        qr = qr.detach()
        # TODO: sp support here
        bsz, seqlen, _ = x.size()
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        k = self.wk(x)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        # rope in indexer is not interleaved
        k_pe_input = k_pe.unsqueeze(2)  # [bsz, seqlen, 1, rope_dim]
        q_rope = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=2)
        k_rope = apply_rotary_pos_emb(k_pe_input, cos, sin, unsqueeze_dim=2)
        k_rope = k_rope.squeeze(2)
        q = torch.cat([q_rope, q_nope], dim=-1)
        k = torch.cat([k_rope, k_nope], dim=-1)
        q = rotate_activation(q)
        k = rotate_activation(k)

        weights = self.weights_proj(x.float()) * self.n_heads ** -0.5
        weights = weights * self.softmax_scale
        index_scores = self._compute_index_scores(q, weights, k)
        if mask is not None:
            index_scores += mask.squeeze(0)

        # =========================================
        # Select top-k indices
        # =========================================
        topk_k = min(self.index_topk, seqlen)
        # [batch, seqlen, index_topk]
        topk_indices = index_scores.topk(topk_k, dim=-1)[1]
        return index_scores, topk_indices

def compute_dsa_indexer_loss(
    index_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: bool,
) -> torch.Tensor:
    """
    Compute KL divergence loss between index_scores and true attention_scores.
    This loss trains the indexer to predict which tokens are important by matching the distribution
    of true attention scores.
    Reference: Section 2.1 of
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf
    Args:
        index_scores: Scores predicted by indexer [batch, seqlen_q, seqlen_k].
        topk_indices: Top-k indices [batch, seqlen_q, index_topk].
        query: Query tensor [seqlen_q, batch, heads, dim].
        key: Key tensor [seqlen_k, batch, heads, dim].
        softmax_scale: Scale coefficient after q @ k^T.
        loss_coeff: Coefficient for the indexer KL divergence loss.
        sparse_loss: bool, whether to use sparse indexer loss. If True, only the topk
            indices will be used to compute the loss.
        pg_collection: Process group collection, must have TP process group.
    Returns:
        index_loss: KL divergence loss (scalar).
    """
    b, sq, np, hn = query.size()
    sk = key.size(1)

    # [b, sq, np, hn] -> [b, np, sq, hn] -> [b * np, sq, hn]
    query = query.permute(0, 2, 1, 3).reshape(b * np, sq, hn)
    # [b, sk, np, hn] -> [b, np, hn, sk] -> [b * np, hn, sk]
    key = key.permute(0, 2, 3, 1).reshape(b * np, hn, sk)
    # Compute attention scores [b * np, sq, sk]
    attention_scores = torch.bmm(query.float(), key.float()) * softmax_scale
    # Reshape to [b, np, sq, sk]
    attention_scores = attention_scores.reshape(b, np, sq, sk)

    # causal_mask [sq, sk]
    causal_mask = torch.triu(
        torch.full((sq, sk), float('-inf'), dtype=torch.float32, device=attention_scores.device),
        diagonal=1,
    )
    # index_mask [b, sq, sk]
    index_mask = torch.full(
        (b, sq, sk), float("-inf"), dtype=torch.float32, device=causal_mask.device
    ).scatter_(-1, topk_indices, 0)

    # [b, np, sq, skv] + [1, 1, sq, skv] -> [b, np, sq, skv]
    attention_scores += causal_mask.view(1, 1, sq, sk)
    if sparse_loss:
        # [b, np, sq, sk] + [b, 1, sq, sk] -> [b, np, sq, sk]
        attention_scores += index_mask.view(b, 1, sq, sk)
        # [b, sq, sk] + [b, sq, sk] -> [b, sq, sk]
        index_scores += index_mask

    # [b, np, sq, sk] -> [b, np, sq, sk]
    attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32)
    # [b, sq, sk] -> [b, sq, sk]
    index_scores = torch.nn.functional.softmax(index_scores, dim=-1, dtype=torch.float32)

    # Sum attention scores across heads.
    # [batch, heads, seqlen_q, seqlen_k] -> [batch, seqlen_q, seqlen_k]
    attention_scores = attention_scores.sum(dim=1)

    # L1 normalize target on the last dimension. Doesn't use abs() because attention_scores are
    # obtained from softmax so they are already non-negative.
    attention_scores = attention_scores / attention_scores.sum(dim=-1, keepdim=True)

    # Compute KL divergence: KL(target || index) = target(x) * log(target(x) / index(x))
    # kl_per_element [b, sq, sk]
    kl_per_element = attention_scores * (
        torch.log(attention_scores + 1e-10) - torch.log(index_scores + 1e-10)
    )

    # [b, sq, sk] -> [b, sq] -> [1]
    # Each token has same weight in the loss.
    kl_div = kl_per_element.sum(dim=-1).mean()

    # Scale by coefficient.
    indexer_loss = kl_div * loss_coeff

    return indexer_loss


class DSAIndexerLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that triggers the backward pass and scales the grad for indexer loss.
    This custom autograd function attaches a KL divergence loss to the activation
    to train the indexer to predict attention scores without affecting the forward pass.
    """

    main_loss_backward_scale: torch.Tensor = None

    @staticmethod
    def forward(ctx, output: torch.Tensor, indexer_loss: torch.Tensor):
        """Preserve the indexer_loss by storing it in the context to avoid garbage collection.
        Args:
            output: The output tensor (activation).
            indexer_loss: The indexer KL divergence loss tensor.
        Returns:
            torch.Tensor: The output tensor unchanged.
        """
        ctx.save_for_backward(indexer_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for indexer loss.
        Args:
            grad_output: The gradient of the output.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled indexer loss
                gradient.
        """
        (indexer_loss,) = ctx.saved_tensors
        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = torch.tensor(
                1.0, device=indexer_loss.device
            )
        indexer_loss_backward_scale = DSAIndexerLossAutoScaler.main_loss_backward_scale
        scaled_indexer_loss_grad = torch.ones_like(indexer_loss) * indexer_loss_backward_scale
        return grad_output, scaled_indexer_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """Set the scale of the indexer loss.
        Args:
            scale: The scale value to set.
        """
        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = scale
        else:
            DSAIndexerLossAutoScaler.main_loss_backward_scale.copy_(scale)

class SparseAttn(DeepseekV3Attention):
    def __init__(self, config: DeepseekV3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.indexer = Indexer(config)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values = None,
        cache_position = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        assert self.q_lora_rank is not None, "must be"

        qr = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q_states = self.q_b_proj(qr)
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        cos, sin = position_embeddings
        if self.config.rope_interleave:  # support using interleaved weights for efficiency
            q_rot = apply_rotary_pos_emb_interleave(q_rot, cos, sin)
            k_rot = apply_rotary_pos_emb_interleave(k_rot, cos, sin)
        else:
            q_rot = apply_rotary_pos_emb(q_rot, cos, sin)
            k_rot = apply_rotary_pos_emb(k_rot, cos, sin)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # ==========================================================================================================
        # TODO: attention_mask should support difference type of inputs, currently assume casual mask
        index_scores, topk_indices = self.indexer(hidden_states, qr, cos, sin, attention_mask)
        index_mask = torch.full((batch_size, seq_length, seq_length), float("-inf"), device=hidden_states.device)
        index_mask.scatter_(-1, topk_indices, 0)
        sparse_mask = index_mask + attention_mask
        # ==========================================================================================================

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            sparse_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        if torch.is_grad_enabled():
            # Compute KL divergence loss between indexer scores and true attention scores
            indexer_loss_coeff = getattr(self.config, 'dsa_indexer_loss_coeff', 0.0)
            if indexer_loss_coeff > 0:
                indexer_loss = compute_dsa_indexer_loss(
                    index_scores,
                    topk_indices,
                    query_states.detach(),
                    key_states.detach(),
                    self.scaling,
                    indexer_loss_coeff,
                    getattr(self.config, "dsa_indexer_use_sparse_loss", False),
                )
                # Attach loss to output
                attn_output = DSAIndexerLossAutoScaler.apply(attn_output, indexer_loss)

        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

def rms_forward(self, hidden_states):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return (self.weight * hidden_states).to(input_dtype)

def create_deepseek_v32_model(config):
    assert config.model_type == "deepseek_v32", "must be"
    model = DeepseekV3ForCausalLM(config)
    DeepseekV3RMSNorm.forward = rms_forward

    # add indexer to attention module
    for i, layer in enumerate(model.model.layers):
        layer.self_attn = SparseAttn(layer.self_attn.config, layer.self_attn.layer_idx)

    return model

def load_config(path):
    r"""Load model config."""
    return AutoConfig.from_pretrained(path, trust_remote_code=True)

if __name__ == "__main__":
    config_path = "/mnt/sfs_turbo/pretrained_models/DevQuasar-DeepSeek-V3.2-Speciale-BF16/config.json"
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    print(config_dict)
    c1 = load_config("/mnt/sfs_turbo/pretrained_models/DevQuasar-DeepSeek-V3.2-Speciale-BF16/")
    print(c1)
    config = DeepseekV32Config(**config_dict)
    print(config)
    with torch.device("meta"):
        model = create_deepseek_v32_model(config)
    print(model)