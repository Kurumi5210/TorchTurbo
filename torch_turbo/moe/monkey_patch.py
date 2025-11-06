import torch
import torch.nn.functional as F
import os
from typing import Optional

from transformers.modeling_flash_attention_utils import _flash_attention_forward
try:
    from transformers.models.qwen3_next.modeling_qwen3_next import apply_mask_to_padding_states
except ImportError:
    pass  # Qwen3Next is only available in transformers>=4.57.0.dev0

try:
    from ring_attn_ascend import zigzag_ring_flash_attn_func
except ImportError:
    pass
    
import sys
from torch_turbo.sequence_parallel.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size,
    SequenceParallelCausalConv1dComm,
)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=2, repeats=n_rep). The hidden states go from (batch,
    seqlen, num_key_value_heads, head_dim) to (batch, seqlen, num_attention_heads, head_dim)
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)

def _ulysses_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask,
    query_length,
    *args,
    position_ids: Optional[torch.Tensor] = None,
    **kwargs,
):
    """Insert all-to-all before and after flash attention.
    DeepSpeed-Ulysses: https://arxiv.org/pdf/2309.14509

    Args:
        query_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads, head_dim)
        key_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads_k, head_dim)
        value_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads_k, head_dim)
        position_ids (torch.Tensor, optional): (batch_size, seqlen/sp_size)

    Returns:
        torch.Tensor: (batch_size, seqlen/sp_size, nheads, head_dim)
    """
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

    ########## AlltoAll for Ulysses ##########
    if ulysses_sp_size > 1:
        assert position_ids is not None, "position_ids is required for Ulysses sequence parallelism"

        # NOTE: repeat kv heads to be divided by sequence parallel. Instead of repeating nheads_q//nheads_k,
        # we choose to repeat sp_size//nheads_k, since flash_attention supports MQA/GQA.
        # For example:
        # - nheads_k=4, sp=8, repeats=2
        # - nheads_k=8, sp=8, repeats=1
        # - nheads_k=16, sp=8, repeats=1
        repeats = max(ulysses_sp_size // key_states.size(2), 1)
        key_states = repeat_kv(key_states, repeats)
        value_states = repeat_kv(value_states, repeats)

        # (bsz, seq_len/n, n_head, head_dim) -> (bsz, seq_len, n_head/n, head_dim)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=1, head_dim=2)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=1, head_dim=2)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=1, head_dim=2)

        # TODO: all_gather position_ids because `prepare_fa2_from_position_ids` needs it, we can eliminate
        # this all_gather by passing cu_seq_lens_q, cu_seq_lens_k, max_length_k, max_length_q explicitly.
        # https://github.com/huggingface/transformers/pull/33932

        # (bsz, seq_len/n) -> (bsz, seq_len)
        position_ids_list = [torch.empty_like(position_ids) for _ in range(ulysses_sp_size)]
        torch.distributed.all_gather(position_ids_list, position_ids, group=get_ulysses_sequence_parallel_group())
        position_ids = torch.concat(position_ids_list, dim=-1)
        if query_length != query_states.size(1):
            assert query_length * ulysses_sp_size == query_states.size(1), "ulysses fa only pass 1/sp query here"
            query_length = query_states.size(1)

    # (bsz, seq_len, n_head/n, head_dim)
    attn_output = _flash_attention_forward(query_states, key_states, value_states, attention_mask, query_length, *args, position_ids=position_ids, **kwargs)

    ########## AlltoAll for Ulysses ##########
    if ulysses_sp_size > 1:
        # (bsz, seq_len, n_head/n, head_dim) -> (bsz, seq_len/n, n_head, head_dim)
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2)

    return attn_output

def _zigzag_ring_flash_attn_forward(
    query_states,
    key_states,
    value_states,
    attention_mask,
    query_length,
    dropout=0,
    deterministic=False,
    sliding_window=None,
    is_causal=True,
    **kwargs,
):
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()
    if ulysses_sp_size > 1:
        attn_output, _, _ = zigzag_ring_flash_attn_func(
            query_states, key_states, value_states, dropout, causal=is_causal, group=get_ulysses_sequence_parallel_group()
        )
    else:
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length,
            is_causal=is_causal,
            dropout=dropout,
            deterministic=deterministic,
            sliding_window=sliding_window,
            **kwargs
        )
    return attn_output

def _ulysses_gated_delta_net_forward(
    self,
    hidden_states,
    cache_params,
    cache_position,
    attention_mask,
):
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

    # Set up dimensions for reshapes later
    batch_size, seq_len, _ = hidden_states.shape

    use_precomputed_states = (
        cache_params is not None
        and cache_params.has_previous_state
        and seq_len == 1
        and cache_position is not None
    )

    # getting projected states from cache if it exists
    if cache_params is not None:
        conv_state = cache_params.conv_states[self.layer_idx]
        recurrent_state = cache_params.recurrent_states[self.layer_idx]

    projected_states_qkvz = self.in_proj_qkvz(hidden_states)
    projected_states_ba = self.in_proj_ba(hidden_states)
    query, key, value, z, b, a = self.fix_query_key_value_ordering(projected_states_qkvz, projected_states_ba)
    query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))

    mixed_qkv = torch.cat((query, key, value), dim=-1)
    mixed_qkv = mixed_qkv.transpose(1, 2)

    if ulysses_sp_size > 1:
        comm = SequenceParallelCausalConv1dComm(self.conv_kernel_size, group=get_ulysses_sequence_parallel_group())

        recv_left = comm.exchange_left_border(mixed_qkv)
        if recv_left is not None:
            mixed_qkv = torch.cat((recv_left, mixed_qkv), dim=2)
        else:
            mixed_qkv = F.pad(mixed_qkv, (comm.left_padding_size, 0))

    if use_precomputed_states:
        # 2. Convolution sequence transformation
        # NOTE: the conv state is updated in `causal_conv1d_update`
        mixed_qkv = self.causal_conv1d_update(
            mixed_qkv,
            conv_state,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.activation,
        )
    else:
        if cache_params is not None:
            conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
            cache_params.conv_states[self.layer_idx] = conv_state
        if self.causal_conv1d_fn is not None:
            mixed_qkv = self.causal_conv1d_fn(
                x=mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=None,
            )
        else:
            if ulysses_sp_size > 1:
                mixed_qkv = self.conv1d(mixed_qkv)
            else:
                mixed_qkv = self.conv1d(mixed_qkv)[:, :, :seq_len]
            mixed_qkv = F.silu(mixed_qkv)

    if ulysses_sp_size > 1:
        # Remove the extra border added for convolution
        mixed_qkv = mixed_qkv[:, :, comm.left_padding_size : comm.left_padding_size + seq_len]

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv,
        [
            self.key_dim,
            self.key_dim,
            self.value_dim,
        ],
        dim=-1,
    )
    query = query.reshape(query.shape[0], query.shape[1], -1, self.head_k_dim)
    key = key.reshape(key.shape[0], key.shape[1], -1, self.head_k_dim)
    value = value.reshape(value.shape[0], value.shape[1], -1, self.head_v_dim)

    beta = b.sigmoid()
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    if ulysses_sp_size > 1:
        # (bsz, seq_len/n, n_head, head_dim) -> (bsz, seq_len, n_head/n, head_dim)
        query = gather_seq_scatter_heads(query, seq_dim=1, head_dim=2)
        key = gather_seq_scatter_heads(key, seq_dim=1, head_dim=2)
        value = gather_seq_scatter_heads(value, seq_dim=1, head_dim=2)
        # (bsz, seq_len/n, n_head) -> (bsz, seq_len, n_head/n)
        beta = gather_seq_scatter_heads(beta, seq_dim=1, head_dim=2)
        g = gather_seq_scatter_heads(g, seq_dim=1, head_dim=2)

    if not use_precomputed_states:
        core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )

    else:
        core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )

    if ulysses_sp_size > 1:
        # (bsz, seq_len, n_head/n, head_dim) -> (bsz, seq_len/n, n_head, head_dim)
        core_attn_out = gather_heads_scatter_seq(core_attn_out, seq_dim=1, head_dim=2)

    # Update cache
    if cache_params is not None:
        cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

    z_shape_og = z.shape
    # reshape input data into 2D tensor
    core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
    z = z.reshape(-1, z.shape[-1])
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(z_shape_og)
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)

    output = self.out_proj(core_attn_out)
    return output

def apply_monkey_patch(model, args):
    """
    Apply monkey patch to the MoE layer for expert parallel and attention for ulysses sequence parallel.
    """
    module = sys.modules[model.__module__]
    try:
        num_attention_heads, num_key_value_heads = model.config.num_attention_heads, model.config.num_key_value_heads
    except AttributeError:
        num_attention_heads, num_key_value_heads = model.config.text_config.num_attention_heads, model.config.text_config.num_key_value_heads
    ulysses_sp_size = args.sp_size
    if args.sp_mode == "ulysses":
        assert num_attention_heads % ulysses_sp_size == 0, f"num_attention_heads {num_attention_heads} must be divisible by ulysses_sp_size {ulysses_sp_size}"
        assert num_key_value_heads % ulysses_sp_size == 0 or ulysses_sp_size % num_key_value_heads == 0, (
            f"num_key_value_heads {num_key_value_heads} must be divisible by ulysses_sp_size {ulysses_sp_size}or vise versa. Upon ulysses_sp_size % num_key_value_heads == 0,kv heads are repeated to ensure correctness."
        )
        if "next" in model.config.model_type:
            linear_num_value_heads = model.config.linear_num_value_heads
            assert linear_num_value_heads % ulysses_sp_size == 0, f"linear_num_value_heads {linear_num_value_heads} must be divisible by ulysses_sp_size {ulysses_sp_size}"

        new_attention_forward = _ulysses_flash_attention_forward
    else:
        new_attention_forward = _zigzag_ring_flash_attn_forward

    if ulysses_sp_size > 1:
        # transformers<=4.47.1
        if hasattr(module, "_flash_attention_forward"):
            module._flash_attention_forward = new_attention_forward
            print(f"Monkey patch _flash_attention_forward in {model.__module__}")
        else:
            # transformers>=4.48.0
            from transformers.integrations import flash_attention

            flash_attention._flash_attention_forward = new_attention_forward
            print(f"Monkey patch _flash_attention_forward in {flash_attention.__name__}")
        if model.config.architectures[0] == "BailingMoeV2ForCausalLM":
            for layer in model.model.layers:
                layer.attention._flash_attention_forward = new_attention_forward
        if "next" in model.config.model_type:
            from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextGatedDeltaNet
            Qwen3NextGatedDeltaNet.forward = _ulysses_gated_delta_net_forward
            print(f"Monkey patch Qwen3NextGatedDeltaNet.forward")

    if args.use_fused_rope:
        import transformers.models.qwen3_moe.modeling_qwen3_moe as modeling_qwen3_moe
        from mindspeed.ops.npu_rotary_position_embedding import npu_rotary_position_embedding
        def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
            cos = cos.unsqueeze(unsqueeze_dim)
            sin = sin.unsqueeze(unsqueeze_dim)
            return npu_rotary_position_embedding(q, cos, sin, 0), npu_rotary_position_embedding(k, cos, sin, 0)
        modeling_qwen3_moe.apply_rotary_pos_emb = apply_rotary_pos_emb
        import torch_turbo.moe.modeling_bailing_moe_v2 as modeling_bailing_moe_v2
        def apply_rotary_pos_emb_bailing(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
            cos = cos.unsqueeze(unsqueeze_dim)
            sin = sin.unsqueeze(unsqueeze_dim)
            rotary_dim = cos.shape[-1]
            q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
            k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
            q_embed = npu_rotary_position_embedding(q_rot, cos, sin, 0)
            k_embed = npu_rotary_position_embedding(k_rot, cos, sin, 0)
            q_embed = torch.cat([q_embed, q_pass], dim=-1)
            k_embed = torch.cat([k_embed, k_pass], dim=-1)
            return q_embed, k_embed
        modeling_bailing_moe_v2.apply_rotary_pos_emb = apply_rotary_pos_emb_bailing

    if args.use_fused_rms_norm:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm
        from torch_npu import npu_rms_norm
        def fused_forward(self, hidden_states: torch.Tensor):
            return npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
        Qwen3MoeRMSNorm.forward = fused_forward
        from .modeling_bailing_moe_v2 import BailingMoeV2RMSNorm
        BailingMoeV2RMSNorm.forward = fused_forward

    if args.use_fused_permute_unpermute:
        from mindspeed.ops.npu_moe_token_permute import npu_moe_token_permute
        from mindspeed.ops.npu_moe_token_unpermute import npu_moe_token_unpermute
        from typing import List, Optional, Tuple
        from torch_turbo.distributed.mappings import all_to_all
        from torch_turbo.transformer.utils  import sort_chunks_by_idxs
        from torch_turbo.moe.token_dispatcher import MoEAlltoAllTokenDispatcher

        def permute(
            tokens,
            routing_map,
            probs: Optional[torch.Tensor] = None,
            num_out_tokens: Optional[int] = None,
            fused: bool = False,
            drop_and_pad: bool = False,
        ):
            expert_indices = torch.nonzero(routing_map, as_tuple=True)[1].view(tokens.shape[0], -1)
            permuted_tokens, sorted_indices = npu_moe_token_permute(tokens, expert_indices)
            return permuted_tokens, sorted_indices

        def unpermute(
            permuted_tokens: torch.Tensor,
            sorted_indices: torch.Tensor,
            restore_shape: torch.Size,
            probs: torch.Tensor = None,
            routing_map: torch.Tensor = None,
            fused: bool = False,
            drop_and_pad: bool = False,
        ):
            # topk = int(routing_map[0].sum(dim=0).item())
            # # If probs is not passed in, the npu_moe_token_unpermute will consider topk to be 1, resulting in an incorrect outcome
            # # link: https://gitee.com/ascend/MindSpeed/blob/1.0.0_core_r0.6.0/docs/ops/npu_fused_moe_token_unpermute.md
            # fake_probs = (torch.ones(routing_map.shape[0], topk, device=permuted_tokens.device, dtype=permuted_tokens.dtype))
            unpermuted_tokens = npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs=probs)
            return unpermuted_tokens

        def token_permutation(
            self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Dispatch tokens to local experts using AlltoAll communication.

            This method performs the following steps:
            1. Preprocess the routing map to get metadata for communication and permutation.
            2. Permute input tokens for AlltoAll communication.
            3. Perform expert parallel AlltoAll communication.
            4. Sort tokens by local expert (if multiple local experts exist).

            Args:
                hidden_states (torch.Tensor): Input token embeddings.
                probs (torch.Tensor): The probabilities of token to experts assignment.
                routing_map (torch.Tensor): The mapping of token to experts assignment.

            Returns:
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                    - Permuted token embeddings for local experts.
                    - Number of tokens per expert.
                    - Permuted probs of each token produced by the router.
            """
            # Preprocess: Get the metadata for communication, permutation and computation operations.
            self.hidden_shape = hidden_states.shape
            self.probs = probs
            self.routing_map = routing_map
            assert probs.dim() == 2, "Expected 2D tensor for probs"
            assert routing_map.dim() == 2, "Expected 2D tensor for token2expert mask"
            assert routing_map.dtype == torch.bool, "Expected bool tensor for mask"
            hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
            tokens_per_expert = self.preprocess(self.routing_map)

            # Permutation 1: input to AlltoAll input
            tokens_per_expert = self._maybe_dtoh_and_synchronize(
                "before_permutation_1", tokens_per_expert
            )
            self.hidden_shape_before_permute = hidden_states.shape
            (
                permutated_local_input_tokens,
                self.reversed_local_input_permutation_mapping,
            ) = permute(
                hidden_states,
                routing_map,
                probs=probs,
                num_out_tokens=self.num_out_tokens,
                fused=self.config.moe_permute_fusion,
                drop_and_pad=self.drop_and_pad,
            )

            # Perform expert parallel AlltoAll communication
            tokens_per_expert = self._maybe_dtoh_and_synchronize(
                "before_ep_alltoall", tokens_per_expert
            )
            global_input_tokens = all_to_all(
                self.ep_group, permutated_local_input_tokens, self.output_splits, self.input_splits
            )

            # Permutation 2: Sort tokens by local expert.
            tokens_per_expert = self._maybe_dtoh_and_synchronize(
                "before_permutation_2", tokens_per_expert
            )
            if self.num_local_experts > 1:
                if self.drop_and_pad:
                    global_input_tokens = (
                        global_input_tokens.view(
                            self.tp_size * self.ep_size,
                            self.num_local_experts,
                            self.capacity,
                            *global_input_tokens.size()[1:],
                        )
                        .transpose(0, 1)
                        .contiguous()
                        .flatten(start_dim=0, end_dim=2)
                    )
                else:
                    global_input_tokens, global_probs = sort_chunks_by_idxs(
                        global_input_tokens,
                        self.num_global_tokens_per_local_expert.ravel(),
                        self.sort_input_by_local_experts,
                        probs=None,
                        fused=self.config.moe_permute_fusion,
                    )

            tokens_per_expert = self._maybe_dtoh_and_synchronize("before_finish", tokens_per_expert)

            return global_input_tokens, tokens_per_expert, probs

        def token_unpermutation(
            self, hidden_states: torch.Tensor, bias: Optional[torch.Tensor] = None, probs: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            """
            Reverse the token permutation to restore the original order.

            This method performs the following steps:
            1. Unsort tokens by local expert (if multiple local experts exist).
            2. Perform expert parallel AlltoAll communication to restore the original order.
            3. Unpermute tokens to restore the original order.

            Args:
                hidden_states (torch.Tensor): Output from local experts.
                bias (torch.Tensor, optional): Bias tensor (not supported).

            Returns:
                Tuple[torch.Tensor, Optional[torch.Tensor]]:
                    - Unpermuted token embeddings in the original order.
                    - None (bias is not supported).
            """
            assert bias is None, "Bias is not supported in MoEAlltoAllTokenDispatcher"

            # Unpermutation 2: Unsort tokens by local expert.
            if self.num_local_experts > 1:
                if self.drop_and_pad:
                    hidden_states = (
                        hidden_states.view(
                            self.num_local_experts,
                            self.tp_size * self.ep_size,
                            self.capacity,
                            *hidden_states.size()[1:],
                        )
                        .transpose(0, 1)
                        .contiguous()
                        .flatten(start_dim=0, end_dim=2)
                    )
                else:
                    hidden_states, _ = sort_chunks_by_idxs(
                        hidden_states,
                        self.num_global_tokens_per_local_expert.T.ravel(),
                        self.restore_output_by_local_experts,
                        fused=self.config.moe_permute_fusion,
                    )

            # Perform expert parallel AlltoAll communication
            # hidden_states: [SEQL, H] -> [SEQL, H/TP]
            permutated_local_input_tokens = all_to_all(
                self.ep_group, hidden_states, self.input_splits, self.output_splits
            )

            # Unpermutation 1: AlltoAll output to output
            # HACK
            probs = probs.to(torch.bfloat16)
            output = unpermute(
                permutated_local_input_tokens,
                self.reversed_local_input_permutation_mapping,
                restore_shape=self.hidden_shape_before_permute,
                routing_map=self.routing_map,
                fused=self.config.moe_permute_fusion,
                drop_and_pad=self.drop_and_pad,
                probs=probs,
            )

            # Reshape the output tensor
            output = output.view(self.hidden_shape)

            # release cached tensors
            self.probs = None
            self.routing_map = None

            return output, None
        MoEAlltoAllTokenDispatcher.token_permutation = token_permutation
        MoEAlltoAllTokenDispatcher.token_unpermutation = token_unpermutation
