from typing import List, Optional, Tuple

import torch

from torch_npu import npu_rms_norm

from torchturbo.models.modules.comm import all_to_all
from torchturbo.models.modules.moe.utils import sort_chunks_by_idxs
from mindspeed.ops.npu_rotary_position_embedding import npu_rotary_position_embedding
from mindspeed.ops.npu_moe_token_permute import npu_moe_token_permute
from mindspeed.ops.npu_moe_token_unpermute import npu_moe_token_unpermute


def apply_rotary_pos_emb_npu(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = npu_rotary_position_embedding(q, cos, sin, 0)
    k_embed = npu_rotary_position_embedding(k, cos, sin, 0)
    return q_embed, k_embed


def apply_rotary_pos_emb_bailing_npu(
    q, k, cos, sin, position_ids=None, unsqueeze_dim=1
):
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


def apply_rotary_pos_emb_kimi_moe(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    q_embed = npu_rotary_position_embedding(q, cos, sin, 0)
    k_embed = npu_rotary_position_embedding(k, cos, sin, 0)
    return q_embed, k_embed


def fused_rms_norm_forward_npu(self, hidden_states: torch.Tensor):
    return npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]


def fused_permute(
    tokens,
    routing_map,
    probs: Optional[torch.Tensor] = None,
    num_out_tokens: Optional[int] = None,
    fused: bool = False,
):
    expert_indices = torch.nonzero(routing_map, as_tuple=True)[1].view(
        tokens.shape[0], -1
    )
    permuted_tokens, sorted_indices = npu_moe_token_permute(tokens, expert_indices)
    return permuted_tokens, sorted_indices


def fused_token_permutation_npu(
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
    ) = fused_permute(
        hidden_states,
        routing_map,
        probs=probs,
        num_out_tokens=self.num_out_tokens,
        fused=self.config.moe_permute_fusion,
    )

    # Perform expert parallel AlltoAll communication
    tokens_per_expert = self._maybe_dtoh_and_synchronize(
        "before_ep_alltoall", tokens_per_expert
    )
    global_input_tokens = all_to_all(
        self.ep_group,
        permutated_local_input_tokens,
        self.output_splits,
        self.input_splits,
    )

    # Permutation 2: Sort tokens by local expert.
    tokens_per_expert = self._maybe_dtoh_and_synchronize(
        "before_permutation_2", tokens_per_expert
    )
    if self.num_local_experts > 1:
        global_input_tokens, global_probs = sort_chunks_by_idxs(
            global_input_tokens,
            self.num_global_tokens_per_local_expert.ravel(),
            self.sort_input_by_local_experts,
            probs=None,
            fused=self.config.moe_permute_fusion,
        )

    tokens_per_expert = self._maybe_dtoh_and_synchronize(
        "before_finish", tokens_per_expert
    )

    return global_input_tokens, tokens_per_expert, probs


def fused_unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
    probs: torch.Tensor = None,
    routing_map: torch.Tensor = None,
    fused: bool = False,
):
    # topk = int(routing_map[0].sum(dim=0).item())
    # # If probs is not passed in, the npu_moe_token_unpermute will consider topk to be 1, resulting in an incorrect outcome
    # # link: https://gitee.com/ascend/MindSpeed/blob/1.0.0_core_r0.6.0/docs/ops/npu_fused_moe_token_unpermute.md
    # fake_probs = (torch.ones(routing_map.shape[0], topk, device=permuted_tokens.device, dtype=permuted_tokens.dtype))
    unpermuted_tokens = npu_moe_token_unpermute(
        permuted_tokens, sorted_indices, probs=probs
    )
    return unpermuted_tokens


def fused_token_unpermutation_npu(
    self,
    hidden_states: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    probs: Optional[torch.Tensor] = None,
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
    probs = probs.to(torch.bfloat16)
    output = fused_unpermute(
        permutated_local_input_tokens,
        self.reversed_local_input_permutation_mapping,
        restore_shape=self.hidden_shape_before_permute,
        routing_map=self.routing_map,
        fused=self.config.moe_permute_fusion,
        probs=probs,
    )

    # Reshape the output tensor
    output = output.view(self.hidden_shape)

    # release cached tensors
    self.probs = None
    self.routing_map = None

    return output, None
