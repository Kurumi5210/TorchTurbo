from megatron.core import parallel_state, tensor_parallel
import torch
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from typing import List, Optional

def moe_forward(self, hidden_states: torch.Tensor):
    if (
        self.training
        and self.config.tensor_model_parallel_size > 1
        and not self.config.sequence_parallel
    ):
        raise ValueError(
            "During training, performance may degrade if MoE and tensor parallelism"
            "are enabled without also enabling sequence parallelism."
        )

    # process MoE
    def custom_forward(hidden_states):
        probs, routing_map, logits = self.router(hidden_states)
        logits = logits.view(-1, logits.shape[-1])
        (dispatched_input, tokens_per_expert, permuted_probs) = (
            self.token_dispatcher.token_permutation(hidden_states, probs, routing_map)
        )
        expert_output, mlp_bias = self.experts(
            dispatched_input, tokens_per_expert, permuted_probs
        )
        output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
        if self.use_shared_expert and not self.shared_expert_overlap:
            # if shared_expert_overlap is True, the expert calculation happens in
            # the token_dispatcher to overlap communications and computations
            output = output + self.shared_experts(hidden_states)
        return output, mlp_bias, logits

    if self.moe_layer_recompute:
        output, mlp_bias, logits = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
    else:
        output, mlp_bias, logits = custom_forward(hidden_states)

    return output, logits

def router_forward(self, input: torch.Tensor):
    """
    Forward pass of the router.

    Args:
        input (torch.Tensor): Input tensor.
    """
    self._maintain_float32_expert_bias()

    # Apply input jitter
    input = self.apply_input_jitter(input)
    logits = self.gating(input)

    scores, routing_map = self.routing(logits)

    return scores, routing_map, logits

def moe_token_dispatcher_all_to_all_init(self, num_local_experts: int, local_expert_indices: List[int], config: TransformerConfig) -> None:
        """
        Initialize the AlltoAll token dispatcher.

        Args:
            num_local_experts (int): Number of local experts on the current device.
            local_expert_indices (List[int]): Indices of local experts on the current device.
            config (TransformerConfig): Configuration for the transformer model.
        """
        self.config = config
        self.shared_experts: Optional[SharedExpertMLP] = None

        self.tp_size = config.expert_tensor_parallel_size or 1
        self.ep_size = config.expert_model_parallel_size or 1
        self.num_local_experts = num_local_experts
        assert config.num_moe_experts is not None
        self.num_experts = config.num_moe_experts
        assert self.num_local_experts > 0, "Expected at least one expert"
        self.local_expert_indices = local_expert_indices
        assert (
            len(self.local_expert_indices) == self.num_local_experts
        ), "Invalid local expert indices"
        for i in range(len(self.local_expert_indices) - 1):
            assert (
                self.local_expert_indices[i] == self.local_expert_indices[i + 1] - 1
            ), "local_expert_indices must be continous"

        # [ep_size]. Represents the number of tokens sent by the current rank to other
        # EP ranks.
        self.input_splits = None
        # [ep_size]. Represents the number of tokens received by the current rank from
        # other EP ranks.
        self.output_splits = None
        # [tp_size]. Represents the number of tokens received by the current rank from
        # other TP ranks.
        self.output_splits_tp = None
        # self.permute_idx_device = torch.device("cuda") if self.config.moe_permute_fusion else None
        self.permute_idx_device = torch.device("cpu")
        input_chunk_idxs = torch.arange(
            self.num_experts * self.tp_size, device=self.permute_idx_device
        )
        # [num_local_experts, tp_size * ep_size]. Sort the input chunks by local experts.
        # print(input_chunk_idxs.device)
        assert not input_chunk_idxs.is_meta, f"Parameter is a meta tensor, which is not expected."
        self.sort_input_by_local_experts = input_chunk_idxs.reshape(
            -1, self.num_local_experts
        ).T.ravel()
        # [tp_size * ep_size, num_local_experts]. Restore the output chunks by local experts.
        self.restore_output_by_local_experts = input_chunk_idxs.reshape(
            self.num_local_experts, -1
        ).T.ravel()

        # Token drop and padding.
        # Drop and pad the input to capacity.
        self.drop_and_pad = self.config.moe_pad_expert_input_to_capacity
        if self.drop_and_pad:
            assert self.config.moe_expert_capacity_factor is not None
            self.moe_expert_capacity_factor = self.config.moe_expert_capacity_factor
        self.capacity = None

        # A cuda stream synchronization is needed in self.token_permutation() in some cases,
        # because there are several non-blocking DtoH data transfers called at
        # `self.cuda_dtoh_point`. The synchronization happens at `self.cuda_sync_point`, which is
        # decided based on the MoE and parallel settings. Valid points are "before_permutation_1",
        # "before_ep_alltoall", "before_permutation_2", "before_finish", and "no_sync".
        self.cuda_sync_point = "no_sync"
        self.cuda_sync_point_priority = {
            "before_permutation_1": 0,
            "before_ep_alltoall": 1,
            "before_permutation_2": 2,
            "before_finish": 3,
            "no_sync": 4,
        }
        self.cuda_dtoh_point = "before_permutation_1"
        self.cuda_dtoh_stream = torch.cuda.Stream()

        self.shared_experts = None
        
def apply_monkey_patch():
    """
    Apply the monkey patch to the MoE layer.
    """
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.moe.router import TopKRouter
    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
    
    TopKRouter.forward = router_forward     
    MoELayer.forward = moe_forward
    MoEAlltoAllTokenDispatcher.__init__ = moe_token_dispatcher_all_to_all_init