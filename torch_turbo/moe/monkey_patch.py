from megatron.core import parallel_state, tensor_parallel
import torch
import os

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

    # debug
    rank = torch.distributed.get_rank()
    # process MoE
    def custom_forward(hidden_states):
        probs, routing_map, logits = self.router(hidden_states)
        probs /= probs.sum(dim=-1, keepdim=True)
        logits = logits.view(-1, logits.shape[-1])
        (dispatched_input, tokens_per_expert, permuted_probs) = (
            self.token_dispatcher.token_permutation(hidden_states, probs, routing_map)
        )
        expert_output, mlp_bias = self.experts(
            dispatched_input, tokens_per_expert, permuted_probs
        )
        expert_output = expert_output * permuted_probs.unsqueeze(-1)
        output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
        if self.use_shared_expert and not self.shared_expert_overlap:
            # if shared_expert_overlap is True, the expert calculation happens in
            # the token_dispatcher to overlap communications and computations
            output = output + self.shared_experts(hidden_states)
        
        if rank == 0 and os.environ.get('TORCH_TURBO_DEBUG', False):
            print(f"hidden_states.shape {hidden_states.shape} hidden_states {hidden_states}")
            print(f"dispatched_input.shape {dispatched_input.shape} dispatched_input {dispatched_input}")
            print(f"tokens_per_expert.shape {tokens_per_expert.shape} tokens_per_expert {tokens_per_expert}")
            print(f"permuted_probs.shape {permuted_probs.shape} permuted_probs {permuted_probs}")
            print(f"before unpermutation: expert_output.shape {expert_output.shape} expert_output {expert_output}")
            print(f"after unpermutation: expert_output.shape {output.shape} expert_output {output}")
            
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
        
def apply_monkey_patch():
    """
    Apply the monkey patch to the MoE layer.
    """
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.moe.router import TopKRouter
    
    TopKRouter.forward = router_forward     
    MoELayer.forward = moe_forward