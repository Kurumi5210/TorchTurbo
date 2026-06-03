from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .moe_config import MoEConfig
from .router import TopKRouter
from .token_dispatcher import MoETokenDispatcher, MoEAlltoAllTokenDispatcher
from .module import ModuleSpec, build_module
from torchturbo.utils.device import is_npu_available


@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None


class BaseMoELayer(torch.nn.Module, ABC):
    """Base class for a mixture of experts layer."""

    def __init__(
        self,
        config: MoEConfig,
        ep_group: dist.ProcessGroup,
        layer_number: Optional[int] = None,
    ):
        super().__init__()

        ep_size = dist.get_world_size(ep_group)
        ep_rank = dist.get_rank(ep_group)

        self.config = config
        self.expert_parallel_size = ep_size
        assert (
            self.expert_parallel_size > 0
        ), "Expected non-negative expert parallel size"

        assert self.config.num_moe_experts % self.expert_parallel_size == 0
        self.num_local_experts = (
            self.config.num_moe_experts // self.expert_parallel_size
        )
        local_expert_indices_offset = ep_rank * self.num_local_experts

        self.use_shared_expert = (
            self.config.moe_shared_expert_intermediate_size is not None
        )

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(
            map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices)
        )
        self.router: TopKRouter = None
        self.experts = None
        self.shared_experts = None
        self.token_dispatcher: Optional[MoETokenDispatcher] = None
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        """Forward method for the MoE layer."""
        pass

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer."""
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)


class MoELayer(BaseMoELayer):
    """
    Mixture of experts Layer **currently only supports no token dropping**.
    """

    def __init__(
        self,
        config: MoEConfig,
        ep_group: dist.ProcessGroup,
        submodules: Optional[MoESubmodules] = None,
        use_fused_swiglu: Optional[bool] = False,
        use_fused_permute_unpermute: Optional[bool] = False,
        use_te: Optional[bool] = False,
        layer_number: Optional[int] = None,
    ):
        self.use_fused_swiglu = use_fused_swiglu
        self.use_fused_permute_unpermute = use_fused_permute_unpermute
        self.use_te = use_te
        self.submodules = submodules
        super(MoELayer, self).__init__(
            config=config, ep_group=ep_group, layer_number=layer_number
        )

        # Initialize router
        self.router = TopKRouter(config=self.config)

        # Initialize token dispatcher
        if config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                ep_group=ep_group,
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )

        # Initialize experts
        self.experts = build_module(
            self.submodules.experts,
            self.num_local_experts,
            self.use_fused_swiglu,
            self.use_te,
            self.config,
            layer_number=self.layer_number,
        )

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = build_module(
                self.submodules.shared_experts, config=self.config
            )

    def forward(self, hidden_states: torch.Tensor):
        # debug
        rank = torch.distributed.get_rank()

        # process MoE
        def custom_forward(hidden_states):
            hidden_states_dtype = hidden_states.dtype
            probs, routing_map, logits = self.router(hidden_states)
            if self.config.moe_router_score_function != "sigmoid":
                probs = probs / probs.sum(dim=-1, keepdim=True)
            logits = logits.view(-1, logits.shape[-1])
            (dispatched_input, tokens_per_expert, permuted_probs) = (
                self.token_dispatcher.token_permutation(
                    hidden_states, probs, routing_map
                )
            )
            expert_output, mlp_bias = self.experts(
                dispatched_input, tokens_per_expert, permuted_probs
            )

            if is_npu_available:
                # apply shape transform for npu fused kernel
                num_tokens, _ = probs.shape
                probs = probs.masked_select(routing_map).view(num_tokens, -1)
                output, mlp_bias = self.token_dispatcher.token_unpermutation(
                    expert_output, mlp_bias, probs
                )
            else:
                expert_output = expert_output * permuted_probs.unsqueeze(-1)
                output, mlp_bias = self.token_dispatcher.token_unpermutation(
                    expert_output, mlp_bias
                )
            output = output.to(hidden_states_dtype)

            if hasattr(self, "shared_expert") and self.shared_expert is not None:
                shared_expert_output = self.shared_expert(hidden_states)
                shared_expert_output = (
                    F.sigmoid(self.shared_expert_gate(hidden_states))
                    * shared_expert_output
                )
                output = output + shared_expert_output

            if hasattr(self, "shared_experts") and self.shared_experts is not None:
                shared_expert_output = self.shared_experts(hidden_states)
                output = output + shared_expert_output

            if rank == 0 and os.environ.get("TORCH_TURBO_DEBUG", False):
                print(
                    f"hidden_states.shape {hidden_states.shape} hidden_states {hidden_states}"
                )
                print(
                    f"dispatched_input.shape {dispatched_input.shape} dispatched_input {dispatched_input}"
                )
                print(
                    f"tokens_per_expert.shape {tokens_per_expert.shape} tokens_per_expert {tokens_per_expert}"
                )
                print(
                    f"permuted_probs.shape {permuted_probs.shape} permuted_probs {permuted_probs}"
                )
                print(
                    f"before unpermutation: expert_output.shape {expert_output.shape} expert_output {expert_output}"
                )
                print(
                    f"after unpermutation: expert_output.shape {output.shape} expert_output {output}"
                )

            return output, mlp_bias, logits

        output, mlp_bias, logits = custom_forward(hidden_states)
        return output, logits
