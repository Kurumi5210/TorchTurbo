import functools
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F


def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""
    return functools.partial(torch.nn.init.normal_, mean=0.0, std=sigma)

@dataclass
class MoEConfig:
    # model architecture
    hidden_size: int = 0
    gated_linear_unit: bool = False
    activation_func: Callable = F.gelu
    num_moe_experts: Optional[int] = None
    # expert parallel
    expert_model_parallel_size: int = 1
    # training
    params_dtype: torch.dtype = torch.float32
    # initilization
    perform_initialization: bool = True
    init_method: Optional[Callable] = None
    init_method_std: float = 0.02
    # MoE related
    """Shared expert total ffn hidden size."""
    moe_shared_expert_intermediate_size: Optional[int] = None
    """MoE Feed-Forward Network hidden size"""
    moe_ffn_hidden_size: Optional[int] = None
    """The load balancing strategy for the router."""
    moe_router_load_balancing_type: str = "none"
    """Number of experts to route to for each token."""
    moe_router_topk: int = 2
    """Number of groups to divide experts into for group-limited routing."""
    moe_router_num_groups: Optional[int] = None
    """Number of selected groups for group-limited routing."""
    moe_router_group_topk: Optional[int] = None
    """Enable pre-softmax(pre-sigmoid) routing for MoE, which means softmax is before the 
    top-k selection."""
    moe_router_pre_softmax: bool = False
    """Scaling factor for routing score in top-k selection, only works when moe_router_pre_softmax
    enabled. """
    moe_router_topk_scaling_factor: Optional[float] = None
    """Score function for MoE routing. Can be "softmax" or "sigmoid"."""
    moe_router_score_function: str = "softmax"
    moe_router_dtype: Optional[str] = None
    moe_router_enable_expert_bias: bool = False
    """The type of token dispatcher to use."""
    moe_token_dispatcher_type: str = "allgather"
    """Fuse token rearrangement ops during token dispatching."""
    moe_permute_fusion: bool = False
    """Apply probs on input of experts instead of applying after activation and glu."""
    moe_apply_probs_on_input: bool = False

    def __post_init__(self):
        if self.init_method is None:
            self.init_method = init_method_normal(self.init_method_std)


@dataclass
class SPPlan:
    # sp module list
    """Can be chosen from [_flash_attention_forward, create_causal_mask, rotary]"""
    apply_modules: List[str] = None


@dataclass
class EPPlan:
    # ep module list
    """Can be chosen from [mlp.experts, mlp.shared_epxerts, token_dispatcher]"""
    apply_modules: List[str] = None


@dataclass
class FullyShardPlan:
    # fully shard module list
    apply_modules: List[str] = None


@dataclass
class ParallelPlan:
    moe_config: MoEConfig = None
    sp_plan: SPPlan = None
    ep_plan: EPPlan = None
    fsdp_plan: FullyShardPlan = None