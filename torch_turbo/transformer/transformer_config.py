# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import enum
import functools
import math
import warnings
from dataclasses import dataclass
from typing import Callable, ContextManager, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""
    return functools.partial(torch.nn.init.normal_, mean=0.0, std=sigma)


@dataclass
class ModelParallelConfig:
    """Base configuration for Megatron Core

    The initialization function has an argument for each parameter.
    """

    ###################
    # Model parallelism
    ###################
    expert_model_parallel_size: int = 1
    """Distributes Moe Experts across sub data parallel dimension."""

    ###################
    # Initialization
    ###################
    perform_initialization: bool = True
    """If true, weights are initialized. This option can be useful when you know you are going to
       load values from a checkpoint.
    """

    ###################
    # Training
    ###################
    params_dtype: torch.dtype = torch.float32
    """dtype used when intializing the weights."""


@dataclass
class TransformerConfig(ModelParallelConfig):
    """Configuration object for megatron-core transformers.

    The initialization function has an argument for each parameter,
    including those in ModelParallelConfig.
    """

    ####################
    # model architecture
    ####################

    hidden_size: int = 0
    """Transformer hidden size."""

    num_attention_heads: int = 0
    """Number of transformer attention heads."""

    ffn_hidden_size: Optional[int] = None
    """Transformer Feed-Forward Network hidden size. This is set to 4*hidden_size
    if not provided."""

    gated_linear_unit: bool = False
    """Use a gated linear unit for the first linear layer in the MLP."""

    activation_func: Callable = F.gelu
    """Activation function to use for the non-linearity in the MLP."""

    num_moe_experts: Optional[int] = None
    """Number of experts to use for MoE layer. When set, it replaces MLP with MoE layer. Set to None
    for no MoE."""

    ####################
    # initialization
    ####################
    init_method: Optional[Callable] = None
    """Method to initialize weights. Note that bias is always set to zero. Should be a function that
    takes a single Tensor and initializes it. If None, will be set to
    megatron.core.utils.init_method_normal(init_method_std) which is torch nn init normal with
    mean=0.0 and std=init_method_std."""

    init_method_std: float = 0.02
    """Standard deviation of the zero mean normal for the default initialization method, not used if
    init_method and output_layer_init_method are provided."""

    ####################
    # MoE related
    ####################
    moe_shared_expert_intermediate_size: Optional[int] = None
    """Shared expert total ffn hidden size.
    It should be equal to 'num_shared_experts * ffn_size_of_each_shared_expert' if
    there are multiple shared experts.
    None means no shared expert."""

    moe_ffn_hidden_size: Optional[int] = None
    """MoE Feed-Forward Network hidden size"""

    moe_router_load_balancing_type: str = "aux_loss"
    """The load balancing strategy for the router. "aux_loss" corresponds to the load balancing loss
    used in GShard and SwitchTransformer; "seq_aux_loss" corresponds to the load balancing loss used
    in DeepSeekV2 and DeepSeekV3, which computes the loss for each individual sample; "sinkhorn"
    corresponds to the balancing algorithm used in S-BASE, and "none" implies no load balancing.
    The default is "aux_loss"."""

    moe_router_topk: int = 2
    """Number of experts to route to for each token."""

    moe_router_num_groups: Optional[int] = None
    """Number of groups to divide experts into for group-limited routing.
    When using group-limited routing:
    1. Experts are divided into 'moe_router_num_groups' equal-sized groups
    2. For each token, 'moe_router_group_topk' groups are selected based on sum of
    top-('moe_router_topk'/'moe_router_group_topk') routing scores within each group
    3. From these selected groups, 'moe_router_topk' individual experts are chosen
    Two common use cases:
    - Device-limited routing: Set 'moe_router_num_groups' equal to expert parallel size (EP)
    to limit each token to experts on a subset of devices
    (See DeepSeek-V2: https://arxiv.org/pdf/2405.04434)
    - Node-limited routing: Set 'moe_router_num_groups' equal to number of nodes in EP group
    to limit each token to experts on a subset of nodes
    (See DeepSeek-V3: https://arxiv.org/pdf/2412.19437)
    """

    moe_router_group_topk: Optional[int] = None
    """Number of selected groups for group-limited routing."""

    moe_router_pre_softmax: bool = False
    """Enable pre-softmax(pre-sigmoid) routing for MoE, which means softmax is before the 
    top-k selection.
    By default, softmax is done after top-k."""

    moe_router_topk_scaling_factor: Optional[float] = None
    """Scaling factor for routing score in top-k selection, only works when moe_router_pre_softmax
    enabled. Defaults to None, which means no scaling."""

    moe_router_score_function: str = "softmax"
    """Score function for MoE routing. Can be "softmax" or "sigmoid"."""

    moe_router_dtype: Optional[str] = None
    """Data type for routing and expert output weighted averaging. Using fp32 or fp64 can
    improve stability especially when the number of experts is large (e.g. finegrained-moe).
    None means no changes for dtype."""

    moe_router_enable_expert_bias: bool = False
    """TopK routing with dynamic per-expert bias in the aux-loss-free load balancing strategy.
    The routing decision is based on the sum of the routing scores and the expert bias.
    See https://arxiv.org/abs/2408.15664 for details."""

    moe_grouped_gemm: bool = False
    """When there are multiple experts per rank, compress multiple local (potentially small) gemms
    in a single kernel launch to improve the utilization and performance by leveraging the Grouped
    GEMM feature introduced since CUTLASS 2.8 (https://github.com/fanshiqing/grouped_gemm).
    """

    moe_use_legacy_grouped_gemm: bool = False
    """Use legacy GroupedMLP rather than TEGroupedMLP.
    Note: The legacy one will be deprecated soon."""

    moe_token_dispatcher_type: str = "allgather"
    """The type of token dispatcher to use. The default is 'allgather'.
    Options are 'allgather','alltoall' and 'flex'."""

    moe_permute_fusion: bool = False
    """Fuse token rearrangement ops during token dispatching."""

    moe_apply_probs_on_input: bool = False
    """Apply probs on input of experts instead of applying after activation and glu."""

    def __post_init__(self):
        """Python dataclass method that is used to modify attributes after initialization.
        See https://docs.python.org/3/library/dataclasses.html#post-init-processing for more
        details.
        """
        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = 4 * self.hidden_size

        if self.expert_model_parallel_size > 1 and self.num_moe_experts is None:
            raise ValueError('num_moe_experts must be non None to use expert-parallel.')

        if self.num_moe_experts is not None and self.num_moe_experts <= 0:
            raise ValueError('num_moe_experts must be non-negative.')

        if self.moe_ffn_hidden_size is None:
            self.moe_ffn_hidden_size = self.ffn_hidden_size

        if self.moe_shared_expert_intermediate_size is not None:
            if self.moe_shared_expert_intermediate_size <= 0:
                raise ValueError(
                    f'moe_shared_expert_intermediate_size must be '
                    f'num_shared_experts * ffn_size_of_each_shared_expert, '
                    f'but got {self.moe_shared_expert_intermediate_size}'
                )

        if self.init_method is None:
            self.init_method = init_method_normal(self.init_method_std)

        if self.moe_router_enable_expert_bias and self.moe_router_score_function != "sigmoid":
            raise ValueError(
                "Expert bias for aux-loss-free routing only supports sigmoid score function."
                "Please set --moe-router-score-function sigmoid for sigmoid score function."
            )

        if (
            self.moe_router_topk == 1
            and self.moe_router_score_function == 'softmax'
            and not self.moe_router_pre_softmax
            and self.moe_router_load_balancing_type != 'sinkhorn'
        ):
            # Requires applying softmax before selecting the top-k when k is 1,
            # since softmax on a [num_tokens, 1] would yield a zero gradient.
            raise ValueError("Please use --moe-router-pre-softmax when topk is 1.")

        if self.moe_router_group_topk:
            if not self.moe_router_num_groups:
                raise ValueError(
                    "When using group limited routing, moe_router_num_groups must be specified."
                )
            else:
                assert self.num_moe_experts % self.moe_router_num_groups == 0, (
                    f"num_moe_experts ({self.num_moe_experts}) should be divisible by "
                    f"moe_router_num_groups ({self.moe_router_num_groups})."
                )
                assert self.moe_router_group_topk <= self.moe_router_num_groups, (
                    f"moe_router_group_topk ({self.moe_router_group_topk}) should be smaller than "
                    f"moe_router_num_groups ({self.moe_router_num_groups})."
                )

        if (
            self.num_moe_experts is not None
            and self.num_moe_experts >= 32
            and not self.moe_router_dtype
        ):
            warnings.warn(
                "Using a large number of experts (e.g. >=32) without fp32 routing. "
                "Consider enabling moe_router_dtype for better numerical stability."
            )