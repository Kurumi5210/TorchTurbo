import torch.nn.functional as F
from transformers import PretrainedConfig

from torch_turbo.distributed.parallel_config import MoEConfig
from torch_turbo.distributed.parallel_state import get_expert_model_parallel_world_size


def _get_base_moe_config(hf_config: PretrainedConfig, **override_config_kwargs) -> dict:
    ep_size = get_expert_model_parallel_world_size()
    base_config_kwargs = dict(
        # model arch
        hidden_size=hf_config.moe_intermediate_size,
        gated_linear_unit=True,
        activation_func=F.silu,
        # parallel
        expert_model_parallel_size=ep_size,
        # moe related params
        moe_ffn_hidden_size=hf_config.moe_intermediate_size,
        moe_router_load_balancing_type="none",
        moe_router_topk=hf_config.num_experts_per_tok,
        moe_router_pre_softmax=True,
        moe_token_dispatcher_type="alltoall",
    )
    base_config_kwargs.update(**override_config_kwargs)
    return base_config_kwargs

def hf_to_moe_config_qwen3(hf_config: PretrainedConfig) -> MoEConfig:
    qwen3_config_kwargs = _get_base_moe_config(
        hf_config = hf_config,
        num_moe_experts = hf_config.num_experts,
    )
    return MoEConfig(**qwen3_config_kwargs)

def hf_to_moe_config_bailing(hf_config: PretrainedConfig) -> MoEConfig:
    args = _get_base_moe_config(
        hf_config = hf_config,
        num_moe_experts = hf_config.num_experts,
        moe_router_num_groups=hf_config.n_group,
        moe_router_group_topk=hf_config.topk_group,
        moe_router_topk_scaling_factor=hf_config.routed_scaling_factor,
        moe_router_score_function=hf_config.score_function,
        moe_router_dtype=hf_config.router_dtype,
        moe_router_enable_expert_bias=hf_config.moe_router_enable_expert_bias,
    )
    return MoEConfig(args)