import torch
from transformers import PretrainedConfig

import torch_turbo.models.kimi_k2.modeling_deepseek as kimi_k2_model
from torch_turbo.distributed.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from torch_turbo.distributed.moe.moe_layer import MoELayer
from torch_turbo.distributed.parallel_plan import MoEConfig
from torch_turbo.models.registry import MODELING_REGISTRY, MOE_CONFIG_REGISTRY, PARALLEL_PLAN_REGISTRY
from torch_turbo.models.config_converter import get_base_moe_config
from torch_turbo.ops import npu_fused_ops
from torch_turbo.utils.device import is_npu_available


def deepseek_moe_forward(self, hidden_states: torch.Tensor):
    output, _ = MoELayer.forward(hidden_states)
    return output


def apply_kimi_k2_patch():
    if is_npu_available:
        kimi_k2_model.apply_rotary_pos_emb = npu_fused_ops.apply_rotary_pos_emb_kimi_moe
        kimi_k2_model.DeepseekV3RMSNorm.forward = npu_fused_ops.fused_rms_norm_forward_npu
        MoEAlltoAllTokenDispatcher.token_permutation = npu_fused_ops.fused_token_permutation_npu
        MoEAlltoAllTokenDispatcher.token_unpermutation = npu_fused_ops.fused_token_unpermutation_npu
    
    # patch moe_layer forward
    MoELayer.forward = deepseek_moe_forward


@MODELING_REGISTRY.register("kimi_k2")
def register_kimi_k2_modeling(hf_config):
    from .modeling_deepseek import DeepseekV3ForCausalLM

    apply_kimi_k2_patch()

    print("Kimi-K2 modeling registered!")
    return DeepseekV3ForCausalLM(hf_config)


@MOE_CONFIG_REGISTRY.register("kimi_k2")
def register_kimi_k2_moe_config(hf_config: PretrainedConfig) -> MoEConfig:
    kimi_k2_config_kwargs = get_base_moe_config(
        hf_config = hf_config,
        num_moe_experts = hf_config.n_routed_experts,
        moe_router_num_groups=hf_config.n_group,
        moe_router_group_topk=hf_config.topk_group,
        moe_router_topk_scaling_factor=hf_config.routed_scaling_factor,
        moe_router_score_function=hf_config.scoring_func,
        moe_router_dtype='fp32',
        moe_router_enable_expert_bias=True,
    )

    print("Kimi-K2 moe config registered!")
    return MoEConfig(**kimi_k2_config_kwargs)


@PARALLEL_PLAN_REGISTRY.register("kimi_k2")
def register_kimi_k2_parallel_plan():
    from .parallel_plan import get_parallel_plan

    print("Kimi-K2 parallel plan registered!")
    return get_parallel_plan()