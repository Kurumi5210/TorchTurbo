from transformers import PretrainedConfig

import torch_turbo.models.bailing_moe_v2.modeling_bailing_moe_v2 as bailing_moe_v2_model
from torch_turbo.distributed.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from torch_turbo.distributed.parallel_plan import MoEConfig
from torch_turbo.models.registry import MODELING_REGISTRY, MOE_CONFIG_REGISTRY, PARALLEL_PLAN_REGISTRY
from torch_turbo.models.config_converter import get_base_moe_config
from torch_turbo.ops import npu_fused_ops
from torch_turbo.utils.device import is_npu_available


def apply_bailing_moe_v2_patch():
    if is_npu_available:
        bailing_moe_v2_model.apply_rotary_pos_emb = npu_fused_ops.apply_rotary_pos_emb_bailing_npu
        bailing_moe_v2_model.BailingMoeV2RMSNorm.forward = npu_fused_ops.fused_rms_norm_forward_npu
        MoEAlltoAllTokenDispatcher.token_permutation = npu_fused_ops.fused_token_permutation_npu
        MoEAlltoAllTokenDispatcher.token_unpermutation = npu_fused_ops.fused_token_unpermutation_npu


@MODELING_REGISTRY.register("bailing_moe_v2")
def register_bailing_moe_v2_modeling():
    from .modeling_bailing_moe_v2 import BailingMoeV2ForCausalLM

    apply_bailing_moe_v2_patch()

    return BailingMoeV2ForCausalLM


@MOE_CONFIG_REGISTRY.register("bailing_moe_v2")
def register_bailing_v2_moe_config(hf_config: PretrainedConfig) -> MoEConfig:
    bailing_moe_v2_config_kwargs = get_base_moe_config(
        hf_config = hf_config,
        num_moe_experts = hf_config.num_experts,
        moe_router_num_groups=hf_config.n_group,
        moe_router_group_topk=hf_config.topk_group,
        moe_router_topk_scaling_factor=hf_config.routed_scaling_factor,
        moe_router_score_function=hf_config.score_function,
        moe_router_dtype=hf_config.router_dtype,
        moe_router_enable_expert_bias=hf_config.moe_router_enable_expert_bias,
    )
    return MoEConfig(**bailing_moe_v2_config_kwargs)


@PARALLEL_PLAN_REGISTRY.register("bailing_moe_v2")
def register_bailing_moe_v2_parallel_plan():
    from .parallel_plan import get_parallel_plan
    return get_parallel_plan()