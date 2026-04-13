import torch
from transformers import PretrainedConfig

import torch_turbo.models.glm_moe_dsa.modeling_glm_moe_dsa as glm5_model
from torch_turbo.distributed.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from torch_turbo.distributed.moe.moe_layer import MoELayer
from torch_turbo.distributed.parallel_plan import MoEConfig
from torch_turbo.models.registry import MODELING_REGISTRY, MOE_CONFIG_REGISTRY, PARALLEL_PLAN_REGISTRY
from torch_turbo.models.config_converter import get_base_moe_config
from torch_turbo.ops import npu_fused_ops
from torch_turbo.utils.device import is_npu_available


def apply_glm5_patch():
    if is_npu_available:
        glm5_model.GlmMoeDsaRMSNorm.forward = npu_fused_ops.fused_rms_norm_forward_npu
        MoEAlltoAllTokenDispatcher.token_permutation = npu_fused_ops.fused_token_permutation_npu
        MoEAlltoAllTokenDispatcher.token_unpermutation = npu_fused_ops.fused_token_unpermutation_npu

        from .npu import npu_fused_dsa_attn_forward
        from .modeling_glm_moe_dsa import GlmMoeDsaAttention
        GlmMoeDsaAttention.forward = npu_fused_dsa_attn_forward
    
    # patch moe_layer forward
    origin_moe_forward = MoELayer.forward
    def deepseek_moe_forward(self, hidden_states: torch.Tensor):
        output, _ = origin_moe_forward(self, hidden_states)
        return output
    MoELayer.forward = deepseek_moe_forward


@MODELING_REGISTRY.register("glm_moe_dsa")
def register_glm5_modeling(hf_config):
    from .modeling_glm_moe_dsa import GlmMoeDsaForCausalLM

    apply_glm5_patch()
    # code path of model creation with sparse attn is same as eager mode
    hf_config._attn_implementation = "eager"

    print("GLM5 modeling registered!")
    model = GlmMoeDsaForCausalLM(hf_config)

    # code path of model forward with sparse attn is same as FA
    model.model._use_flash_attention_2 = True
    model.config._attn_implementation = "flash_attention_2"
    return model


@MOE_CONFIG_REGISTRY.register("glm_moe_dsa")
def register_glm5_moe_config(hf_config: PretrainedConfig) -> MoEConfig:
    glm5_config_kwargs = get_base_moe_config(
        hf_config = hf_config,
        num_moe_experts = hf_config.n_routed_experts,
        moe_router_num_groups=hf_config.n_group,
        moe_router_group_topk=hf_config.topk_group,
        moe_router_topk_scaling_factor=hf_config.routed_scaling_factor,
        moe_router_score_function=hf_config.scoring_func,
        moe_router_dtype='fp32',
        moe_router_enable_expert_bias=True,
    )

    print("GLM5 moe config registered!")
    return MoEConfig(**glm5_config_kwargs)


@PARALLEL_PLAN_REGISTRY.register("glm_moe_dsa")
def register_glm5_parallel_plan():
    from .parallel_plan import get_parallel_plan

    print("GLM5 parallel plan registered!")
    return get_parallel_plan()