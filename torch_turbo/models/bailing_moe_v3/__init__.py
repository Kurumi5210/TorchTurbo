from transformers import PretrainedConfig

from torch_turbo.distributed.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from torch_turbo.distributed.parallel_plan import MoEConfig
from torch_turbo.models.registry import MODELING_REGISTRY, MOE_CONFIG_REGISTRY, PARALLEL_PLAN_REGISTRY, PARAM_NAME_CONVERTER_REGISTRY
from torch_turbo.models.name_converter import BaseWeightConverter
from torch_turbo.models.config_converter import get_base_moe_config
from torch_turbo.ops import npu_fused_ops
from torch_turbo.utils.device import is_npu_available


class BailingMoeV3WeightConverter(BaseWeightConverter):
    def get_hf_gate_bias_name(self):
        return "gate.expert_bias"


def apply_bailing_moe_v3_patch():
    import torch_turbo.models.bailing_moe_v3.modeling_bailing_moe_v3 as bailing_moe_v3_model

    if is_npu_available:
        bailing_moe_v3_model.apply_rotary_pos_emb = npu_fused_ops.apply_rotary_pos_emb_bailing_npu
        # To further improve accuracy
        # bailing_moe_v3_model.BailingMoeV3RMSNorm.forward = npu_fused_ops.fused_rms_norm_forward_npu
        MoEAlltoAllTokenDispatcher.token_permutation = npu_fused_ops.fused_token_permutation_npu
        MoEAlltoAllTokenDispatcher.token_unpermutation = npu_fused_ops.fused_token_unpermutation_npu


@MODELING_REGISTRY.register("bailing_hybrid")
def register_bailing_moe_v3_modeling(hf_config):
    from .modeling_bailing_moe_v3 import BailingMoeV3ForCausalLM

    apply_bailing_moe_v3_patch()

    print("BailingMoEV3 modeling registered!")
    return BailingMoeV3ForCausalLM(hf_config)


@MOE_CONFIG_REGISTRY.register("bailing_hybrid")
def register_bailing_v3_moe_config(hf_config: PretrainedConfig) -> MoEConfig:
    bailing_moe_v3_config_kwargs = get_base_moe_config(
        hf_config = hf_config,
        num_moe_experts = hf_config.num_experts,
        moe_router_num_groups=hf_config.n_group,
        moe_router_group_topk=hf_config.topk_group,
        moe_router_topk_scaling_factor=hf_config.routed_scaling_factor,
        moe_router_score_function=hf_config.score_function,
        moe_router_dtype=hf_config.router_dtype,
        moe_router_enable_expert_bias=hf_config.moe_router_enable_expert_bias,
    )
    return MoEConfig(**bailing_moe_v3_config_kwargs)


@PARALLEL_PLAN_REGISTRY.register("bailing_hybrid")
def register_bailing_moe_v3_parallel_plan():
    from .parallel_plan import get_parallel_plan
    return get_parallel_plan()


@PARAM_NAME_CONVERTER_REGISTRY.register("bailing_hybrid")
def register_bailing_moe_v3_param_converter(hf_config, moe_config):
    return BailingMoeV3WeightConverter(hf_config, moe_config)