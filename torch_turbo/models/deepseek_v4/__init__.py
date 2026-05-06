
import torch
from transformers import PretrainedConfig

import torch_turbo.models.deepseek_v4.modeling_deepseek_v4 as deepseek_v4_model
from torch_turbo.distributed.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from torch_turbo.distributed.moe.moe_layer import MoELayer
from torch_turbo.distributed.parallel_plan import MoEConfig
from torch_turbo.models.registry import MODELING_REGISTRY, MOE_CONFIG_REGISTRY, PARALLEL_PLAN_REGISTRY, PARAM_NAME_CONVERTER_REGISTRY
from torch_turbo.models.config_converter import get_base_moe_config
from torch_turbo.ops import npu_fused_ops
from torch_turbo.utils.device import is_npu_available



def apply_dsv4_patch():
    if is_npu_available:
        deepseek_v4_model.DeepseekV4RMSNorm.forward = npu_fused_ops.fused_rms_norm_forward_npu
        MoEAlltoAllTokenDispatcher.token_permutation = npu_fused_ops.fused_token_permutation_npu
        MoEAlltoAllTokenDispatcher.token_unpermutation = npu_fused_ops.fused_token_unpermutation_npu

        from .npu import npu_sfa, npu_li
        from .modeling_deepseek_v4 import SparseAttention, LiCompute
        #SparseAttention.forward = npu_sfa
        #LiCompute.forward = npu_li

    # patch moe_layer forward
    origin_moe_forward = MoELayer.forward
    def deepseek_moe_forward(self, hidden_states: torch.Tensor, input_ids = None):
        output, _ = origin_moe_forward(self, hidden_states, input_ids)
        return output
    MoELayer.forward = deepseek_moe_forward


@MODELING_REGISTRY.register("deepseek_v4")
def register_dsv4_modeling(hf_config):
    from .modeling_deepseek_v4 import DeepseekV4ForCausalLM

    apply_dsv4_patch()
    # code path of model creation with sparse attn is same as eager mode
    hf_config._attn_implementation = "eager"

    print("Deepseek V4 modeling registered!")
    model = DeepseekV4ForCausalLM(hf_config)

    # code path of model forward with sparse attn is same as FA
    model._use_flash_attention_2 = True
    model.config._attn_implementation = "flash_attention_2"
    return model



@PARAM_NAME_CONVERTER_REGISTRY.register("deepseek_v4")
def register_dsv4_name_converter(config, moe_config):
    from .dsv4_weight_conerter import DeepseekV4WeightConverter
    converter = DeepseekV4WeightConverter(config, moe_config)
    return converter

@MOE_CONFIG_REGISTRY.register("deepseek_v4")
def register_dsv4_moe_config(hf_config: PretrainedConfig) -> MoEConfig:
    dsv4_config_kwargs = get_base_moe_config(
        hf_config = hf_config,
        num_moe_experts = hf_config.n_routed_experts,
        moe_router_topk_scaling_factor=hf_config.routed_scaling_factor,
        moe_router_score_function=hf_config.scoring_func,
        moe_router_dtype='fp32',
        moe_swiglu_limit=getattr(hf_config, 'swiglu_limit', 0.0),
        vocab_size=hf_config.vocab_size,
        num_hash_layers=getattr(hf_config, 'num_hash_layers', 0.0),
        moe_router_enable_expert_bias=True,
    )

    print("Deepseek V4 moe config registered!")
    return MoEConfig(**dsv4_config_kwargs)


@PARALLEL_PLAN_REGISTRY.register("deepseek_v4")
def register_dsv4_parallel_plan():
    from .parallel_plan import get_parallel_plan

    print("Deepseek V4 parallel plan registered!")
    return get_parallel_plan()