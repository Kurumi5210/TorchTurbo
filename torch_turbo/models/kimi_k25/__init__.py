from transformers import PretrainedConfig

from torch_turbo.distributed.parallel_plan import MoEConfig
from torch_turbo.models.registry import MODELING_REGISTRY, MOE_CONFIG_REGISTRY, PARALLEL_PLAN_REGISTRY
from torch_turbo.models.config_converter import get_base_moe_config
from torch_turbo.distributed.sequence_parallel.ulysses import patch_vlm_for_ulysses_input_slicing


from ..kimi_k2 import apply_kimi_k2_patch

@MODELING_REGISTRY.register("kimi_k25")
def register_kimi_k25_modeling(hf_config):
    from .modeling_kimi_k25 import KimiK25ForConditionalGeneration
    from ..kimi_k2.modeling_deepseek import DeepseekV3ForCausalLM

    patch_vlm_for_ulysses_input_slicing(DeepseekV3ForCausalLM)

    apply_kimi_k2_patch()

    print("Kimi-K2.5 modeling registered!")
    return KimiK25ForConditionalGeneration(hf_config)


@MOE_CONFIG_REGISTRY.register("kimi_k25")
def register_kimi_k25_moe_config(hf_config: PretrainedConfig) -> MoEConfig:
    language_config = hf_config.text_config
    kimi_k25_config_kwargs = get_base_moe_config(
        hf_config = language_config,
        num_moe_experts = language_config.n_routed_experts,
        moe_router_num_groups=language_config.n_group,
        moe_router_group_topk=language_config.topk_group,
        moe_router_topk_scaling_factor=language_config.routed_scaling_factor,
        moe_router_score_function=language_config.scoring_func,
        moe_router_dtype='fp32',
        moe_router_enable_expert_bias=True,
    )

    print("Kimi-K2.5 moe config registered!")
    return MoEConfig(**kimi_k25_config_kwargs)

@PARALLEL_PLAN_REGISTRY.register("kimi_k25")
def register_kimi_k25_parallel_plan():
    from .parallel_plan import get_parallel_plan

    print("Kimi-K2.5 parallel plan registered!")
    return get_parallel_plan()