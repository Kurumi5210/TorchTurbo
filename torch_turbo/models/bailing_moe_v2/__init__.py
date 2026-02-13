from torch_turbo.models.registry import MODELING_REGISTRY, PARALLEL_CONFIG_REGISTRY


@MODELING_REGISTRY.register("bailing_moe")
def register_bailing_moe_v2_modeling():
    from .modeling_bailing_moe_v2 import BailingMoeV2ForCausalLM
    return BailingMoeV2ForCausalLM

@PARALLEL_CONFIG_REGISTRY.register("bailing_moe")
def register_bailing_moe_v2_parallel_config():
    from .parallel_config import get_parallel_config
    return get_parallel_config()