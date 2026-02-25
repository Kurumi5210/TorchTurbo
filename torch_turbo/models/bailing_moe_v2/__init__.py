import torch_turbo.models.bailing_moe_v2.modeling_bailing_moe_v2 as bailing_moe_v2_model
from torch_turbo.distributed.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from torch_turbo.models.registry import MODELING_REGISTRY, PARALLEL_CONFIG_REGISTRY
from torch_turbo.ops import npu_fused_ops
from torch_turbo.utils.device import is_npu_available


def apply_bailing_moe_v2_patch():
    if is_npu_available:
        bailing_moe_v2_model.apply_rotary_pos_emb = npu_fused_ops.apply_rotary_pos_emb_bailing_npu
        bailing_moe_v2_model.BailingMoeV2RMSNorm.forward = npu_fused_ops.fused_rms_norm_forward_npu
        MoEAlltoAllTokenDispatcher.token_permutation = npu_fused_ops.fused_token_permutation_npu
        MoEAlltoAllTokenDispatcher.token_unpermutation = npu_fused_ops.fused_token_unpermutation_npu


@MODELING_REGISTRY.register("bailing_moe")
def register_bailing_moe_v2_modeling():
    from .modeling_bailing_moe_v2 import BailingMoeV2ForCausalLM

    apply_bailing_moe_v2_patch()

    return BailingMoeV2ForCausalLM

@PARALLEL_CONFIG_REGISTRY.register("bailing_moe")
def register_bailing_moe_v2_parallel_config():
    from .parallel_config import get_parallel_config
    return get_parallel_config()