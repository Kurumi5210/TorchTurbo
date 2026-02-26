import logging

import transformers.models.qwen3_moe.modeling_qwen3_moe as qewn3_moe_model

from torch_turbo.distributed.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from torch_turbo.distributed.parallel_state import get_parallel_state
from torch_turbo.distributed.sequence_parallel.utils import create_causal_mask
from torch_turbo.models.registry import MODELING_REGISTRY, PARALLEL_CONFIG_REGISTRY
from torch_turbo.ops import npu_fused_ops
from torch_turbo.utils.device import is_npu_available


def apply_qwen3_moe_patch():
    if is_npu_available:
        qewn3_moe_model.apply_rotary_pos_emb = npu_fused_ops.apply_rotary_pos_emb_npu
        qewn3_moe_model.Qwen3MoeRMSNorm.forward = npu_fused_ops.fused_rms_norm_forward_npu
        MoEAlltoAllTokenDispatcher.token_permutation = npu_fused_ops.fused_token_permutation_npu
        MoEAlltoAllTokenDispatcher.token_unpermutation = npu_fused_ops.fused_token_unpermutation_npu
    
    parallel_state = get_parallel_state()
    if parallel_state.sp_enabled():
        qewn3_moe_model.create_causal_mask = create_causal_mask


@MODELING_REGISTRY.register("qwen3_moe")
def register_qwen3_moe_modeling(hf_config):
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

    apply_qwen3_moe_patch()

    print("Qwen3 MoE modeling registered!")
    return Qwen3MoeForCausalLM(hf_config)


@PARALLEL_CONFIG_REGISTRY.register("qwen3_moe")
def register_qwen3_moe_parallel_config():
    from .parallel_config import get_parallel_config

    print("Qwen3 MoE parallel config registered!")
    return get_parallel_config()