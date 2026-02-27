import torch
import transformers.models.qwen3_moe.modeling_qwen3_moe as qewn3_moe_model
from transformers import PretrainedConfig

from torch_turbo.distributed.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from torch_turbo.distributed.parallel_state import get_parallel_state
from torch_turbo.distributed.parallel_plan import MoEConfig
from torch_turbo.distributed.sequence_parallel.utils import create_causal_mask
from torch_turbo.models.registry import MODELING_REGISTRY, MOE_CONFIG_REGISTRY, PARALLEL_PLAN_REGISTRY
from torch_turbo.models.config_converter import get_base_moe_config
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

    print("Qwen3MoE modeling registered!")
    return Qwen3MoeForCausalLM(hf_config)


@MOE_CONFIG_REGISTRY.register("qwen3_moe")
def register_qwen3_moe_config(hf_config: PretrainedConfig) -> MoEConfig:
    qwen3_config_kwargs = get_base_moe_config(
        hf_config = hf_config,
        num_moe_experts = hf_config.num_experts,
    )
    print("Qwen3MoE moe config registered!")
    return MoEConfig(**qwen3_config_kwargs)


@PARALLEL_PLAN_REGISTRY.register("qwen3_moe")
def register_qwen3_moe_parallel_plan():
    from .parallel_plan import get_parallel_plan

    print("Qwen3MoE parallel plan registered!")
    return get_parallel_plan()