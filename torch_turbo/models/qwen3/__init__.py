from transformers import PretrainedConfig


from torch_turbo.distributed.parallel_state import get_parallel_state
from torch_turbo.distributed.sequence_parallel.utils import create_causal_mask
from torch_turbo.models.registry import MODELING_REGISTRY,  PARALLEL_PLAN_REGISTRY
from torch_turbo.ops import npu_fused_ops
from torch_turbo.utils.device import is_npu_available


def apply_qwen3_patch():
    import transformers.models.qwen3.modeling_qwen3 as qwen3_model
    if is_npu_available:
        qwen3_model.apply_rotary_pos_emb = npu_fused_ops.apply_rotary_pos_emb_npu
        qwen3_model.Qwen3RMSNorm.forward = npu_fused_ops.fused_rms_norm_forward_npu
    
    parallel_state = get_parallel_state()
    if parallel_state.sp_enabled():
        print(f"Monkey patched create_causal_mask in Qwen3 Model!")
        qwen3_model.create_causal_mask = create_causal_mask


@MODELING_REGISTRY.register("qwen3")
def register_qwen3_modeling(hf_config):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    
    apply_qwen3_patch()

    print("Qwen3 modeling registered!")
    return Qwen3ForCausalLM(hf_config)


@PARALLEL_PLAN_REGISTRY.register("qwen3")
def register_qwen3_parallel_plan():
    from .parallel_plan import get_parallel_plan

    print("Qwen3 parallel plan registered!")
    return get_parallel_plan()