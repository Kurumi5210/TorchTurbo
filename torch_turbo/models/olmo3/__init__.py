from transformers import PretrainedConfig

from torch_turbo.models.registry import MODELING_REGISTRY, PARALLEL_PLAN_REGISTRY
from torch_turbo.models.config_converter import get_base_moe_config
from torch_turbo.distributed.parallel_state import get_parallel_state
from torch_turbo.distributed.sequence_parallel.utils import create_causal_mask, create_sliding_window_causal_mask

from torch_turbo.ops import npu_fused_ops
from torch_turbo.utils.device import is_npu_available

def apply_olmo3_patch():
    import torch_turbo.models.olmo3.modeling_olmo3 as olmo3_model
    if is_npu_available:
        olmo3_model.apply_rotary_pos_emb = npu_fused_ops.apply_rotary_pos_emb_npu
        olmo3_model.Olmo3RMSNorm.forward = npu_fused_ops.fused_rms_norm_forward_npu

    parallel_state = get_parallel_state()

    print("Monkey patched causal_mask in HF Olmo3 Model!")
    olmo3_model.create_causal_mask = create_causal_mask
    olmo3_model.create_sliding_window_causal_mask = create_sliding_window_causal_mask


@MODELING_REGISTRY.register("olmo3")
def register_olmo3_modeling(hf_config):
    from .modeling_olmo3 import Olmo3ForCausalLM

    apply_olmo3_patch()

    print("Olmo3 modeling registered!")
    return Olmo3ForCausalLM(hf_config)


@PARALLEL_PLAN_REGISTRY.register("olmo3")
def register_olmo3_parallel_plan():
    from .parallel_plan import get_parallel_plan

    print("Olmo3 parallel plan registered!")
    return get_parallel_plan()