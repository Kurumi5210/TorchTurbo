from typing_extensions import override
import torch
from torch_turbo.models.name_converter import BaseWeightConverter


class DeepseekV4WeightConverter(BaseWeightConverter):
    def __init__(self, config, moe_config):
        super().__init__(config, moe_config)

    @override
    def get_expert_gate_name(self):
        return 'w1'

    @override
    def get_expert_up_name(self):
        return 'w3'

    @override
    def get_expert_down_name(self):
        return 'w2'

    @override
    def get_hf_expert_mlp_name(self):
        return 'ffn'

    def get_router_hash_name(self):
        return "router.tid2eid"

    @override
    def get_hf_gate_bias_name(self):
        return "gate.bias"

    def get_hf_gate_hash_name(self):
        return "gate.tid2eid"

    def get_special_param_names(self):
        return ["hc_head_base", "hc_head_fn", "hc_head_scale",
                "hc_attn_base", "hc_attn_fn", "hc_attn_scale",
                "hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale",
                "attn.attn_sink", "compressor.ape"]

    @override
    def get_default_hf_dtype(self, param_name):
        if self.get_hf_gate_hash_name() in param_name:
            return torch.int64
        if self.get_hf_gate_bias_name() in param_name:
            return torch.float32
        for name in self.get_special_param_names():
            if name in param_name:
                return torch.float32
        return self.dtype

    @override
    def is_persist_buffer(self, name):
        if self.get_router_bias_name() in name:
            return True
        if self.get_router_hash_name() in name:
            return True
        return False

    @override
    def convert_persist_buffer(self, name):
        if self.get_router_bias_name() in name:
            return name.replace(self.get_router_bias_name(), self.get_hf_gate_bias_name())
        if self.get_router_hash_name() in name:
            return name.replace(self.get_router_hash_name(), self.get_hf_gate_hash_name())
        return name