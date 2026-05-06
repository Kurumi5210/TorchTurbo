import torch
import torch.distributed as dist

from torch_turbo.distributed.parallel_state import get_expert_model_parallel_rank, get_expert_model_parallel_world_size, get_world_mesh


class BaseWeightConverter:
    def __init__(self, config, moe_config):
        self.world_mesh = get_world_mesh()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.ep_rank = get_expert_model_parallel_rank()
        self.ep_size = get_expert_model_parallel_world_size()
        self.moe_config = moe_config
        self.model_type = config.model_type
        self.dtype = config.torch_dtype

    def get_experts_name(self):
        return 'experts'

    def get_expert_gate_name(self):
        return 'gate_proj'

    def get_expert_up_name(self):
        return 'up_proj'

    def get_experts_weight1_name(self):
        return 'weight1'

    def get_expert_down_name(self):
        return 'down_proj'

    def get_experts_weight2_name(self):
        return 'weight2'

    def get_moe_expert_mlp_name(self):
        return 'mlp'

    def get_hf_expert_mlp_name(self):
        return 'mlp'

    def get_moe_router_name(self):
        return 'router'

    def get_hf_gate_name(self):
        return 'gate'
    
    def get_router_bias_name(self):
        return "router.expert_bias"
    
    def get_hf_gate_bias_name(self):
        return "gate.e_score_correction_bias"

    def get_default_hf_dtype(self, param_name):
        return self.dtype

    def is_expert(self, name):
        return self.get_experts_name() in name

    def is_expert_weight1(self, name):
        return self.is_expert(name) and self.get_experts_weight1_name() in name

    def is_expert_weight2(self, name):
        return self.is_expert(name) and self.get_experts_weight2_name() in name
    
    def is_router_param(self, name):
        return self.get_moe_router_name() in name
    
    def is_persist_buffer(self, name):
        return self.get_router_bias_name() in name

    def convert_persist_buffer(self, name):
        if self.get_router_bias_name() in name:
            return name.replace(self.get_router_bias_name(), self.get_hf_gate_bias_name())
        return name
        
    def convert_expert_weight1_to_hf(self, name):
        assert self.is_expert_weight1(name), "must be"
        prefix = name[:-8].replace(self.get_moe_expert_mlp_name(), self.get_hf_expert_mlp_name())
        gates = []
        ups = []
        for i in range(self.ep_size):
            num_local_experts = self.moe_config.num_moe_experts // self.ep_size
            for j in range(num_local_experts):
                index = i * num_local_experts + j
                gate =f"{prefix}.{index}.{self.get_expert_gate_name()}.weight"
                up = f"{prefix}.{index}.{self.get_expert_up_name()}.weight"
                gates.append(gate)
                ups.append(up)
        return gates, ups

    def convert_expert_weight2_to_hf(self, name):
        assert self.is_expert_weight2(name), "must be"
        prefix = name[:-8]
        downs = []
        for i in range(self.ep_size):
            num_local_experts = self.moe_config.num_moe_experts // self.ep_size
            for j in range(num_local_experts):
                index = i * num_local_experts + j
                down =f"{prefix}.{index}.{self.get_expert_down_name()}.weight"
                downs.append(down)
        return downs

    def convert_router_param_to_hf(self, name):
        assert self.get_moe_router_name() in name, "must be"
        return name.replace(self.get_moe_router_name(), self.get_hf_gate_name())