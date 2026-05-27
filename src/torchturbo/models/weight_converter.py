import torch
import torch.distributed as dist

from torchturbo.models.modules.moe.moe_layer import BaseMoELayer

class WeightConverterMixin:
    def get_moe_config(self):
        for _, module in self.named_modules():
            if isinstance(module, BaseMoELayer):
                moe_config = module.config
                return moe_config
        return None

    def get_weight_converter(self, mesh):
        return BaseWeightConverter(self.config, self.get_moe_config(), mesh)


class BaseWeightConverter:
    def __init__(self, config, moe_config, mesh):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.world_mesh = mesh
        self.ep_mesh = mesh["ep"]
        self.ep_group = self.ep_mesh.get_group()
        self.ep_rank = dist.get_rank(group=self.ep_group)
        self.ep_size = dist.get_world_size(group=self.ep_group)
        self.moe_config = moe_config
        self.config = config
        self.model_type = config.model_type
        self.dtype = self.get_save_dtype(config.dtype)

    def get_save_dtype(self, dtype_str):
        DTYPE_MAP = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if dtype_str in DTYPE_MAP:
            return DTYPE_MAP[dtype_str]
        return torch.bfloat16

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
        if self.get_hf_gate_bias_name() in param_name:
            return torch.float32
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

    def convert_expert_weight1_to_hf(self, name, only_local_experts = False):
        assert self.is_expert_weight1(name), "must be"
        assert self.moe_config != None, "must be"
        prefix = name[:-8].replace(self.get_moe_expert_mlp_name(), self.get_hf_expert_mlp_name())
        gates = []
        ups = []
        for i in range(self.ep_size):
            if only_local_experts and i != self.ep_rank:
                continue
            num_local_experts = self.moe_config.num_moe_experts // self.ep_size
            for j in range(num_local_experts):
                index = i * num_local_experts + j
                gate =f"{prefix}.{index}.{self.get_expert_gate_name()}.weight"
                up = f"{prefix}.{index}.{self.get_expert_up_name()}.weight"
                gates.append(gate)
                ups.append(up)
        return gates, ups

    def get_gate_weight_shape(self):
        return (self.config.moe_intermediate_size, self.config.hidden_size)

    def get_gate_weight_bytes(self, element_size):
        return self.config.moe_intermediate_size * self.config.hidden_size * element_size

    def get_up_weight_shape(self):
        return (self.config.moe_intermediate_size, self.config.hidden_size)

    def get_up_weight_bytes(self, element_size):
        return self.config.moe_intermediate_size * self.config.hidden_size * element_size

    def get_down_weight_shape(self):
        return (self.config.hidden_size, self.config.moe_intermediate_size)

    def get_down_weight_bytes(self, element_size):
        return self.config.moe_intermediate_size * self.config.hidden_size * element_size

    def get_local_ep_w1_hf_state_dict(self, name, tensor):
        assert self.is_expert_weight1(name), "must be"
        assert self.moe_config != None, "must be"
        num_local_experts = self.moe_config.num_moe_experts // self.ep_size
        experts = torch.split(tensor, self.config.hidden_size)
        start_index = self.ep_rank * num_local_experts
        prefix = name[:-8].replace(self.get_moe_expert_mlp_name(), self.get_hf_expert_mlp_name())
        local_ep_sd = {}
        for i in range(num_local_experts):
            index = start_index + i
            gate =f"{prefix}.{index}.{self.get_expert_gate_name()}.weight"
            up = f"{prefix}.{index}.{self.get_expert_up_name()}.weight"
            expert2 = torch.split(experts[i].t(), self.config.moe_intermediate_size)
            gate_tensor = expert2[0]
            up_tensor = expert2[1]
            local_ep_sd[gate] = gate_tensor
            local_ep_sd[up] = up_tensor
        return local_ep_sd

    def convert_expert_weight2_to_hf(self, name, only_local_experts = False):
        assert self.is_expert_weight2(name), "must be"
        assert self.moe_config != None, "must be"
        prefix = name[:-8].replace(self.get_moe_expert_mlp_name(), self.get_hf_expert_mlp_name())
        downs = []
        for i in range(self.ep_size):
            if only_local_experts and i != self.ep_rank:
                continue
            num_local_experts = self.moe_config.num_moe_experts // self.ep_size
            for j in range(num_local_experts):
                index = i * num_local_experts + j
                down =f"{prefix}.{index}.{self.get_expert_down_name()}.weight"
                downs.append(down)
        return downs

    def get_local_ep_w2_hf_state_dict(self, name, tensor):
        assert self.is_expert_weight2(name), "must be"
        assert self.moe_config != None, "must be"
        num_local_experts = self.moe_config.num_moe_experts // self.ep_size
        start_index = self.ep_rank * num_local_experts
        prefix = name[:-8].replace(self.get_moe_expert_mlp_name(), self.get_hf_expert_mlp_name())
        experts = torch.split(tensor, self.config.moe_intermediate_size)
        local_ep_sd = {}
        for i in range(num_local_experts):
            index = start_index + i
            down =f"{prefix}.{index}.{self.get_expert_down_name()}.weight"
            down_tensor = experts[i].t()
            local_ep_sd[down] = down_tensor
        return local_ep_sd

    def convert_router_param_to_hf(self, name):
        assert self.get_moe_router_name() in name, "must be"
        return name.replace(self.get_moe_router_name(), self.get_hf_gate_name())