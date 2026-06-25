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
        self.dtype = self.get_save_dtype(config)

    def get_save_dtype(self, config):
        if hasattr(config, 'dtype'):
            return config.dtype
        dtype_str = "bfloat16"
        if hasattr(config, "torch_dtype"):
            dtype_str = config.torch_dtype
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

    def _per_expert_w1_keys(self, name):
        """Return (gates, ups) key lists for per-expert checkpoint format."""
        prefix = name[:-8].replace(self.get_moe_expert_mlp_name(), self.get_hf_expert_mlp_name())
        gates, ups = [], []
        for i in range(self.ep_size):
            num_local = self.moe_config.num_moe_experts // self.ep_size
            for j in range(num_local):
                idx = i * num_local + j
                gates.append(f"{prefix}.{idx}.{self.get_expert_gate_name()}.weight")
                ups.append(f"{prefix}.{idx}.{self.get_expert_up_name()}.weight")
        return gates, ups

    def _per_expert_w2_keys(self, name):
        """Return downs key list for per-expert checkpoint format."""
        prefix = name[:-8].replace(self.get_moe_expert_mlp_name(), self.get_hf_expert_mlp_name())
        downs = []
        for i in range(self.ep_size):
            num_local = self.moe_config.num_moe_experts // self.ep_size
            for j in range(num_local):
                idx = i * num_local + j
                downs.append(f"{prefix}.{idx}.{self.get_expert_down_name()}.weight")
        return downs

    def iter_load_weight1(self, local_sd, name, ep_size, num_local_experts, device, dtype):
        """Yield one [num_local*H, 2I] tensor per EP group from per-expert HF tensors."""
        gates, ups = self._per_expert_w1_keys(name)
        for i in range(ep_size):
            weights = []
            for j in range(num_local_experts):
                idx = i * num_local_experts + j
                gate = local_sd.pop(gates[idx]).to(device).to(dtype)
                up = local_sd.pop(ups[idx]).to(device).to(dtype)
                weights.append(torch.cat([gate, up]))
            tensor = torch.stack(weights, dim=0)
            yield tensor.permute(0, 2, 1).reshape(-1, tensor.shape[1])

    def iter_load_weight2(self, local_sd, name, ep_size, num_local_experts, device, dtype):
        """Yield one [num_local*I, H] tensor per EP group from per-expert HF tensors."""
        downs = self._per_expert_w2_keys(name)
        for i in range(ep_size):
            weights = []
            for j in range(num_local_experts):
                idx = i * num_local_experts + j
                down = local_sd.pop(downs[idx]).to(device).to(dtype)
                weights.append(down.t())
            tensor = torch.stack(weights, dim=0)
            yield tensor.reshape(-1, tensor.shape[-1])

    def get_weight1_save_entries(self, name, element_size, ep_rank, ep_size):
        """Return (local_keys, size_map) for weight1 save allocation."""
        ep_gates, ep_ups = self.convert_expert_weight1_to_hf(name, only_local_experts=True)
        gates, ups = self.convert_expert_weight1_to_hf(name)
        gate_size = self.get_gate_weight_bytes(element_size)
        up_size = self.get_up_weight_bytes(element_size)
        local_keys = list(ep_gates) + list(ep_ups)
        size_map = {g: gate_size for g in gates}
        size_map.update({u: up_size for u in ups})
        return local_keys, size_map

    def get_weight2_save_entries(self, name, element_size, ep_rank, ep_size):
        """Return (local_keys, size_map) for weight2 save allocation."""
        ep_downs = self.convert_expert_weight2_to_hf(name, only_local_experts=True)
        downs = self.convert_expert_weight2_to_hf(name)
        down_size = self.get_down_weight_bytes(element_size)
        local_keys = list(ep_downs)
        size_map = {d: down_size for d in downs}
        return local_keys, size_map

    @property
    def fused_experts(self):
        return False


class FusedExpertWeightConverter(BaseWeightConverter):
    """Weight converter for models with fused expert tensors.

    Handles checkpoints where expert weights are stored as 3D fused tensors
    (e.g. gate_up_proj [E, 2*I, H]) rather than per-expert 2D tensors.
    """

    @property
    def fused_experts(self):
        return True

    def get_fused_w1_name(self):
        return 'gate_up_proj'

    def get_fused_w2_name(self):
        return 'down_proj'

    def _expert_prefix(self, name):
        return name.rsplit('.', 1)[0]

    def convert_expert_weight1_to_hf(self, name, only_local_experts=False):
        prefix = self._expert_prefix(name)
        hf_prefix = prefix.replace(
            self.get_moe_expert_mlp_name(), self.get_hf_expert_mlp_name()
        )
        fused_name = f"{hf_prefix}.{self.get_fused_w1_name()}"
        return fused_name

    def convert_expert_weight2_to_hf(self, name, only_local_experts=False):
        prefix = self._expert_prefix(name)
        hf_prefix = prefix.replace(
            self.get_moe_expert_mlp_name(), self.get_hf_expert_mlp_name()
        )
        fused_name = f"{hf_prefix}.{self.get_fused_w2_name()}"
        return fused_name

    def get_fused_w1_shape(self):
        num_experts = self.moe_config.num_moe_experts
        return (num_experts, 2 * self.config.moe_intermediate_size, self.config.hidden_size)

    def get_fused_w2_shape(self):
        num_experts = self.moe_config.num_moe_experts
        return (num_experts, self.config.hidden_size, self.config.moe_intermediate_size)

    def get_fused_w1_bytes(self, element_size):
        s = self.get_fused_w1_shape()
        return s[0] * s[1] * s[2] * element_size

    def get_fused_w2_bytes(self, element_size):
        s = self.get_fused_w2_shape()
        return s[0] * s[1] * s[2] * element_size

    def get_local_ep_w1_hf_state_dict(self, name, tensor):
        assert self.is_expert_weight1(name), "must be"
        assert self.moe_config is not None, "must be"
        num_local_experts = self.moe_config.num_moe_experts // self.ep_size
        w1 = tensor.reshape(num_local_experts, self.config.hidden_size, -1)
        w1 = w1.permute(0, 2, 1).contiguous()  # [num_local, 2I, H]
        # All-gather across EP group to reconstruct full [E, 2I, H]
        w1_list = [torch.zeros_like(w1) for _ in range(self.ep_size)]
        dist.all_gather(w1_list, w1, group=self.ep_group)
        full_w1 = torch.cat(w1_list, dim=0)  # [E, 2I, H]
        hf_name = self.convert_expert_weight1_to_hf(name)
        return {hf_name: full_w1}

    def get_local_ep_w2_hf_state_dict(self, name, tensor):
        assert self.is_expert_weight2(name), "must be"
        assert self.moe_config is not None, "must be"
        num_local_experts = self.moe_config.num_moe_experts // self.ep_size
        w2 = tensor.reshape(num_local_experts, self.config.moe_intermediate_size, self.config.hidden_size)
        w2 = w2.permute(0, 2, 1).contiguous()  # [num_local, H, I]
        # All-gather across EP group to reconstruct full [E, H, I]
        w2_list = [torch.zeros_like(w2) for _ in range(self.ep_size)]
        dist.all_gather(w2_list, w2, group=self.ep_group)
        full_w2 = torch.cat(w2_list, dim=0)  # [E, H, I]
        hf_name = self.convert_expert_weight2_to_hf(name)
        return {hf_name: full_w2}

    def get_gate_weight_shape(self):
        raise AssertionError("get_gate_weight_shape is per-expert only, not supported for fused format")

    def get_gate_weight_bytes(self, element_size):
        raise AssertionError("get_gate_weight_bytes is per-expert only, not supported for fused format")

    def get_up_weight_shape(self):
        raise AssertionError("get_up_weight_shape is per-expert only, not supported for fused format")

    def get_up_weight_bytes(self, element_size):
        raise AssertionError("get_up_weight_bytes is per-expert only, not supported for fused format")

    def get_down_weight_shape(self):
        raise AssertionError("get_down_weight_shape is per-expert only, not supported for fused format")

    def get_down_weight_bytes(self, element_size):
        raise AssertionError("get_down_weight_bytes is per-expert only, not supported for fused format")

    def iter_load_weight1(self, local_sd, name, ep_size, num_local_experts, device, dtype):
        """Yield from fused [E, 2I, H] tensor, one [num_local*H, 2I] per EP group.

        Falls back to per-expert loading if the fused key is not in checkpoint.
        """
        hf_name = self.convert_expert_weight1_to_hf(name)
        if hf_name not in local_sd:
            yield from super().iter_load_weight1(
                local_sd, name, ep_size, num_local_experts, device, dtype,
            )
            return
        full_weight = local_sd.pop(hf_name).to(device).to(dtype)
        try:
            for i in range(ep_size):
                chunk = full_weight[i * num_local_experts : (i + 1) * num_local_experts]
                yield chunk.permute(0, 2, 1).reshape(-1, chunk.shape[1])
        finally:
            del full_weight

    def iter_load_weight2(self, local_sd, name, ep_size, num_local_experts, device, dtype):
        """Yield from fused [E, H, I] tensor, one [num_local*I, H] per EP group.

        Falls back to per-expert loading if the fused key is not in checkpoint.
        """
        hf_name = self.convert_expert_weight2_to_hf(name)
        if hf_name not in local_sd:
            yield from super().iter_load_weight2(
                local_sd, name, ep_size, num_local_experts, device, dtype,
            )
            return
        full_weight = local_sd.pop(hf_name).to(device).to(dtype)
        try:
            for i in range(ep_size):
                chunk = full_weight[i * num_local_experts : (i + 1) * num_local_experts]
                permuted = chunk.permute(0, 2, 1)
                yield permuted.reshape(-1, permuted.shape[-1])
        finally:
            del full_weight

    def get_weight1_save_entries(self, name, element_size, ep_rank, ep_size):
        """Return (local_keys, size_map) for fused weight1 save."""
        fused_name = self.convert_expert_weight1_to_hf(name)
        fused_size = (
            self.moe_config.num_moe_experts
            * 2 * self.config.moe_intermediate_size
            * self.config.hidden_size
            * element_size
        )
        return [fused_name], {fused_name: fused_size}

    def get_weight2_save_entries(self, name, element_size, ep_rank, ep_size):
        """Return (local_keys, size_map) for fused weight2 save."""
        fused_name = self.convert_expert_weight2_to_hf(name)
        fused_size = (
            self.moe_config.num_moe_experts
            * self.config.hidden_size
            * self.config.moe_intermediate_size
            * element_size
        )
        return [fused_name], {fused_name: fused_size}