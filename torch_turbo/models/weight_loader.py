import gc
import json
import re
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import load_file
from torch.distributed._tensor import distribute_tensor, DTensor

from torch_turbo.distributed.parallel_state import (
    get_expert_mesh,
    get_world_mesh,
    get_expert_model_parallel_rank,
    get_expert_model_parallel_world_size
)
from torch_turbo.models.registry import get_parallel_plan


HF_BUFFER_NAME_MAP = {
    "base": {"router": "gate"},
    "kimi_k2": {"expert_bias": "e_score_correction_bias"},
    "kimi_k25": {"expert_bias": "e_score_correction_bias"}
}


class DistParamLoader:
    def __init__(self, model, hf_config, device, args):
        self.model = model
        self.hf_config = hf_config
        parallel_plan = get_parallel_plan(hf_config)
        self.moe_config = parallel_plan.moe_config
        self.device = device
        self.path = Path(args.path)
        self.ep_mesh = get_expert_mesh()
        self.world_mesh = get_world_mesh()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.ep_rank = get_expert_model_parallel_rank()
        self.ep_world_size = get_expert_model_parallel_world_size()
        # log related
        self.start_time = time.perf_counter()
        self.prev_time = self.start_time
        # build meta
        self.layer_pattern = re.compile(r"(?:^|\.)model\.layers\.(\d+)(?:\.|$)")
        self.param_to_filename = self._load_index_json()
        self.param_to_rank, self.layer_to_rank = self._build_param_to_rank()
        self.local_sd = self._load_local_shards()
        self.use_te = args.use_transformer_engine

    def _load_index_json(self):
        index_path = self.path / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        with open(index_path) as f:
            index = json.load(f)
        return index["weight_map"]

    def _get_layer_id(self, param_name):
        m = self.layer_pattern.search(param_name)
        return int(m.group(1)) if m else None

    def _sync_log_rank0(self, content, update_prev = True):
        torch.cuda.synchronize()
        dist.barrier()
        cur = time.perf_counter()
        if self.rank == 0:
            print(f"{content} in {cur - self.prev_time:.2f}s (total {cur - self.start_time:.2f}s)", flush=True)
        if update_prev:
            self.prev_time = cur

    def _build_param_to_rank(self):
        param_to_rank = {}
        layer_rank_map = {}

        for param_name in self.param_to_filename.keys():
            layer_id = self._get_layer_id(param_name)
            if layer_id is not None:
                rank = (layer_id + 1) % self.world_size
                param_to_rank[param_name] = rank
                layer_rank_map[layer_id] = rank
            else:
                param_to_rank[param_name] = 0

        self._sync_log_rank0(f"Distributed {len(layer_rank_map)} layers across {self.world_size} ranks")
        return param_to_rank, layer_rank_map

    def _load_local_shards(self):
        self.local_params = {
            name: filename
            for name, filename in self.param_to_filename.items()
            if self.param_to_rank[name] == self.rank
        }

        local_sd = {}
        if self.local_params:
            files_needed = set(self.local_params.values())
            files = [self.path / f for f in sorted(files_needed)]
            print(f"[Rank {self.rank}] Loading {len(files)} files with {len(self.local_params)} parameters...", flush=True)
            t0 = time.perf_counter()
            for f in files:
                file_params = load_file(f.as_posix(), device="cpu")
                for param_name, tensor in file_params.items():
                    if param_name in self.local_params:
                        local_sd[param_name] = tensor
            print(f"[Rank {self.rank}] Loaded {len(local_sd)} params in {time.perf_counter() - t0:.2f}s", flush=True)
        else:
            print(f"[Rank {self.rank}] No parameters to load")

        self._sync_log_rank0(f"[Load File] Load finished")
        return local_sd

    def _broadcast_tensor(self, name, shape=None, dtype=None):
        owner = self.param_to_rank[name]
        if self.rank == owner:
            tensor = self.local_sd[name].to(self.device, dtype=dtype)
        else:
            tensor = torch.empty(shape, device=self.device, dtype=dtype)
        dist.broadcast(tensor, src=owner)
        return tensor

    def _get_owner_rank(self, name):
        layer_id = self._get_layer_id(name)
        if layer_id is not None:
            return self.layer_to_rank[layer_id]
        else:
            return self.param_to_rank[name]

    def _get_expert_keys(self, prefix, is_weight1):
        keys = []
        for i in range(self.moe_config.num_moe_experts):
            if is_weight1:
                keys.extend([f"{prefix}.{i}.gate_proj.weight", f"{prefix}.{i}.up_proj.weight"])
            else:
                keys.append(f"{prefix}.{i}.down_proj.weight")
        return keys
    
    def _load_experts_to_device(self):
        for name, _ in self.model.named_parameters():
            if "mlp.experts" not in name or self._get_owner_rank(name) != self.rank:
                continue

            prefix = name[:-8]
            is_weight1 = "weight1" in name
            for key in self._get_expert_keys(prefix, is_weight1):
                assert key in self.local_sd, f"{key} must be in rank {self.rank} local sd"
                self.local_sd[key] = self.local_sd[key].to(self.device)

    def _te_load_experts_to_device(self):
        for name, _ in self.model.named_parameters():
            if "mlp.experts" not in name or "weight0" not in name or self._get_owner_rank(name) != self.rank:
                continue

            prefix = ".".join(name.split(".")[:5])
            is_fc1 = "fc1_linear" in name
            for key in self._get_expert_keys(prefix, is_fc1):
                assert key in self.local_sd, f"{key} must be in rank {self.rank} local sd"
                self.local_sd[key] = self.local_sd[key].to(self.device)

    def _broadcast_empty_tensor(self, param, owner, ep_size):
        new_shape = (param.shape[0], param.shape[1])
        tensor = torch.empty(
            new_shape,
            device=self.device,
            dtype=param.dtype,
        )
        for i in range(ep_size):
            # get broadcast tensor
            dist.broadcast(tensor, src=owner)
            torch.cuda.synchronize()
            if self.rank % ep_size == i:
                result = tensor.clone()
        return result

    def _broadcast_to_ep_ranks(self, name, param, prefix, is_weight1):
        owner = self._get_owner_rank(name)
        ep_size = self.ep_world_size
        result = None

        if owner == self.rank:
            for i in range(ep_size):
                num_local_experts = self.moe_config.num_moe_experts // ep_size
                weights = []
                for j in range(num_local_experts):
                    index = i * num_local_experts + j
                    if is_weight1:
                        gate = self.local_sd.pop(f"{prefix}.{index}.gate_proj.weight")
                        up = self.local_sd.pop(f"{prefix}.{index}.up_proj.weight")
                        weights.append(torch.cat([gate, up]))
                    else:
                        down = self.local_sd.pop(f"{prefix}.{index}.down_proj.weight")
                        weights.append(down.t())
                tensor = torch.stack(weights, dim=0)
                tensor = tensor.permute(0, 2, 1) if is_weight1 else tensor
                tensor = tensor.reshape(-1, tensor.shape[-1])
                if self.rank % ep_size == i:
                    result = tensor
                 # broadcast to ep_group
                dist.broadcast(tensor, src=owner)
                torch.cuda.synchronize()
        else:
            result = self._broadcast_empty_tensor(param, owner, ep_size)

        return result

    def _te_broadcast_to_ep_ranks(self, name, param, prefix_lst):
        prefix = ".".join(prefix_lst[:5])
        owner = self._get_owner_rank(name)
        ep_size = self.ep_world_size
        num_local_experts = self.moe_config.num_moe_experts // ep_size
        result = None
        if owner == self.rank:
            for i in range(ep_size):
                expert_idx = i * num_local_experts + int(re.findall(r"\d+", prefix_lst[-1])[0])
                if "fc1_linear" in name:
                    gate = self.local_sd.pop(f"{prefix}.{expert_idx}.gate_proj.weight")
                    up = self.local_sd.pop(f"{prefix}.{expert_idx}.up_proj.weight")
                    tensor = torch.cat([gate, up])
                elif "fc2_linear" in name:
                    down = self.local_sd.pop(f"{prefix}.{expert_idx}.down_proj.weight")
                    tensor = down
                if self.rank % ep_size == i:
                    result = tensor
                dist.broadcast(tensor, src=owner)
                torch.cuda.synchronize()
        else:
            result = self._broadcast_empty_tensor(param, owner, ep_size)

        return result

    def _hf_buffer_name_convert(self, name):
        for turbo_name, hf_name in  HF_BUFFER_NAME_MAP["base"].items():
            name = name.replace(turbo_name, hf_name)

        model_type = self.hf_config.model_type
        if model_type in HF_BUFFER_NAME_MAP.keys():
            for turbo_name, hf_name in  HF_BUFFER_NAME_MAP[model_type].items():
                name = name.replace(turbo_name, hf_name)
        return name

    def _broadcast_buffers(self):
         # Load buffers (e.g., bailing expert bias)
        for name, buf in self.model.named_buffers():
            if "router.expert_bias" in name:
                hf_buffer_name = self._hf_buffer_name_convert(name)
                full = self._broadcast_tensor(hf_buffer_name, buf.shape, buf.dtype)
                buf.copy_(full)

    def _broadcast_params(self):
        num_param_tensors = len(list(self.model.parameters()))
        i = 0
        for name, param in self.model.named_parameters():
            if "mlp.experts" in name:
                if self.use_te:
                    prefix = name.split(".")
                    full = self._te_broadcast_to_ep_ranks(name, param, prefix)
                else:
                    is_weight1 = "weight1" in name
                    prefix = name[:-8]
                    full = self._broadcast_to_ep_ranks(name, param, prefix, is_weight1)
                mesh = self.ep_mesh['edp']
            elif "mlp.router" in name:
                hf_param_name = name.replace("router", "gate")
                full = self._broadcast_tensor(hf_param_name, param.shape, param.dtype)
                mesh = self.world_mesh
            else:
                full = self._broadcast_tensor(name, param.shape, param.dtype)
                mesh = self.world_mesh

            sharded_tensor = distribute_tensor(full, mesh, param.placements, src_data_rank=None)
            # print(f"name: {name}, deivce: {self.device}, rank: {self.rank}, mesh: {mesh}, param: {param.shape}, full: {full.shape}, sharded_tensor: {sharded_tensor.shape}")

            with torch.no_grad():
                param.copy_(nn.Parameter(sharded_tensor))
            del sharded_tensor, full
            i += 1
            mem_free, mem_total = torch.cuda.mem_get_info()
            mem_used = mem_total - mem_free
            utilization = mem_used / mem_total
            self._sync_log_rank0(f"[Rank {self.rank}, mem {utilization:.1%}] Broadcast {i}/{num_param_tensors} params", False)
            if utilization > 0.8:
                torch.cuda.empty_cache()
                gc.collect()
                new_free, _ = torch.cuda.mem_get_info()
                freed_gb = (new_free - mem_free) / 1024 ** 3
                print(f"Rank {self.rank} Cleaned {freed_gb:.2f} GB ({new_free} - {mem_free}) GPU memory (utilization: {utilization:.1%} -> {(mem_total - new_free)/mem_total:.1%})", flush=True)

    def _validation(self):
        # Validation
        for p in self.model.parameters():
            assert isinstance(p, DTensor), f"Parameter {p} is not DTensor"

    def _cleanup(self):
        self.local_sd = None
        gc.collect()
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.empty_cache()

    def load_weights(self):
        # move experts from cpu to npu
        if self.use_te:
            self._te_load_experts_to_device()
        else:
            self._load_experts_to_device()
        self._sync_log_rank0(f"[Rank {self.rank}] load experts to gpu")

        # buffers
        self._broadcast_buffers()
        self._sync_log_rank0(f"[Rank {self.rank}] broadcast buffers")

        # parameters
        self._broadcast_params()
        self._sync_log_rank0(f"[Rank {self.rank}] Broadcast/sharding done")

        self._cleanup()

        self._validation()
