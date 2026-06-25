import gc
import json
import time
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import load_file
from torch.distributed._tensor import distribute_tensor, DTensor

from torchturbo.engine.peft.naming import strip_peft_prefix, is_peft_adapter_param
from torchturbo.engine.peft.weights import get_peft_init_config, init_lora_tensor

class DistParamLoader:
    def __init__(
        self,
        model,
        path,
        rank,
        mesh,
        world_mesh,
        device,
    ):
        self.model = model
        self.hf_config = model.config
        self.device = device
        self.path = Path(path)
        self.mesh = mesh
        self.world_mesh = world_mesh
        self.rank = rank
        self.world_size = dist.get_world_size()
        self.ep_group = mesh["ep"].get_group()
        self.ep_rank = dist.get_rank(group=self.ep_group)
        self.ep_world_size = dist.get_world_size(group=self.ep_group)
        self.param_converter = model.get_weight_converter(mesh)
        self.moe_config = self.param_converter.moe_config
        # log related
        self.start_time = time.perf_counter()
        self.prev_time = self.start_time
        # build meta
        self.layer_pattern = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
        self.param_to_filename = self._load_index_json()
        self.param_to_rank, self.layer_to_rank = self._build_param_to_rank()

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
                        local_sd[param_name] = tensor.clone()
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
        dist.broadcast(tensor, src=owner, async_op=False)
        return tensor

    def _get_owner_rank(self, name):
        layer_id = self._get_layer_id(name)
        if layer_id is not None:
            return self.layer_to_rank[layer_id]
        else:
            return self.param_to_rank[name]

    def _broadcast_empty_tensor(self, param, owner, ep_size):
        new_shape = (param.shape[0], param.shape[1])
        tensor = torch.empty(
            new_shape,
            device=self.device,
            dtype=param.dtype,
        )
        for i in range(ep_size):
            # get broadcast tensor
            dist.broadcast(tensor, src=owner, async_op=False)
            torch.cuda.synchronize()
            if self.rank % ep_size == i:
                result = tensor.clone()
        return result

    def _broadcast_expert_weight1(self, name, param):
        owner = self._get_owner_rank(name)
        ep_size = self.ep_world_size
        result = None

        if owner == self.rank:
            num_local_experts = self.moe_config.num_moe_experts // ep_size
            for i, tensor in enumerate(
                self.param_converter.iter_load_weight1(
                    self.local_sd, name, ep_size, num_local_experts,
                    self.device, param.dtype,
                )
            ):
                if self.rank % ep_size == i:
                    result = tensor
                dist.broadcast(tensor, src=owner, async_op=False)
        else:
            result = self._broadcast_empty_tensor(param, owner, ep_size)

        return result

    def _broadcast_expert_weight2(self, name, param):
        owner = self._get_owner_rank(name)
        ep_size = self.ep_world_size
        result = None

        if owner == self.rank:
            num_local_experts = self.moe_config.num_moe_experts // ep_size
            for i, tensor in enumerate(
                self.param_converter.iter_load_weight2(
                    self.local_sd, name, ep_size, num_local_experts,
                    self.device, param.dtype,
                )
            ):
                if self.rank % ep_size == i:
                    result = tensor
                dist.broadcast(tensor, src=owner, async_op=False)
        else:
            result = self._broadcast_empty_tensor(param, owner, ep_size)

        return result

    def _fix_buffer_dtype(self):
        for name, submodule in self.model.named_modules():
            for buf_name, buf in submodule.named_buffers(recurse=False):
                full_name = f"{name}.{buf_name}" if name else buf_name
                clean_name = strip_peft_prefix(full_name)
                if not self.param_converter.is_persist_buffer(clean_name):
                    continue
                hf_buffer_name = self.param_converter.convert_persist_buffer(clean_name)
                hf_dtype = self.param_converter.get_default_hf_dtype(hf_buffer_name)
                old_dtype = buf.dtype
                if old_dtype != hf_dtype:
                    new_buf = buf.detach().to(hf_dtype)
                    submodule.register_buffer(buf_name, new_buf)
                    print(f"Converted {name}.{buf_name}: {old_dtype} -> {hf_dtype}")

    def _broadcast_buffers(self):
        self._fix_buffer_dtype()
        for name, buf in self.model.named_buffers():
            clean_name = strip_peft_prefix(name)
            if self.param_converter.is_persist_buffer(clean_name):
                hf_buffer_name = self.param_converter.convert_persist_buffer(clean_name)
                if hf_buffer_name not in self.param_to_filename:
                    if self.rank == 0:
                        print(f"[Weight Loader] SKIP buffer (not in checkpoint): {clean_name} (original: {name})", flush=True)
                    continue
                full = self._broadcast_tensor(hf_buffer_name, buf.shape, buf.dtype)
                with torch.no_grad():
                    buf.copy_(full)

    def _init_and_broadcast_lora(self, clean_name, param):
        """Initialize a LoRA adapter param on the source rank and broadcast.

        Expert LoRA params are initialized on edp_rank=0 and broadcast
        within the edp group.  Non-expert LoRA params are initialized on
        global rank 0 and broadcast to the world.

        Initialization follows the PEFT ``init_lora_weights`` config.

        Returns ``(full_tensor, mesh)``.
        """
        is_a = (".lora_A." in clean_name
                or ".lora_A2." in clean_name
                or ".lora_embedding_A." in clean_name)
        # GroupedMLPLoRA params are direct children of the .experts. module
        # (e.g. model.layers.1.mlp.experts.lora_A.default.weight).
        # shared_experts use standard lora.Linear and have an intermediate
        # module (e.g. model.layers.1.mlp.shared_experts.gate_proj.lora_A.default.weight).
        # lora_A2 / lora_B2 are unique to GroupedMLPLoRA.
        is_expert = (
            ".experts.lora_A2." in clean_name
            or ".experts.lora_B2." in clean_name
            or ".experts.lora_A." in clean_name
            or ".experts.lora_B." in clean_name
        )
        init_lora_weights, r = get_peft_init_config(self.model)

        if is_expert:
            edp_mesh = self.mesh["edp"]
            edp_group = edp_mesh.get_group()
            edp_rank = edp_mesh.get_local_rank()
            # src for dist.broadcast must be a global rank.
            edp_src_global = dist.get_global_rank(edp_group, 0)

            if edp_rank == 0:
                full = torch.empty(param.shape, device=self.device, dtype=param.dtype)
                init_lora_tensor(full, is_a, init_lora_weights, r)
            else:
                full = torch.empty(param.shape, device=self.device, dtype=param.dtype)

            dist.broadcast(full, src=edp_src_global, group=edp_group, async_op=False)
            return full, edp_mesh
        else:
            if self.rank == 0:
                full = torch.empty(param.shape, device=self.device, dtype=param.dtype)
                init_lora_tensor(full, is_a, init_lora_weights, r)
            else:
                full = torch.empty(param.shape, device=self.device, dtype=param.dtype)

            dist.broadcast(full, src=0, async_op=False)
            return full, self.world_mesh

    def _broadcast_params(self):
        num_param_tensors = len(list(self.model.parameters()))
        i = 0
        for name, param in self.model.named_parameters():
            # Strip PEFT prefixes (base_model.model. and base_layer.)
            # so that parameter names match the HF checkpoint format.
            clean_name = strip_peft_prefix(name)

            # LoRA adapter parameters: initialize on the source rank of the
            # corresponding process group, then broadcast.  They do not exist
            # in the base model checkpoint.
            if is_peft_adapter_param(clean_name):
                full, mesh = self._init_and_broadcast_lora(clean_name, param)
            elif self.param_converter.is_expert_weight1(clean_name):
                full = self._broadcast_expert_weight1(clean_name, param)
                mesh = self.mesh['edp']
            elif self.param_converter.is_expert_weight2(clean_name):
                full = self._broadcast_expert_weight2(clean_name, param)
                mesh = self.mesh['edp']
            elif self.param_converter.is_router_param(clean_name):
                hf_param_name = self.param_converter.convert_router_param_to_hf(clean_name)
                if hf_param_name not in self.param_to_filename:
                    if self.rank == 0:
                        print(f"[Weight Loader] SKIP (not in checkpoint): {clean_name} (original: {name})", flush=True)
                    i += 1
                    continue
                full = self._broadcast_tensor(hf_param_name, param.shape, param.dtype)
                mesh = self.world_mesh
            else:
                if clean_name not in self.param_to_filename:
                    if self.rank == 0:
                        print(f"[Weight Loader] SKIP (not in checkpoint): {clean_name} (original: {name})", flush=True)
                    i += 1
                    continue
                full = self._broadcast_tensor(clean_name, param.shape, param.dtype)
                mesh = self.world_mesh

            if isinstance(param, DTensor):
                sharded_tensor = distribute_tensor(
                    full, mesh, param.placements, src_data_rank=None
                )
            else:
                sharded_tensor = full
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

    def _cleanup(self):
        self.local_sd = None
        gc.collect()
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.empty_cache()

    def load_weights(self):
        self.model.to_empty(device=self.device)
        self.local_sd = self._load_local_shards()

        # buffers
        self._broadcast_buffers()
        self._sync_log_rank0(f"[Rank {self.rank}] broadcast buffers")

        # parameters
        self._broadcast_params()
        self._sync_log_rank0(f"[Rank {self.rank}] Broadcast/sharding done")

        self._cleanup()