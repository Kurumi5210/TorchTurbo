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

from torchturbo.models.modules.moe.moe_layer import BaseMoELayer


class MoeParamType(Enum):
    """Types of MoE parameters."""

    EXPERTS = "experts"
    # Not used yet
    SHARED_EXPERTS = "shared_experts"
    ROUTER = "router"
    ROUTER_BUFFER = "router_buffer"


@dataclass
class MoeParamInfo:
    """Information about an MoE parameter."""

    type: MoeParamType
    moe_prefix: str
    num_moe_experts: int
    local_expert_indices: list  # Expert indices this rank should load (for EP)


HF_BUFFER_NAME_MAP = {
    "base": {"router": "gate"},
    "kimi_k25": {"expert_bias": "e_score_correction_bias"},
    "glm_moe_dsa": {"expert_bias": "e_score_correction_bias"},
}


class DistParamLoader:
    def __init__(
        self,
        model,
        path,
        rank,
        mesh,
        world_mesh,
        device,
        use_transformer_engine=False,
    ):
        self.model = model
        self.config = model.config
        self.path = Path(path)
        self.mesh = mesh
        self.ep_mesh = mesh["ep"]
        self.ep_group = self.ep_mesh.get_group()
        self.world_mesh = world_mesh
        self.rank = rank
        self.world_size = dist.get_world_size()
        self.device = device
        self.ep_rank = dist.get_rank(group=self.ep_group)
        self.ep_world_size = dist.get_world_size(group=self.ep_group)
        self.use_te = use_transformer_engine
        # log related
        self.start_time = time.perf_counter()
        self.prev_time = self.start_time
        # build meta
        self.param_to_filename = self._load_index_json()
        # build module mapping for MoE detection (must be before _build_checkpoint_mapping)
        self._moe_param_info = self._build_moe_param_info()
        # build checkpoint key mapping: non-expert -> owner rank, expert -> whether to load
        self._checkpoint_owner_map, self._local_expert_keys = (
            self._build_checkpoint_mapping()
        )
        self.local_sd = self._load_local_shards()

    def _load_index_json(self):
        index_path = self.path / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        with open(index_path) as f:
            index = json.load(f)
        return index["weight_map"]

    def _get_layer_id(self, param_name):
        """Extract layer ID from parameter name.

        Works with any prefix (model, vision_tower, multi_modal_projector, etc.)
        by finding the number after 'layers' using regex.
        """
        match = re.search(r"\.layers\.(\d+)\.", param_name)
        if match:
            return int(match.group(1))
        return None

    def _sync_log_rank0(self, content, update_prev=True):
        torch.cuda.synchronize()
        dist.barrier()
        cur = time.perf_counter()
        if self.rank == 0:
            print(
                f"{content} in {cur - self.prev_time:.2f}s (total {cur - self.start_time:.2f}s)",
                flush=True,
            )
        if update_prev:
            self.prev_time = cur

    def _build_checkpoint_mapping(self):
        """Build checkpoint key mappings for distributed loading.

        Returns:
            checkpoint_owner_map: Dict mapping non-expert checkpoint keys to owner rank.
                                  Layer params are distributed across ranks for load balancing.
            local_expert_keys: Set of expert checkpoint keys this EP rank should load.
        """
        checkpoint_owner_map = {}
        local_expert_keys = set()

        # Build non-expert owner map from checkpoint index
        for param_name in self.param_to_filename.keys():
            if ".experts." not in param_name:
                layer_id = self._get_layer_id(param_name)
                if layer_id is not None:
                    checkpoint_owner_map[param_name] = (layer_id + 1) % self.world_size
                else:
                    checkpoint_owner_map[param_name] = 0

        # Build expert keys from model parameters
        # Group by moe_prefix to avoid duplicate processing
        seen_moe_layers = set()
        for name, param in self.model.named_parameters():
            info = self._moe_param_info.get(name)
            if info is None or info.type != MoeParamType.EXPERTS:
                continue
            if info.moe_prefix in seen_moe_layers:
                continue
            seen_moe_layers.add(info.moe_prefix)

            hf_prefix = info.moe_prefix.replace("moe_layer", "mlp")
            for idx in info.local_expert_indices:
                # Add all expert weight types for this layer
                local_expert_keys.add(f"{hf_prefix}.experts.{idx}.gate_proj.weight")
                local_expert_keys.add(f"{hf_prefix}.experts.{idx}.up_proj.weight")
                local_expert_keys.add(f"{hf_prefix}.experts.{idx}.down_proj.weight")

        return checkpoint_owner_map, local_expert_keys

    def _load_local_shards(self):
        """Load checkpoint tensors that belong to this rank.

        For non-expert params: load based on layer ownership.
        For expert params: only load tensors for this EP rank's expert indices.
        """
        # Non-expert params: files based on checkpoint ownership
        local_params = {
            name: filename
            for name, filename in self.param_to_filename.items()
            if name in self._checkpoint_owner_map
            and self._checkpoint_owner_map[name] == self.rank
        }

        # Expert params: files based on local expert keys
        expert_params = {
            name: filename
            for name, filename in self.param_to_filename.items()
            if name in self._local_expert_keys
        }

        local_sd = {}
        files_needed = set(local_params.values()) | set(expert_params.values())

        if files_needed:
            files = [self.path / f for f in sorted(files_needed)]
            print(
                f"[Rank {self.rank}] Loading {len(files)} files with {len(local_params)} non-expert + {len(expert_params)} expert parameters...",
                flush=True,
            )
            t0 = time.perf_counter()
            for f in files:
                file_params = load_file(f.as_posix(), device="cpu")
                for param_name, tensor in file_params.items():
                    if ".experts." in param_name:
                        # Expert param: load only if this EP rank needs it
                        if param_name in self._local_expert_keys:
                            local_sd[param_name] = tensor
                    elif param_name in local_params:
                        # Non-expert param: load if this rank owns it
                        local_sd[param_name] = tensor
            print(
                f"[Rank {self.rank}] Loaded {len(local_sd)} params in {time.perf_counter() - t0:.2f}s",
                flush=True,
            )
        else:
            print(f"[Rank {self.rank}] No parameters to load")

        self._sync_log_rank0("[Load File] Load finished")
        return local_sd

    def _build_moe_param_info(self):
        """Build mapping from parameter names to MoE module info.

        Returns a dict mapping param/buffer name to MoeParamInfo.
        """
        moe_info = {}
        for name, module in self.model.named_modules():
            if isinstance(module, BaseMoELayer):
                num_experts = module.config.num_moe_experts
                local_expert_indices = module.local_expert_indices
                # Found an MoE layer, mark all its parameters
                for param_name, _ in module.named_parameters(recurse=True):
                    full_name = f"{name}.{param_name}" if name else param_name

                    # Ignore shared experts
                    if "shared_experts" in param_name:
                        continue

                    if "experts" in param_name:
                        moe_info[full_name] = MoeParamInfo(
                            type=MoeParamType.EXPERTS,
                            moe_prefix=name,
                            num_moe_experts=num_experts,
                            local_expert_indices=local_expert_indices,
                        )
                    elif "router" in param_name:
                        moe_info[full_name] = MoeParamInfo(
                            type=MoeParamType.ROUTER,
                            moe_prefix=name,
                            num_moe_experts=num_experts,
                            local_expert_indices=local_expert_indices,
                        )
                # Mark buffers too
                for buf_name, _ in module.named_buffers(recurse=True):
                    full_name = f"{name}.{buf_name}" if name else buf_name
                    # Only expert_bias now
                    if "router.expert_bias" in buf_name:
                        moe_info[full_name] = MoeParamInfo(
                            type=MoeParamType.ROUTER_BUFFER,
                            moe_prefix=name,
                            num_moe_experts=num_experts,
                            local_expert_indices=local_expert_indices,
                        )
        return moe_info

    def _get_moe_param_type(self, name) -> Optional[MoeParamType]:
        """Get MoE parameter type, or None if not an MoE parameter."""
        info = self._moe_param_info.get(name)
        return info.type if info else None

    def _get_hf_mlp_prefix(self, name):
        """Get HF checkpoint prefix (using 'mlp') from model parameter name.

        The model may use 'moe_layer' or 'mlp' as the module name, but HF checkpoints
        always use 'mlp'. This converts the prefix accordingly.
        """
        info = self._moe_param_info.get(name)
        if info:
            # e.g., 'model.layers.0.moe_layer' -> 'model.layers.0.mlp'
            return info.moe_prefix.replace("moe_layer", "mlp")
        # Fallback: extract prefix from name
        if ".experts" in name:
            return name[: name.rfind(".experts")]
        return name[:-8] if name.endswith(".weight") else name

    def _broadcast_tensor(self, name, shape=None, dtype=None):
        owner = self._checkpoint_owner_map[name]
        if self.rank == owner:
            tensor = self.local_sd[name].to(self.device, dtype=dtype)
        else:
            tensor = torch.empty(shape, device=self.device, dtype=dtype)
        dist.broadcast(tensor, src=owner)
        return tensor

    def _get_expert_keys(self, prefix, is_weight1, local_expert_indices):
        """Get checkpoint keys for the given expert indices.

        Args:
            prefix: HF checkpoint prefix (e.g., 'model.layers.0.mlp')
            is_weight1: True for gate+up weights, False for down weights
            local_expert_indices: List of expert indices this rank should load
        """
        keys = []
        for i in local_expert_indices:
            if is_weight1:
                keys.extend(
                    [f"{prefix}.{i}.gate_proj.weight", f"{prefix}.{i}.up_proj.weight"]
                )
            else:
                keys.append(f"{prefix}.{i}.down_proj.weight")
        return keys

    def _load_experts_to_device(self):
        """Move expert tensors from CPU to GPU.

        With EP-direct loading, each rank has already loaded only its expert tensors,
        so we just need to move them to GPU.
        """
        for name, _ in self.model.named_parameters():
            if self._get_moe_param_type(name) != MoeParamType.EXPERTS:
                continue

            info = self._moe_param_info[name]
            prefix = self._get_hf_mlp_prefix(name)
            prefix += ".experts"
            is_weight1 = "weight1" in name

            # Only load expert tensors for this EP rank's indices
            for key in self._get_expert_keys(
                prefix, is_weight1, info.local_expert_indices
            ):
                if key in self.local_sd:
                    self.local_sd[key] = self.local_sd[key].to(self.device)

    def _te_load_experts_to_device(self):
        """Move expert tensors from CPU to GPU for Transformer Engine backend."""
        for name, _ in self.model.named_parameters():
            if (
                self._get_moe_param_type(name) != MoeParamType.EXPERTS
                or "weight0" not in name
            ):
                continue

            info = self._moe_param_info[name]
            prefix = self._get_hf_mlp_prefix(name)
            prefix += ".experts"
            is_fc1 = "fc1_linear" in name

            # Only load expert tensors for this EP rank's indices
            for key in self._get_expert_keys(prefix, is_fc1, info.local_expert_indices):
                if key in self.local_sd:
                    self.local_sd[key] = self.local_sd[key].to(self.device)

    def _assemble_expert_tensor(self, name, prefix, is_weight1, local_expert_indices):
        """Assemble expert tensor from locally loaded checkpoint tensors.

        No broadcast needed - each EP rank has its own expert tensors already loaded.
        """
        weights = []
        for idx in local_expert_indices:
            if is_weight1:
                gate = self.local_sd.pop(f"{prefix}.{idx}.gate_proj.weight")
                up = self.local_sd.pop(f"{prefix}.{idx}.up_proj.weight")
                weights.append(torch.cat([gate, up]))
            else:
                down = self.local_sd.pop(f"{prefix}.{idx}.down_proj.weight")
                weights.append(down.t())

        tensor = torch.stack(weights, dim=0)
        tensor = tensor.permute(0, 2, 1) if is_weight1 else tensor
        tensor = tensor.reshape(-1, tensor.shape[-1]).contiguous()
        return tensor

    def _te_assemble_expert_tensor(self, name, prefix, is_fc1, local_expert_indices):
        """Assemble expert tensor for Transformer Engine backend.

        TE uses weight0/weight1 format with individual expert indices in the tensor name.
        """
        # For TE, we only have one expert index per tensor (weight0 or weight1 per expert)
        # local_expert_indices should have exactly one element for TE
        if len(local_expert_indices) != 1:
            raise ValueError(
                f"TE backend expects exactly one local expert, got {len(local_expert_indices)}"
            )

        expert_idx = local_expert_indices[0]
        if is_fc1:
            gate = self.local_sd.pop(f"{prefix}.{expert_idx}.gate_proj.weight")
            up = self.local_sd.pop(f"{prefix}.{expert_idx}.up_proj.weight")
            tensor = torch.cat([gate, up])
        else:
            tensor = self.local_sd.pop(f"{prefix}.{expert_idx}.down_proj.weight")
        return tensor

    def _hf_buffer_name_convert(self, name):
        for turbo_name, hf_name in HF_BUFFER_NAME_MAP["base"].items():
            name = name.replace(turbo_name, hf_name)

        model_type = self.config.model_type
        if model_type in HF_BUFFER_NAME_MAP.keys():
            for turbo_name, hf_name in HF_BUFFER_NAME_MAP[model_type].items():
                name = name.replace(turbo_name, hf_name)
        return name

    def _broadcast_buffers(self):
        # Load buffers (e.g., expert bias in router)
        for name, buf in self.model.named_buffers():
            if self._get_moe_param_type(name) == MoeParamType.ROUTER_BUFFER:
                # Get HF prefix for checkpoint lookup
                hf_prefix = self._get_hf_mlp_prefix(name)
                # Extract the buffer name suffix (e.g., "router.expert_bias")
                info = self._moe_param_info.get(name)
                if info:
                    moe_prefix = info.moe_prefix
                    suffix = name[len(moe_prefix) + 1 :]  # +1 for the dot
                    # Replace router with gate for HF convention
                    hf_suffix = suffix.replace("router", "gate")
                    hf_buffer_name = f"{hf_prefix}.{hf_suffix}"
                else:
                    hf_buffer_name = name.replace("router", "gate")
                hf_buffer_name = self._hf_buffer_name_convert(hf_buffer_name)
                full = self._broadcast_tensor(hf_buffer_name, buf.shape, buf.dtype)
                buf.copy_(full)

    def _broadcast_params(self):
        num_param_tensors = len(list(self.model.parameters()))
        i = 0
        for name, param in self.model.named_parameters():
            moe_type = self._get_moe_param_type(name)
            if moe_type == MoeParamType.EXPERTS:
                info = self._moe_param_info[name]
                prefix = self._get_hf_mlp_prefix(name)
                prefix += ".experts"
                if self.use_te:
                    is_fc1 = "fc1_linear" in name
                    full = self._te_assemble_expert_tensor(
                        name, prefix, is_fc1, info.local_expert_indices
                    )
                else:
                    is_weight1 = "weight1" in name
                    full = self._assemble_expert_tensor(
                        name, prefix, is_weight1, info.local_expert_indices
                    )
                mesh = self.mesh["edp"]
            elif moe_type == MoeParamType.ROUTER:
                # Get HF prefix and convert router -> gate for HF convention
                hf_prefix = self._get_hf_mlp_prefix(name)
                info = self._moe_param_info.get(name)
                if info:
                    moe_prefix = info.moe_prefix
                    suffix = name[len(moe_prefix) + 1 :]  # +1 for the dot
                    hf_suffix = suffix.replace("router", "gate")
                    hf_param_name = f"{hf_prefix}.{hf_suffix}"
                else:
                    hf_param_name = name.replace("router", "gate")
                full = self._broadcast_tensor(hf_param_name, param.shape, param.dtype)
                mesh = self.world_mesh
            else:
                full = self._broadcast_tensor(name, param.shape, param.dtype)
                mesh = self.world_mesh

            sharded_tensor = distribute_tensor(
                full, mesh, param.placements, src_data_rank=None
            )

            with torch.no_grad():
                param.copy_(nn.Parameter(sharded_tensor))
            del sharded_tensor, full
            i += 1
            mem_free, mem_total = torch.cuda.mem_get_info()
            mem_used = mem_total - mem_free
            utilization = mem_used / mem_total
            self._sync_log_rank0(
                f"[Rank {self.rank}, mem {utilization:.1%}] Loaded {i}/{num_param_tensors} params",
                False,
            )
            if utilization > 0.8:
                torch.cuda.empty_cache()
                gc.collect()
                new_free, _ = torch.cuda.mem_get_info()
                freed_gb = (new_free - mem_free) / 1024**3
                print(
                    f"Rank {self.rank} Cleaned {freed_gb:.2f} GB ({new_free} - {mem_free}) GPU memory (utilization: {utilization:.1%} -> {(mem_total - new_free)/mem_total:.1%})",
                    flush=True,
                )

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
        # create empty tensor for model params on target device
        self.model.to_empty(device=self.device)

        # move experts from cpu to gpu
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
