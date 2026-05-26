import re
import os
import json
import shutil
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import psutil
import gc

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from safetensors.torch import save_file

from .weight_loader import HF_BUFFER_NAME_MAP
from torchturbo.models.modules.comm import all_gather_tensor, all_gather_tensor_chunked
from torchturbo.models.modules.moe.moe_layer import BaseMoELayer


class MoeParamType(Enum):
    """Types of MoE parameters."""

    EXPERT = "expert"
    ROUTER = "router"
    ROUTER_BUFFER = "router_buffer"


@dataclass
class MoeParamInfo:
    """Information about an MoE parameter."""

    type: MoeParamType
    moe_prefix: str
    num_moe_experts: int
    local_expert_indices: list
    hidden_size: int = None
    moe_ffn_hidden_size: int = None
    num_local_experts: int = None


class DistParamSaver:
    def __init__(
        self,
        model,
        rank,
        mesh,
        use_transformer_engine,
        path=None,
        processing_class=None,
    ):
        self.model = model

        self.rank = rank
        self.mesh = mesh
        self.ep_mesh = mesh["ep"]
        self.ep_group = self.ep_mesh.get_group(0)
        self.ep_size = dist.get_world_size(group=self.ep_group)

        self.use_te = use_transformer_engine
        self.processing_class = processing_class
        self.src_path = Path(path) if path else None

        config = model.config
        self.model_type = config.model_type if hasattr(config, "model_type") else None

        # Build MoE parameter info mapping using isinstance checks
        self._moe_param_info = self._build_moe_param_info()

    def _build_moe_param_info(self):
        """Build mapping from parameter names to MoE module info.

        Uses isinstance checks to detect MoE layers instead of string matching.

        Returns a dict mapping param/buffer name to MoeParamInfo.
        """
        moe_info = {}
        for name, module in self.model.named_modules():
            if isinstance(module, BaseMoELayer):
                moe_config = module.config
                num_experts = moe_config.num_moe_experts
                hidden_size = moe_config.hidden_size
                moe_ffn_hidden_size = getattr(
                    moe_config,
                    "moe_ffn_hidden_size",
                    getattr(moe_config, "moe_intermediate_size", None),
                )
                num_local_experts = module.num_local_experts
                local_expert_indices = module.local_expert_indices

                # Found an MoE layer, mark all its parameters
                for param_name, _ in module.named_parameters(recurse=True):
                    # Ignore shared experts
                    if "shared_experts" in param_name:
                        continue

                    full_name = f"{name}.{param_name}" if name else param_name

                    if "experts" in param_name:
                        moe_info[full_name] = MoeParamInfo(
                            type=MoeParamType.EXPERT,
                            moe_prefix=name,
                            num_moe_experts=num_experts,
                            local_expert_indices=local_expert_indices,
                            hidden_size=hidden_size,
                            moe_ffn_hidden_size=moe_ffn_hidden_size,
                            num_local_experts=num_local_experts,
                        )
                    elif "router" in param_name:
                        moe_info[full_name] = MoeParamInfo(
                            type=MoeParamType.ROUTER,
                            moe_prefix=name,
                            num_moe_experts=num_experts,
                            local_expert_indices=local_expert_indices,
                            hidden_size=hidden_size,
                            moe_ffn_hidden_size=moe_ffn_hidden_size,
                            num_local_experts=num_local_experts,
                        )

                # Mark buffers too
                for buf_name, _ in module.named_buffers(recurse=True):
                    full_name = f"{name}.{buf_name}" if name else buf_name
                    if "router" in buf_name:
                        moe_info[full_name] = MoeParamInfo(
                            type=MoeParamType.ROUTER_BUFFER,
                            moe_prefix=name,
                            num_moe_experts=num_experts,
                            local_expert_indices=local_expert_indices,
                            hidden_size=hidden_size,
                            moe_ffn_hidden_size=moe_ffn_hidden_size,
                            num_local_experts=num_local_experts,
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
        # Fallback: extract prefix from name (consistent with weight_loader)
        if ".experts." in name:
            return name[: name.rfind(".experts.")]
        return name

    def _load_index_json(self, path):
        """Load the model.safetensors.index.json file."""
        index_path = Path(path) / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        with open(index_path) as f:
            index = json.load(f)
        return index["weight_map"], index.get("metadata", {})

    def _copy_non_safetensor_files(self, src_path, dst_path):
        """Copy all non-safetensor files from source to destination."""
        dst_path = Path(dst_path)
        dst_path.mkdir(parents=True, exist_ok=True)

        for item in Path(src_path).iterdir():
            if item.is_file():
                # Skip safetensors files and index json
                if (
                    item.suffix == ".safetensors"
                    or item.name == "model.safetensors.index.json"
                ):
                    continue
                shutil.copy2(item, dst_path / item.name)
            elif item.is_dir():
                # Copy directories recursively
                shutil.copytree(item, dst_path / item.name, dirs_exist_ok=True)

    def _distribute_files_to_ranks(self, weight_map, world_size):
        """Distribute safetensor files across ranks based on number of files."""
        files = sorted(set(weight_map.values()))
        file_to_rank = {}
        for i, filename in enumerate(files):
            file_to_rank[filename] = i % world_size
        return file_to_rank, files

    def _get_local_params_to_save(self, weight_map, world_size):
        """Get the parameter names and files that this rank needs to save."""
        file_to_rank, files = self._distribute_files_to_ranks(weight_map, world_size)
        local_files = {f for f in files if file_to_rank[f] == self.rank}
        local_params = {name for name, fn in weight_map.items() if fn in local_files}
        return local_params, local_files

    def _get_local_hf_state_dict(self, local_params):
        """Build state_dict containing only the parameters this rank needs to save.

        Uses MoE parameter info mapping to detect MoE layers instead of string matching.
        """
        full_sd = {}

        cleanup_freq = 50
        param_count = 0

        def _gc_cleanup():
            gc.collect()
            gpu_mem = self._get_gpu_memory_usage_percent()
            cpu_mem = self._get_cpu_memory_usage_percent()
            if torch.distributed.get_rank() == 0:
                print(f"⚠️  Memory alert - GPU: {gpu_mem:.1f}%, CPU: {cpu_mem:.1f}%")

        # Cache for gathered expert tensors to avoid redundant gathering
        # Key: (moe_prefix, is_weight1), Value: gathered tensor
        gathered_expert_tensors = {}

        for name, param in self.model.named_parameters():
            param_count += 1
            if torch.distributed.get_rank() == 0:
                print(f"[saving] param_name: {name}")

            # Gather DTensor to full tensor
            if isinstance(param, DTensor):
                full_tensor = param.full_tensor()
            else:
                full_tensor = param

            moe_type = self._get_moe_param_type(name)

            if moe_type == MoeParamType.EXPERT:
                info = self._moe_param_info[name]
                hf_prefix = self._get_hf_mlp_prefix(name)
                num_experts = info.num_moe_experts
                num_local_experts = info.num_local_experts
                hidden_size = info.hidden_size
                moe_ffn_hidden_size = info.moe_ffn_hidden_size

                if hidden_size is None or moe_ffn_hidden_size is None:
                    raise ValueError(
                        f"Cannot find hidden_size or moe_ffn_hidden_size for {name}"
                    )

                # Determine if this is weight1 (gate+up) or weight2 (down)
                is_weight1 = "weight1" in name

                # Use cached gathered tensor if available
                cache_key = (info.moe_prefix, is_weight1)
                if cache_key in gathered_expert_tensors:
                    gathered = gathered_expert_tensors[cache_key]
                else:
                    # Gather from expert parallel ranks
                    gathered = all_gather_tensor_chunked(
                        full_tensor, group=self.ep_group, num_chunks=self.ep_size
                    )
                    gathered_expert_tensors[cache_key] = gathered

                if is_weight1:
                    # weight1 contains gate_proj and up_proj combined
                    # Local shape: [hidden_size * num_local_experts, moe_ffn_hidden_size * 2]
                    # Gathered shape: [hidden_size * num_experts, moe_ffn_hidden_size * 2]
                    # The gathered tensor is already in natural expert order:
                    # [E0, E1, ..., E_{num_local-1} from rank 0, E0, E1, ... from rank 1, ...]
                    # Since local_expert_indices are assigned as [ep_rank * num_local + i for i in range(num_local)],
                    # the concatenation by rank order produces correct global expert order.

                    # Split into individual experts
                    experts = torch.split(gathered, hidden_size)

                    for i in range(num_experts):
                        gate_proj_name = f"{hf_prefix}.experts.{i}.gate_proj.weight"
                        up_proj_name = f"{hf_prefix}.experts.{i}.up_proj.weight"

                        # Transpose and split gate/up
                        # expert[i] shape: [H, F*2] -> [F*2, H]
                        expert_weight = experts[i].t()
                        # Split into [F, H] each
                        gate_up_split = torch.split(expert_weight, moe_ffn_hidden_size)

                        if gate_proj_name in local_params:
                            full_sd[gate_proj_name] = gate_up_split[0].cpu()
                        if up_proj_name in local_params:
                            full_sd[up_proj_name] = gate_up_split[1].cpu()
                        del expert_weight, gate_up_split

                else:
                    # weight2 contains down_proj
                    # Local shape: [moe_ffn_hidden_size * num_local_experts, hidden_size]
                    # Gathered shape: [moe_ffn_hidden_size * num_experts, hidden_size]
                    # Same as weight1: gathered tensor is already in natural expert order.

                    # Split into individual experts
                    experts = torch.split(gathered, moe_ffn_hidden_size)

                    for i in range(num_experts):
                        down_proj_name = f"{hf_prefix}.experts.{i}.down_proj.weight"
                        if down_proj_name in local_params:
                            # Transpose for HF format: [F, H] -> [H, F]
                            full_sd[down_proj_name] = experts[i].t().cpu()

                del full_tensor
                del gathered, experts

            elif moe_type == MoeParamType.ROUTER:
                # Router weights - rename router -> gate for HF convention
                info = self._moe_param_info[name]
                hf_prefix = self._get_hf_mlp_prefix(name)

                # Get the suffix after moe_prefix
                suffix = name[len(info.moe_prefix) + 1 :]  # +1 for the dot
                hf_suffix = suffix.replace("router", "gate")
                new_name = f"{hf_prefix}.{hf_suffix}"

                # Apply model-specific buffer name mapping
                if self.model_type in HF_BUFFER_NAME_MAP:
                    new_name = new_name.replace(
                        "expert_bias",
                        HF_BUFFER_NAME_MAP[self.model_type].get(
                            "expert_bias", "expert_bias"
                        ),
                    )

                if new_name in local_params:
                    full_sd[new_name] = full_tensor.cpu()
                del full_tensor

            else:
                # Non-MoE parameter
                if name in local_params:
                    full_sd[name] = full_tensor.cpu()
                del full_tensor

            if param_count % cleanup_freq == 0:
                _gc_cleanup()

        # Handle buffers (e.g., expert_bias in router)
        for name, buf in self.model.named_buffers():
            param_count += 1
            if torch.distributed.get_rank() == 0:
                print(f"[saving] buf_name: {name}")

            moe_type = self._get_moe_param_type(name)

            if moe_type == MoeParamType.ROUTER_BUFFER:
                info = self._moe_param_info[name]
                hf_prefix = self._get_hf_mlp_prefix(name)

                # Get the suffix after moe_prefix
                suffix = name[len(info.moe_prefix) + 1 :]  # +1 for the dot
                hf_suffix = suffix.replace("router", "gate")
                new_name = f"{hf_prefix}.{hf_suffix}"

                # Apply model-specific buffer name mapping
                if self.model_type in HF_BUFFER_NAME_MAP:
                    new_name = new_name.replace(
                        "expert_bias",
                        HF_BUFFER_NAME_MAP[self.model_type].get(
                            "expert_bias", "expert_bias"
                        ),
                    )

                if new_name in local_params:
                    if isinstance(buf, DTensor):
                        full_sd[new_name] = buf.full_tensor().cpu()
                    else:
                        full_sd[new_name] = buf.cpu()
            else:
                # Non-MoE buffer
                # Apply model-specific buffer name mapping
                new_name = name
                if self.model_type in HF_BUFFER_NAME_MAP:
                    if "router" in name and "expert_bias" in name:
                        new_name = name.replace("router", "gate")
                        new_name = new_name.replace(
                            "expert_bias",
                            HF_BUFFER_NAME_MAP[self.model_type].get(
                                "expert_bias", "expert_bias"
                            ),
                        )
                if new_name in local_params:
                    if isinstance(buf, DTensor):
                        full_sd[new_name] = buf.full_tensor().cpu()
                    else:
                        full_sd[new_name] = buf.cpu()

            del buf
            if param_count % cleanup_freq == 0:
                _gc_cleanup()

        return full_sd

    def _get_gpu_memory_usage_percent(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            if reserved > 0:
                return (allocated / reserved) * 100
        return 0.0

    def _get_cpu_memory_usage_percent(self):
        return psutil.virtual_memory().percent

    def _get_te_local_hf_state_dict(self, local_params):
        """Build state_dict containing only the parameters this rank needs to save (TE version).

        Uses MoE parameter info mapping to detect MoE layers instead of string matching.
        """
        full_sd = {}

        cleanup_freq = 50
        param_count = 0

        def _gc_cleanup():
            gc.collect()
            gpu_mem = self._get_gpu_memory_usage_percent()
            cpu_mem = self._get_cpu_memory_usage_percent()
            if torch.distributed.get_rank() == 0:
                print(f"⚠️  Memory alert - GPU: {gpu_mem:.1f}%, CPU: {cpu_mem:.1f}%")

        # Cache for gathered expert tensors
        gathered_expert_tensors = {}

        for name, param in self.model.named_parameters():
            param_count += 1
            if torch.distributed.get_rank() == 0:
                print(f"[saving] param_name: {name}")

            # Gather DTensor to full tensor
            if isinstance(param, DTensor):
                full_tensor = param.full_tensor()
            else:
                full_tensor = param

            moe_type = self._get_moe_param_type(name)

            if moe_type == MoeParamType.EXPERT:
                info = self._moe_param_info[name]
                hf_prefix = self._get_hf_mlp_prefix(name)
                num_experts = info.num_moe_experts
                num_local_experts = info.num_local_experts
                hidden_size = info.hidden_size
                moe_ffn_hidden_size = info.moe_ffn_hidden_size

                if hidden_size is None or moe_ffn_hidden_size is None:
                    raise ValueError(
                        f"Cannot find hidden_size or moe_ffn_hidden_size for {name}"
                    )

                # Determine if this is fc1 (gate+up) or fc2 (down)
                is_fc1 = "fc1_linear" in name

                # For TE, extract local expert index from parameter name
                # TE format: layers.X.mlp.experts.fc1_linear.weight_Y where Y is local expert idx
                local_expert_match = re.search(r"weight_(\d+)$", name)
                if local_expert_match:
                    local_expert_idx = int(local_expert_match.group(1))
                else:
                    # Fallback: extract from parameter name
                    local_expert_idx = int(re.findall(r"\d+", name.split(".")[-1])[0])

                # Create cache key that includes local_expert_idx
                cache_key = (info.moe_prefix, is_fc1, local_expert_idx)
                if cache_key in gathered_expert_tensors:
                    gathered = gathered_expert_tensors[cache_key]
                else:
                    # For TE, gather from expert parallel ranks
                    # all_gather_tensor concatenates along first dim by rank order
                    gathered = all_gather_tensor(full_tensor, group=self.ep_group)
                    gathered_expert_tensors[cache_key] = gathered

                if is_fc1:
                    # fc1_linear contains gate_proj and up_proj combined
                    # Local shape: [moe_ffn_hidden_size * 2, hidden_size] (TE uses transposed)
                    # Gathered shape: [ep_size * moe_ffn_hidden_size * 2, hidden_size]
                    # But we process one local_expert_idx at a time

                    # Split by ep_size to get each rank's contribution
                    experts_fc1 = torch.split(gathered, moe_ffn_hidden_size * 2)

                    for ep_rank in range(self.ep_size):
                        # Calculate global expert index
                        expert_idx = ep_rank * num_local_experts + local_expert_idx

                        gate_proj_name = (
                            f"{hf_prefix}.experts.{expert_idx}.gate_proj.weight"
                        )
                        up_proj_name = (
                            f"{hf_prefix}.experts.{expert_idx}.up_proj.weight"
                        )

                        # Split gate and up: [F*2, H] -> two [F, H] tensors
                        gate_up_split = torch.split(
                            experts_fc1[ep_rank], moe_ffn_hidden_size
                        )

                        if gate_proj_name in local_params:
                            full_sd[gate_proj_name] = gate_up_split[0].cpu()
                        if up_proj_name in local_params:
                            full_sd[up_proj_name] = gate_up_split[1].cpu()
                        del gate_up_split
                    del experts_fc1

                else:
                    # fc2_linear is down_proj
                    # Local shape: [hidden_size, moe_ffn_hidden_size] (TE uses transposed)
                    # Gathered shape: [ep_size * hidden_size, moe_ffn_hidden_size]

                    # Split by ep_size to get each rank's contribution
                    experts_fc2 = torch.split(gathered, hidden_size)

                    for ep_rank in range(self.ep_size):
                        # Calculate global expert index
                        expert_idx = ep_rank * num_local_experts + local_expert_idx

                        down_proj_name = (
                            f"{hf_prefix}.experts.{expert_idx}.down_proj.weight"
                        )
                        if down_proj_name in local_params:
                            full_sd[down_proj_name] = experts_fc2[ep_rank].cpu()
                    del experts_fc2

                del full_tensor

            elif moe_type == MoeParamType.ROUTER:
                # Router weights - rename router -> gate for HF convention
                info = self._moe_param_info[name]
                hf_prefix = self._get_hf_mlp_prefix(name)

                # Get the suffix after moe_prefix
                suffix = name[len(info.moe_prefix) + 1 :]  # +1 for the dot
                hf_suffix = suffix.replace("router", "gate")
                new_name = f"{hf_prefix}.{hf_suffix}"

                if new_name in local_params:
                    full_sd[new_name] = full_tensor.cpu()
                del full_tensor

            else:
                # Non-MoE parameter
                if name in local_params:
                    full_sd[name] = full_tensor.cpu()
                del full_tensor

            if param_count % cleanup_freq == 0:
                _gc_cleanup()

        # Handle buffers
        for name, buf in self.model.named_buffers():
            param_count += 1
            if torch.distributed.get_rank() == 0:
                print(f"[saving] buf_name: {name}")

            moe_type = self._get_moe_param_type(name)

            if moe_type == MoeParamType.ROUTER_BUFFER:
                info = self._moe_param_info[name]
                hf_prefix = self._get_hf_mlp_prefix(name)

                # Get the suffix after moe_prefix
                suffix = name[len(info.moe_prefix) + 1 :]
                hf_suffix = suffix.replace("router", "gate")
                new_name = f"{hf_prefix}.{hf_suffix}"

                # Apply model-specific buffer name mapping
                if self.model_type in HF_BUFFER_NAME_MAP:
                    new_name = new_name.replace(
                        "expert_bias",
                        HF_BUFFER_NAME_MAP[self.model_type].get(
                            "expert_bias", "expert_bias"
                        ),
                    )

                if new_name in local_params:
                    if isinstance(buf, DTensor):
                        full_sd[new_name] = buf.full_tensor().cpu()
                    else:
                        full_sd[new_name] = buf.cpu()
            else:
                # Non-MoE buffer
                new_name = name
                if self.model_type in HF_BUFFER_NAME_MAP:
                    if "router" in name and "expert_bias" in name:
                        new_name = name.replace("router", "gate")
                        new_name = new_name.replace(
                            "expert_bias",
                            HF_BUFFER_NAME_MAP[self.model_type].get(
                                "expert_bias", "expert_bias"
                            ),
                        )
                if new_name in local_params:
                    if isinstance(buf, DTensor):
                        full_sd[new_name] = buf.full_tensor().cpu()
                    else:
                        full_sd[new_name] = buf.cpu()

            del buf
            if param_count % cleanup_freq == 0:
                _gc_cleanup()

        return full_sd

    def save_model(self, save_path, use_hf_format: bool = True):
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        save_path = Path(save_path)
        if use_hf_format:
            save_path = save_path / "huggingface"

        if rank == 0:
            save_path.mkdir(parents=True, exist_ok=True)

        torch.distributed.barrier()  # Ensure directory is created

        assert self.src_path is not None, "must be"

        weight_map, metadata = self._load_index_json(self.src_path)
        local_params, local_files = self._get_local_params_to_save(
            weight_map, world_size
        )

        # Each rank only collects the parameters it needs to save
        if self.use_te:
            state_dict = self._get_te_local_hf_state_dict(local_params)
        else:
            state_dict = self._get_local_hf_state_dict(local_params)

        # Convert tensors to bfloat16 and ensure contiguous
        for k in list(state_dict.keys()):
            v = state_dict[k]
            if isinstance(v, torch.Tensor):
                v_bf16 = v.to(torch.bfloat16)
                if not v_bf16.is_contiguous():
                    v_bf16 = v_bf16.contiguous()
                state_dict[k] = v_bf16
                del v, v_bf16

        # Save shards assigned to this rank
        for filename in local_files:
            shard_dict = {}
            for param_name, fn in weight_map.items():
                if fn == filename and param_name in state_dict:
                    shard_dict[param_name] = state_dict[param_name]
            if shard_dict:
                print(
                    f"[Rank {rank}] Saving {filename} with {len(shard_dict)} params..."
                )
                save_file(shard_dict, (save_path / filename).as_posix())

        torch.distributed.barrier()  # Wait for all ranks to finish saving shards

        # Rank 0 saves index and copies other files
        if rank == 0:
            # Reconstruct full weight_map for index (only need param_name -> filename mapping)
            new_weight_map = {
                param_name: filename for param_name, filename in weight_map.items()
            }

            index_content = {"metadata": metadata, "weight_map": new_weight_map}
            with open(save_path / "model.safetensors.index.json", "w") as f:
                json.dump(index_content, f, indent=2)

            self._copy_non_safetensor_files(self.src_path, save_path)

            if self.processing_class is not None:
                self.processing_class.save_pretrained(save_path)

            print(f"Rank: {rank} - Saved hf model to {save_path.absolute()}")

        torch.distributed.barrier()
