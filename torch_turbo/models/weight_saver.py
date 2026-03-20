import json
import re
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.distributed._tensor import DTensor

from .weight_loader import HF_BUFFER_NAME_MAP
from torch_turbo.distributed.comm import all_gather_tensor, all_gather_tensor_chunked
from torch_turbo.distributed.parallel_state import get_expert_mesh
from torch_turbo.models.registry import get_parallel_plan


class DistParamSaver:
    def __init__(self, model, args, processing_class=None):
        self.model = model
        self.moe_config = get_parallel_plan(model.config).moe_config

        ep_mesh = get_expert_mesh()
        self.ep = ep_mesh["ep"]
        ep_group = self.ep.get_group(0)
        self.ep_size = dist.get_world_size(group=ep_group)
        self.rank = dist.get_rank()

        self.use_te = args.use_transformer_engine
        self.processing_class = processing_class
        self.src_path = Path(args.path) if hasattr(args, 'path') and args.path else None
        self.model_type = model.config.model_type

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
                if item.suffix == '.safetensors' or item.name == 'model.safetensors.index.json':
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
        """Build state_dict containing only the parameters this rank needs to save."""
        full_sd = {}
        if hasattr(self.model.config, "vision_config"):
            config = self.model.config.text_config
        else:
            config = self.model.config

        for name, param in self.model.named_parameters():
            if isinstance(param, DTensor):
                full_tensor = param.full_tensor()
            else:
                full_tensor = param

            if "mlp.experts" in name:
                prefix = name[:-8]
                num_experts = self.moe_config.num_moe_experts
                full_tensor = all_gather_tensor_chunked(full_tensor, ep_name="ep", num_chunks=self.ep_size)
                if "weight1" in name:
                    experts = torch.split(full_tensor, config.hidden_size)
                    for i in range(num_experts):
                        gate_proj = f"{prefix}.{i}.gate_proj.weight"
                        up_proj = f"{prefix}.{i}.up_proj.weight"
                        expert2 = torch.split(experts[i].t(), config.moe_intermediate_size)
                        if gate_proj in local_params:
                            full_sd[gate_proj] = expert2[0].cpu()
                        if up_proj in local_params:
                            full_sd[up_proj] = expert2[1].cpu()
                else:
                    # weight2
                    experts = torch.split(full_tensor, config.moe_intermediate_size)
                    for i in range(num_experts):
                        down_proj = f"{prefix}.{i}.down_proj.weight"
                        if down_proj in local_params:
                            full_sd[down_proj] = experts[i].t().cpu()
            elif "mlp.router" in name:
                new_name = name.replace("router", "gate")
                if self.model_type in HF_BUFFER_NAME_MAP.keys():
                    new_name = new_name.replace("expert_bias", HF_BUFFER_NAME_MAP[self.model_type]["expert_bias"])
                if new_name in local_params:
                    full_sd[new_name] = full_tensor.cpu()
            else:
                if name in local_params:
                    full_sd[name] = full_tensor.cpu()

        # Handle buffers
        for name, buf in self.model.named_buffers():
            if self.model_type in HF_BUFFER_NAME_MAP.keys():
                if "mlp.router" in name and "expert_bias" in name:
                    new_name = name.replace("router", "gate")
                    new_name = new_name.replace("expert_bias", HF_BUFFER_NAME_MAP[self.model_type]["expert_bias"])
                    name = new_name
                    if name in local_params:
                        if isinstance(buf, DTensor):
                            full_sd[name] = buf.full_tensor().cpu()
                        else:
                            full_sd[name] = buf.cpu()

        return full_sd

    def _get_te_local_hf_state_dict(self, local_params):
        """Build state_dict containing only the parameters this rank needs to save (TE version)."""
        full_sd = {}
        if hasattr(self.model.config, "vision_config"):
            config = self.model.config.text_config
        else:
            config = self.model.config

        for name, param in self.model.named_parameters():
            if isinstance(param, DTensor):
                full_tensor = param.full_tensor()
            else:
                full_tensor = param

            if "mlp.experts" in name:
                full_tensor = all_gather_tensor(full_tensor, ep_name="ep")
                prefix_lst = name.split(".")
                prefix = ".".join(name.split(".")[:5])
                num_local_experts = self.moe_config.num_moe_experts // self.ep_size

                if "fc1_linear" in name:
                    experts = torch.split(full_tensor, config.moe_intermediate_size * 2)
                    for i in range(self.ep_size):
                        expert_idx = i * num_local_experts + int(re.findall(r"\d+", prefix_lst[-1])[0])
                        gate_proj = f"{prefix}.{expert_idx}.gate_proj.weight"
                        up_proj = f"{prefix}.{expert_idx}.up_proj.weight"
                        expert2 = torch.split(experts[i], config.moe_intermediate_size)
                        if gate_proj in local_params:
                            full_sd[gate_proj] = expert2[0].cpu()
                        if up_proj in local_params:
                            full_sd[up_proj] = expert2[1].cpu()
                elif "fc2_linear" in name:
                    experts = torch.split(full_tensor, config.hidden_size)
                    for i in range(self.ep_size):
                        expert_idx = i * num_local_experts + int(re.findall(r"\d+", prefix_lst[-1])[0])
                        down_proj = f"{prefix}.{expert_idx}.down_proj.weight"
                        if down_proj in local_params:
                            full_sd[down_proj] = experts[i].cpu()
            elif "mlp.router" in name:
                new_name = name.replace("router", "gate")
                if new_name in local_params:
                    full_sd[new_name] = full_tensor.cpu()
            else:
                if name in local_params:
                    full_sd[name] = full_tensor.cpu()

        # Handle buffers
        for name, buf in self.model.named_buffers():
            if self.model_type in HF_BUFFER_NAME_MAP.keys():
                if "mlp.router" in name and "expert_bias" in name:
                    new_name = name.replace("router", "gate")
                    new_name = new_name.replace("expert_bias", HF_BUFFER_NAME_MAP[self.model_type]["expert_bias"])
                    name = new_name
                    if name in local_params:
                        if isinstance(buf, DTensor):
                            full_sd[name] = buf.full_tensor().cpu()
                        else:
                            full_sd[name] = buf.cpu()

        return full_sd

    def save_model(self, save_path):
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        save_path = Path(save_path)
        if rank == 0:
            save_path.mkdir(parents=True, exist_ok=True)

        torch.distributed.barrier()  # Ensure directory is created

        assert self.src_path is not None, "must be"

        weight_map, metadata = self._load_index_json(self.src_path)
        local_params, local_files = self._get_local_params_to_save(weight_map, world_size)

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
                print(f"[Rank {rank}] Saving {filename} with {len(shard_dict)} params...")
                save_file(shard_dict, (save_path / filename).as_posix())

        torch.distributed.barrier()  # Wait for all ranks to finish saving shards

        # Rank 0 saves index and copies other files
        if rank == 0:
            # Reconstruct full weight_map for index (only need param_name -> filename mapping)
            new_weight_map = {param_name: filename for param_name, filename in weight_map.items()}

            index_content = {
                "metadata": metadata,
                "weight_map": new_weight_map
            }
            with open(save_path / "model.safetensors.index.json", "w") as f:
                json.dump(index_content, f, indent=2)

            self._copy_non_safetensor_files(self.src_path, save_path)

            if self.processing_class is not None:
                self.processing_class.save_pretrained(save_path)

            print(f"Rank: {rank} - Saved hf model to {save_path.absolute()}")

        torch.distributed.barrier()