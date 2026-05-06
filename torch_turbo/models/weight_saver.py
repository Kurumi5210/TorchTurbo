import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
import psutil
import gc

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.distributed._tensor import DTensor

from .weight_loader import HF_BUFFER_NAME_MAP
from torch_turbo.distributed.comm import all_gather_tensor, all_gather_tensor_chunked
from torch_turbo.distributed.parallel_state import get_expert_mesh
from torch_turbo.models.registry import get_parallel_plan
from torch_turbo.models.registry import get_param_converter

logger = logging.getLogger(__name__)

# If root logger is not configured, add a default handler for this module
if not logging.getLogger().handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    _handler.setLevel(logging.INFO)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


class DistParamSaver:
    def __init__(self, model, args, processing_class=None):
        self.model = model
        self.moe_config = model.moe_config

        ep_mesh = get_expert_mesh()
        self.ep = ep_mesh["ep"]
        ep_group = self.ep.get_group(0)
        self.ep_size = dist.get_world_size(group=ep_group)
        self.rank = dist.get_rank()

        self.use_te = args.use_transformer_engine
        self.processing_class = processing_class
        self.src_path = Path(args.path) if hasattr(args, 'path') and args.path else None
        self.model_type = model.config.model_type
        self.param_converter = get_param_converter(model.config, self.moe_config)

        # Async transfer setup (one-time use)
        self._transfer_stream = torch.cuda.Stream()
        self._pending_events = []

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

    def _save_expert_weight1_to_hf(self, name, full_tensor, local_params, full_sd):
        gates, ups = self.param_converter.convert_expert_weight1_to_hf(name)
        num_experts = self.moe_config.num_moe_experts
        full_tensor = all_gather_tensor(full_tensor, ep_name="ep")
        experts = torch.split(full_tensor, self.config.hidden_size)
        for i in range(num_experts):
            expert2 = torch.split(experts[i].t(), self.config.moe_intermediate_size)
            if gates[i] in local_params:
                full_sd[gates[i]] = self._async_cpu_copy(gates[i], expert2[0])
            if ups[i] in local_params:
                full_sd[ups[i]] = self._async_cpu_copy(ups[i], expert2[1])
            del expert2
        del experts

    def _save_expert_weight2_to_hf(self, name, full_tensor, local_params, full_sd):
        downs = self.param_converter.convert_expert_weight2_to_hf(name)
        num_experts = self.moe_config.num_moe_experts
        full_tensor = all_gather_tensor(full_tensor, ep_name="ep")
        # weight2
        experts = torch.split(full_tensor, self.config.moe_intermediate_size)
        for i in range(num_experts):
            if downs[i] in local_params:
                full_sd[downs[i]] = self._async_cpu_copy(downs[i], experts[i].t())
        del experts

    def _save_router_param_to_hf(self, name, full_tensor, local_params, full_sd):
        hf_param_name = self.param_converter.convert_router_param_to_hf(name)
        if hf_param_name in local_params:
            full_sd[hf_param_name] = self._async_cpu_copy(hf_param_name, full_tensor)

    def _get_local_hf_state_dict(self, local_params):
        """Build state_dict containing only the parameters this rank needs to save."""
        start_time = time.perf_counter()
        rank = torch.distributed.get_rank()
        if rank == 0:
            logger.info(f"_get_local_hf_state_dict starting...")

        full_sd = {}
        if hasattr(self.model.config, "vision_config"):
            self.config = self.model.config.text_config
        else:
            self.config = self.model.config

        cleanup_freq = 50
        param_count = 0

        def _gc_cleanup():
            gpu_before = self._get_gpu_memory_usage_percent()
            cpu_before = self._get_cpu_memory_usage_percent()
            torch.cuda.empty_cache()
            gc.collect()
            gpu_after = self._get_gpu_memory_usage_percent()
            cpu_after = self._get_cpu_memory_usage_percent()
            if torch.distributed.get_rank() == 0:
                logger.info(f"Memory alert - GPU: {gpu_before:.1f}% -> {gpu_after:.1f}% , CPU: {cpu_before:.1f}% -> {cpu_after:.1f}%")

        for name, param in self.model.named_parameters():
            param_count += 1
            if torch.distributed.get_rank() == 0:
                logger.info(f"[saving] param_name: {name}")
            if isinstance(param, DTensor):
                full_tensor = param.full_tensor()
            else:
                full_tensor = param

            if self.param_converter.is_expert_weight1(name):
                self._save_expert_weight1_to_hf(name, full_tensor, local_params, full_sd)
            elif self.param_converter.is_expert_weight2(name):
                self._save_expert_weight2_to_hf(name, full_tensor, local_params, full_sd)
            elif self.param_converter.is_router_param(name):
                self._save_router_param_to_hf(name, full_tensor, local_params, full_sd)
            else:
                if name in local_params:
                    full_sd[name] = self._async_cpu_copy(name, full_tensor)
            del full_tensor

            if self._get_cpu_memory_usage_percent() >= 80 or self._get_gpu_memory_usage_percent() >= 80:
                self._sync_all_transfers()
                _gc_cleanup()

        for name, buf in self.model.named_buffers():
            if self.param_converter.is_persist_buffer(name):
                hf_buffer_name = self.param_converter.convert_persist_buffer(name)
                param_count += 1
                if torch.distributed.get_rank() == 0:
                    logger.info(f"[saving] buf_name: {name}->{hf_buffer_name}")
                if isinstance(buf, DTensor):
                    buf = buf.full_tensor()
                if hf_buffer_name in local_params:
                    full_sd[hf_buffer_name] = self._async_cpu_copy(hf_buffer_name, buf)
                del buf
                if param_count % cleanup_freq == 0:
                    _gc_cleanup()

        # Final sync before returning
        self._sync_all_transfers()
        missing_params = local_params - set(full_sd.keys())
        extra_params = set(full_sd.keys()) - local_params
        if len(full_sd.keys()) != len(local_params):
            logger.error(f"[Rank {self.rank}] Missing params ({len(missing_params)}): {missing_params}")
            logger.error(f"[Rank {self.rank}] Extra params ({len(extra_params)}): {extra_params}")

        elapsed = time.perf_counter() - start_time
        if rank == 0:
            logger.info(f"_get_local_hf_state_dict completed in {elapsed:.2f}s")

        return full_sd

    def _async_cpu_copy(self, name, tensor):
        """Async GPU->CPU copy using pinned memory."""
        dtype = self.param_converter.get_default_hf_dtype(name)
        tensor = tensor.to(dtype)
        if True:
            return tensor.cpu()

        # following code has bug in npu
        # Allocate fresh pinned buffer each time
        cpu_pinned = torch.empty(
            tensor.shape,
            dtype=tensor.dtype,
            device='cpu',
            pin_memory=True
        )

        # Non-blocking copy on dedicated stream
        with torch.cuda.stream(self._transfer_stream):
            cpu_pinned.copy_(tensor, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self._transfer_stream)
            self._pending_events.append(event)

        return cpu_pinned

    def _sync_all_transfers(self):
        """Wait for all pending async transfers to complete."""
        if not self._pending_events:
            return

        for event in self._pending_events:
            event.wait()
        self._pending_events.clear()

        if self._transfer_stream is not None:
            torch.cuda.current_stream().wait_stream(self._transfer_stream)

    def _get_gpu_memory_usage_percent(self):
        mem_free, mem_total = torch.cuda.mem_get_info()
        mem_used = mem_total - mem_free
        utilization = (mem_used / mem_total) * 100.0
        return utilization

    def _get_cpu_memory_usage_percent(self):
        return psutil.virtual_memory().percent

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
                            full_sd[gate_proj] = self._async_cpu_copy(gate_proj, expert2[0])
                        if up_proj in local_params:
                            full_sd[up_proj] = self._async_cpu_copy(up_proj, expert2[1])
                elif "fc2_linear" in name:
                    experts = torch.split(full_tensor, config.hidden_size)
                    for i in range(self.ep_size):
                        expert_idx = i * num_local_experts + int(re.findall(r"\d+", prefix_lst[-1])[0])
                        down_proj = f"{prefix}.{expert_idx}.down_proj.weight"
                        if down_proj in local_params:
                            full_sd[down_proj] = self._async_cpu_copy(down_proj, experts[i])
            elif "mlp.router" in name:
                new_name = name.replace("router", "gate")
                if new_name in local_params:
                    full_sd[new_name] = self._async_cpu_copy(new_name, full_tensor)
            else:
                if name in local_params:
                    full_sd[name] = self._async_cpu_copy(name, full_tensor)

        # Handle buffers
        for name, buf in self.model.named_buffers():
            if buf in self.model._non_persistent_buffers_set:
                continue
            if self.model_type in HF_BUFFER_NAME_MAP.keys():
                if "mlp.router" in name:
                    name = name.replace("router", "gate")
                    if "expert_bias" in name:
                        name = name.replace("expert_bias", HF_BUFFER_NAME_MAP[self.model_type]["expert_bias"])
                    if isinstance(buf, DTensor):
                        buf = buf.full_tensor()
                    if name in local_params:
                        full_sd[name] = self._async_cpu_copy(name, buf)
            del buf

        self._sync_all_transfers()
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
        local_params, local_files = self._get_local_params_to_save(weight_map, world_size)

        # Each rank only collects the parameters it needs to save
        if self.use_te:
            state_dict = self._get_te_local_hf_state_dict(local_params)
        else:
            state_dict = self._get_local_hf_state_dict(local_params)

        # Save shards assigned to this rank
        for filename in local_files:
            shard_dict = {}
            for param_name, fn in weight_map.items():
                if fn == filename and param_name in state_dict:
                    shard_dict[param_name] = state_dict[param_name]
            if shard_dict:
                logger.info(f"[Rank {rank}] Saving {filename} with {len(shard_dict)} params...")
                file_start = time.perf_counter()
                file_path = save_path / filename
                save_file(shard_dict, file_path.as_posix())
                # Force flush to filesystem to ensure data is written to NAS
                with open(file_path, 'rb+') as f:
                    f.flush()
                    os.fsync(f.fileno())
                file_elapsed = time.perf_counter() - file_start
                logger.info(f"[Rank {rank}] Saved {filename} in {file_elapsed:.2f}s")
        os.sync()
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
            os.sync()

            if self.processing_class is not None:
                self.processing_class.save_pretrained(save_path)

            logger.info(f"Rank: {rank} - Saved hf model to {save_path.absolute()}")

        torch.distributed.barrier()