import gc
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import load_file
from torch.distributed._tensor import distribute_tensor, DTensor

from torchturbo.engine.utils import is_peft_adapter_param, strip_peft_prefix
from torchturbo.engine.peft.weights import get_peft_init_config, init_lora_tensor
from torchturbo.logging import get_logger

logger = get_logger(__name__)


class _PendingTask:
    """Holds state for an async broadcast that needs to be awaited and processed.

    After handle.wait(), the full_tensor contains valid data and can be
    distributed into the model parameter via distribute_tensor + copy_.
    """
    __slots__ = ('handle', 'full_tensor', 'param', 'mesh', 'target_dtype')

    def __init__(self, handle, full_tensor, param, mesh, target_dtype=None):
        self.handle = handle
        self.full_tensor = full_tensor
        self.param = param
        self.mesh = mesh
        self.target_dtype = target_dtype


class DistParamLoader:
    # Fraction of total GPU memory to reserve as headroom during broadcasts.
    # Controlled by TORCHTURBO_BROADCAST_MEM_RESERVE env var (default 0.1).
    _MEM_RESERVE_FRACTION = float(os.environ.get("TORCHTURBO_BROADCAST_MEM_RESERVE", "0.1"))
    # Max bytes for in-flight broadcast tensors (default 5 GB).
    # Env var TORCHTURBO_BROADCAST_MEM_BUDGET_CAP is in GB.
    _BROADCAST_MEM_BUDGET_CAP = int(float(os.environ.get("TORCHTURBO_BROADCAST_MEM_BUDGET_CAP", "5")) * 1024**3)

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

        # EP topology
        self.ep_group = mesh["ep"].get_group()
        self.ep_rank = dist.get_rank(group=self.ep_group)
        self.ep_world_size = dist.get_world_size(group=self.ep_group)

        # EDP topology (mirrors weight_saver)
        self.edp_group = mesh["edp"].get_group()
        self.edp_rank = mesh["edp"].get_local_rank()
        current_ep_idx = mesh.get_local_rank("ep")
        self.edp_ranks = mesh.mesh[:, current_ep_idx].tolist()

        self.param_converter = model.get_weight_converter(mesh)
        self.moe_config = self.param_converter.moe_config

        # Assignment data structures (mirrors weight_saver)
        self.param_size_map = {}
        self.param_to_rank = {}
        self.rank_shard_bytes = {}
        self.rank_to_params = [[] for _ in range(self.world_size)]
        self.sharded_param_groups = []
        self.WEIGHT_SHARD_SIZE = 5 * (1024 ** 3)

        # log related
        self.start_time = time.perf_counter()
        self.prev_time = self.start_time

        # build meta (replaces _build_param_to_rank)
        self.param_to_filename = self._load_index_json()
        self.local_params = self._build_meta_info()

    # ── Index & Meta Info ──────────────────────────────────────────

    def _load_index_json(self):
        index_path = self.path / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        with open(index_path) as f:
            index = json.load(f)
        return index["weight_map"]

    def _sync_log_rank0(self, content, update_prev=True):
        torch.cuda.synchronize()
        dist.barrier()
        cur = time.perf_counter()
        if self.rank == 0:
            logger.info(f"{content} in {cur - self.prev_time:.2f}s (total {cur - self.start_time:.2f}s)")
        if update_prev:
            self.prev_time = cur

    # ── Assignment Helpers (mirrors weight_saver) ──────────────────

    def _update_shard_bytes(self, rank, num_bytes):
        if rank in self.rank_shard_bytes:
            self.rank_shard_bytes[rank] += num_bytes
        else:
            self.rank_shard_bytes[rank] = num_bytes

    def _select_low_pressure_rank(self):
        lowest = float('inf')
        result = 0
        for r in range(self.world_size):
            if r not in self.rank_shard_bytes:
                return r
            if lowest > self.rank_shard_bytes[r]:
                result = r
                lowest = self.rank_shard_bytes[r]
        return result

    # ── Build Meta Info ────────────────────────────────────────────

    def _build_meta_info(self):
        """Classify model params, compute sizes, and assign to ranks.

        This mirrors ShardedParamManager._build_meta_info in weight_saver.py.
        Each rank determines which HF parameter names it needs to read from disk.
        """
        start_time = time.perf_counter()
        if self.rank == 0:
            logger.info("[DistParamLoader] _build_meta_info starting...")

        common_params = []
        expert_param_groups = []  # list of (model_param_name, [local_hf_names], total_local_bytes)

        for name, param in self.model.named_parameters():
            clean_name = strip_peft_prefix(name)
            if is_peft_adapter_param(clean_name):
                continue

            common_keys, expert_group, size_map = (
                self.param_converter.get_param_load_entries(
                    clean_name, param.numel(),
                    param_to_filename=self.param_to_filename,
                )
            )
            # Register virtual key → filename mapping for fused expert format.
            # Must happen before the filtering step so that virtual keys in
            # size_map pass the param_to_filename check.
            virtual_mapping = self.param_converter.get_virtual_key_filenames(
                clean_name, self.param_to_filename
            )
            self.param_to_filename.update(virtual_mapping)

            # Loader filters by param_to_filename (checkpoint must contain the key)
            filtered_keys = [k for k in common_keys if k in self.param_to_filename]
            filtered_size_map = {k: v for k, v in size_map.items() if k in self.param_to_filename}
            self.param_size_map.update(filtered_size_map)
            common_params.extend(filtered_keys)
            if expert_group is not None:
                expert_param_groups.append(expert_group)

        for name, buf in self.model.named_buffers():
            clean_name = strip_peft_prefix(name)
            if self.param_converter.is_persist_buffer(clean_name):
                hf_buffer_name = self.param_converter.convert_persist_buffer(clean_name)
                if hf_buffer_name in self.param_to_filename:
                    common_params.append(hf_buffer_name)
                    self.param_size_map[hf_buffer_name] = buf.numel() * buf.element_size()

        self._assign_param_to_ranks(expert_param_groups, self.edp_ranks, common_params)

        elapsed = time.perf_counter() - start_time
        if self.rank == 0:
            logger.info(f"[DistParamLoader] _build_meta_info completed in {elapsed:.2f}s")
            num_files = len(set(
                self.param_to_filename[p]
                for p in self.rank_to_params[self.rank]
                if p in self.param_to_filename
            ))
            logger.info(f"[DistParamLoader] Rank 0 will read {num_files} files for "
                       f"{len(self.rank_to_params[self.rank])} params")

        return set(self.rank_to_params[self.rank])

    def _assign_param_to_ranks(self, expert_param_groups, edp_ranks, common_params):
        """Two-phase param-to-rank assignment.

        Phase 1: Expert params → EDP ranks, shard-size balanced round-robin.
        Phase 2: Common params → all world ranks, file-grouped least-loaded.
        """
        # ── Phase 1: Expert params → EDP ranks ──
        num_edp_ranks = len(edp_ranks)
        accumulated_bytes = 0
        shard_params = []
        i = 0

        for _model_name, hf_names, group_bytes in expert_param_groups:
            # Flush current shard before adding this group if it would cross the boundary.
            # This prevents splitting a model param's HF names across ranks.
            if accumulated_bytes > 0 and accumulated_bytes + group_bytes >= self.WEIGHT_SHARD_SIZE:
                dst_rank = edp_ranks[i % num_edp_ranks]
                # Commit this shard
                shard_bytes = 0
                for p in shard_params:
                    self.param_to_rank[p] = dst_rank
                    shard_bytes += self.param_size_map.get(p, 0)
                self._update_shard_bytes(dst_rank, shard_bytes)
                if dst_rank == self.rank:
                    self.sharded_param_groups.append(shard_params)
                shard_params = []
                accumulated_bytes = 0
                i += 1

            shard_params.extend(hf_names)
            accumulated_bytes += group_bytes

        # Flush remaining
        if shard_params:
            dst_rank = edp_ranks[i % num_edp_ranks]
            shard_bytes = 0
            for p in shard_params:
                self.param_to_rank[p] = dst_rank
                shard_bytes += self.param_size_map.get(p, 0)
            self._update_shard_bytes(dst_rank, shard_bytes)
            if dst_rank == self.rank:
                self.sharded_param_groups.append(shard_params)

        # Merge param_to_rank across EP group (each EP rank assigns different HF names)
        param_to_rank_list = [None] * self.ep_world_size
        dist.all_gather_object(param_to_rank_list, self.param_to_rank, group=self.ep_group)
        for rank_map in param_to_rank_list:
            self.param_to_rank.update(rank_map)

        # Recompute rank_shard_bytes from full expert assignments across all EP ranks
        self.rank_shard_bytes = {}
        for param, r in self.param_to_rank.items():
            self._update_shard_bytes(r, self.param_size_map.get(param, 0))

        # ── Phase 2: Common params → all world ranks, file-grouped ──
        file_to_common_params = {}
        for param in common_params:
            filename = self.param_to_filename.get(param)
            if filename is not None:
                file_to_common_params.setdefault(filename, []).append(param)

        # Sort file groups by descending total bytes for better bin-packing
        sorted_file_groups = sorted(
            file_to_common_params.items(),
            key=lambda kv: sum(self.param_size_map.get(p, 0) for p in kv[1]),
            reverse=True,
        )

        for _filename, params in sorted_file_groups:
            dst_rank = self._select_low_pressure_rank()
            shard_bytes = sum(self.param_size_map.get(p, 0) for p in params)
            for p in params:
                self.param_to_rank[p] = dst_rank
            self._update_shard_bytes(dst_rank, shard_bytes)
            if dst_rank == self.rank:
                self.sharded_param_groups.append(params)

        # Build rank_to_params index
        for param, r in self.param_to_rank.items():
            self.rank_to_params[r].append(param)

    # ── Load Local Shards ──────────────────────────────────────────

    def _load_local_shards(self):
        # Group local params by source file for efficient loading
        files_needed = {}
        for param_name in self.local_params:
            filename = self.param_to_filename.get(param_name)
            if filename is not None:
                files_needed.setdefault(filename, []).append(param_name)

        local_sd = {}
        if files_needed:
            files = [self.path / f for f in sorted(files_needed.keys())]
            logger.info(f"[Rank {self.rank}] Loading {len(files)} files with "
                       f"{len(self.local_params)} parameters...")
            t0 = time.perf_counter()
            for f in files:
                file_params = load_file(f.as_posix(), device="cpu")
                needed_in_file = set(files_needed[f.name])

                # Separate virtual keys from physical keys.
                # Virtual keys (fused expert per-EP slices) need slicing
                # from the physical tensor rather than direct loading.
                virtual_needed = {p for p in needed_in_file
                                 if self.param_converter.is_virtual_key(p)}
                physical_needed = needed_in_file - virtual_needed

                # Physical keys: load directly (unchanged)
                for param_name, tensor in file_params.items():
                    if param_name in physical_needed:
                        local_sd[param_name] = tensor.clone()

                # Virtual keys: look up physical tensor, slice by ep_rank
                # encoded in the virtual key name, store under virtual key.
                # Slicing happens on CPU (file_params is CPU tensors),
                # so no GPU memory pressure.
                for virtual_key in virtual_needed:
                    physical_key = self.param_converter.get_physical_key(virtual_key)
                    if physical_key in file_params:
                        local_sd[virtual_key] = self.param_converter.slice_virtual_key(
                            virtual_key, file_params[physical_key]
                        )

            logger.info(f"[Rank {self.rank}] Loaded {len(local_sd)} params in "
                       f"{time.perf_counter() - t0:.2f}s")
        else:
            logger.info(f"[Rank {self.rank}] No parameters to load")

        self._sync_log_rank0("[Load File] Load finished")
        return local_sd

    # ── Async Broadcast Helpers ────────────────────────────────────

    def _async_broadcast_tensor(self, name, shape=None, broadcast_dtype=None, group=None):
        """Issue an async broadcast and return (handle, tensor).

        Broadcasts in broadcast_dtype (defaults to checkpoint dtype). Owner
        keeps the loaded dtype without conversion; non-owners allocate in
        broadcast_dtype. group=None means the world group. Dtype conversion
        to param.dtype is deferred to _flush_pending_tasks.
        """
        owner = self.param_to_rank[name]
        if self.rank == owner:
            assert self.local_sd[name].dtype == broadcast_dtype, "must be"
            tensor = self.local_sd[name].to(self.device)
        else:
            tensor = torch.empty(shape, device=self.device, dtype=broadcast_dtype)
        handle = dist.broadcast(tensor, src=owner, group=group, async_op=True)
        return handle, tensor

    def _async_broadcast_expert_weight(self, name, param):
        """Async broadcast of expert weight1 or weight2 within EDP group.

        The converter handles format-specific assembly (per-expert vs fused)
        and weight1 vs weight2 dispatch. Owner rank assembles the tensor;
        non-owners allocate empty. Broadcasts in checkpoint dtype, returns
        (handle, tensor). Dtype conversion deferred to _flush_pending_tasks.
        """
        hf_key = self.param_converter.get_expert_broadcast_key(name)
        owner = self.param_to_rank[hf_key]
        hf_dtype = self.param_converter.get_default_hf_dtype(name)

        # Pre-allocate in checkpoint dtype
        tensor = torch.empty(param.shape, device=self.device, dtype=hf_dtype)
        if self.rank == owner:
            self.param_converter.assemble_expert_weight_for_broadcast(
                tensor, self.local_sd, name
            )

        handle = dist.broadcast(tensor, src=owner, group=self.edp_group, async_op=True)
        return handle, tensor

    def _async_init_and_broadcast_lora(self, clean_name, param):
        """Issue async broadcast for a LoRA adapter param.

        Returns (handle, tensor, mesh).
        """
        is_a = (".lora_A." in clean_name
                or ".lora_A2." in clean_name
                or ".lora_embedding_A." in clean_name)
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
            edp_src_global = dist.get_global_rank(edp_group, 0)

            if edp_rank == 0:
                full = torch.empty(param.shape, device=self.device, dtype=param.dtype)
                init_lora_tensor(full, is_a, init_lora_weights, r)
            else:
                full = torch.empty(param.shape, device=self.device, dtype=param.dtype)

            handle = dist.broadcast(full, src=edp_src_global, group=edp_group, async_op=True)
            return handle, full, edp_mesh
        else:
            if self.rank == 0:
                full = torch.empty(param.shape, device=self.device, dtype=param.dtype)
                init_lora_tensor(full, is_a, init_lora_weights, r)
            else:
                full = torch.empty(param.shape, device=self.device, dtype=param.dtype)

            handle = dist.broadcast(full, src=0, async_op=True)
            return handle, full, self.world_mesh

    # ── Memory Budget Sync ─────────────────────────────────────────

    def _sync_memory_budget(self):
        """Compute a synchronized memory budget across all ranks.

        Each rank computes its local budget (free memory - reserve),
        then all-reduce (MIN) ensures every rank uses the same conservative
        value so broadcast collectives never hang due to mismatched flush timing.
        """
        mem_free, mem_total = torch.cuda.mem_get_info()
        reserve_bytes = int(mem_total * self._MEM_RESERVE_FRACTION)
        local_budget = max(mem_free - reserve_bytes, 0)

        budget_tensor = torch.tensor([local_budget], dtype=torch.float, device=self.device)
        dist.all_reduce(budget_tensor, op=dist.ReduceOp.MIN)
        budget_bytes = min(int(budget_tensor.item()), self._BROADCAST_MEM_BUDGET_CAP)

        if self.rank == 0:
            logger.info(
                f"[Broadcast] Memory budget: {budget_bytes / 1024**3:.2f} GB "
                f"(local free: {mem_free / 1024**3:.2f} GB, "
                f"reserve: {reserve_bytes / 1024**3:.2f} GB)"
            )
        return budget_bytes

    # ── Flush Pending Tasks ────────────────────────────────────────

    def _flush_pending_tasks(self, tasks):
        """Wait for async broadcasts, convert dtype, distribute and copy into params.

        For each task: wait handle → convert broadcast dtype to target dtype
        (e.g., bf16→fp32) if needed → distribute_tensor (if DTensor) → copy_.
        """
        for task in tasks:
            task.handle.wait()

            # Convert broadcast dtype → target dtype (e.g., bf16 → fp32)
            if task.target_dtype is not None and task.full_tensor.dtype != task.target_dtype:
                task.full_tensor = task.full_tensor.to(task.target_dtype)

            if isinstance(task.param, DTensor):
                sharded_tensor = distribute_tensor(
                    task.full_tensor, task.mesh, task.param.placements, src_data_rank=None
                )
            else:
                sharded_tensor = task.full_tensor

            with torch.no_grad():
                task.param.copy_(nn.Parameter(sharded_tensor))
            del task.full_tensor

        del tasks

        # Memory cleanup after each batch
        mem_free, mem_total = torch.cuda.mem_get_info()
        mem_used = mem_total - mem_free
        utilization = mem_used / mem_total
        if utilization > (1 - self._MEM_RESERVE_FRACTION):
            before_free = mem_free
            torch.cuda.empty_cache()
            gc.collect()
            new_free, _ = torch.cuda.mem_get_info()
            freed_gb = (new_free - before_free) / 1024 ** 3
            logger.info(f"Rank {self.rank} Cleaned {freed_gb:.2f} GB GPU memory "
                       f"(utilization: {utilization:.1%} -> {(mem_total - new_free)/mem_total:.1%})")

    # ── Buffer Handling ────────────────────────────────────────────

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
                    logger.info(f"Converted {name}.{buf_name}: {old_dtype} -> {hf_dtype}")

    def _broadcast_buffers(self):
        """Broadcast buffers synchronously (they are typically small biases)."""
        self._fix_buffer_dtype()
        for name, buf in self.model.named_buffers():
            clean_name = strip_peft_prefix(name)
            if self.param_converter.is_persist_buffer(clean_name):
                hf_buffer_name = self.param_converter.convert_persist_buffer(clean_name)
                if hf_buffer_name not in self.param_to_filename:
                    if self.rank == 0:
                        logger.info(f"[Weight Loader] SKIP buffer (not in checkpoint): {clean_name} (original: {name})")
                    continue
                if hf_buffer_name not in self.param_to_rank:
                    if self.rank == 0:
                        logger.info(f"[Weight Loader] SKIP buffer (not assigned): {clean_name} (original: {name})")
                    continue
                owner = self.param_to_rank[hf_buffer_name]
                if self.rank == owner:
                    full = self.local_sd[hf_buffer_name].to(self.device, dtype=buf.dtype)
                else:
                    full = torch.empty(buf.shape, device=self.device, dtype=buf.dtype)
                dist.broadcast(full, src=owner, async_op=False)
                with torch.no_grad():
                    buf.copy_(full)
                del full

    # ── Broadcast All Params ───────────────────────────────────────

    def _broadcast_params(self):
        pending = []
        pending_bytes = 0
        num_params = len(list(self.model.parameters()))
        processed = 0

        # Dynamic memory budget: reserve a fraction of total GPU memory,
        # use the remaining free memory for in-flight broadcast tensors.
        # Must synchronize across ranks so all ranks flush at the same time,
        # otherwise broadcast collectives will hang.
        budget_bytes = self._sync_memory_budget()

        for name, param in self.model.named_parameters():
            clean_name = strip_peft_prefix(name)
            target_dtype = param.dtype

            if is_peft_adapter_param(clean_name):
                handle, full, mesh = self._async_init_and_broadcast_lora(clean_name, param)
            elif self.param_converter.is_expert_weight(clean_name):
                handle, full = self._async_broadcast_expert_weight(clean_name, param)
                mesh = self.mesh['edp']
            else:
                hf_name = self.param_converter.to_common_hf_name(clean_name)
                if hf_name not in self.param_to_filename:
                    if self.rank == 0:
                        logger.info(f"[Weight Loader] SKIP (not in checkpoint): {clean_name} (original: {name})")
                    continue
                if hf_name not in self.param_to_rank:
                    if self.rank == 0:
                        logger.info(f"[Weight Loader] SKIP (not assigned): {clean_name} (original: {name})")
                    continue
                # Broadcast in checkpoint dtype; dtype conversion deferred to flush
                hf_dtype = self.param_converter.get_default_hf_dtype(clean_name)
                handle, full = self._async_broadcast_tensor(hf_name, param.shape, hf_dtype, group=None)
                mesh = self.world_mesh

            tensor_bytes = full.numel() * full.element_size()

            # Flush existing pending if adding this tensor would exceed the memory budget
            if pending and pending_bytes + tensor_bytes > budget_bytes:
                self._flush_pending_tasks(pending)
                pending = []
                pending_bytes = 0
                if self.rank == 0:
                    logger.info(f"[Broadcast] {processed}/{num_params} params")

            pending.append(_PendingTask(handle, full, param, mesh, target_dtype))
            pending_bytes += tensor_bytes
            processed += 1

        if pending:
            self._flush_pending_tasks(pending)

        self._sync_log_rank0(f"[Rank {self.rank}] Broadcast {processed}/{num_params} params done")

    # ── Cleanup ────────────────────────────────────────────────────

    def _cleanup(self):
        self.local_sd = None
        gc.collect()
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.empty_cache()

    # ── Public API ─────────────────────────────────────────────────

    def load_weights(self):
        self.model.to_empty(device=self.device)
        self.local_sd = self._load_local_shards()

        # buffers (small, kept synchronous)
        self._broadcast_buffers()
        self._sync_log_rank0(f"[Rank {self.rank}] broadcast buffers")

        # parameters (pipelined async broadcasts)
        self._broadcast_params()
        self._sync_log_rank0(f"[Rank {self.rank}] Broadcast/sharding done")

        self._cleanup()
