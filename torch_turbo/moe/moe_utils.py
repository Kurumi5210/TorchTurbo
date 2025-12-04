import os
import re
import time

from pathlib import Path
from safetensors.torch import load_file
import json
import gc
import torch
import torch.distributed as dist
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._tensor import distribute_tensor, DTensor
from torch import nn
from torch.nn.utils.clip_grad import _clip_grads_with_norm_, _get_total_norm

from torch_turbo.moe.monkey_patch import apply_monkey_patch
from torch_turbo.utils.torch_utils import torch_version, load_safetensors_dir
from torch_turbo.utils.device import get_device_name, get_device_id, is_npu_available
from torch_turbo.transformer.transformer_config import TransformerConfig
from torch_turbo.transformer.module import ModuleSpec
from torch_turbo.moe.moe_layer import MoESubmodules, MoELayer
from torch_turbo.moe.experts import GroupedMLP
from torch_turbo.distributed.parallel import (
    get_expert_model_parallel_group,
    get_expert_model_parallel_rank,
    get_expert_model_parallel_world_size
)
from torch.distributed.tensor.placement_types import Shard
if is_npu_available:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu

if torch_version() >= (2, 6, 0):
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.fsdp._fully_shard import FSDPModule
else:
    from torch.distributed._composable.fsdp import fully_shard
    from torch.distributed._composable.fsdp.fully_shard import FSDPModule


device_name = get_device_name()

WORLD_MESH = None
EXPERT_MESH = None
FIRST_RANK = 0
_GRADIENT_SCALING_FACTOR = 1.0


def is_rank_in_mesh(target_rank, device_mesh):
    return target_rank in device_mesh.mesh.flatten().tolist()

def is_rank_in_group(target_rank, process_group):
    group_ranks = dist.get_process_group_ranks(process_group)
    return target_rank in group_ranks

def slice_tensor_for_ep_rank(tensor, ep_rank, ep_world_size):
    assert ep_world_size > 0, "must be"

    total_dim = tensor.shape[0]
    chunk_size = total_dim // ep_world_size

    start = ep_rank * chunk_size
    end = start + chunk_size

    return tensor[start:end]

def all_gather_tensor(local_tensor, ep_name="ep"):
    """ gather expert tensors
    """
    group = EXPERT_MESH[ep_name].get_group(0)
    ep_world_size = dist.get_world_size(group=group)
    output_shape = list(local_tensor.shape)
    output_shape[0] = output_shape[0] * ep_world_size
    output = torch.empty(output_shape, dtype=local_tensor.dtype, device=local_tensor.device)
    dist.all_gather_into_tensor(output, local_tensor, group=group)

    return output

def convert_param_name(key, reverse_key_mapping):
    for pattern, replacement in reverse_key_mapping.items():
        replacement = replacement.lstrip("^")  # strip off un-needed chars and patterns
        replacement = re.sub(r"\(.*\)", "", replacement)
        key, n_replace = re.subn(pattern, replacement, key)
        # Early exit of the loop
        if n_replace > 0:
            return key
    return key

def fix_rope_config_for_bailing(config):
    rotary_dim = getattr(config, "rotary_dim", None)
    head_dim = getattr(config, "head_dim", None)
    partial_rotary_factor = getattr(config, "partial_rotary_factor", None)
    assert head_dim != None, "must be"
    assert rotary_dim != None or partial_rotary_factor != None, "must be"
    if partial_rotary_factor == None:
        # Ring-1T does not set partial_rotary_factor
        partial_rotary_factor = rotary_dim / head_dim
        config.partial_rotary_factor = partial_rotary_factor

class DistParamLoad:
    def __init__(self, model, config, device, args, ep_mesh, world_mesh):
        self.model = model
        self.config = config
        self.device = device
        self.path = Path(args.path)
        self.ep_mesh = ep_mesh
        self.world_mesh = world_mesh
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.ep_rank = get_expert_model_parallel_rank()
        self.ep_world_size = get_expert_model_parallel_world_size()
        # log related
        self.start_time = time.perf_counter()
        self.prev_time = self.start_time
        # build meta
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
        parts = param_name.split('.')
        if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
            return int(parts[2])
        return None

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

    def _load_experts_to_device(self):
        def get_expert_keys(prefix, is_weight1):
            keys = []
            for i in range(self.config.num_experts):
                if is_weight1:
                    keys.extend([f"{prefix}.{i}.gate_proj.weight", f"{prefix}.{i}.up_proj.weight"])
                else:
                    keys.append(f"{prefix}.{i}.down_proj.weight")
            return keys

        for name, _ in self.model.named_parameters():
            if "mlp.experts" not in name or self._get_owner_rank(name) != self.rank:
                continue

            prefix = name[:-8]
            is_weight1 = "weight1" in name
            for key in get_expert_keys(prefix, is_weight1):
                assert key in self.local_sd, f"{key} must be in rank {self.rank} local sd"
                self.local_sd[key] = self.local_sd[key].to(self.device)

    def _te_load_experts_to_device(self):
        def get_expert_keys(prefix, is_fc1):
            keys = []
            for i in range(self.config.num_experts):
                if is_fc1:
                    keys.extend([f"{prefix}.{i}.gate_proj.weight", f"{prefix}.{i}.up_proj.weight"])
                else:
                    keys.append(f"{prefix}.{i}.down_proj.weight")
            return keys

        for name, _ in self.model.named_parameters():
            if "mlp.experts" not in name or "weight0" not in name or self._get_owner_rank(name) != self.rank:
                continue
            prefix = ".".join(name.split(".")[:5])
            is_fc1 = "fc1_linear" in name
            for key in get_expert_keys(prefix, is_fc1):
                assert key in self.local_sd, f"{key} must be in rank {self.rank} local sd"
                self.local_sd[key] = self.local_sd[key].to(self.device)

    def _te_broadcast_to_ep_ranks(self, name, param, prefix_lst):
        prefix = ".".join(prefix_lst[:5])
        owner = self._get_owner_rank(name)
        ep_size = self.ep_world_size
        num_local_experts = self.config.num_experts // ep_size
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
            new_shape = (param.shape[0], param.shape[1])
            tensor = torch.empty(
                new_shape,
                device=self.device,
                dtype=param.dtype,
            )
            for i in range(ep_size):
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
                num_local_experts = self.config.num_experts // ep_size
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

    def _broadcast_buffers(self):
         # Load buffers (e.g., bailing expert bias)
        for name, buf in self.model.named_buffers():
            if "router.expert_bias" in name:
                full = self._broadcast_tensor(name.replace("router", "gate"), buf.shape, buf.dtype)
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
                full = self._broadcast_tensor(name.replace("router", "gate"), param.shape, param.dtype)
                mesh = self.world_mesh
            else:
                full = self._broadcast_tensor(name, param.shape, param.dtype)
                mesh = self.world_mesh

            sharded_tensor = distribute_tensor(full, mesh, param.placements, src_data_rank=None)

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
                freed_gb = (new_free - mem_free) / 1024**3
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

    def load(self):
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


class MoeUtils:
    @staticmethod
    def initialize_moe_layer_from_hf_config(config, args, **config_kwargs):
        num_moe_experts = config.num_experts
        expert_model_parallel_size = get_expert_model_parallel_world_size()
        default_config_kwargs = dict(
            num_layers=1,
            expert_model_parallel_size=expert_model_parallel_size,
            moe_ffn_hidden_size=config.moe_intermediate_size,
            moe_token_dispatcher_type="alltoall",
            gated_linear_unit=True,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_moe_experts=num_moe_experts,
            moe_router_topk=config.num_experts_per_tok,
            use_cpu_initialization=True,
            fp8=None,
            add_bias_linear=False,
            moe_router_pre_softmax=True,
            activation_func=F.silu,
            moe_router_load_balancing_type="none",
        )
        if config.architectures[0] == "BailingMoeV2ForCausalLM":
            default_config_kwargs.update(
                moe_router_score_function=config.score_function,
                moe_router_group_topk=config.topk_group,
                moe_router_num_groups=config.n_group,
                moe_router_dtype=config.router_dtype, 
                moe_router_enable_expert_bias=config.moe_router_enable_expert_bias,
                moe_router_topk_scaling_factor=config.routed_scaling_factor,
            )
        default_config_kwargs.update(**config_kwargs)
        tf_config = TransformerConfig(**default_config_kwargs)
        experts = ModuleSpec(module=GroupedMLP, submodules=None)
        moe_submodules = MoESubmodules(experts=experts, shared_experts=None)

        grouped_mlp = MoELayer(tf_config, moe_submodules, args.use_fused_swiglu, args.use_fused_permute_unpermute, args.use_transformer_engine)

        return grouped_mlp

    @staticmethod
    def patch_moe_layer(model, hf_config, args, device):
        assert ("moe" in hf_config.model_type) or ("next" in hf_config.model_type), f'{hf_config.model_type} must be moe model'
        for layer in model.model.layers:
            # patch token_dispatcher
            if not hasattr(layer.mlp, "token_dispatcher"):
                continue
            token_dispatcher = layer.mlp.token_dispatcher
            token_dispatcher.permute_idx_device = device
            input_chunk_idxs = torch.arange(
                token_dispatcher.num_experts * token_dispatcher.tp_size, device=token_dispatcher.permute_idx_device
            )
            token_dispatcher.sort_input_by_local_experts = input_chunk_idxs.reshape(
                -1, token_dispatcher.num_local_experts
            ).T.ravel()
            # [tp_size * ep_size, num_local_experts]. Restore the output chunks by local experts.
            token_dispatcher.restore_output_by_local_experts = input_chunk_idxs.reshape(
                token_dispatcher.num_local_experts, -1
            ).T.ravel()

        # patch rotary embedding
        rotary_emb = model.model.rotary_emb
        if rotary_emb.config.architectures[0] == "BailingMoeV2ForCausalLM":
            fix_rope_config_for_bailing(rotary_emb.config)
        inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(rotary_emb.config, device)
        # HACK align to deepspeed
        rotary_emb.register_buffer("inv_freq", inv_freq.to(dtype=torch.bfloat16), persistent=False)

    @staticmethod
    def scale_moe_gradient(params):
        assert params[0].device_mesh == MoeUtils.get_edp_mesh()
        grads = [p.grad for p in params]
        torch._foreach_mul_(grads, _GRADIENT_SCALING_FACTOR)

    @staticmethod
    @torch.no_grad()
    def clip_grad_norm(model, max_norm=1.0, norm_type=2.0):
        params = []
        grads = []
        moe_params = []
        moe_grads = []
        edp = MoeUtils.get_edp_mesh()
        for param in model.parameters():
            if param.device_mesh == edp:
                moe_params.append(param)
                moe_grads.append(param.grad)
            else:
                params.append(param)
                grads.append(param.grad)

        MoeUtils.scale_moe_gradient(moe_params)

        dp_total_norm = _get_total_norm(grads, norm_type=norm_type)
        edp_total_norm = _get_total_norm(moe_grads, norm_type=norm_type)
        dp_total_norm = dp_total_norm.full_tensor().unsqueeze(0)
        edp_total_norm = edp_total_norm.full_tensor().unsqueeze(0)
        edp_total_norm = all_gather_tensor(edp_total_norm, ep_name="ep")
        total_norm = torch.linalg.vector_norm(torch.cat([edp_total_norm, dp_total_norm]), norm_type)
        _clip_grads_with_norm_(params, max_norm, total_norm)
        _clip_grads_with_norm_(moe_params, max_norm, total_norm)

        return total_norm

    @staticmethod
    def transform_moe_layer(model, hf_config, args):
        assert ("moe" in hf_config.model_type) or ("next" in hf_config.model_type), f'{hf_config.model_type} must be moe model'
        if hasattr(hf_config, "first_k_dense_replace"):
            first_moe_layer = hf_config.first_k_dense_replace
        else:
            first_moe_layer = 0
        for i, layer in enumerate(model.model.layers):
            if i < first_moe_layer:
                continue
            moe = MoeUtils.initialize_moe_layer_from_hf_config(hf_config, args)
            if "bailing_moe" in hf_config.model_type:
                bailing_shared_experts = layer.mlp.shared_experts.__class__(model.config, intermediate_size=model.config.moe_intermediate_size * model.config.num_shared_experts)
            layer.mlp = moe
            if "next" in hf_config.model_type:
                from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextMLP
                layer.mlp.shared_expert = Qwen3NextMLP(model.config, intermediate_size=model.config.shared_expert_intermediate_size)
                layer.mlp.shared_expert_gate = torch.nn.Linear(model.config.hidden_size, 1, bias=False)
            if "bailing_moe" in hf_config.model_type:
                layer.mlp.shared_experts = bailing_shared_experts

    @staticmethod
    def get_world_mesh():
        assert WORLD_MESH != None, "must be"
        return WORLD_MESH

    @staticmethod
    def get_expert_mesh():
        assert EXPERT_MESH != None, "must be"
        return EXPERT_MESH

    @staticmethod
    def get_ep_mesh():
        assert EXPERT_MESH != None, "must be"
        return EXPERT_MESH['ep']

    @staticmethod
    def get_edp_mesh():
        assert EXPERT_MESH != None, "must be"
        return EXPERT_MESH['edp']

    @staticmethod
    def create_fsdp2_moe_model(args, device):
        MoeUtils.init_mesh(args)

        config = AutoConfig.from_pretrained(args.path, trust_remote_code=True)
        model = MoeUtils.build_model(args, config)
        model = MoeUtils.fsdp_wrap(model, device, {}, config)
        MoeUtils.patch_moe_layer(model, config, args, device)
        MoeUtils.load_weights_from_hf(model, config, device, args)
        return model

    @staticmethod
    def create_fsdp2_moe_optimizer(model, optim_config):
        params = []
        moe_params = []
        edp = MoeUtils.get_edp_mesh()
        for param in model.parameters():
            if param.device_mesh != edp:
                params.append(param)
            else:
                moe_params.append(param)

        optimizer_params = [{"params": params}, {"params": moe_params}]
        optimizer = torch.optim.AdamW(
            optimizer_params,
            lr=optim_config.get("lr", 1e-4),
            betas=optim_config.get("betas", (0.9, 0.999)),
            weight_decay=optim_config.get("weight_decay", 1e-2),
        )

        return optimizer

    @staticmethod
    def init_mesh(args):
        # TODO: change device name to a global variable to switch between cuda and npu
        world_size = args.world_size
        world_mesh = init_device_mesh(device_name, (world_size,))
        ep_mesh = init_device_mesh(device_name, (world_size // args.ep_size, args.ep_size), mesh_dim_names=("edp", "ep"))
        gradient_scaling_factor = (1 / args.ep_size) if args.average_in_collective else 1.0

        def is_same_group(process_group, device_mesh):
            mesh_group = device_mesh.get_group(0)
            mesh_ranks = dist.get_process_group_ranks(mesh_group)
            given_ranks = dist.get_process_group_ranks(process_group)
            return mesh_ranks == given_ranks

        # verify ep group is the same with ep_mesh
        megatron_ep_group = get_expert_model_parallel_group()
        assert is_same_group(megatron_ep_group, ep_mesh['ep'])

        global WORLD_MESH
        WORLD_MESH = world_mesh
        global EXPERT_MESH
        EXPERT_MESH = ep_mesh
        global _GRADIENT_SCALING_FACTOR
        _GRADIENT_SCALING_FACTOR = gradient_scaling_factor

    @staticmethod
    def build_model(args, config, torch_dtype=torch.bfloat16):
        # DEBUG
        # config.num_hidden_layers = 1
        with torch.device("meta"):
            if config.architectures[0] == "BailingMoeV2ForCausalLM":
                from .modeling_bailing_moe_v2 import BailingMoeV2ForCausalLM
                model = BailingMoeV2ForCausalLM(config)
            else:
                model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
            apply_monkey_patch(model, args)
            MoeUtils.transform_moe_layer(model, config, args)

        model = model.to(dtype=torch_dtype)
        return model

    @staticmethod
    def save_model(model, save_path, args, processing_class=None):
        use_te = args.use_transformer_engine
        if use_te:
            state_dict = MoeUtils.get_te_full_hf_state_dict(model)
        else:
            state_dict = MoeUtils.get_full_hf_state_dict(model)

        rank = torch.distributed.get_rank()
        if rank == 0:
            with torch.device("meta"):
                if model.config.architectures[0] == "BailingMoeV2ForCausalLM":
                    from .modeling_bailing_moe_v2 import BailingMoeV2ForCausalLM
                    save_model = BailingMoeV2ForCausalLM(model.config)
                else:    
                    save_model = AutoModelForCausalLM.from_config(model.config, torch_dtype=torch.bfloat16)
                
            save_model.to_empty(device="cpu")

            bf16_state_dict = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}

            hf_local_path = os.path.join(save_path, "huggingface")
            os.makedirs(hf_local_path, exist_ok=True)
            save_model.save_pretrained(hf_local_path, state_dict=bf16_state_dict)
            if processing_class is not None:
                processing_class.save_pretrained(hf_local_path)
            print(f"Rank: {rank} - Saved hf model to {os.path.abspath(hf_local_path)}")

        # wait for rank0 to dump hf_model to local
        torch.distributed.barrier()

    @staticmethod
    def set_moe_forward_prefetch_for_fsdp2_ep(fsdp2_model, layer_cls, expert_cls, model_cls, config=None):
        model_modules = []
        layer_modules = []
        expert_modules = []
        if hasattr(config, "first_k_dense_replace"):
            first_k_dense_replace = config.first_k_dense_replace
        else:
            first_k_dense_replace = 0

        for name, named_module in fsdp2_model.named_modules():
            if isinstance(named_module, FSDPModule) and isinstance(named_module, model_cls):
                model_modules.append(named_module)

            if isinstance(named_module, FSDPModule) and isinstance(named_module, layer_cls):
                layer_modules.append(named_module)

            if isinstance(named_module, FSDPModule) and isinstance(named_module, expert_cls):
                expert_modules.append(named_module)

        assert len(model_modules) == 1
        assert len(layer_modules) == len(expert_modules) + first_k_dense_replace

        for i in range(len(layer_modules)):
            # layer prefetch layer
            # expert prefetch expert
            cur_layer = layer_modules[i]
            j = i - first_k_dense_replace
            if j >= 0:
                cur_expert = expert_modules[j]

            # deal with forward
            if i == 0:
                if j == 0:
                    model_modules[0].set_modules_to_forward_prefetch([cur_layer, cur_expert])
                else:
                    model_modules[0].set_modules_to_forward_prefetch([cur_layer])
            else:
                # layer_{i - 1} prefetch layer_i
                if j == 0:
                    layer_modules[i - 1].set_modules_to_forward_prefetch([cur_expert, cur_layer])
                else:
                    layer_modules[i - 1].set_modules_to_forward_prefetch([cur_layer])
                
                if j > 0:
                    expert_modules[j - 1].set_modules_to_forward_prefetch([cur_expert])
            

            # deal with backward
            if i > 0:
                # layer_i prefetch layer_{i - 1}
                layer_backward_prefetch_states = [layer_modules[i - 1]._get_fsdp_state()]
                setattr(layer_modules[i], 'backward_prefetch_states', layer_backward_prefetch_states)

            if j > 0:
                # layer_i prefetch layer_{i - 1}
                expert_backward_prefetch_states = [expert_modules[j - 1]._get_fsdp_state()]
                setattr(expert_modules[j], 'backward_prefetch_states', expert_backward_prefetch_states)

    @staticmethod
    def fsdp_wrap(model, device, fsdp_kwargs={}, config=None):
        outer_cls = model.model.layers[0].__class__
        if hasattr(config, "first_k_dense_replace"):
            first_k_dense_replace = config.first_k_dense_replace
            inter_cls = model.model.layers[config.first_k_dense_replace].mlp.experts.__class__
        else:
            first_k_dense_replace = 0
            inter_cls = model.model.layers[0].mlp.experts.__class__
        outermost_cls = model.__class__
        if config.architectures[0] == "BailingMoeV2ForCausalLM":
            fully_shard(model.model.word_embeddings, mesh = MoeUtils.get_world_mesh(), **fsdp_kwargs)
            fully_shard(model.lm_head, mesh = MoeUtils.get_world_mesh(), **fsdp_kwargs)
        for i, layer in enumerate(model.model.layers):
            # wrap moe mlp in edp
            if i >= first_k_dense_replace:
                fully_shard(layer.mlp.experts, mesh = MoeUtils.get_edp_mesh(), **fsdp_kwargs)
                setattr(layer.mlp.experts, "expert_forward_prefetch", True)
            # wrap other parts in dp
            fully_shard(layer, mesh = MoeUtils.get_world_mesh(), **fsdp_kwargs)
            setattr(layer, "moe_forward_prefetch", True)
        # wrap remaining parts
        fully_shard(model, mesh = MoeUtils.get_world_mesh(), **fsdp_kwargs)

        MoeUtils.set_moe_forward_prefetch_for_fsdp2_ep(model, outer_cls, inter_cls, outermost_cls, config)
        model.to_empty(device=device)
        return model

    # HF model:
    # Qwen3MoeSparseMoeBlock(
    #   (gate): Linear(in_features=2048, out_features=128, bias=False)
    #   (experts): ModuleList(
    #     (0-127): 128 x Qwen3MoeMLP(
    #       (gate_proj): Linear(in_features=2048, out_features=768, bias=False)
    #       (up_proj): Linear(in_features=2048, out_features=768, bias=False)
    #       (down_proj): Linear(in_features=768, out_features=2048, bias=False)
    #       (act_fn): SiLU()
    #     )
    #   )
    # )
    # name layers.0.mlp.gate.weight, param.shape torch.Size([128, 2048]), param.device meta
    # name layers.0.mlp.experts.0.gate_proj.weight, param.shape torch.Size([768, 2048]), param.device meta
    # name layers.0.mlp.experts.0.up_proj.weight, param.shape torch.Size([768, 2048]), param.device meta
    # name layers.0.mlp.experts.0.down_proj.weight, param.shape torch.Size([2048, 768]), param.device meta
    #
    # Megatron model:
    # MoELayer(
    #   (router): TopKRouter()
    #   (experts): GroupedMLP()
    # )
    # name router.weight, param.shape torch.Size([128, 2048]), param.device meta
    # name experts.weight1, param.shape torch.Size([2048, 49152]), param.device meta
    # name experts.weight2, param.shape torch.Size([24576, 2048]), param.device meta
    @staticmethod
    def load_weights_from_hf(model, config, device, args):
        legacy = os.environ.get("TT_LEGACY_MODEL_LOAD", None)
        if legacy is None:
            dpl = DistParamLoad(model, config, device, args, EXPERT_MESH, WORLD_MESH)
            dpl.load()
            return
        rank = torch.distributed.get_rank()
        # only load state dict from rank 0, and broadcast to alll other ranks according device mesh
        if rank == FIRST_RANK:
            print("Start loading param to memory...", flush=True)
            t0 = time.perf_counter()
            sd = load_safetensors_dir(args.path)
            t1 = time.perf_counter()
            print(f"Finish loading param to memory in {t1 - t0:.2f} s", flush=True)
        torch.distributed.barrier()

        ep = EXPERT_MESH['ep']
        edp = EXPERT_MESH['edp']

        ep_rank = get_expert_model_parallel_rank()
        ep_world_size = get_expert_model_parallel_world_size()

        bcast_start = time.perf_counter()
        sharded_sd = {}
        for buffer_name, buffer in model.named_buffers():
            if "router.expert_bias" in buffer_name:
                if rank == FIRST_RANK:          
                    new_name = buffer_name.replace("router", "gate")
                    full_tensor = sd[new_name].to(device)
                else:
                    full_tensor = torch.empty(
                        buffer.shape,
                        device=device,
                    )
                dist.broadcast(full_tensor, src=FIRST_RANK)
                sharded_sd[buffer_name] = full_tensor
        for param_name, param in model.named_parameters():
            if rank == FIRST_RANK:
                if "mlp.experts" in param_name:
                    prefix = param_name[:-8]
                    if "weight1" in param_name:
                        w1 = []
                        for i in range(config.num_experts):
                            gate_proj = f"{prefix}.{i}.gate_proj.weight"
                            up_proj = f"{prefix}.{i}.up_proj.weight"
                            fc1 = torch.cat([sd[gate_proj].to(device), sd[up_proj].to(device)])
                            w1.append(fc1)
                        full_tensor = torch.stack(w1, dim=0)
                        del w1
                        # [e, 2 * i, h] -> [e, h, 2 * i]
                        full_tensor = full_tensor.permute(0, 2, 1)
                    else:
                        # weight2
                        w2 = []
                        for i in range(config.num_experts):
                            down_proj = f"{prefix}.{i}.down_proj.weight"
                            w2.append(sd[down_proj].to(device).t())
                        full_tensor = torch.stack(w2, dim=0)
                        del w2
                    # broadcast to ep_group
                    full_tensor = full_tensor.reshape(-1, full_tensor.shape[-1])
                    dist.broadcast(full_tensor, src=rank, group=ep.get_group(0))

                    # each ep slice its part
                    full_tensor = slice_tensor_for_ep_rank(full_tensor, ep_rank, ep_world_size)
                    mesh = edp
                elif "mlp.router" in param_name:
                    new_name = param_name.replace("router", "gate")
                    full_tensor = sd[new_name].to(device)
                    mesh = WORLD_MESH
                else:
                    full_tensor = sd[param_name].to(device)
                    mesh = WORLD_MESH
            else:
                if "mlp.experts" in param_name:
                    new_shape = (param.shape[0] * ep_world_size, param.shape[1])
                    full_tensor = torch.empty(
                        new_shape,
                        device=device,
                        dtype=param.dtype,
                    )
                    # get broadcast tensor
                    if is_rank_in_mesh(FIRST_RANK, ep):
                        dist.broadcast(full_tensor, src=FIRST_RANK, group=ep.get_group(0))
                    # slice tensor
                    full_tensor = slice_tensor_for_ep_rank(full_tensor, ep_rank, ep_world_size)
                    mesh = edp
                else:
                    full_tensor = torch.empty(
                        param.size(),
                        device=device,
                        dtype=param.dtype,
                    )
                    mesh = WORLD_MESH
            sharded_tensor = distribute_tensor(full_tensor, mesh, param.placements)
            sharded_sd[param_name] = nn.Parameter(sharded_tensor)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        bcast_elapsed = time.perf_counter() - bcast_start
        print(f"[Rank {rank}] broadcast done in {bcast_elapsed:.2f} s", flush=True)
        model.load_state_dict(sharded_sd, assign=True)
        for param in model.parameters():
            assert isinstance(param, DTensor), "must be"

    @staticmethod
    def get_full_hf_state_dict(model):
        rank = torch.distributed.get_rank()
        ep = EXPERT_MESH['ep']

        # Only rank 0 will save hf model and,
        # offload to cpu to save LLMs which may be too large to fit in one GPU
        full_sd = {}

        for name, param in model.named_parameters():
            if isinstance(param, DTensor):
                full_tensor = param.full_tensor()
            else:
                full_tensor = param
            if rank == FIRST_RANK:
                if "mlp.experts" in name:
                    prefix = name[:-8]
                    num_experts = model.config.num_experts
                    full_tensor = all_gather_tensor(full_tensor, ep_name="ep")
                    if "weight1" in name:
                        experts = torch.split(full_tensor, model.config.hidden_size)
                        for i in range(num_experts):
                            gate_proj = f"{prefix}.{i}.gate_proj.weight"
                            up_proj = f"{prefix}.{i}.up_proj.weight"
                            expert2 = torch.split(experts[i].t(), model.config.moe_intermediate_size)
                            full_sd[gate_proj] = expert2[0].cpu()
                            full_sd[up_proj] = expert2[1].cpu()
                    else:
                        # weight2
                        experts = torch.split(full_tensor, model.config.moe_intermediate_size)
                        for i in range(num_experts):
                            down_proj = f"{prefix}.{i}.down_proj.weight"
                            full_sd[down_proj] = experts[i].t().cpu()
                elif "mlp.router" in name:
                    new_name = name.replace("router", "gate")
                    full_sd[new_name] = full_tensor.cpu()
                else:
                    full_sd[name] = full_tensor.cpu()
            elif is_rank_in_mesh(FIRST_RANK, ep) and "mlp.experts" in name:
                full_tensor = all_gather_tensor(full_tensor, ep_name="ep")
                del full_tensor
            else:
                del full_tensor

        return full_sd
    
    @staticmethod
    def get_te_full_hf_state_dict(model):
        rank = torch.distributed.get_rank()
        ep = EXPERT_MESH['ep']

        # Only rank 0 will save hf model and,
        # offload to cpu to save LLMs which may be too large to fit in one GPU
        full_sd = {}

        for name, param in model.named_parameters():
            if isinstance(param, DTensor):
                full_tensor = param.full_tensor()
            else:
                full_tensor = param
            if rank == FIRST_RANK:
                if "mlp.experts" in name:
                    full_tensor = all_gather_tensor(full_tensor, ep_name="ep")
                    prefix_lst = name.split(".")
                    prefix = ".".join(name.split(".")[:5])
                    group = EXPERT_MESH["ep"].get_group(0)
                    ep_size = dist.get_world_size(group=group)
                    num_local_experts = model.config.num_experts // ep_size
                        
                    if "fc1_linear" in name:
                        experts = torch.split(full_tensor, model.config.moe_intermediate_size * 2)
                        for i in range(ep_size):
                            expert_idx = i * num_local_experts + int(re.findall(r"\d+", prefix_lst[-1])[0])
                            gate_proj = f"{prefix}.{expert_idx}.gate_proj.weight"
                            up_proj = f"{prefix}.{expert_idx}.up_proj.weight"
                            expert2 = torch.split(experts[i], model.config.moe_intermediate_size)
                            full_sd[gate_proj] = expert2[0].cpu()
                            full_sd[up_proj] = expert2[1].cpu()
                    elif "fc2_linear" in name:
                        experts = torch.split(full_tensor, model.config.hidden_size)
                        for i in range(ep_size):
                            expert_idx = i * num_local_experts + int(re.findall(r"\d+", prefix_lst[-1])[0])
                            down_proj = f"{prefix}.{expert_idx}.down_proj.weight"
                            full_sd[down_proj] = experts[i].cpu()
                elif "mlp.router" in name:
                    new_name = name.replace("router", "gate")
                    full_sd[new_name] = full_tensor.cpu()
                else:
                    full_sd[name] = full_tensor.cpu()
            elif is_rank_in_mesh(FIRST_RANK, ep) and "mlp.experts" in name:
                full_tensor = all_gather_tensor(full_tensor, ep_name="ep")
                del full_tensor
            else:
                del full_tensor

        return full_sd

    @staticmethod
    def load_checkpoint_shard_async(model, save_path, ckpt_group, extra={}):
        """
        Load an FSDP checkpoint for this rank.
        Every rank will load its own model shard.
        """
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        checkpoint_id = save_path
        optimizer = extra.get("optimizer", None)
        lr_scheduler = extra.get("lr_scheduler", None)
        state_dict = dict()
        model_sd = model.state_dict()
        # HACK: change device mesh to use dcp
        for name in list(model_sd.keys()):
            if "mlp.experts.weight" in name:
                local_tensor = model_sd[name].to_local()
                model_sd[name] = DTensor.from_local(local_tensor, device_mesh=WORLD_MESH, placements=[Shard(0)])
        state_dict["model"] = model_sd
        if optimizer is not None:
            state_dict["optimizer"] = optimizer.state_dict()
        if lr_scheduler is not None:
            state_dict["lr_scheduler"] = lr_scheduler.state_dict()
        
        dcp.state_dict_loader.load(state_dict, checkpoint_id=checkpoint_id, process_group=ckpt_group)
        # HACK: resume device mesh
        for name in list(state_dict["model"].keys()):
            if "expert_bias" in name:
                model_sd[name] = model.state_dict()[name]
            if "mlp.experts.weight" in name:
                local_tensor = state_dict["model"][name].to_local()
                old_mesh = model.state_dict()[name].device_mesh
                state_dict["model"][name] = DTensor.from_local(local_tensor, device_mesh=old_mesh, placements=[Shard(0)])
        model.load_state_dict(state_dict["model"])
        if optimizer is not None:
            optimizer.load_state_dict(state_dict["optimizer"])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(state_dict["lr_scheduler"])

        # Load RNG state
        rng_path = os.path.join(save_path, f"rng_world_size_{world_size}_rank_{rank}.pt")
        if os.path.exists(rng_path):
            rng_states = torch.load(rng_path, weights_only=False)
            import random
            import numpy as np
            random.setstate(rng_states["python"])
            np.random.set_state(rng_states["numpy"])
            torch.random.set_rng_state(rng_states["cpu"])
            # Load device-specific RNG state based on device type
            device_name = get_device_name()
            if device_name == "npu":
                torch.npu.random.set_rng_state(rng_states["npu"])
            elif device_name == "cuda":
                torch.cuda.random.set_rng_state(rng_states["cuda"])
            print(f"Rank: {rank} - Loaded RNG state from {os.path.abspath(rng_path)}")
        else:
            print(f"Rank: {rank} - RNG checkpoint not found at {rng_path}")

        # wait for everyone to load checkpoints
        torch.distributed.barrier()

    @staticmethod
    def load_checkpoint_shard(model, save_path, extra={}):
        """
        Load an FSDP checkpoint for this rank.
        Every rank will load its own model shard.
        """
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        model_path = os.path.join(save_path, f"model_world_size_{world_size}_rank_{rank}.pt")
        model_state_dict = torch.load(model_path, weights_only=False)
        model.load_state_dict(model_state_dict)
        print(f"Rank: {rank} - Loaded model from {os.path.abspath(model_path)}")

        optimizer = extra.get("optimizer", None)
        if optimizer is not None:
            # load optimizer state dict
            optim_path = os.path.join(save_path, f"optim_world_size_{world_size}_rank_{rank}.pt")
            if os.path.exists(optim_path):
                optimizer_state_dict = torch.load(optim_path, weights_only=False)
                optimizer.load_state_dict(optimizer_state_dict)
                print(f"Rank: {rank} - Loaded optimizer from {os.path.abspath(optim_path)}")
            else:
                print(f"Rank: {rank} - Optimizer checkpoint not found at {optim_path}")

        lr_scheduler = extra.get("lr_scheduler", None)
        if lr_scheduler is not None:
            # load lr scheduler state dict
            scheduler_path = os.path.join(save_path, f"scheduler_world_size_{world_size}_rank_{rank}.pt")
            if os.path.exists(scheduler_path):
                lr_scheduler_state_dict = torch.load(scheduler_path, weights_only=False)
                lr_scheduler.load_state_dict(lr_scheduler_state_dict)
                print(f"Rank: {rank} - Loaded lr scheduler from {os.path.abspath(scheduler_path)}")
            else:
                print(f"Rank: {rank} - Lr scheduler checkpoint not found at {scheduler_path}")

        # Load RNG state
        rng_path = os.path.join(save_path, f"rng_world_size_{world_size}_rank_{rank}.pt")
        if os.path.exists(rng_path):
            rng_states = torch.load(rng_path, weights_only=False)
            import random
            import numpy as np
            random.setstate(rng_states["python"])
            np.random.set_state(rng_states["numpy"])
            torch.random.set_rng_state(rng_states["cpu"])
            # Load device-specific RNG state based on device type
            device_name = get_device_name()
            if device_name == "npu":
                torch.npu.random.set_rng_state(rng_states["npu"])
            elif device_name == "cuda":
                torch.cuda.random.set_rng_state(rng_states["cuda"])
            print(f"Rank: {rank} - Loaded RNG state from {os.path.abspath(rng_path)}")
        else:
            print(f"Rank: {rank} - RNG checkpoint not found at {rng_path}")

        # wait for everyone to load checkpoints
        torch.distributed.barrier()

    @staticmethod
    def save_checkpoint_shard_async(model, save_path, ckpt_group, save_only_model=False, extra={}):
        """
        Save an FSDP checkpoint for this rank.
        Every rank will save its own model shard.
        """
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        optimizer = extra.get("optimizer", None)
        lr_scheduler = extra.get("lr_scheduler", None)
        state_dict = dict()
        model_sd = model.state_dict()
        # HACK: change device mesh to use dcp
        for name in list(model_sd.keys()):
            if "mlp.experts.weight" in name:
                local_tensor = model_sd[name].to_local()
                model_sd[name] = DTensor.from_local(local_tensor, device_mesh=WORLD_MESH, placements=[Shard(0)])
        state_dict["model"] =  model_sd
        if not save_only_model:
            if optimizer is not None:
                state_dict["optimizer"] = optimizer.state_dict()
            if lr_scheduler is not None:
                state_dict["lr_scheduler"] = lr_scheduler.state_dict()
        future = dcp.state_dict_saver.async_save(state_dict, checkpoint_id=save_path, process_group=ckpt_group)
        # Save RNG state
        rng_path = os.path.join(save_path, f"rng_world_size_{world_size}_rank_{rank}.pt")
        import random
        import numpy as np
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        # Save device-specific RNG state based on device type
        device_name = get_device_name()
        if device_name == "npu":
            rng_states["npu"] = torch.npu.random.get_rng_state()
        elif device_name == "cuda":
            rng_states["cuda"] = torch.cuda.random.get_rng_state()
        torch.save(rng_states, rng_path)
        print(f"Rank: {rank} - Saved rng to {os.path.abspath(rng_path)}")

        # wait for everyone to save checkpoints
        torch.distributed.barrier()
        
        return future

    @staticmethod
    def save_checkpoint_shard(model, save_path, save_only_model=False, extra={}):
        """
        Save an FSDP checkpoint for this rank.
        Every rank will save its own model shard.
        """
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        model_path = os.path.join(save_path, f"model_world_size_{world_size}_rank_{rank}.pt")
        model_state_dict = model.state_dict()
        torch.save(model_state_dict, model_path)
        print(f"Rank: {rank} - Saved model to {os.path.abspath(model_path)}")

        if not save_only_model:
            optimizer = extra.get("optimizer", None)
            if optimizer is not None:
                # save optimizer state dict
                optim_path = os.path.join(save_path, f"optim_world_size_{world_size}_rank_{rank}.pt")
                optimizer_state_dict = optimizer.state_dict()
                torch.save(optimizer_state_dict, optim_path)
                print(f"Rank: {rank} - Saved optimizer to {os.path.abspath(optim_path)}")

            lr_scheduler = extra.get("lr_scheduler", None)
            if lr_scheduler is not None:
                # save lr scheduler state dict
                scheduler_path = os.path.join(save_path, f"scheduler_world_size_{world_size}_rank_{rank}.pt")
                lr_scheduler_state_dict = lr_scheduler.state_dict()
                torch.save(lr_scheduler_state_dict, scheduler_path)
                print(f"Rank: {rank} - Saved lr scheduler to {os.path.abspath(scheduler_path)}")

            # Save RNG state
            rng_path = os.path.join(save_path, f"rng_world_size_{world_size}_rank_{rank}.pt")
            import random
            import numpy as np
            rng_states = {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "cpu": torch.random.get_rng_state(),
            }
            # Save device-specific RNG state based on device type
            device_name = get_device_name()
            if device_name == "npu":
                rng_states["npu"] = torch.npu.random.get_rng_state()
            elif device_name == "cuda":
                rng_states["cuda"] = torch.cuda.random.get_rng_state()
            torch.save(rng_states, rng_path)
            print(f"Rank: {rank} - Saved rng to {os.path.abspath(rng_path)}")

        # wait for everyone to save checkpoints
        torch.distributed.barrier()

    @staticmethod
    def pt2hf(args, save_path, hf_save_path, config_path):
        MoeUtils.init_mesh(args)
        config = AutoConfig.from_pretrained(config_path)
        model = MoeUtils.build_model(args, config)
        model = MoeUtils.fsdp_wrap(model, get_device_id())
        MoeUtils.patch_moe_layer(model, config, args, get_device_id())
        MoeUtils.load_checkpoint_shard(model, save_path)
        MoeUtils.save_model(model, hf_save_path)

    @staticmethod
    def per_tensor_generate(model):
        """ per-tensor generator
        """
        if hasattr(model, "_checkpoint_conversion_mapping"):
            reverse_key_mapping = {v: k for k, v in model._checkpoint_conversion_mapping.items()}
        else:
            reverse_key_mapping = {}

        for name, param in model.named_parameters():
            if "mlp.experts" in name:
                prefix = name[:-8]
                num_experts = model.config.num_experts
                full_tensor = param.full_tensor()
                if "weight1" in name:
                    full_tensor = all_gather_tensor(full_tensor, ep_name="ep")
                    experts = torch.split(full_tensor, model.config.hidden_size)
                    for i in range(num_experts):
                        gate_proj = f"{prefix}.{i}.gate_proj.weight"
                        up_proj = f"{prefix}.{i}.up_proj.weight"
                        expert2 = torch.split(experts[i].t(), model.config.moe_intermediate_size)
                        yield from zip([convert_param_name(gate_proj, reverse_key_mapping)], [expert2[0]])
                        yield from zip([convert_param_name(up_proj, reverse_key_mapping)], [expert2[1]])
                else:
                    # weight2
                    full_tensor = all_gather_tensor(full_tensor, ep_name="ep")
                    experts = torch.split(full_tensor, model.config.moe_intermediate_size)
                    for i in range(num_experts):
                        down_proj = f"{prefix}.{i}.down_proj.weight"
                        yield from zip([convert_param_name(down_proj, reverse_key_mapping)], [experts[i].t()])

            elif "mlp.router" in name:
                new_name = name.replace("router", "gate")
                yield from zip([convert_param_name(new_name, reverse_key_mapping)], [param.full_tensor()])
            else:
                yield from zip([convert_param_name(name, reverse_key_mapping)], [param.full_tensor()])

    @staticmethod
    def load_safetensors_dir_base(model_dir):
        # quick start for 1 layer hack when model is very large
        from pathlib import Path
        from safetensors.torch import load_file

        state_dict = {}
        model_path = Path(model_dir)

        for safetensor_file in model_path.glob("*.safetensors"):
            file_path = safetensor_file.as_posix()
            state_dict.update(load_file(file_path, device="cpu"))

        return state_dict
