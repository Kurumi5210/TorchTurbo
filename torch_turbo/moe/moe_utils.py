from typing import Any
from torch_turbo.moe.megatron_utils import MegatronUtils
from torch_turbo.moe.monkey_patch import apply_monkey_patch
from torch_turbo.utils.torch_utils import torch_version
from torch.distributed.device_mesh import init_device_mesh
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.clip_grad import _clip_grads_with_norm_, _get_total_norm

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

import megatron.core.parallel_state as ps
from megatron.core.jit import jit_fuser

# TODO, torch 2.5 and torch 2.6 package name is not the same
from torch.distributed._tensor import (
    DeviceMesh,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)
if torch_version() >= (2, 6, 0):
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
    from torch.distributed.fsdp._fully_shard import FSDPModule
else:
    from torch.distributed._composable.fsdp import fully_shard
    from torch.distributed._composable.fsdp._fsdp_state import FSDPState
    from torch.distributed._composable.fsdp._fsdp_param_group import FSDPParamGroup
    from torch.distributed._composable.fsdp.fully_shard import FSDPModule


WORLD_MESH = None
EXPERT_MESH = None
FIRST_RANK = 0
_GRADIENT_SCALING_FACTOR = 1.0

def load_safetensors_dir(model_dir: str) -> dict:
    from pathlib import Path
    from safetensors.torch import load_file

    state_dict = {}
    model_path = Path(model_dir)

    for safetensor_file in model_path.glob("*.safetensors"):
        file_path = safetensor_file.as_posix()
        state_dict.update(load_file(file_path, device="cpu"))

    return state_dict

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
    import re
    for pattern, replacement in reverse_key_mapping.items():
        replacement = replacement.lstrip("^")  # strip off un-needed chars and patterns
        replacement = re.sub(r"\(.*\)", "", replacement)
        key, n_replace = re.subn(pattern, replacement, key)
        # Early exit of the loop
        if n_replace > 0:
            return key
    return key

# BackwardPrefetchContext is designed to switch prefetch logic between forward & backward.
#   During forward, prefetch next layer
#   During backward, prefetch previous layer
class BackwardPrefetchContext:
    def __init__(self):
        self.in_backward = False

        def pre_forward_hook(state, module, args, kwargs):
            old_prefetch_list = None
            cur_prefetch_list = getattr(module, "backward_prefetch_states", [])
            if self.in_backward and cur_prefetch_list != []:
                old_prefetch_list = state._states_to_forward_prefetch
                state._states_to_forward_prefetch = cur_prefetch_list

            res = self.old_pre_forward(state, module, args, kwargs)

            if old_prefetch_list is not None:
                state._states_to_forward_prefetch = old_prefetch_list

            return res

        self.old_pre_forward = FSDPState._pre_forward
        FSDPState._pre_forward = pre_forward_hook


        def post_forward_hook(state, module: nn.Module, input: Any, output: Any):
            old_mesh_info = None
            cur_prefetch_list = getattr(module, "backward_prefetch_states", [])
            if self.in_backward and cur_prefetch_list != []:
                old_mesh_info = state._fsdp_param_group.post_forward_mesh_info
                state._fsdp_param_group.post_forward_mesh_info = None

            res = self.old_post_forward(state, module, input, output)

            if old_mesh_info != None:
                state._fsdp_param_group.post_forward_mesh_info = old_mesh_info

            return res

        self.old_post_forward = FSDPState._post_forward
        FSDPState._post_forward = post_forward_hook

        def backward_prefetch_hook(group):
            return

        self.old_backward_prefetch = FSDPParamGroup._backward_prefetch
        FSDPParamGroup._backward_prefetch = backward_prefetch_hook


    def __enter__(self):
        self.in_backward = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.in_backward = False

fsdp2_backward_prefetch_manager = BackwardPrefetchContext()

class MoeUtils:
    @staticmethod
    def patch_moe_layer(model, hf_config, device):
        assert "moe" in hf_config.model_type, f'{hf_config.model_type} must be moe model'
        for layer in model.model.layers:
            # patch token_dispatcher
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
            
            # patch grouped mlp
            @jit_fuser
            def glu(x, probs):
                x = torch.chunk(x, 2, dim=-1)
                return layer.mlp.experts.config.activation_func(x[0]) * x[1]
            layer.mlp.experts.activation_func_with_probs = glu
        
        # patch rotary embedding
        rotary_emb = model.model.rotary_emb
        inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(rotary_emb.config, device)
        rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def scale_moe_gradient(params):
        assert params[0].device_mesh == MoeUtils.get_edp_mesh()
        grads = [p.grad for p in params]
        torch._foreach_mul_(grads, _GRADIENT_SCALING_FACTOR)
    
    @staticmethod
    def clip_grad_norm(model, max_norm=1.0, norm_type=2.0):
        params = []
        grads = []
        moe_params = []
        moe_grads = []
        for param in model.parameters():
            if param.device_mesh == MoeUtils.get_edp_mesh():
                moe_params.append(param)
                moe_grads.append(param.grad)
            else:
                params.append(param)
                grads.append(param.grad)

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
    def transform_moe_layer(model, hf_config):
        assert "moe" in hf_config.model_type, f'{hf_config.model_type} must be moe model'
        for layer in model.model.layers:
            moe = MegatronUtils.initialize_moe_layer_from_hf_config(hf_config)
            layer.mlp = moe

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

        config = AutoConfig.from_pretrained(args.path)
        model = MoeUtils.build_model(config)
        model = MoeUtils.fsdp_wrap(model, device)
        MoeUtils.patch_moe_layer(model, config, device)
        MoeUtils.load_weights_from_hf(model, config, device, args.path)
        return model

    @staticmethod
    def init_mesh(args):
        # TODO: change device name to a global variable to switch between cuda and npu
        world_size = args.world_size
        world_mesh = init_device_mesh("npu", (world_size,))
        ep_mesh = init_device_mesh("npu", (world_size // args.ep_size, args.ep_size), mesh_dim_names=("edp", "ep"))
        gradient_scaling_factor = (1 / args.ep_size) if args.average_in_collective else 1.0

        def is_same_group(process_group, device_mesh):
            mesh_group = device_mesh.get_group(0)
            mesh_ranks = dist.get_process_group_ranks(mesh_group)
            given_ranks = dist.get_process_group_ranks(process_group)
            return mesh_ranks == given_ranks

        # verify ep group is the same with ep_mesh
        megatron_ep_group = ps.get_expert_model_parallel_group()
        assert is_same_group(megatron_ep_group, ep_mesh['ep'])

        global WORLD_MESH
        WORLD_MESH = world_mesh
        global EXPERT_MESH
        EXPERT_MESH = ep_mesh
        global _GRADIENT_SCALING_FACTOR
        _GRADIENT_SCALING_FACTOR = gradient_scaling_factor

    @staticmethod
    def build_model(config):
        # DEBUG
        # config.num_hidden_layers = 1
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
            apply_monkey_patch()
            MoeUtils.transform_moe_layer(model, config)
            
        model = model.to(dtype=config.torch_dtype)
        return model

    @staticmethod
    def build_model_from_config(args):
        # DEBUG
        config = AutoConfig.from_pretrained(args.path)
        config.num_hidden_layers = 1
        model = AutoModel.from_config(config)
        return model

    @staticmethod
    def set_moe_forward_prefetch_for_fsdp2_ep(fsdp2_model, layer_cls, expert_cls, model_cls):
        model_modules = []
        layer_modules = []
        expert_modules = []

        for name, named_module in fsdp2_model.named_modules():
            if isinstance(named_module, FSDPModule) and isinstance(named_module, model_cls):
                model_modules.append(named_module)

            if isinstance(named_module, FSDPModule) and isinstance(named_module, layer_cls):
                layer_modules.append(named_module)

            if isinstance(named_module, FSDPModule) and isinstance(named_module, expert_cls):
                expert_modules.append(named_module)

        assert len(model_modules) == 1
        assert len(layer_modules) == len(expert_modules)

        for i in range(len(layer_modules)):
            # layer prefetch layer
            # expert prefetch expert
            cur_expert = expert_modules[i]
            cur_layer = layer_modules[i]

            # deal with forward
            if i == 0:
               model_modules[0].set_modules_to_forward_prefetch([cur_layer, cur_expert])
            else:
                # layer_{i - 1} prefetch layer_i
                layer_modules[i - 1].set_modules_to_forward_prefetch([cur_layer])
                expert_modules[i - 1].set_modules_to_forward_prefetch([cur_expert])

            # deal with backward
            if i > 0:
                # layer_i prefetch layer_{i - 1}
                layer_backward_prefetch_states = [layer_modules[i - 1]._get_fsdp_state()]
                setattr(layer_modules[i], 'backward_prefetch_states', layer_backward_prefetch_states)
                expert_backward_prefetch_states = [expert_modules[i - 1]._get_fsdp_state()]
                setattr(expert_modules[i], 'backward_prefetch_states', expert_backward_prefetch_states)

    @staticmethod
    def fsdp_wrap(model, device):
        outer_cls = model.model.layers[0].__class__
        inter_cls = model.model.layers[0].mlp.experts.__class__
        outermost_cls = model.__class__
        for layer in model.model.layers:
            # wrap moe mlp in edp
            fully_shard(layer.mlp.experts, mesh = MoeUtils.get_edp_mesh())
            setattr(layer.mlp.experts, "expert_forward_prefetch", True)
            # wrap other parts in dp
            fully_shard(layer, mesh = MoeUtils.get_world_mesh())
            setattr(layer, "moe_forward_prefetch", True)
        # wrap remaining parts
        fully_shard(model, mesh = MoeUtils.get_world_mesh())

        MoeUtils.set_moe_forward_prefetch_for_fsdp2_ep(model, outer_cls, inter_cls, outermost_cls)
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
    def load_weights_from_hf(model, config, device, path):
        rank = torch.distributed.get_rank()
        # only load state dict from rank 0, and broadcast to alll other ranks according device mesh
        if rank == FIRST_RANK:
            sd = load_safetensors_dir(path)
        torch.distributed.barrier()

        ep = EXPERT_MESH['ep']
        edp = EXPERT_MESH['edp']

        ep_rank = ps.get_expert_model_parallel_rank()
        ep_world_size = ps.get_expert_model_parallel_world_size()
        
        sharded_sd = {}
        for param_name, param in model.named_parameters():
            if rank == FIRST_RANK:
                if "mlp.experts" in param_name:
                    prefix = param_name[:-8]
                    if "weight1" in param_name:
                        w1 = []
                        for i in range(config.num_experts):
                            gate_proj = f"{prefix}.{i}.gate_proj.weight"
                            up_proj = f"{prefix}.{i}.up_proj.weight"
                            fc1 = torch.cat([sd[gate_proj], sd[up_proj]])
                            w1.append(fc1)
                        full_tensor = torch.stack(w1, dim=0).to(device)
                        # [e, 2 * i, h] -> [e, h, 2 * i]
                        full_tensor = full_tensor.permute(0, 2, 1)
                    else:
                        # weight2
                        w2 = []
                        for i in range(config.num_experts):
                            down_proj = f"{prefix}.{i}.down_proj.weight"
                            w2.append(sd[down_proj].t())
                        full_tensor = torch.stack(w2, dim=0).to(device)
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

        model.load_state_dict(sharded_sd, assign=True)
        for param in model.parameters():
            assert isinstance(param, DTensor), "must be"

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
