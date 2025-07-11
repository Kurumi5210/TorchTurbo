from torch_turbo.moe.megatron_utils import MegatronUtils
from torch_turbo.moe.monkey_patch import apply_monkey_patch
from torch.distributed.device_mesh import init_device_mesh
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
import torch.nn.functional as F

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
from torch.distributed.fsdp import fully_shard

WORLD_MESH = None
EXPERT_MESH = None
FIRST_RANK = 0

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

class MoeUtils:
    @staticmethod
    def patch_moe_layer(model, hf_config, device):
        assert "moe" in hf_config.model_type, f'{hf_config.model_type} must be moe model'
        for layer in model.layers:
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
        rotary_emb = model.rotary_emb
        inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(rotary_emb.config, device)
        rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)

        
    @staticmethod
    def transform_moe_layer(model, hf_config):
        assert "moe" in hf_config.model_type, f'{hf_config.model_type} must be moe model'
        for layer in model.layers:
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

    @staticmethod
    def build_model(config):
        # DEBUG
        # config.num_hidden_layers = 1
        with torch.device("meta"):
            model = AutoModel.from_config(config)
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
    def fsdp_wrap(model, device):
        for layer in model.layers:
            # wrap moe mlp in edp
            fully_shard(layer.mlp, mesh = MoeUtils.get_edp_mesh())
            # wrap other parts in dp
            fully_shard(layer, mesh = MoeUtils.get_world_mesh())
        # wrap remaining parts
        fully_shard(model, mesh = MoeUtils.get_world_mesh())
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
                            gate_proj = f"model.{prefix}.{i}.gate_proj.weight"
                            up_proj = f"model.{prefix}.{i}.up_proj.weight"
                            fc1 = torch.cat([sd[gate_proj], sd[up_proj]])
                            w1.append(fc1)
                        full_tensor = torch.stack(w1, dim=0).to(device)
                        # [e, 2 * i, h] -> [e, h, 2 * i]
                        full_tensor = full_tensor.permute(0, 2, 1)
                    else:
                        # weight2
                        w2 = []
                        for i in range(config.num_experts):
                            down_proj = f"model.{prefix}.{i}.down_proj.weight"
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
                    hf_name = f"model.{new_name}"
                    full_tensor = sd[hf_name].to(device)
                    # broadcast to ep_group
                    dist.broadcast(full_tensor, src=rank, group=ep.get_group(0))
                    mesh = edp
                else:
                    hf_name = f"model.{param_name}"
                    full_tensor = sd[hf_name].to(device)
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
                elif "mlp.router" in param_name:
                    full_tensor = torch.empty(
                        param.size(),
                        device=device,
                        dtype=param.dtype,
                    )
                    # get broadcast tensor
                    if is_rank_in_mesh(FIRST_RANK, ep):
                        dist.broadcast(full_tensor, src=FIRST_RANK, group=ep.get_group(0))
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
        for name, param in model.named_parameters():
            if "mlp.experts" in name:
                prefix = name[:-8]
                num_experts = model.config.num_experts
                full_tensor = param.full_tensor()
                if "weight1" in name:
                    full_tensor = all_gather_tensor(full_tensor, ep_name="ep")
                    experts = torch.split(full_tensor, model.config.hidden_size)
                    for i in range(num_experts):
                        gate_proj = f"model.{prefix}.{i}.gate_proj.weight"
                        up_proj = f"model.{prefix}.{i}.up_proj.weight"
                        expert2 = torch.split(experts[i].t(), model.config.moe_intermediate_size)
                        yield from zip([gate_proj], [expert2[0]])
                        yield from zip([up_proj], [expert2[1]])
                else:
                    # weight2
                    full_tensor = all_gather_tensor(full_tensor, ep_name="ep")
                    experts = torch.split(full_tensor, model.config.moe_intermediate_size)
                    for i in range(num_experts):
                        down_proj = f"model.{prefix}.{i}.down_proj.weight"
                        yield from zip([down_proj], [experts[i].t()])

            elif "mlp.router" in name:
                new_name = name.replace("router", "gate")
                hf_name = f"model.{new_name}"
                yield from zip([hf_name], [param])
            else:
                hf_name = f"model.{name}"
                yield from zip([hf_name], [param])
