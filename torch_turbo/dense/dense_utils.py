import os
from torch_turbo.utils.torch_utils import load_safetensors_dir
from torch_turbo.utils.torch_utils import torch_version
from torch.distributed.device_mesh import init_device_mesh
import torch
from torch import nn

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

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
    from torch.distributed.fsdp._fully_shard import FSDPModule
else:
    from torch.distributed._composable.fsdp import fully_shard
    from torch.distributed._composable.fsdp.fully_shard import FSDPModule

WORLD_MESH = None
FIRST_RANK = 0

class DenseUtils:
    @staticmethod
    def get_world_mesh():
        assert WORLD_MESH != None, "must be"
        return WORLD_MESH

    @staticmethod
    def create_fsdp2_model(args, device):
        DenseUtils.init_mesh(args)

        config = AutoConfig.from_pretrained(args.path)
        model = DenseUtils.build_model(config)
        model = DenseUtils.fsdp_wrap(model, device)
        # patch rotary embedding
        # model.to_empty() will clear the buffers, so we need to patch after that
        rotary_emb = model.model.rotary_emb
        inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(rotary_emb.config, device)
        rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)
        
        DenseUtils.load_weights_from_hf(model, config, device, args.path)
        return model

    @staticmethod
    def init_mesh(args):
        # TODO: change device name to a global variable to switch between cuda and npu
        world_size = args.world_size
        world_mesh = init_device_mesh("npu", (world_size,))
        
        global WORLD_MESH
        WORLD_MESH = world_mesh

    @staticmethod
    def build_model(config):
        # DEBUG
        # config.num_hidden_layers = 5
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
            
        model = model.to(dtype=config.torch_dtype)
        return model
    
    @staticmethod
    def save_model(model, save_path):
        rank = torch.distributed.get_rank()
        sharded_sd = model.state_dict()
        cpu_state_dict = {}
        for param_name, sharded_param in sharded_sd.items():
            full_param = sharded_param.full_tensor()
            if rank == 0:
                cpu_state_dict[param_name] = full_param.cpu()
            else:
                del full_param
                
        if rank == 0:
            with torch.device("meta"):
                save_model = AutoModelForCausalLM.from_config(model.config)
            save_model.to_empty(device="cpu")
            
            hf_local_path = os.path.join(save_path, "huggingface")
            os.makedirs(hf_local_path, exist_ok=True)
            save_model.save_pretrained(hf_local_path, state_dict=cpu_state_dict)
        
        # wait for rank0 to dump hf_model to local
        torch.distributed.barrier()

    @staticmethod
    def set_forward_prefetch_for_fsdp2(fsdp2_model, layer_cls, model_cls):
        model_modules = []
        layer_modules = []

        for name, named_module in fsdp2_model.named_modules():
            if isinstance(named_module, FSDPModule) and isinstance(named_module, model_cls):
                model_modules.append(named_module)

            if isinstance(named_module, FSDPModule) and isinstance(named_module, layer_cls):
                layer_modules.append(named_module)

        assert len(model_modules) == 1

        for i in range(len(layer_modules)):
            # layer prefetch layer
            cur_layer = layer_modules[i]

            # deal with forward
            if i == 0:
               model_modules[0].set_modules_to_forward_prefetch([cur_layer])
            else:
                # layer_{i - 1} prefetch layer_i
                layer_modules[i - 1].set_modules_to_forward_prefetch([cur_layer])

            # deal with backward
            if i > 0:
                # layer_i prefetch layer_{i - 1}
                layer_backward_prefetch_states = [layer_modules[i - 1]._get_fsdp_state()]
                setattr(layer_modules[i], 'backward_prefetch_states', layer_backward_prefetch_states)

    @staticmethod
    def fsdp_wrap(model, device):
        outer_cls = model.model.layers[0].__class__
        outermost_cls = model.__class__
        for layer in model.model.layers:
            fully_shard(layer, mesh = DenseUtils.get_world_mesh())
            setattr(layer, "fsdp2_forward_prefetch", True)
        # wrap remaining parts
        fully_shard(model, mesh = DenseUtils.get_world_mesh())

        DenseUtils.set_forward_prefetch_for_fsdp2(model, outer_cls, outermost_cls)
        model.to_empty(device=device)
        return model

    @staticmethod
    def load_weights_from_hf(model, config, device, path):
        rank = torch.distributed.get_rank()
        # only load state dict from rank 0, and broadcast to all other ranks according device mesh
        if rank == FIRST_RANK:
            sd = load_safetensors_dir(path)
        torch.distributed.barrier()
        sharded_sd = {}
        for param_name, param in model.named_parameters():
            if rank == FIRST_RANK:
                full_tensor = sd[param_name].to(device)
            else:
                full_tensor = torch.empty(
                    param.size(),
                    device=device,
                    dtype=param.dtype,
                )
            sharded_tensor = distribute_tensor(full_tensor, WORLD_MESH, param.placements)
            sharded_sd[param_name] = nn.Parameter(sharded_tensor)

        model.load_state_dict(sharded_sd, assign=True)
        for param in model.parameters():
            assert isinstance(param, DTensor), "must be"
