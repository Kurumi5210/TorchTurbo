import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
import torch.nn.functional as F

from torch.distributed.device_mesh import init_device_mesh

from torch.distributed._tensor import (
    DeviceMesh,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)
from torch.distributed.fsdp import fully_shard


import torch_npu

from transformers import AutoConfig, AutoModel


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    torch_npu.npu.set_device(rank)

    dist.barrier()

def cleanup():
    dist.destroy_process_group()

def build_model(path):
    config = AutoConfig.from_pretrained(path)
    with torch.device("meta"):
        model = AutoModel.from_config(config)

    return model

def is_rank_in_mesh(target_rank, device_mesh):
    return target_rank in device_mesh.mesh.flatten().tolist()


def load_safetensors_dir(model_dir: str) -> dict:
    from pathlib import Path
    from safetensors.torch import load_file

    state_dict = {}
    model_path = Path(model_dir)

    for safetensor_file in model_path.glob("*.safetensors"):
        file_path = safetensor_file.as_posix()
        state_dict.update(load_file(file_path, device="cpu"))

    return state_dict


def init_model(model, rank, path, mesh, ep_mesh, device):
    dtype = torch.bfloat16
    ep = ep_mesh['ep']
    sd = {}
    if rank == 0:
        sd = load_safetensors_dir(path)

    torch.distributed.barrier()

    if is_rank_in_mesh(0, ep):
        if rank == 0:
            sds = [sd]
        else:
            sds = [None]
        dist.broadcast_object_list(sds, src=0, group=ep.get_group(0))
        sd = sds[0]

    torch.distributed.barrier()

    meta_sharded_sd = model.state_dict()
    sharded_sd = {}
    if is_rank_in_mesh(0, ep):
        for param_name, sharded_meta_param in meta_sharded_sd.items():
            full_tensor = sd[f'model.{param_name}'].to(device)
            mesh = sharded_meta_param.device_mesh
            sharded_tensor = distribute_tensor(
                full_tensor, mesh, sharded_meta_param.placements
            )
            sharded_sd[param_name] = nn.Parameter(sharded_tensor)
    else:
        for param_name, sharded_meta_param in meta_sharded_sd.items():
            full_tensor = torch.empty(
                sharded_meta_param.size(),
                device=device,
                dtype=sharded_meta_param.dtype,
            )
            mesh = sharded_meta_param.device_mesh
            sharded_tensor = distribute_tensor(
                full_tensor, mesh, sharded_meta_param.placements
            )
            sharded_sd[param_name] = nn.Parameter(sharded_tensor)

    model.load_state_dict(sharded_sd, assign=True)

    model.unshard()  # no lazy init yet
    for layer in model.layers:
        layer.unshard()

    if rank == 2:
        for param_name, param in model.named_parameters():
            sd = load_safetensors_dir(path)
            #print(f'param_name {param_name}, shape {param.shape}')
            full_tensor = sd[f'model.{param_name}'].to(device)
            if not torch.allclose(full_tensor, param):
                print(f'--->param_name {param_name} not same as sd loaded in model path')


def fsdp_wrap(model, device, world_mesh, mesh):
    for layer in model.layers:
        # simulate moe
        fully_shard(layer, mesh = mesh['edp'])
    # simulate non-moe part
    fully_shard(model, mesh = world_mesh)
    model.to_empty(device=device)
    return model

def print_sharding_info(model):
    for name, param in model.named_parameters():
        if not isinstance(param, torch.Tensor) or not hasattr(param, '_spec'):
            continue

        placements = param._spec.placements
        local_tensor = param.to_local()

        shard_info = []
        global_shape = param.size()
        for i, placement in enumerate(placements):
            if isinstance(placement, torch.distributed._tensor.Shard):
                shard_dim = placement.dim
                num_shards = param.device_mesh.size(i)
                shard_size = global_shape[shard_dim] // num_shards
                shard_info.append(
                    f"Mesh dim {i}: Shard(dim={shard_dim}) "
                    f"shards={num_shards} per_shard_size={shard_size}"
                )
            elif isinstance(placement, torch.distributed._tensor.Replicate):
                shard_info.append(f"Mesh dim {i}: Replicate()")

        print(f"Parameter: {name}")
        print(f"Global shape: {list(global_shape)}")
        print(f"Local shape: {list(local_tensor.shape)}")
        print(f"Sharding strategy: {shard_info}")
        print("-" * 50)

def run(rank, args):
    world_size = args.world_size
    setup(rank, world_size)
    device = torch.device(f"npu:{rank}")

    world_mesh = init_device_mesh("npu", (world_size,))
    mesh = init_device_mesh("npu", (world_size // 2, 2), mesh_dim_names=("edp", "ep"))

    model = build_model(args.path)
    model = fsdp_wrap(model, device, world_mesh, mesh)
    if torch.distributed.get_rank() == 0:
        print(model)
        #print_sharding_info(model)

    init_model(model, rank, args.path, world_mesh, mesh, device)


    cleanup()


def main(args):
    mp.spawn(run, args=(args,), nprocs=args.world_size, join=True)


# command python t1.py --path /sfs_turbo/pretrained_models/Qwen3-0.6B
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=16)
    parser.add_argument("--path", type=str)
    args = parser.parse_args()

    main(args)