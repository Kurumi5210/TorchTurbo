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

from torch_turbo.moe.megatron_utils import MegatronUtils
from torch_turbo.moe.moe_utils import MoeUtils


def setup(rank, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group("hccl", rank=rank, world_size=args.world_size)
    torch_npu.npu.set_device(rank)
    MegatronUtils.initialize_model_parallel(expert_model_parallel_size=args.ep_size)

    dist.barrier()

def cleanup():
    dist.destroy_process_group()

def print_model_params(model):
    for name, param in model.named_parameters():
        print(f'name {name}, param.shape {param.shape}, param.device {param.device}')

def build_model(path):
    config = AutoConfig.from_pretrained(path)
    with torch.device("meta"):
        model = AutoModel.from_config(config)

        if torch.distributed.get_rank() == 0:
            print(model.layers[0].mlp)
            print_model_params(model.layers[0].mlp)
            print('-'*50)

        MoeUtils.transform_moe_layer(model, config)

        if torch.distributed.get_rank() == 0:
            print(model.layers[0].mlp)
            print_model_params(model.layers[0].mlp)
            print('-'*50)

    return model

def run(rank, args):
    world_size = args.world_size
    setup(rank, args)
    device = torch.device(f"npu:{rank}")

    world_mesh = init_device_mesh("npu", (world_size,))
    mesh = init_device_mesh("npu", (world_size // 2, 2), mesh_dim_names=("edp", "ep"))

    model = build_model(args.path)


    cleanup()


def main(args):
    mp.spawn(run, args=(args,), nprocs=args.world_size, join=True)


# command:
#   export PYTHONPATH=/sfs_turbo/xiyun/rl/3d/opensource/Megatron-LM:/sfs_turbo/xiyun/rl/3d/opensource/MindSpeed:/sfs_turbo/xiyun/rl/new/test/torch_turbo:$PYTHONPATH
#   python example/fsdp2/run_qwen3_ep_npu.py  --path /sfs_turbo/pretrained_models/Qwen3-30B-A3B
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=16)
    parser.add_argument("--ep_size", type=int, default=4)
    parser.add_argument("--path", type=str)
    args = parser.parse_args()

    main(args)
