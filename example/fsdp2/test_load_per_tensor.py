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

from torch_turbo.distributed.parallel import initialize_model_parallel
from torch_turbo.moe.moe_utils import MoeUtils, load_safetensors_dir, FIRST_RANK


def setup(rank, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group("hccl", rank=rank, world_size=args.world_size)
    torch_npu.npu.set_device(rank)
    initialize_model_parallel(expert_model_parallel_size=args.ep_size)

    dist.barrier()


def cleanup():
    dist.barrier()
    dist.destroy_process_group()


def compare_gen(data, golden_data, device="cpu"):
    """data compare for generator
    """
    for key, value in data:
        assert key in golden_data, f"key {key} not in golden_data"
        if isinstance(value, DTensor):
            value = value.full_tensor()
        torch.allclose(value.to(device), golden_data[key])

def run(rank, args):
    enable_record = False
    if enable_record:
        torch.npu.memory._record_memory_history()
    world_size = args.world_size
    setup(rank, args)
    device = torch.device(f"npu:{rank}")

    model = MoeUtils.create_fsdp2_moe_model(args, device)
    sd_gen = MoeUtils.per_tensor_generate(model)
    if enable_record:
        torch.npu.memory._dump_snapshot(f"./memory/rank_{rank}_mem.pickle")
        torch.npu.memory._record_memory_history(enabled=None)
    if rank == FIRST_RANK:
        print(model)
        for param_name, param in model.named_parameters():
            print(f'dd param_name {param_name}, param.shape {param.shape}, type {type(param)}')
    for name, param in sd_gen:
        if rank == FIRST_RANK:
            print("[CHECK] name, param:", name, param.shape)
    golden_data = load_safetensors_dir(args.path)
    compare_gen(sd_gen, golden_data)

    cleanup()


def main(args):
    mp.spawn(run, args=(args,), nprocs=args.world_size, join=True)


# command:
#   export PYTHONPATH=/sfs_turbo/xiyun/rl/3d/opensource/Megatron-LM:/sfs_turbo/xiyun/rl/3d/opensource/MindSpeed:/sfs_turbo/xiyun/rl/new/test/tt/torchturbo:$PYTHONPATH
#   python example/fsdp2/run_qwen3_ep_npu.py  --path /sfs_turbo/pretrained_models/Qwen3-30B-A3B
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=16)
    parser.add_argument("--ep_size", type=int, default=4)
    parser.add_argument("--path", type=str)
    args = parser.parse_args()

    main(args)
