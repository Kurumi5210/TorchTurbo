import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh, Replicate, Shard
from transformers import AutoModelForCausalLM
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
import torch.nn.functional as F

import torch_npu

from torchturbo.torch_turbo.distributed.parallel_state import (
    initialize_model_parallel,
    get_expert_model_parallel_rank,
    get_expert_model_parallel_world_size
)
from torch_turbo.moe.moe_utils import MoeUtils, load_safetensors_dir, slice_tensor_for_ep_rank
from torch_turbo.utils.torch_utils import compare_tensors

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoConfig

def verify_weight_loading(path, model, device):
    rank = dist.get_rank()
    ep_rank = get_expert_model_parallel_rank()
    ep_world_size = get_expert_model_parallel_world_size()
    
    # Step 1: Load original state dict from safetensors
    sd = load_safetensors_dir(path)
    dist.barrier()
    # Step 2: Verify weights
    for param_name, param in model.named_parameters():
        if "mlp.experts" in param_name:
            prefix = param_name[:-8]
            if "weight1" in param_name:
                w1 = []
                for i in range(model.config.num_experts):
                    gate_proj = f"{prefix}.{i}.gate_proj.weight"
                    up_proj = f"{prefix}.{i}.up_proj.weight"
                    fc1 = torch.cat([sd[gate_proj], sd[up_proj]])
                    w1.append(fc1)
                full_tensor = torch.stack(w1, dim=0).to(device)
                full_tensor = full_tensor.permute(0, 2, 1)
            else:
                # weight2
                w2 = []
                for i in range(model.config.num_experts):
                    down_proj = f"{prefix}.{i}.down_proj.weight"
                    w2.append(sd[down_proj].t())
                full_tensor = torch.stack(w2, dim=0).to(device)
            full_tensor = full_tensor.reshape(-1, full_tensor.shape[-1])
            # each ep slice its part
            full_tensor = slice_tensor_for_ep_rank(full_tensor, ep_rank, ep_world_size)
        elif "mlp.router" in param_name:
            new_name = param_name.replace("router", "gate")
            full_tensor = sd[new_name].to(device)
        else:
            full_tensor = sd[param_name].to(device)
        # Compare
        full_tensor_cpu = full_tensor.cpu().float()
        param_tensor_cpu = param.full_tensor().cpu().float()
        compare_tensors(param_name, full_tensor_cpu, param_tensor_cpu, rank)
        dist.barrier()
    
    print("finish verifying weight loading")
        
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

def run(rank, args):
    setup(rank, args)
    device = torch.device(f"npu:{rank}")

    MoeUtils.init_mesh(args)
    config = AutoConfig.from_pretrained(args.golden_path)
    model = MoeUtils.build_model(config)
    model = MoeUtils.fsdp_wrap(model, device)
    MoeUtils.patch_moe_layer(model, config, device)
    MoeUtils.load_checkpoint_shard(model, args.path)
    
    verify_weight_loading(args.golden_path, model, device)

    cleanup()


def main(args):
    mp.spawn(run, args=(args,), nprocs=args.world_size, join=True)


# command:
#   export PYTHONPATH=/sfs_turbo/xiyun/rl/3d/opensource/Megatron-LM:/sfs_turbo/xiyun/rl/3d/opensource/MindSpeed:/sfs_turbo/xiyun/rl/new/test/tt/torchturbo:$PYTHONPATH
#   python example/fsdp2/test_load_weight.py  --path /sfs_turbo/pretrained_models/Qwen3-30B-A3B
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=16)
    parser.add_argument("--ep_size", type=int, default=4)
    parser.add_argument("--path", type=str, default="/sfs_turbo/qinjue/models/Qwen3-30B-A3B/")
    parser.add_argument("--golden_path", type=str, default="/sfs_turbo/pretrained_models/Qwen3-30B-A3B")
    parser.add_argument("--profiling", default=False, action="store_true")
    parser.add_argument("--average_in_collective", default=True, action="store_true")
    args = parser.parse_args()

    main(args)