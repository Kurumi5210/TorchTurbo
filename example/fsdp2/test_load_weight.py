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


from torch_turbo.moe.megatron_utils import MegatronUtils
from torch_turbo.moe.moe_utils import MoeUtils, load_safetensors_dir, slice_tensor_for_ep_rank

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoConfig
import megatron.core.parallel_state as ps

def verify_weight_loading(path, model, device):
    config = AutoConfig.from_pretrained(path)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    ep = MoeUtils.get_ep_mesh()
    ep_rank = ps.get_expert_model_parallel_rank()
    ep_world_size = ps.get_expert_model_parallel_world_size()
    # print(f"Rank {rank} - EP Rank: {ep_rank}, EP World Size: {ep_world_size}")
    
    # Step 1: Load original state dict from safetensors
    sd = load_safetensors_dir(path)
    dist.barrier()
    
    # Step 2: Verify weights
    for param_name, param in model.named_parameters():
        # if "embed_tokens" in param_name:
        #     # print(f"Rank {rank} - {param_name} value: local tensor value: {param.to_local()}")
        #     print(f"Rank {rank} - {param_name} value: local tensor value: {param.to_local()}")
        #     # print(f"Rank {rank} - {param_name} value: full tensor value: {param.full_tensor()}")
        # continue
        # assert not isinstance(param, DTensor), f"{param_name} must not be DTensor"
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
            else:
                # weight2
                w2 = []
                for i in range(config.num_experts):
                    down_proj = f"model.{prefix}.{i}.down_proj.weight"
                    w2.append(sd[down_proj].t())
                full_tensor = torch.stack(w2, dim=0).to(device)
            # broadcast to ep_group
            full_tensor = full_tensor.reshape(-1, full_tensor.shape[-1])

            # each ep slice its part
            full_tensor = slice_tensor_for_ep_rank(full_tensor, ep_rank, ep_world_size)
            if "weight1" in param_name:
                full_tensor = full_tensor.t()
            
        elif "mlp.router" in param_name:
            new_name = param_name.replace("router", "gate")
            hf_name = f"model.{new_name}"
            full_tensor = sd[hf_name].to(device)
        else:
            hf_name = f"model.{param_name}"
            full_tensor = sd[hf_name].to(device)
        # Compare
        full_tensor_cpu = full_tensor.cpu().float()
        param_tensor_cpu = param.full_tensor().cpu().float()
        # 首先检查形状是否匹配
        if full_tensor_cpu.shape != param_tensor_cpu.shape:
            print(f"❌ Shape mismatch for {param_name}:")
            print(f"  Expected shape: {full_tensor_cpu.shape}")
            print(f"  Actual shape: {param_tensor_cpu.shape}")
            dist.barrier()
            if rank == 0:
                print("\n❌ Weight loading failed due to shape mismatch!")
            exit(1)

        # 然后检查数值是否匹配
        if not torch.allclose(full_tensor_cpu, param_tensor_cpu, atol=1e-5) and rank == 0:
            # 计算最大差异
            diff = torch.abs(full_tensor_cpu - param_tensor_cpu)
            max_diff = torch.max(diff)
            mean_diff = torch.mean(diff)
            
            print(f"❌ Value mismatch for {param_name}, device={device}:")
            print(f"  Max difference: {max_diff.item():.6f}, device={device}")
            print(f"  Mean difference: {mean_diff.item():.6f}, device={device}")
            print(f"  Expected values sample: {full_tensor_cpu.flatten()[:-1]}, device={device}")
            print(f"  Actual values sample: {param_tensor_cpu.flatten()[:-1]}, device={device}")
            print(f"  Actual shape: {param_tensor_cpu.shape}, device={device}")
            print(f"  Original DTensor values sample: {param.to_local().flatten()[:-1]}, device={device}")
            print(f"  Original DTensor shape: {param.to_local().shape}, device={device}")
            
            dist.barrier()
            if rank == 0:
                print("\n❌ Weight loading failed due to value mismatch!")
            exit(1)
        
    # dist.barrier()
    if rank == 0:
        print("\n✅ All weights matched successfully!")
        
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

def run(rank, args):
    world_size = args.world_size
    setup(rank, args)
    device = torch.device(f"npu:{rank}")

    model = MoeUtils.create_fsdp2_moe_model(args, device)
    dist.barrier()
    
    verify_weight_loading(args.path, model, device)

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
    parser.add_argument("--path", type=str)
    args = parser.parse_args()

    main(args)