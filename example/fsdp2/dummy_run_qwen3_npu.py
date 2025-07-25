import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
import torch.nn.functional as F

import torch_npu

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from torch_turbo.utils.fsdp2_backward_prefetch import fsdp2_backward_prefetch_manager
from torch_turbo.utils.torch_utils import compare_tensors
from dataset_utils import get_dataloader, get_dataset, data_to_device
from torch_turbo.dense.dense_utils import DenseUtils

def test_forward(args, model, dataloader, device, rank):
    def check_meta_tensors(module):
        for name, param in module.named_parameters():
            assert not param.is_meta, f"Parameter {name} is a meta tensor, which is not expected."
    check_meta_tensors(model)
    dist.barrier()
    # build reference model
    config = AutoConfig.from_pretrained(args.path)
    config.num_hidden_layers = model.config.num_hidden_layers
    ref_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=args.path, config=config)
    ref_model = ref_model.to(device, dtype=config.torch_dtype)
    model = model.to(dtype=config.torch_dtype)
    DenseUtils.save_model(model, args.save_path)
    
    # store intermediate outputs
    intermediate_outputs = {}
    intermediate_outputs_ref = {}
    
    if rank == 0:
        print(f"Reference model: {ref_model}")
        print(f"Model: {model}")
    
    for batch in dataloader:
        batch = data_to_device(batch, device)
        output = model(**batch)
        # output = output["last_hidden_state"]
        intermediate_outputs["logits"] = output["logits"]
        output_ref = ref_model(**batch)
        # output_ref = output_ref["last_hidden_state"]
        intermediate_outputs_ref["logits"] = output_ref["logits"]
        for key in intermediate_outputs.keys():
            compare_tensors(key, intermediate_outputs[key][0], intermediate_outputs_ref[key][0], rank)

def train(args, model, dataloader, device, rank):
    dist.barrier()

    if rank == 0:
        print(f"Model: {model}")

    lr = 1e-5
    betas = (0.9, 0.999)
    weight_decay = 1e-2
    
    optimizer = torch.optim.AdamW(model.parameters(),
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})

    if args.profiling:
        from torch_turbo.utils.npu_utils import get_profiler
        prof = get_profiler()
        prof.start()

    import time

    i = 0
    for batch in dataloader:
        start_time = time.time()
        optimizer.zero_grad()
        batch = data_to_device(batch, device)
        output = model(**batch)
        output = output["logits"]

        with fsdp2_backward_prefetch_manager:
            output.sum().backward()

        optimizer.step()
        if args.profiling:
            prof.step()
        cur_time = time.time()
        print(f'--->step {i}, time {cur_time - start_time}s')
        i += 1
    if args.profiling:
        prof.stop()
    print(f'train done')    

def setup(rank, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group("hccl", rank=rank, world_size=args.world_size)
    torch_npu.npu.set_device(rank)

    dist.barrier()

def cleanup():
    dist.barrier()
    dist.destroy_process_group()

def run(rank, args):
    world_size = args.world_size
    setup(rank, args)
    device = torch.device(f"npu:{rank}")

    model = DenseUtils.create_fsdp2_model(args, device)
    dist.barrier()
    batch_size_per_gpu = 1
    max_train_step = 10
    dp_size = world_size
    total_batch_size = batch_size_per_gpu * dp_size
    total_data_size = max_train_step * total_batch_size
    seq_len = 2048 * 4
    use_distributed_dataloader = True
    
    if use_distributed_dataloader:
        data_size = total_data_size
    else:
        data_size = total_data_size // dp_size
    dataset = get_dataset(seq_length=seq_len, data_size=data_size)
    dataloader = get_dataloader(dataset, batch_size_per_gpu, rank, dp_size, use_distributed_dataloader)
    
    train(args, model, dataloader, device, rank)
    cleanup()


def main(args):
    mp.spawn(run, args=(args,), nprocs=args.world_size, join=True)


# command:
#   export PYTHONPATH=/sfs_turbo/xiyun/rl/3d/opensource/Megatron-LM:/sfs_turbo/xiyun/rl/3d/opensource/MindSpeed:/sfs_turbo/xiyun/rl/new/test/tt/torchturbo:$PYTHONPATH
#   python example/fsdp2/dummy_run_qwen3_ep_npu.py  --path /sfs_turbo/pretrained_models/Qwen3-30B-A3B
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=16)
    parser.add_argument("--ep_size", type=int, default=4)
    parser.add_argument("--path", type=str, default="/sfs_turbo/pretrained_models/Qwen3-32B/")
    parser.add_argument("--save_path", type=str, default="/sfs_turbo/qinjue/models/Qwen3-32B/")
    parser.add_argument("--profiling", default=False, action="store_true")
    parser.add_argument("--average_in_collective", default=True, action="store_true")
    args = parser.parse_args()

    main(args)