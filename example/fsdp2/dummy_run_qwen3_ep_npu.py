import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
import torch.nn.functional as F

import torch_npu

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from torch_turbo.moe.megatron_utils import MegatronUtils
from torch_turbo.moe.moe_utils import MoeUtils
from torch.utils.data import DataLoader, Dataset
import functools
from torch.utils.data.distributed import DistributedSampler
from collections.abc import Mapping

def collate_sentences_lm(samples, input_name, label_name):
    if len(samples) == 0:
        return {}

    if len(samples[0]) == 0:
        return {}

    src_tokens = torch.stack([s["source"] for s in samples], 0)
    tgt_tokens = torch.stack([s["target"] for s in samples], 0)

    batch = {
        input_name: src_tokens,
        label_name: tgt_tokens,
    }
    return batch

class BenchmarkLMDataset(Dataset):
    def __init__(
        self,
        vocab_size=10000,
        max_source_positions=1024,
        total_samples=10000,
        null_dataset=False,
    ):
        self.vocab_size = vocab_size
        self.max_source_positions = max_source_positions
        self.total_samples = total_samples
        self.null_dataset = null_dataset

    def __getitem__(self, index):
        if self.null_dataset:
            return {}
        length = self.max_source_positions
        source = torch.randint(1, self.vocab_size, (length,))
        target = source.clone()
        return {
            "source": source,
            "target": target,
        }

    def __len__(self):
        return self.total_samples

def get_dataset(seq_length=128, vocab_size=32000, data_size=1000, null_dataset=False):
    return BenchmarkLMDataset(
        vocab_size=vocab_size, max_source_positions=seq_length, total_samples=data_size, null_dataset=null_dataset
    )


def get_dataloader(dataset, batch_size, rank, dp_size, use_distributed):
    dataloader_args = {"batch_size": batch_size, "drop_last": True, "shuffle": False, "num_workers": 2}
    input_name = "input_ids"
    label_name = "labels"
    dataloader_args["collate_fn"] = functools.partial(
        collate_sentences_lm, input_name=input_name, label_name=label_name
    )
    if use_distributed:
        sampler = DistributedSampler(dataset, shuffle=False, num_replicas=dp_size, rank=rank)
    else:
        sampler = None
    dataloader = DataLoader(dataset, sampler=sampler, **dataloader_args)
    return dataloader

def recursively_apply(
    func, data, *args, test_type=lambda t: isinstance(t, torch.Tensor), error_on_other_type=False, **kwargs
):
    if isinstance(data, Mapping):
        return type(data)(
            {
                k: recursively_apply(
                    func, v, *args, test_type=test_type, error_on_other_type=error_on_other_type, **kwargs
                )
                for k, v in data.items()
            }
        )
    elif isinstance(data, (tuple, list)):
        return type(data)(
            [
                recursively_apply(
                    func, v, *args, test_type=test_type, error_on_other_type=error_on_other_type, **kwargs
                )
                for v in data
            ]
        )
    elif test_type(data):
        return func(data, *args, **kwargs)
    elif error_on_other_type:
        raise TypeError(
            f"Can't apply {func.__name__} on object of type {type(data)}, only of nested list/tuple/dicts of objects "
            f"that satisfy {test_type.__name__}."
        )
    return data


def data_to_device(data, device, non_blocking=False):
    def to_device(data, device, non_blocking=False):
        if device == "pin":
            data = data.pin_memory()
        else:
            data = data.to(device, non_blocking=non_blocking)
        return data

    return recursively_apply(to_device, data, device, non_blocking=non_blocking)

def test_foward(args, model, dataloader, device, rank):
    def check_meta_tensors(module):
        for name, param in module.named_parameters():
            assert not param.is_meta, f"Parameter {name} is a meta tensor, which is not expected."
    check_meta_tensors(model)
    dist.barrier()
    # build reference model
    config = AutoConfig.from_pretrained(args.path)
    config.num_hidden_layers = 1
    ref_model = AutoModel.from_pretrained(pretrained_model_name_or_path=args.path, config=config)
    ref_model = ref_model.to(device)
    
    if rank == 0:
        print(f"Reference model: {ref_model}")
        for param_name, param in ref_model.named_parameters():
            print(f'param_name {param_name}, param.shape {param.shape}')
        print(f"Model: {model}")
        for param_name, param in model.named_parameters():
            print(f'param_name {param_name}, param.shape {param.shape}')
    
    # store attention outputs
    attention_outputs = {}

    def get_attention_hook(name):
        def hook(module, input, output):
            attention_outputs[name] = output
        return hook

    # register hook for each attn layer
    for i, layer in enumerate(model.layers):
        layer.self_attn.register_forward_hook(get_attention_hook(f"layer_{i}_attn"))
    # register hook for reference model
    for i, layer in enumerate(ref_model.layers):
        layer.self_attn.register_forward_hook(get_attention_hook(f"ref_layer_{i}_attn"))
            
    for batch in dataloader:
        batch = data_to_device(batch, device)
        output = model(**batch)
        output = output["last_hidden_state"]
        output_ref = ref_model(**batch)
        output_ref = output_ref["last_hidden_state"]
        if rank == 0:
            print(f"batch: {batch}")
            print(f"attn output: {attention_outputs['layer_0_attn']}")
            print(f"attn output ref: {attention_outputs['ref_layer_0_attn']}")
            if not torch.allclose(output.float(), output_ref.float(), atol=0.0):
                print(f"Rank {rank} - value mismatch! \n")
                print(f"Rank {rank} - Output: {output}")
                print(f"Rank {rank} - Output Ref: {output_ref}")

def setup(rank, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12346'
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
    batch_size_per_gpu = 1
    max_train_step = 1
    dp_size = world_size
    total_batch_size = batch_size_per_gpu * dp_size
    total_data_size = max_train_step * total_batch_size
    seq_len = 1
    use_distributed_dataloader = True
    
    if use_distributed_dataloader:
        data_size = total_data_size
    else:
        data_size = total_data_size // dp_size
    dataset = get_dataset(seq_length=seq_len, data_size=data_size)
    dataloader = get_dataloader(dataset, batch_size_per_gpu, rank, dp_size, use_distributed_dataloader)
    test_foward(args, model, dataloader, device, rank)

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
    parser.add_argument("--path", type=str)
    args = parser.parse_args()

    main(args)
