import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from transformers import AutoConfig, AutoModelForCausalLM

import torch_turbo.distributed.parallel_state as ps
from dataset_utils import get_dataloader, get_dataset, data_to_device
from torch_turbo.distributed.parallize_model import build_fsdp_turbo_model
from torch_turbo.distributed.fsdp.clip_grad_norm import clip_grad_norm, scale_moe_gradient
from torch_turbo.distributed.fsdp.moe_fwd_bwd_prefetch import fsdp2_backward_prefetch_manager
from torch_turbo.distributed.moe.utils import slice_tensor_for_ep_rank
from torch_turbo.utils.torch_utils import compare_tensors
from torch_turbo.utils.device import get_device_name, get_device_id, get_nccl_backend, is_npu_available


def test_forward(args, model, dataloader, device, rank):
    def check_meta_tensors(module):
        for name, param in module.named_parameters():
            assert not param.is_meta, f"Parameter {name} is a meta tensor, which is not expected."
    check_meta_tensors(model)
    dist.barrier()
    # build reference model
    config = AutoConfig.from_pretrained(args.path, trust_remote_code=True)
    config.num_hidden_layers = model.config.num_hidden_layers
    ref_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=args.path, config=config, trust_remote_code=True)
    ref_model = ref_model.to(device, dtype=config.torch_dtype)
    if hasattr(config, "layer_types"):
        ref_model.config.layer_types = ref_model.config.layer_types[:config.num_hidden_layers]
    model = model.to(dtype=config.torch_dtype)

    # store intermediate outputs
    intermediate_outputs = {}
    intermediate_outputs_ref = {}

    def get_hook(name, is_ref = False):
        def generic_hook(module, input, output):
            if is_ref == False:
                intermediate_outputs[name] = output
            else:
                intermediate_outputs_ref[name] = output

        return generic_hook

    model.model.rotary_emb.register_forward_hook(get_hook(f"rotary_embed"))
    for i, layer in enumerate(model.model.layers):
        layer.mlp.register_forward_hook(get_hook(f"layer_{i}_experts"))
    ref_model.model.rotary_emb.register_forward_hook(get_hook(f"rotary_embed", True))
    for i, layer in enumerate(ref_model.model.layers):
        layer.mlp.register_forward_hook(get_hook(f"layer_{i}_experts", True))

    if rank == 0:
        print(f"Reference model: {ref_model}")
        print(f"Model: {model}")

    for batch in dataloader:
        batch = data_to_device(batch, device)
        output = model(**batch)
        intermediate_outputs["logits"] = output["logits"]
        output_ref = ref_model(**batch)
        intermediate_outputs_ref["logits"] = output_ref["logits"]
        for key in intermediate_outputs.keys():
            compare_tensors(key, intermediate_outputs[key][0], intermediate_outputs_ref[key][0], rank)

def test_backward(args, model, dataloader, device, rank, prof=None):
    from torch_turbo.utils.torch_utils import torch_version
    if torch_version() >= (2, 6, 0):
        from torch.distributed.fsdp import fully_shard
    else:
        from torch.distributed._composable.fsdp import fully_shard
    ep_rank = ps.get_expert_model_parallel_rank()
    ep_world_size = ps.get_expert_model_parallel_world_size()
    lr = 1e-2
    betas = (0.9, 0.999)
    weight_decay = 0.0
    params = []
    moe_params = []
    for param in model.parameters():
        assert len(param.device_mesh.shape) == 1
        if param.device_mesh == ps.get_world_mesh():
            params.append(param)
        elif param.device_mesh != ps.get_world_mesh():
            moe_params.append(param)

    optimizer = torch.optim.AdamW([{"params": params}, {"params": moe_params}],
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )

    # build reference model
    config = AutoConfig.from_pretrained(args.path, trust_remote_code=True)
    config.num_hidden_layers = model.config.num_hidden_layers
    ref_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=args.path, config=config, trust_remote_code=True)
    if rank == 0:
        print(f"ref model: {ref_model}")
        print(f"fsdp model: {model}")
    ref_model = ref_model.to(device, dtype=config.torch_dtype)
    for layer in ref_model.model.layers:
        # wrap moe mlp in edp
        fully_shard(layer.mlp, mesh = ps.get_world_mesh())
        # wrap other parts in dp
        fully_shard(layer, mesh = ps.get_world_mesh())
    # wrap remaining parts
    fully_shard(ref_model, mesh = ps.get_world_mesh())
    ref_optimizer = torch.optim.AdamW(ref_model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    model = model.to(dtype=config.torch_dtype)
    criterion = nn.MSELoss()

    def get_params_and_grads(model):
        params = {}
        grads = {}
        dist.barrier()
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.full_tensor().detach().clone()
                params[name] = param.full_tensor().detach().clone()

        return params, grads

    for batch in dataloader:
        batch = data_to_device(batch, device)
        output = model(**batch)
        # generate random tensor in [-1, 1]
        min_val, max_val = -1.0, 1.0
        labels = (max_val - min_val) * torch.rand(size=output["logits"].shape, device=device, dtype=config.torch_dtype) + min_val
        loss = criterion(output["logits"], labels)
        optimizer.zero_grad()
        loss.backward()
        scale_moe_gradient(moe_params)
        total_norm = clip_grad_norm(model, max_norm=1.0)
        total_norm = total_norm.to(device)
        optimizer.step()
        params, grads = get_params_and_grads(model)

        # ref model backward
        ref_output = ref_model(**batch)
        ref_loss = criterion(ref_output["logits"], labels)
        ref_optimizer.zero_grad()
        ref_loss.backward()
        # the implementation of clip_grad_norm in verl
        from torch.nn.utils.clip_grad import _clip_grads_with_norm_, _get_total_norm
        ref_grads = [p.grad for p in ref_model.parameters() if p.grad is not None]
        ref_total_norm = _get_total_norm(ref_grads)
        ref_total_norm = ref_total_norm.to(device)
        _clip_grads_with_norm_(ref_model.parameters(), max_norm=1.0, total_norm=ref_total_norm)
        ref_optimizer.step()
        ref_params, ref_grads = get_params_and_grads(ref_model)
        compare_tensors("total_norm", total_norm.detach().clone(), ref_total_norm.full_tensor().detach().clone(), rank)
        if prof is not None and rank == 0:
            prof.step()
        if rank < ep_world_size:
            for name in params.keys():
                if "mlp.experts" in name:
                    prefix = name[:-8]
                    if "weight1" in name:
                        w1 = []
                        grad = []
                        for i in range(config.num_experts):
                            gate_proj = f"{prefix}.{i}.gate_proj.weight"
                            up_proj = f"{prefix}.{i}.up_proj.weight"
                            fc1 = torch.cat([ref_params[gate_proj], ref_params[up_proj]])
                            fc1_grad = torch.cat([ref_grads[gate_proj], ref_grads[up_proj]])
                            w1.append(fc1)
                            grad.append(fc1_grad)
                        w1 = torch.stack(w1, dim=0).to(device)
                        grad = torch.stack(grad, dim=0).to(device)
                        # [e, 2 * i, h] -> [e, h, 2 * i]
                        experts = w1.permute(0, 2, 1)
                        grad = grad.permute(0, 2, 1)
                    else:
                        # weight2
                        w2 = []
                        grad = []
                        for i in range(config.num_experts):
                            down_proj = f"{prefix}.{i}.down_proj.weight"
                            w2.append(ref_params[down_proj].t())
                            grad.append(ref_grads[down_proj].t())
                        experts = torch.stack(w2, dim=0).to(device)
                        grad = torch.stack(grad, dim=0).to(device)
                    experts = experts.reshape(-1, experts.shape[-1])
                    grad = grad.reshape(-1, grad.shape[-1])
                    experts = slice_tensor_for_ep_rank(experts, ep_rank, ep_world_size)
                    grad = slice_tensor_for_ep_rank(grad, ep_rank, ep_world_size)
                    compare_tensors(name, params[name], experts, rank)
                    compare_tensors(name + "_grad", grads[name], grad, rank)
                elif "mlp.router" in name:
                    new_name = name.replace("router", "gate")
                    compare_tensors(name, params[name], ref_params[new_name], rank)
                    compare_tensors(name + "_grad", grads[name], ref_grads[new_name], rank)
                else:
                    compare_tensors(name, params[name], ref_params[name], rank)
                    compare_tensors(name + "_grad", grads[name], ref_grads[name], rank)

    if rank == 0:
        print("-" * 30)
        print(" " * 10 + "TEST DONE")

def train(args, model, dataloader, device, rank):
    dist.barrier()

    if rank == 0:
        print(f"Model: {model}")

    lr = 1e-5
    betas = (0.9, 0.999)
    weight_decay = 1e-2
    params = []
    moe_params = []
    for param in model.parameters():
        if param.device_mesh == ps.get_world_mesh():
            params.append(param)
        elif param.device_mesh != ps.get_world_mesh():
            moe_params.append(param)

    optimizer = torch.optim.AdamW([{"params": params}, {"params": moe_params}],
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

        scale_moe_gradient(moe_params)
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
    os.environ['MASTER_PORT'] = '12346'
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group(backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}", rank=rank, world_size=args.world_size)

    if is_npu_available:
        import torch_npu
        from torch_npu.contrib import transfer_to_npu
        torch_npu.npu.set_device(rank)
    else:
        torch.cuda.set_device(rank)
    ps.initialize_model_parallel(expert_model_parallel_size=args.ep_size)

    dist.barrier()

def cleanup():
    dist.barrier()
    dist.destroy_process_group()

def run(rank, args):
    world_size = args.world_size
    setup(rank, args)
    device = get_device_id()

    model = build_fsdp_turbo_model(args, {}, device)
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

    # train(args, model, dataloader, device, rank)
    test_forward(args, model, dataloader, device, rank)
    # test_backward(args, model, dataloader, device, rank)
    cleanup()


def main(args):
    mp.spawn(run, args=(args,), nprocs=args.world_size, join=True)


# command:
#   export PYTHONPATH=/sfs_turbo/xiyun/rl/3d/opensource/MindSpeed:/sfs_turbo/xiyun/rl/new/test/tt/torchturbo:$PYTHONPATH
#   python example/fsdp2/dummy_run_qwen3_ep_npu.py  --path /sfs_turbo/pretrained_models/Qwen3-30B-A3B
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=16)
    parser.add_argument("--ep_size", type=int, default=8)
    parser.add_argument("--sp_size", type=int, default=1)
    parser.add_argument("--sp_mode", type=str, default="ulysses", choices=["zigzag_ring", "ulysses"])
    parser.add_argument("--path", type=str, default="/sfs_turbo/pretrained_models/Qwen3-30B-A3B/")
    parser.add_argument("--save_path", type=str, default="/sfs_turbo/qinjue/models/Qwen3-30B-A3B/")
    parser.add_argument("--profiling", default=False, action="store_true")
    parser.add_argument("--average_in_collective", default=True, action="store_true")
    parser.add_argument("--use_fused_swiglu", default=False, action="store_true")
    parser.add_argument("--use_fused_rope", default=False, action="store_true")
    parser.add_argument("--use_fused_rms_norm", default=False, action="store_true")
    parser.add_argument("--use_fused_permute_unpermute", default=False, action="store_true")
    parser.add_argument("--use_transformer_engine", default=False, action="store_true")

    args = parser.parse_args()

    main(args)