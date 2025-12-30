import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoConfig, AutoModelForCausalLM
from io_hooker import InputOutputHooker, io_hook_wrapper
import gc
import sys

def cleanup():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    gc.collect()
    sys.exit(0)

@torch.no_grad()
def get_rand_batch(vocab_size, device, bs=1, seq_length=512):
    input_ids = torch.randint(0, vocab_size, (bs, seq_length), device=device)

    causal_mask = torch.triu(
        torch.full(
            (seq_length, seq_length), float('-inf'), dtype=torch.float32, device=device
        ),
        diagonal=1,
    )
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, seq_length, seq_length)
    # causal_mask = None

    position_ids = torch.arange(seq_length, device=device).long().unsqueeze(0).expand(bs, -1)

    return {
        "input_ids": input_ids,
        "labels": input_ids,
        "attention_mask": causal_mask,
        "position_ids": position_ids,
    }

def setup(rank, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    torch.cuda.set_device(rank)

    dist.barrier()

def find_attention_block(layer):
    for name in ['self_attn', 'attention', 'attn']:
        if hasattr(layer, name):
            return getattr(layer, name)
    raise AttributeError(f"Layer of type {type(layer)} does not have a self_attn/attention/attn property!")

def run(rank, args):
    torch.manual_seed(1234)
    setup(rank, args)
    device = torch.device(f"cuda:{rank}")

    config = AutoConfig.from_pretrained(args.path, trust_remote_code=True)
    config.num_hidden_layers = 20
    if torch.distributed.get_rank() == 0:
        print(config)

    model = AutoModelForCausalLM.from_pretrained(args.path, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = model.to(device)

    if torch.distributed.get_rank() == 0:
        print(model)

    hooker = InputOutputHooker(model, "/input/zhisan/dump/ds_infer_rank_0.pt", num_record_forward=1)

    # Params: io_hooker, hook_name, original_method
    # for idx, layer in enumerate(model.model.layers):
    #     orig_forward = layer.__class__.forward
    #     layer.forward = io_hook_wrapper(hooker, f'decoder_layer{idx}_forward', orig_forward).__get__(layer, layer.__class__)
    for idx, layer in enumerate(model.model.layers):
        attn = find_attention_block(layer)
        if not hasattr(attn, "lightning_attn_ops"):
            print(f"Skip layer {idx}: {type(attn)} has no lightning_attn_ops")
            continue
        for mode in attn.lightning_attn_ops:
            orig_fn = attn.lightning_attn_ops[mode]
            hooked_fn = io_hook_wrapper(hooker, f'decoder_layer{idx}_att_op_{mode}', orig_fn)
            attn.lightning_attn_ops[mode] = hooked_fn

    for _ in range(hooker.num_record_forward):
        with hooker:
            batch = get_rand_batch(config.vocab_size, device)
            hooker.model_inputs_outputs.append({"input": (batch["input_ids"],)})
            output = model(**batch)
            hooker.model_inputs_outputs.append({"logits": (output.logits,)})
            from transformers.loss.loss_utils import ForCausalLMLoss
            loss = ForCausalLMLoss(output['logits'], batch["input_ids"], config.vocab_size)
            print(f'loss {loss}')
            hooker.model_inputs_outputs.append({"loss": (loss,)})
        print(output)

    cleanup()


def main(args):
    mp.spawn(run, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/input/zhisan/ling-mini-2.5-256k/")
    args = parser.parse_args()
    args = argparse.Namespace(
        torch_turbo_enabled=False,
        world_size=8,
        path=args.path,
    )

    main(args)