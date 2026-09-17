import os
import json

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
import torch.nn.functional as F

from torch_turbo.utils.device import is_npu_available
if is_npu_available:
    import torch_npu.contrib.transfer_to_npu as _

from torchturbo.torch_turbo.distributed.parallel_state import initialize_model_parallel
from torch_turbo.moe.moe_utils import MoeUtils

from transformers import AutoConfig

from io_hooker import InputOutputHooker, io_hook_wrapper
import torch_turbo.models.deepseek.modeling_deepseek_v32 as DeepseekV32
from torch_turbo.models.deepseek.modeling_deepseek_v32 import Indexer

def setup(rank, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    torch.cuda.set_device(rank)
    initialize_model_parallel(expert_model_parallel_size=args.ep_size)

    dist.barrier()

def cleanup():
    dist.barrier()
    dist.destroy_process_group()

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
    position_ids = torch.arange(seq_length, device=device).long().unsqueeze(0).expand(bs, -1)

    return {
        "input_ids": input_ids,
        "labels": input_ids,
        "attention_mask": causal_mask,
        "position_ids": position_ids,
    }

def align_hf_rope(args, batch, hooker, device):
    from inference.model import ModelArgs, precompute_freqs_cis
    with open(args.path + "/config.json", "r") as f:
        config = json.load(f)
    model_args = ModelArgs(
        # 基本模型参数
        max_batch_size=1,
        dtype="bf16",
        max_seq_len=config["max_position_embeddings"],
        vocab_size=config["vocab_size"],
        dim=config["hidden_size"],
        inter_dim=config["intermediate_size"],
        moe_inter_dim=config["moe_intermediate_size"],
        n_layers=config["num_hidden_layers"],
        n_dense_layers=config["first_k_dense_replace"],
        n_heads=config["num_attention_heads"],
        # MoE 相关参数
        n_routed_experts=config["n_routed_experts"],
        n_shared_experts=config["n_shared_experts"],
        n_activated_experts=config["num_experts_per_tok"],
        n_expert_groups=config["n_group"],
        n_limited_groups=config["topk_group"],
        score_func=config["scoring_func"],
        route_scale=config["routed_scaling_factor"],
        # MLA (Multi-Head Latent Attention) 相关参数
        q_lora_rank=config["q_lora_rank"],
        kv_lora_rank=config["kv_lora_rank"],
        qk_nope_head_dim=config["qk_nope_head_dim"],
        qk_rope_head_dim=config["qk_rope_head_dim"],
        v_head_dim=config["v_head_dim"],
        # RoPE (Rotary Positional Embedding) 相关参数
        original_seq_len=config["rope_scaling"]["original_max_position_embeddings"],
        rope_theta=config["rope_theta"],
        rope_factor=config["rope_scaling"]["factor"],
        beta_fast=config["rope_scaling"]["beta_fast"],
        beta_slow=config["rope_scaling"]["beta_slow"],
        mscale=config["rope_scaling"]["mscale"],
        # Indexer 相关参数
        index_n_heads=config["index_n_heads"],
        index_head_dim=config["index_head_dim"],
        index_topk=config["index_topk"],
    )
    freqs_cis = precompute_freqs_cis(model_args)
    seqlen = batch["input_ids"].size(1)
    freqs_cis = freqs_cis[0:seqlen]
    cos = freqs_cis.real
    cos = torch.cat([cos, cos], dim=-1)
    sin = freqs_cis.imag
    sin = torch.cat([sin, sin], dim=-1)
    cos1 = cos.unsqueeze(0).to(device)
    sin1 = sin.unsqueeze(0).to(device)

    orig_apply_rotary_pos_emb = DeepseekV32.apply_rotary_pos_emb
    def apply_rotary_pos_emb_hook(q, cos, sin, unsqueeze_dim=1):
        assert cos.shape == cos1.shape
        input = (q, cos1, sin1)
        output = orig_apply_rotary_pos_emb(q, cos1, sin1, unsqueeze_dim)
        hooker.append(f'layer.apply_rotary_emb', (input, (output,)))
        return output
    DeepseekV32.apply_rotary_pos_emb = apply_rotary_pos_emb_hook

    orig_apply_rotary_pos_emb_interleave = DeepseekV32.apply_rotary_pos_emb_interleave
    def apply_rotary_pos_emb_interleave_hook(q, cos, sin, unsqueeze_dim=1):
        input = (q, cos1, sin1)
        output = orig_apply_rotary_pos_emb_interleave(q, cos1, sin1, unsqueeze_dim)
        hooker.append(f'layer.apply_rotary_emb', (input, (output,)))
        return output
    DeepseekV32.apply_rotary_pos_emb_interleave = apply_rotary_pos_emb_interleave_hook

def run(rank, args):
    world_size = args.world_size
    setup(rank, args)
    device = torch.device(f"cuda:{rank}")

    config = AutoConfig.from_pretrained(args.path, trust_remote_code=True)
    config.num_hidden_layers = 4
    if torch.distributed.get_rank() == 0:
        print(config)

    MoeUtils.init_mesh(args)
    model = MoeUtils.build_model(args, config)
    model = MoeUtils.fsdp_wrap(model, device, {}, config)
    MoeUtils.patch_moe_layer(model, config, args, device)
    MoeUtils.load_weights_from_hf(model, config, device, args)
    if torch.distributed.get_rank() == 0:
        print(model)

    hooker = InputOutputHooker(model, "/tmp/tt_infer_rank_0.pt")

    Indexer._compute_index_scores = io_hook_wrapper(hooker, "layer.compute_index_scores", Indexer._compute_index_scores)
    DeepseekV32.rotate_activation = io_hook_wrapper(hooker, "layer.rotate_activation", DeepseekV32.rotate_activation)

    torch.manual_seed(666)
    batch = get_rand_batch(config.vocab_size, device)

    align_hf_rope(args, batch, hooker, device)

    torch.manual_seed(888)
    with hooker:
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
    parser.add_argument("--path", type=str, default="/path/to/DeepSeek-V3.2")
    args = parser.parse_args()
    args = argparse.Namespace(
        torch_turbo_enabled=True,
        world_size=8,
        ep_size=4,
        sp_size=1,
        sp_mode="ulysses",
        average_in_collective=True,
        use_fused_rope=False,
        use_fused_rms_norm=False,
        use_fused_permute_unpermute=False,
        use_transformer_engine=False,
        path=args.path,
    )

    main(args)