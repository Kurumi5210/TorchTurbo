import os
import json
from collections import defaultdict
from typing import Dict, List, Set, Optional
import torch
from safetensors.torch import load_file
import inference.model as HFModel
from inference.model import ModelArgs, Transformer, precompute_freqs_cis
from io_hooker import InputOutputHooker, io_hook_wrapper


mapping = {
    "embed_tokens": ("embed", 0),
    "input_layernorm": ("attn_norm", None),
    "post_attention_layernorm": ("ffn_norm", None),
    "q_proj": ("wq", 0),
    "q_a_proj": ("wq_a", None),
    "q_a_layernorm": ("q_norm", None),
    "q_b_proj": ("wq_b", 0),
    "kv_a_proj_with_mqa": ("wkv_a", None),
    "kv_a_layernorm": ("kv_norm", None),
    "kv_b_proj": ("wkv_b", 0),
    "o_proj": ("wo", 1),
    "gate": ("gate", None),
    "gate_proj": ("w1", 0),
    "down_proj": ("w2", 1),
    "up_proj": ("w3", 0),
    "norm": ("norm", None),
    "lm_head": ("head", 0),
    "scale": ("scale", None),
    "wq_b": ("wq_b", None),
    "wk": ("wk", None),
    "k_norm": ("k_norm", None),
    "weights_proj": ("weights_proj", None),
}

def convert_param_names(maps):
    new_map = {}
    for name in maps.keys():
        if "model.layers.61" in name:
            continue
        value = maps[name]
        if name.startswith("model."):
            name = name[len("model."):]
        name = name.replace("self_attn", "attn")
        name = name.replace("mlp", "ffn")
        name = name.replace("weight_scale_inv", "scale")
        name = name.replace("e_score_correction_bias", "bias")
        key = name.split(".")[-2]
        assert key in mapping, f"Key {key} not found in mapping"
        new_key, _ = mapping[key]
        name = name.replace(key, new_key)
        new_map[name] = value
    return new_map

def load_state_dict(model, path: str) -> torch.nn.Module:
    model_state_dict = model.state_dict()
    model_param_names = set(model_state_dict.keys())

    index_path = os.path.join(path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"File not exists: {index_path}")

    with open(index_path, 'r', encoding='utf-8') as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})
    weight_map = convert_param_names(weight_map)

    file_to_params = defaultdict(list)
    missing_params = []

    for param_name in model_param_names:
        if param_name in weight_map:
            file_name = weight_map[param_name]
            file_to_params[file_name].append(param_name)
        else:
            missing_params.append(param_name)

    if missing_params:
        print(f"  Warning: {len(missing_params)} params not found")
        for p in missing_params:
            print(f"    - {p}")

    loaded_state_dict = {}

    for i, (file_name, param_names) in enumerate(sorted(file_to_params.items()), 1):
        file_path = os.path.join(path, file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Safetensor not exits: {file_path}")

        file_state_dict = convert_param_names(load_file(file_path))

        for param_name in param_names:
            if param_name in file_state_dict:
                loaded_state_dict[param_name] = file_state_dict[param_name]
            else:
                print(f"    warning: param {param_name} not found in {file_name}")

    shape_mismatches = []
    for param_name, loaded_tensor in list(loaded_state_dict.items()):
        if param_name in model_state_dict:
            model_shape = model_state_dict[param_name].shape
            loaded_shape = loaded_tensor.shape
            if model_shape != loaded_shape:
                shape_mismatches.append((param_name, model_shape, loaded_shape))
                del loaded_state_dict[param_name]

    if shape_mismatches:
        print(f"  Warning: {len(shape_mismatches)} param are skipped due to shape mismatch:")
        for name, model_shape, loaded_shape in shape_mismatches:
            print(f"    - {name}: model expect {model_shape}, but got {loaded_shape}")
        return model

    model.load_state_dict(loaded_state_dict, strict=False)

    # 打印总结
    loaded_count = len(loaded_state_dict)
    total_count = len(model_param_names)
    print(f"  Success loaded {loaded_count}/{total_count} params")

    return model

def main(path):
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)

    with open(path + "/config.json", "r") as f:
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

    model_args.n_layers = 4
    model = Transformer(model_args)
    model = model.to(dtype=torch.bfloat16)
    model.freqs_cis = precompute_freqs_cis(model_args)
    model = load_state_dict(model, path)
    print(model)

    hooker = InputOutputHooker(model, "/tmp/hf_infer_rank_0.pt")

    HFModel.bf16_index = io_hook_wrapper(hooker, "layer.compute_index_scores", HFModel.bf16_index)
    HFModel.rotate_activation = io_hook_wrapper(hooker, "layer.rotate_activation", HFModel.rotate_activation)
    HFModel.apply_rotary_emb = io_hook_wrapper(hooker, "layer.apply_rotary_emb", HFModel.apply_rotary_emb)

    torch.manual_seed(666)
    x = torch.randint(0, model_args.vocab_size, (1, 512))

    print("Running inference test...")
    torch.manual_seed(888)
    with hooker:
        hooker.model_inputs_outputs.append({"input": (x,)})
        logits = model(x)
        hooker.model_inputs_outputs.append({"logits": (logits,)})
        from transformers.loss.loss_utils import ForCausalLMLoss
        loss = ForCausalLMLoss(logits, x, config['vocab_size'])
        print(f'loss {loss}')
        hooker.model_inputs_outputs.append({"loss": (loss,)})

    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Logits data type: {logits.dtype}")
    print("\nInference test completed successfully!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/mnt/sfs_turbo/pretrained_models/DeepSeek-V3.2-Speciale-BF16/")
    args = parser.parse_args()

    main(args.path)
