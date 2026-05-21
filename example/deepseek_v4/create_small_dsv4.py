#!/usr/bin/env python3
"""
裁剪 DeepSeek V4 模型到前 N 层，并保存新的权重。

用法:
    python create_small_dsv4.py --input_dir /path/to/deepseek-v4 --output_dir /path/to/output --num_layers 4
"""

import argparse
import json
import os
import shutil
from collections import OrderedDict

from safetensors import safe_open
from safetensors.torch import save_file


def extract_layer_index(key):
    """从 key 中提取 layer 索引，支持两种格式:
    - model.layers.0.xxx -> 0
    - layers.0.xxx -> 0
    """
    if "model.layers." in key:
        parts = key.split(".")
        try:
            return int(parts[2])
        except (ValueError, IndexError):
            pass

    if key.startswith("layers."):
        parts = key.split(".")
        try:
            return int(parts[1])
        except (ValueError, IndexError):
            pass

    return None


def load_index_json(input_dir):
    """加载 model.safetensors.index.json，返回 weight_map 和 metadata"""
    index_path = os.path.join(input_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return None, None

    with open(index_path, "r") as f:
        index_data = json.load(f)

    return index_data.get("weight_map", {}), index_data.get("metadata", {})


def is_mtp_key(key):
    """检查是否是 mtp 层的权重"""
    return key.startswith("mtp.") or ".mtp." in key


def select_needed_files(weight_map, num_layers):
    """根据 weight_map 和所需层数，选择需要读取的文件"""
    needed_files = set()
    all_layers = set()

    for key, filename in weight_map.items():
        # 跳过 mtp 层
        if is_mtp_key(key):
            continue

        layer_idx = extract_layer_index(key)

        if layer_idx is None:
            # 非 layer 相关权重 (embed, norm, lm_head 等)，需要读取
            needed_files.add(filename)
        elif layer_idx < num_layers:
            needed_files.add(filename)
            all_layers.add(layer_idx)

    return needed_files, all_layers


def load_weights_from_files(input_dir, weight_map, needed_files):
    """只从需要的文件中加载权重"""
    all_weights = {}

    for filename in needed_files:
        filepath = os.path.join(input_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping")
            continue

        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in f.keys():
                all_weights[key] = f.get_tensor(key)

    return all_weights


def load_all_safetensors(input_dir):
    """回退方案：加载所有 safetensors 文件"""
    weight_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".safetensors")])

    all_weights = {}
    for wf in weight_files:
        filepath = os.path.join(input_dir, wf)
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in f.keys():
                all_weights[key] = f.get_tensor(key)

    return all_weights


def filter_layers(weights, num_layers):
    """只保留前 num_layers 层的权重，排除 mtp 层"""
    filtered = OrderedDict()

    for key, tensor in weights.items():
        # 跳过 mtp 层
        if is_mtp_key(key):
            continue

        layer_idx = extract_layer_index(key)

        if layer_idx is not None:
            if layer_idx < num_layers:
                filtered[key] = tensor
        else:
            filtered[key] = tensor

    return filtered


def save_in_shards(weights, output_dir, shard_size="5GB"):
    """将权重保存为 safetensors 分片文件"""
    os.makedirs(output_dir, exist_ok=True)

    if shard_size.endswith("GB"):
        max_shard_size = int(shard_size[:-2]) * (1024**3)
    elif shard_size.endswith("MB"):
        max_shard_size = int(shard_size[:-2]) * (1024**2)
    else:
        max_shard_size = 5 * (1024**3)

    current_shard = OrderedDict()
    current_size = 0
    shard_idx = 0
    shard_files = []

    for key, tensor in weights.items():
        tensor_size = tensor.numel() * tensor.element_size()

        if current_size + tensor_size > max_shard_size and current_shard:
            shard_filename = f"model-{shard_idx:05d}-of-xxxx.safetensors"
            shard_path = os.path.join(output_dir, shard_filename)
            save_file(current_shard, shard_path)
            shard_files.append(shard_filename)
            print(f"  Saved {shard_filename} ({len(current_shard)} tensors, {current_size / 1e9:.2f} GB)")

            current_shard = OrderedDict()
            current_size = 0
            shard_idx += 1

        current_shard[key] = tensor
        current_size += tensor_size

    if current_shard:
        shard_filename = f"model-{shard_idx:05d}-of-xxxx.safetensors"
        shard_path = os.path.join(output_dir, shard_filename)
        save_file(current_shard, shard_path)
        shard_files.append(shard_filename)
        print(f"  Saved {shard_filename} ({len(current_shard)} tensors, {current_size / 1e9:.2f} GB)")
        shard_idx += 1

    # 重命名更新总数
    weight_map = {}
    for i, old_name in enumerate(shard_files):
        new_name = f"model-{i:05d}-of-{shard_idx:05d}.safetensors"
        old_path = os.path.join(output_dir, old_name)
        new_path = os.path.join(output_dir, new_name)
        if old_name != new_name:
            os.rename(old_path, new_path)

        # 记录新 weight_map
        with safe_open(new_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                weight_map[key] = new_name

    return weight_map


def save_index_json(output_dir, weight_map, num_layers, total_size=0):
    """保存新的 model.safetensors.index.json"""
    index_data = {
        "metadata": {
            "format": "pt",
            "num_layers": num_layers,
            "total_size": total_size,
        },
        "weight_map": weight_map,
    }

    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=2)
    print(f"Saved model.safetensors.index.json")


def copy_config_files(input_dir, output_dir, num_layers):
    """复制并修改配置文件"""
    config_path = os.path.join(input_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

        original_layers = config.get("num_hidden_layers", None)
        config["num_hidden_layers"] = num_layers
        print(f"  Updated num_hidden_layers: {original_layers} -> {num_layers}")

        output_config_path = os.path.join(output_dir, "config.json")
        with open(output_config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  Saved config.json")

    # 复制其他文件
    for filename in [
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ]:
        src = os.path.join(input_dir, filename)
        if os.path.exists(src):
            dst = os.path.join(output_dir, filename)
            shutil.copy2(src, dst)
            print(f"  Copied {filename}")


def main():
    parser = argparse.ArgumentParser(description="裁剪 DeepSeek V4 模型到前 N 层")
    parser.add_argument(
        "--input_dir", type=str, required=True, help="输入模型权重目录 (Hugging Face 格式)"
    )
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--num_layers", type=int, required=True, help="保留的层数")
    parser.add_argument(
        "--shard_size", type=str, default="5GB", help="每个分片的大小 (默认: 5GB)"
    )

    args = parser.parse_args()

    print(f"Loading index from {args.input_dir}...")
    weight_map, _ = load_index_json(args.input_dir)

    if weight_map:
        print(f"  Found index.json with {len(weight_map)} tensors")

        # 计算需要的文件
        needed_files, all_layers = select_needed_files(weight_map, args.num_layers)
        print(f"  Total layers in model: {max(all_layers) + 1 if all_layers else 0}")
        print(f"  Target layers: 0-{args.num_layers - 1}")
        print(f"  Need to read {len(needed_files)} files out of {len(set(weight_map.values()))}")

        # 只加载需要的文件
        print(f"\nLoading weights from selected files...")
        weights = load_weights_from_files(args.input_dir, weight_map, needed_files)
    else:
        print("  No index.json found, loading all .safetensors files...")
        weights = load_all_safetensors(args.input_dir)

    print(f"  Loaded {len(weights)} tensors")

    # 统计原始层数 (排除 mtp 层)
    layer_indices = {
        extract_layer_index(k)
        for k in weights.keys()
        if not is_mtp_key(k) and extract_layer_index(k) is not None
    }
    original_num_layers = max(layer_indices) + 1 if layer_indices else 0
    print(f"  Original model has {original_num_layers} layers")

    if args.num_layers > original_num_layers:
        print(
            f"  Warning: requested {args.num_layers} layers but model only has {original_num_layers}"
        )
        args.num_layers = original_num_layers

    print(f"\nFiltering to first {args.num_layers} layers...")
    filtered_weights = filter_layers(weights, args.num_layers)
    print(f"  Kept {len(filtered_weights)} tensors")

    # 清理内存
    del weights

    print(f"\nSaving to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)

    new_weight_map = save_in_shards(filtered_weights, args.output_dir, args.shard_size)

    # 计算总大小
    total_size = sum(t.numel() * t.element_size() for t in filtered_weights.values())

    save_index_json(args.output_dir, new_weight_map, args.num_layers, total_size)

    print("\nCopying config files...")
    copy_config_files(args.input_dir, args.output_dir, args.num_layers)

    print("\n" + "=" * 50)
    print("Done!")
    print(f"  Original model: {original_num_layers} layers")
    print(f"  Pruned model: {args.num_layers} layers")
    print(f"  Output: {args.output_dir}")
    print(f"  Total size: {total_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()