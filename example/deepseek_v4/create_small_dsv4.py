#!/usr/bin/env python3
"""Prune a DeepSeek V4 model to its first N layers and save new weights.

Usage:
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
    """Extract a layer index from a weight-map key. Supported formats:
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
    """Load the safetensors index and return its weight map and metadata."""
    index_path = os.path.join(input_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return None, None

    with open(index_path, "r") as f:
        index_data = json.load(f)

    return index_data.get("weight_map", {}), index_data.get("metadata", {})


def is_mtp_key(key):
    """Return whether a key belongs to an MTP layer."""
    return key.startswith("mtp.") or ".mtp." in key


def count_mtp_stages(weight_map):
    """Count MTP stages represented by a checkpoint weight map."""
    stages = set()
    for key in weight_map:
        if is_mtp_key(key):
            stages.add(int(key.split("mtp.")[1].split(".", 1)[0]))
    return max(stages) + 1 if stages else 0


def select_needed_files(weight_map, num_layers, keep_mtp):
    """Select checkpoint files needed for the requested layers."""
    needed_files = set()
    all_layers = set()

    for key, filename in weight_map.items():
        if is_mtp_key(key):
            if keep_mtp:
                needed_files.add(filename)
            continue

        layer_idx = extract_layer_index(key)

        if layer_idx is None:
            # Read non-layer weights such as embeddings, norms, and the head.
            needed_files.add(filename)
        elif layer_idx < num_layers:
            needed_files.add(filename)
            all_layers.add(layer_idx)

    return needed_files, all_layers


def load_weights_from_files(input_dir, weight_map, needed_files):
    """Load weights only from the selected checkpoint files."""
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
    """Fallback path: load every safetensors file."""
    weight_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".safetensors")])

    all_weights = {}
    for wf in weight_files:
        filepath = os.path.join(input_dir, wf)
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in f.keys():
                all_weights[key] = f.get_tensor(key)

    return all_weights


def filter_layers(weights, num_layers, keep_mtp):
    """Keep the first num_layers layers and optionally keep MTP weights."""
    filtered = OrderedDict()

    for key, tensor in weights.items():
        if is_mtp_key(key):
            if keep_mtp:
                filtered[key] = tensor
            continue

        layer_idx = extract_layer_index(key)

        if layer_idx is not None:
            if layer_idx < num_layers:
                filtered[key] = tensor
        else:
            filtered[key] = tensor

    return filtered


def save_in_shards(weights, output_dir, shard_size="5GB"):
    """Save weights as sharded safetensors files."""
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

    # Rename shards after the final shard count is known.
    weight_map = {}
    for i, old_name in enumerate(shard_files):
        new_name = f"model-{i:05d}-of-{shard_idx:05d}.safetensors"
        old_path = os.path.join(output_dir, old_name)
        new_path = os.path.join(output_dir, new_name)
        if old_name != new_name:
            os.rename(old_path, new_path)

        # Record the new weight map.
        with safe_open(new_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                weight_map[key] = new_name

    return weight_map


def save_index_json(output_dir, weight_map, num_layers, total_size=0):
    """Save a new model.safetensors.index.json file."""
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


def copy_config_files(input_dir, output_dir, num_layers, keep_mtp, dspark, mtp_stages):
    """Copy and update model configuration files."""
    config_path = os.path.join(input_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

        original_layers = config.get("num_hidden_layers", None)
        config["num_hidden_layers"] = num_layers
        print(f"  Updated num_hidden_layers: {original_layers} -> {num_layers}")

        if keep_mtp:
            config["num_nextn_predict_layers"] = mtp_stages
            if config.get("compress_ratios"):
                tail = (
                    [0] * mtp_stages
                    if dspark
                    else [config["compress_ratios"][-1]] * mtp_stages
                )
                config["compress_ratios"] = config["compress_ratios"][:num_layers] + tail
            if dspark:
                # main_proj consumes one hidden state per target layer, so keep
                # the target count and move the targets into the kept layers.
                num_target_layers = len(config["dspark_target_layer_ids"])
                config["dspark_target_layer_ids"] = list(
                    range(num_layers - num_target_layers, num_layers)
                )
                print(
                    "  Updated DSpark target layers: "
                    f"{config['dspark_target_layer_ids']}"
                )
            print(f"  Updated MTP layers: {config['num_nextn_predict_layers']}")

        output_config_path = os.path.join(output_dir, "config.json")
        with open(output_config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  Saved config.json")

    # Copy auxiliary tokenizer and generation files.
    for filename in [
        "tokenizer_config.json",
        "tokenizer.json",
        "generation_config.json",
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
    parser = argparse.ArgumentParser(description="Prune DeepSeek V4 to its first N layers")
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Input Hugging Face checkpoint directory"
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_layers", type=int, required=True, help="Number of layers to keep")
    parser.add_argument(
        "--shard_size", type=str, default="5GB", help="Maximum shard size (default: 5GB)"
    )
    parser.add_argument("--keep_mtp", action="store_true", help="Keep mtp.* weights")
    parser.add_argument(
        "--dspark",
        action="store_true",
        help="Input is a DSpark checkpoint; remap its target layers into the kept layers",
    )

    args = parser.parse_args()
    if args.dspark and not args.keep_mtp:
        parser.error("--dspark requires --keep_mtp")

    print(f"Loading index from {args.input_dir}...")
    weight_map, _ = load_index_json(args.input_dir)
    mtp_stages = count_mtp_stages(weight_map) if weight_map else 0
    if args.dspark and mtp_stages == 0:
        parser.error("no mtp.* stages found in the checkpoint index")
    if args.dspark:
        with open(os.path.join(args.input_dir, "config.json"), "r") as f:
            num_target_layers = len(json.load(f)["dspark_target_layer_ids"])
        if args.num_layers < num_target_layers:
            parser.error(
                f"--num_layers must be at least {num_target_layers} to keep "
                "every DSpark target layer"
            )

    if weight_map:
        print(f"  Found index.json with {len(weight_map)} tensors")

        # Determine which shard files must be read.
        needed_files, all_layers = select_needed_files(weight_map, args.num_layers, args.keep_mtp)
        print(f"  Total layers in model: {max(all_layers) + 1 if all_layers else 0}")
        print(f"  Target layers: 0-{args.num_layers - 1}")
        print(f"  Need to read {len(needed_files)} files out of {len(set(weight_map.values()))}")

        # Load only the required shard files.
        print(f"\nLoading weights from selected files...")
        weights = load_weights_from_files(args.input_dir, weight_map, needed_files)
    else:
        print("  No index.json found, loading all .safetensors files...")
        weights = load_all_safetensors(args.input_dir)

    print(f"  Loaded {len(weights)} tensors")

    # Count original backbone layers, excluding MTP.
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
    filtered_weights = filter_layers(weights, args.num_layers, args.keep_mtp)
    print(f"  Kept {len(filtered_weights)} tensors")

    # Release the unfiltered tensors.
    del weights

    print(f"\nSaving to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)

    new_weight_map = save_in_shards(filtered_weights, args.output_dir, args.shard_size)

    # Compute the final checkpoint size.
    total_size = sum(t.numel() * t.element_size() for t in filtered_weights.values())

    save_index_json(args.output_dir, new_weight_map, args.num_layers, total_size)

    print("\nCopying config files...")
    copy_config_files(
        args.input_dir,
        args.output_dir,
        args.num_layers,
        args.keep_mtp,
        args.dspark,
        mtp_stages,
    )

    print("\n" + "=" * 50)
    print("Done!")
    print(f"  Original model: {original_num_layers} layers")
    print(f"  Pruned model: {args.num_layers} layers")
    print(f"  Output: {args.output_dir}")
    print(f"  Total size: {total_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
