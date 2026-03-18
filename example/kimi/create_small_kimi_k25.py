#!/usr/bin/env python3
"""
Create a small Kimi K2.5 model by combining K2.5 1T's vision tower with K2 16B's language model.

This script:
1. Loads Kimi K2.5 1T model to extract vision tower and projector weights
2. Loads Kimi K2 16B model to extract language model weights
3. Creates a new K2.5 model with K2 16B's language model configuration
4. Adjusts the vision projector to match K2 16B's hidden size
5. Saves the combined model

Usage:
    export PYTHONPATH=/opt/MindSpeed:$PYTHONPATH
    python create_small_kimi_k25.py \
        --k25_dir /path/to/Kimi-K2.5-1T \
        --k2_dir /path/to/Kimi-K2-16B \
        --output_dir /path/to/output/Kimi-K2.5-Small

Example:
    python create_small_kimi_k25.py \
        --k25_dir /sfs_turbo/pretrained_models/Kimi-K2.5-bf16/ \
        --k2_dir /sfs_turbo/hw/shiyang/model/Moonlight-16B-A3B-Instruct \
        --output_dir /tmp/Kimi-K2.5-Small/
"""

import argparse
import os
import sys
import shutil
import glob
import torch
from pathlib import Path
from transformers import AutoConfig
from safetensors import safe_open

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Register custom models
from torch_turbo.models.kimi_k25.modeling_kimi_k25 import KimiK25ForConditionalGeneration
from torch_turbo.models.kimi_k25.configuration_kimi_k25 import KimiK25Config
from torch_turbo.models.kimi_k2.configuration_deepseek import DeepseekV3Config


def parse_args():
    parser = argparse.ArgumentParser(description="Create small Kimi K2.5 model")
    parser.add_argument(
        "--k25_dir",
        type=str,
        required=True,
        help="Path to Kimi K2.5 1T model directory"
    )
    parser.add_argument(
        "--k2_dir",
        type=str,
        required=True,
        help="Path to Kimi K2 16B model directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save the combined model"
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="2GB",
        help="Maximum shard size when saving the model"
    )
    parser.add_argument(
        "--tokenizer_source",
        type=str,
        choices=["k25", "k2"],
        default="k25",
        help="Which model to copy tokenizer from (default: k25)"
    )
    return parser.parse_args()


def load_k25_config(k25_dir):
    """Load K2.5 config."""
    print(f"Loading K2.5 config from {k25_dir}")
    config = AutoConfig.from_pretrained(k25_dir, trust_remote_code=True)
    return config


def load_k2_config(k2_dir):
    """Load K2 16B config for language model."""
    print(f"Loading K2 16B config from {k2_dir}")
    config = AutoConfig.from_pretrained(k2_dir, trust_remote_code=True)
    return config


def create_small_k25_config(k25_config, k2_config):
    """Create a new K2.5 config based on K2.5, only updating language model size/shape configs from K2."""
    print("Creating new K2.5 config with K2 16B language model settings...")

    # Get K2's hidden size (for projector adjustment)
    k2_hidden_size = k2_config.hidden_size
    print(f"K2 16B hidden size: {k2_hidden_size}")

    # Get K2.5 vision config
    k25_vision_config = k25_config.vision_config

    # Start from K2.5 vision config, only adjust text_hidden_size
    new_vision_config_dict = {
        'patch_size': k25_vision_config.patch_size,
        'init_pos_emb_height': k25_vision_config.init_pos_emb_height,
        'init_pos_emb_width': k25_vision_config.init_pos_emb_width,
        'init_pos_emb_time': k25_vision_config.init_pos_emb_time,
        'pos_emb_type': k25_vision_config.pos_emb_type,
        'vt_num_attention_heads': k25_vision_config.vt_num_attention_heads,
        'vt_num_hidden_layers': k25_vision_config.vt_num_hidden_layers,
        'vt_hidden_size': k25_vision_config.vt_hidden_size,
        'vt_intermediate_size': k25_vision_config.vt_intermediate_size,
        'merge_kernel_size': k25_vision_config.merge_kernel_size,
        'video_attn_type': k25_vision_config.video_attn_type,
        'merge_type': k25_vision_config.merge_type,
        'mm_projector_type': k25_vision_config.mm_projector_type,
        'mm_hidden_size': k25_vision_config.mm_hidden_size,
        'projector_hidden_act': k25_vision_config.projector_hidden_act,
        'projector_ln_eps': k25_vision_config.projector_ln_eps,
        'text_hidden_size': k2_hidden_size,  # Only change: adjust to K2's hidden size
        '_attn_implementation': getattr(k25_vision_config, '_attn_implementation', 'flash_attention_2'),
    }

    # Start from K2.5 text config, only update size/shape related configs from K2
    # These are the configs that affect tensor shapes in language model
    k25_text_config = k25_config.text_config
    size_related_keys = [
        'hidden_size',
        'intermediate_size',
        'moe_intermediate_size',
        'num_hidden_layers',
        'num_attention_heads',
        'num_key_value_heads',
        'vocab_size',
        'kv_lora_rank',
        'q_lora_rank',
        'qk_rope_head_dim',
        'v_head_dim',
        'qk_nope_head_dim',
        'n_shared_experts',
        'n_routed_experts',
        'routed_scaling_factor',
        'num_experts_per_tok',
        'num_nextn_predict_layers',
        'n_group',
        'topk_group',
    ]

    # Copy from K2.5 text config, then override size-related from K2
    new_text_config_dict = k25_text_config.to_dict()
    for key in size_related_keys:
        if hasattr(k2_config, key):
            k2_val = getattr(k2_config, key)
            old_val = getattr(k25_text_config, key, 'N/A')
            # Always set the value, including None (important for q_lora_rank)
            new_text_config_dict[key] = k2_val
            if old_val != k2_val:
                print(f"  text_config.{key}: {old_val} -> {k2_val}")

    # Also copy torch_dtype from K2 if present
    if hasattr(k2_config, 'torch_dtype'):
        new_text_config_dict['torch_dtype'] = k2_config.torch_dtype

    # Create new K2.5 config with all necessary fields
    new_config = KimiK25Config(
        text_config=new_text_config_dict,
        vision_config=new_vision_config_dict,
        ignore_index=getattr(k25_config, 'ignore_index', -100),
        media_placeholder_token_id=getattr(k25_config, 'media_placeholder_token_id', 163605),
        pad_token_id=getattr(k25_config, 'pad_token_id', 0),
        use_unified_vision_chunk=getattr(k25_config, 'use_unified_vision_chunk', True),
        video_placeholder=getattr(k25_config, 'video_placeholder', "<|kimi_k25_video_placeholder|>"),
    )

    # Set additional top-level config fields from K2.5
    for key in ['architectures', 'auto_map', 'bos_token_id', 'eos_token_id', 'tie_word_embeddings', 'dtype']:
        if hasattr(k25_config, key):
            setattr(new_config, key, getattr(k25_config, key))

    return new_config


def create_model_with_meta_device(config):
    """Create model on meta device for efficient loading."""
    print("Creating model on meta device...")
    with torch.device("meta"):
        model = KimiK25ForConditionalGeneration(config)
    return model


def adjust_projector_weights(name, param, k2_hidden_size):
    """Adjust projector output layer weights to match K2 16B hidden size.

    Returns adjusted weight if this is the output layer, otherwise returns None.
    """
    # Check if this is the final projection layer (proj.2)
    if 'mm_projector.proj.2.' not in name:
        return None

    old_weight = param
    old_out_features = old_weight.shape[0]

    if old_out_features == k2_hidden_size:
        return old_weight  # No adjustment needed

    print(f"  Adjusting {name}: {old_weight.shape} -> output size {k2_hidden_size}")

    if len(old_weight.shape) == 2:
        # Weight matrix: [out_features, in_features]
        new_weight = torch.zeros(k2_hidden_size, old_weight.shape[1], dtype=old_weight.dtype)
        if old_out_features > k2_hidden_size:
            # Reduce dimension: just take the first k2_hidden_size rows
            # Alternative: could use mean pooling, but simple truncation is more common
            new_weight = old_weight[:k2_hidden_size, :]
        else:
            # Expand dimension: copy with zero padding
            new_weight[:old_out_features, :] = old_weight
        return new_weight
    elif len(old_weight.shape) == 1:
        # Bias vector
        new_weight = torch.zeros(k2_hidden_size, dtype=old_weight.dtype)
        if old_out_features > k2_hidden_size:
            new_weight = old_weight[:k2_hidden_size]
        else:
            new_weight[:old_out_features] = old_weight
        return new_weight

    return None


def find_safetensors_files(model_dir):
    """Find all safetensors files in the model directory."""
    safetensor_files = glob.glob(os.path.join(model_dir, "*.safetensors"))
    safetensor_files.extend(glob.glob(os.path.join(model_dir, "**/*.safetensors"), recursive=True))
    # Remove duplicates and sort
    safetensor_files = sorted(set(safetensor_files))
    return safetensor_files


def load_vision_weights_from_safetensors(model_state_dict, k25_dir, k2_hidden_size):
    """Load vision tower and projector weights from K2.5 safetensors, adjusting projector output size."""
    print(f"Loading vision weights from {k25_dir}...")

    # Find all safetensor files
    safetensor_files = find_safetensors_files(k25_dir)
    print(f"  Found {len(safetensor_files)} safetensor files")

    # Get keys we need (vision tower + projector, excluding language_model)
    needed_keys = set()
    for key in model_state_dict.keys():
        if not key.startswith('language_model.'):
            needed_keys.add(key)

    copied_keys = []
    shape_mismatch_keys = []
    adjusted_keys = []

    # Load weights file by file
    for safetensor_file in safetensor_files:
        print(f"  Processing {os.path.basename(safetensor_file)}...")
        with safe_open(safetensor_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                # Skip language model weights
                if key.startswith('language_model.'):
                    continue

                # Skip if not needed
                if key not in needed_keys:
                    continue

                param = f.get_tensor(key)

                # Handle projector output layer - needs adjustment
                if 'mm_projector.proj.2.' in key:
                    adjusted_weight = adjust_projector_weights(key, param, k2_hidden_size)
                    if adjusted_weight is not None:
                        if key in model_state_dict and model_state_dict[key].shape == adjusted_weight.shape:
                            model_state_dict[key].copy_(adjusted_weight)
                            adjusted_keys.append(key)
                        elif key in model_state_dict:
                            shape_mismatch_keys.append((key, model_state_dict[key].shape, adjusted_weight.shape))
                        continue

                # Copy other weights directly if shapes match
                if key in model_state_dict:
                    if model_state_dict[key].shape == param.shape:
                        model_state_dict[key].copy_(param)
                        copied_keys.append(key)
                    else:
                        shape_mismatch_keys.append((key, str(model_state_dict[key].shape), str(param.shape)))

    print(f"  Copied {len(copied_keys)} vision weights")
    print(f"  Adjusted {len(adjusted_keys)} projector weights")
    if shape_mismatch_keys:
        print(f"  Shape mismatch keys ({len(shape_mismatch_keys)}): {shape_mismatch_keys[:5]}...")

    return model_state_dict


def load_language_weights_from_safetensors(model_state_dict, k2_dir):
    """Load language model weights from K2 16B safetensors."""
    print(f"Loading K2 16B language model from {k2_dir}...")

    # Find all safetensor files
    safetensor_files = find_safetensors_files(k2_dir)
    print(f"  Found {len(safetensor_files)} safetensor files")

    # Get K2 keys (without language_model. prefix) that we need
    needed_keys = set()
    for our_key in model_state_dict.keys():
        if our_key.startswith('language_model.'):
            k2_key = our_key[len('language_model.'):]
            needed_keys.add(k2_key)

    copied_keys = []
    shape_mismatch_keys = []
    skipped_keys = []

    # Load weights file by file
    for safetensor_file in safetensor_files:
        print(f"  Processing {os.path.basename(safetensor_file)}...")
        with safe_open(safetensor_file, framework="pt", device="cpu") as f:
            for k2_key in f.keys():
                # Build our key with language_model. prefix
                our_key = f"language_model.{k2_key}"

                if our_key not in model_state_dict:
                    skipped_keys.append(k2_key)
                    continue

                param = f.get_tensor(k2_key)

                if model_state_dict[our_key].shape == param.shape:
                    model_state_dict[our_key].copy_(param)
                    copied_keys.append(our_key)
                else:
                    shape_mismatch_keys.append((our_key, model_state_dict[our_key].shape, param.shape))

    # Check for missing keys (keys we need but not found in K2)
    found_keys = set(copied_keys)
    not_found_keys = []
    for our_key in model_state_dict.keys():
        if our_key.startswith('language_model.') and our_key not in found_keys:
            not_found_keys.append(our_key)

    print(f"  Copied {len(copied_keys)} language model weights")
    if shape_mismatch_keys:
        print(f"  Shape mismatch keys ({len(shape_mismatch_keys)}): {shape_mismatch_keys[:5]}...")
    if not_found_keys:
        print(f"  Not found keys ({len(not_found_keys)}): {not_found_keys[:5]}...")

    return model_state_dict


def copy_tokenizer_and_processor(k25_dir, output_dir):
    """Copy tokenizer, processor and other necessary files from K2.5 to output directory."""
    print(f"Copying necessary files from {k25_dir}...")

    # Required files to copy
    files_to_copy = [
        # Tokenizer related
        'tiktoken.model',
        'tokenizer_config.json',
        'tokenization_kimi.py',
        # Config related
        'generation_config.json',
        'preprocessor_config.json',
        'chat_template.jinja',
        # Model related Python files
        'modeling_kimi_k25.py',
        'modeling_deepseek.py',
        'configuration_kimi_k25.py',
        'configuration_deepseek.py',
        'kimi_k25_processor.py',
        'kimi_k25_vision_processing.py',
        'media_utils.py',
        'tool_declaration_ts.py',
        # Other necessary files
        'README.md',
        'LICENSE',
        'THIRD_PARTY_NOTICES.md',
    ]

    copied = []
    for f in files_to_copy:
        src = os.path.join(k25_dir, f)
        if os.path.exists(src):
            dst = os.path.join(output_dir, f)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(f)

    print(f"  Copied files: {copied}")


def save_model(model, state_dict, output_dir, max_shard_size, tokenizer_dir):
    """Save the combined model and copy tokenizer files."""
    print(f"Saving model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(
        output_dir,
        state_dict=state_dict,
        max_shard_size=max_shard_size,
        safe_serialization=True
    )
    print(f"Model saved to {output_dir}")

    # Copy tokenizer and processor files
    copy_tokenizer_and_processor(tokenizer_dir, output_dir)


def main():
    args = parse_args()

    print("=" * 60)
    print("Creating small Kimi K2.5 model")
    print("=" * 60)

    # Step 1: Load configs
    k25_config = load_k25_config(args.k25_dir)
    k2_config = load_k2_config(args.k2_dir)

    # Get original hidden sizes
    original_hidden_size = k25_config.text_config.hidden_size
    k2_hidden_size = k2_config.hidden_size

    print(f"\nOriginal K2.5 hidden size: {original_hidden_size}")
    print(f"K2 16B hidden size: {k2_hidden_size}")

    # Step 2: Create new config
    new_config = create_small_k25_config(k25_config, k2_config)

    # Step 3: Create model on meta device
    model = create_model_with_meta_device(new_config)
    model = model.to(dtype=torch.bfloat16)
    model.to_empty(device="cpu")

    # Get the model's state dict structure (for key checking and shape info)
    model_state_dict = model.state_dict()

    # Step 4: Load weights
    # First, load and adjust vision weights
    model_state_dict = load_vision_weights_from_safetensors(
        model_state_dict, args.k25_dir, k2_hidden_size
    )

    # Then, load language model weights
    model_state_dict = load_language_weights_from_safetensors(model_state_dict, args.k2_dir)

    # Step 5: Load state dict into model
    print("Loading state dict into model...")
    model.load_state_dict(model_state_dict, strict=True)

    # Step 6: Determine tokenizer source
    tokenizer_dir = args.k25_dir if args.tokenizer_source == "k25" else args.k2_dir
    print(f"Using tokenizer from: {tokenizer_dir}")

    # Step 7: Save model
    save_model(model, model_state_dict, args.output_dir, args.max_shard_size, tokenizer_dir)

    print("\n" + "=" * 60)
    print("Done! Small Kimi K2.5 model created successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()