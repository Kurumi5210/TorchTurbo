#!/usr/bin/env bash
# =============================================================================
# 离线 tokenize — ms-swift 路径(示例:science SFT 数据)
#
# 用法: 
# bash scripts/offline_tokenizer/swift_tokenize_run.sh
# 不用加 --packing，swift 的 packing 由训练时的 PackingDataset 完成。

# 训练侧把 --dataset 换成 --cached_dataset <output_dir>/train
# =============================================================================
set -euo pipefail

python scripts/offline_tokenizer/swift_tokenize.py \
    --model_type turbo_qwen3_5_moe \
    --model /sfs_turbo/pretrained_models/Qwen3.5-35B-A3B/ \
    --template qwen3_5 \
    --dataset /mnt/sfs_turbo/torchturbo/datasets/debug_data/data_0630/sampled_500.jsonl \
    --max_length 4096 \
    --dataset_num_proc 32 \
    --output_dir ../dataset/science_swift \
    --exist_ok true
