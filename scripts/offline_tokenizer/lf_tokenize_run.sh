#!/usr/bin/env bash
# =============================================================================
# 离线 tokenize — LLaMA-Factory 路径
#
# 用法:
# bash scripts/offline_tokenizer/lf_tokenize_run.sh
# 重新 tokenize 已存在的产物目录时加 --overwrite。
#
# 训练侧设置 --tokenized_path 即可。
# 若数据集包含图像，请不要设置 IMAGE_PLACEHOLDER。
# =============================================================================
set -euo pipefail

export PYTHONPATH=/mnt/sfs_turbo/torchturbo/stable/LLaMA-Factory/src/:${PYTHONPATH:-}

export VIDEO_PLACEHOLDER="<qwen3_5_video_placeholder>"
export AUDIO_PLACEHOLDER="<qwen3_5_audio_placeholder>"
export IMAGE_PLACEHOLDER="<qwen3_5_image_placeholder>"

python scripts/offline_tokenizer/lf_tokenize.py \
    --stage sft \
    --do_train \
    --finetuning_type full \
    --model_name_or_path /sfs_turbo/pretrained_models/Qwen3.5-35B-A3B/ \
    --dataset_dir dataset/npu \
    --dataset science \
    --template qwen3_5_nothink \
    --cutoff_len 4096 \
    --preprocessing_num_workers 32 \
    --packing true \
    --neat_packing true \
    --overwrite \
    --tokenized_path ../dataset/science_lf
