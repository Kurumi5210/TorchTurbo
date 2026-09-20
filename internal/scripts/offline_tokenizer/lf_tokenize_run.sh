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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TORCHTURBO_SRC_DIR="${TORCHTURBO_SRC_DIR:-${REPO_ROOT}/src}"

OUTPUT_DIR=${OUTPUT_DIR:-/mnt/sfs_turbo/${USER_NAME}/datasets/cache/lf/}
export HF_DATASETS_CACHE=${OUTPUT_DIR}

if [ ! -d "${TORCHTURBO_SRC_DIR}/torchturbo" ]; then
    echo "Error: TORCHTURBO_SRC_DIR does not point to torchturbo src: ${TORCHTURBO_SRC_DIR}" >&2
    echo "Set TORCHTURBO_SRC_DIR=/path/to/torchturbo/src or run this script from the torchturbo repo." >&2
    exit 1
fi

export PYTHONPATH="${TORCHTURBO_SRC_DIR}:${PYTHONPATH:-}"
PYTHON=${PYTHON:-python3}

export VIDEO_PLACEHOLDER="<qwen3_5_video_placeholder>"
export AUDIO_PLACEHOLDER="<qwen3_5_audio_placeholder>"
export IMAGE_PLACEHOLDER="<qwen3_5_image_placeholder>"

"${PYTHON}" "${SCRIPT_DIR}/lf_tokenize.py" \
    --stage sft \
    --do_train \
    --finetuning_type full \
    --model_name_or_path /sfs_turbo/pretrained_models/Qwen3.5-35B-A3B/ \
    --dataset_dir "${REPO_ROOT}/dataset/npu" \
    --dataset science \
    --template qwen3_5_nothink \
    --cutoff_len 4096 \
    --preprocessing_num_workers 32 \
    --packing true \
    --neat_packing true \
    --overwrite \
    --tokenized_path "${REPO_ROOT}/dataset/science_lf"
