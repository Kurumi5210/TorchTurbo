#!/usr/bin/env bash
# =============================================================================
#
# Usage:
#   bash internal/scripts/offline_tokenizer/lf_tokenize_run_mdata.sh
#   bash internal/scripts/runs/qwen3_5_moe/lf_mdata_cached_text_sft.sh
#
# =============================================================================
set -euo pipefail

[ -n "$USER_NAME" ] || { echo "Error: USER_NAME is not set" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TORCHTURBO_SRC_DIR="${TORCHTURBO_SRC_DIR:-${REPO_ROOT}/src}"

MDATA_PLUGIN="${REPO_ROOT}/internal/dataset/mdata/mdata_plugin.py"

OUTPUT_DIR=${OUTPUT_DIR:-/mnt/sfs_turbo/${USER_NAME}/datasets/cache/lf/math44w}
export HF_DATASETS_CACHE="/mnt/sfs_turbo/${USER_NAME}/datasets/cache/lf/math44w"


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

# --tt_plugin loads the mdata LF plugin (patches _get_merged_dataset), which
# self-resolves its colocated datasets.json; --dataset mdata_math is LF's
# native flag the patch intercepts.
"${PYTHON}" "${SCRIPT_DIR}/lf_tokenize.py" \
    --stage sft \
    --do_train \
    --finetuning_type full \
    --model_name_or_path /sfs_turbo/pretrained_models/Qwen3.5-35B-A3B/ \
    --tt_plugin ${MDATA_PLUGIN} \
    --dataset math-44w \
    --template qwen3_5 \
    --cutoff_len 32768 \
    --preprocessing_num_workers 96 \
    --packing true \
    --neat_packing true \
    --overwrite \
    --cache_dir ${OUTPUT_DIR} \
    --tokenized_path ${OUTPUT_DIR}
