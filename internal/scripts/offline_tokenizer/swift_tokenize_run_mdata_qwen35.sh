#!/usr/bin/env bash
#
# Usage:
#   step 1: bash internal/scripts/offline_tokenizer/swift_tokenize_run_mdata_qwen35.sh
#   step 2: bash internal/scripts/runs/qwen3_5_moe/swift_mllm_mdata_cached_sft_debug.sh

set -euo pipefail

[ -n "$USER_NAME" ] || { echo "Error: USER_NAME is not set" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TORCHTURBO_SRC_DIR="${TORCHTURBO_SRC_DIR:-${REPO_ROOT}/src}"

MDATA_PLUGIN="${REPO_ROOT}/internal/dataset/mdata/mdata_plugin.py"

OUTPUT_DIR=/mnt/sfs_turbo/${USER_NAME}/datasets/cache/swift/math44w
MODELSCOPE_CACHE=${OUTPUT_DIR}
export HF_DATASETS_CACHE=/mnt/sfs_turbo/${USER_NAME}/datasets/cache/swift/math44w


if [ ! -d "${TORCHTURBO_SRC_DIR}/torchturbo" ]; then
    echo "Error: TORCHTURBO_SRC_DIR does not point to torchturbo src: ${TORCHTURBO_SRC_DIR}" >&2
    echo "Set TORCHTURBO_SRC_DIR=/path/to/torchturbo/src or run this script from the torchturbo repo." >&2
    exit 1
fi

export PYTHONPATH="${TORCHTURBO_SRC_DIR}:${PYTHONPATH:-}"
PYTHON=${PYTHON:-python3}

# --tt_plugin loads the mdata swift plugin (registers the ``mdata`` dataset),
# which self-resolves its colocated datasets.json; --dataset mdata_math is
# swift's native dataset flag the plugin registered.
"${PYTHON}" "${SCRIPT_DIR}/swift_tokenize.py" \
    --tt_plugin ${MDATA_PLUGIN} \
    --dataset math-44w \
    --model_type turbo_qwen3_5_moe \
    --model /sfs_turbo/pretrained_models/Qwen3.5-35B-A3B \
    --dataset_shuffle false \
    --check_model false \
    --template qwen3_5 \
    --max_length 32768 \
    --dataset_num_proc 96 \
    --output_dir ${OUTPUT_DIR} \
    --exist_ok true