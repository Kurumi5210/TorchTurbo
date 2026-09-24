#!/usr/bin/env bash
#
# Usage:
#   step 1: bash internal/scripts/offline_tokenizer/swift_tokenize_run_mdata_ling3.sh
#   step 2: bash internal/scripts/runs/bailing_v3/swift_tiny_mdata_cached_sft.sh

set -euo pipefail

[ -n "$USER_NAME" ] || { echo "Error: USER_NAME is not set" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TORCHTURBO_SRC_DIR="${TORCHTURBO_SRC_DIR:-${REPO_ROOT}/src}"

MDATA_PLUGIN="${REPO_ROOT}/internal/dataset/mdata_demo/mdata_plugin.py"

OUTPUT_DIR=/mnt/sfs_turbo/${USER_NAME}/datasets/cache/swift/math44w
MODELSCOPE_CACHE=${OUTPUT_DIR}
export HF_DATASETS_CACHE=${OUTPUT_DIR}
rm -rf ${OUTPUT_DIR}


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
    --model_type turbo_bailing_hybrid \
    --model /mnt/sfs_turbo/pretrained_models/bailing_v3/ling_tiny_v3_formal_256k_enhance_0420_wthink/merge-step-mean-0baeb1658f223ec4bcfa8d7cf6b7e242 \
    --check_model false \
    --template ling_v3_eos \
    --max_length 131072 \
    --dataset_num_proc 96 \
    --output_dir ${OUTPUT_DIR} \
    --exist_ok true