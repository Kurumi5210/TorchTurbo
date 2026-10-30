#!/usr/bin/env bash
# Launch the Qwen3.8 alignment path with explicit data and node identity.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${MODEL:?Set the policy checkpoint directory.}"
: "${DATASET:?Set the DPO dataset file.}"
: "${OUTPUT_DIR:?Set the same fresh OUTPUT_DIR on both nodes.}"
: "${MASTER_ADDR:?Set the master node address.}"
: "${NODE_RANK:?Set NODE_RANK to 0 or 1.}"
export MODEL DATASET OUTPUT_DIR MASTER_ADDR NODE_RANK
export REF_MODEL=${REF_MODEL:-${MODEL}}
export NNODES=${NNODES:-2}
export NPROC_PER_NODE=${NPROC_PER_NODE:-16}
export SP_SIZE=${SP_SIZE:-16}
export HASH_MODEL_WEIGHTS=${HASH_MODEL_WEIGHTS:-1}

if [[ ${DRY_RUN:-0} != 1 ]]; then
    [[ -f "${MODEL}/model.safetensors.index.json" ]] || { echo "Model index is absent: ${MODEL}" >&2; exit 1; }
    [[ -f "${DATASET}" ]] || { echo "Dataset is absent: ${DATASET}" >&2; exit 1; }
fi
exec bash "${script_dir}/swift_text_dpo_debug.sh"
