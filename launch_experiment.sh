#!/usr/bin/env bash
# Install dependencies and run the fixed Qwen3.8 global_step_176 experiment.
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
run_dir=${repo_root}/internal/scripts/runs/qwen3_8_dense

export MODEL=/mnt/sfs_turbo/qdy/ckpt/gspo-mxc-38-0904-xiaxiao-nothink-mtp-cp-gdpo-epoch-rmv20-global_step_176
export REF_MODEL=${MODEL}
export DATASET=/mnt/sfs_turbo/cx02308786/business_user_flag_answer.json
export OUTPUT_DIR=/mnt/sfs_turbo/cx02308786/ckpt/qwen38-dpo-global176-aligned-20261029
export NNODES=2
export NPROC_PER_NODE=16
export SP_SIZE=16
export HASH_MODEL_WEIGHTS=1
export MAX_STEPS=-1
export DATA_TRACE_DIR=${OUTPUT_DIR}/data-trace
export TT_CONFIG=${run_dir}/swift_text_dpo_debug.yaml
unset RESUME_FROM_CHECKPOINT

: "${MASTER_ADDR:?Platform must set MASTER_ADDR.}"
export MASTER_ADDR
export NODE_RANK=${NODE_RANK:-${RANK:?Platform must set NODE_RANK or RANK.}}

if [[ ${DRY_RUN:-0} == 1 ]]; then
    exec bash "${run_dir}/run_npu_alignment.sh"
fi

mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${OUTPUT_DIR}/bootstrap-${NODE_RANK}.txt") 2>&1
trap 'printf "[bootstrap] exit_code=%s\n" "$?"' EXIT
printf '[bootstrap] code=%s output=%s node=%s\n' "${repo_root}" "${OUTPUT_DIR}" "${NODE_RANK}"
git -C "${repo_root}" rev-parse HEAD
if [[ ${NODE_RANK} == 0 ]]; then
    cp "$0" "${OUTPUT_DIR}/launch_experiment.sh"
fi

cd "${run_dir}"
bash install_swift_deps.sh
bash run_npu_alignment.sh
