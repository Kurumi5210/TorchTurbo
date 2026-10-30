#!/bin/bash
# Qwen3.8-Dense-27B text DPO with TorchTurbo via ms-swift.
#
# Qwen3.8-Dense uses the Qwen3.5 Dense implementation in Transformers and
# TorchTurbo (model_type=qwen3_5), so the Swift registration is
# turbo_qwen3_5 and the chat template is qwen3_5.
#
# Usage:
#   bash internal/scripts/runs/qwen3_8_dense/install_swift_deps.sh
#   bash internal/scripts/runs/qwen3_8_dense/swift_text_dpo.sh

set -o pipefail
set -eux

[ -n "$USER_NAME" ] || { echo "Error: USER_NAME is not set" >&2; exit 1; }

# The released weight path can be overridden when the platform publishes a
# revision under a different NAS directory.
MODEL=${MODEL:-/sfs_turbo/pretrained_models/Qwen3.8-27B}
REF_MODEL=${REF_MODEL:-${MODEL}}
MODEL_TYPE=${MODEL_TYPE:-turbo_qwen3_5}
TEMPLATE=${TEMPLATE:-qwen3_5}

# Qwen3.8-27B supports the 128K context window. Keep the value overridable for
# smoke tests and for jobs whose A3 allocation requires a shorter sequence.
SEQ_LEN=${SEQ_LEN:-131072}
OUTPUT_DIR=${OUTPUT_DIR:-/sfs_turbo/${USER_NAME}/ckpt/$(basename "${MODEL}")/swift/text_dpo_seq$((${SEQ_LEN}/1024))k_node${NNODES:-1}}

# DPO currently supports full-parameter training in the TorchTurbo Swift
# adapter. The variable is retained so the launcher has the same interface as
# the other run scripts.
FINETUNING_TYPE=${FINETUNING_TYPE:-full}
DATASET=${DATASET:-/mnt/sfs_turbo/torchturbo/datasets/orca_dpo_pairs/swift_orca_rlhf.jsonl}
TT_CONFIG=${TT_CONFIG:-"$(dirname "$0")/swift_text_dpo.yaml"}
PYTHON=${PYTHON:-python3}

# DPO and optimizer settings. All are environment-overridable for platform
# sweeps without editing this launcher.
BETA=${BETA:-0.1}
LOSS_TYPE=${LOSS_TYPE:-sigmoid}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:-200}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-cosine}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
LOGGING_STEPS=${LOGGING_STEPS:-1}
SAVE_STRATEGY=${SAVE_STRATEGY:-steps}
SAVE_STEPS=${SAVE_STEPS:-2000}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-2}
DATASET_NUM_PROC=${DATASET_NUM_PROC:-96}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}

source /usr/local/Ascend/ascend-toolkit/set_env.sh

export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export ACL_DEVICE_SYNC_TIMEOUT=14400
export HCCL_OP_EXPANSION_MODE="AI_CPU"
export ASCEND_GLOBAL_LOG_LEVEL=3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_IF_BASE_PORT=29972
export GLOO_SOCKET_IFNAME=eth0
export MULTI_STREAM_MEMORY_REUSE=1
export TENSORBOARD_LOGGING_DIR=/home/admin/logs/tfevent
export WANDB_DISABLED=True
export TRANSFORMERS_VERBOSITY=error

# Distributed launch environment.
export NNODES=${NNODES:-1}
export NODE_RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-23566}
export NPUS_PER_NODE=${NPUS_PER_NODE:-16}
export NPROC_PER_NODE=${NPROC_PER_NODE:-${NPUS_PER_NODE}}
export WORLD_SIZE=$((${NPUS_PER_NODE} * ${NNODES}))
export ASCEND_PROCESS_LOG_PATH=$OUTPUT_DIR/ascend_log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUN_SWIFT="${REPO_ROOT}/scripts/runs/run_swift.py"
SWIFT_SRC_DIR=${SWIFT_SRC_DIR:-}
export PYTHONPATH="${REPO_ROOT}/src${SWIFT_SRC_DIR:+:${SWIFT_SRC_DIR}}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/log-$NODE_RANK.txt"
if [ "$NODE_RANK" -eq 0 ]; then
    cp -f "$0" "$OUTPUT_DIR/"
    cp -f "$TT_CONFIG" "$OUTPUT_DIR/"
fi
{ date "+%Y%m%d-%H%M%S"; echo "$OUTPUT_DIR"; } >> "$LOG_FILE"

DISTRIBUTED_ARGS=(
    --nproc_per_node "${NPROC_PER_NODE}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

SWIFT_ARGS=(
    dpo
    --rlhf_type dpo
    --model_type "${MODEL_TYPE}"
    --model "${MODEL}"
    --ref_model "${REF_MODEL}"
    --ref_model_type "${MODEL_TYPE}"
    --template "${TEMPLATE}"
    --dataset "${DATASET}"
    --load_from_cache_file true
    --max_length "${SEQ_LEN}"
    --packing true
    --packing_length "${SEQ_LEN}"
    --attn_impl flash_attn
    --dataset_num_proc "${DATASET_NUM_PROC}"
    --tuner_type "${FINETUNING_TYPE}"
    --num_train_epochs "${NUM_TRAIN_EPOCHS}"
    --max_steps "${MAX_STEPS}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --beta "${BETA}"
    --loss_type "${LOSS_TYPE}"
    --sync_ref_model false
    --learning_rate "${LEARNING_RATE}"
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}"
    --warmup_ratio "${WARMUP_RATIO}"
    --logging_steps "${LOGGING_STEPS}"
    --save_strategy "${SAVE_STRATEGY}"
    --save_steps "${SAVE_STEPS}"
    --save_total_limit "${SAVE_TOTAL_LIMIT}"
    --bf16 true
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
    --report_to tensorboard
    --output_dir "${OUTPUT_DIR}"
    --ddp_timeout 1800000000
    --tt_config "${TT_CONFIG}"
)

if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    SWIFT_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

# Swift's DPO pipeline builds this path as an independent, eval-only reference
# model; only the policy is connected to the optimizer.
"${PYTHON}" -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" \
    "${RUN_SWIFT}" "${SWIFT_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"
