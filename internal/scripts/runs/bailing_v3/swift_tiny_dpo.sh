#!/bin/bash
# Bailing V3 DPO with TorchTurbo via ms-swift.

set -o pipefail
set -eux

[ -n "$USER_NAME" ] || { echo "Error: USER_NAME is not set" >&2; exit 1; }

MODEL=${MODEL:-/mnt/sfs_turbo/pretrained_models/bailing_v3/ling_tiny_v3_formal_256k_enhance_0420_wthink/merge-step-mean-0baeb1658f223ec4bcfa8d7cf6b7e242}
SEQ_LEN=${SEQ_LEN:-65536}
OUTPUT_DIR=${OUTPUT_DIR:-/sfs_turbo/${USER_NAME}/ckpt/$(basename "${MODEL}")/swift/tiny_dpo_seq$((${SEQ_LEN}/1024))k_node${NNODES:-1}}
FINETUNING_TYPE=${FINETUNING_TYPE:-full}
DATASET=${DATASET:-/mnt/sfs_turbo/torchturbo/datasets/orca_dpo_pairs/swift_orca_rlhf.jsonl}
TT_CONFIG=${TT_CONFIG:-"$(dirname "$0")/swift_tiny_dpo.yaml"}
PYTHON=${PYTHON:-python3}

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_EVENT_TIMEOUT=14500
export ACL_DEVICE_SYNC_TIMEOUT=14400
export HCCL_OP_EXPANSION_MODE="AI_CPU"
export ASCEND_GLOBAL_LOG_LEVEL=3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_IF_BASE_PORT=29972
export GLOO_SOCKET_IFNAME=eth0
export MULTI_STREAM_MEMORY_REUSE=1
export WANDB_DISABLED=True
export TRANSFORMERS_VERBOSITY=error
export NNODES=${NNODES:-1}
export NODE_RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-23556}
export NPUS_PER_NODE=${NPUS_PER_NODE:-16}
export NPROC_PER_NODE=${NPROC_PER_NODE:-${NPUS_PER_NODE}}
export WORLD_SIZE=$((${NPUS_PER_NODE} * ${NNODES}))
export ASCEND_PROCESS_LOG_PATH=$OUTPUT_DIR/log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUN_SWIFT="${REPO_ROOT}/scripts/runs/run_swift.py"
SWIFT_SRC_DIR=${SWIFT_SRC_DIR:-}
export PYTHONPATH="${REPO_ROOT}/src${SWIFT_SRC_DIR:+:${SWIFT_SRC_DIR}}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/log-$NODE_RANK.txt"
if [ "$NODE_RANK" -eq 0 ]; then
    cp -f "$0" "$OUTPUT_DIR/"
fi
{ date "+%Y%m%d-%H%M%S"; echo "$OUTPUT_DIR"; } >> "$LOG_FILE"

"${PYTHON}" -m torch.distributed.run \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    "${RUN_SWIFT}" dpo \
    --rlhf_type dpo \
    --model_type turbo_bailing_hybrid \
    --model "${MODEL}" \
    --check_model false \
    --template ling_v3 \
    --add_non_thinking_prefix true \
    --dataset "${DATASET}" \
    --load_from_cache_file true \
    --max_length "${SEQ_LEN}" \
    --attn_impl flash_attn \
    --dataset_num_proc 96 \
    --tuner_type "${FINETUNING_TYPE}" \
    --dataset_shuffle false \
    --train_dataloader_shuffle false \
    --tt_config "${TT_CONFIG}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --beta 0.1 \
    --loss_type sigmoid \
    --learning_rate 1.25e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.0 \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 2 \
    --bf16 true \
    --dataloader_num_workers 8 \
    --report_to tensorboard \
    --output_dir "${OUTPUT_DIR}" \
    --ddp_timeout 1800000000 \
    2>&1 | tee "$LOG_FILE"
