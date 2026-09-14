#!/bin/bash
# LLaDA 2.1 TEXT SFT with torchturbo
#
# Usage:
#   bash internal/scripts/runs/llada2/install_lf_deps.sh
#   bash internal/scripts/runs/llada2/lf_text_sft.sh

set -o pipefail
set -eux

[ -n "$USER_NAME" ] || { echo "Error: USER_NAME is not set" >&2; exit 1; }

# model name or path
MODEL=${MODEL:-/mnt/sfs_turbo/models/LLaDA2.1-mini}
# sequence length
SEQ_LEN=${SEQ_LEN:-4096}
# output directory
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/sfs_turbo/${USER_NAME}/ckpt/$(basename "${MODEL}")/lf/text_sft_seq$((${SEQ_LEN}/1024))k_node${NNODES:-1}}
# tuner type: 'full' | 'lora'
FINETUNING_TYPE=${FINETUNING_TYPE:-full}
# training dataset (LF dataset id resolved via --dataset_dir, or a local path)
DATASET=${DATASET:-gsm8k_bd}
# turbo engine flags (sp/ep, freeze, mtp, ...) live in the sibling yaml.
TT_CONFIG=${TT_CONFIG:-"$(dirname "$0")/lf_text_sft.yaml"}



source /usr/local/Ascend/ascend-toolkit/set_env.sh
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_EVENT_TIMEOUT=14500
export ACL_DEVICE_SYNC_TIMEOUT=14400
export ASCEND_GLOBAL_LOG_LEVEL=3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_IF_BASE_PORT=29972
export GLOO_SOCKET_IFNAME=eth0
export MULTI_STREAM_MEMORY_REUSE=2
export WANDB_DISABLED=True
export ALLOW_EXTRA_ARGS=True
export TRANSFORMERS_VERBOSITY=error
export SWANLAB_MODE=disabled
export NNODES=${NNODES:-1}
export NPUS_PER_NODE=${NPUS_PER_NODE:-16}
export NODE_RANK=${RANK:-0}
export WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-23556}
export ASCEND_PROCESS_LOG_PATH=$OUTPUT_DIR/log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUN_LF="${REPO_ROOT}/scripts/runs/run_lf.py"
DATASET_DIR="${REPO_ROOT}/internal/dataset/npu/"
TORCHTURBO_SRC_DIR="${TORCHTURBO_SRC_DIR:-${REPO_ROOT}/src}"
[ -d "${TORCHTURBO_SRC_DIR}/torchturbo" ] || { echo "bad TORCHTURBO_SRC_DIR: ${TORCHTURBO_SRC_DIR}" >&2; exit 1; }
export PYTHONPATH="${TORCHTURBO_SRC_DIR}:/opt/MindSpeed${PYTHONPATH:+:${PYTHONPATH}}"

DISTRIBUTED_ARGS="
    --nproc_per_node ${NPROC_PER_NODE:-${NPUS_PER_NODE:-16}} \
    --nnodes ${NNODES:-1} \
    --node_rank ${NODE_RANK:-0} \
    --master_addr=${MASTER_ADDR:-127.0.0.1} \
    --master_port=${MASTER_PORT:-23556}
"

mkdir -p $OUTPUT_DIR
LOG_FILE=$OUTPUT_DIR/log-$NODE_RANK.txt
if [ "${NODE_RANK:-0}" -eq 0 ]; then
    cp -f "$0" $OUTPUT_DIR/
fi
{ date "+%Y%m%d-%H%M%S"; echo "$OUTPUT_DIR"; } >> $LOG_FILE

torchrun $DISTRIBUTED_ARGS $RUN_LF \ \
    --tt_config ${TT_CONFIG} \
    --stage sft \
    --model_name_or_path ${MODEL} \
    --trust_remote_code \
    --disable_shuffling \
    --do_train \
    --flash_attn sdpa \
    --dataset ${DATASET} \
    --dataset_dir ${DATASET_DIR} \
    --template default \
    --finetuning_type full \
    --cutoff_len ${SEQ_LEN} \
    --neat_packing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --max_steps 50 \
    --logging_steps 1 \
    --save_steps 200000000 \
    --eval_steps 200000000 \
    --bf16 \
    --seed 42 \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_output_dir \
    --ddp_timeout 1800000000 \
    --overwrite_cache 2>&1 | tee ${LOG_FILE}
