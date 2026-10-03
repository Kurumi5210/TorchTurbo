#!/bin/bash
# Bailing MoE V3 VL MLLM SFT with torchturbo
#
# Usage:
#   bash internal/scripts/runs/bailing_moe_v3_vl/install_lf_deps.sh
#   bash internal/scripts/runs/bailing_moe_v3_vl/lf_mllm_sft.sh

set -o pipefail
set -eux

[ -n "$USER_NAME" ] || { echo "Error: USER_NAME is not set" >&2; exit 1; }

# model name or path
# Bailing VL tiny weights
# MODEL=${MODEL:-/mnt/sfs_turbo/pretrained_models/tiny_v3_vl_256k_enhance_formal_select28}
MODEL=${MODEL:-/mnt/aicompiler-a3/common/ckpt/common_base_model/ling_vl_flash_midtrain_model_merge_select36}
# sequence length
SEQ_LEN=${SEQ_LEN:-131072}
# output directory
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/sfs_turbo/${USER_NAME}/ckpt/$(basename "${MODEL}")/lf/mllm_sft_seq$((${SEQ_LEN}/1024))k_node${NNODES:-1}}
# tuner type: 'full' | 'lora'
FINETUNING_TYPE=${FINETUNING_TYPE:-full}
# training dataset id resolved via --dataset_dir
DATASET=${DATASET:-med_debug}
# turbo engine flags (sp/ep, freeze, mtp, ...) live in the sibling yaml.
TT_CONFIG=${TT_CONFIG:-"$(dirname "$0")/lf_mllm_sft.yaml"}

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
export GLOO_LOG_LEVEL=DEBUG
export MULTI_STREAM_MEMORY_REUSE=2 # 多流
export WANDB_DISABLED=True
export ALLOW_EXTRA_ARGS=True
export TRANSFORMERS_VERBOSITY=error
export NNODES=${NNODES:-1}
export NPUS_PER_NODE=${NPUS_PER_NODE:-16}
export NODE_RANK=${RANK:-0}
export WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-23556}
export ASCEND_PROCESS_LOG_PATH=$OUTPUT_DIR/log
export LING_SFT_AUTO_THINKING=${LING_SFT_AUTO_THINKING:-1}

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

torchrun $DISTRIBUTED_ARGS $RUN_LF \
    --tt_config ${TT_CONFIG} \
    --report_to tensorboard \
    --disable_shuffling \
    --stage sft \
    --model_name_or_path ${MODEL} \
    --trust_remote_code \
    --do_train \
    --val_size 0 \
    --flash_attn fa2 \
    --dataset_dir ${DATASET_DIR} \
    --dataset ${DATASET} \
    --max_samples 1000 \
    --preprocessing_num_workers $((NNODES * 16)) \
    --cutoff_len ${SEQ_LEN} \
    --neat_packing \
    --template bailing_moe_v3_vl \
    --preserve_thinking true \
    --image_max_pixels 3145728 \
    --image_min_pixels 4096 \
    --finetuning_type ${FINETUNING_TYPE} \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_accumulation_steps 1 \
    --gradient_checkpointing \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --overwrite_cache \
    --save_strategy "steps" \
    --eval_steps 200000000 \
    --save_steps 200000000 \
    --warmup_ratio 0.0 \
    --learning_rate 1.25e-5 \
    --save_total_limit 1 \
    --plot_loss \
    --seed 42 \
    --ddp_timeout 1800000000 \
    --bf16 \
    --num_train_epochs 5 2>&1 | tee ${LOG_FILE}
