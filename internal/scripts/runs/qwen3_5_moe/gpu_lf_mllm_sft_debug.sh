#!/bin/bash
# Qwen3.5-MoE TEXT SFT with torchturbo
#   bash scripts/runs/qwen3_5_moe/install_lf_deps.sh
#   bash scripts/runs/qwen3_5_moe/gpu_lf_mllm_sft_debug.sh
#
# Usage:
#   bash internal/scripts/runs/qwen3_5_moe/install_lf_deps.sh
#   bash internal/scripts/runs/qwen3_5_moe/gpu_lf_mllm_sft_debug.sh

set -o pipefail
set -eux

[ -n "$USER_NAME" ] || { echo "Error: USER_NAME is not set" >&2; exit 1; }

# model name or path
#MODEL=${MODEL:-/pcache-mnt/ro/checkpoint/54229/345412/265265966/1/rootmodelQwen3.5-397B-A17B-BASE-hgf-asys}
MODEL=${MODEL:-/pcache-mnt/ro/checkpoint/54229/345412/264967436/1/rootmodelQwen3.5-35B-A3B-Base-hgf-asys}
# sequence length
SEQ_LEN=${SEQ_LEN:-32768}
# output directory
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/sfs_turbo/${USER_NAME}/ckpt/$(basename "${MODEL}")/lf/mllm_sft_debug_seq$((${SEQ_LEN}/1024))k_node${NNODES:-1}}
# tuner type: 'full' | 'lora'
FINETUNING_TYPE=${FINETUNING_TYPE:-full}
# training dataset (LF dataset id resolved via --dataset_dir, or a local path)
DATASET=${DATASET:-med_debug}
# turbo engine flags (sp/ep, freeze, mtp, ...) live in the sibling yaml.
TT_CONFIG=${TT_CONFIG:-"$(dirname "$0")/gpu_lf_mllm_sft_debug.yaml"}

packing_mode=packing_sep

if [ ! -d /mnt/medmm-s3/common/data/debug_data ]; then
    mkdir -p /mnt/medmm-s3/common/data
    ln -s /input/torchturbo/stable/dataset/med /mnt/medmm-s3/common/data/debug_data
fi

export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export VIDEO_PLACEHOLDER="<qwen3_5_video_placeholder>"
export AUDIO_PLACEHOLDER="<qwen3_5_audio_placeholder>"
export WANDB_DISABLED=True
export ALLOW_EXTRA_ARGS=True
export TRANSFORMERS_VERBOSITY=error
export NNODES=${WORLD_SIZE:-1}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export NODE_RANK=${RANK:-0}
export WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
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
    --nproc_per_node ${NPROC_PER_NODE:-${GPUS_PER_NODE:-8}} \
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
    --num_train_epochs 1 \
    --report_to tensorboard \
    --logging_dir /home/admin/logs/tfevent \
    --gradient_checkpointing \
    --dataset_dir ${DATASET_DIR} \
    --max_steps 1000 \
    --max_samples 1000 \
    --trust_remote_code \
    --disable_shuffling \
    --stage sft \
    --model_name_or_path ${MODEL} \
    --do_train \
    --flash_attn fa2 \
    --dataset ${DATASET} \
    --preprocessing_num_workers 64 \
    --cutoff_len ${SEQ_LEN} \
    --neat_packing \
    --template qwen3_5 \
    --finetuning_type ${FINETUNING_TYPE} \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_accumulation_steps 1 \
    --lr_scheduler_type cosine_warmup_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr":1e-7}' \
    --logging_steps 1 \
    --overwrite_cache \
    --save_strategy "steps" \
    --eval_steps 200000000 \
    --save_steps 200000000 \
    --warmup_ratio 0.05 \
    --learning_rate 1e-5 \
    --save_total_limit 1 \
    --plot_loss \
    --seed 42 \
    --bf16 \
    --include_num_input_tokens_seen \
    --ddp_timeout 1800000000 2>&1 | tee ${LOG_FILE}