#!/bin/bash
# Qwen3.5-MoE (M)LLM SFT with torchturbo via ms-swift — turbo takeover model build + training.
# --- env (shared with the megatron script where backend-agnostic) -----------
#
# Usage:
#   bash internal/scripts/runs/qwen3_5_moe/install_swift_deps.sh
#   bash internal/scripts/runs/qwen3_5_moe/gpu_swift_mllm_sft.sh

set -o pipefail
set -eux

[ -n "$USER_NAME" ] || { echo "Error: USER_NAME is not set" >&2; exit 1; }

# model name or path
MODEL=${MODEL:-/pcache-mnt/ro/checkpoint/54229/345412/265265966/1/rootmodelQwen3.5-397B-A17B-BASE-hgf-asys}
# sequence length
SEQ_LEN=${SEQ_LEN:-32768}
# output directory
OUTPUT_DIR=${OUTPUT_DIR:-/sfs_turbo/${USER_NAME}/ckpt/$(basename "${MODEL}")/swift/gpu_swift_mllm_sft_node${NNODES:-1}}
# tuner type: 'full' | 'lora'
FINETUNING_TYPE=${FINETUNING_TYPE:-full                # swift args: 'full' | 'lora' (megatron uses --finetune)}
# training dataset (LF dataset id resolved via --dataset_dir, or a local path)
DATASET=${DATASET:-/mnt/medmm-s3/common/data/debug_data/data_0630/sampled_sft_data_count_151372_0630.jsonl}
# turbo engine flags (sp/ep, freeze, mtp, ...) live in the sibling yaml.
TT_CONFIG=${TT_CONFIG:-"$(dirname "$0")/gpu_swift_mllm_sft.yaml"}

dataset_num_proc=96            # swift --dataset_num_proc.
dataloader_num_workers=8       # swift args: --dataloader_num_workers
model_type=turbo_qwen3_5_moe   # turbo-registered model_type (loader = TurboModelLoader)
template=qwen3_5               # swift's built-in (M)LLM template for qwen3.5

if [ ! -d /mnt/medmm-s3/common/data/debug_data ]; then
    mkdir -p /mnt/medmm-s3/common/data
    ln -s /input/torchturbo/stable/dataset/med /mnt/medmm-s3/common/data/debug_data
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export TENSORBOARD_LOGGING_DIR=/home/admin/logs/tfevent
export MAX_PIXELS=${MAX_PIXELS:-3145728}
export VIDEO_MAX_PIXELS=${VIDEO_MAX_PIXELS:-50176}
export FPS_MAX_FRAMES=${FPS_MAX_FRAMES:-12}
export IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-3072}
export VIDEO_MAX_TOKEN_NUM=${VIDEO_MAX_TOKEN_NUM:-49}
export NNODES=${WORLD_SIZE:-${NNODES:-1}}
export NODE_RANK=${RANK:-${NODE_RANK:-0}}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-23556}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export WANDB_DISABLED=True
export TRANSFORMERS_VERBOSITY=error
export ASCEND_PROCESS_LOG_PATH=$OUTPUT_DIR/log

# python interpreter (defaults to python3)
PYTHON=${PYTHON:-python3}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUN_SWIFT="${REPO_ROOT}/scripts/runs/run_swift.py"
# ms-swift repo root (the dir containing the `swift` package); empty if pip-installed.
SWIFT_SRC_DIR=${SWIFT_SRC_DIR:-}
export PYTHONPATH="${REPO_ROOT}/src${SWIFT_SRC_DIR:+:${SWIFT_SRC_DIR}}${PYTHONPATH:+:${PYTHONPATH}}"

TORCHRUN_FLAGS=(--nproc_per_node "${NPROC_PER_NODE}"
                --nnodes "${NNODES}"
                --node_rank "${NODE_RANK}"
                --master_addr "${MASTER_ADDR}"
                --master_port "${MASTER_PORT}")

mkdir -p $OUTPUT_DIR
LOG_FILE=$OUTPUT_DIR/log-$NODE_RANK.txt
if [ "${NODE_RANK:-0}" -eq 0 ]; then
    cp -f "$0" $OUTPUT_DIR/
fi
{ date "+%Y%m%d-%H%M%S"; echo "$OUTPUT_DIR"; } >> $LOG_FILE

"${PYTHON}" -m torch.distributed.run "${TORCHRUN_FLAGS[@]}" "${RUN_SWIFT}" \
    --model_type "${model_type}" \
    --model "${MODEL}" \
    --template "${template}" \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --dataset "${DATASET}" \
    --load_from_cache_file true \
    --max_length "${SEQ_LEN}" \
    --packing true \
    --packing_length "${SEQ_LEN}" \
    --attn_impl flash_attn \
    --dataset_num_proc "${dataset_num_proc}" \
    --tuner_type "${FINETUNING_TYPE}" \
    --tt_config ${TT_CONFIG} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr":1e-7}' \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.05 \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 2 \
    --bf16 true \
    --dataloader_num_workers "${dataloader_num_workers}" \
    --report_to tensorboard \
    --output_dir "${OUTPUT_DIR}" \
    --ddp_timeout 1800000000 \
    2>&1 | tee "${LOG_FILE}"