#!/usr/bin/env bash
# Qwen3.5-MoE TEXT SFT with torchturbo
# Usage:
#   bash scripts/runs/gpu_deps/qwen3_5_moe_deps.sh
#   bash scripts/runs/gpu_lf_qwen3_5_moe_mllm_sft.sh

set -o pipefail

set -ex

export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN


export VIDEO_PLACEHOLDER="<qwen3_5_video_placeholder>"
export AUDIO_PLACEHOLDER="<qwen3_5_audio_placeholder>"
#export IMAGE_PLACEHOLDER="<qwen3_5_image_placeholder>"


if [ -z "$USER_NAME" ]; then
    echo "Error: USER_NAME is not set"
    exit 1
fi

model_name_or_path=/pcache-mnt/ro/checkpoint/54229/345412/265265966/1/rootmodelQwen3.5-397B-A17B-BASE-hgf-asys/
#model_name_or_path=/pcache-mnt/ro/checkpoint/54229/345412/264967436/1/rootmodelQwen3.5-35B-A3B-Base-hgf-asys

seq_length=$((1024 * 32))
sp_size=4
ep_size=8
dataset=med_debug
packing_mode=packing_sep
finetuning_type=full

if [ ! -d /mnt/medmm-s3/common/data/debug_data ]; then
    mkdir -p /mnt/medmm-s3/common/data
    ln -s /input/torchturbo/stable/dataset/med /mnt/medmm-s3/common/data/debug_data
fi

export WANDB_DISABLED=True
export ALLOW_EXTRA_ARGS=True
export TRANSFORMERS_VERBOSITY=error

export NNODES=${WORLD_SIZE:-1}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export NODE_RANK=${RANK:-0}
export WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-23556}


seq_length_k=$((seq_length / 1024))k
model_basename=$(basename "$model_name_or_path")
output_dir=/input/asystem/${USER_NAME}/ckpt/${model_basename}/lf/text_sft_seq${seq_length_k}_sp${sp_size}ep${ep_size}_node${NNODES}/

mkdir -p $output_dir
if [ -d $output_dir ]; then
  echo 'Exist: ', $output_dir
else
  echo 'Not exist: ', $output_dir
fi

log_file=$output_dir/log-$NODE_RANK.txt
echo $output_dir
cp -r $0 $output_dir
cur_time=$(date "+%Y%m%d-%H%M%S")
echo $cur_time >> $log_file
echo $output_dir >> $log_file

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --node_rank $NODE_RANK
"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_LF="${SCRIPT_DIR}/run_lf.py"
DATASET_DIR="${REPO_ROOT}/dataset/gpu/"
TORCHTURBO_SRC_DIR="${TORCHTURBO_SRC_DIR:-${REPO_ROOT}/src}"

if [ ! -d "${TORCHTURBO_SRC_DIR}/torchturbo" ]; then
    echo "Error: TORCHTURBO_SRC_DIR does not point to torchturbo src: ${TORCHTURBO_SRC_DIR}" >&2
    echo "Set TORCHTURBO_SRC_DIR=/path/to/torchturbo/src or run this script from the torchturbo repo." >&2
    exit 1
fi

export PYTHONPATH=/input/torchturbo/stable/LLaMA-Factory/src/:$PYTHONPATH

torchrun $DISTRIBUTED_ARGS $RUN_LF \
    --tt_sp_size $sp_size \
    --tt_ep_size $ep_size \
    --tt_freeze_vision_tower \
    --tt_freeze_multi_modal_projector \
    --tt_ac_use_reentrant true \
    --num_train_epochs 1 \
    --report_to tensorboard \
    --logging_dir /home/admin/logs/tfevent \
    --gradient_checkpointing \
    --dataset_dir ${DATASET_DIR} \
    --trust_remote_code \
    --stage sft \
    --model_name_or_path $model_name_or_path \
    --do_train \
    --flash_attn fa2 \
    --dataset $dataset \
    --preprocessing_num_workers 64 \
    --cutoff_len ${seq_length} \
    --neat_packing \
    --template qwen3_5 \
    --finetuning_type ${finetuning_type} \
    --output_dir $output_dir \
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
    --bf16 \
    --include_num_input_tokens_seen \
    --torch_empty_cache_steps 10 \
    --ddp_timeout 1800000000 2>&1 | tee $log_file