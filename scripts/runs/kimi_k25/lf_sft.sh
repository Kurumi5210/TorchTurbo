#!/usr/bin/env bash
# kimi_k25 SFT training with torchturbo
# Usage:
#   bash scripts/runs/kimi_k25/lf_sft.sh

set -o pipefail

set -ex

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
export GLOO_LOG_LEVEL=DEBUG
export MULTI_STREAM_MEMORY_REUSE=2 # 多流

#model_name_or_path=/sfs_turbo/pretrained_models/Kimi-K2.5-bf16-new/
model_name_or_path=/mnt/aicompiler-a3/common/pretrained_models/Kimi-K2.5-bf16-new/
output_dir=/sfs_turbo/${USER_NAME}/sft/ckpt/kimi_k25/lf/

seq_length=$((1024 * 4))
dataset=llava_1k_zh
packing_mode=packing_sep
finetuning_type=full

# # Deterministic computation
# export HCCL_DETERMINISTIC=True
# export LCCL_DETERMINISTIC=1
# export CLOSE_MATMUL_K_SHIFT=1
# export ATB_MATMUL_SHUFFLE_K_ENABLE=0
# export ATB_LLM_LCOC_ENABLE=0

export WANDB_DISABLED=True
export ALLOW_EXTRA_ARGS=True
export PYTHONPATH=/opt/MindSpeed:$PYTHONPATH
export TRANSFORMERS_VERBOSITY=error

export NNODES=${NNODES:-1}
export NPUS_PER_NODE=${NPUS_PER_NODE:-16}
export NODE_RANK=${RANK:-0}
export WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-23556}

export ASCEND_PROCESS_LOG_PATH=$output_dir/log

mkdir -p $output_dir
if [ -d $output_dir ]; then
  echo 'Exist: ', $output_dir
else
  echo 'Not exist: ', $output_dir
fi

log_file=$output_dir/log-$NODE_RANK.txt
echo $output_dir
if [ "$NODE_RANK" -eq 0 ]; then
    cp -f "$0" "$output_dir/"
fi
cur_time=$(date "+%Y%m%d-%H%M%S")
echo $cur_time >> $log_file
echo $output_dir >> $log_file

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --node_rank $NODE_RANK
"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUN_LF="${SCRIPT_DIR}/../run_lf.py"
DATASET_DIR="${REPO_ROOT}/dataset/npu/"
TORCHTURBO_SRC_DIR="${TORCHTURBO_SRC_DIR:-${REPO_ROOT}/src}"

if [ ! -d "${TORCHTURBO_SRC_DIR}/torchturbo" ]; then
    echo "Error: TORCHTURBO_SRC_DIR does not point to torchturbo src: ${TORCHTURBO_SRC_DIR}" >&2
    echo "Set TORCHTURBO_SRC_DIR=/path/to/torchturbo/src or run this script from the torchturbo repo." >&2
    exit 1
fi

export PYTHONPATH="${TORCHTURBO_SRC_DIR}:$PYTHONPATH"
export PYTHONPATH=/mnt/sfs_turbo/torchturbo/stable/LLaMA-Factory/src/:$PYTHONPATH

torchrun $DISTRIBUTED_ARGS $RUN_LF \
    --tt_config "${SCRIPT_DIR}/lf_sft.yaml" \
    --disable_shuffling \
    --stage sft \
    --model_name_or_path $model_name_or_path \
    --do_train \
    --val_size 0 \
    --flash_attn fa2 \
    --dataset_dir ${DATASET_DIR} \
    --dataset $dataset \
    --preprocessing_num_workers $((NNODES * 16)) \
    --cutoff_len ${seq_length} \
    --template kimi_k25 \
    --finetuning_type ${finetuning_type} \
    --output_dir $output_dir \
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
    --neat_packing \
    --max_steps 50 \
    --trust_remote_code True
