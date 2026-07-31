#!/usr/bin/env bash
# LLaDA 2.1 TEXT SFT with torchturbo
# Usage:
#   bash scripts/runs/npu_deps/qwen3_5_moe_deps.sh
#   bash scripts/runs/lf_llada2_text_sft.sh

set -o pipefail
set -ex

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export ASCEND_GLOBAL_LOG_LEVEL=3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_IF_BASE_PORT=29972
export GLOO_SOCKET_IFNAME=eth0
export MULTI_STREAM_MEMORY_REUSE=2

if [ -z "$USER_NAME" ]; then
    echo "Error: USER_NAME is not set"
    exit 1
fi

model_name_or_path=/mnt/sfs_turbo/models/LLaDA2.1-mini

# seq_length = cutoff_len = per-sample real truncation limit.
# BD beam length = 2 * cutoff_len (noisy + clean).

# torchturbo --tt_* flags:
#   --tt_bd_mask_low/--tt_bd_mask_high : BD per-sample noise ratio sigma range U[low,high], default 0.3/0.8.
#   --tt_cap_lambda                    : CAP confidence-loss weight; >0 enables (e.g. 2.0), default 0 = off.
#   --tt_bd_block_size                 : BD block-diagonal attention block size; cutoff_len ceil-aligns to it, default 32.

seq_length=$((1024 * 4))
sp_size=2
ep_size=8
dataset=gsm8k_bd

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

export ASCEND_PROCESS_LOG_PATH=$output_dir/log

model_basename=$(basename "$model_name_or_path")
seq_length_k=$((seq_length / 1024))k
output_dir=/mnt/sfs_turbo/${USER_NAME}/ckpt/${model_basename}/lf/text_sft_seq${seq_length_k}_sp${sp_size}ep${ep_size}_node${NNODES}/

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
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --node_rank $NODE_RANK
"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_LF="${SCRIPT_DIR}/run_lf.py"
DATASET_DIR="${REPO_ROOT}/dataset/npu/"
TORCHTURBO_SRC_DIR="${TORCHTURBO_SRC_DIR:-${REPO_ROOT}/src}"

if [ ! -d "${TORCHTURBO_SRC_DIR}/torchturbo" ]; then
    echo "Error: TORCHTURBO_SRC_DIR does not point to torchturbo src: ${TORCHTURBO_SRC_DIR}" >&2
    echo "Set TORCHTURBO_SRC_DIR=/path/to/torchturbo/src or run this script from the torchturbo repo." >&2
    exit 1
fi

export PYTHONPATH=/mnt/sfs_turbo/torchturbo/stable/LLaMA-Factory/src/:$PYTHONPATH

torchrun $DISTRIBUTED_ARGS $RUN_LF \
    --tt_sp_size $sp_size \
    --tt_ep_size $ep_size \
    --tt_bd_mask_low 0.3 \
    --tt_bd_mask_high 0.8 \
    --tt_bd_block_size 32 \
    --stage sft \
    --model_name_or_path $model_name_or_path \
    --trust_remote_code \
    --disable_shuffling \
    --do_train \
    --flash_attn sdpa \
    --dataset $dataset \
    --dataset_dir ${DATASET_DIR} \
    --template default \
    --finetuning_type full \
    --cutoff_len ${seq_length} \
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
    --output_dir $output_dir \
    --overwrite_output_dir \
    --ddp_timeout 1800000000 \
    --overwrite_cache 2>&1 | tee -a $log_file
