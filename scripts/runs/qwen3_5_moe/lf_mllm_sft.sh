#!/usr/bin/env bash
# Qwen3.5-MoE MLLM SFT with torchturbo
# Usage:
#   bash scripts/runs/npu_deps/qwen3_5_moe_deps.sh
#   bash scripts/runs/qwen3_5_moe/lf_mllm_sft.sh

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
export GLOO_LOG_LEVEL=DEBUG
export MULTI_STREAM_MEMORY_REUSE=2 # 多流

# ==============================================================================
# TorchTurbo freeze flags
# ==============================================================================
#   --tt_freeze_vision_tower            Freeze visual encoder weights
#   --tt_freeze_multi_modal_projector   Freeze vision→text projector
#   --tt_freeze_language_model          Freeze language backbone
# ==============================================================================
# Supported models & max SP size
# ==============================================================================
#   Qwen3.5-35B-A3B        linear_num_value_heads=32      max SP=32
#   Qwen3.5-122B-A10B      linear_num_value_heads=64      max SP=64
#   Qwen3.5-397B-A17B      linear_num_value_heads=64      max SP=64
# ==============================================================================
# Test datasets
# ==============================================================================
#   SFT  MLLM      llava_1k_zh       visual instruction tuning
#   SFT  TEXT      science           text-only instruction
#   CPT  MLLM      security          multimodal continual pretraining
#   CPT  TEXT      wikipedia_en      text-only pretraining
#   DPO  TEXT      orca_rlhf         text-only preference alignment
# ==============================================================================

if [ -z "$USER_NAME" ]; then
    echo "Error: USER_NAME is not set"
    exit 1
fi

#model_name_or_path=/sfs_turbo/pretrained_models/Qwen3.5-35B-A3B
model_name_or_path=/mnt/aicompiler-a3/common/pretrained_models/Qwen3.5-35B-A3B

seq_length=$((1024 * 4))
sp_size=2
ep_size=8
dataset=llava_1k_zh
packing_mode=packing_sep
finetuning_type=full

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

model_basename=$(basename "$model_name_or_path")
seq_length_k=$((seq_length / 1024))k
output_dir=/mnt/sfs_turbo/${USER_NAME}/ckpt/${model_basename}/lf/mllm_sft_seq${seq_length_k}_sp${sp_size}ep${ep_size}_node${NNODES}/

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

export PYTHONPATH=/mnt/sfs_turbo/torchturbo/stable/LLaMA-Factory/src/:$PYTHONPATH

torchrun $DISTRIBUTED_ARGS $RUN_LF \
    --tt_config "${SCRIPT_DIR}/lf_mllm_sft.yaml" \
    --num_train_epochs 1 \
    --report_to tensorboard \
    --logging_dir /home/admin/logs/tfevent \
    --gradient_checkpointing \
    --dataset_dir ${DATASET_DIR} \
    --max_steps 100 \
    --trust_remote_code \
    --disable_shuffling \
    --stage sft \
    --model_name_or_path $model_name_or_path \
    --do_train \
    --flash_attn fa2 \
    --dataset $dataset \
    --preprocessing_num_workers 16 \
    --cutoff_len ${seq_length} \
    --neat_packing \
    --template qwen3_5_nothink \
    --finetuning_type ${finetuning_type} \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_accumulation_steps 1 \
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
    --bf16 \
    --ddp_timeout 1800000000 2>&1 | tee $log_file
