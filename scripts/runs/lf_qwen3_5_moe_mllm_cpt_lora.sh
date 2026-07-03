#!/usr/bin/env bash
# Qwen3.5-MoE LoRA MLLM CPT with torchturbo
# Usage:
#   bash scripts/runs/npu_deps/qwen3_5_moe_deps.sh
#   bash scripts/runs/lf_qwen3_5_moe_mllm_cpt_lora.sh

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
# LoRA hyperparameters
# ==============================================================================
#   --finetuning_type lora   Enable LoRA instead of full fine-tuning
#   --lora_rank N            Low-rank dimension (bottleneck size)
#   --lora_alpha N           Scaling factor: effective lr ∝ alpha / rank
#   --lora_target all        Apply LoRA to all attention linear layers
#   --lora_dropout N         Dropout on LoRA adapters
#   --use_rslora             Use rank-stabilized LoRA (rsLoRA)
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

model_name_or_path=/sfs_turbo/pretrained_models/Qwen3.5-35B-A3B

seq_length=$((1024 * 4))
sp_size=2
ep_size=8
dataset=security
finetuning_type=lora

# LoRA hyperparameters
lora_rank=8
lora_alpha=16
lora_target=all
lora_dropout=0.0

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
output_dir=/mnt/sfs_turbo/${USER_NAME}/ckpt/${model_basename}/lf/mllm_cpt_lora_seq${seq_length_k}_sp${sp_size}ep${ep_size}_node${NNODES}/

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
    --num_train_epochs 1 \
    --report_to tensorboard \
    --logging_dir /home/admin/logs/tfevent \
    --gradient_checkpointing \
    --dataset_dir ${DATASET_DIR} \
    --max_steps 100 \
    --trust_remote_code \
    --disable_shuffling \
    --stage pt \
    --model_name_or_path $model_name_or_path \
    --do_train \
    --flash_attn fa2 \
    --dataset $dataset \
    --preprocessing_num_workers 16 \
    --cutoff_len ${seq_length} \
    --template qwen3_5_no_think \
    --finetuning_type ${finetuning_type} \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --lora_target ${lora_target} \
    --lora_dropout ${lora_dropout} \
    --use_rslora \
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
    --learning_rate 5e-4 \
    --save_total_limit 1 \
    --plot_loss \
    --seed 42 \
    --bf16 \
    --tt_freeze_vision_tower \
    --ddp_timeout 1800000000 2>&1 | tee -a $log_file
