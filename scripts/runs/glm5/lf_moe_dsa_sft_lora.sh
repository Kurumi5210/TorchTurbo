#!/usr/bin/env bash
# glm_moe_dsa SFT LoRA training with torchturbo
# Usage:
#   bash scripts/runs/glm5/lf_moe_dsa_sft_lora.sh

set -o pipefail

set -ex

# source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /mnt/sfs_turbo/hw/dengshumin/CANN_2/ascend-toolkit/set_env.sh
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_EVENT_TIMEOUT=14500
export ACL_DEVICE_SYNC_TIMEOUT=14400
export ASCEND_GLOBAL_LOG_LEVEL=3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_ASYNC_ERROR_HANDLING=0
source /mnt/sfs_turbo/hw/dengshumin/CANN_2/nnal/atb/set_env.sh || true

export HCCL_IF_BASE_PORT=29972
export GLOO_SOCKET_IFNAME=eth0
export GLOO_LOG_LEVEL=DEBUG
export MULTI_STREAM_MEMORY_REUSE=2 # 多流

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

model_name_or_path=/mnt/sfs_turbo/pretrained_models/GLM-5.2

seq_length=$((1024 * 64))
dataset=science
packing_mode=packing_sep
finetuning_type=lora
sp_size=32
ep_size=32

# LoRA hyperparameters
lora_rank=8
lora_alpha=16
lora_target=all
lora_dropout=0.0

export WANDB_DISABLED=True
export ALLOW_EXTRA_ARGS=True
export PYTHONPATH=/opt/MindSpeed:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}
export TRANSFORMERS_VERBOSITY=error

export NNODES=${NNODES:-1}
export NPUS_PER_NODE=${NPUS_PER_NODE:-16}
export NODE_RANK=${RANK:-0}
export WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-23556}

model_basename=$(basename "$model_name_or_path")
seq_length_k=$((seq_length / 1024))k
output_dir=/sfs_turbo/zlj/ckpt/${model_basename}/lf/sft_lora_seq${seq_length_k}_sp${sp_size}ep${ep_size}_node${NNODES}/

export ASCEND_PROCESS_LOG_PATH=$output_dir/log

export OOM_SNAPSHOT_ENABLE=1
export OOM_SNAPSHOT_PATH=/sfs_turbo/shanyicheng.syc
export TRITON_ALL_BLOCKS_PARALLEL=1

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
RUN_LF="$(dirname "$0")/../run_lf.py"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUN_LF="${SCRIPT_DIR}/../run_lf.py"
DATASET_DIR="${REPO_ROOT}/dataset/npu/"
TORCHTURBO_SRC_DIR="${TORCHTURBO_SRC_DIR:-${REPO_ROOT}/src}"

#export ASCEND_LAUNCH_BLOCKING=1

#export PYTHONPATH=/sfs_turbo/zlj/workspace/LLaMA-Factory/src:$PYTHONPATH
export PYTHONPATH=/mnt/sfs_turbo/torchturbo/stable/LLaMA-Factory/src/:$PYTHONPATH
#export PYTHONPATH=/sfs_turbo/zlj/workspace/torchturbo/src:$PYTHONPATH

torchrun $DISTRIBUTED_ARGS $RUN_LF \
    --tt_sp_size $sp_size \
    --tt_ep_size $ep_size \
    --num_train_epochs 1 \
    --report_to tensorboard \
    --logging_dir /home/admin/logs/tfevent \
    --gradient_checkpointing \
    --trust_remote_code \
    --disable_shuffling \
    --stage sft \
    --model_name_or_path $model_name_or_path \
    --do_train \
    --flash_attn fa2 \
    --dataset $dataset \
    --preprocessing_num_workers $((NNODES * 16)) \
    --cutoff_len ${seq_length} \
    --neat_packing \
    --template glm4_7 \
    --finetuning_type ${finetuning_type} \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --lora_target ${lora_target} \
    --lora_dropout ${lora_dropout} \
    --use_rslora \
    --tt_ac_use_reentrant true \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --overwrite_cache \
    --dataset_dir ${DATASET_DIR} \
    --save_strategy "steps" \
    --eval_steps 200000000 \
    --save_steps 200000000 \
    --warmup_ratio 0.0 \
    --learning_rate 1e-5 \
    --save_total_limit 1 \
    --plot_loss \
    --seed 42 \
    --bf16 \
    --ddp_timeout 1800000000 2>&1| tee $log_file