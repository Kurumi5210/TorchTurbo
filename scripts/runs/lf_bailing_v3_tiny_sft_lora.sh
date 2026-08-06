#!/usr/bin/env bash
# bailing_v3_tiny LoRA SFT training with torchturbo
# Usage:
#   bash scripts/runs/lf_bailing_v3_tiny_sft_lora.sh
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

if [ -z "$USER_NAME" ]; then
    echo "Error: USER_NAME is not set"
    exit 1
fi

#model_name_or_path=/sfs_turbo/pretrained_models/bailing_v3/sft_ring_tiny_v3_formal_0402_rc52_baseline_100w_safegate_0403_iter_0000324/
model_name_or_path=/mnt/aicompiler-a3/common/pretrained_models/bailing_v3/sft_ring_tiny_v3_formal_0402_rc52_baseline_100w_safegate_0403_iter_0000324/
output_dir=/mnt/sfs_turbo/${USER_NAME}/ckpt/ling_v3/lf_lora/

seq_length=$((1024 * 4))
dataset=science
packing_mode=packing_sep
finetuning_type=lora

# LoRA hyperparameters
lora_rank=8
lora_alpha=16
lora_target=all
lora_dropout=0.0

# LoRA weight initialization strategy.
# Supported values:
#   true   - Standard Kaiming uniform for A, zeros for B (default)
#   false  - Skip initialization (useful for loading pre-trained adapters)
#   gaussian - Gaussian init for A, zeros for B
#   pissa  - PiSSA initialization
#   pissa_niter_<N> - PiSSA with FSVD (N iteration steps), e.g. pissa_niter_16
#   loftq  - LoftQ initialization
pissa_init=false

export PATH=/usr/local/Ascend/tools/bishengir/bin:$PATH

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
RUN_LF="$(dirname "$0")/run_lf.py"
DATASET_DIR="$(dirname "$0")/../../dataset/npu/"

export PYTHONPATH=/mnt/sfs_turbo/xiyun.wc/opensource/LlamaFactory/src/:$PYTHONPATH
export PYTHONPATH=/mnt/sfs_turbo/xiyun.wc/torchturbo/src:$PYTHONPATH
export PYTHONPATH=/mnt/sfs_turbo/xiyun.wc/flash-linear-attention/:$PYTHONPATH

torchrun $DISTRIBUTED_ARGS $RUN_LF \
    --tt_sp_size 2 \
    --tt_ep_size 8 \
    --num_train_epochs 1 \
    --report_to tensorboard \
    --logging_dir /home/admin/logs/tfevent \
    --gradient_checkpointing \
    --dataset_dir ${DATASET_DIR} \
    --max_samples 1000 \
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
    --template bailing_v3 \
    --finetuning_type ${finetuning_type} \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --lora_target ${lora_target} \
    --lora_dropout ${lora_dropout} \
    $([ "$pissa_init" = true ] && echo "--pissa_init") \
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
    --ddp_timeout 1800000000 2>&1 | tee $log_file