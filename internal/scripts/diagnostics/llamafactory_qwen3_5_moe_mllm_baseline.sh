#!/usr/bin/env bash
# Qwen3.5-MoE baseline diagnostics

# env:
# pip install trl==0.9.6

# run:
# conda activate sftenv; cd /sfs_turbo/zhisan/0604
# bash torchturbo/scripts/diagnostics/llamafactory_qwen3_5_moe_mllm_baseline.sh

# diagnostics:
# python torchturbo/scripts/diagnostics/compare_diagnostics.py \
#     --mode compare --output_dir /sfs_turbo/zhisan/0604/diagnostics/qwen3_5_moe_mllm/ \
#     --verbose_compare > /sfs_turbo/zhisan/0604/diagnostics/qwen3_5_moe_mllm/diagnostics.log

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

# CRITICAL: disable torchturbo for baseline
export TORCH_TURBO_DISABLE=1


model_name_or_path=/sfs_turbo/zhisan/models/Qwen3___5-35B-A3B
output_dir=/sfs_turbo/zhisan/0604/diagnostics_cpt/qwen3_5_moe_mllm

seq_length=$((1024 * 1))
dataset=security
finetuning_type=full

export WANDB_DISABLED=True
export ALLOW_EXTRA_ARGS=True
export TRANSFORMERS_VERBOSITY=error

NPUS_PER_NODE=${NPUS_PER_NODE:-16}
export NNODES=${NNODES:-1}
export NODE_RANK=${RANK:-0}
export WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-23556}

export ASCEND_PROCESS_LOG_PATH=$output_dir/log

mkdir -p $output_dir

log_file=$output_dir/log-$NODE_RANK.txt
echo $output_dir
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

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET_DIR="$(dirname "$0")/../../dataset/"
COMPARE_SCRIPT="$SCRIPT_DIR/compare_diagnostics.py"

export PYTHONPATH=/opt/MindSpeed:$PYTHONPATH
export PYTHONPATH=/sfs_turbo/zhisan/0604/LLaMA-Factory/src:$PYTHONPATH
export PYTHONPATH=/sfs_turbo/zhisan/0604/torchturbo_qwen35:$PYTHONPATH
export PYTHONPATH=/sfs_turbo/zhisan/0604/transformers_4_57_3/src:$PYTHONPATH

export TORCH_TURBO_SP_SIZE=2
export TORCH_TURBO_EP_SIZE=8

torchrun $DISTRIBUTED_ARGS "$COMPARE_SCRIPT" \
    --mode baseline \
    --stage pt \
    --model_name_or_path $model_name_or_path \
    --flash_attn fa2 \
    --dataset_dir ${DATASET_DIR} \
    --dataset $dataset \
    --cutoff_len ${seq_length} \
    --template qwen3_5 \
    --output_dir $output_dir \
    --bf16 \
    --seed 42 \
    --per_device_train_batch_size 1 \
    --learning_rate 1.25e-5 \
    --dump_params \
    --dump_inputs \
    --dump_layer_io \
    --rank 0 \
    --trainer_class llamafactory.train.pt.torch_turbo_trainer.TorchTurboTrainer \
    --extra_lf_args --preprocessing_num_workers $((NNODES * 16)) --max_samples 64 --print_param_status True --trust_remote_code True --torch_turbo_enabled --torch_turbo_world_size $((NNODES * 16)) --torch_turbo_ep_size ${TORCH_TURBO_EP_SIZE} --torch_turbo_sp_size ${TORCH_TURBO_SP_SIZE} --torch_turbo_path $model_name_or_path --torch_turbo_save_path $output_dir

    # [baseline] Step 1 loss = 4.2461