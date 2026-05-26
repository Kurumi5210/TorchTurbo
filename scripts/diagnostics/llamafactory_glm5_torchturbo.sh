#!/usr/bin/env bash

set -o pipefail

set -ex

# source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /mnt/sfs_turbo/hw/dengshumin/CANN_2/ascend-toolkit/set_env.sh
source /mnt/sfs_turbo/hw/dengshumin/CANN_2/nnal/atb/set_env.sh || true
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export ASCEND_GLOBAL_LOG_LEVEL=3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_IF_BASE_PORT=29972
export GLOO_SOCKET_IFNAME=eth0
export GLOO_LOG_LEVEL=DEBUG
export MULTI_STREAM_MEMORY_REUSE=2 # 多流

model_name_or_path=/sfs_turbo/pretrained_models/GLM-5
output_dir=/sfs_turbo/zlj/output

seq_length=$((1024 * 4))
dataset=science
packing_mode=packing_sep
finetuning_type=full

export WANDB_DISABLED=True
export ALLOW_EXTRA_ARGS=True
export PYTHONPATH=/opt/MindSpeed:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}
export TRANSFORMERS_VERBOSITY=error

# export NNODES=${NNODES:-1}
export NNODES=1
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

export PYTHONPATH=/sfs_turbo/zlj/workspace/glm5/v3/LLaMA-Factory/src:$PYTHONPATH
export PYTHONPATH=/sfs_turbo/zlj/workspace/glm5/v3/torchturbo/src:$PYTHONPATH
export TORCH_TURBO_SP_SIZE=2
export TORCH_TURBO_EP_SIZE=8

torchrun $DISTRIBUTED_ARGS compare_diagnostics.py \
    --mode torchturbo \
    --model_name_or_path $model_name_or_path \
    --flash_attn fa2 \
    --dataset $dataset \
    --cutoff_len ${seq_length} \
    --template glm4_7 \
    --output_dir $output_dir \
    --bf16 \
    --neat_packing \
    --seed 42 \
    --per_device_train_batch_size 1 \
    --learning_rate 1.25e-5 \
    --dump_params \
    --dump_inputs \
    --dump_layer_io \
    --rank 0 \
    --extra_lf_args --plot_loss --overwrite_cache --preprocessing_num_workers $((NNODES * 16)) --max_samples 64 --print_param_status True --trust_remote_code True
