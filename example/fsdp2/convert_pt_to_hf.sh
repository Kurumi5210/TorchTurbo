set -o pipefail

set -ex

export PATH=/opt/pyvenv/bin:$PATH
source /opt/pyvenv/bin/activate
export PATH_ASCEND=/usr/local/Ascend
export LD_LIBRARY_PATH=${PATH_ASCEND}/ascend-toolkit/latest/aarch64-linux:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${PATH_ASCEND}/driver/tools/hccn_tool:$LD_LIBRARY_PATH
source ${PATH_ASCEND}/ascend-toolkit/set_env.sh
source ${PATH_ASCEND}/nnal/atb/set_env.sh
LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

export HCCL_CONNECT_TIMEOUT=1800
#export HCCL_ASYNC_ERROR_HANDLING=0
#export INF_NAN_MODE_ENABLE=1
export ASCEND_GLOBAL_LOG_LEVEL=3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_IF_BASE_PORT=29972
export GLOO_SOCKET_IFNAME=eth0
export GLOO_LOG_LEVEL=DEBUG

source /sfs_turbo/qinjue/commands/env.sh

export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
export PYTHONPATH=/sfs_turbo/xiyun/rl/3d/opensource/Megatron-LM:/sfs_turbo/dsh/MindSpeed:/sfs_turbo/qinjue/torchturbo:$PYTHONPATH

export WORLD_SIZE=${WORLD_SIZE:-1}
export NODE_RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-23556}

save_path=/sfs_turbo/qinjue/ckpt/Qwen3-235B/checkpoint-20
hf_save_path=/sfs_turbo/qinjue/ckpt/Qwen3-235B/hf_ckpt
config_path=/sfs_turbo/pretrained_models/Qwen3-235B-A22B-Thinking-2507/

output_dir=/sfs_turbo/qinjue/qwen3_235b_sft/ckpt/

mkdir -p $output_dir
if [ -d $output_dir ]; then
  echo 'Exist: ', $output_dir
else
  echo 'Not exist: ', $output_dir
fi
log_file=$output_dir/log-$NODE_RANK.txt

train_ep=16

torchrun \
    --nproc_per_node=16 \
    --nnodes=${WORLD_SIZE} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    examples/pt2hf.py \
    --save_path $save_path \
    --hf_save_path $hf_save_path \
    --world_size $((WORLD_SIZE * 16)) \
    --ep_size ${train_ep} \
    --config_path $config_path 2>&1 | tee $log_file
