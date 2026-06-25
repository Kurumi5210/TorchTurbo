#!/usr/bin/env bash
# qwen3_moe SFT training with torchturbo
# Usage:
#   bash scripts/runs/lf_qwen3_5_moe_sft.sh

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

export VIDEO_PLACEHOLDER="<qwen3_5_video_placeholder>"
export AUDIO_PLACEHOLDER="<qwen3_5_audio_placeholder>"
export IMAGE_PLACEHOLDER="<qwen3_5_image_placeholder>"

model_name_or_path=/sfs_turbo/zhisan/models/Qwen3___5-35B-A3B/
output_dir=/sfs_turbo/zhisan/0604/lf_logs_qwen35/run_qwen3_5_moe_debug_35b_sft_text_sp1ep8/ckpt

seq_length=$((1024 * 4))
dataset=science
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
RUN_LF="$(dirname "$0")/run_lf.py"
DATASET_DIR="$(dirname "$0")/../../dataset/"

export PYTHONPATH=/sfs_turbo/zhisan/0604/LlamaFactory/src:$PYTHONPATH
export PYTHONPATH=/sfs_turbo/zhisan/0604/torchturbo/src:$PYTHONPATH
export PYTHONPATH=/sfs_turbo/zhisan/0604/transformers_4_57_3_pure/src:$PYTHONPATH

torchrun $DISTRIBUTED_ARGS $RUN_LF \
    --tt_sp_size 1 \
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
    --template qwen3_5 \
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
    --ddp_timeout 1800000000 2>&1 | tee -a $log_file


# sp1 4k
# {'loss': 0.4877, 'grad_norm': 2.5924270153045654, 'learning_rate': 1.25e-05, 'epoch': 0.03}
# {'loss': 0.4523, 'grad_norm': 2.2152011394500732, 'learning_rate': 1.2465761846051709e-05, 'epoch': 0.07}
# {'loss': 0.4305, 'grad_norm': 1.8862305879592896, 'learning_rate': 1.2363422504586286e-05, 'epoch': 0.1}
# {'loss': 0.4155, 'grad_norm': 1.2942569255828857, 'learning_rate': 1.219410322684471e-05, 'epoch': 0.13}
# {'loss': 0.394, 'grad_norm': 1.2239418029785156, 'learning_rate': 1.1959659110266256e-05, 'epoch': 0.17}
# {'loss': 0.4007, 'grad_norm': 1.1116782426834106, 'learning_rate': 1.1662658773652743e-05, 'epoch': 0.2}
# {'loss': 0.3749, 'grad_norm': 1.0385875701904297, 'learning_rate': 1.1306356214843423e-05, 'epoch': 0.23}
# {'loss': 0.3799, 'grad_norm': 1.0991425514221191, 'learning_rate': 1.0894655159233715e-05, 'epoch': 0.27}
# {'loss': 0.4553, 'grad_norm': 1.0655317306518555, 'learning_rate': 1.0432066289742864e-05, 'epoch': 0.3}
# {'loss': 0.3853, 'grad_norm': 0.9212357997894287, 'learning_rate': 9.923657826827957e-06, 'epoch': 0.33}
# {'loss': 0.4005, 'grad_norm': 1.0315406322479248, 'learning_rate': 9.375000000000001e-06, 'epoch': 0.37}
# {'loss': 0.3702, 'grad_norm': 0.8967673778533936, 'learning_rate': 8.792104019223753e-06, 'epoch': 0.4}
# {'loss': 0.397, 'grad_norm': 0.9210770130157471, 'learning_rate': 8.181356214843423e-06, 'epoch': 0.43}
# {'loss': 0.411, 'grad_norm': 0.9503166675567627, 'learning_rate': 7.5494480676109954e-06, 'epoch': 0.47}
# {'loss': 0.4047, 'grad_norm': 1.0049320459365845, 'learning_rate': 6.903302895422835e-06, 'epoch': 0.5}
# {'loss': 0.3833, 'grad_norm': 1.0398653745651245, 'learning_rate': 6.25e-06, 'epoch': 0.53}
# {'loss': 0.397, 'grad_norm': 0.8819916844367981, 'learning_rate': 5.596697104577167e-06, 'epoch': 0.57}
# {'loss': 0.3973, 'grad_norm': 0.8654693365097046, 'learning_rate': 4.950551932389005e-06, 'epoch': 0.6}
# {'loss': 0.3917, 'grad_norm': 0.954013466835022, 'learning_rate': 4.3186437851565795e-06, 'epoch': 0.63}
# {'loss': 0.3103, 'grad_norm': 0.770961344242096, 'learning_rate': 3.7078959807762497e-06, 'epoch': 0.67}
# {'loss': 0.3656, 'grad_norm': 0.8647118806838989, 'learning_rate': 3.1250000000000014e-06, 'epoch': 0.7}
# {'loss': 0.3562, 'grad_norm': 0.8048088550567627, 'learning_rate': 2.5763421731720436e-06, 'epoch': 0.73}
# {'loss': 0.3723, 'grad_norm': 0.8900460600852966, 'learning_rate': 2.0679337102571383e-06, 'epoch': 0.77}
# {'loss': 0.3866, 'grad_norm': 0.8597457408905029, 'learning_rate': 1.6053448407662854e-06, 'epoch': 0.8}
# {'loss': 0.362, 'grad_norm': 0.8389605283737183, 'learning_rate': 1.1936437851565791e-06, 'epoch': 0.83}
# {'loss': 0.374, 'grad_norm': 0.8449287414550781, 'learning_rate': 8.373412263472582e-07, 'epoch': 0.87}
# {'loss': 0.4185, 'grad_norm': 0.8638467192649841, 'learning_rate': 5.403408897337439e-07, 'epoch': 0.9}
# {'loss': 0.3669, 'grad_norm': 2.4158074855804443, 'learning_rate': 3.0589677315529047e-07, 'epoch': 0.93}
# {'loss': 0.343, 'grad_norm': 0.8659589886665344, 'learning_rate': 1.3657749541371446e-07, 'epoch': 0.97}
# {'loss': 0.3044, 'grad_norm': 1.111772894859314, 'learning_rate': 3.423815394829194e-08, 'epoch': 1.0}
