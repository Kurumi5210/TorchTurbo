#!/usr/bin/env bash
# Qwen3.5-MoE LoRA TEXT SFT with torchturbo
# Usage:
#   bash scripts/runs/lf_qwen3_5_moe_text_sft_lora.sh

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
output_dir=/sfs_turbo/zhisan/0604/lf_logs_qwen35/run_qwen3_5_moe_text_sft_lora_sp2ep8/ckpt

seq_length=$((1024 * 4))
dataset=science
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
    --template qwen3_5 \
    --finetuning_type ${finetuning_type} \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --lora_target ${lora_target} \
    --lora_dropout ${lora_dropout} \
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
    --ddp_timeout 1800000000 2>&1 | tee -a $log_file

    # --use_rslora \

# sp2 4k
# {'loss': 0.4881, 'grad_norm': 0.17927393317222595, 'learning_rate': 0.0005, 'epoch': 0.03}
# {'loss': 0.4614, 'grad_norm': 0.14600823819637299, 'learning_rate': 0.0004986304738420684, 'epoch': 0.07}
# {'loss': 0.4559, 'grad_norm': 0.21046696603298187, 'learning_rate': 0.0004945369001834514, 'epoch': 0.1}
# {'loss': 0.4218, 'grad_norm': 0.10214848071336746, 'learning_rate': 0.0004877641290737884, 'epoch': 0.13}
# {'loss': 0.3992, 'grad_norm': 0.09394528716802597, 'learning_rate': 0.0004783863644106502, 'epoch': 0.17}
# {'loss': 0.4091, 'grad_norm': 0.10295327752828598, 'learning_rate': 0.00046650635094610973, 'epoch': 0.2}
# {'loss': 0.3765, 'grad_norm': 0.07790575921535492, 'learning_rate': 0.0004522542485937369, 'epoch': 0.23}
# {'loss': 0.3782, 'grad_norm': 0.07589846849441528, 'learning_rate': 0.00043578620636934855, 'epoch': 0.27}
# {'loss': 0.4579, 'grad_norm': 0.08114815503358841, 'learning_rate': 0.0004172826515897146, 'epoch': 0.3}
# {'loss': 0.3853, 'grad_norm': 0.06619919836521149, 'learning_rate': 0.0003969463130731183, 'epoch': 0.33}
# {'loss': 0.4001, 'grad_norm': 0.07304911315441132, 'learning_rate': 0.000375, 'epoch': 0.37}
# {'loss': 0.3671, 'grad_norm': 0.06727302819490433, 'learning_rate': 0.0003516841607689501, 'epoch': 0.4}
# {'loss': 0.3958, 'grad_norm': 0.06569687277078629, 'learning_rate': 0.00032725424859373687, 'epoch': 0.43}
# {'loss': 0.4093, 'grad_norm': 0.0711311399936676, 'learning_rate': 0.0003019779227044398, 'epoch': 0.47}
# {'loss': 0.4043, 'grad_norm': 0.07132981717586517, 'learning_rate': 0.0002761321158169134, 'epoch': 0.5}
# {'loss': 0.3799, 'grad_norm': 0.07110428810119629, 'learning_rate': 0.00025, 'epoch': 0.53}
# {'loss': 0.3951, 'grad_norm': 0.058793604373931885, 'learning_rate': 0.00022386788418308668, 'epoch': 0.57}
# {'loss': 0.395, 'grad_norm': 0.05676395073533058, 'learning_rate': 0.0001980220772955602, 'epoch': 0.6}
# {'loss': 0.3911, 'grad_norm': 0.0642547607421875, 'learning_rate': 0.00017274575140626317, 'epoch': 0.63}
# {'loss': 0.3088, 'grad_norm': 0.05138992890715599, 'learning_rate': 0.00014831583923105, 'epoch': 0.67}
# {'loss': 0.3655, 'grad_norm': 0.062136322259902954, 'learning_rate': 0.00012500000000000006, 'epoch': 0.7}
# {'loss': 0.355, 'grad_norm': 0.05740569531917572, 'learning_rate': 0.00010305368692688174, 'epoch': 0.73}
# {'loss': 0.3702, 'grad_norm': 0.05665472149848938, 'learning_rate': 8.271734841028553e-05, 'epoch': 0.77}
# {'loss': 0.3873, 'grad_norm': 0.059933140873909, 'learning_rate': 6.421379363065141e-05, 'epoch': 0.8}
# {'loss': 0.3622, 'grad_norm': 0.060353998094797134, 'learning_rate': 4.7745751406263163e-05, 'epoch': 0.83}
# {'loss': 0.3746, 'grad_norm': 0.05793457478284836, 'learning_rate': 3.3493649053890325e-05, 'epoch': 0.87}
# {'loss': 0.4193, 'grad_norm': 0.056960579007864, 'learning_rate': 2.1613635589349755e-05, 'epoch': 0.9}
# {'loss': 0.3663, 'grad_norm': 0.06032372638583183, 'learning_rate': 1.2235870926211617e-05, 'epoch': 0.93}
# {'loss': 0.3439, 'grad_norm': 0.053760528564453125, 'learning_rate': 5.463099816548578e-06, 'epoch': 0.97}
# {'loss': 0.3336, 'grad_norm': 0.07081377506256104, 'learning_rate': 1.3695261579316775e-06, 'epoch': 1.0}
