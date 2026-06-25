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
output_dir=/sfs_turbo/zhisan/0604/lf_logs_qwen35/run_qwen3_5_moe_debug_35b_sft_text_sp2ep8/ckpt

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

# sp2 4k
# {'loss': 0.4881, 'grad_norm': 2.5076005458831787, 'learning_rate': 1.25e-05, 'epoch': 0.03}
# {'loss': 0.4517, 'grad_norm': 2.230151891708374, 'learning_rate': 1.2465761846051709e-05, 'epoch': 0.07}
# {'loss': 0.4305, 'grad_norm': 1.8571326732635498, 'learning_rate': 1.2363422504586286e-05, 'epoch': 0.1}
# {'loss': 0.4161, 'grad_norm': 1.3235825300216675, 'learning_rate': 1.219410322684471e-05, 'epoch': 0.13}
# {'loss': 0.3975, 'grad_norm': 1.4016706943511963, 'learning_rate': 1.1959659110266256e-05, 'epoch': 0.17}
# {'loss': 0.4016, 'grad_norm': 1.208695650100708, 'learning_rate': 1.1662658773652743e-05, 'epoch': 0.2}
# {'loss': 0.3757, 'grad_norm': 1.0309809446334839, 'learning_rate': 1.1306356214843423e-05, 'epoch': 0.23}
# {'loss': 0.3799, 'grad_norm': 1.0955591201782227, 'learning_rate': 1.0894655159233715e-05, 'epoch': 0.27}
# {'loss': 0.4543, 'grad_norm': 1.055594801902771, 'learning_rate': 1.0432066289742864e-05, 'epoch': 0.3}
# {'loss': 0.3853, 'grad_norm': 0.9120645523071289, 'learning_rate': 9.923657826827957e-06, 'epoch': 0.33}
# {'loss': 0.4009, 'grad_norm': 1.0075644254684448, 'learning_rate': 9.375000000000001e-06, 'epoch': 0.37}
# {'loss': 0.3699, 'grad_norm': 0.9029478430747986, 'learning_rate': 8.792104019223753e-06, 'epoch': 0.4}
# {'loss': 0.3977, 'grad_norm': 0.925118625164032, 'learning_rate': 8.181356214843423e-06, 'epoch': 0.43}
# {'loss': 0.4106, 'grad_norm': 1.0324761867523193, 'learning_rate': 7.5494480676109954e-06, 'epoch': 0.47}
# {'loss': 0.4043, 'grad_norm': 0.9997653961181641, 'learning_rate': 6.903302895422835e-06, 'epoch': 0.5}
# {'loss': 0.3833, 'grad_norm': 1.040412425994873, 'learning_rate': 6.25e-06, 'epoch': 0.53}
# {'loss': 0.3978, 'grad_norm': 0.894476592540741, 'learning_rate': 5.596697104577167e-06, 'epoch': 0.57}
# {'loss': 0.3973, 'grad_norm': 0.8922469019889832, 'learning_rate': 4.950551932389005e-06, 'epoch': 0.6}
# {'loss': 0.3921, 'grad_norm': 0.9621519446372986, 'learning_rate': 4.3186437851565795e-06, 'epoch': 0.63}
# {'loss': 0.3106, 'grad_norm': 0.7643373012542725, 'learning_rate': 3.7078959807762497e-06, 'epoch': 0.67}
# {'loss': 0.3652, 'grad_norm': 0.836165189743042, 'learning_rate': 3.1250000000000014e-06, 'epoch': 0.7}
# {'loss': 0.3557, 'grad_norm': 0.944053590297699, 'learning_rate': 2.5763421731720436e-06, 'epoch': 0.73}
# {'loss': 0.3716, 'grad_norm': 0.8589189648628235, 'learning_rate': 2.0679337102571383e-06, 'epoch': 0.77}
# {'loss': 0.3866, 'grad_norm': 0.8552417159080505, 'learning_rate': 1.6053448407662854e-06, 'epoch': 0.8}
# {'loss': 0.3615, 'grad_norm': 0.8224872350692749, 'learning_rate': 1.1936437851565791e-06, 'epoch': 0.83}
# {'loss': 0.3734, 'grad_norm': 0.8490666151046753, 'learning_rate': 8.373412263472582e-07, 'epoch': 0.87}
# {'loss': 0.4182, 'grad_norm': 0.8546963930130005, 'learning_rate': 5.403408897337439e-07, 'epoch': 0.9}
# {'loss': 0.3667, 'grad_norm': 0.8741291165351868, 'learning_rate': 3.0589677315529047e-07, 'epoch': 0.93}
# {'loss': 0.3431, 'grad_norm': 0.7589819431304932, 'learning_rate': 1.3657749541371446e-07, 'epoch': 0.97}
# {'loss': 0.3067, 'grad_norm': 1.1188074350357056, 'learning_rate': 3.423815394829194e-08, 'epoch': 1.0}
