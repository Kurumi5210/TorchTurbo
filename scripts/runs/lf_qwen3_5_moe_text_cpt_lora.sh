#!/usr/bin/env bash
# Qwen3.5-MoE LoRA TEXT CPT with torchturbo
# Usage:
#   bash scripts/runs/lf_qwen3_5_moe_text_cpt_lora.sh

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
output_dir=/sfs_turbo/zhisan/0604/lf_logs_qwen35/run_qwen3_5_moe_text_cpt_lora_sp2ep8/ckpt

seq_length=$((1024 * 4))
dataset=wikipedia_en
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
    --max_samples 3000 \
    --trust_remote_code \
    --disable_shuffling \
    --stage pt \
    --model_name_or_path $model_name_or_path \
    --do_train \
    --flash_attn fa2 \
    --dataset $dataset \
    --preprocessing_num_workers 16 \
    --cutoff_len ${seq_length} \
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
# {'loss': 1.8271, 'grad_norm': 0.22169271111488342, 'learning_rate': 0.0005, 'epoch': 0.03}
# {'loss': 1.8882, 'grad_norm': 0.1231415793299675, 'learning_rate': 0.0004989335440737586, 'epoch': 0.06}
# {'loss': 1.7876, 'grad_norm': 0.0994306281208992, 'learning_rate': 0.0004957432749209755, 'epoch': 0.09}
# {'loss': 1.8765, 'grad_norm': 0.07847367227077484, 'learning_rate': 0.0004904564107932048, 'epoch': 0.12}
# {'loss': 1.7642, 'grad_norm': 0.08089607954025269, 'learning_rate': 0.00048311805735108893, 'epoch': 0.15}
# {'loss': 1.7207, 'grad_norm': 0.08588796108961105, 'learning_rate': 0.00047379082283876563, 'epoch': 0.18}
# {'loss': 1.6621, 'grad_norm': 0.07750597596168518, 'learning_rate': 0.0004625542839324036, 'epoch': 0.21}
# {'loss': 1.6807, 'grad_norm': 0.07799782603979111, 'learning_rate': 0.00044950430682005996, 'epoch': 0.24}
# {'loss': 1.6709, 'grad_norm': 0.07793066650629044, 'learning_rate': 0.00043475222930516476, 'epoch': 0.26}
# {'loss': 1.7925, 'grad_norm': 0.07637333124876022, 'learning_rate': 0.0004184239109116393, 'epoch': 0.29}
# {'loss': 1.771, 'grad_norm': 0.08897679299116135, 'learning_rate': 0.0004006586590948141, 'epoch': 0.32}
# {'loss': 1.7168, 'grad_norm': 0.07358987629413605, 'learning_rate': 0.00038160804071933895, 'epoch': 0.35}
# {'loss': 1.7341, 'grad_norm': 0.21836301684379578, 'learning_rate': 0.0003614345889441346, 'epoch': 0.38}
# {'loss': 1.7666, 'grad_norm': 0.07563623040914536, 'learning_rate': 0.0003403104165467883, 'epoch': 0.41}
# {'loss': 1.8115, 'grad_norm': 0.07720393687486649, 'learning_rate': 0.0003184157475180208, 'epoch': 0.44}
# {'loss': 1.7183, 'grad_norm': 0.07238943129777908, 'learning_rate': 0.0002959373794541426, 'epoch': 0.47}
# {'loss': 1.6406, 'grad_norm': 0.0731380358338356, 'learning_rate': 0.0002730670898658255, 'epoch': 0.5}
# {'loss': 1.6572, 'grad_norm': 0.06913821399211884, 'learning_rate': 0.00025, 'epoch': 0.53}
# {'loss': 1.6953, 'grad_norm': 0.07525192201137543, 'learning_rate': 0.00022693291013417452, 'epoch': 0.56}
# {'loss': 1.7881, 'grad_norm': 0.07563211023807526, 'learning_rate': 0.0002040626205458574, 'epoch': 0.59}
# {'loss': 1.6567, 'grad_norm': 0.07782697677612305, 'learning_rate': 0.0001815842524819793, 'epoch': 0.62}
# {'loss': 1.7432, 'grad_norm': 0.06875443458557129, 'learning_rate': 0.00015968958345321176, 'epoch': 0.65}
# {'loss': 1.7656, 'grad_norm': 0.06893914937973022, 'learning_rate': 0.00013856541105586545, 'epoch': 0.68}
# {'loss': 1.6514, 'grad_norm': 0.06722293049097061, 'learning_rate': 0.00011839195928066102, 'epoch': 0.71}
# {'loss': 1.7388, 'grad_norm': 0.06638005375862122, 'learning_rate': 9.934134090518593e-05, 'epoch': 0.74}
# {'loss': 1.7749, 'grad_norm': 0.06572806090116501, 'learning_rate': 8.15760890883607e-05, 'epoch': 0.76}
# {'loss': 1.7461, 'grad_norm': 0.07324633002281189, 'learning_rate': 6.524777069483526e-05, 'epoch': 0.79}
# {'loss': 1.8433, 'grad_norm': 0.06826777756214142, 'learning_rate': 5.049569317994013e-05, 'epoch': 0.82}
# {'loss': 1.8013, 'grad_norm': 0.06806404143571854, 'learning_rate': 3.7445716067596506e-05, 'epoch': 0.85}
# {'loss': 1.7388, 'grad_norm': 0.07202189415693283, 'learning_rate': 2.6209177161234443e-05, 'epoch': 0.88}
# {'loss': 1.3502, 'grad_norm': 0.06694286316633224, 'learning_rate': 1.6881942648911074e-05, 'epoch': 0.91}
# {'loss': 1.7773, 'grad_norm': 0.07058112323284149, 'learning_rate': 9.54358920679524e-06, 'epoch': 0.94}
# {'loss': 1.709, 'grad_norm': 0.06460283696651459, 'learning_rate': 4.256725079024554e-06, 'epoch': 0.97}
# {'loss': 1.7319, 'grad_norm': 0.06821692734956741, 'learning_rate': 1.066455926241383e-06, 'epoch': 1.0}
