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

model_name_or_path=/sfs_turbo/zhisan/models/Qwen3___5-35B-A3B/
output_dir=/sfs_turbo/zhisan/0604/lf_logs_qwen35/run_qwen3_5_moe_debug_35b_sft_mllm_sp1ep8/ckpt

seq_length=$((1024 * 4))
dataset=llava_1k_zh
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
    --max_steps 30 \
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
    --tt_freeze_vision_tower \
    --ddp_timeout 1800000000 2>&1 | tee -a $log_file

    # --tt_freeze_vision_tower \
    # --tt_freeze_multi_modal_projector \
    # --tt_freeze_language_model \

# sp1 4k [Full training]
# {'loss': 1.3477, 'grad_norm': 48.74412155151367, 'learning_rate': 1.25e-05, 'epoch': 0.17}
# {'loss': 1.0647, 'grad_norm': 8.817215919494629, 'learning_rate': 1.2465761846051709e-05, 'epoch': 0.33}
# {'loss': 1.0837, 'grad_norm': 7.5216755867004395, 'learning_rate': 1.2363422504586286e-05, 'epoch': 0.5}
# {'loss': 1.0393, 'grad_norm': 3.0703768730163574, 'learning_rate': 1.219410322684471e-05, 'epoch': 0.67}
# {'loss': 1.0703, 'grad_norm': 6.063371658325195, 'learning_rate': 1.1959659110266256e-05, 'epoch': 0.83}
# {'loss': 0.9929, 'grad_norm': 3.708191156387329, 'learning_rate': 1.1662658773652743e-05, 'epoch': 1.0}
# {'loss': 0.8582, 'grad_norm': 4.347257137298584, 'learning_rate': 1.1306356214843423e-05, 'epoch': 1.17}
# {'loss': 0.697, 'grad_norm': 3.1219208240509033, 'learning_rate': 1.0894655159233715e-05, 'epoch': 1.33}
# {'loss': 0.7135, 'grad_norm': 4.073367595672607, 'learning_rate': 1.0432066289742864e-05, 'epoch': 1.5}
# {'loss': 0.6067, 'grad_norm': 4.083878517150879, 'learning_rate': 9.923657826827957e-06, 'epoch': 1.67}
# {'loss': 0.7949, 'grad_norm': 9.958456039428711, 'learning_rate': 9.375000000000001e-06, 'epoch': 1.83}
# {'loss': 0.6583, 'grad_norm': 20.975284576416016, 'learning_rate': 8.792104019223753e-06, 'epoch': 2.0}
# {'loss': 0.5918, 'grad_norm': 4.59750509262085, 'learning_rate': 8.181356214843423e-06, 'epoch': 2.17}
# {'loss': 0.4071, 'grad_norm': 3.176424980163574, 'learning_rate': 7.5494480676109954e-06, 'epoch': 2.33}
# {'loss': 0.4411, 'grad_norm': 5.944131851196289, 'learning_rate': 6.903302895422835e-06, 'epoch': 2.5}
# {'loss': 0.3524, 'grad_norm': 3.7579140663146973, 'learning_rate': 6.25e-06, 'epoch': 2.67}
# {'loss': 0.601, 'grad_norm': 2.9563751220703125, 'learning_rate': 5.596697104577167e-06, 'epoch': 2.83}
# {'loss': 0.4694, 'grad_norm': 2.9349725246429443, 'learning_rate': 4.950551932389005e-06, 'epoch': 3.0}
# {'loss': 0.3331, 'grad_norm': 2.8228893280029297, 'learning_rate': 4.3186437851565795e-06, 'epoch': 3.17}
# {'loss': 0.2072, 'grad_norm': 3.379931688308716, 'learning_rate': 3.7078959807762497e-06, 'epoch': 3.33}
# {'loss': 0.3092, 'grad_norm': 5.017539024353027, 'learning_rate': 3.1250000000000014e-06, 'epoch': 3.5}
# {'loss': 0.2213, 'grad_norm': 3.740523338317871, 'learning_rate': 2.5763421731720436e-06, 'epoch': 3.67}
# {'loss': 0.4248, 'grad_norm': 2.903871774673462, 'learning_rate': 2.0679337102571383e-06, 'epoch': 3.83}
# {'loss': 0.313, 'grad_norm': 3.1027579307556152, 'learning_rate': 1.6053448407662854e-06, 'epoch': 4.0}
# {'loss': 0.2011, 'grad_norm': 2.6836917400360107, 'learning_rate': 1.1936437851565791e-06, 'epoch': 4.17}
# {'loss': 0.1292, 'grad_norm': 3.423800230026245, 'learning_rate': 8.373412263472582e-07, 'epoch': 4.33}
# {'loss': 0.2392, 'grad_norm': 3.3973190784454346, 'learning_rate': 5.403408897337439e-07, 'epoch': 4.5}
# {'loss': 0.1587, 'grad_norm': 2.195711612701416, 'learning_rate': 3.0589677315529047e-07, 'epoch': 4.67}
# {'loss': 0.3458, 'grad_norm': 5.280300617218018, 'learning_rate': 1.3657749541371446e-07, 'epoch': 4.83}
# {'loss': 0.2631, 'grad_norm': 2.4790239334106445, 'learning_rate': 3.423815394829194e-08, 'epoch': 5.0}

# sp1 4k [Freeze vision encoder]
# {'loss': 1.3477, 'grad_norm': 26.841442108154297, 'learning_rate': 1.25e-05, 'epoch': 0.17}
# {'loss': 1.0642, 'grad_norm': 2.7189204692840576, 'learning_rate': 1.2465761846051709e-05, 'epoch': 0.33}
# {'loss': 1.0806, 'grad_norm': 2.5816848278045654, 'learning_rate': 1.2363422504586286e-05, 'epoch': 0.5}
# {'loss': 1.0369, 'grad_norm': 1.5181465148925781, 'learning_rate': 1.219410322684471e-05, 'epoch': 0.67}
# {'loss': 1.0693, 'grad_norm': 1.7750855684280396, 'learning_rate': 1.1959659110266256e-05, 'epoch': 0.83}
# {'loss': 0.9883, 'grad_norm': 1.35183584690094, 'learning_rate': 1.1662658773652743e-05, 'epoch': 1.0}
# {'loss': 0.8623, 'grad_norm': 1.3906352519989014, 'learning_rate': 1.1306356214843423e-05, 'epoch': 1.17}
# {'loss': 0.6643, 'grad_norm': 2.051624059677124, 'learning_rate': 1.0894655159233715e-05, 'epoch': 1.33}
# {'loss': 0.6904, 'grad_norm': 1.7173423767089844, 'learning_rate': 1.0432066289742864e-05, 'epoch': 1.5}
# {'loss': 0.641, 'grad_norm': 1.5941121578216553, 'learning_rate': 9.923657826827957e-06, 'epoch': 1.67}
# {'loss': 0.7392, 'grad_norm': 3.1454038619995117, 'learning_rate': 9.375000000000001e-06, 'epoch': 1.83}
# {'loss': 0.6229, 'grad_norm': 2.8993821144104004, 'learning_rate': 8.792104019223753e-06, 'epoch': 2.0}
# {'loss': 0.5297, 'grad_norm': 1.5980730056762695, 'learning_rate': 8.181356214843423e-06, 'epoch': 2.17}
# {'loss': 0.4179, 'grad_norm': 2.1139748096466064, 'learning_rate': 7.5494480676109954e-06, 'epoch': 2.33}
# {'loss': 0.4208, 'grad_norm': 2.542346954345703, 'learning_rate': 6.903302895422835e-06, 'epoch': 2.5}
# {'loss': 0.3607, 'grad_norm': 3.1339423656463623, 'learning_rate': 6.25e-06, 'epoch': 2.67}
# {'loss': 0.5122, 'grad_norm': 2.5515859127044678, 'learning_rate': 5.596697104577167e-06, 'epoch': 2.83}
# {'loss': 0.3596, 'grad_norm': 2.078458309173584, 'learning_rate': 4.950551932389005e-06, 'epoch': 3.0}
# {'loss': 0.2748, 'grad_norm': 3.2309741973876953, 'learning_rate': 4.3186437851565795e-06, 'epoch': 3.17}
# {'loss': 0.2575, 'grad_norm': 4.736144542694092, 'learning_rate': 3.7078959807762497e-06, 'epoch': 3.33}
# {'loss': 0.2837, 'grad_norm': 4.042264461517334, 'learning_rate': 3.1250000000000014e-06, 'epoch': 3.5}
# {'loss': 0.2459, 'grad_norm': 2.3866636753082275, 'learning_rate': 2.5763421731720436e-06, 'epoch': 3.67}
# {'loss': 0.3651, 'grad_norm': 1.6830756664276123, 'learning_rate': 2.0679337102571383e-06, 'epoch': 3.83}
# {'loss': 0.2211, 'grad_norm': 2.5307466983795166, 'learning_rate': 1.6053448407662854e-06, 'epoch': 4.0}
# {'loss': 0.1769, 'grad_norm': 2.0420100688934326, 'learning_rate': 1.1936437851565791e-06, 'epoch': 4.17}
# {'loss': 0.1857, 'grad_norm': 1.258335828781128, 'learning_rate': 8.373412263472582e-07, 'epoch': 4.33}
# {'loss': 0.2156, 'grad_norm': 1.6437913179397583, 'learning_rate': 5.403408897337439e-07, 'epoch': 4.5}
# {'loss': 0.1829, 'grad_norm': 1.802366852760315, 'learning_rate': 3.0589677315529047e-07, 'epoch': 4.67}
# {'loss': 0.2967, 'grad_norm': 2.2470548152923584, 'learning_rate': 1.3657749541371446e-07, 'epoch': 4.83}
# {'loss': 0.1895, 'grad_norm': 2.3770031929016113, 'learning_rate': 3.423815394829194e-08, 'epoch': 5.0}
