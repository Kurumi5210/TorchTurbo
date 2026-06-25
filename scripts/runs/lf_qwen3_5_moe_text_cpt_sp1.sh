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
output_dir=/sfs_turbo/zhisan/0604/lf_logs_qwen35/run_qwen3_5_moe_debug_35b_cpt_text_sp1ep8/ckpt

seq_length=$((1024 * 4))
dataset=wikipedia_en
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
# {'loss': 1.8286, 'grad_norm': 3.9378724098205566, 'learning_rate': 1.25e-05, 'epoch': 0.03}
# {'loss': 1.8853, 'grad_norm': 1.6682169437408447, 'learning_rate': 1.2473338601843965e-05, 'epoch': 0.06}
# {'loss': 1.7817, 'grad_norm': 1.564901351928711, 'learning_rate': 1.2393581873024388e-05, 'epoch': 0.09}
# {'loss': 1.8706, 'grad_norm': 1.1936328411102295, 'learning_rate': 1.2261410269830121e-05, 'epoch': 0.12}
# {'loss': 1.7607, 'grad_norm': 1.1829205751419067, 'learning_rate': 1.2077951433777224e-05, 'epoch': 0.15}
# {'loss': 1.7178, 'grad_norm': 1.2041122913360596, 'learning_rate': 1.1844770570969141e-05, 'epoch': 0.18}
# {'loss': 1.6621, 'grad_norm': 1.2232331037521362, 'learning_rate': 1.156385709831009e-05, 'epoch': 0.21}
# {'loss': 1.6821, 'grad_norm': 1.1875028610229492, 'learning_rate': 1.12376076705015e-05, 'epoch': 0.24}
# {'loss': 1.6709, 'grad_norm': 1.1519207954406738, 'learning_rate': 1.086880573262912e-05, 'epoch': 0.26}
# {'loss': 1.79, 'grad_norm': 1.102183222770691, 'learning_rate': 1.0460597772790983e-05, 'epoch': 0.29}
# {'loss': 1.7705, 'grad_norm': 1.396924614906311, 'learning_rate': 1.0016466477370354e-05, 'epoch': 0.32}
# {'loss': 1.7192, 'grad_norm': 1.1549383401870728, 'learning_rate': 9.540201017983475e-06, 'epoch': 0.35}
# {'loss': 1.7349, 'grad_norm': 1.571650743484497, 'learning_rate': 9.035864723603366e-06, 'epoch': 0.38}
# {'loss': 1.77, 'grad_norm': 1.1090632677078247, 'learning_rate': 8.507760413669708e-06, 'epoch': 0.41}
# {'loss': 1.8145, 'grad_norm': 1.074083924293518, 'learning_rate': 7.960393687950519e-06, 'epoch': 0.44}
# {'loss': 1.7207, 'grad_norm': 1.124229073524475, 'learning_rate': 7.398434486353566e-06, 'epoch': 0.47}
# {'loss': 1.6431, 'grad_norm': 1.126220941543579, 'learning_rate': 6.826677246645638e-06, 'epoch': 0.5}
# {'loss': 1.6582, 'grad_norm': 0.98422771692276, 'learning_rate': 6.25e-06, 'epoch': 0.53}
# {'loss': 1.6973, 'grad_norm': 1.061273217201233, 'learning_rate': 5.673322753354363e-06, 'epoch': 0.56}
# {'loss': 1.791, 'grad_norm': 1.088426947593689, 'learning_rate': 5.1015655136464345e-06, 'epoch': 0.59}
# {'loss': 1.6602, 'grad_norm': 1.1443241834640503, 'learning_rate': 4.5396063120494825e-06, 'epoch': 0.62}
# {'loss': 1.7441, 'grad_norm': 1.0435595512390137, 'learning_rate': 3.9922395863302945e-06, 'epoch': 0.65}
# {'loss': 1.7661, 'grad_norm': 1.0463144779205322, 'learning_rate': 3.4641352763966364e-06, 'epoch': 0.68}
# {'loss': 1.6528, 'grad_norm': 1.0044201612472534, 'learning_rate': 2.9597989820165255e-06, 'epoch': 0.71}
# {'loss': 1.7412, 'grad_norm': 0.9787085056304932, 'learning_rate': 2.483533522629648e-06, 'epoch': 0.74}
# {'loss': 1.7769, 'grad_norm': 1.002127766609192, 'learning_rate': 2.0394022272090177e-06, 'epoch': 0.76}
# {'loss': 1.7471, 'grad_norm': 1.0355020761489868, 'learning_rate': 1.6311942673708814e-06, 'epoch': 0.79}
# {'loss': 1.8452, 'grad_norm': 1.0185641050338745, 'learning_rate': 1.2623923294985032e-06, 'epoch': 0.82}
# {'loss': 1.8057, 'grad_norm': 1.0172268152236938, 'learning_rate': 9.361429016899126e-07, 'epoch': 0.85}
# {'loss': 1.7388, 'grad_norm': 1.027198314666748, 'learning_rate': 6.552294290308611e-07, 'epoch': 0.88}
# {'loss': 1.3505, 'grad_norm': 1.0402350425720215, 'learning_rate': 4.220485662227769e-07, 'epoch': 0.91}
# {'loss': 1.7808, 'grad_norm': 1.0669480562210083, 'learning_rate': 2.38589730169881e-07, 'epoch': 0.94}
# {'loss': 1.7085, 'grad_norm': 0.9537035226821899, 'learning_rate': 1.0641812697561385e-07, 'epoch': 0.97}
# {'loss': 1.7192, 'grad_norm': 1.072786808013916, 'learning_rate': 2.6661398156034577e-08, 'epoch': 1.0}
