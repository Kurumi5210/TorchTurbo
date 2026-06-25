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
output_dir=/sfs_turbo/zhisan/0604/lf_logs_qwen35/run_qwen3_5_moe_debug_35b_cpt_mllm_sp1ep8/ckpt

seq_length=$((1024 * 4))
dataset=security
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
    --tt_freeze_vision_tower \
    --ddp_timeout 1800000000 2>&1 | tee -a $log_file

    # --tt_freeze_vision_tower \
    # --tt_freeze_multi_modal_projector \
    # --tt_freeze_language_model \

# sp1 4k [Full training]
# {'loss': 2.7878, 'grad_norm': 167.0164031982422, 'learning_rate': 1.25e-05, 'epoch': 0.04}
# {'loss': 3.1948, 'grad_norm': 27.848358154296875, 'learning_rate': 1.2465761846051709e-05, 'epoch': 0.07}
# {'loss': 3.3311, 'grad_norm': 18.456605911254883, 'learning_rate': 1.2363422504586286e-05, 'epoch': 0.11}
# {'loss': 2.7759, 'grad_norm': 16.09052085876465, 'learning_rate': 1.219410322684471e-05, 'epoch': 0.15}
# {'loss': 2.1172, 'grad_norm': 28.613672256469727, 'learning_rate': 1.1959659110266256e-05, 'epoch': 0.19}
# {'loss': 2.5671, 'grad_norm': 14.379622459411621, 'learning_rate': 1.1662658773652743e-05, 'epoch': 0.22}
# {'loss': 2.6362, 'grad_norm': 9.389266967773438, 'learning_rate': 1.1306356214843423e-05, 'epoch': 0.26}
# {'loss': 2.7461, 'grad_norm': 21.2520751953125, 'learning_rate': 1.0894655159233715e-05, 'epoch': 0.3}
# {'loss': 1.9482, 'grad_norm': 22.284120559692383, 'learning_rate': 1.0432066289742864e-05, 'epoch': 0.33}
# {'loss': 2.2283, 'grad_norm': 5.673595905303955, 'learning_rate': 9.923657826827957e-06, 'epoch': 0.37}
# {'loss': 2.5747, 'grad_norm': 22.02975845336914, 'learning_rate': 9.375000000000001e-06, 'epoch': 0.41}
# {'loss': 2.3413, 'grad_norm': 21.837749481201172, 'learning_rate': 8.792104019223753e-06, 'epoch': 0.44}
# {'loss': 2.4231, 'grad_norm': 13.860767364501953, 'learning_rate': 8.181356214843423e-06, 'epoch': 0.48}
# {'loss': 2.7227, 'grad_norm': 21.858718872070312, 'learning_rate': 7.5494480676109954e-06, 'epoch': 0.52}
# {'loss': 2.8613, 'grad_norm': 14.460535049438477, 'learning_rate': 6.903302895422835e-06, 'epoch': 0.56}
# {'loss': 2.7024, 'grad_norm': 5.51029634475708, 'learning_rate': 6.25e-06, 'epoch': 0.59}
# {'loss': 2.6924, 'grad_norm': 11.799822807312012, 'learning_rate': 5.596697104577167e-06, 'epoch': 0.63}
# {'loss': 2.8125, 'grad_norm': 16.740833282470703, 'learning_rate': 4.950551932389005e-06, 'epoch': 0.67}
# {'loss': 2.3821, 'grad_norm': 31.41050910949707, 'learning_rate': 4.3186437851565795e-06, 'epoch': 0.7}
# {'loss': 2.3821, 'grad_norm': 12.759076118469238, 'learning_rate': 3.7078959807762497e-06, 'epoch': 0.74}
# {'loss': 2.276, 'grad_norm': 85.52666473388672, 'learning_rate': 3.1250000000000014e-06, 'epoch': 0.78}
# {'loss': 2.8154, 'grad_norm': 12.86805534362793, 'learning_rate': 2.5763421731720436e-06, 'epoch': 0.81}
# {'loss': 1.9519, 'grad_norm': 25.197654724121094, 'learning_rate': 2.0679337102571383e-06, 'epoch': 0.85}
# {'loss': 2.3813, 'grad_norm': 8.679388999938965, 'learning_rate': 1.6053448407662854e-06, 'epoch': 0.89}
# {'loss': 2.3413, 'grad_norm': 16.472143173217773, 'learning_rate': 1.1936437851565791e-06, 'epoch': 0.93}
# {'loss': 2.2944, 'grad_norm': 23.745912551879883, 'learning_rate': 8.373412263472582e-07, 'epoch': 0.96}
# {'loss': 2.1044, 'grad_norm': 10.459434509277344, 'learning_rate': 5.403408897337439e-07, 'epoch': 1.0}
# {'loss': 2.0184, 'grad_norm': 7.691360950469971, 'learning_rate': 3.0589677315529047e-07, 'epoch': 1.04}
# {'loss': 2.062, 'grad_norm': 12.313516616821289, 'learning_rate': 1.3657749541371446e-07, 'epoch': 1.07}
# {'loss': 2.0752, 'grad_norm': 12.198158264160156, 'learning_rate': 3.423815394829194e-08, 'epoch': 1.11}

# sp1 4k [Freeze vision encoder]
# {'loss': 2.7878, 'grad_norm': 43.19639205932617, 'learning_rate': 1.25e-05, 'epoch': 0.04}
# {'loss': 3.1958, 'grad_norm': 6.648967742919922, 'learning_rate': 1.2465761846051709e-05, 'epoch': 0.07}
# {'loss': 3.335, 'grad_norm': 7.3032989501953125, 'learning_rate': 1.2363422504586286e-05, 'epoch': 0.11}
# {'loss': 2.7759, 'grad_norm': 5.373886585235596, 'learning_rate': 1.219410322684471e-05, 'epoch': 0.15}
# {'loss': 2.1208, 'grad_norm': 5.0425333976745605, 'learning_rate': 1.1959659110266256e-05, 'epoch': 0.19}
# {'loss': 2.5691, 'grad_norm': 3.4442567825317383, 'learning_rate': 1.1662658773652743e-05, 'epoch': 0.22}
# {'loss': 2.6323, 'grad_norm': 3.7255055904388428, 'learning_rate': 1.1306356214843423e-05, 'epoch': 0.26}
# {'loss': 2.7524, 'grad_norm': 4.533050537109375, 'learning_rate': 1.0894655159233715e-05, 'epoch': 0.3}
# {'loss': 1.948, 'grad_norm': 2.697153329849243, 'learning_rate': 1.0432066289742864e-05, 'epoch': 0.33}
# {'loss': 2.2224, 'grad_norm': 2.674626111984253, 'learning_rate': 9.923657826827957e-06, 'epoch': 0.37}
# {'loss': 2.5786, 'grad_norm': 3.614774465560913, 'learning_rate': 9.375000000000001e-06, 'epoch': 0.41}
# {'loss': 2.3354, 'grad_norm': 4.357975959777832, 'learning_rate': 8.792104019223753e-06, 'epoch': 0.44}
# {'loss': 2.4341, 'grad_norm': 8.289466857910156, 'learning_rate': 8.181356214843423e-06, 'epoch': 0.48}
# {'loss': 2.7144, 'grad_norm': 3.2478864192962646, 'learning_rate': 7.5494480676109954e-06, 'epoch': 0.52}
# {'loss': 2.8555, 'grad_norm': 4.356831073760986, 'learning_rate': 6.903302895422835e-06, 'epoch': 0.56}
# {'loss': 2.6965, 'grad_norm': 2.776939630508423, 'learning_rate': 6.25e-06, 'epoch': 0.59}
# {'loss': 2.6855, 'grad_norm': 3.323093891143799, 'learning_rate': 5.596697104577167e-06, 'epoch': 0.63}
# {'loss': 2.8125, 'grad_norm': 5.064888000488281, 'learning_rate': 4.950551932389005e-06, 'epoch': 0.67}
# {'loss': 2.3455, 'grad_norm': 21.347925186157227, 'learning_rate': 4.3186437851565795e-06, 'epoch': 0.7}
# {'loss': 2.3804, 'grad_norm': 3.7711732387542725, 'learning_rate': 3.7078959807762497e-06, 'epoch': 0.74}
# {'loss': 2.2653, 'grad_norm': 6.370126724243164, 'learning_rate': 3.1250000000000014e-06, 'epoch': 0.78}
# {'loss': 2.8027, 'grad_norm': 3.354085922241211, 'learning_rate': 2.5763421731720436e-06, 'epoch': 0.81}
# {'loss': 1.9561, 'grad_norm': 3.6556506156921387, 'learning_rate': 2.0679337102571383e-06, 'epoch': 0.85}
# {'loss': 2.3813, 'grad_norm': 3.1383843421936035, 'learning_rate': 1.6053448407662854e-06, 'epoch': 0.89}
# {'loss': 2.311, 'grad_norm': 3.903881311416626, 'learning_rate': 1.1936437851565791e-06, 'epoch': 0.93}
# {'loss': 2.2925, 'grad_norm': 2.7255873680114746, 'learning_rate': 8.373412263472582e-07, 'epoch': 0.96}
# {'loss': 2.0675, 'grad_norm': 3.040667772293091, 'learning_rate': 5.403408897337439e-07, 'epoch': 1.0}
# {'loss': 1.9819, 'grad_norm': 3.2622017860412598, 'learning_rate': 3.0589677315529047e-07, 'epoch': 1.04}
# {'loss': 1.9849, 'grad_norm': 6.354218006134033, 'learning_rate': 1.3657749541371446e-07, 'epoch': 1.07}
# {'loss': 2.1299, 'grad_norm': 6.089993000030518, 'learning_rate': 3.423815394829194e-08, 'epoch': 1.11}
