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
output_dir=/sfs_turbo/zhisan/0604/lf_logs_qwen35/run_qwen3_5_moe_mllm_cpt_mtp_sp2ep8/ckpt

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
    --tt_sp_size 2 \
    --tt_ep_size 8 \
    --tt_enable_mtp true \
    --tt_mtp_loss_scaling_factor 0.1 \
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
# {'loss': 3.0964, 'grad_norm': 176.85333251953125, 'learning_rate': 1.25e-05, 'mtp_loss/0': 2.34375, 'mtp_loss/avg': 2.34375, 'mtp_loss/main': 1.96875, 'mtp_loss/scaled': 0.234375, 'epoch': 0.04}
# {'loss': 3.561, 'grad_norm': 32.193809509277344, 'learning_rate': 1.2465761846051709e-05, 'mtp_loss/0': 3.53125, 'mtp_loss/avg': 3.53125, 'mtp_loss/main': 3.03125, 'mtp_loss/scaled': 0.353516, 'epoch': 0.07}
# {'loss': 3.7021, 'grad_norm': 29.666162490844727, 'learning_rate': 1.2363422504586286e-05, 'mtp_loss/0': 4.25, 'mtp_loss/avg': 4.25, 'mtp_loss/main': 3.796875, 'mtp_loss/scaled': 0.425781, 'epoch': 0.11}
# {'loss': 3.0977, 'grad_norm': 12.142522811889648, 'learning_rate': 1.219410322684471e-05, 'mtp_loss/0': 2.921875, 'mtp_loss/avg': 2.921875, 'mtp_loss/main': 2.53125, 'mtp_loss/scaled': 0.292969, 'epoch': 0.15}
# {'loss': 2.3577, 'grad_norm': 14.845982551574707, 'learning_rate': 1.1959659110266256e-05, 'mtp_loss/0': 2.5, 'mtp_loss/avg': 2.5, 'mtp_loss/main': 2.234375, 'mtp_loss/scaled': 0.25, 'epoch': 0.19}
# {'loss': 2.8616, 'grad_norm': 10.678339958190918, 'learning_rate': 1.1662658773652743e-05, 'mtp_loss/0': 0.941406, 'mtp_loss/avg': 0.941406, 'mtp_loss/main': 0.863281, 'mtp_loss/scaled': 0.094238, 'epoch': 0.22}
# {'loss': 2.9253, 'grad_norm': 10.687296867370605, 'learning_rate': 1.1306356214843423e-05, 'mtp_loss/0': 3.359375, 'mtp_loss/avg': 3.359375, 'mtp_loss/main': 2.953125, 'mtp_loss/scaled': 0.335938, 'epoch': 0.26}
# {'loss': 3.0723, 'grad_norm': 27.919029235839844, 'learning_rate': 1.0894655159233715e-05, 'mtp_loss/0': 3.265625, 'mtp_loss/avg': 3.265625, 'mtp_loss/main': 2.859375, 'mtp_loss/scaled': 0.326172, 'epoch': 0.3}
# {'loss': 2.1702, 'grad_norm': 204.32931518554688, 'learning_rate': 1.0432066289742864e-05, 'mtp_loss/0': 2.578125, 'mtp_loss/avg': 2.578125, 'mtp_loss/main': 2.25, 'mtp_loss/scaled': 0.257812, 'epoch': 0.33}
# {'loss': 2.4946, 'grad_norm': 73.35203552246094, 'learning_rate': 9.923657826827957e-06, 'mtp_loss/0': 2.390625, 'mtp_loss/avg': 2.390625, 'mtp_loss/main': 2.078125, 'mtp_loss/scaled': 0.239258, 'epoch': 0.37}
# {'loss': 2.9097, 'grad_norm': 24.924415588378906, 'learning_rate': 9.375000000000001e-06, 'mtp_loss/0': 3.34375, 'mtp_loss/avg': 3.34375, 'mtp_loss/main': 2.953125, 'mtp_loss/scaled': 0.333984, 'epoch': 0.41}
# {'loss': 2.6543, 'grad_norm': 25.38645362854004, 'learning_rate': 8.792104019223753e-06, 'mtp_loss/0': 2.28125, 'mtp_loss/avg': 2.28125, 'mtp_loss/main': 1.921875, 'mtp_loss/scaled': 0.228516, 'epoch': 0.44}
# {'loss': 2.7515, 'grad_norm': 16.18285369873047, 'learning_rate': 8.181356214843423e-06, 'mtp_loss/0': 3.09375, 'mtp_loss/avg': 3.09375, 'mtp_loss/main': 2.65625, 'mtp_loss/scaled': 0.308594, 'epoch': 0.48}
# {'loss': 3.0547, 'grad_norm': 8.166523933410645, 'learning_rate': 7.5494480676109954e-06, 'mtp_loss/0': 4.09375, 'mtp_loss/avg': 4.09375, 'mtp_loss/main': 3.578125, 'mtp_loss/scaled': 0.410156, 'epoch': 0.52}
# {'loss': 3.2051, 'grad_norm': 11.009378433227539, 'learning_rate': 6.903302895422835e-06, 'mtp_loss/0': 3.40625, 'mtp_loss/avg': 3.40625, 'mtp_loss/main': 2.984375, 'mtp_loss/scaled': 0.339844, 'epoch': 0.56}
# {'loss': 3.0269, 'grad_norm': 8.557579040527344, 'learning_rate': 6.25e-06, 'mtp_loss/0': 2.59375, 'mtp_loss/avg': 2.59375, 'mtp_loss/main': 2.234375, 'mtp_loss/scaled': 0.259766, 'epoch': 0.59}
# {'loss': 3.0107, 'grad_norm': 9.502875328063965, 'learning_rate': 5.596697104577167e-06, 'mtp_loss/0': 1.484375, 'mtp_loss/avg': 1.484375, 'mtp_loss/main': 1.304688, 'mtp_loss/scaled': 0.148438, 'epoch': 0.63}
# {'loss': 3.1641, 'grad_norm': 32.63252639770508, 'learning_rate': 4.950551932389005e-06, 'mtp_loss/0': 3.671875, 'mtp_loss/avg': 3.671875, 'mtp_loss/main': 3.359375, 'mtp_loss/scaled': 0.367188, 'epoch': 0.67}
# {'loss': 2.6384, 'grad_norm': 55.976722717285156, 'learning_rate': 4.3186437851565795e-06, 'mtp_loss/0': 2.953125, 'mtp_loss/avg': 2.953125, 'mtp_loss/main': 2.703125, 'mtp_loss/scaled': 0.294922, 'epoch': 0.7}
# {'loss': 2.6733, 'grad_norm': 18.97660255432129, 'learning_rate': 3.7078959807762497e-06, 'mtp_loss/0': 1.632812, 'mtp_loss/avg': 1.632812, 'mtp_loss/main': 1.429688, 'mtp_loss/scaled': 0.163086, 'epoch': 0.74}
# {'loss': 2.5549, 'grad_norm': 33.94186019897461, 'learning_rate': 3.1250000000000014e-06, 'mtp_loss/0': 2.703125, 'mtp_loss/avg': 2.703125, 'mtp_loss/main': 2.359375, 'mtp_loss/scaled': 0.269531, 'epoch': 0.78}
# {'loss': 3.1445, 'grad_norm': 9.352150917053223, 'learning_rate': 2.5763421731720436e-06, 'mtp_loss/0': 3.625, 'mtp_loss/avg': 3.625, 'mtp_loss/main': 3.15625, 'mtp_loss/scaled': 0.363281, 'epoch': 0.81}
# {'loss': 2.1875, 'grad_norm': 7.265544891357422, 'learning_rate': 2.0679337102571383e-06, 'mtp_loss/0': 1.203125, 'mtp_loss/avg': 1.203125, 'mtp_loss/main': 1.070312, 'mtp_loss/scaled': 0.120117, 'epoch': 0.85}
# {'loss': 2.6616, 'grad_norm': 8.709660530090332, 'learning_rate': 1.6053448407662854e-06, 'mtp_loss/0': 2.5625, 'mtp_loss/avg': 2.5625, 'mtp_loss/main': 2.078125, 'mtp_loss/scaled': 0.255859, 'epoch': 0.89}
# {'loss': 2.6299, 'grad_norm': 21.3804874420166, 'learning_rate': 1.1936437851565791e-06, 'mtp_loss/0': 1.867188, 'mtp_loss/avg': 1.867188, 'mtp_loss/main': 1.671875, 'mtp_loss/scaled': 0.186523, 'epoch': 0.93}
# {'loss': 2.5713, 'grad_norm': 5.898529529571533, 'learning_rate': 8.373412263472582e-07, 'mtp_loss/0': 3.875, 'mtp_loss/avg': 3.875, 'mtp_loss/main': 3.21875, 'mtp_loss/scaled': 0.386719, 'epoch': 0.96}
# {'loss': 2.3638, 'grad_norm': 11.442935943603516, 'learning_rate': 5.403408897337439e-07, 'mtp_loss/0': 3.96875, 'mtp_loss/avg': 3.96875, 'mtp_loss/main': 3.546875, 'mtp_loss/scaled': 0.396484, 'epoch': 1.0}
# {'loss': 2.2734, 'grad_norm': 33.56008529663086, 'learning_rate': 3.0589677315529047e-07, 'mtp_loss/0': 1.695312, 'mtp_loss/avg': 1.695312, 'mtp_loss/main': 1.4375, 'mtp_loss/scaled': 0.169922, 'epoch': 1.04}
# {'loss': 2.3394, 'grad_norm': 14.412690162658691, 'learning_rate': 1.3657749541371446e-07, 'mtp_loss/0': 2.8125, 'mtp_loss/avg': 2.8125, 'mtp_loss/main': 2.3125, 'mtp_loss/scaled': 0.28125, 'epoch': 1.07}
# {'loss': 2.4756, 'grad_norm': 14.115264892578125, 'learning_rate': 3.423815394829194e-08, 'mtp_loss/0': 2.9375, 'mtp_loss/avg': 2.9375, 'mtp_loss/main': 2.34375, 'mtp_loss/scaled': 0.292969, 'epoch': 1.11}

# sp1 4k [Freeze vision encoder]
# {'loss': 3.0964, 'grad_norm': 45.54030227661133, 'learning_rate': 1.25e-05, 'mtp_loss/0': 2.34375, 'mtp_loss/avg': 2.34375, 'mtp_loss/main': 1.96875, 'mtp_loss/scaled': 0.234375, 'epoch': 0.04}
# {'loss': 3.561, 'grad_norm': 6.506494522094727, 'learning_rate': 1.2465761846051709e-05, 'mtp_loss/0': 3.53125, 'mtp_loss/avg': 3.53125, 'mtp_loss/main': 3.03125, 'mtp_loss/scaled': 0.353516, 'epoch': 0.07}
# {'loss': 3.7041, 'grad_norm': 7.053343296051025, 'learning_rate': 1.2363422504586286e-05, 'mtp_loss/0': 4.25, 'mtp_loss/avg': 4.25, 'mtp_loss/main': 3.796875, 'mtp_loss/scaled': 0.425781, 'epoch': 0.11}
# {'loss': 3.0977, 'grad_norm': 5.4920525550842285, 'learning_rate': 1.219410322684471e-05, 'mtp_loss/0': 2.921875, 'mtp_loss/avg': 2.921875, 'mtp_loss/main': 2.53125, 'mtp_loss/scaled': 0.292969, 'epoch': 0.15}
# {'loss': 2.3616, 'grad_norm': 4.993106365203857, 'learning_rate': 1.1959659110266256e-05, 'mtp_loss/0': 2.515625, 'mtp_loss/avg': 2.515625, 'mtp_loss/main': 2.234375, 'mtp_loss/scaled': 0.251953, 'epoch': 0.19}
# {'loss': 2.8635, 'grad_norm': 3.593689203262329, 'learning_rate': 1.1662658773652743e-05, 'mtp_loss/0': 0.941406, 'mtp_loss/avg': 0.941406, 'mtp_loss/main': 0.863281, 'mtp_loss/scaled': 0.094238, 'epoch': 0.22}
# {'loss': 2.9302, 'grad_norm': 3.8152554035186768, 'learning_rate': 1.1306356214843423e-05, 'mtp_loss/0': 3.390625, 'mtp_loss/avg': 3.390625, 'mtp_loss/main': 2.953125, 'mtp_loss/scaled': 0.339844, 'epoch': 0.26}
# {'loss': 3.0649, 'grad_norm': 5.181483268737793, 'learning_rate': 1.0894655159233715e-05, 'mtp_loss/0': 3.265625, 'mtp_loss/avg': 3.265625, 'mtp_loss/main': 2.84375, 'mtp_loss/scaled': 0.326172, 'epoch': 0.3}
# {'loss': 2.1704, 'grad_norm': 3.004561424255371, 'learning_rate': 1.0432066289742864e-05, 'mtp_loss/0': 2.578125, 'mtp_loss/avg': 2.578125, 'mtp_loss/main': 2.25, 'mtp_loss/scaled': 0.257812, 'epoch': 0.33}
# {'loss': 2.478, 'grad_norm': 3.026089668273926, 'learning_rate': 9.923657826827957e-06, 'mtp_loss/0': 2.359375, 'mtp_loss/avg': 2.359375, 'mtp_loss/main': 2.078125, 'mtp_loss/scaled': 0.236328, 'epoch': 0.37}
# {'loss': 2.8701, 'grad_norm': 6.433638572692871, 'learning_rate': 9.375000000000001e-06, 'mtp_loss/0': 3.328125, 'mtp_loss/avg': 3.328125, 'mtp_loss/main': 2.90625, 'mtp_loss/scaled': 0.332031, 'epoch': 0.41}
# {'loss': 2.6064, 'grad_norm': 4.699110984802246, 'learning_rate': 8.792104019223753e-06, 'mtp_loss/0': 2.234375, 'mtp_loss/avg': 2.234375, 'mtp_loss/main': 1.90625, 'mtp_loss/scaled': 0.223633, 'epoch': 0.44}
# {'loss': 2.7222, 'grad_norm': 8.642070770263672, 'learning_rate': 8.181356214843423e-06, 'mtp_loss/0': 3.046875, 'mtp_loss/avg': 3.046875, 'mtp_loss/main': 2.609375, 'mtp_loss/scaled': 0.304688, 'epoch': 0.48}
# {'loss': 3.0278, 'grad_norm': 3.5964014530181885, 'learning_rate': 7.5494480676109954e-06, 'mtp_loss/0': 4.03125, 'mtp_loss/avg': 4.03125, 'mtp_loss/main': 3.53125, 'mtp_loss/scaled': 0.402344, 'epoch': 0.52}
# {'loss': 3.1831, 'grad_norm': 4.595099925994873, 'learning_rate': 6.903302895422835e-06, 'mtp_loss/0': 3.375, 'mtp_loss/avg': 3.375, 'mtp_loss/main': 2.96875, 'mtp_loss/scaled': 0.337891, 'epoch': 0.56}
# {'loss': 3.0083, 'grad_norm': 3.046936273574829, 'learning_rate': 6.25e-06, 'mtp_loss/0': 2.578125, 'mtp_loss/avg': 2.578125, 'mtp_loss/main': 2.234375, 'mtp_loss/scaled': 0.257812, 'epoch': 0.59}
# {'loss': 2.9956, 'grad_norm': 3.792293071746826, 'learning_rate': 5.596697104577167e-06, 'mtp_loss/0': 1.476562, 'mtp_loss/avg': 1.476562, 'mtp_loss/main': 1.296875, 'mtp_loss/scaled': 0.147461, 'epoch': 0.63}
# {'loss': 3.1323, 'grad_norm': 4.919723987579346, 'learning_rate': 4.950551932389005e-06, 'mtp_loss/0': 3.71875, 'mtp_loss/avg': 3.71875, 'mtp_loss/main': 3.375, 'mtp_loss/scaled': 0.371094, 'epoch': 0.67}
# {'loss': 2.6147, 'grad_norm': 22.923797607421875, 'learning_rate': 4.3186437851565795e-06, 'mtp_loss/0': 3.0, 'mtp_loss/avg': 3.0, 'mtp_loss/main': 2.75, 'mtp_loss/scaled': 0.300781, 'epoch': 0.7}
# {'loss': 2.6548, 'grad_norm': 3.7097465991973877, 'learning_rate': 3.7078959807762497e-06, 'mtp_loss/0': 1.625, 'mtp_loss/avg': 1.625, 'mtp_loss/main': 1.421875, 'mtp_loss/scaled': 0.162109, 'epoch': 0.74}
# {'loss': 2.5247, 'grad_norm': 28.05980682373047, 'learning_rate': 3.1250000000000014e-06, 'mtp_loss/0': 2.671875, 'mtp_loss/avg': 2.671875, 'mtp_loss/main': 2.328125, 'mtp_loss/scaled': 0.267578, 'epoch': 0.78}
# {'loss': 3.126, 'grad_norm': 3.705904960632324, 'learning_rate': 2.5763421731720436e-06, 'mtp_loss/0': 3.5625, 'mtp_loss/avg': 3.5625, 'mtp_loss/main': 3.09375, 'mtp_loss/scaled': 0.355469, 'epoch': 0.81}
# {'loss': 2.1746, 'grad_norm': 3.6643052101135254, 'learning_rate': 2.0679337102571383e-06, 'mtp_loss/0': 1.195312, 'mtp_loss/avg': 1.195312, 'mtp_loss/main': 1.054688, 'mtp_loss/scaled': 0.119629, 'epoch': 0.85}
# {'loss': 2.6548, 'grad_norm': 3.24611496925354, 'learning_rate': 1.6053448407662854e-06, 'mtp_loss/0': 2.5625, 'mtp_loss/avg': 2.5625, 'mtp_loss/main': 2.078125, 'mtp_loss/scaled': 0.255859, 'epoch': 0.89}
# {'loss': 2.5723, 'grad_norm': 3.992034912109375, 'learning_rate': 1.1936437851565791e-06, 'mtp_loss/0': 1.84375, 'mtp_loss/avg': 1.84375, 'mtp_loss/main': 1.65625, 'mtp_loss/scaled': 0.18457, 'epoch': 0.93}
# {'loss': 2.5586, 'grad_norm': 2.896852731704712, 'learning_rate': 8.373412263472582e-07, 'mtp_loss/0': 3.859375, 'mtp_loss/avg': 3.859375, 'mtp_loss/main': 3.203125, 'mtp_loss/scaled': 0.386719, 'epoch': 0.96}
# {'loss': 2.3127, 'grad_norm': 3.2903099060058594, 'learning_rate': 5.403408897337439e-07, 'mtp_loss/0': 3.953125, 'mtp_loss/avg': 3.953125, 'mtp_loss/main': 3.515625, 'mtp_loss/scaled': 0.394531, 'epoch': 1.0}
# {'loss': 2.22, 'grad_norm': 3.519047260284424, 'learning_rate': 3.0589677315529047e-07, 'mtp_loss/0': 1.640625, 'mtp_loss/avg': 1.640625, 'mtp_loss/main': 1.382812, 'mtp_loss/scaled': 0.164062, 'epoch': 1.04}
# {'loss': 2.2148, 'grad_norm': 6.77000617980957, 'learning_rate': 1.3657749541371446e-07, 'mtp_loss/0': 2.6875, 'mtp_loss/avg': 2.6875, 'mtp_loss/main': 2.21875, 'mtp_loss/scaled': 0.269531, 'epoch': 1.07}
# {'loss': 2.3667, 'grad_norm': 6.408984184265137, 'learning_rate': 3.423815394829194e-08, 'mtp_loss/0': 2.734375, 'mtp_loss/avg': 2.734375, 'mtp_loss/main': 2.171875, 'mtp_loss/scaled': 0.273438, 'epoch': 1.11}

# sp2 4k [Freeze vision encoder]
# {'loss': 3.0977, 'grad_norm': 40.74612808227539, 'learning_rate': 1.25e-05, 'mtp_loss/0': 1.421875, 'mtp_loss/avg': 1.421875, 'mtp_loss/main': 1.398438, 'mtp_loss/scaled': 0.14209, 'epoch': 0.04}
# {'loss': 3.5566, 'grad_norm': 6.453285217285156, 'learning_rate': 1.2465761846051709e-05, 'mtp_loss/0': 1.609375, 'mtp_loss/avg': 1.609375, 'mtp_loss/main': 1.375, 'mtp_loss/scaled': 0.161133, 'epoch': 0.07}
# {'loss': 3.6973, 'grad_norm': 7.679423809051514, 'learning_rate': 1.2363422504586286e-05, 'mtp_loss/0': 1.769531, 'mtp_loss/avg': 1.769531, 'mtp_loss/main': 1.5625, 'mtp_loss/scaled': 0.177246, 'epoch': 0.11}
# {'loss': 3.0957, 'grad_norm': 5.712965965270996, 'learning_rate': 1.219410322684471e-05, 'mtp_loss/0': 1.582031, 'mtp_loss/avg': 1.582031, 'mtp_loss/main': 1.398438, 'mtp_loss/scaled': 0.158203, 'epoch': 0.15}
# {'loss': 2.3635, 'grad_norm': 5.044900894165039, 'learning_rate': 1.1959659110266256e-05, 'mtp_loss/0': 1.472656, 'mtp_loss/avg': 1.472656, 'mtp_loss/main': 1.289062, 'mtp_loss/scaled': 0.147461, 'epoch': 0.19}
# {'loss': 2.8635, 'grad_norm': 3.874448299407959, 'learning_rate': 1.1662658773652743e-05, 'mtp_loss/0': 0.375, 'mtp_loss/avg': 0.375, 'mtp_loss/main': 0.34668, 'mtp_loss/scaled': 0.037476, 'epoch': 0.22}
# {'loss': 2.9355, 'grad_norm': 3.891644239425659, 'learning_rate': 1.1306356214843423e-05, 'mtp_loss/0': 1.828125, 'mtp_loss/avg': 1.828125, 'mtp_loss/main': 1.609375, 'mtp_loss/scaled': 0.182617, 'epoch': 0.26}
# {'loss': 3.0601, 'grad_norm': 4.860438346862793, 'learning_rate': 1.0894655159233715e-05, 'mtp_loss/0': 1.746094, 'mtp_loss/avg': 1.746094, 'mtp_loss/main': 1.539062, 'mtp_loss/scaled': 0.174316, 'epoch': 0.3}
# {'loss': 2.1709, 'grad_norm': 2.8910040855407715, 'learning_rate': 1.0432066289742864e-05, 'mtp_loss/0': 1.511719, 'mtp_loss/avg': 1.511719, 'mtp_loss/main': 1.3125, 'mtp_loss/scaled': 0.151367, 'epoch': 0.33}
# {'loss': 2.4731, 'grad_norm': 2.9213380813598633, 'learning_rate': 9.923657826827957e-06, 'mtp_loss/0': 1.511719, 'mtp_loss/avg': 1.511719, 'mtp_loss/main': 1.34375, 'mtp_loss/scaled': 0.151367, 'epoch': 0.37}
# {'loss': 2.8735, 'grad_norm': 3.7023980617523193, 'learning_rate': 9.375000000000001e-06, 'mtp_loss/0': 1.957031, 'mtp_loss/avg': 1.957031, 'mtp_loss/main': 1.742188, 'mtp_loss/scaled': 0.195312, 'epoch': 0.41}
# {'loss': 2.6011, 'grad_norm': 4.773199558258057, 'learning_rate': 8.792104019223753e-06, 'mtp_loss/0': 1.089844, 'mtp_loss/avg': 1.089844, 'mtp_loss/main': 0.908203, 'mtp_loss/scaled': 0.108887, 'epoch': 0.44}
# {'loss': 2.7212, 'grad_norm': 8.066452026367188, 'learning_rate': 8.181356214843423e-06, 'mtp_loss/0': 1.589844, 'mtp_loss/avg': 1.589844, 'mtp_loss/main': 1.386719, 'mtp_loss/scaled': 0.15918, 'epoch': 0.48}
# {'loss': 3.0215, 'grad_norm': 3.4972853660583496, 'learning_rate': 7.5494480676109954e-06, 'mtp_loss/0': 1.839844, 'mtp_loss/avg': 1.839844, 'mtp_loss/main': 1.578125, 'mtp_loss/scaled': 0.183594, 'epoch': 0.52}
# {'loss': 3.1694, 'grad_norm': 4.7954230308532715, 'learning_rate': 6.903302895422835e-06, 'mtp_loss/0': 1.742188, 'mtp_loss/avg': 1.742188, 'mtp_loss/main': 1.511719, 'mtp_loss/scaled': 0.174316, 'epoch': 0.56}
# {'loss': 3.0024, 'grad_norm': 3.0752649307250977, 'learning_rate': 6.25e-06, 'mtp_loss/0': 0.966797, 'mtp_loss/avg': 0.966797, 'mtp_loss/main': 0.801758, 'mtp_loss/scaled': 0.09668, 'epoch': 0.59}
# {'loss': 2.9985, 'grad_norm': 3.801319122314453, 'learning_rate': 5.596697104577167e-06, 'mtp_loss/0': 1.158203, 'mtp_loss/avg': 1.158203, 'mtp_loss/main': 1.007812, 'mtp_loss/scaled': 0.115967, 'epoch': 0.63}
# {'loss': 3.1328, 'grad_norm': 5.073177337646484, 'learning_rate': 4.950551932389005e-06, 'mtp_loss/0': 1.738281, 'mtp_loss/avg': 1.738281, 'mtp_loss/main': 1.582031, 'mtp_loss/scaled': 0.173828, 'epoch': 0.67}
# {'loss': 2.6211, 'grad_norm': 23.00812530517578, 'learning_rate': 4.3186437851565795e-06, 'mtp_loss/0': 1.660156, 'mtp_loss/avg': 1.660156, 'mtp_loss/main': 1.46875, 'mtp_loss/scaled': 0.166016, 'epoch': 0.7}
# {'loss': 2.6531, 'grad_norm': 3.7367310523986816, 'learning_rate': 3.7078959807762497e-06, 'mtp_loss/0': 1.019531, 'mtp_loss/avg': 1.019531, 'mtp_loss/main': 0.898438, 'mtp_loss/scaled': 0.101807, 'epoch': 0.74}
# {'loss': 2.5193, 'grad_norm': 6.340987205505371, 'learning_rate': 3.1250000000000014e-06, 'mtp_loss/0': 1.496094, 'mtp_loss/avg': 1.496094, 'mtp_loss/main': 1.324219, 'mtp_loss/scaled': 0.149902, 'epoch': 0.78}
# {'loss': 3.1182, 'grad_norm': 3.784090042114258, 'learning_rate': 2.5763421731720436e-06, 'mtp_loss/0': 1.773438, 'mtp_loss/avg': 1.773438, 'mtp_loss/main': 1.558594, 'mtp_loss/scaled': 0.177246, 'epoch': 0.81}
# {'loss': 2.1748, 'grad_norm': 3.784266233444214, 'learning_rate': 2.0679337102571383e-06, 'mtp_loss/0': 0.451172, 'mtp_loss/avg': 0.451172, 'mtp_loss/main': 0.398438, 'mtp_loss/scaled': 0.045105, 'epoch': 0.85}
# {'loss': 2.6523, 'grad_norm': 3.2591960430145264, 'learning_rate': 1.6053448407662854e-06, 'mtp_loss/0': 1.382812, 'mtp_loss/avg': 1.382812, 'mtp_loss/main': 1.179688, 'mtp_loss/scaled': 0.138184, 'epoch': 0.89}
# {'loss': 2.5728, 'grad_norm': 4.047006607055664, 'learning_rate': 1.1936437851565791e-06, 'mtp_loss/0': 1.429688, 'mtp_loss/avg': 1.429688, 'mtp_loss/main': 1.238281, 'mtp_loss/scaled': 0.143066, 'epoch': 0.93}
# {'loss': 2.5571, 'grad_norm': 2.842097282409668, 'learning_rate': 8.373412263472582e-07, 'mtp_loss/0': 1.457031, 'mtp_loss/avg': 1.457031, 'mtp_loss/main': 1.236328, 'mtp_loss/scaled': 0.145996, 'epoch': 0.96}
# {'loss': 2.2954, 'grad_norm': 3.643988847732544, 'learning_rate': 5.403408897337439e-07, 'mtp_loss/0': 1.392578, 'mtp_loss/avg': 1.392578, 'mtp_loss/main': 1.220703, 'mtp_loss/scaled': 0.13916, 'epoch': 1.0}
# {'loss': 2.1985, 'grad_norm': 3.7102978229522705, 'learning_rate': 3.0589677315529047e-07, 'mtp_loss/0': 0.966797, 'mtp_loss/avg': 0.966797, 'mtp_loss/main': 0.830078, 'mtp_loss/scaled': 0.09668, 'epoch': 1.04}
# {'loss': 2.2119, 'grad_norm': 7.367356300354004, 'learning_rate': 1.3657749541371446e-07, 'mtp_loss/0': 1.183594, 'mtp_loss/avg': 1.183594, 'mtp_loss/main': 0.953125, 'mtp_loss/scaled': 0.118652, 'epoch': 1.07}
# {'loss': 2.3867, 'grad_norm': 6.349632740020752, 'learning_rate': 3.423815394829194e-08, 'mtp_loss/0': 1.230469, 'mtp_loss/avg': 1.230469, 'mtp_loss/main': 0.980469, 'mtp_loss/scaled': 0.123291, 'epoch': 1.11}
