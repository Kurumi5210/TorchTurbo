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
output_dir=/sfs_turbo/zhisan/0604/lf_logs_qwen35/run_qwen3_5_moe_text_cpt_mtp_sp2ep8/ckpt

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
    --tt_sp_size 2 \
    --tt_ep_size 8 \
    --tt_enable_mtp true \
    --tt_mtp_loss_scaling_factor 0.1 \
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
# {'loss': 2.0479, 'grad_norm': 4.363673686981201, 'learning_rate': 1.25e-05, 'mtp_loss/0': 2.265625, 'mtp_loss/avg': 2.265625, 'mtp_loss/main': 1.945312, 'mtp_loss/scaled': 0.226562, 'epoch': 0.03}
# {'loss': 2.1128, 'grad_norm': 1.947240948677063, 'learning_rate': 1.2473338601843965e-05, 'mtp_loss/0': 2.25, 'mtp_loss/avg': 2.25, 'mtp_loss/main': 1.851562, 'mtp_loss/scaled': 0.224609, 'epoch': 0.06}
# {'loss': 1.9917, 'grad_norm': 1.6951688528060913, 'learning_rate': 1.2393581873024388e-05, 'mtp_loss/0': 1.648438, 'mtp_loss/avg': 1.648438, 'mtp_loss/main': 1.40625, 'mtp_loss/scaled': 0.165039, 'epoch': 0.09}
# {'loss': 2.0884, 'grad_norm': 1.3672596216201782, 'learning_rate': 1.2261410269830121e-05, 'mtp_loss/0': 2.140625, 'mtp_loss/avg': 2.140625, 'mtp_loss/main': 1.875, 'mtp_loss/scaled': 0.213867, 'epoch': 0.12}
# {'loss': 1.9692, 'grad_norm': 1.252256989479065, 'learning_rate': 1.2077951433777224e-05, 'mtp_loss/0': 1.226562, 'mtp_loss/avg': 1.226562, 'mtp_loss/main': 1.015625, 'mtp_loss/scaled': 0.122559, 'epoch': 0.15}
# {'loss': 1.9209, 'grad_norm': 1.2684870958328247, 'learning_rate': 1.1844770570969141e-05, 'mtp_loss/0': 2.265625, 'mtp_loss/avg': 2.265625, 'mtp_loss/main': 1.882812, 'mtp_loss/scaled': 0.226562, 'epoch': 0.18}
# {'loss': 1.8584, 'grad_norm': 1.201442003250122, 'learning_rate': 1.156385709831009e-05, 'mtp_loss/0': 2.125, 'mtp_loss/avg': 2.125, 'mtp_loss/main': 1.78125, 'mtp_loss/scaled': 0.212891, 'epoch': 0.21}
# {'loss': 1.8833, 'grad_norm': 1.2558608055114746, 'learning_rate': 1.12376076705015e-05, 'mtp_loss/0': 2.140625, 'mtp_loss/avg': 2.140625, 'mtp_loss/main': 1.851562, 'mtp_loss/scaled': 0.213867, 'epoch': 0.24}
# {'loss': 1.8687, 'grad_norm': 1.2144880294799805, 'learning_rate': 1.086880573262912e-05, 'mtp_loss/0': 2.015625, 'mtp_loss/avg': 2.015625, 'mtp_loss/main': 1.703125, 'mtp_loss/scaled': 0.201172, 'epoch': 0.26}
# {'loss': 1.9985, 'grad_norm': 1.1675916910171509, 'learning_rate': 1.0460597772790983e-05, 'mtp_loss/0': 1.804688, 'mtp_loss/avg': 1.804688, 'mtp_loss/main': 1.546875, 'mtp_loss/scaled': 0.180664, 'epoch': 0.29}
# {'loss': 1.9795, 'grad_norm': 1.3465832471847534, 'learning_rate': 1.0016466477370354e-05, 'mtp_loss/0': 2.375, 'mtp_loss/avg': 2.375, 'mtp_loss/main': 2.03125, 'mtp_loss/scaled': 0.237305, 'epoch': 0.32}
# {'loss': 1.9209, 'grad_norm': 1.4752854108810425, 'learning_rate': 9.540201017983475e-06, 'mtp_loss/0': 1.96875, 'mtp_loss/avg': 1.96875, 'mtp_loss/main': 1.632812, 'mtp_loss/scaled': 0.197266, 'epoch': 0.35}
# {'loss': 1.9497, 'grad_norm': 1.714293360710144, 'learning_rate': 9.035864723603366e-06, 'mtp_loss/0': 1.148438, 'mtp_loss/avg': 1.148438, 'mtp_loss/main': 0.984375, 'mtp_loss/scaled': 0.114746, 'epoch': 0.38}
# {'loss': 1.978, 'grad_norm': 1.1726704835891724, 'learning_rate': 8.507760413669708e-06, 'mtp_loss/0': 2.0, 'mtp_loss/avg': 2.0, 'mtp_loss/main': 1.703125, 'mtp_loss/scaled': 0.200195, 'epoch': 0.41}
# {'loss': 2.0278, 'grad_norm': 1.1115986108779907, 'learning_rate': 7.960393687950519e-06, 'mtp_loss/0': 1.992188, 'mtp_loss/avg': 1.992188, 'mtp_loss/main': 1.6875, 'mtp_loss/scaled': 0.199219, 'epoch': 0.44}
# {'loss': 1.9199, 'grad_norm': 1.1405376195907593, 'learning_rate': 7.398434486353566e-06, 'mtp_loss/0': 2.015625, 'mtp_loss/avg': 2.015625, 'mtp_loss/main': 1.734375, 'mtp_loss/scaled': 0.201172, 'epoch': 0.47}
# {'loss': 1.8394, 'grad_norm': 1.1299941539764404, 'learning_rate': 6.826677246645638e-06, 'mtp_loss/0': 1.921875, 'mtp_loss/avg': 1.921875, 'mtp_loss/main': 1.585938, 'mtp_loss/scaled': 0.192383, 'epoch': 0.5}
# {'loss': 1.8535, 'grad_norm': 1.059196949005127, 'learning_rate': 6.25e-06, 'mtp_loss/0': 2.015625, 'mtp_loss/avg': 2.015625, 'mtp_loss/main': 1.710938, 'mtp_loss/scaled': 0.201172, 'epoch': 0.53}
# {'loss': 1.8955, 'grad_norm': 1.0942983627319336, 'learning_rate': 5.673322753354363e-06, 'mtp_loss/0': 1.875, 'mtp_loss/avg': 1.875, 'mtp_loss/main': 1.632812, 'mtp_loss/scaled': 0.1875, 'epoch': 0.56}
# {'loss': 2.002, 'grad_norm': 1.138553500175476, 'learning_rate': 5.1015655136464345e-06, 'mtp_loss/0': 1.921875, 'mtp_loss/avg': 1.921875, 'mtp_loss/main': 1.617188, 'mtp_loss/scaled': 0.192383, 'epoch': 0.59}
# {'loss': 1.8545, 'grad_norm': 1.2308863401412964, 'learning_rate': 4.5396063120494825e-06, 'mtp_loss/0': 1.875, 'mtp_loss/avg': 1.875, 'mtp_loss/main': 1.625, 'mtp_loss/scaled': 0.1875, 'epoch': 0.62}
# {'loss': 1.9478, 'grad_norm': 1.058465600013733, 'learning_rate': 3.9922395863302945e-06, 'mtp_loss/0': 1.859375, 'mtp_loss/avg': 1.859375, 'mtp_loss/main': 1.59375, 'mtp_loss/scaled': 0.185547, 'epoch': 0.65}
# {'loss': 1.9731, 'grad_norm': 1.1250320672988892, 'learning_rate': 3.4641352763966364e-06, 'mtp_loss/0': 1.851562, 'mtp_loss/avg': 1.851562, 'mtp_loss/main': 1.601562, 'mtp_loss/scaled': 0.185547, 'epoch': 0.68}
# {'loss': 1.8447, 'grad_norm': 1.0872479677200317, 'learning_rate': 2.9597989820165255e-06, 'mtp_loss/0': 2.328125, 'mtp_loss/avg': 2.328125, 'mtp_loss/main': 1.96875, 'mtp_loss/scaled': 0.232422, 'epoch': 0.71}
# {'loss': 1.9468, 'grad_norm': 1.059775710105896, 'learning_rate': 2.483533522629648e-06, 'mtp_loss/0': 2.1875, 'mtp_loss/avg': 2.1875, 'mtp_loss/main': 1.890625, 'mtp_loss/scaled': 0.21875, 'epoch': 0.74}
# {'loss': 1.9834, 'grad_norm': 1.069171667098999, 'learning_rate': 2.0394022272090177e-06, 'mtp_loss/0': 2.203125, 'mtp_loss/avg': 2.203125, 'mtp_loss/main': 1.960938, 'mtp_loss/scaled': 0.220703, 'epoch': 0.76}
# {'loss': 1.9521, 'grad_norm': 1.111958622932434, 'learning_rate': 1.6311942673708814e-06, 'mtp_loss/0': 1.742188, 'mtp_loss/avg': 1.742188, 'mtp_loss/main': 1.492188, 'mtp_loss/scaled': 0.173828, 'epoch': 0.79}
# {'loss': 2.0596, 'grad_norm': 1.2121622562408447, 'learning_rate': 1.2623923294985032e-06, 'mtp_loss/0': 2.484375, 'mtp_loss/avg': 2.484375, 'mtp_loss/main': 2.171875, 'mtp_loss/scaled': 0.248047, 'epoch': 0.82}
# {'loss': 2.0166, 'grad_norm': 1.0939844846725464, 'learning_rate': 9.361429016899126e-07, 'mtp_loss/0': 2.0625, 'mtp_loss/avg': 2.0625, 'mtp_loss/main': 1.742188, 'mtp_loss/scaled': 0.206055, 'epoch': 0.85}
# {'loss': 1.9414, 'grad_norm': 1.124304175376892, 'learning_rate': 6.552294290308611e-07, 'mtp_loss/0': 2.421875, 'mtp_loss/avg': 2.421875, 'mtp_loss/main': 2.15625, 'mtp_loss/scaled': 0.242188, 'epoch': 0.88}
# {'loss': 1.5094, 'grad_norm': 1.9408479928970337, 'learning_rate': 4.220485662227769e-07, 'mtp_loss/0': 1.882812, 'mtp_loss/avg': 1.882812, 'mtp_loss/main': 1.59375, 'mtp_loss/scaled': 0.188477, 'epoch': 0.91}
# {'loss': 1.9878, 'grad_norm': 1.14474618434906, 'learning_rate': 2.38589730169881e-07, 'mtp_loss/0': 1.75, 'mtp_loss/avg': 1.75, 'mtp_loss/main': 1.46875, 'mtp_loss/scaled': 0.174805, 'epoch': 0.94}
# {'loss': 1.9097, 'grad_norm': 1.0125349760055542, 'learning_rate': 1.0641812697561385e-07, 'mtp_loss/0': 2.03125, 'mtp_loss/avg': 2.03125, 'mtp_loss/main': 1.679688, 'mtp_loss/scaled': 0.203125, 'epoch': 0.97}
# {'loss': 1.9229, 'grad_norm': 1.1170684099197388, 'learning_rate': 2.6661398156034577e-08, 'mtp_loss/0': 2.203125, 'mtp_loss/avg': 2.203125, 'mtp_loss/main': 1.890625, 'mtp_loss/scaled': 0.220703, 'epoch': 1.0}

# sp2 4k
# {'loss': 2.0459, 'grad_norm': 4.0908355712890625, 'learning_rate': 1.25e-05, 'mtp_loss/0': 1.117188, 'mtp_loss/avg': 1.117188, 'mtp_loss/main': 0.953125, 'mtp_loss/scaled': 0.111816, 'epoch': 0.03}
# {'loss': 2.1128, 'grad_norm': 1.8382312059402466, 'learning_rate': 1.2473338601843965e-05, 'mtp_loss/0': 1.101562, 'mtp_loss/avg': 1.101562, 'mtp_loss/main': 0.914062, 'mtp_loss/scaled': 0.110107, 'epoch': 0.06}
# {'loss': 1.9922, 'grad_norm': 1.7362375259399414, 'learning_rate': 1.2393581873024388e-05, 'mtp_loss/0': 0.962891, 'mtp_loss/avg': 0.962891, 'mtp_loss/main': 0.820312, 'mtp_loss/scaled': 0.096436, 'epoch': 0.09}
# {'loss': 2.0908, 'grad_norm': 1.3031859397888184, 'learning_rate': 1.2261410269830121e-05, 'mtp_loss/0': 1.117188, 'mtp_loss/avg': 1.117188, 'mtp_loss/main': 0.964844, 'mtp_loss/scaled': 0.111816, 'epoch': 0.12}
# {'loss': 1.9692, 'grad_norm': 1.251407265663147, 'learning_rate': 1.2077951433777224e-05, 'mtp_loss/0': 0.886719, 'mtp_loss/avg': 0.886719, 'mtp_loss/main': 0.753906, 'mtp_loss/scaled': 0.088623, 'epoch': 0.15}
# {'loss': 1.9214, 'grad_norm': 1.2678889036178589, 'learning_rate': 1.1844770570969141e-05, 'mtp_loss/0': 1.042969, 'mtp_loss/avg': 1.042969, 'mtp_loss/main': 0.886719, 'mtp_loss/scaled': 0.104248, 'epoch': 0.18}
# {'loss': 1.856, 'grad_norm': 1.2052472829818726, 'learning_rate': 1.156385709831009e-05, 'mtp_loss/0': 0.941406, 'mtp_loss/avg': 0.941406, 'mtp_loss/main': 0.785156, 'mtp_loss/scaled': 0.094238, 'epoch': 0.21}
# {'loss': 1.8828, 'grad_norm': 1.2583389282226562, 'learning_rate': 1.12376076705015e-05, 'mtp_loss/0': 0.994141, 'mtp_loss/avg': 0.994141, 'mtp_loss/main': 0.855469, 'mtp_loss/scaled': 0.099365, 'epoch': 0.24}
# {'loss': 1.8701, 'grad_norm': 1.2126132249832153, 'learning_rate': 1.086880573262912e-05, 'mtp_loss/0': 1.003906, 'mtp_loss/avg': 1.003906, 'mtp_loss/main': 0.847656, 'mtp_loss/scaled': 0.100342, 'epoch': 0.26}
# {'loss': 2.0005, 'grad_norm': 1.1612530946731567, 'learning_rate': 1.0460597772790983e-05, 'mtp_loss/0': 0.90625, 'mtp_loss/avg': 0.90625, 'mtp_loss/main': 0.763672, 'mtp_loss/scaled': 0.09082, 'epoch': 0.29}
# {'loss': 1.978, 'grad_norm': 1.3127413988113403, 'learning_rate': 1.0016466477370354e-05, 'mtp_loss/0': 1.199219, 'mtp_loss/avg': 1.199219, 'mtp_loss/main': 1.015625, 'mtp_loss/scaled': 0.119873, 'epoch': 0.32}
# {'loss': 1.9204, 'grad_norm': 1.2249350547790527, 'learning_rate': 9.540201017983475e-06, 'mtp_loss/0': 0.986328, 'mtp_loss/avg': 0.986328, 'mtp_loss/main': 0.841797, 'mtp_loss/scaled': 0.098633, 'epoch': 0.35}
# {'loss': 1.9497, 'grad_norm': 1.6686650514602661, 'learning_rate': 9.035864723603366e-06, 'mtp_loss/0': 0.597656, 'mtp_loss/avg': 0.597656, 'mtp_loss/main': 0.512695, 'mtp_loss/scaled': 0.059814, 'epoch': 0.38}
# {'loss': 1.9775, 'grad_norm': 1.1757371425628662, 'learning_rate': 8.507760413669708e-06, 'mtp_loss/0': 0.996094, 'mtp_loss/avg': 0.996094, 'mtp_loss/main': 0.847656, 'mtp_loss/scaled': 0.099609, 'epoch': 0.41}
# {'loss': 2.0264, 'grad_norm': 1.1183357238769531, 'learning_rate': 7.960393687950519e-06, 'mtp_loss/0': 0.984375, 'mtp_loss/avg': 0.984375, 'mtp_loss/main': 0.835938, 'mtp_loss/scaled': 0.098389, 'epoch': 0.44}
# {'loss': 1.9209, 'grad_norm': 1.1337876319885254, 'learning_rate': 7.398434486353566e-06, 'mtp_loss/0': 0.953125, 'mtp_loss/avg': 0.953125, 'mtp_loss/main': 0.818359, 'mtp_loss/scaled': 0.095215, 'epoch': 0.47}
# {'loss': 1.8379, 'grad_norm': 1.1525239944458008, 'learning_rate': 6.826677246645638e-06, 'mtp_loss/0': 0.996094, 'mtp_loss/avg': 0.996094, 'mtp_loss/main': 0.804688, 'mtp_loss/scaled': 0.099609, 'epoch': 0.5}
# {'loss': 1.8521, 'grad_norm': 1.0492746829986572, 'learning_rate': 6.25e-06, 'mtp_loss/0': 1.078125, 'mtp_loss/avg': 1.078125, 'mtp_loss/main': 0.919922, 'mtp_loss/scaled': 0.10791, 'epoch': 0.53}
# {'loss': 1.8965, 'grad_norm': 1.2497711181640625, 'learning_rate': 5.673322753354363e-06, 'mtp_loss/0': 0.992188, 'mtp_loss/avg': 0.992188, 'mtp_loss/main': 0.861328, 'mtp_loss/scaled': 0.099121, 'epoch': 0.56}
# {'loss': 2.0039, 'grad_norm': 1.1693906784057617, 'learning_rate': 5.1015655136464345e-06, 'mtp_loss/0': 1.042969, 'mtp_loss/avg': 1.042969, 'mtp_loss/main': 0.882812, 'mtp_loss/scaled': 0.104248, 'epoch': 0.59}
# {'loss': 1.8579, 'grad_norm': 1.21783447265625, 'learning_rate': 4.5396063120494825e-06, 'mtp_loss/0': 0.980469, 'mtp_loss/avg': 0.980469, 'mtp_loss/main': 0.847656, 'mtp_loss/scaled': 0.098145, 'epoch': 0.62}
# {'loss': 1.9497, 'grad_norm': 1.1696712970733643, 'learning_rate': 3.9922395863302945e-06, 'mtp_loss/0': 0.925781, 'mtp_loss/avg': 0.925781, 'mtp_loss/main': 0.792969, 'mtp_loss/scaled': 0.092529, 'epoch': 0.65}
# {'loss': 1.9741, 'grad_norm': 1.114098310470581, 'learning_rate': 3.4641352763966364e-06, 'mtp_loss/0': 0.958984, 'mtp_loss/avg': 0.958984, 'mtp_loss/main': 0.824219, 'mtp_loss/scaled': 0.095947, 'epoch': 0.68}
# {'loss': 1.8462, 'grad_norm': 1.1626179218292236, 'learning_rate': 2.9597989820165255e-06, 'mtp_loss/0': 1.097656, 'mtp_loss/avg': 1.097656, 'mtp_loss/main': 0.939453, 'mtp_loss/scaled': 0.109863, 'epoch': 0.71}
# {'loss': 1.9473, 'grad_norm': 1.037534475326538, 'learning_rate': 2.483533522629648e-06, 'mtp_loss/0': 1.007812, 'mtp_loss/avg': 1.007812, 'mtp_loss/main': 0.871094, 'mtp_loss/scaled': 0.10083, 'epoch': 0.74}
# {'loss': 1.9854, 'grad_norm': 1.0841609239578247, 'learning_rate': 2.0394022272090177e-06, 'mtp_loss/0': 1.109375, 'mtp_loss/avg': 1.109375, 'mtp_loss/main': 0.964844, 'mtp_loss/scaled': 0.11084, 'epoch': 0.76}
# {'loss': 1.9531, 'grad_norm': 1.0932981967926025, 'learning_rate': 1.6311942673708814e-06, 'mtp_loss/0': 0.859375, 'mtp_loss/avg': 0.859375, 'mtp_loss/main': 0.730469, 'mtp_loss/scaled': 0.085938, 'epoch': 0.79}
# {'loss': 2.061, 'grad_norm': 1.0840489864349365, 'learning_rate': 1.2623923294985032e-06, 'mtp_loss/0': 1.242188, 'mtp_loss/avg': 1.242188, 'mtp_loss/main': 1.082031, 'mtp_loss/scaled': 0.124268, 'epoch': 0.82}
# {'loss': 2.0166, 'grad_norm': 1.0733293294906616, 'learning_rate': 9.361429016899126e-07, 'mtp_loss/0': 1.125, 'mtp_loss/avg': 1.125, 'mtp_loss/main': 0.972656, 'mtp_loss/scaled': 0.112549, 'epoch': 0.85}
# {'loss': 1.9438, 'grad_norm': 1.0982338190078735, 'learning_rate': 6.552294290308611e-07, 'mtp_loss/0': 1.222656, 'mtp_loss/avg': 1.222656, 'mtp_loss/main': 1.09375, 'mtp_loss/scaled': 0.122314, 'epoch': 0.88}
# {'loss': 1.5111, 'grad_norm': 1.1150529384613037, 'learning_rate': 4.220485662227769e-07, 'mtp_loss/0': 0.941406, 'mtp_loss/avg': 0.941406, 'mtp_loss/main': 0.785156, 'mtp_loss/scaled': 0.094238, 'epoch': 0.91}
# {'loss': 1.9897, 'grad_norm': 1.1284080743789673, 'learning_rate': 2.38589730169881e-07, 'mtp_loss/0': 0.894531, 'mtp_loss/avg': 0.894531, 'mtp_loss/main': 0.757812, 'mtp_loss/scaled': 0.089355, 'epoch': 0.94}
# {'loss': 1.9092, 'grad_norm': 1.0228674411773682, 'learning_rate': 1.0641812697561385e-07, 'mtp_loss/0': 1.074219, 'mtp_loss/avg': 1.074219, 'mtp_loss/main': 0.912109, 'mtp_loss/scaled': 0.107422, 'epoch': 0.97}
# {'loss': 1.9238, 'grad_norm': 1.1848528385162354, 'learning_rate': 2.6661398156034577e-08, 'mtp_loss/0': 1.015625, 'mtp_loss/avg': 1.015625, 'mtp_loss/main': 0.876953, 'mtp_loss/scaled': 0.101562, 'epoch': 1.0}
