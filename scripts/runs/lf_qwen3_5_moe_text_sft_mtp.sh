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
output_dir=/sfs_turbo/zhisan/0604/lf_logs_qwen35/run_qwen3_5_moe_text_sft_mtp_sp2ep8/ckpt

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
    --tt_enable_mtp true \
    --tt_mtp_loss_scaling_factor 0.1 \
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
# {'loss': 0.551, 'grad_norm': 2.7708380222320557, 'learning_rate': 1.25e-05, 'mtp_loss/0': 0.287109, 'mtp_loss/avg': 0.287109, 'mtp_loss/main': 0.228516, 'mtp_loss/scaled': 0.028687, 'epoch': 0.03}
# {'loss': 0.5144, 'grad_norm': 2.3812966346740723, 'learning_rate': 1.2465761846051709e-05, 'mtp_loss/0': 0.757812, 'mtp_loss/avg': 0.757812, 'mtp_loss/main': 0.542969, 'mtp_loss/scaled': 0.075684, 'epoch': 0.07}
# {'loss': 0.4872, 'grad_norm': 1.9980570077896118, 'learning_rate': 1.2363422504586286e-05, 'mtp_loss/0': 0.535156, 'mtp_loss/avg': 0.535156, 'mtp_loss/main': 0.412109, 'mtp_loss/scaled': 0.053467, 'epoch': 0.1}
# {'loss': 0.4735, 'grad_norm': 1.3962830305099487, 'learning_rate': 1.219410322684471e-05, 'mtp_loss/0': 0.730469, 'mtp_loss/avg': 0.730469, 'mtp_loss/main': 0.507812, 'mtp_loss/scaled': 0.073242, 'epoch': 0.13}
# {'loss': 0.4479, 'grad_norm': 1.4141160249710083, 'learning_rate': 1.1959659110266256e-05, 'mtp_loss/0': 0.574219, 'mtp_loss/avg': 0.574219, 'mtp_loss/main': 0.427734, 'mtp_loss/scaled': 0.057373, 'epoch': 0.17}
# {'loss': 0.4576, 'grad_norm': 1.1995792388916016, 'learning_rate': 1.1662658773652743e-05, 'mtp_loss/0': 0.753906, 'mtp_loss/avg': 0.753906, 'mtp_loss/main': 0.527344, 'mtp_loss/scaled': 0.075195, 'epoch': 0.2}
# {'loss': 0.4265, 'grad_norm': 1.120602011680603, 'learning_rate': 1.1306356214843423e-05, 'mtp_loss/0': 0.498047, 'mtp_loss/avg': 0.498047, 'mtp_loss/main': 0.345703, 'mtp_loss/scaled': 0.049805, 'epoch': 0.23}
# {'loss': 0.4313, 'grad_norm': 1.1950894594192505, 'learning_rate': 1.0894655159233715e-05, 'mtp_loss/0': 0.742188, 'mtp_loss/avg': 0.742188, 'mtp_loss/main': 0.507812, 'mtp_loss/scaled': 0.074219, 'epoch': 0.27}
# {'loss': 0.5186, 'grad_norm': 1.160326361656189, 'learning_rate': 1.0432066289742864e-05, 'mtp_loss/0': 0.605469, 'mtp_loss/avg': 0.605469, 'mtp_loss/main': 0.453125, 'mtp_loss/scaled': 0.060547, 'epoch': 0.3}
# {'loss': 0.4384, 'grad_norm': 1.0047599077224731, 'learning_rate': 9.923657826827957e-06, 'mtp_loss/0': 0.511719, 'mtp_loss/avg': 0.511719, 'mtp_loss/main': 0.378906, 'mtp_loss/scaled': 0.05127, 'epoch': 0.33}
# {'loss': 0.4556, 'grad_norm': 1.1106029748916626, 'learning_rate': 9.375000000000001e-06, 'mtp_loss/0': 0.679688, 'mtp_loss/avg': 0.679688, 'mtp_loss/main': 0.498047, 'mtp_loss/scaled': 0.067871, 'epoch': 0.37}
# {'loss': 0.4199, 'grad_norm': 0.969395637512207, 'learning_rate': 8.792104019223753e-06, 'mtp_loss/0': 0.484375, 'mtp_loss/avg': 0.484375, 'mtp_loss/main': 0.351562, 'mtp_loss/scaled': 0.04834, 'epoch': 0.4}
# {'loss': 0.4531, 'grad_norm': 1.0091909170150757, 'learning_rate': 8.181356214843423e-06, 'mtp_loss/0': 0.699219, 'mtp_loss/avg': 0.699219, 'mtp_loss/main': 0.503906, 'mtp_loss/scaled': 0.069824, 'epoch': 0.43}
# {'loss': 0.4661, 'grad_norm': 1.0163527727127075, 'learning_rate': 7.5494480676109954e-06, 'mtp_loss/0': 0.597656, 'mtp_loss/avg': 0.597656, 'mtp_loss/main': 0.414062, 'mtp_loss/scaled': 0.059814, 'epoch': 0.47}
# {'loss': 0.4611, 'grad_norm': 1.074491024017334, 'learning_rate': 6.903302895422835e-06, 'mtp_loss/0': 0.519531, 'mtp_loss/avg': 0.519531, 'mtp_loss/main': 0.386719, 'mtp_loss/scaled': 0.052002, 'epoch': 0.5}
# {'loss': 0.4361, 'grad_norm': 1.1236445903778076, 'learning_rate': 6.25e-06, 'mtp_loss/0': 0.427734, 'mtp_loss/avg': 0.427734, 'mtp_loss/main': 0.34375, 'mtp_loss/scaled': 0.042725, 'epoch': 0.53}
# {'loss': 0.4518, 'grad_norm': 0.9631696343421936, 'learning_rate': 5.596697104577167e-06, 'mtp_loss/0': 0.574219, 'mtp_loss/avg': 0.574219, 'mtp_loss/main': 0.408203, 'mtp_loss/scaled': 0.057373, 'epoch': 0.57}
# {'loss': 0.4518, 'grad_norm': 0.9477324485778809, 'learning_rate': 4.950551932389005e-06, 'mtp_loss/0': 0.441406, 'mtp_loss/avg': 0.441406, 'mtp_loss/main': 0.333984, 'mtp_loss/scaled': 0.044189, 'epoch': 0.6}
# {'loss': 0.446, 'grad_norm': 1.045698642730713, 'learning_rate': 4.3186437851565795e-06, 'mtp_loss/0': 0.339844, 'mtp_loss/avg': 0.339844, 'mtp_loss/main': 0.249023, 'mtp_loss/scaled': 0.033936, 'epoch': 0.63}
# {'loss': 0.3522, 'grad_norm': 1.1128824949264526, 'learning_rate': 3.7078959807762497e-06, 'mtp_loss/0': 0.292969, 'mtp_loss/avg': 0.292969, 'mtp_loss/main': 0.219727, 'mtp_loss/scaled': 0.029297, 'epoch': 0.67}
# {'loss': 0.4157, 'grad_norm': 0.894106924533844, 'learning_rate': 3.1250000000000014e-06, 'mtp_loss/0': 0.589844, 'mtp_loss/avg': 0.589844, 'mtp_loss/main': 0.427734, 'mtp_loss/scaled': 0.059082, 'epoch': 0.7}
# {'loss': 0.4047, 'grad_norm': 1.133778691291809, 'learning_rate': 2.5763421731720436e-06, 'mtp_loss/0': 0.263672, 'mtp_loss/avg': 0.263672, 'mtp_loss/main': 0.206055, 'mtp_loss/scaled': 0.026367, 'epoch': 0.73}
# {'loss': 0.4228, 'grad_norm': 0.9337994456291199, 'learning_rate': 2.0679337102571383e-06, 'mtp_loss/0': 0.527344, 'mtp_loss/avg': 0.527344, 'mtp_loss/main': 0.390625, 'mtp_loss/scaled': 0.052734, 'epoch': 0.77}
# {'loss': 0.4402, 'grad_norm': 0.9359905123710632, 'learning_rate': 1.6053448407662854e-06, 'mtp_loss/0': 0.734375, 'mtp_loss/avg': 0.734375, 'mtp_loss/main': 0.539062, 'mtp_loss/scaled': 0.073242, 'epoch': 0.8}
# {'loss': 0.4116, 'grad_norm': 0.8935600519180298, 'learning_rate': 1.1936437851565791e-06, 'mtp_loss/0': 0.453125, 'mtp_loss/avg': 0.453125, 'mtp_loss/main': 0.324219, 'mtp_loss/scaled': 0.04541, 'epoch': 0.83}
# {'loss': 0.4277, 'grad_norm': 0.9160745739936829, 'learning_rate': 8.373412263472582e-07, 'mtp_loss/0': 0.554688, 'mtp_loss/avg': 0.554688, 'mtp_loss/main': 0.388672, 'mtp_loss/scaled': 0.05542, 'epoch': 0.87}
# {'loss': 0.4773, 'grad_norm': 0.931013822555542, 'learning_rate': 5.403408897337439e-07, 'mtp_loss/0': 0.367188, 'mtp_loss/avg': 0.367188, 'mtp_loss/main': 0.275391, 'mtp_loss/scaled': 0.036621, 'epoch': 0.9}
# {'loss': 0.417, 'grad_norm': 0.9334030151367188, 'learning_rate': 3.0589677315529047e-07, 'mtp_loss/0': 0.609375, 'mtp_loss/avg': 0.609375, 'mtp_loss/main': 0.419922, 'mtp_loss/scaled': 0.061035, 'epoch': 0.93}
# {'loss': 0.3906, 'grad_norm': 0.8155432939529419, 'learning_rate': 1.3657749541371446e-07, 'mtp_loss/0': 0.464844, 'mtp_loss/avg': 0.464844, 'mtp_loss/main': 0.330078, 'mtp_loss/scaled': 0.046387, 'epoch': 0.97}
# {'loss': 0.3487, 'grad_norm': 1.2100911140441895, 'learning_rate': 3.423815394829194e-08, 'mtp_loss/0': 0.550781, 'mtp_loss/avg': 0.550781, 'mtp_loss/main': 0.390625, 'mtp_loss/scaled': 0.055176, 'epoch': 1.0}

# sp2 4k
# {'loss': 0.5514, 'grad_norm': 2.7056403160095215, 'learning_rate': 1.25e-05, 'mtp_loss/0': 0.24707, 'mtp_loss/avg': 0.24707, 'mtp_loss/main': 0.187744, 'mtp_loss/scaled': 0.024689, 'epoch': 0.03}
# {'loss': 0.5144, 'grad_norm': 2.332282066345215, 'learning_rate': 1.2465761846051709e-05, 'mtp_loss/0': 0.270508, 'mtp_loss/avg': 0.270508, 'mtp_loss/main': 0.194092, 'mtp_loss/scaled': 0.027039, 'epoch': 0.07}
# {'loss': 0.4873, 'grad_norm': 1.9479658603668213, 'learning_rate': 1.2363422504586286e-05, 'mtp_loss/0': 0.268555, 'mtp_loss/avg': 0.268555, 'mtp_loss/main': 0.19873, 'mtp_loss/scaled': 0.026855, 'epoch': 0.1}
# {'loss': 0.473, 'grad_norm': 1.4117895364761353, 'learning_rate': 1.219410322684471e-05, 'mtp_loss/0': 0.380859, 'mtp_loss/avg': 0.380859, 'mtp_loss/main': 0.260742, 'mtp_loss/scaled': 0.038086, 'epoch': 0.13}
# {'loss': 0.4487, 'grad_norm': 1.4622678756713867, 'learning_rate': 1.1959659110266256e-05, 'mtp_loss/0': 0.259766, 'mtp_loss/avg': 0.259766, 'mtp_loss/main': 0.191895, 'mtp_loss/scaled': 0.026001, 'epoch': 0.17}
# {'loss': 0.4572, 'grad_norm': 1.2124379873275757, 'learning_rate': 1.1662658773652743e-05, 'mtp_loss/0': 0.323242, 'mtp_loss/avg': 0.323242, 'mtp_loss/main': 0.231934, 'mtp_loss/scaled': 0.032288, 'epoch': 0.2}
# {'loss': 0.428, 'grad_norm': 1.1238784790039062, 'learning_rate': 1.1306356214843423e-05, 'mtp_loss/0': 0.230469, 'mtp_loss/avg': 0.230469, 'mtp_loss/main': 0.166504, 'mtp_loss/scaled': 0.023071, 'epoch': 0.23}
# {'loss': 0.432, 'grad_norm': 1.2066570520401, 'learning_rate': 1.0894655159233715e-05, 'mtp_loss/0': 0.324219, 'mtp_loss/avg': 0.324219, 'mtp_loss/main': 0.229004, 'mtp_loss/scaled': 0.032471, 'epoch': 0.27}
# {'loss': 0.5188, 'grad_norm': 1.1581734418869019, 'learning_rate': 1.0432066289742864e-05, 'mtp_loss/0': 0.293945, 'mtp_loss/avg': 0.293945, 'mtp_loss/main': 0.213867, 'mtp_loss/scaled': 0.029419, 'epoch': 0.3}
# {'loss': 0.4394, 'grad_norm': 0.9957149028778076, 'learning_rate': 9.923657826827957e-06, 'mtp_loss/0': 0.337891, 'mtp_loss/avg': 0.337891, 'mtp_loss/main': 0.234375, 'mtp_loss/scaled': 0.033752, 'epoch': 0.33}
# {'loss': 0.4559, 'grad_norm': 1.0857481956481934, 'learning_rate': 9.375000000000001e-06, 'mtp_loss/0': 0.330078, 'mtp_loss/avg': 0.330078, 'mtp_loss/main': 0.242676, 'mtp_loss/scaled': 0.032959, 'epoch': 0.37}
# {'loss': 0.4205, 'grad_norm': 0.9571061134338379, 'learning_rate': 8.792104019223753e-06, 'mtp_loss/0': 0.256348, 'mtp_loss/avg': 0.256348, 'mtp_loss/main': 0.183594, 'mtp_loss/scaled': 0.025635, 'epoch': 0.4}
# {'loss': 0.4531, 'grad_norm': 0.985826313495636, 'learning_rate': 8.181356214843423e-06, 'mtp_loss/0': 0.352539, 'mtp_loss/avg': 0.352539, 'mtp_loss/main': 0.25293, 'mtp_loss/scaled': 0.035278, 'epoch': 0.43}
# {'loss': 0.4672, 'grad_norm': 1.0218065977096558, 'learning_rate': 7.5494480676109954e-06, 'mtp_loss/0': 0.277344, 'mtp_loss/avg': 0.277344, 'mtp_loss/main': 0.196289, 'mtp_loss/scaled': 0.02771, 'epoch': 0.47}
# {'loss': 0.4611, 'grad_norm': 1.0765632390975952, 'learning_rate': 6.903302895422835e-06, 'mtp_loss/0': 0.291992, 'mtp_loss/avg': 0.291992, 'mtp_loss/main': 0.210449, 'mtp_loss/scaled': 0.029236, 'epoch': 0.5}
# {'loss': 0.4359, 'grad_norm': 1.1595889329910278, 'learning_rate': 6.25e-06, 'mtp_loss/0': 0.255371, 'mtp_loss/avg': 0.255371, 'mtp_loss/main': 0.193359, 'mtp_loss/scaled': 0.025574, 'epoch': 0.53}
# {'loss': 0.4528, 'grad_norm': 0.955569863319397, 'learning_rate': 5.596697104577167e-06, 'mtp_loss/0': 0.288086, 'mtp_loss/avg': 0.288086, 'mtp_loss/main': 0.210938, 'mtp_loss/scaled': 0.028809, 'epoch': 0.57}
# {'loss': 0.4521, 'grad_norm': 0.9445566534996033, 'learning_rate': 4.950551932389005e-06, 'mtp_loss/0': 0.197754, 'mtp_loss/avg': 0.197754, 'mtp_loss/main': 0.142578, 'mtp_loss/scaled': 0.019775, 'epoch': 0.6}
# {'loss': 0.4465, 'grad_norm': 1.0343819856643677, 'learning_rate': 4.3186437851565795e-06, 'mtp_loss/0': 0.253906, 'mtp_loss/avg': 0.253906, 'mtp_loss/main': 0.181641, 'mtp_loss/scaled': 0.025452, 'epoch': 0.63}
# {'loss': 0.3525, 'grad_norm': 0.8404765129089355, 'learning_rate': 3.7078959807762497e-06, 'mtp_loss/0': 0.186523, 'mtp_loss/avg': 0.186523, 'mtp_loss/main': 0.137207, 'mtp_loss/scaled': 0.018677, 'epoch': 0.67}
# {'loss': 0.4156, 'grad_norm': 0.898656964302063, 'learning_rate': 3.1250000000000014e-06, 'mtp_loss/0': 0.232422, 'mtp_loss/avg': 0.232422, 'mtp_loss/main': 0.171387, 'mtp_loss/scaled': 0.023254, 'epoch': 0.7}
# {'loss': 0.4045, 'grad_norm': 0.8555459976196289, 'learning_rate': 2.5763421731720436e-06, 'mtp_loss/0': 0.213867, 'mtp_loss/avg': 0.213867, 'mtp_loss/main': 0.161377, 'mtp_loss/scaled': 0.021423, 'epoch': 0.73}
# {'loss': 0.4233, 'grad_norm': 0.9156928658485413, 'learning_rate': 2.0679337102571383e-06, 'mtp_loss/0': 0.28125, 'mtp_loss/avg': 0.28125, 'mtp_loss/main': 0.204102, 'mtp_loss/scaled': 0.028137, 'epoch': 0.77}
# {'loss': 0.4399, 'grad_norm': 0.9203101396560669, 'learning_rate': 1.6053448407662854e-06, 'mtp_loss/0': 0.329102, 'mtp_loss/avg': 0.329102, 'mtp_loss/main': 0.240234, 'mtp_loss/scaled': 0.032837, 'epoch': 0.8}
# {'loss': 0.4116, 'grad_norm': 0.8852235078811646, 'learning_rate': 1.1936437851565791e-06, 'mtp_loss/0': 0.236816, 'mtp_loss/avg': 0.236816, 'mtp_loss/main': 0.168945, 'mtp_loss/scaled': 0.023682, 'epoch': 0.83}
# {'loss': 0.4281, 'grad_norm': 0.9124271273612976, 'learning_rate': 8.373412263472582e-07, 'mtp_loss/0': 0.265137, 'mtp_loss/avg': 0.265137, 'mtp_loss/main': 0.189453, 'mtp_loss/scaled': 0.026489, 'epoch': 0.87}
# {'loss': 0.4774, 'grad_norm': 0.9169250726699829, 'learning_rate': 5.403408897337439e-07, 'mtp_loss/0': 0.239258, 'mtp_loss/avg': 0.239258, 'mtp_loss/main': 0.17334, 'mtp_loss/scaled': 0.023926, 'epoch': 0.9}
# {'loss': 0.4172, 'grad_norm': 0.934613823890686, 'learning_rate': 3.0589677315529047e-07, 'mtp_loss/0': 0.257324, 'mtp_loss/avg': 0.257324, 'mtp_loss/main': 0.182129, 'mtp_loss/scaled': 0.025757, 'epoch': 0.93}
# {'loss': 0.3914, 'grad_norm': 0.8168951272964478, 'learning_rate': 1.3657749541371446e-07, 'mtp_loss/0': 0.202637, 'mtp_loss/avg': 0.202637, 'mtp_loss/main': 0.146484, 'mtp_loss/scaled': 0.020264, 'epoch': 0.97}
# {'loss': 0.351, 'grad_norm': 1.7252073287963867, 'learning_rate': 3.423815394829194e-08, 'mtp_loss/0': 0.293945, 'mtp_loss/avg': 0.293945, 'mtp_loss/main': 0.210938, 'mtp_loss/scaled': 0.029358, 'epoch': 1.0}
