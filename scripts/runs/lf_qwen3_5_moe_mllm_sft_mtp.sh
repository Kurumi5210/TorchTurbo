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
output_dir=/sfs_turbo/zhisan/0604/lf_logs_qwen35/run_qwen3_5_moe_mllm_sft_mtp_sp2ep8/ckpt

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
# {'loss': 1.5063, 'grad_norm': 53.37123489379883, 'learning_rate': 1.25e-05, 'mtp_loss/0': 1.195312, 'mtp_loss/avg': 1.195312, 'mtp_loss/main': 0.949219, 'mtp_loss/scaled': 0.119629, 'epoch': 0.17}
# {'loss': 1.1924, 'grad_norm': 6.32449197769165, 'learning_rate': 1.2465761846051709e-05, 'mtp_loss/0': 1.367188, 'mtp_loss/avg': 1.367188, 'mtp_loss/main': 1.164062, 'mtp_loss/scaled': 0.136719, 'epoch': 0.33}
# {'loss': 1.2085, 'grad_norm': 26.799489974975586, 'learning_rate': 1.2363422504586286e-05, 'mtp_loss/0': 1.203125, 'mtp_loss/avg': 1.203125, 'mtp_loss/main': 1.039062, 'mtp_loss/scaled': 0.120117, 'epoch': 0.5}
# {'loss': 1.1731, 'grad_norm': 6.156775951385498, 'learning_rate': 1.219410322684471e-05, 'mtp_loss/0': 1.09375, 'mtp_loss/avg': 1.09375, 'mtp_loss/main': 0.910156, 'mtp_loss/scaled': 0.109375, 'epoch': 0.67}
# {'loss': 1.1982, 'grad_norm': 5.745540618896484, 'learning_rate': 1.1959659110266256e-05, 'mtp_loss/0': 1.109375, 'mtp_loss/avg': 1.109375, 'mtp_loss/main': 0.929688, 'mtp_loss/scaled': 0.11084, 'epoch': 0.83}
# {'loss': 1.1042, 'grad_norm': 3.3816912174224854, 'learning_rate': 1.1662658773652743e-05, 'mtp_loss/0': 1.234375, 'mtp_loss/avg': 1.234375, 'mtp_loss/main': 1.054688, 'mtp_loss/scaled': 0.123535, 'epoch': 1.0}
# {'loss': 0.9529, 'grad_norm': 2.4966542720794678, 'learning_rate': 1.1306356214843423e-05, 'mtp_loss/0': 0.820312, 'mtp_loss/avg': 0.820312, 'mtp_loss/main': 0.605469, 'mtp_loss/scaled': 0.082031, 'epoch': 1.17}
# {'loss': 0.7146, 'grad_norm': 3.3144311904907227, 'learning_rate': 1.0894655159233715e-05, 'mtp_loss/0': 0.902344, 'mtp_loss/avg': 0.902344, 'mtp_loss/main': 0.71875, 'mtp_loss/scaled': 0.090332, 'epoch': 1.33}
# {'loss': 0.9753, 'grad_norm': 6.800325870513916, 'learning_rate': 1.0432066289742864e-05, 'mtp_loss/0': 1.070312, 'mtp_loss/avg': 1.070312, 'mtp_loss/main': 0.921875, 'mtp_loss/scaled': 0.106934, 'epoch': 1.5}
# {'loss': 0.7922, 'grad_norm': 3.2232370376586914, 'learning_rate': 9.923657826827957e-06, 'mtp_loss/0': 0.796875, 'mtp_loss/avg': 0.796875, 'mtp_loss/main': 0.613281, 'mtp_loss/scaled': 0.07959, 'epoch': 1.67}
# {'loss': 0.8433, 'grad_norm': 5.49159049987793, 'learning_rate': 9.375000000000001e-06, 'mtp_loss/0': 0.867188, 'mtp_loss/avg': 0.867188, 'mtp_loss/main': 0.679688, 'mtp_loss/scaled': 0.086914, 'epoch': 1.83}
# {'loss': 0.6766, 'grad_norm': 4.939660549163818, 'learning_rate': 8.792104019223753e-06, 'mtp_loss/0': 0.683594, 'mtp_loss/avg': 0.683594, 'mtp_loss/main': 0.492188, 'mtp_loss/scaled': 0.068359, 'epoch': 2.0}
# {'loss': 0.5463, 'grad_norm': 2.831777572631836, 'learning_rate': 8.181356214843423e-06, 'mtp_loss/0': 0.445312, 'mtp_loss/avg': 0.445312, 'mtp_loss/main': 0.235352, 'mtp_loss/scaled': 0.044434, 'epoch': 2.17}
# {'loss': 0.4087, 'grad_norm': 21.956010818481445, 'learning_rate': 7.5494480676109954e-06, 'mtp_loss/0': 0.613281, 'mtp_loss/avg': 0.613281, 'mtp_loss/main': 0.410156, 'mtp_loss/scaled': 0.061279, 'epoch': 2.33}
# {'loss': 0.7266, 'grad_norm': 3.193096399307251, 'learning_rate': 6.903302895422835e-06, 'mtp_loss/0': 0.949219, 'mtp_loss/avg': 0.949219, 'mtp_loss/main': 0.753906, 'mtp_loss/scaled': 0.094727, 'epoch': 2.5}
# {'loss': 0.4743, 'grad_norm': 6.246068000793457, 'learning_rate': 6.25e-06, 'mtp_loss/0': 0.574219, 'mtp_loss/avg': 0.574219, 'mtp_loss/main': 0.363281, 'mtp_loss/scaled': 0.057373, 'epoch': 2.67}
# {'loss': 0.6068, 'grad_norm': 4.385340213775635, 'learning_rate': 5.596697104577167e-06, 'mtp_loss/0': 0.695312, 'mtp_loss/avg': 0.695312, 'mtp_loss/main': 0.482422, 'mtp_loss/scaled': 0.069336, 'epoch': 2.83}
# {'loss': 0.3997, 'grad_norm': 4.589730739593506, 'learning_rate': 4.950551932389005e-06, 'mtp_loss/0': 0.423828, 'mtp_loss/avg': 0.423828, 'mtp_loss/main': 0.191406, 'mtp_loss/scaled': 0.04248, 'epoch': 3.0}
# {'loss': 0.2825, 'grad_norm': 3.387213945388794, 'learning_rate': 4.3186437851565795e-06, 'mtp_loss/0': 0.265625, 'mtp_loss/avg': 0.265625, 'mtp_loss/main': 0.085938, 'mtp_loss/scaled': 0.026611, 'epoch': 3.17}
# {'loss': 0.3367, 'grad_norm': 3.5663232803344727, 'learning_rate': 3.7078959807762497e-06, 'mtp_loss/0': 0.550781, 'mtp_loss/avg': 0.550781, 'mtp_loss/main': 0.353516, 'mtp_loss/scaled': 0.055176, 'epoch': 3.33}
# {'loss': 0.4838, 'grad_norm': 2.6044602394104004, 'learning_rate': 3.1250000000000014e-06, 'mtp_loss/0': 0.789062, 'mtp_loss/avg': 0.789062, 'mtp_loss/main': 0.558594, 'mtp_loss/scaled': 0.079102, 'epoch': 3.5}
# {'loss': 0.3337, 'grad_norm': 2.9647016525268555, 'learning_rate': 2.5763421731720436e-06, 'mtp_loss/0': 0.484375, 'mtp_loss/avg': 0.484375, 'mtp_loss/main': 0.265625, 'mtp_loss/scaled': 0.04834, 'epoch': 3.67}
# {'loss': 0.4391, 'grad_norm': 3.3731954097747803, 'learning_rate': 2.0679337102571383e-06, 'mtp_loss/0': 0.589844, 'mtp_loss/avg': 0.589844, 'mtp_loss/main': 0.359375, 'mtp_loss/scaled': 0.059082, 'epoch': 3.83}
# {'loss': 0.2704, 'grad_norm': 3.755493402481079, 'learning_rate': 1.6053448407662854e-06, 'mtp_loss/0': 0.277344, 'mtp_loss/avg': 0.277344, 'mtp_loss/main': 0.073242, 'mtp_loss/scaled': 0.02771, 'epoch': 4.0}
# {'loss': 0.1733, 'grad_norm': 2.3438477516174316, 'learning_rate': 1.1936437851565791e-06, 'mtp_loss/0': 0.211914, 'mtp_loss/avg': 0.211914, 'mtp_loss/main': 0.042725, 'mtp_loss/scaled': 0.02124, 'epoch': 4.17}
# {'loss': 0.2473, 'grad_norm': 3.060194730758667, 'learning_rate': 8.373412263472582e-07, 'mtp_loss/0': 0.490234, 'mtp_loss/avg': 0.490234, 'mtp_loss/main': 0.246094, 'mtp_loss/scaled': 0.049072, 'epoch': 4.33}
# {'loss': 0.3607, 'grad_norm': 3.2260584831237793, 'learning_rate': 5.403408897337439e-07, 'mtp_loss/0': 0.710938, 'mtp_loss/avg': 0.710938, 'mtp_loss/main': 0.445312, 'mtp_loss/scaled': 0.071289, 'epoch': 4.5}
# {'loss': 0.2624, 'grad_norm': 2.0041446685791016, 'learning_rate': 3.0589677315529047e-07, 'mtp_loss/0': 0.449219, 'mtp_loss/avg': 0.449219, 'mtp_loss/main': 0.211914, 'mtp_loss/scaled': 0.044922, 'epoch': 4.67}
# {'loss': 0.3778, 'grad_norm': 3.9460742473602295, 'learning_rate': 1.3657749541371446e-07, 'mtp_loss/0': 0.558594, 'mtp_loss/avg': 0.558594, 'mtp_loss/main': 0.314453, 'mtp_loss/scaled': 0.055908, 'epoch': 4.83}
# {'loss': 0.2355, 'grad_norm': 2.125126600265503, 'learning_rate': 3.423815394829194e-08, 'mtp_loss/0': 0.255859, 'mtp_loss/avg': 0.255859, 'mtp_loss/main': 0.049072, 'mtp_loss/scaled': 0.025635, 'epoch': 5.0}

# sp1 4k [Freeze vision encoder]
# {'loss': 1.5063, 'grad_norm': 29.63215446472168, 'learning_rate': 1.25e-05, 'mtp_loss/0': 1.195312, 'mtp_loss/avg': 1.195312, 'mtp_loss/main': 0.949219, 'mtp_loss/scaled': 0.119629, 'epoch': 0.17}
# {'loss': 1.1924, 'grad_norm': 3.107937812805176, 'learning_rate': 1.2465761846051709e-05, 'mtp_loss/0': 1.367188, 'mtp_loss/avg': 1.367188, 'mtp_loss/main': 1.164062, 'mtp_loss/scaled': 0.136719, 'epoch': 0.33}
# {'loss': 1.2075, 'grad_norm': 2.659048080444336, 'learning_rate': 1.2363422504586286e-05, 'mtp_loss/0': 1.210938, 'mtp_loss/avg': 1.210938, 'mtp_loss/main': 1.039062, 'mtp_loss/scaled': 0.121094, 'epoch': 0.5}
# {'loss': 1.1594, 'grad_norm': 1.5999250411987305, 'learning_rate': 1.219410322684471e-05, 'mtp_loss/0': 1.085938, 'mtp_loss/avg': 1.085938, 'mtp_loss/main': 0.898438, 'mtp_loss/scaled': 0.108398, 'epoch': 0.67}
# {'loss': 1.1943, 'grad_norm': 1.9004141092300415, 'learning_rate': 1.1959659110266256e-05, 'mtp_loss/0': 1.101562, 'mtp_loss/avg': 1.101562, 'mtp_loss/main': 0.925781, 'mtp_loss/scaled': 0.110352, 'epoch': 0.83}
# {'loss': 1.1047, 'grad_norm': 1.4650225639343262, 'learning_rate': 1.1662658773652743e-05, 'mtp_loss/0': 1.226562, 'mtp_loss/avg': 1.226562, 'mtp_loss/main': 1.046875, 'mtp_loss/scaled': 0.122559, 'epoch': 1.0}
# {'loss': 0.9685, 'grad_norm': 1.4834688901901245, 'learning_rate': 1.1306356214843423e-05, 'mtp_loss/0': 0.84375, 'mtp_loss/avg': 0.84375, 'mtp_loss/main': 0.628906, 'mtp_loss/scaled': 0.084473, 'epoch': 1.17}
# {'loss': 0.7615, 'grad_norm': 2.1711883544921875, 'learning_rate': 1.0894655159233715e-05, 'mtp_loss/0': 0.957031, 'mtp_loss/avg': 0.957031, 'mtp_loss/main': 0.765625, 'mtp_loss/scaled': 0.095703, 'epoch': 1.33}
# {'loss': 0.7784, 'grad_norm': 1.8405334949493408, 'learning_rate': 1.0432066289742864e-05, 'mtp_loss/0': 0.929688, 'mtp_loss/avg': 0.929688, 'mtp_loss/main': 0.753906, 'mtp_loss/scaled': 0.092773, 'epoch': 1.5}
# {'loss': 0.7217, 'grad_norm': 1.6986956596374512, 'learning_rate': 9.923657826827957e-06, 'mtp_loss/0': 0.742188, 'mtp_loss/avg': 0.742188, 'mtp_loss/main': 0.558594, 'mtp_loss/scaled': 0.074219, 'epoch': 1.67}
# {'loss': 0.826, 'grad_norm': 3.1977577209472656, 'learning_rate': 9.375000000000001e-06, 'mtp_loss/0': 0.847656, 'mtp_loss/avg': 0.847656, 'mtp_loss/main': 0.671875, 'mtp_loss/scaled': 0.084961, 'epoch': 1.83}
# {'loss': 0.7102, 'grad_norm': 3.2860662937164307, 'learning_rate': 8.792104019223753e-06, 'mtp_loss/0': 0.679688, 'mtp_loss/avg': 0.679688, 'mtp_loss/main': 0.496094, 'mtp_loss/scaled': 0.067871, 'epoch': 2.0}
# {'loss': 0.611, 'grad_norm': 1.70380437374115, 'learning_rate': 8.181356214843423e-06, 'mtp_loss/0': 0.494141, 'mtp_loss/avg': 0.494141, 'mtp_loss/main': 0.275391, 'mtp_loss/scaled': 0.049316, 'epoch': 2.17}
# {'loss': 0.4929, 'grad_norm': 2.1756510734558105, 'learning_rate': 7.5494480676109954e-06, 'mtp_loss/0': 0.707031, 'mtp_loss/avg': 0.707031, 'mtp_loss/main': 0.503906, 'mtp_loss/scaled': 0.070801, 'epoch': 2.33}
# {'loss': 0.4866, 'grad_norm': 2.5677850246429443, 'learning_rate': 6.903302895422835e-06, 'mtp_loss/0': 0.742188, 'mtp_loss/avg': 0.742188, 'mtp_loss/main': 0.53125, 'mtp_loss/scaled': 0.074219, 'epoch': 2.5}
# {'loss': 0.4267, 'grad_norm': 3.318876028060913, 'learning_rate': 6.25e-06, 'mtp_loss/0': 0.519531, 'mtp_loss/avg': 0.519531, 'mtp_loss/main': 0.332031, 'mtp_loss/scaled': 0.052002, 'epoch': 2.67}
# {'loss': 0.5831, 'grad_norm': 2.7200489044189453, 'learning_rate': 5.596697104577167e-06, 'mtp_loss/0': 0.675781, 'mtp_loss/avg': 0.675781, 'mtp_loss/main': 0.470703, 'mtp_loss/scaled': 0.067383, 'epoch': 2.83}
# {'loss': 0.4288, 'grad_norm': 1.8672512769699097, 'learning_rate': 4.950551932389005e-06, 'mtp_loss/0': 0.439453, 'mtp_loss/avg': 0.439453, 'mtp_loss/main': 0.208984, 'mtp_loss/scaled': 0.043945, 'epoch': 3.0}
# {'loss': 0.3264, 'grad_norm': 2.663102388381958, 'learning_rate': 4.3186437851565795e-06, 'mtp_loss/0': 0.296875, 'mtp_loss/avg': 0.296875, 'mtp_loss/main': 0.104492, 'mtp_loss/scaled': 0.029663, 'epoch': 3.17}
# {'loss': 0.3086, 'grad_norm': 4.696621417999268, 'learning_rate': 3.7078959807762497e-06, 'mtp_loss/0': 0.539062, 'mtp_loss/avg': 0.539062, 'mtp_loss/main': 0.320312, 'mtp_loss/scaled': 0.053955, 'epoch': 3.33}
# {'loss': 0.3341, 'grad_norm': 4.583480358123779, 'learning_rate': 3.1250000000000014e-06, 'mtp_loss/0': 0.632812, 'mtp_loss/avg': 0.632812, 'mtp_loss/main': 0.400391, 'mtp_loss/scaled': 0.063477, 'epoch': 3.5}
# {'loss': 0.3063, 'grad_norm': 3.388155460357666, 'learning_rate': 2.5763421731720436e-06, 'mtp_loss/0': 0.455078, 'mtp_loss/avg': 0.455078, 'mtp_loss/main': 0.245117, 'mtp_loss/scaled': 0.04541, 'epoch': 3.67}
# {'loss': 0.4323, 'grad_norm': 1.9014803171157837, 'learning_rate': 2.0679337102571383e-06, 'mtp_loss/0': 0.59375, 'mtp_loss/avg': 0.59375, 'mtp_loss/main': 0.361328, 'mtp_loss/scaled': 0.059326, 'epoch': 3.83}
# {'loss': 0.2735, 'grad_norm': 1.7650352716445923, 'learning_rate': 1.6053448407662854e-06, 'mtp_loss/0': 0.28125, 'mtp_loss/avg': 0.28125, 'mtp_loss/main': 0.069336, 'mtp_loss/scaled': 0.028076, 'epoch': 4.0}
# {'loss': 0.2152, 'grad_norm': 1.736554741859436, 'learning_rate': 1.1936437851565791e-06, 'mtp_loss/0': 0.236328, 'mtp_loss/avg': 0.236328, 'mtp_loss/main': 0.053711, 'mtp_loss/scaled': 0.023682, 'epoch': 4.17}
# {'loss': 0.2338, 'grad_norm': 1.2228206396102905, 'learning_rate': 8.373412263472582e-07, 'mtp_loss/0': 0.486328, 'mtp_loss/avg': 0.486328, 'mtp_loss/main': 0.227539, 'mtp_loss/scaled': 0.048584, 'epoch': 4.33}
# {'loss': 0.2645, 'grad_norm': 1.4779858589172363, 'learning_rate': 5.403408897337439e-07, 'mtp_loss/0': 0.59375, 'mtp_loss/avg': 0.59375, 'mtp_loss/main': 0.341797, 'mtp_loss/scaled': 0.059326, 'epoch': 4.5}
# {'loss': 0.2461, 'grad_norm': 1.5557546615600586, 'learning_rate': 3.0589677315529047e-07, 'mtp_loss/0': 0.429688, 'mtp_loss/avg': 0.429688, 'mtp_loss/main': 0.202148, 'mtp_loss/scaled': 0.042969, 'epoch': 4.67}
# {'loss': 0.3591, 'grad_norm': 2.1603190898895264, 'learning_rate': 1.3657749541371446e-07, 'mtp_loss/0': 0.550781, 'mtp_loss/avg': 0.550781, 'mtp_loss/main': 0.3125, 'mtp_loss/scaled': 0.055176, 'epoch': 4.83}
# {'loss': 0.2312, 'grad_norm': 2.1054787635803223, 'learning_rate': 3.423815394829194e-08, 'mtp_loss/0': 0.239258, 'mtp_loss/avg': 0.239258, 'mtp_loss/main': 0.040771, 'mtp_loss/scaled': 0.023926, 'epoch': 5.0}

# sp2 4k [Freeze vision encoder]
# {'loss': 1.5083, 'grad_norm': 29.73491859436035, 'learning_rate': 1.25e-05, 'mtp_loss/0': 0.613281, 'mtp_loss/avg': 0.613281, 'mtp_loss/main': 0.499023, 'mtp_loss/scaled': 0.061401, 'epoch': 0.17}
# {'loss': 1.1924, 'grad_norm': 3.0814876556396484, 'learning_rate': 1.2465761846051709e-05, 'mtp_loss/0': 0.701172, 'mtp_loss/avg': 0.701172, 'mtp_loss/main': 0.597656, 'mtp_loss/scaled': 0.070068, 'epoch': 0.33}
# {'loss': 1.2144, 'grad_norm': 3.146371603012085, 'learning_rate': 1.2363422504586286e-05, 'mtp_loss/0': 0.589844, 'mtp_loss/avg': 0.589844, 'mtp_loss/main': 0.505859, 'mtp_loss/scaled': 0.05896, 'epoch': 0.5}
# {'loss': 1.1624, 'grad_norm': 1.781233787536621, 'learning_rate': 1.219410322684471e-05, 'mtp_loss/0': 0.585938, 'mtp_loss/avg': 0.585938, 'mtp_loss/main': 0.490234, 'mtp_loss/scaled': 0.058594, 'epoch': 0.67}
# {'loss': 1.1951, 'grad_norm': 2.034144401550293, 'learning_rate': 1.1959659110266256e-05, 'mtp_loss/0': 0.541016, 'mtp_loss/avg': 0.541016, 'mtp_loss/main': 0.453125, 'mtp_loss/scaled': 0.054199, 'epoch': 0.83}
# {'loss': 1.1074, 'grad_norm': 1.478384017944336, 'learning_rate': 1.1662658773652743e-05, 'mtp_loss/0': 0.605469, 'mtp_loss/avg': 0.605469, 'mtp_loss/main': 0.519531, 'mtp_loss/scaled': 0.060547, 'epoch': 1.0}
# {'loss': 0.9656, 'grad_norm': 1.508519172668457, 'learning_rate': 1.1306356214843423e-05, 'mtp_loss/0': 0.421875, 'mtp_loss/avg': 0.421875, 'mtp_loss/main': 0.327148, 'mtp_loss/scaled': 0.042114, 'epoch': 1.17}
# {'loss': 0.7517, 'grad_norm': 2.369755983352661, 'learning_rate': 1.0894655159233715e-05, 'mtp_loss/0': 0.448242, 'mtp_loss/avg': 0.448242, 'mtp_loss/main': 0.358398, 'mtp_loss/scaled': 0.044922, 'epoch': 1.33}
# {'loss': 0.7944, 'grad_norm': 1.8567891120910645, 'learning_rate': 1.0432066289742864e-05, 'mtp_loss/0': 0.451172, 'mtp_loss/avg': 0.451172, 'mtp_loss/main': 0.364258, 'mtp_loss/scaled': 0.045166, 'epoch': 1.5}
# {'loss': 0.7383, 'grad_norm': 1.6886405944824219, 'learning_rate': 9.923657826827957e-06, 'mtp_loss/0': 0.418945, 'mtp_loss/avg': 0.418945, 'mtp_loss/main': 0.320312, 'mtp_loss/scaled': 0.04187, 'epoch': 1.67}
# {'loss': 0.8311, 'grad_norm': 2.9944396018981934, 'learning_rate': 9.375000000000001e-06, 'mtp_loss/0': 0.416992, 'mtp_loss/avg': 0.416992, 'mtp_loss/main': 0.326172, 'mtp_loss/scaled': 0.041626, 'epoch': 1.83}
# {'loss': 0.6993, 'grad_norm': 2.847301959991455, 'learning_rate': 8.792104019223753e-06, 'mtp_loss/0': 0.408203, 'mtp_loss/avg': 0.408203, 'mtp_loss/main': 0.313477, 'mtp_loss/scaled': 0.040771, 'epoch': 2.0}
# {'loss': 0.5995, 'grad_norm': 1.7331005334854126, 'learning_rate': 8.181356214843423e-06, 'mtp_loss/0': 0.246094, 'mtp_loss/avg': 0.246094, 'mtp_loss/main': 0.14209, 'mtp_loss/scaled': 0.024597, 'epoch': 2.17}
# {'loss': 0.489, 'grad_norm': 2.178990602493286, 'learning_rate': 7.5494480676109954e-06, 'mtp_loss/0': 0.307617, 'mtp_loss/avg': 0.307617, 'mtp_loss/main': 0.214844, 'mtp_loss/scaled': 0.030701, 'epoch': 2.33}
# {'loss': 0.5012, 'grad_norm': 2.3878931999206543, 'learning_rate': 6.903302895422835e-06, 'mtp_loss/0': 0.348633, 'mtp_loss/avg': 0.348633, 'mtp_loss/main': 0.248535, 'mtp_loss/scaled': 0.034912, 'epoch': 2.5}
# {'loss': 0.4337, 'grad_norm': 2.959557294845581, 'learning_rate': 6.25e-06, 'mtp_loss/0': 0.299805, 'mtp_loss/avg': 0.299805, 'mtp_loss/main': 0.193359, 'mtp_loss/scaled': 0.029968, 'epoch': 2.67}
# {'loss': 0.5812, 'grad_norm': 2.6108803749084473, 'learning_rate': 5.596697104577167e-06, 'mtp_loss/0': 0.323242, 'mtp_loss/avg': 0.323242, 'mtp_loss/main': 0.217773, 'mtp_loss/scaled': 0.032288, 'epoch': 2.83}
# {'loss': 0.413, 'grad_norm': 2.3086965084075928, 'learning_rate': 4.950551932389005e-06, 'mtp_loss/0': 0.299805, 'mtp_loss/avg': 0.299805, 'mtp_loss/main': 0.185791, 'mtp_loss/scaled': 0.030029, 'epoch': 3.0}
# {'loss': 0.3257, 'grad_norm': 3.252000331878662, 'learning_rate': 4.3186437851565795e-06, 'mtp_loss/0': 0.152344, 'mtp_loss/avg': 0.152344, 'mtp_loss/main': 0.054077, 'mtp_loss/scaled': 0.015259, 'epoch': 3.17}
# {'loss': 0.3091, 'grad_norm': 4.668210506439209, 'learning_rate': 3.7078959807762497e-06, 'mtp_loss/0': 0.226562, 'mtp_loss/avg': 0.226562, 'mtp_loss/main': 0.116699, 'mtp_loss/scaled': 0.022644, 'epoch': 3.33}
# {'loss': 0.338, 'grad_norm': 3.9145853519439697, 'learning_rate': 3.1250000000000014e-06, 'mtp_loss/0': 0.291992, 'mtp_loss/avg': 0.291992, 'mtp_loss/main': 0.174805, 'mtp_loss/scaled': 0.029175, 'epoch': 3.5}
# {'loss': 0.3021, 'grad_norm': 2.3214271068573, 'learning_rate': 2.5763421731720436e-06, 'mtp_loss/0': 0.257324, 'mtp_loss/avg': 0.257324, 'mtp_loss/main': 0.141357, 'mtp_loss/scaled': 0.025757, 'epoch': 3.67}
# {'loss': 0.4198, 'grad_norm': 1.7579163312911987, 'learning_rate': 2.0679337102571383e-06, 'mtp_loss/0': 0.274414, 'mtp_loss/avg': 0.274414, 'mtp_loss/main': 0.157715, 'mtp_loss/scaled': 0.027405, 'epoch': 3.83}
# {'loss': 0.2696, 'grad_norm': 2.5683655738830566, 'learning_rate': 1.6053448407662854e-06, 'mtp_loss/0': 0.237305, 'mtp_loss/avg': 0.237305, 'mtp_loss/main': 0.121948, 'mtp_loss/scaled': 0.023743, 'epoch': 4.0}
# {'loss': 0.2228, 'grad_norm': 2.1179141998291016, 'learning_rate': 1.1936437851565791e-06, 'mtp_loss/0': 0.122559, 'mtp_loss/avg': 0.122559, 'mtp_loss/main': 0.027527, 'mtp_loss/scaled': 0.012268, 'epoch': 4.17}
# {'loss': 0.2341, 'grad_norm': 1.4319422245025635, 'learning_rate': 8.373412263472582e-07, 'mtp_loss/0': 0.198242, 'mtp_loss/avg': 0.198242, 'mtp_loss/main': 0.071899, 'mtp_loss/scaled': 0.019836, 'epoch': 4.33}
# {'loss': 0.2655, 'grad_norm': 1.72998046875, 'learning_rate': 5.403408897337439e-07, 'mtp_loss/0': 0.269531, 'mtp_loss/avg': 0.269531, 'mtp_loss/main': 0.139893, 'mtp_loss/scaled': 0.026917, 'epoch': 4.5}
# {'loss': 0.2322, 'grad_norm': 1.9285333156585693, 'learning_rate': 3.0589677315529047e-07, 'mtp_loss/0': 0.236328, 'mtp_loss/avg': 0.236328, 'mtp_loss/main': 0.109131, 'mtp_loss/scaled': 0.023621, 'epoch': 4.67}
# {'loss': 0.3487, 'grad_norm': 2.277651786804199, 'learning_rate': 1.3657749541371446e-07, 'mtp_loss/0': 0.253418, 'mtp_loss/avg': 0.253418, 'mtp_loss/main': 0.130615, 'mtp_loss/scaled': 0.02533, 'epoch': 4.83}
# {'loss': 0.236, 'grad_norm': 2.3385021686553955, 'learning_rate': 3.423815394829194e-08, 'mtp_loss/0': 0.224609, 'mtp_loss/avg': 0.224609, 'mtp_loss/main': 0.1073, 'mtp_loss/scaled': 0.022461, 'epoch': 5.0}
