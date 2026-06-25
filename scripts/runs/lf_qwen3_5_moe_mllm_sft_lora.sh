#!/usr/bin/env bash
# Qwen3.5-MoE LoRA MLLM SFT with torchturbo
# Usage:
#   bash scripts/runs/lf_qwen3_5_moe_mllm_sft_lora.sh

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
output_dir=/sfs_turbo/zhisan/0604/lf_logs_qwen35/run_qwen3_5_moe_mllm_sft_lora_sp2ep8/ckpt

seq_length=$((1024 * 4))
dataset=llava_1k_zh
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
    --tt_freeze_vision_tower \
    --ddp_timeout 1800000000 2>&1 | tee -a $log_file

    # --tt_freeze_vision_tower \
    # --tt_freeze_multi_modal_projector \
    # --tt_freeze_language_model \

    # --use_rslora \

# sp2 4k [Full training]
# {'loss': 1.3501, 'grad_norm': 1.9952099323272705, 'learning_rate': 0.0005, 'epoch': 0.17}
# {'loss': 1.0825, 'grad_norm': 0.5239965915679932, 'learning_rate': 0.0004986304738420684, 'epoch': 0.33}
# {'loss': 1.0784, 'grad_norm': 0.22794142365455627, 'learning_rate': 0.0004945369001834514, 'epoch': 0.5}
# {'loss': 1.0396, 'grad_norm': 0.2855709195137024, 'learning_rate': 0.0004877641290737884, 'epoch': 0.67}
# {'loss': 1.072, 'grad_norm': 1.3388960361480713, 'learning_rate': 0.0004783863644106502, 'epoch': 0.83}
# {'loss': 1.0129, 'grad_norm': 0.2416631281375885, 'learning_rate': 0.00046650635094610973, 'epoch': 1.0}
# {'loss': 0.9751, 'grad_norm': 0.1516360640525818, 'learning_rate': 0.0004522542485937369, 'epoch': 1.17}
# {'loss': 0.9343, 'grad_norm': 0.18152466416358948, 'learning_rate': 0.00043578620636934855, 'epoch': 1.33}
# {'loss': 0.9365, 'grad_norm': 0.18293501436710358, 'learning_rate': 0.0004172826515897146, 'epoch': 1.5}
# {'loss': 0.9343, 'grad_norm': 0.6447073817253113, 'learning_rate': 0.0003969463130731183, 'epoch': 1.67}
# {'loss': 0.9536, 'grad_norm': 0.2601969838142395, 'learning_rate': 0.000375, 'epoch': 1.83}
# {'loss': 0.8884, 'grad_norm': 0.14886628091335297, 'learning_rate': 0.0003516841607689501, 'epoch': 2.0}
# {'loss': 0.8367, 'grad_norm': 0.31802141666412354, 'learning_rate': 0.00032725424859373687, 'epoch': 2.17}
# {'loss': 0.7976, 'grad_norm': 0.2823852598667145, 'learning_rate': 0.0003019779227044398, 'epoch': 2.33}
# {'loss': 0.7687, 'grad_norm': 0.19867642223834991, 'learning_rate': 0.0002761321158169134, 'epoch': 2.5}
# {'loss': 0.801, 'grad_norm': 0.7484733462333679, 'learning_rate': 0.00025, 'epoch': 2.67}
# {'loss': 0.8239, 'grad_norm': 0.6085671186447144, 'learning_rate': 0.00022386788418308668, 'epoch': 2.83}
# {'loss': 0.7394, 'grad_norm': 0.39517641067504883, 'learning_rate': 0.0001980220772955602, 'epoch': 3.0}
# {'loss': 0.6854, 'grad_norm': 0.2024737000465393, 'learning_rate': 0.00017274575140626317, 'epoch': 3.17}
# {'loss': 0.6669, 'grad_norm': 0.23659688234329224, 'learning_rate': 0.00014831583923105, 'epoch': 3.33}
# {'loss': 0.6359, 'grad_norm': 0.1985343098640442, 'learning_rate': 0.00012500000000000006, 'epoch': 3.5}
# {'loss': 0.6792, 'grad_norm': 1.1379449367523193, 'learning_rate': 0.00010305368692688174, 'epoch': 3.67}
# {'loss': 0.7307, 'grad_norm': 0.3010111153125763, 'learning_rate': 8.271734841028553e-05, 'epoch': 3.83}
# {'loss': 0.6262, 'grad_norm': 0.3545139729976654, 'learning_rate': 6.421379363065141e-05, 'epoch': 4.0}
# {'loss': 0.5864, 'grad_norm': 0.29829445481300354, 'learning_rate': 4.7745751406263163e-05, 'epoch': 4.17}
# {'loss': 0.5927, 'grad_norm': 0.33056196570396423, 'learning_rate': 3.3493649053890325e-05, 'epoch': 4.33}
# {'loss': 0.5748, 'grad_norm': 0.21123714745044708, 'learning_rate': 2.1613635589349755e-05, 'epoch': 4.5}
# {'loss': 0.6312, 'grad_norm': 0.27702268958091736, 'learning_rate': 1.2235870926211617e-05, 'epoch': 4.67}
# {'loss': 0.6861, 'grad_norm': 0.2886561453342438, 'learning_rate': 5.463099816548578e-06, 'epoch': 4.83}
# {'loss': 0.5916, 'grad_norm': 0.5534628033638, 'learning_rate': 1.3695261579316775e-06, 'epoch': 5.0}

# sp2 4k [Freeze vision encoder]
# {'loss': 1.3501, 'grad_norm': 1.989679217338562, 'learning_rate': 0.0005, 'epoch': 0.17}
# {'loss': 1.0842, 'grad_norm': 0.424142450094223, 'learning_rate': 0.0004986304738420684, 'epoch': 0.33}
# {'loss': 1.0793, 'grad_norm': 0.4307427406311035, 'learning_rate': 0.0004945369001834514, 'epoch': 0.5}
# {'loss': 1.0388, 'grad_norm': 0.3388862609863281, 'learning_rate': 0.0004877641290737884, 'epoch': 0.67}
# {'loss': 1.073, 'grad_norm': 0.24500489234924316, 'learning_rate': 0.0004783863644106502, 'epoch': 0.83}
# {'loss': 1.012, 'grad_norm': 0.30147767066955566, 'learning_rate': 0.00046650635094610973, 'epoch': 1.0}
# {'loss': 0.9761, 'grad_norm': 0.7525965571403503, 'learning_rate': 0.0004522542485937369, 'epoch': 1.17}
# {'loss': 0.9368, 'grad_norm': 0.5907979011535645, 'learning_rate': 0.00043578620636934855, 'epoch': 1.33}
# {'loss': 0.9407, 'grad_norm': 0.6475263237953186, 'learning_rate': 0.0004172826515897146, 'epoch': 1.5}
# {'loss': 0.9355, 'grad_norm': 0.37071892619132996, 'learning_rate': 0.0003969463130731183, 'epoch': 1.67}
# {'loss': 0.9387, 'grad_norm': 0.2745262086391449, 'learning_rate': 0.000375, 'epoch': 1.83}
# {'loss': 0.894, 'grad_norm': 0.17472296953201294, 'learning_rate': 0.0003516841607689501, 'epoch': 2.0}
# {'loss': 0.8416, 'grad_norm': 0.2180529683828354, 'learning_rate': 0.00032725424859373687, 'epoch': 2.17}
# {'loss': 0.8035, 'grad_norm': 0.2039031982421875, 'learning_rate': 0.0003019779227044398, 'epoch': 2.33}
# {'loss': 0.7772, 'grad_norm': 0.4266115427017212, 'learning_rate': 0.0002761321158169134, 'epoch': 2.5}
# {'loss': 0.8079, 'grad_norm': 0.3416644334793091, 'learning_rate': 0.00025, 'epoch': 2.67}
# {'loss': 0.8047, 'grad_norm': 0.5677202343940735, 'learning_rate': 0.00022386788418308668, 'epoch': 2.83}
# {'loss': 0.7458, 'grad_norm': 0.23640425503253937, 'learning_rate': 0.0001980220772955602, 'epoch': 3.0}
# {'loss': 0.6904, 'grad_norm': 0.19368243217468262, 'learning_rate': 0.00017274575140626317, 'epoch': 3.17}
# {'loss': 0.674, 'grad_norm': 0.2927948832511902, 'learning_rate': 0.00014831583923105, 'epoch': 3.33}
# {'loss': 0.6416, 'grad_norm': 0.4944034218788147, 'learning_rate': 0.00012500000000000006, 'epoch': 3.5}
# {'loss': 0.6886, 'grad_norm': 0.23644280433654785, 'learning_rate': 0.00010305368692688174, 'epoch': 3.67}
# {'loss': 0.7088, 'grad_norm': 0.891928493976593, 'learning_rate': 8.271734841028553e-05, 'epoch': 3.83}
# {'loss': 0.6327, 'grad_norm': 0.23216232657432556, 'learning_rate': 6.421379363065141e-05, 'epoch': 4.0}
# {'loss': 0.592, 'grad_norm': 0.3323267698287964, 'learning_rate': 4.7745751406263163e-05, 'epoch': 4.17}
# {'loss': 0.5983, 'grad_norm': 0.3155030608177185, 'learning_rate': 3.3493649053890325e-05, 'epoch': 4.33}
# {'loss': 0.5817, 'grad_norm': 0.19332203269004822, 'learning_rate': 2.1613635589349755e-05, 'epoch': 4.5}
# {'loss': 0.6382, 'grad_norm': 0.44400322437286377, 'learning_rate': 1.2235870926211617e-05, 'epoch': 4.67}
# {'loss': 0.6701, 'grad_norm': 0.8867807984352112, 'learning_rate': 5.463099816548578e-06, 'epoch': 4.83}
# {'loss': 0.6003, 'grad_norm': 0.21541479229927063, 'learning_rate': 1.3695261579316775e-06, 'epoch': 5.0}
