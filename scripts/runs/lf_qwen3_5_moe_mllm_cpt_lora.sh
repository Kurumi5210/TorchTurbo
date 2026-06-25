#!/usr/bin/env bash
# Qwen3.5-MoE LoRA MLLM CPT with torchturbo
# Usage:
#   bash scripts/runs/lf_qwen3_5_moe_mllm_cpt_lora.sh

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
output_dir=/sfs_turbo/zhisan/0604/lf_logs_qwen35/run_qwen3_5_moe_mllm_cpt_lora_sp2ep8/ckpt

seq_length=$((1024 * 4))
dataset=security
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
    --tt_freeze_vision_tower \
    --use_rslora \
    --ddp_timeout 1800000000 2>&1 | tee -a $log_file

    # --tt_freeze_vision_tower \
    # --tt_freeze_multi_modal_projector \
    # --tt_freeze_language_model \

    # --use_rslora \

# sp2 4k [Full training]
# {'loss': 2.7891, 'grad_norm': 5.516116619110107, 'learning_rate': 0.0005, 'epoch': 0.04}
# {'loss': 3.2031, 'grad_norm': 1.3978757858276367, 'learning_rate': 0.0004986304738420684, 'epoch': 0.07}
# {'loss': 3.3613, 'grad_norm': 2.11080002784729, 'learning_rate': 0.0004945369001834514, 'epoch': 0.11}
# {'loss': 2.7983, 'grad_norm': 1.2775804996490479, 'learning_rate': 0.0004877641290737884, 'epoch': 0.15}
# {'loss': 2.1541, 'grad_norm': 1.0859019756317139, 'learning_rate': 0.0004783863644106502, 'epoch': 0.19}
# {'loss': 2.5955, 'grad_norm': 0.6708095073699951, 'learning_rate': 0.00046650635094610973, 'epoch': 0.22}
# {'loss': 2.6514, 'grad_norm': 0.87071293592453, 'learning_rate': 0.0004522542485937369, 'epoch': 0.26}
# {'loss': 2.7612, 'grad_norm': 1.7142983675003052, 'learning_rate': 0.00043578620636934855, 'epoch': 0.3}
# {'loss': 1.9598, 'grad_norm': 0.8282343149185181, 'learning_rate': 0.0004172826515897146, 'epoch': 0.33}
# {'loss': 2.2432, 'grad_norm': 0.8707656860351562, 'learning_rate': 0.0003969463130731183, 'epoch': 0.37}
# {'loss': 2.6021, 'grad_norm': 0.9878096580505371, 'learning_rate': 0.000375, 'epoch': 0.41}
# {'loss': 2.3296, 'grad_norm': 2.2737159729003906, 'learning_rate': 0.0003516841607689501, 'epoch': 0.44}
# {'loss': 2.4792, 'grad_norm': 1.4718208312988281, 'learning_rate': 0.00032725424859373687, 'epoch': 0.48}
# {'loss': 2.7461, 'grad_norm': 1.3546971082687378, 'learning_rate': 0.0003019779227044398, 'epoch': 0.52}
# {'loss': 2.8711, 'grad_norm': 1.1564478874206543, 'learning_rate': 0.0002761321158169134, 'epoch': 0.56}
# {'loss': 2.7095, 'grad_norm': 0.882752537727356, 'learning_rate': 0.00025, 'epoch': 0.59}
# {'loss': 2.7012, 'grad_norm': 4.973069190979004, 'learning_rate': 0.00022386788418308668, 'epoch': 0.63}
# {'loss': 2.832, 'grad_norm': 1.6004101037979126, 'learning_rate': 0.0001980220772955602, 'epoch': 0.67}
# {'loss': 2.4177, 'grad_norm': 5.651004314422607, 'learning_rate': 0.00017274575140626317, 'epoch': 0.7}
# {'loss': 2.3904, 'grad_norm': 0.8141946792602539, 'learning_rate': 0.00014831583923105, 'epoch': 0.74}
# {'loss': 2.2747, 'grad_norm': 4.145059108734131, 'learning_rate': 0.00012500000000000006, 'epoch': 0.78}
# {'loss': 2.8486, 'grad_norm': 1.4589073657989502, 'learning_rate': 0.00010305368692688174, 'epoch': 0.81}
# {'loss': 1.9558, 'grad_norm': 1.1077793836593628, 'learning_rate': 8.271734841028553e-05, 'epoch': 0.85}
# {'loss': 2.3965, 'grad_norm': 2.1518285274505615, 'learning_rate': 6.421379363065141e-05, 'epoch': 0.89}
# {'loss': 2.363, 'grad_norm': 1.8123269081115723, 'learning_rate': 4.7745751406263163e-05, 'epoch': 0.93}
# {'loss': 2.3225, 'grad_norm': 0.910576581954956, 'learning_rate': 3.3493649053890325e-05, 'epoch': 0.96}
# {'loss': 2.3157, 'grad_norm': 1.2836370468139648, 'learning_rate': 2.1613635589349755e-05, 'epoch': 1.0}
# {'loss': 2.2578, 'grad_norm': 0.9423073530197144, 'learning_rate': 1.2235870926211617e-05, 'epoch': 1.04}
# {'loss': 2.6436, 'grad_norm': 9.874869346618652, 'learning_rate': 5.463099816548578e-06, 'epoch': 1.07}
# {'loss': 2.7671, 'grad_norm': 1.8092881441116333, 'learning_rate': 1.3695261579316775e-06, 'epoch': 1.11}

# sp2 4k [Freeze vision encoder]
# {'loss': 2.7891, 'grad_norm': 2.281747579574585, 'learning_rate': 0.0005, 'epoch': 0.04}
# {'loss': 3.2041, 'grad_norm': 0.42556264996528625, 'learning_rate': 0.0004986304738420684, 'epoch': 0.07}
# {'loss': 3.3721, 'grad_norm': 0.5432040095329285, 'learning_rate': 0.0004945369001834514, 'epoch': 0.11}
# {'loss': 2.7993, 'grad_norm': 0.5931195616722107, 'learning_rate': 0.0004877641290737884, 'epoch': 0.15}
# {'loss': 2.1523, 'grad_norm': 0.4755932092666626, 'learning_rate': 0.0004783863644106502, 'epoch': 0.19}
# {'loss': 2.5986, 'grad_norm': 0.3221801519393921, 'learning_rate': 0.00046650635094610973, 'epoch': 0.22}
# {'loss': 2.6533, 'grad_norm': 0.4931413531303406, 'learning_rate': 0.0004522542485937369, 'epoch': 0.26}
# {'loss': 2.7563, 'grad_norm': 0.39001843333244324, 'learning_rate': 0.00043578620636934855, 'epoch': 0.3}
# {'loss': 1.9657, 'grad_norm': 0.2030654400587082, 'learning_rate': 0.0004172826515897146, 'epoch': 0.33}
# {'loss': 2.2456, 'grad_norm': 0.206773042678833, 'learning_rate': 0.0003969463130731183, 'epoch': 0.37}
# {'loss': 2.5996, 'grad_norm': 0.30323296785354614, 'learning_rate': 0.000375, 'epoch': 0.41}
# {'loss': 2.3315, 'grad_norm': 0.358756422996521, 'learning_rate': 0.0003516841607689501, 'epoch': 0.44}
# {'loss': 2.4746, 'grad_norm': 0.5146746039390564, 'learning_rate': 0.00032725424859373687, 'epoch': 0.48}
# {'loss': 2.7534, 'grad_norm': 0.2368728220462799, 'learning_rate': 0.0003019779227044398, 'epoch': 0.52}
# {'loss': 2.8721, 'grad_norm': 0.3389914035797119, 'learning_rate': 0.0002761321158169134, 'epoch': 0.56}
# {'loss': 2.7153, 'grad_norm': 0.259469211101532, 'learning_rate': 0.00025, 'epoch': 0.59}
# {'loss': 2.7021, 'grad_norm': 0.28969791531562805, 'learning_rate': 0.00022386788418308668, 'epoch': 0.63}
# {'loss': 2.8291, 'grad_norm': 0.4093320667743683, 'learning_rate': 0.0001980220772955602, 'epoch': 0.67}
# {'loss': 2.4331, 'grad_norm': 2.5464043617248535, 'learning_rate': 0.00017274575140626317, 'epoch': 0.7}
# {'loss': 2.3896, 'grad_norm': 0.2859818637371063, 'learning_rate': 0.00014831583923105, 'epoch': 0.74}
# {'loss': 2.264, 'grad_norm': 0.45659613609313965, 'learning_rate': 0.00012500000000000006, 'epoch': 0.78}
# {'loss': 2.8467, 'grad_norm': 0.28585925698280334, 'learning_rate': 0.00010305368692688174, 'epoch': 0.81}
# {'loss': 1.9575, 'grad_norm': 0.2897636890411377, 'learning_rate': 8.271734841028553e-05, 'epoch': 0.85}
# {'loss': 2.4014, 'grad_norm': 0.2701222002506256, 'learning_rate': 6.421379363065141e-05, 'epoch': 0.89}
# {'loss': 2.3738, 'grad_norm': 0.32080063223838806, 'learning_rate': 4.7745751406263163e-05, 'epoch': 0.93}
# {'loss': 2.3315, 'grad_norm': 0.24139900505542755, 'learning_rate': 3.3493649053890325e-05, 'epoch': 0.96}
# {'loss': 2.2957, 'grad_norm': 0.20569802820682526, 'learning_rate': 2.1613635589349755e-05, 'epoch': 1.0}
# {'loss': 2.238, 'grad_norm': 0.2153601050376892, 'learning_rate': 1.2235870926211617e-05, 'epoch': 1.04}
# {'loss': 2.6367, 'grad_norm': 0.44074925780296326, 'learning_rate': 5.463099816548578e-06, 'epoch': 1.07}
# {'loss': 2.6919, 'grad_norm': 0.5093758702278137, 'learning_rate': 1.3695261579316775e-06, 'epoch': 1.11}
