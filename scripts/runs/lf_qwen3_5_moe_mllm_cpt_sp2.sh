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
output_dir=/sfs_turbo/zhisan/0604/lf_logs_qwen35/run_qwen3_5_moe_debug_35b_cpt_mllm_sp2ep8/ckpt

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

# sp2 4k [Full training]
# {'loss': 2.7891, 'grad_norm': 83.25856018066406, 'learning_rate': 1.25e-05, 'epoch': 0.04}
# {'loss': 3.1963, 'grad_norm': 22.255374908447266, 'learning_rate': 1.2465761846051709e-05, 'epoch': 0.07}
# {'loss': 3.3311, 'grad_norm': 41.21863555908203, 'learning_rate': 1.2363422504586286e-05, 'epoch': 0.11}
# {'loss': 2.7847, 'grad_norm': 16.837425231933594, 'learning_rate': 1.219410322684471e-05, 'epoch': 0.15}
# {'loss': 2.1287, 'grad_norm': 11.973355293273926, 'learning_rate': 1.1959659110266256e-05, 'epoch': 0.19}
# {'loss': 2.573, 'grad_norm': 12.879950523376465, 'learning_rate': 1.1662658773652743e-05, 'epoch': 0.22}
# {'loss': 2.6421, 'grad_norm': 11.747886657714844, 'learning_rate': 1.1306356214843423e-05, 'epoch': 0.26}
# {'loss': 2.7529, 'grad_norm': 59.371986389160156, 'learning_rate': 1.0894655159233715e-05, 'epoch': 0.3}
# {'loss': 1.9562, 'grad_norm': 4.579838752746582, 'learning_rate': 1.0432066289742864e-05, 'epoch': 0.33}
# {'loss': 2.228, 'grad_norm': 5.539555072784424, 'learning_rate': 9.923657826827957e-06, 'epoch': 0.37}
# {'loss': 2.5859, 'grad_norm': 24.24497413635254, 'learning_rate': 9.375000000000001e-06, 'epoch': 0.41}
# {'loss': 2.3369, 'grad_norm': 24.60919952392578, 'learning_rate': 8.792104019223753e-06, 'epoch': 0.44}
# {'loss': 2.4497, 'grad_norm': 18.504188537597656, 'learning_rate': 8.181356214843423e-06, 'epoch': 0.48}
# {'loss': 2.7285, 'grad_norm': 30.928983688354492, 'learning_rate': 7.5494480676109954e-06, 'epoch': 0.52}
# {'loss': 2.8604, 'grad_norm': 46.815513610839844, 'learning_rate': 6.903302895422835e-06, 'epoch': 0.56}
# {'loss': 2.7056, 'grad_norm': 7.283843040466309, 'learning_rate': 6.25e-06, 'epoch': 0.59}
# {'loss': 2.6987, 'grad_norm': 31.250627517700195, 'learning_rate': 5.596697104577167e-06, 'epoch': 0.63}
# {'loss': 2.8286, 'grad_norm': 66.15345001220703, 'learning_rate': 4.950551932389005e-06, 'epoch': 0.67}
# {'loss': 2.4045, 'grad_norm': 46.45777130126953, 'learning_rate': 4.3186437851565795e-06, 'epoch': 0.7}
# {'loss': 2.3909, 'grad_norm': 10.236305236816406, 'learning_rate': 3.7078959807762497e-06, 'epoch': 0.74}
# {'loss': 2.2963, 'grad_norm': 35.715579986572266, 'learning_rate': 3.1250000000000014e-06, 'epoch': 0.78}
# {'loss': 2.8193, 'grad_norm': 14.50294017791748, 'learning_rate': 2.5763421731720436e-06, 'epoch': 0.81}
# {'loss': 1.9578, 'grad_norm': 9.119998931884766, 'learning_rate': 2.0679337102571383e-06, 'epoch': 0.85}
# {'loss': 2.3867, 'grad_norm': 7.089620590209961, 'learning_rate': 1.6053448407662854e-06, 'epoch': 0.89}
# {'loss': 2.3083, 'grad_norm': 20.632265090942383, 'learning_rate': 1.1936437851565791e-06, 'epoch': 0.93}
# {'loss': 2.2988, 'grad_norm': 8.87476634979248, 'learning_rate': 8.373412263472582e-07, 'epoch': 0.96}
# {'loss': 2.0389, 'grad_norm': 9.048565864562988, 'learning_rate': 5.403408897337439e-07, 'epoch': 1.0}
# {'loss': 1.954, 'grad_norm': 7.5061774253845215, 'learning_rate': 3.0589677315529047e-07, 'epoch': 1.04}
# {'loss': 2.0239, 'grad_norm': 15.009380340576172, 'learning_rate': 1.3657749541371446e-07, 'epoch': 1.07}
# {'loss': 2.3237, 'grad_norm': 19.726247787475586, 'learning_rate': 3.423815394829194e-08, 'epoch': 1.11}

# sp2 4k [Freeze vision encoder]
# {'loss': 2.7891, 'grad_norm': 38.61737823486328, 'learning_rate': 1.25e-05, 'epoch': 0.04}
# {'loss': 3.1943, 'grad_norm': 6.260625839233398, 'learning_rate': 1.2465761846051709e-05, 'epoch': 0.07}
# {'loss': 3.3291, 'grad_norm': 6.734350204467773, 'learning_rate': 1.2363422504586286e-05, 'epoch': 0.11}
# {'loss': 2.7769, 'grad_norm': 5.280318737030029, 'learning_rate': 1.219410322684471e-05, 'epoch': 0.15}
# {'loss': 2.1177, 'grad_norm': 5.5141777992248535, 'learning_rate': 1.1959659110266256e-05, 'epoch': 0.19}
# {'loss': 2.5681, 'grad_norm': 3.3031790256500244, 'learning_rate': 1.1662658773652743e-05, 'epoch': 0.22}
# {'loss': 2.6411, 'grad_norm': 3.828721523284912, 'learning_rate': 1.1306356214843423e-05, 'epoch': 0.26}
# {'loss': 2.7432, 'grad_norm': 4.442893981933594, 'learning_rate': 1.0894655159233715e-05, 'epoch': 0.3}
# {'loss': 1.948, 'grad_norm': 2.683587074279785, 'learning_rate': 1.0432066289742864e-05, 'epoch': 0.33}
# {'loss': 2.2166, 'grad_norm': 2.6041951179504395, 'learning_rate': 9.923657826827957e-06, 'epoch': 0.37}
# {'loss': 2.5737, 'grad_norm': 3.4283909797668457, 'learning_rate': 9.375000000000001e-06, 'epoch': 0.41}
# {'loss': 2.3345, 'grad_norm': 4.429203987121582, 'learning_rate': 8.792104019223753e-06, 'epoch': 0.44}
# {'loss': 2.4236, 'grad_norm': 8.192304611206055, 'learning_rate': 8.181356214843423e-06, 'epoch': 0.48}
# {'loss': 2.7104, 'grad_norm': 3.2273619174957275, 'learning_rate': 7.5494480676109954e-06, 'epoch': 0.52}
# {'loss': 2.856, 'grad_norm': 5.804811000823975, 'learning_rate': 6.903302895422835e-06, 'epoch': 0.56}
# {'loss': 2.6965, 'grad_norm': 3.0373878479003906, 'learning_rate': 6.25e-06, 'epoch': 0.59}
# {'loss': 2.689, 'grad_norm': 3.3159303665161133, 'learning_rate': 5.596697104577167e-06, 'epoch': 0.63}
# {'loss': 2.8047, 'grad_norm': 4.387152194976807, 'learning_rate': 4.950551932389005e-06, 'epoch': 0.67}
# {'loss': 2.3167, 'grad_norm': 20.433841705322266, 'learning_rate': 4.3186437851565795e-06, 'epoch': 0.7}
# {'loss': 2.3811, 'grad_norm': 3.4022576808929443, 'learning_rate': 3.7078959807762497e-06, 'epoch': 0.74}
# {'loss': 2.2656, 'grad_norm': 6.147268295288086, 'learning_rate': 3.1250000000000014e-06, 'epoch': 0.78}
# {'loss': 2.7998, 'grad_norm': 3.375831365585327, 'learning_rate': 2.5763421731720436e-06, 'epoch': 0.81}
# {'loss': 1.9514, 'grad_norm': 3.4734227657318115, 'learning_rate': 2.0679337102571383e-06, 'epoch': 0.85}
# {'loss': 2.3779, 'grad_norm': 2.9968483448028564, 'learning_rate': 1.6053448407662854e-06, 'epoch': 0.89}
# {'loss': 2.3103, 'grad_norm': 3.8708722591400146, 'learning_rate': 1.1936437851565791e-06, 'epoch': 0.93}
# {'loss': 2.2927, 'grad_norm': 2.9601521492004395, 'learning_rate': 8.373412263472582e-07, 'epoch': 0.96}
# {'loss': 2.0492, 'grad_norm': 3.1044294834136963, 'learning_rate': 5.403408897337439e-07, 'epoch': 1.0}
# {'loss': 1.9651, 'grad_norm': 3.353863000869751, 'learning_rate': 3.0589677315529047e-07, 'epoch': 1.04}
# {'loss': 1.9761, 'grad_norm': 6.321893215179443, 'learning_rate': 1.3657749541371446e-07, 'epoch': 1.07}
# {'loss': 2.104, 'grad_norm': 6.307958602905273, 'learning_rate': 3.423815394829194e-08, 'epoch': 1.11}
