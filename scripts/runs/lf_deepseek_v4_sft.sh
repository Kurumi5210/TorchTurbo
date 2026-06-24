#!/usr/bin/env bash
# DeepSeek V4 SFT training with torchturbo
# Usage:
#   cd /a3_inference/itask/workdir/zss02308789/610/torchturbo-v3
#   bash scripts/runs/lf_deepseek_v4_sft.sh

set -o pipefail
set -ex

source /usr/local/Ascend/ascend-toolkit/set_env.sh

export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export ASCEND_GLOBAL_LOG_LEVEL=3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-29972}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export GLOO_LOG_LEVEL=${GLOO_LOG_LEVEL:-DEBUG}
export MULTI_STREAM_MEMORY_REUSE=2

WORKDIR=${WORKDIR:-/a3_inference/itask/workdir/zss02308789/610}
model_name_or_path=${model_name_or_path:-${WORKDIR}/small_dsv4}
output_dir=${output_dir:-${WORKDIR}/output-2/deepseek_v4_sft}

seq_length=${seq_length:-$((1024 * 8))}
dataset=${dataset:-science}
finetuning_type=${finetuning_type:-full}

export WANDB_DISABLED=${WANDB_DISABLED:-True}
export ALLOW_EXTRA_ARGS=True
export PYTHONPATH=/opt/MindSpeed:${PYTHONPATH}
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-error}
export DISABLE_VERSION_CHECK=${DISABLE_VERSION_CHECK:-1}

export NPUS_PER_NODE=${NPUS_PER_NODE:-16}
export NODE_RANK=${RANK:-0}
export WORLD_SIZE=${NPUS_PER_NODE}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-23556}

export ASCEND_PROCESS_LOG_PATH=${output_dir}/log
mkdir -p ${output_dir}

log_file=${output_dir}/log-${NODE_RANK}.txt
cur_time=$(date "+%Y%m%d-%H%M%S")
echo ${cur_time} >> ${log_file}
echo ${output_dir} >> ${log_file}
cp -r $0 ${output_dir}/

DISTRIBUTED_ARGS="
    --nproc_per_node ${NPUS_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --node_rank ${NODE_RANK}
"

RUN_LF=${WORKDIR}/torchturbo-v3/scripts/runs/run_lf.py

export PYTHONPATH=${WORKDIR}/LLaMA-Factory-v3/src:${PYTHONPATH}
export PYTHONPATH=${WORKDIR}/torchturbo-v3/src:${PYTHONPATH}

python - <<PY
import os
for path in [
    os.environ.get("model_name_or_path", "${model_name_or_path}"),
    "${WORKDIR}/LLaMA-Factory-v3/src",
    "${WORKDIR}/torchturbo-v3/src",
    "${RUN_LF}",
]:
    print(path, "OK" if os.path.exists(path) else "MISSING")
PY

torchrun ${DISTRIBUTED_ARGS} ${RUN_LF} \
    --tt_sp_size ${TT_SP_SIZE:-2} \
    --tt_ep_size ${TT_EP_SIZE:-8} \
    --disable_shuffling \
    --stage sft \
    --model_name_or_path ${model_name_or_path} \
    --do_train \
    --val_size 0 \
    --dataset ${dataset} \
    --dataset_dir /a3_inference/itask/workdir/zss02308789/610/torchturbo-v3/dataset \
    --preprocessing_num_workers ${PREPROCESSING_NUM_WORKERS:-16} \
    --cutoff_len ${seq_length} \
    --flash_attn fa2 \
    --template ${template:-default} \
    --finetuning_type ${finetuning_type} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE:-1} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS:-1} \
    --eval_accumulation_steps 1 \
    --gradient_checkpointing \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --overwrite_cache \
    --save_strategy "steps" \
    --eval_steps 200000000 \
    --save_steps 200000000 \
    --warmup_ratio 0.0 \
    --learning_rate ${LEARNING_RATE:-1.25e-5} \
    --save_total_limit 1 \
    --plot_loss \
    --seed 42 \
    --ddp_timeout 1800000000 \
    --bf16 \
    --include_num_input_tokens_seen \
    --max_steps ${MAX_STEPS:-1}
