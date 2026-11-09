#!/usr/bin/env bash
# Run the same tau2 SFT recipe against this checkout in full or LoRA mode.
set -euo pipefail
mode=${1:?Use run_full.sh or run_lora.sh}
shift
case ${mode} in full|lora) ;; *) echo 'Expected full or lora.' >&2; exit 1 ;; esac
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=${TT_REPO:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}
PYTHON=${PYTHON:-python3}
MODEL=${MODEL:?Set MODEL in the job configuration}
DATA_DIR=${DATA_DIR:-${REPO_ROOT}/data/tau2}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/outputs/tau2_mfu_${mode}}
TT_CONFIG=${TT_CONFIG:-${SCRIPT_DIR}/shared.yaml}
MAX_STEPS=${MAX_STEPS:--1}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-${EPOCHS:-3}}
SEQ_LEN=${SEQ_LEN:-32768}
SEED=${SEED:-42}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
SAVE_STRATEGY=${SAVE_STRATEGY:-no}

export NNODES=${NNODES:?Set NNODES in the job configuration}
export NPROC_PER_NODE=${NPROC_PER_NODE:-${NPUS_PER_NODE:-}}
: "${NPROC_PER_NODE:?Set NPROC_PER_NODE or NPUS_PER_NODE in the job configuration}"
export NODE_RANK=${NODE_RANK:-${RANK:-}}
: "${NODE_RANK:?Set NODE_RANK or RANK in the job configuration}"
export MASTER_ADDR=${MASTER_ADDR:-${MASTER_IP:-}}
: "${MASTER_ADDR:?Set MASTER_ADDR or MASTER_IP in the job configuration}"
export MASTER_PORT=${MASTER_PORT:-23560}
export WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
export PYTHONPATH="${REPO_ROOT}/src${SWIFT_SRC_DIR:+:${SWIFT_SRC_DIR}}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export ENABLE_THINKING=false
export WANDB_DISABLED=true
export HF_HOME=${OUTPUT_DIR}/cache/huggingface
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HF_MODULES_CACHE=${HF_HOME}/modules
export ASCEND_PROCESS_LOG_PATH=${OUTPUT_DIR}/ascend-logs
HXB_MANIFEST=${OUTPUT_DIR}/prepared/node-${NODE_RANK}/manifest.json

prepare_command=(
    "${PYTHON}" "${SCRIPT_DIR}/prepare_hxb_data.py"
    --train "${TRAIN_FILE:-${DATA_DIR}/train.jsonl}"
    --valid "${VALID_FILE:-${DATA_DIR}/valid.jsonl}"
    --model "${MODEL}" --output-dir "${OUTPUT_DIR}/prepared/node-${NODE_RANK}"
    --tt-config "${TT_CONFIG}" --max-length "${SEQ_LEN}"
)
command=(
    "${PYTHON}" -m torch.distributed.run
    --nproc_per_node "${NPROC_PER_NODE}" --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}" --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}"
    --no_python bash "${SCRIPT_DIR}/run_sft_worker.sh" /tmp
    "${PYTHON}" "${REPO_ROOT}/scripts/runs/run_swift.py" sft
    --model_type turbo_bailing_hybrid --model "${MODEL}" --check_model false
    --template ling_v3 --enable_thinking false --add_non_thinking_prefix true
    --loss_scale ignore_empty_think --dataset hxb_sft
    --tt_plugin "${SCRIPT_DIR}/hxb_sft_plugin.py:${HXB_MANIFEST}"
    --split_dataset_ratio 0 --eval_strategy no --do_eval false
    --load_from_cache_file false --strict true
    --max_length "${SEQ_LEN}" --packing true --packing_length "${SEQ_LEN}" --packing_num_proc 1
    --attn_impl flash_attn --dataset_num_proc 1
    --tuner_type "${mode}" --tt_config "${TT_CONFIG}"
    --per_device_train_batch_size 1 --gradient_accumulation_steps 1
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" --max_steps "${MAX_STEPS}"
    --seed "${SEED}" --data_seed "${SEED}"
    --optim adamw_torch --learning_rate "${LEARNING_RATE}" --lr_scheduler_type constant
    --warmup_steps 0 --warmup_ratio 0 --weight_decay 0
    --adam_beta1 0.9 --adam_beta2 0.999 --adam_epsilon 1e-8 --max_grad_norm 1.0
    --logging_steps 1 --logging_first_step true --save_strategy "${SAVE_STRATEGY}"
    --bf16 true --dataloader_num_workers 0 --report_to tensorboard
    --output_dir "${OUTPUT_DIR}" --ddp_timeout 7200
)
if [[ ${mode} == lora ]]; then
    command+=(
        --lora_rank 8 --lora_alpha 16 --lora_dropout 0
        --target_modules q_proj k_proj v_proj o_proj kv_a_proj_with_mqa kv_b_proj
        --use_rslora false --use_dora false
    )
fi
command+=("$@")
if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf '[tau2-mfu] prepare: '
    printf '%q ' "${prepare_command[@]}"
    printf '\n'
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${OUTPUT_DIR}/bootstrap-node-${NODE_RANK}.log") 2>&1
printf '[tau2-mfu] repo=%s mode=%s output=%s\n' "${REPO_ROOT}" "${mode}" "${OUTPUT_DIR}"
git -c "safe.directory=${REPO_ROOT}" -C "${REPO_ROOT}" rev-parse HEAD
if [[ ${NODE_RANK} == 0 ]]; then
    cp "$0" "${TT_CONFIG}" "${OUTPUT_DIR}/"
fi
"${prepare_command[@]}"
printf '%q ' "${command[@]}" > "${OUTPUT_DIR}/command-node-${NODE_RANK}.sh"
printf '\n' >> "${OUTPUT_DIR}/command-node-${NODE_RANK}.sh"

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_EVENT_TIMEOUT=14500
export ACL_DEVICE_SYNC_TIMEOUT=14400
export HCCL_OP_EXPANSION_MODE=AI_CPU
export ASCEND_GLOBAL_LOG_LEVEL=3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-29972}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export MULTI_STREAM_MEMORY_REUSE=1
"${command[@]}" 2>&1 | tee "${OUTPUT_DIR}/train-node-${NODE_RANK}.log"
