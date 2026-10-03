#!/bin/bash
# Bailing MoE V3 VL (M)LLM SFT with torchturbo via ms-swift
#
# Usage:
#   bash internal/scripts/runs/bailing_moe_v3_vl/install_swift_deps.sh
#   bash internal/scripts/runs/bailing_moe_v3_vl/swift_mllm_sft.sh

set -o pipefail
set -eux

[ -n "$USER_NAME" ] || { echo "Error: USER_NAME is not set" >&2; exit 1; }

# model name or path
# Bailing VL tiny weights
# MODEL=${MODEL:-/mnt/sfs_turbo/pretrained_models/tiny_v3_vl_256k_enhance_formal_select28}
MODEL=${MODEL:-/mnt/aicompiler-a3/common/ckpt/common_base_model/ling_vl_flash_midtrain_model_merge_select36}
# sequence length
SEQ_LEN=${SEQ_LEN:-131072}
# output directory
OUTPUT_DIR=${OUTPUT_DIR:-/sfs_turbo/${USER_NAME}/ckpt/$(basename "${MODEL}")/swift/mllm_sft_node${NNODES:-1}}
# tuner type: 'full' | 'lora'
FINETUNING_TYPE=${FINETUNING_TYPE:-full}
# training dataset (swift dataset id or a local path)
DATASET=${DATASET:-/mnt/sfs_turbo/torchturbo/datasets/debug_data/data_0630/sampled_sft_data_count_151372_0630.jsonl}
# turbo engine flags (sp/ep, freeze, mtp, ...) live in the sibling yaml.
TT_CONFIG=${TT_CONFIG:-"$(dirname "$0")/swift_mllm_sft.yaml"}

dataset_num_proc=96
dataloader_num_workers=8       # swift args: --dataloader_num_workers
model_type=turbo_bailing_moe_v3_vl   # turbo-registered model_type (loader = TurboModelLoader)
template=bailing_moe_v3_vl           # swift template for bailing moe v3 vl

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_EVENT_TIMEOUT=14500
export ACL_DEVICE_SYNC_TIMEOUT=14400
export HCCL_OP_EXPANSION_MODE="AI_CPU"
export ASCEND_GLOBAL_LOG_LEVEL=3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_IF_BASE_PORT=29972
export GLOO_SOCKET_IFNAME=eth0
export GLOO_LOG_LEVEL=DEBUG
export MULTI_STREAM_MEMORY_REUSE=1 # 多流
export TENSORBOARD_LOGGING_DIR=/home/admin/logs/tfevent
export NNODES=${NNODES:-1}
export NODE_RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-23556}
export NPUS_PER_NODE=${NPUS_PER_NODE:-16}
export NPROC_PER_NODE=${NPROC_PER_NODE:-${NPUS_PER_NODE}}
export WORLD_SIZE=$((${NPUS_PER_NODE} * ${NNODES}))
export WANDB_DISABLED=True
export TRANSFORMERS_VERBOSITY=error
export ASCEND_PROCESS_LOG_PATH=$OUTPUT_DIR/log
export LING_SFT_AUTO_THINKING=${LING_SFT_AUTO_THINKING:-1}

# python interpreter (defaults to python3)
PYTHON=${PYTHON:-python3}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUN_SWIFT="${REPO_ROOT}/scripts/runs/run_swift.py"
# ms-swift repo root (the dir containing the `swift` package); empty if pip-installed.
SWIFT_SRC_DIR=${SWIFT_SRC_DIR:-}
export PYTHONPATH="${REPO_ROOT}/src${SWIFT_SRC_DIR:+:${SWIFT_SRC_DIR}}${PYTHONPATH:+:${PYTHONPATH}}"

export HF_DATASETS_CACHE=/mnt/sfs_turbo/${USER_NAME}/datasets/cache/swift/sampled_sft_data_count_151372_0630

DISTRIBUTED_ARGS="
    --nproc_per_node ${NPROC_PER_NODE:-${NPUS_PER_NODE:-16}} \
    --nnodes ${NNODES:-1} \
    --node_rank ${NODE_RANK:-0} \
    --master_addr=${MASTER_ADDR:-127.0.0.1} \
    --master_port=${MASTER_PORT:-23556}
"

mkdir -p $OUTPUT_DIR
LOG_FILE=$OUTPUT_DIR/log-$NODE_RANK.txt
if [ "${NODE_RANK:-0}" -eq 0 ]; then
    cp -f "$0" $OUTPUT_DIR/
fi
{ date "+%Y%m%d-%H%M%S"; echo "$OUTPUT_DIR"; } >> $LOG_FILE

"${PYTHON}" -m torch.distributed.run \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    "${RUN_SWIFT}" \
    --model_type "${model_type}" \
    --model "${MODEL}" \
    --check_model false \
    --template "${template}" \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --dataset "${DATASET}" \
    --load_from_cache_file true \
    --max_length "${SEQ_LEN}" \
    --max_pixels 3145728 \
    --packing true \
    --packing_length "${SEQ_LEN}" \
    --attn_impl flash_attn \
    --dataset_num_proc "${dataset_num_proc}" \
    --tuner_type "${FINETUNING_TYPE}" \
    --tt_config ${TT_CONFIG} \
    --num_train_epochs 5 \
    --seed 42 \
    --data_seed 42 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr":1e-7}' \
    --warmup_ratio 0.05 \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 2 \
    --bf16 true \
    --dataloader_num_workers "${dataloader_num_workers}" \
    --report_to tensorboard \
    --output_dir "${OUTPUT_DIR}" \
    --ddp_timeout 1800000000 \
    2>&1 | tee "${LOG_FILE}"
