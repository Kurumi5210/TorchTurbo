#!/bin/bash
# Qwen3.5-MoE MLLM CPT with torchturbo via ms-swift (pt stage). The xg corpus
# must be converted offline (internal/scripts/dataset/convert_lf_to_swift.py);
# the msswift data_check gate rejects unconverted rows at startup.
#
# Usage:
#   bash internal/scripts/runs/qwen3_5_moe/install_swift_deps.sh
#   bash internal/scripts/runs/qwen3_5_moe/swift_mllm_pt.sh

set -o pipefail
set -eux

[ -n "$USER_NAME" ] || { echo "Error: USER_NAME is not set" >&2; exit 1; }


# model name or path
MODEL=${MODEL:-/sfs_turbo/pretrained_models/Qwen3.5-35B-A3B}
# output directory
OUTPUT_DIR=${OUTPUT_DIR:-/sfs_turbo/${USER_NAME}/ckpt/qwen35_35B/swift_mllm_pt}
# sequence length
SEQ_LEN=${SEQ_LEN:-4096}
# tuner type: 'full' | 'lora'
FINETUNING_TYPE=${FINETUNING_TYPE:-full}
# training dataset (path to a local swift-standard jsonl, see header)
DATASET=${DATASET:-/mnt/sfs_turbo/torchturbo/datasets/xg/swift_data_output_full.jsonl}
# turbo engine flags (sp/ep, freeze, AC, moe aux loss, chunk loss, MFU) live in
TT_CONFIG=${TT_CONFIG:-"$(dirname "$0")/swift_mllm_pt.yaml"}


source /usr/local/Ascend/ascend-toolkit/set_env.sh

export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_EVENT_TIMEOUT=14500
export ACL_DEVICE_SYNC_TIMEOUT=14400
# AI_CPU is default for A3; AIV does not support multi-comm-domain parallelism
export HCCL_OP_EXPANSION_MODE="AI_CPU"
export ASCEND_GLOBAL_LOG_LEVEL=3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_IF_BASE_PORT=29972
export GLOO_SOCKET_IFNAME=eth0
export MULTI_STREAM_MEMORY_REUSE=1
export TENSORBOARD_LOGGING_DIR=/home/admin/logs/tfevent
# NOTE: no MAX_PIXELS-style env here -- the installed qwen_vl_utils hardcodes
# those constants and ignores the env; the image cap is set via --max_pixels.

export WANDB_DISABLED=True
export TRANSFORMERS_VERBOSITY=error
# distributed-launch env (swift's BaseArguments also reads NPROC_PER_NODE /
# RANK / WORLD_SIZE for its own dist setup)
export NNODES=${NNODES:-1}
export NODE_RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-23556}
export NPUS_PER_NODE=${NPUS_PER_NODE:-16}
export NPROC_PER_NODE=${NPROC_PER_NODE:-${NPUS_PER_NODE}}
export WORLD_SIZE=$((${NPUS_PER_NODE} * ${NNODES}))
export ASCEND_PROCESS_LOG_PATH=$OUTPUT_DIR/ascend_log

# python interpreter (defaults to python3)
PYTHON=${PYTHON:-python3}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUN_SWIFT="${REPO_ROOT}/scripts/runs/run_swift.py"
# ms-swift repo root (the dir containing the `swift/` package); empty if
# ms-swift is pip-installed.
SWIFT_SRC_DIR=${SWIFT_SRC_DIR:-}
export PYTHONPATH="${REPO_ROOT}/src${SWIFT_SRC_DIR:+:${SWIFT_SRC_DIR}}${PYTHONPATH:+:${PYTHONPATH}}"

DISTRIBUTED_ARGS="
    --nproc_per_node ${NPROC_PER_NODE} \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT}
"

mkdir -p $OUTPUT_DIR
LOG_FILE=$OUTPUT_DIR/log-$NODE_RANK.txt
if [ "$NODE_RANK" -eq 0 ]; then
    cp -f "$0" $OUTPUT_DIR/
fi
{ date "+%Y%m%d-%H%M%S"; echo "$OUTPUT_DIR"; } >> $LOG_FILE

# --max_pixels 589824 (=768*768): LF's mm_plugin falls back to 768*768 when the
# processor has no image_max_pixels (the case for Qwen3.5); match it here.
"${PYTHON}" -m torch.distributed.run $DISTRIBUTED_ARGS "${RUN_SWIFT}" \
    pt \
    --model_type turbo_qwen3_5_moe \
    --model $MODEL \
    --template qwen3_5 \
    --dataset $DATASET \
    --load_from_cache_file true \
    --max_length ${SEQ_LEN} \
    --packing true \
    --packing_length ${SEQ_LEN} \
    --max_pixels 589824 \
    --attn_impl flash_attn \
    --dataset_num_proc 96 \
    --tuner_type ${FINETUNING_TYPE} \
    --dataset_shuffle false \
    --train_dataloader_shuffle false \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1.25e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 2 \
    --bf16 true \
    --dataloader_num_workers 8 \
    --max_steps 200 --save_strategy no --report_to tensorboard \
    --output_dir $OUTPUT_DIR \
    --ddp_timeout 1800000000 \
    --tt_config ${TT_CONFIG} \
    2>&1 | tee $LOG_FILE
