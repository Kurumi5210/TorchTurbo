#!/bin/bash
# Qwen3.5-MoE text DPO short debug run; runs eight steps without checkpoints.

set -o pipefail
set -eux

[ -n "$USER_NAME" ] || { echo "Error: USER_NAME is not set" >&2; exit 1; }

# model name or path
MODEL=${MODEL:-/sfs_turbo/pretrained_models/Qwen3.5-35B-A3B}
# output directory
OUTPUT_DIR=${OUTPUT_DIR:-/sfs_turbo/${USER_NAME}/ckpt/qwen35_35B/swift_text_dpo_debug}
# sequence length
SEQ_LEN=${SEQ_LEN:-4096}
# tuner type: v1 supports 'full' only (independent ref model)
FINETUNING_TYPE=${FINETUNING_TYPE:-full}
# Preference dataset in Swift's chosen/rejected JSONL format.
DATASET=${DATASET:-/mnt/sfs_turbo/torchturbo/datasets/orca_dpo_pairs/swift_orca_rlhf.jsonl}
# turbo engine flags (sp/ep, AC, moe aux loss, MFU) live in the sibling yaml.
TT_CONFIG=${TT_CONFIG:-"$(dirname "$0")/swift_text_dpo_debug.yaml"}

source /usr/local/Ascend/ascend-toolkit/set_env.sh

export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_EVENT_TIMEOUT=14500
export ACL_DEVICE_SYNC_TIMEOUT=14400
# Use AI_CPU for A3 multi-communication domains.
export HCCL_OP_EXPANSION_MODE="AI_CPU"
export ASCEND_GLOBAL_LOG_LEVEL=3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_IF_BASE_PORT=29972
export GLOO_SOCKET_IFNAME=eth0
export MULTI_STREAM_MEMORY_REUSE=1
export TENSORBOARD_LOGGING_DIR=/home/admin/logs/tfevent

export WANDB_DISABLED=True
export TRANSFORMERS_VERBOSITY=error
# Distributed launch environment.
export NNODES=${NNODES:-1}
export NODE_RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-23566}
export NPUS_PER_NODE=${NPUS_PER_NODE:-16}
export NPROC_PER_NODE=${NPROC_PER_NODE:-${NPUS_PER_NODE}}
export WORLD_SIZE=$((${NPUS_PER_NODE} * ${NNODES}))
export ASCEND_PROCESS_LOG_PATH=$OUTPUT_DIR/ascend_log

# python interpreter (defaults to python3)
PYTHON=${PYTHON:-python3}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUN_SWIFT="${REPO_ROOT}/scripts/runs/run_swift.py"

# Optional ms-swift source root.
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

# The dpo stage uses the model path as its frozen reference by default.
"${PYTHON}" -m torch.distributed.run $DISTRIBUTED_ARGS "${RUN_SWIFT}" \
    dpo \
    --rlhf_type dpo \
    --model_type turbo_qwen3_5_moe \
    --model $MODEL \
    --template qwen3_5 \
    --dataset $DATASET \
    --load_from_cache_file true \
    --max_length ${SEQ_LEN} \
    --attn_impl flash_attn \
    --dataset_num_proc 96 \
    --tuner_type ${FINETUNING_TYPE} \
    --dataset_shuffle false \
    --train_dataloader_shuffle false \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --logging_steps 1 \
    --bf16 true \
    --dataloader_num_workers 8 \
    --max_steps 8 --save_strategy no --report_to tensorboard \
    --output_dir $OUTPUT_DIR \
    --ddp_timeout 1800000000 \
    --tt_config ${TT_CONFIG} \
    2>&1 | tee $LOG_FILE

# Verify zero initial rewards and ln(2) before updates.
if [ "$NODE_RANK" -eq 0 ]; then
    "${PYTHON}" - "$LOG_FILE" <<'EOF'
import math
import re
import sys

with open(sys.argv[1], encoding="utf-8", errors="replace") as fh:
    metric_lines = [ln for ln in fh if "'rewards/chosen':" in ln]
if not metric_lines:
    sys.exit("DPO step-1 gate: no metric lines found in the log")


def _val(line, key):
    m = re.search(rf"'{re.escape(key)}': ([-0-9.eE]+)", line)
    return float(m.group(1)) if m else None


first = metric_lines[0]
rc = _val(first, "rewards/chosen")
rr = _val(first, "rewards/rejected")
loss = _val(first, "loss")
print(f"DPO step-1 gate: rewards/chosen={rc} rewards/rejected={rr} loss={loss}")
if rc != 0.0 or rr != 0.0:
    sys.exit("DPO step-1 gate FAILED: nonzero rewards before the first update")
if loss is None or abs(loss - math.log(2)) > 1e-6:
    sys.exit("DPO step-1 gate FAILED: loss != ln2 before the first update")
print("DPO step-1 gate passed: ref == policy at init")
EOF
fi
