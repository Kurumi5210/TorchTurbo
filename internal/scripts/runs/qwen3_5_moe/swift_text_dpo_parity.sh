#!/bin/bash
# Compare native Swift and TorchTurbo DPO on a tiny Qwen3.5-MoE model.
set -o pipefail
set -eux

BACKEND=${BACKEND:?native|turbo}
[ -n "$USER_NAME" ] || { echo "Error: USER_NAME is not set" >&2; exit 1; }

MODEL=${MODEL:-/sfs_turbo/chenxiao/RLAscend/tiny-qwen35-4layer}
DATASET=${DATASET:-/sfs_turbo/chenxiao/RLAscend/dpo_parity/orca_32.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-/sfs_turbo/${USER_NAME}/ckpt/qwen35_35B/dpo_parity_${BACKEND}}
BF16=${BF16:-true}
TT_CONFIG="$(dirname "$0")/swift_text_dpo_parity.yaml"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
export ASCEND_GLOBAL_LOG_LEVEL=3
export WANDB_DISABLED=True
export TRANSFORMERS_VERBOSITY=error

PYTHON=${PYTHON:-python3}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

COMMON_ARGS="
    --rlhf_type dpo \
    --model $MODEL \
    --template qwen3_5 \
    --dataset $DATASET \
    --max_length 2048 \
    --attn_impl flash_attn \
    --tuner_type full \
    --dataset_shuffle false \
    --train_dataloader_shuffle false \
    --seed 42 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0 \
    --logging_steps 1 \
    --save_strategy no \
    --bf16 ${BF16} \
    --dataloader_num_workers 0 \
    --max_steps 10 \
    --router_aux_loss_coef 0 \
    --output_dir $OUTPUT_DIR \
    --ddp_timeout 1800000000
"

mkdir -p $OUTPUT_DIR
LOG_FILE=$OUTPUT_DIR/log.txt

if [ "$BACKEND" = "native" ]; then
    # Native Swift path.
    swift rlhf \
        --model_type qwen3_5_moe \
        $COMMON_ARGS \
        2>&1 | tee $LOG_FILE
else
    "${PYTHON}" -m torch.distributed.run --nproc_per_node 1 \
        "${REPO_ROOT}/scripts/runs/run_swift.py" \
        dpo \
        --model_type turbo_qwen3_5_moe \
        $COMMON_ARGS \
        --tt_config ${TT_CONFIG} \
        2>&1 | tee $LOG_FILE
fi
