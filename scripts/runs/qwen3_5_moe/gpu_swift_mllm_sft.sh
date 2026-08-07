#!/usr/bin/env bash
# Qwen3.5-MoE (M)LLM SFT with torchturbo via ms-swift — turbo takeover model build + training.
#
# This script uses ms-swift's NATIVE HF/FSDP2 SFT path end to end (turbo takeover
# runs through swift's own SwiftSft pipeline + TrainerFactory + ModelLoader).
# It is the turbo takeover counterpart of the team's historical `megatron sft`
# script — same training intent, but using the HF-path argument vocabulary
# (see swift/trainers/arguments.py), NOT the megatron one. Key differences
# from the megatron script are called out inline.

set -o pipefail
set -ex

# --- env (shared with the megatron script where backend-agnostic) -----------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
# TensorBoard event log dir. HF's TensorBoardCallback reads ONLY this env var
# for its log_dir (the ``--logging_dir`` arg is ignored by the TB callback), so
# this is the single source of truth for where events.out.tfevents.* land.
# rank0 (is_world_process_zero) is the writer.
export TENSORBOARD_LOGGING_DIR=/home/admin/logs/tfevent
# MLLM data-processing caps (same as the megatron script; these are read by
# swift's mm_plugin, not backend-specific). Pinned to the exact values the
# `megatron sft` run used, so turbo filters the same rows the megatron path
# does. Defaults here mirror megatron; override via the environment for
# one-off adjustments.
export MAX_PIXELS=${MAX_PIXELS:-3145728}
export VIDEO_MAX_PIXELS=${VIDEO_MAX_PIXELS:-50176}
export FPS_MAX_FRAMES=${FPS_MAX_FRAMES:-12}
export IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-3072}
export VIDEO_MAX_TOKEN_NUM=${VIDEO_MAX_TOKEN_NUM:-49}

if [ -z "$USER_NAME" ]; then
    echo "Error: USER_NAME is not set"
    exit 1
fi

if [ ! -d /mnt/medmm-s3/common/data/debug_data ]; then
    mkdir -p /mnt/medmm-s3/common/data
    ln -s /input/torchturbo/stable/dataset/med /mnt/medmm-s3/common/data/debug_data
fi

# --- model / data -----------------------------------------------------------
#model_name_or_path=/pcache-mnt/ro/checkpoint/54229/345412/264967436/1/rootmodelQwen3.5-35B-A3B-Base-hgf-asys
model_name_or_path=/pcache-mnt/ro/checkpoint/54229/345412/265265966/1/rootmodelQwen3.5-397B-A17B-BASE-hgf-asys/
dataset=/mnt/medmm-s3/common/data/debug_data/data_0630/sampled_sft_data_count_151372_0630.jsonl
dataset_num_proc=96            # swift --dataset_num_proc. multiprocess dataset.map is now safe:
                                # TurboSwiftSft / TurboSftArguments are module-level classes, so
                                # dill pickles them by reference (no by-value save_type -> no
                                # "_abc._abc_data"). If the map STILL fails with that error, re-run
                                # with TURBO_DIAG_PICKLE=1 (run_swift.py) to pinpoint the real
                                # unpicklable leaf.
dataloader_num_workers=8       # swift args: --dataloader_num_workers

model_type=turbo_qwen3_5_moe   # turbo-registered model_type (loader = TurboModelLoader)
template=qwen3_5               # swift's built-in (M)LLM template for qwen3.5

seq_length=$((1024 * 32))
tuner_type=full                # swift args: 'full' | 'lora' (megatron uses --finetune)

# --- distributed-launch env (fed to torchrun below; swift's BaseArguments
#     also reads NPROC_PER_NODE / RANK / WORLD_SIZE for its own dist setup) ---
export NNODES=${WORLD_SIZE:-${NNODES:-1}}
export NODE_RANK=${RANK:-${NODE_RANK:-0}}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-23556}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# turbo-owned parallelism (engine/backend FSDP2 plan). NOTE: turbo owns
# sequence + expert parallelism, so swift's `--sequence_parallel` is left off.
sp_size=4
ep_size=8

output_dir=/input/asystem/${USER_NAME}/ckpt/qwen35_35B/swift_sft_sp${sp_size}ep${ep_size}_node${NNODES}
mkdir -p "$output_dir"
log_file=$output_dir/log-$NODE_RANK.txt
if [ "$NODE_RANK" -eq 0 ]; then
    cp -f "$0" "$output_dir/"
fi
{ date "+%Y%m%d-%H%M%S"; echo "$output_dir"; } >> "$log_file"

# --- paths: make torchturbo + ms-swift importable ---------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TORCHTURBO_SRC_DIR="${TORCHTURBO_SRC_DIR:-${REPO_ROOT}/src}"
PLUGIN="${TORCHTURBO_SRC_DIR}/torchturbo/adapters/msswift/plugin.py"
# ms-swift repo root (the dir containing the `swift/` package); leave empty if
# ms-swift is `pip install -e`'d.
SWIFT_SRC_DIR=${SWIFT_SRC_DIR}

[ -d "${TORCHTURBO_SRC_DIR}/torchturbo" ] || { echo "bad TORCHTURBO_SRC_DIR: ${TORCHTURBO_SRC_DIR}" >&2; exit 1; }
[ -f "${PLUGIN}" ] || { echo "plugin not found: ${PLUGIN}" >&2; exit 1; }

export PYTHONPATH="${TORCHTURBO_SRC_DIR}${SWIFT_SRC_DIR:+:${SWIFT_SRC_DIR}}:${PYTHONPATH}"
export WANDB_DISABLED=True
export TRANSFORMERS_VERBOSITY=error

PYTHON=${PYTHON:-python3}
# Launch the turbo worker entry directly under torchrun (the same flags swift's
# own cli_main re-exec passes). We bypass `swift sft` on purpose: run_swift.py
# imports the msswift plugin BEFORE sft_main, swapping SwiftSft -> TurboSwiftSft
# in time for TurboSftArguments to parse the `--tt_*` knobs. (`--external_plugins`
# fires too late — during BaseArguments.__post_init__, after sft_main has already
# started constructing plain SwiftSft — so --tt_* land in remaining_argv and are
# rejected.)
RUN_SWIFT="${REPO_ROOT}/scripts/runs/run_swift.py"
TORCHRUN_FLAGS=(--nproc_per_node "${NPROC_PER_NODE}"
                --nnodes "${NNODES}"
                --node_rank "${NODE_RANK}"
                --master_addr "${MASTER_ADDR}"
                --master_port "${MASTER_PORT}")

"${PYTHON}" -m torch.distributed.run "${TORCHRUN_FLAGS[@]}" "${RUN_SWIFT}" \
    --model_type "${model_type}" \
    --model "${model_name_or_path}" \
    --template "${template}" \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --dataset "${dataset}" \
    --load_from_cache_file true \
    --max_length "${seq_length}" \
    --packing true \
    --packing_length "${seq_length}" \
    --attn_impl flash_attn \
    --dataset_num_proc "${dataset_num_proc}" \
    --tuner_type "${tuner_type}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
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
    --output_dir "${output_dir}" \
    --ddp_timeout 1800000000 \
    --tt_sp_size "${sp_size}" \
    --tt_ep_size "${ep_size}" \
    --tt_freeze_vision_tower true \
    --tt_freeze_multi_modal_projector true \
    --tt_ac_use_reentrant \
    2>&1 | tee "$log_file"