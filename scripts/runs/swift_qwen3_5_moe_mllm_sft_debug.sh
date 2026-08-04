#!/bin/sh
# Qwen3.5-MoE (M)LLM SFT with torchturbo via ms-swift — turbo接管 model build + training.
# Ascend NPU counterpart of gpu_swift_qwen3_5_moe_mllm_sft_debug.sh.
#

set -o pipefail
set -ex

# --- Ascend env (mirrors lf_qwen3_5_moe_mllm_sft.sh; replaces the GPU
#     PYTORCH_CUDA_ALLOC_CONF / NCCL_DEBUG block) ----------------------------
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export ASCEND_GLOBAL_LOG_LEVEL=3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_IF_BASE_PORT=29972
export GLOO_SOCKET_IFNAME=eth0
export GLOO_LOG_LEVEL=DEBUG
export MULTI_STREAM_MEMORY_REUSE=1 # 多流
# MLLM data-processing caps (read by swift's mm_plugin, not backend-specific).
# Defaults here mirror the GPU debug script; override via the environment for
# one-off adjustments.
export MAX_PIXELS=${MAX_PIXELS:-3145728}
export VIDEO_MAX_PIXELS=${VIDEO_MAX_PIXELS:-50176}
export FPS_MAX_FRAMES=${FPS_MAX_FRAMES:-12}
export IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-3072}
export VIDEO_MAX_TOKEN_NUM=${VIDEO_MAX_TOKEN_NUM:-49}
export HCCL_EVENT_TIMEOUT=7200
export ASCEND_LAUNCH_BLOCKING=1
export ACL_DEVICE_SYNC_TIMEOUT=3600

if [ -z "$USER_NAME" ]; then
    echo "Error: USER_NAME is not set"
    exit 1
fi
# --- model / data -----------------------------------------------------------
# NPU-side paths (same checkpoint family the GPU debug script uses, just the
# NPU-side /sfs_turbo mount — see lf_qwen3_5_moe_mllm_sft.sh).

# model_name_or_path=/sfs_turbo/pretrained_models/Qwen3.5-397B-A17B
model_name_or_path=/sfs_turbo/pretrained_models/Qwen3.5-35B-A3B

# swift MLLM SFT dataset id (the NPU debug MLLM dataset, same one lf_qwen3_5_moe
# _mllm_sft.sh targets). For an offline NPU box, override DATASET to a local
# .jsonl path swift can load directly, e.g.:
#   DATASET=/mnt/sfs_turbo/.../sampled_sft_data_0630_fixed.jsonl bash <this>
#dataset=/mnt/aicompiler-a3/common/data/MedOneModel_debug_data/data_0630/sampled_sft_data_count_151372_0630_fixed.jsonl
dataset=/mnt/sfs_turbo/torchturbo/datasets/sampled_sft_data_count_151372_0630_fixed.jsonl
#dataset=/mnt/sfs_turbo/data/data/gsm8k/gsm8k_jsonl/gsm8k_train.jsonl

dataset_num_proc=96            # swift --dataset_num_proc. multiprocess dataset.map is now safe:
                                # TurboSwiftSft / TurboSftArguments are module-level classes, so
                                # dill pickles them by reference (no by-value save_type -> no
                                # "_abc._abc_data"). If the map STILL fails with that error, re-run
                                # with TURBO_DIAG_PICKLE=1 (run_swift.py) to pinpoint the real
                                # unpicklable leaf.
dataloader_num_workers=8       # swift args: --dataloader_num_workers

model_type=turbo_qwen3_5_moe   # turbo-registered model_type (loader = TurboModelLoader)
template=qwen3_5               # swift's built-in (M)LLM template for qwen3.5

seq_length=$((1024 * 4))
tuner_type=full                # swift args: 'full' | 'lora' (megatron uses --finetune)

# --- distributed-launch env (fed to torchrun below; swift's BaseArguments
#     also reads NPROC_PER_NODE / RANK / WORLD_SIZE for its own dist setup) ---
export NNODES=${NNODES:-1}
export NODE_RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-23556}
export NPUS_PER_NODE=${NPUS_PER_NODE:-16}
export NPROC_PER_NODE=${NPROC_PER_NODE:-${NPUS_PER_NODE}}
export WORLD_SIZE=$((${NPUS_PER_NODE} * ${NNODES}))
# NOTE: no CUDA_VISIBLE_DEVICES on NPU — torch_npu + transfer_to_npu bind each
# rank to its NPU from LOCAL_RANK (see NPU specifics in the header).

# turbo-owned parallelism (engine/backend FSDP2 plan). NOTE: turbo owns
# sequence + expert parallelism, so swift's `--sequence_parallel` is left off.
# Qwen3.5-35B-A3B: linear_num_value_heads=32, max SP=32.
sp_size=4
ep_size=8

output_dir=/sfs_turbo/${USER_NAME}/ckpt/qwen35_35B/swift_sft_sp${sp_size}ep${ep_size}_node${NNODES}
mkdir -p "$output_dir"
log_file=$output_dir/log-$NODE_RANK.txt
export ASCEND_PROCESS_LOG_PATH=${output_dir}/ascend_log
cp -r "$0" "$output_dir"
{ date "+%Y%m%d-%H%M%S"; echo "$output_dir"; } >> "$log_file"

# --- paths: make torchturbo + ms-swift importable ---------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
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

"${PYTHON}" -m torch.distributed.run \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    "${RUN_SWIFT}" \
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
    --logging_dir /home/admin/logs/tfevent \
    --output_dir "${output_dir}" \
    --ddp_timeout 1800000000 \
    --tt_sp_size "${sp_size}" \
    --tt_ep_size "${ep_size}" \
    --tt_freeze_vision_tower true \
    --tt_freeze_multi_modal_projector true \
    --tt_ac_use_reentrant \
    2>&1 | tee "$log_file"
