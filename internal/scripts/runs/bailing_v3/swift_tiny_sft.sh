#!/bin/sh
# Bailing V3 SFT with torchturbo via ms-swift

set -o pipefail
set -ex

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
export GLOO_LOG_LEVEL=DEBUG
export MULTI_STREAM_MEMORY_REUSE=1 # 多流
export TENSORBOARD_LOGGING_DIR=/home/admin/logs/tfevent


if [ -z "$USER_NAME" ]; then
    echo "Error: USER_NAME is not set"
    exit 1
fi
# --- model / data -----------------------------------------------------------

model_name_or_path=/mnt/sfs_turbo/pretrained_models/bailing_v3/ling_tiny_v3_formal_256k_enhance_0420_wthink/merge-step-mean-0baeb1658f223ec4bcfa8d7cf6b7e242/

dataset=/mnt/sfs_turbo/torchturbo/datasets/am_think/science.jsonl

dataset_num_proc=96
dataloader_num_workers=8          # swift args: --dataloader_num_workers

model_type=turbo_bailing_hybrid   # turbo-registered model_type (loader = TurboModelLoader)

# Template choice:
# - We ship a 'ling_v3' template byte-for-byte aligned to the ling v3 chat_template.jinja.
# - swift itself ships four bailing templates: ling / ling2 / ring2 / ring2_5.
# We compared all four on science dataset; for the current ling v3 checkpoint
# 'ring2' fits best (lowest loss), and is the selection below. Pick whichever template
# best matches your own dataset, or fall back to the one that simply yields the lower loss.
# The 5 templates differ mainly in:
# - thinking on/off: ring2 / ring2_5 / ling_v3 are thinking templates (is_thinking=True);
#   ling / ling2 are not. (ling2 still bakes a literal "detailed thinking off" into its
#   system, and ling_v3 injects a dynamic "detailed thinking on|off" directive; ling /
#   ring2 / ring2_5 carry no such directive.)
# - turn separator / suffix:
#     ling, ring2        -> chat_sep=[], suffix=''          (roles sit back-to-back)
#     ling2              -> '<|role_end|>'
#     ring2_5, ling_v3   -> '<|role_end|>\n\n'
# - newline spacing in system/prompt: ling / ling2 / ring2 keep the role tags flush (no
#   '\n'); ring2_5 and ling_v3 pad them with '\n' (e.g. prompt becomes
#   '<role>HUMAN</role>\n{{QUERY}}<|role_end|>\n\n<role>ASSISTANT</role>\n').
#
# Detailed results (step 1):
# ring2
# {'loss': 0.50526637, 'grad_norm': 2.29390478, 'learning_rate': 1.25e-05, 'aux_loss': 1.78196633, 'aux_loss_scaled': 1.78e-06, 'epoch': 0.00144928, 'global_step/max_steps': '1/690', 'elapsed_time': '22s', 'remaining_time': '4h 9m 48s', 'memory(GiB)': 40.27, 'train_speed(s/it)': 21.752806}
#
# ling2
# {'loss': 0.57480443, 'grad_norm': 3.27337599, 'learning_rate': 1.25e-05, 'aux_loss': 1.66832302, 'aux_loss_scaled': 1.67e-06, 'epoch': 0.001443, 'global_step/max_steps': '1/693', 'elapsed_time': '23s', 'remaining_time': '4h 22m 4s', 'memory(GiB)': 38.49, 'train_speed(s/it)': 22.723208}
#
# ring2_5
# {'loss': 0.53941792, 'grad_norm': 4.29061985, 'learning_rate': 1.25e-05, 'aux_loss': 1.75054086, 'aux_loss_scaled': 1.75e-06, 'epoch': 0.00144092, 'global_step/max_steps': '1/694', 'elapsed_time': '23s', 'remaining_time': '4h 27m 21s', 'memory(GiB)': 40.17, 'train_speed(s/it)': 23.146866}
#
# ling_v3
# {'loss': 0.62988997, 'grad_norm': 9.66740417, 'learning_rate': 1.25e-05, 'aux_loss': 1.66859701, 'aux_loss_scaled': 1.67e-06, 'epoch': 0.00143885, 'global_step/max_steps': '1/695', 'elapsed_time': '22s', 'remaining_time': '4h 9m 3s', 'memory(GiB)': 39.37, 'train_speed(s/it)': 21.531849}
template=ling_v3

seq_length=$((1024 * 32))
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
sp_size=4
ep_size=8

output_dir=/mnt/sfs_turbo/${USER_NAME}/ckpt/bailing_v3/swift_sft_sp${sp_size}ep${ep_size}_node${NNODES}
mkdir -p "$output_dir"
log_file=$output_dir/log-$NODE_RANK.txt
export ASCEND_PROCESS_LOG_PATH=${output_dir}/ascend_log
if [ "$NODE_RANK" -eq 0 ]; then
    cp -f "$0" "$output_dir/"
fi
{ date "+%Y%m%d-%H%M%S"; echo "$output_dir"; } >> "$log_file"

# --- paths: make torchturbo + ms-swift importable ---------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
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
    --output_dir "${output_dir}" \
    --ddp_timeout 1800000000 \
    --tt_sp_size "${sp_size}" \
    --tt_ep_size "${ep_size}" \
    --tt_moe_aux_loss_coeff 1e-6 \
    --tt_ac_use_reentrant \
    --tt_enable_chunk_loss \
    --tt_log_mfu \
    2>&1 | tee "$log_file"