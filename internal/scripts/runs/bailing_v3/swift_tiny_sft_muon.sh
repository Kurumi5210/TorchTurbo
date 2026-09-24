#!/bin/sh
# Bailing V3 SFT with torchturbo via ms-swift and the engine Muon optimizer

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

# antllm sft_pre_pack output dir (holds meta.json / encode_data / pack_data).
# Already tokenized, templated and packed, so it is fed through
# --tt_antllm_pack_dataset below instead of --dataset, and --packing is off.
dataset=/sfs_turbo/shanyicheng.syc/dataset/pre_pack_v4

dataset_num_proc=96
dataloader_num_workers=8          # swift args: --dataloader_num_workers

model_type=turbo_bailing_hybrid   # turbo-registered model_type (loader = TurboModelLoader)

template=ling_v3

seq_length=$((1024 * 128))
tuner_type=full                # swift args: 'full' | 'lora' (megatron uses --finetune)

# --- distributed-launch env (fed to torchrun below; swift's BaseArguments
#     also reads NPROC_PER_NODE / RANK / WORLD_SIZE for its own dist setup) ---
export NNODES=${NNODES:-16}
export NODE_RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-23556}
export NPUS_PER_NODE=${NPUS_PER_NODE:-16}
export NPROC_PER_NODE=${NPROC_PER_NODE:-${NPUS_PER_NODE}}
export WORLD_SIZE=$((${NPUS_PER_NODE} * ${NNODES}))
# NOTE: no CUDA_VISIBLE_DEVICES on NPU — torch_npu + transfer_to_npu bind each
# rank to its NPU from LOCAL_RANK (see NPU specifics in the header).

output_dir=/mnt/sfs_turbo/${USER_NAME}/ckpt/bailing_v3/swift_sft_muon1_sp8ep8_node${NNODES}
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
# ms-swift repo root (the dir containing the `swift/` package); leave empty if
# ms-swift is `pip install -e`'d.
SWIFT_SRC_DIR=${SWIFT_SRC_DIR}

[ -d "${TORCHTURBO_SRC_DIR}/torchturbo" ] || { echo "bad TORCHTURBO_SRC_DIR: ${TORCHTURBO_SRC_DIR}" >&2; exit 1; }

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
    --tt_antllm_pack_dataset "${dataset}" \
    --load_from_cache_file true \
    --max_length "${seq_length}" \
    --packing_length "${seq_length}" \
    --attn_impl flash_attn \
    --dataset_num_proc "${dataset_num_proc}" \
    --tuner_type "${tuner_type}" \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --seed 1234 \
    --data_seed 1234 \
    --lr_scheduler_type cosine \
    --warmup_steps 50 \
    --logging_steps 1 \
    --save_strategy epoch \
    --save_total_limit 50 \
    --bf16 true \
    --dataloader_num_workers "${dataloader_num_workers}" \
    --report_to tensorboard \
    --output_dir "${output_dir}" \
    --ddp_timeout 1800000000 \
    --tt_config "${SCRIPT_DIR}/swift_tiny_sft_muon.yaml" \
    2>&1 | tee "$log_file"
