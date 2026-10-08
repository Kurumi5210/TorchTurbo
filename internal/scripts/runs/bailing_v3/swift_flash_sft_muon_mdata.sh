#!/bin/bash
# Bailing V3 SFT with torchturbo via ms-swift and Muon optimizer
#
# Usage:
#   bash internal/scripts/runs/bailing_v3/install_swift_deps.sh
#   bash internal/scripts/runs/bailing_v3/swift_flash_sft_muon_mdata.sh

set -o pipefail
set -eux

[ -n "$USER_NAME" ] || { echo "Error: USER_NAME is not set" >&2; exit 1; }

# model name or path
MODEL=${MODEL:-/mnt/sfs_turbo/pretrained_models/ling_flash_v3_wothink_mix_antcc_select_32/}
# sequence length
SEQ_LEN=${SEQ_LEN:-$((1024 * 128))}
# output directory
OUTPUT_DIR=${OUTPUT_DIR:-/sfs_turbo/${USER_NAME}/ckpt/$(basename "${MODEL}")/swift/swift_flash_sft_muon_node${NNODES:-1}}
# tuner type: 'full' | 'lora'
FINETUNING_TYPE=${FINETUNING_TYPE:-full} # swift args: 'full' | 'lora' (megatron uses --finetune)
# turbo engine flags (sp/ep,mtp, muon optimizer, ...) live in the sibling yaml.
TT_CONFIG=${TT_CONFIG:-"$(dirname "$0")/swift_flash_sft_muon_mdata.yaml"}

dataset_num_proc=96
dataloader_num_workers=8          # swift args: --dataloader_num_workers
model_type=turbo_bailing_hybrid   # turbo-registered model_type (loader = TurboModelLoader)
template=ling_v3

# python interpreter (defaults to python3)
PYTHON=${PYTHON:-python3}

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUN_SWIFT="${REPO_ROOT}/scripts/runs/run_swift.py"
# ms-swift repo root (the dir containing the `swift` package); empty if pip-installed.
SWIFT_SRC_DIR=${SWIFT_SRC_DIR:-}
export PYTHONPATH="${REPO_ROOT}/src${SWIFT_SRC_DIR:+:${SWIFT_SRC_DIR}}${PYTHONPATH:+:${PYTHONPATH}}"

export HF_DATASETS_CACHE=/mnt/sfs_turbo/${USER_NAME}/datasets/cache/swift/science

# antllm_prepack plugin (resolved against REPO_ROOT here). The pack config is YOUR
# pack.yaml (`dataset` block: args/pack_args); the bound plugin is the source of truth.
PACK_PLUGIN="${REPO_ROOT}/internal/dataset/antllm_pack/antllm_pack_plugin.py"
PACK_YAML="${PACK_YAML:-${SCRIPT_DIR}/prepack_mdata_config.yaml}"   # DEMO: your pack config (dataset block)
[ -f "${PACK_PLUGIN}" ] || { echo "missing antllm_prepack plugin: ${PACK_PLUGIN}" >&2; exit 1; }
[ -f "${PACK_YAML}" ] || { echo "missing pack config, PACK_YAML: ${PACK_YAML}" >&2; exit 1; }
SAVE_PATH="$(python3 -c 'import sys,yaml;d=yaml.safe_load(open(sys.argv[1]))["dataset"];print(d.get("args",{}).get("save_path") or d.get("pack_args",{}).get("save_path"))' "${PACK_YAML}")"
[ -n "${SAVE_PATH}" ] || { echo "could not read dataset.args.save_path from ${PACK_YAML}" >&2; exit 1; }
echo "pack config: ${PACK_YAML}  dataset/save_path: ${SAVE_PATH}"

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
    --tt_plugin "${PACK_PLUGIN}:${PACK_YAML}" \
    --dataset "${SAVE_PATH}" \
    --load_from_cache_file true \
    --max_length "${SEQ_LEN}" \
    --packing_length "${SEQ_LEN}" \
    --attn_impl flash_attn \
    --dataset_num_proc "${dataset_num_proc}" \
    --tuner_type "${FINETUNING_TYPE}" \
    --tt_config ${TT_CONFIG} \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --seed 1234 \
    --data_seed 1234 \
    --lr_scheduler_type cosine \
    --warmup_steps 50 \
    --logging_steps 1 \
    --save_strategy epoch \
    --bf16 true \
    --dataloader_num_workers "${dataloader_num_workers}" \
    --report_to tensorboard \
    --output_dir "${OUTPUT_DIR}" \
    --ddp_timeout 1800000000 \
    2>&1 | tee "${LOG_FILE}"
