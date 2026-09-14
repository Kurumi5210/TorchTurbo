#!/bin/bash
# Qwen3.5-MoE MLLM SFT with torchturbo
#   mkdir -p /mnt/medmm-s3/common/data/debug_data
#   ln -s /sfs_turbo/zhisan/datasets/mnt/medmm-s3/common/data/debug_data/images /mnt/medmm-s3/common/data/debug_data/images
#
# Usage:
#   bash internal/scripts/runs/qwen3_5_moe/install_lf_deps.sh
#   bash internal/scripts/runs/qwen3_5_moe/lf_mllm_sft_med.sh

set -o pipefail
set -eux

[ -n "$USER_NAME" ] || { echo "Error: USER_NAME is not set" >&2; exit 1; }

# model name or path
MODEL=${MODEL:-/mnt/aicompiler-a3/common/pretrained_models/Qwen3.5-35B-A3B}
# sequence length
SEQ_LEN=${SEQ_LEN:-32768}
# output directory
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/sfs_turbo/${USER_NAME}/ckpt/$(basename "${MODEL}")/lf/mllm_sft_med_seq$((${SEQ_LEN}/1024))k_node${NNODES:-1}}
# tuner type: 'full' | 'lora'
FINETUNING_TYPE=${FINETUNING_TYPE:-full}
# training dataset (LF dataset id resolved via --dataset_dir, or a local path)
DATASET=${DATASET:-med_debug}
# turbo engine flags (sp/ep, freeze, mtp, ...) live in the sibling yaml.
TT_CONFIG=${TT_CONFIG:-"$(dirname "$0")/lf_mllm_sft_med.yaml"}

packing_mode=packing_sep

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
export MULTI_STREAM_MEMORY_REUSE=2 # 多流
export VIDEO_PLACEHOLDER="<qwen3_5_video_placeholder>"
export AUDIO_PLACEHOLDER="<qwen3_5_audio_placeholder>"
export WANDB_DISABLED=True
export ALLOW_EXTRA_ARGS=True
export TRANSFORMERS_VERBOSITY=error
export NNODES=${NNODES:-1}
export NPUS_PER_NODE=${NPUS_PER_NODE:-16}
export NODE_RANK=${RANK:-0}
export WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-23556}
export ASCEND_PROCESS_LOG_PATH=$OUTPUT_DIR/log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUN_LF="${REPO_ROOT}/scripts/runs/run_lf.py"
DATASET_DIR="${REPO_ROOT}/internal/dataset/npu/"
TORCHTURBO_SRC_DIR="${TORCHTURBO_SRC_DIR:-${REPO_ROOT}/src}"
[ -d "${TORCHTURBO_SRC_DIR}/torchturbo" ] || { echo "bad TORCHTURBO_SRC_DIR: ${TORCHTURBO_SRC_DIR}" >&2; exit 1; }
export PYTHONPATH="${TORCHTURBO_SRC_DIR}:/opt/MindSpeed${PYTHONPATH:+:${PYTHONPATH}}"

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

torchrun $DISTRIBUTED_ARGS $RUN_LF \
    --tt_config ${TT_CONFIG} \
    --num_train_epochs 1 \
    --report_to tensorboard \
    --logging_dir /home/admin/logs/tfevent \
    --gradient_checkpointing \
    --dataset_dir ${DATASET_DIR} \
    --max_steps 10000 \
    --trust_remote_code \
    --disable_shuffling \
    --stage sft \
    --model_name_or_path ${MODEL} \
    --do_train \
    --flash_attn fa2 \
    --dataset ${DATASET} \
    --preprocessing_num_workers 32 \
    --cutoff_len ${SEQ_LEN} \
    --neat_packing \
    --template qwen3_5_nothink \
    --finetuning_type ${FINETUNING_TYPE} \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --overwrite_cache \
    --save_strategy "steps" \
    --eval_steps 200000000 \
    --save_steps 200000000 \
    --warmup_ratio 0.0 \
    --learning_rate 1.25e-5 \
    --save_total_limit 1 \
    --plot_loss \
    --seed 42 \
    --bf16 \
    --ddp_timeout 1800000000 2>&1 | tee ${LOG_FILE}

# qwen3_5_nothink 32k sp4ep16
# {'loss': 2.625, 'grad_norm': 175.26144409179688, 'learning_rate': 1.25e-05, 'epoch': 0.0}
# {'loss': 0.7847, 'grad_norm': 43.624210357666016, 'learning_rate': 1.2499999691574866e-05, 'epoch': 0.0}
# {'loss': 0.326, 'grad_norm': 10.339645385742188, 'learning_rate': 1.2499998766299491e-05, 'epoch': 0.01}
# {'loss': 0.9221, 'grad_norm': 18.688390731811523, 'learning_rate': 1.2499997224173967e-05, 'epoch': 0.01}
# {'loss': 1.1755, 'grad_norm': 21.321134567260742, 'learning_rate': 1.249999506519845e-05, 'epoch': 0.01}
# {'loss': 1.0345, 'grad_norm': 8.787310600280762, 'learning_rate': 1.2499992289373147e-05, 'epoch': 0.01}
# {'loss': 0.4557, 'grad_norm': 5.321507453918457, 'learning_rate': 1.2499988896698336e-05, 'epoch': 0.02}
# {'loss': 0.2572, 'grad_norm': 4.364945411682129, 'learning_rate': 1.2499984887174352e-05, 'epoch': 0.02}
# {'loss': 0.2287, 'grad_norm': 3.510512590408325, 'learning_rate': 1.249998026080159e-05, 'epoch': 0.02}
# {'loss': 0.1914, 'grad_norm': 2.5690150260925293, 'learning_rate': 1.2499975017580503e-05, 'epoch': 0.02}

# qwen3_5 32k sp4ep16
# {'loss': 1.5168, 'grad_norm': 190.84646606445312, 'learning_rate': 1.25e-05, 'epoch': 0.0}
# {'loss': 0.2592, 'grad_norm': 27.695114135742188, 'learning_rate': 1.2499999691574866e-05, 'epoch': 0.0}
# {'loss': 0.2037, 'grad_norm': 14.601760864257812, 'learning_rate': 1.2499998766299491e-05, 'epoch': 0.01}
# {'loss': 0.5064, 'grad_norm': 24.964548110961914, 'learning_rate': 1.2499997224173967e-05, 'epoch': 0.01}
# {'loss': 0.746, 'grad_norm': 29.61608123779297, 'learning_rate': 1.249999506519845e-05, 'epoch': 0.01}
# {'loss': 0.6617, 'grad_norm': 18.157182693481445, 'learning_rate': 1.2499992289373147e-05, 'epoch': 0.01}
# {'loss': 0.2778, 'grad_norm': 4.864065170288086, 'learning_rate': 1.2499988896698336e-05, 'epoch': 0.02}
# {'loss': 0.151, 'grad_norm': 2.9993717670440674, 'learning_rate': 1.2499984887174352e-05, 'epoch': 0.02}
# {'loss': 0.1057, 'grad_norm': 1.7375272512435913, 'learning_rate': 1.249998026080159e-05, 'epoch': 0.02}
# {'loss': 0.0986, 'grad_norm': 1.3906617164611816, 'learning_rate': 1.2499975017580503e-05, 'epoch': 0.02}
