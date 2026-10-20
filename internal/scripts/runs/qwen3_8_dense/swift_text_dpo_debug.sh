#!/usr/bin/env bash
# Qwen3.8 DPO data alignment with Swift 4.4.2 and effective Megatron DP=2.
set -euo pipefail

: "${MODEL:?Set MODEL to the HF policy checkpoint (the GPU run used RL step 176).}"
: "${DATASET:?Set DATASET to business_user_flag_answer.json.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new run directory.}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON=${PYTHON:-python3}
REF_MODEL=${REF_MODEL:-${MODEL}}
TT_CONFIG=${TT_CONFIG:-${SCRIPT_DIR}/swift_text_dpo_debug.yaml}
SP_SIZE=${SP_SIZE:-16}
NNODES=${NNODES:-2}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
NODE_RANK=${NODE_RANK:-${RANK:-0}}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23566}
NUM_PROCESSES=$((NNODES * NPROC_PER_NODE))
if (( SP_SIZE < 1 || NUM_PROCESSES != 2 * SP_SIZE )); then
    echo "Exact DP batch comparison requires NNODES * NPROC_PER_NODE = 2 * SP_SIZE." >&2
    exit 1
fi

GLOBAL_BATCH_SIZE=24
MICRO_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=12
MAX_STEPS=${MAX_STEPS:--1}
DATA_TRACE_DIR=${DATA_TRACE_DIR:-${OUTPUT_DIR}/data-trace}
SWIFT_SRC_DIR=${SWIFT_SRC_DIR:-}
export PYTHONPATH="${REPO_ROOT}/src${SWIFT_SRC_DIR:+:${SWIFT_SRC_DIR}}${PYTHONPATH:+:${PYTHONPATH}}"

COMMAND=(
    "${PYTHON}" -m torch.distributed.run
    --nproc_per_node "${NPROC_PER_NODE}" --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}" --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}"
    "${REPO_ROOT}/scripts/runs/run_swift.py" dpo
    --rlhf_type dpo --model_type turbo_qwen3_5 --ref_model_type turbo_qwen3_5
    --model "${MODEL}" --ref_model "${REF_MODEL}"
    --template qwen3_8 --template_backend swift --agent_template qwen3_5
    --enable_thinking false --preserve_thinking true --add_non_thinking_prefix true
    --loss_scale last_round+ignore_empty_think
    --dataset "${DATASET}" --split_dataset_ratio 0.01
    --dataset_shuffle true --train_dataloader_shuffle true --seed 42 --data_seed 42
    --load_from_cache_file false --lazy_tokenize false
    --truncation_strategy delete --max_length 128000
    --packing true --padding_free true --packing_length 128000
    --packing_strategy binpack --packing_num_proc 1
    --dataset_num_proc 4 --dataloader_num_workers 2 --dataloader_drop_last true
    --dpo_megatron_data true --dpo_data_trace_dir "${DATA_TRACE_DIR}"
    --beta 0.1 --rpo_alpha 0.1 --label_smoothing 0.05 --loss_type sigmoid
    --sync_ref_model false
    --tuner_type full --num_train_epochs 1 --max_steps "${MAX_STEPS}"
    --per_device_train_batch_size "${MICRO_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --learning_rate 1e-6 --lr_scheduler_type cosine_with_min_lr
    --lr_scheduler_kwargs '{"min_lr":1e-7}' --warmup_ratio 0.05
    --optim adamw_torch --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-8
    --weight_decay 0.1 --max_grad_norm 1.0 --bf16 true --attn_impl flash_attn
    --freeze_llm false --freeze_vit true --freeze_aligner true
    --tt_freeze_language_model false --tt_freeze_vision_tower true --tt_freeze_multi_modal_projector true
    --tt_enable_mtp false --tt_sp_size "${SP_SIZE}" --tt_gbs "${GLOBAL_BATCH_SIZE}"
    --tt_config "${TT_CONFIG}"
    --save_strategy epoch --save_steps 20000 --save_only_model true
    --eval_strategy no --logging_steps 1 --report_to tensorboard
    --output_dir "${OUTPUT_DIR}" --add_version false --ddp_timeout 1800000000
)
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    COMMAND+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

# The dependency installer for this model pins Swift 4.4.2 and binpacking 1.5.2.
"${PYTHON}" - <<'PY'
import importlib.metadata
import swift
if swift.__version__ != "4.4.2":
    raise SystemExit(f"This backport requires Swift 4.4.2; imported {swift.__version__} from {swift.__file__}")
if importlib.metadata.version("binpacking") != "1.5.2":
    raise SystemExit("Install binpacking==1.5.2 to keep the packing algorithm fixed.")
PY

set +u
source "${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
set -u
export HCCL_CONNECT_TIMEOUT=7200 HCCL_EXEC_TIMEOUT=7200 ACL_DEVICE_SYNC_TIMEOUT=14400
export HCCL_OP_EXPANSION_MODE=AI_CPU HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-29972}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_GLOBAL_LOG_LEVEL=3 MULTI_STREAM_MEMORY_REUSE=1
export ASCEND_PROCESS_LOG_PATH="${OUTPUT_DIR}/ascend_log"
export IMAGE_MAX_TOKEN_NUM=1024 VIDEO_MAX_TOKEN_NUM=128 FPS_MAX_FRAMES=12
export TOKENIZERS_PARALLELISM=false

mkdir -p "${OUTPUT_DIR}"
if [[ "${NODE_RANK}" == 0 ]]; then
    cp "$0" "${TT_CONFIG}" "${OUTPUT_DIR}/"
    printf '%q ' "${COMMAND[@]}" > "${OUTPUT_DIR}/command.txt"
    export MODEL REF_MODEL DATASET OUTPUT_DIR REPO_ROOT
    "${PYTHON}" - <<'PY'
import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def checkpoint(path: str) -> dict:
    root = Path(path)
    names = ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
             "special_tokens_map.json", "chat_template.jinja", "model.safetensors.index.json")
    files = [root / name for name in names if (root / name).is_file()]
    if os.environ.get("HASH_MODEL_WEIGHTS") == "1":
        files.extend(sorted(root.glob("*.safetensors")))
    return {"path": path, "sha256": {p.name: digest(p) for p in files},
            "weight_shards_hashed": os.environ.get("HASH_MODEL_WEIGHTS") == "1"}

model = checkpoint(os.environ["MODEL"])
manifest = {
    "dataset_sha256": digest(Path(os.environ["DATASET"])),
    "model": model,
    "reference": model if os.environ["REF_MODEL"] == os.environ["MODEL"] else checkpoint(os.environ["REF_MODEL"]),
    "versions": {name: importlib.metadata.version(name) for name in
                 ("ms-swift", "torch", "transformers", "datasets", "binpacking", "trl")},
    "git_head": subprocess.check_output(["git", "-C", os.environ["REPO_ROOT"], "rev-parse", "HEAD"], text=True).strip(),
    "dp_size": 2, "micro_batch_size": 1, "gradient_accumulation_steps": 12,
    "global_batch_size": 24, "sampler_seed": "epoch", "dataset_seed": 42,
}
Path(os.environ["OUTPUT_DIR"], "launch-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY
fi

"${COMMAND[@]}" 2>&1 | tee "${OUTPUT_DIR}/log-${NODE_RANK}.txt"
