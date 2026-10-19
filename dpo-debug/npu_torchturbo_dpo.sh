#!/usr/bin/env bash
set -eo pipefail

# ==================== Training parameters ====================

export MODEL="/sfs_turbo/pretrained_models/Qwen3.8-27B"
export REF_MODEL="$MODEL"
export MODEL_TYPE=turbo_qwen3_5
export TEMPLATE=qwen3_5
export DPO_DEBUG_EXPERIMENT=history_think
export FINETUNING_TYPE=full

export DATASET="/workspace/bin/mf_qwen3.5_397b_a17b_instruct-Main-Selected/DPO/AntWealth-WealthMxc-AllIntent-All/20261013_business_user_flag_answer.json"

export SEQ_LEN=128000
export NUM_TRAIN_EPOCHS=1
export MAX_STEPS=-1

# Two A3 nodes, 32 NPUs, SP=16. The engine derives GA=12 internally.
export GLOBAL_BATCH_SIZE=24
export PER_DEVICE_TRAIN_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=1

export BETA=0.1
export LOSS_TYPE=sigmoid
export RPO_ALPHA=0.1
export LABEL_SMOOTHING=0.05
export LOSS_SCALE=last_round+ignore_empty_think
export ADD_NON_THINKING_PREFIX=true

export LEARNING_RATE=1e-6
export LR_SCHEDULER_TYPE=cosine_with_min_lr
export LR_SCHEDULER_KWARGS='{"min_lr": 1e-7}'
export WARMUP_RATIO=0.05
export MAX_GRAD_NORM=1.0

export SAVE_STRATEGY=epoch
export SAVE_STEPS=2000
export SAVE_TOTAL_LIMIT=3
export SAVE_ONLY_MODEL=false

export LOGGING_STEPS=1
export DATASET_NUM_PROC=4
export DATALOADER_NUM_WORKERS=2
export SEED=42
export DATA_SEED=42

# TorchTurbo does not support evaluation during DPO yet.
export SPLIT_DATASET_RATIO=0.0
export EVAL_STRATEGY=no
export PER_DEVICE_EVAL_BATCH_SIZE=1

# ==================== Platform environment ====================
# The submission script and AIStudio provide the variables below.

: "${PACKAGE_DIR:?}" "${OUTPUT_DIR:?}"
: "${ARCHIVE_SHA256:?}" "${USER_NAME:?}"
: "${NNODES:?}" "${NPUS_PER_NODE:?}"

export LOG_DIR="${LOG_DIR:-/sfs_turbo/fanyaxin.fyx/logs/$(basename "$OUTPUT_DIR")}"
mkdir -p "$LOG_DIR"

export RANK="${NODE_RANK:-${RANK:-${WORKER_RANK:-}}}"

if [[ ! "$RANK" =~ ^[01]$ || "$NNODES" != 2 || "$NPUS_PER_NODE" != 16 ]]; then
  echo 'Expected two A3 nodes, 16 NPUs each, and platform node rank 0 or 1.' >&2
  exit 2
fi

if [[ -z "${MASTER_ADDR:-}" || "$MASTER_ADDR" == 127.0.0.1 || "$MASTER_ADDR" == localhost ]]; then
  echo 'AIStudio must supply a reachable MASTER_ADDR for both nodes.' >&2
  exit 2
fi

export MASTER_PORT="${MASTER_PORT:-23566}"
export NPROC_PER_NODE=16

# ==================== Data and dependency checks ====================

for item in \
  "$PACKAGE_DIR/dpo_debug_swift_template.py" \
  "$MODEL/config.json" \
  "$REF_MODEL/config.json" \
  "$DATASET" \
  /mnt/sfs_turbo/torchturbo/stable/flash_linear_attention-0.6.0-py3-none-any.whl \
  /mnt/sfs_turbo/torchturbo/stable/triton_ascend-3.2.2-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl
do
  [[ -f "$item" ]] || {
    echo "Missing required file: $item" >&2
    exit 2
  }
done

python3 "$PACKAGE_DIR/check_dataset.py" "$DATASET" \
  --manifest "$PACKAGE_DIR/data_manifest.json"

printf '%s  %s\n' \
  "$ARCHIVE_SHA256" \
  "$PACKAGE_DIR/torchturbo-c9c1ad2.tar.gz" | sha256sum -c -

runtime_dir="$(mktemp -d /tmp/qwen38-dpo.XXXXXX)"
tar -xzf "$PACKAGE_DIR/torchturbo-c9c1ad2.tar.gz" -C "$runtime_dir"
cd "$runtime_dir"

# Apply this branch's template to the existing platform package.
python3 - <<'PY'
import ast
import hashlib
import json
import os
from pathlib import Path

source = Path(os.environ["PACKAGE_DIR"]) / "dpo_debug_swift_template.py"
contents = source.read_bytes()
tree = ast.parse(contents)
variants = [
    ast.literal_eval(node.value)
    for node in tree.body
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "DPO_DEBUG_EXPERIMENT"
            for target in node.targets)
]
if variants != [os.environ["DPO_DEBUG_EXPERIMENT"]]:
    raise RuntimeError("The packaged template does not match this debug experiment.")

target = Path("src/torchturbo/models/qwen3_5/swift_template.py")
if not target.is_file():
    raise RuntimeError(f"The unpacked runtime has no template at {target}.")
target.write_bytes(contents)
metadata = {
    "experiment": variants[0],
    "template_sha256": hashlib.sha256(contents).hexdigest(),
    "swift_version": "4.4.2",
}
Path("dpo-debug-overlay.json").write_text(json.dumps(metadata, indent=2) + "\n")
print("DPO debug overlay:", json.dumps(metadata), flush=True)
PY

source /usr/local/Ascend/ascend-toolkit/set_env.sh

export HF_HOME="$runtime_dir/hf_cache"
export HF_MODULES_CACHE="$HF_HOME/modules"

bash internal/scripts/runs/qwen3_8_dense/install_swift_deps.sh

python3 - <<'PY'
import swift

if swift.__version__ != "4.4.2":
    raise RuntimeError(f"DPO debug comparison requires Swift 4.4.2, got {swift.__version__}.")
PY

# ==================== Connect to the official launcher ====================

python3 - <<'PY'
from pathlib import Path

p = Path("internal/scripts/runs/qwen3_8_dense/swift_text_dpo.sh")
s = p.read_text()

# Let Swift filter invalid and overlong samples without changing the source file.
anchor = "    --load_from_cache_file true\n"
assert s.count(anchor) == 1
s = s.replace(
    anchor,
    "    --load_from_cache_file false\n"
    "    --strict false\n"
    "    --truncation_strategy delete\n",
)

s = s.replace(
    'LOG_FILE="$OUTPUT_DIR/log-$NODE_RANK.txt"',
    'LOG_FILE="$LOG_DIR/log-$NODE_RANK.txt"',
)
s = s.replace(
    "export ASCEND_PROCESS_LOG_PATH=$OUTPUT_DIR/ascend_log",
    'export ASCEND_PROCESS_LOG_PATH="$LOG_DIR/ascend_log"',
)
s = s.replace(
    "export TENSORBOARD_LOGGING_DIR=/home/admin/logs/tfevent",
    'export TENSORBOARD_LOGGING_DIR="$LOG_DIR/tfevent"',
)
s = s.replace(
    "    --report_to tensorboard\n",
    '    --report_to tensorboard\n'
    '    --logging_dir "$LOG_DIR/tfevent"\n',
)

anchor = '    --tt_config "${TT_CONFIG}"\n'
assert s.count(anchor) == 1

extra_args = '''    --tt_gbs "${GLOBAL_BATCH_SIZE}"
    --rpo_alpha "${RPO_ALPHA}"
    --label_smoothing "${LABEL_SMOOTHING}"
    --max_grad_norm "${MAX_GRAD_NORM}"
    --lr_scheduler_kwargs "${LR_SCHEDULER_KWARGS}"
    --split_dataset_ratio "${SPLIT_DATASET_RATIO}"
    --eval_strategy "${EVAL_STRATEGY}"
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
    --seed "${SEED}"
    --data_seed "${DATA_SEED}"
    --save_only_model "${SAVE_ONLY_MODEL}"
    --loss_scale "${LOSS_SCALE}"
    --add_non_thinking_prefix "${ADD_NON_THINKING_PREFIX}"
'''

s = s.replace(anchor, extra_args + anchor)
p.write_text(s)
PY

bash -n internal/scripts/runs/qwen3_8_dense/swift_text_dpo.sh

# ==================== Persist the effective configuration ====================

mkdir -p "$OUTPUT_DIR"

if [[ "$RANK" == 0 ]]; then
  cp "$PACKAGE_DIR/run_qwen38_27b_dpo.sh" \
    "$OUTPUT_DIR/run_qwen38_27b_dpo.sh"
  cp src/torchturbo/models/qwen3_5/swift_template.py \
    "$OUTPUT_DIR/dpo-debug-swift-template.py"
  cp dpo-debug-overlay.json "$OUTPUT_DIR/dpo-debug-overlay.json"

  python3 - <<'PY'
import json
import os
from pathlib import Path

raw = os.environ.get("SUBMISSION_CONFIG_JSON")
if raw:
    config = json.loads(raw)
else:
    config = json.loads(
        (Path(os.environ["PACKAGE_DIR"]) / "config.json").read_text()
    )

config.update(
    model=os.environ["MODEL"],
    ref_model=os.environ["REF_MODEL"],
)
config["dpo_debug"] = json.loads(Path("dpo-debug-overlay.json").read_text())
config["dataset"] = os.environ["DATASET"].split(
    "/mf_qwen3.5_397b_a17b_instruct-Main-Selected/", 1
)[1]

keys = """
SEQ_LEN NUM_TRAIN_EPOCHS MAX_STEPS GLOBAL_BATCH_SIZE
PER_DEVICE_TRAIN_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS
BETA LOSS_TYPE RPO_ALPHA LABEL_SMOOTHING LOSS_SCALE
ADD_NON_THINKING_PREFIX LEARNING_RATE LR_SCHEDULER_TYPE
LR_SCHEDULER_KWARGS WARMUP_RATIO MAX_GRAD_NORM SAVE_STRATEGY
SAVE_STEPS SAVE_TOTAL_LIMIT SAVE_ONLY_MODEL LOGGING_STEPS
DATASET_NUM_PROC DATALOADER_NUM_WORKERS SEED DATA_SEED
SPLIT_DATASET_RATIO EVAL_STRATEGY PER_DEVICE_EVAL_BATCH_SIZE
""".split()

config.setdefault("training", {}).update(
    {key: os.environ[key] for key in keys}
)

(Path(os.environ["OUTPUT_DIR"]) / "submission-config.json").write_text(
    json.dumps(config, ensure_ascii=False, indent=2) + "\n"
)
PY

  printf '%s\n' \
    'c9c1ad2cf9b0ad2f59fbbc566c390c13775fc4cf' \
    > "$OUTPUT_DIR/torchturbo-commit.txt"
fi

# ==================== Launch training ====================

unset RESUME_FROM_CHECKPOINT
export PYTHON=python3

exec bash internal/scripts/runs/qwen3_8_dense/swift_text_dpo.sh
