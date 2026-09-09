#!/usr/bin/env bash
# One-command turbo-vs-vllm precision comparison on Ascend NPU.
#
# Runs the full 4-stage pipeline (gen_input → vllm → turbo → compare) that
# diffs torchturbo's training-side modeling against vLLM prefill, operator by
# operator, and emits a CLI report + browser-viewable graph.json.
#
# Usage:
#   bash scripts/rl/run_turbo_vs_vllm.sh
#
# Env overrides:
#   MODEL       HF model path (default /sfs_turbo/pretrained_models/Qwen3.5-4B)
#   OUT         output dir (default ../tmp/gate)
#   GPU_ID      NPU device id (default 0)
#   DTYPE       compute dtype (default bfloat16)
#   INPUT_MODE  text | image_text (default text)
#   SEQ_LEN     token length (default 128 for text, 512 for image_text)
#   GEN_EXTRA   extra flags forwarded only to gen_input (e.g. --fake_image)
#   RUN_EXTRA   extra flags forwarded to BOTH turbo and vllm (e.g. --hook_filter 'layers\.0\.mlp')
#
# Pipeline (each stage is a separate process; all read the same --out_dir, so
# turbo/vllm consume the shared input_spec.pt produced by gen_input):
#   [1/4] gen_input  → $OUT/input_spec.pt        (shared tokens; single source of truth)
#   [2/4] vllm       → $OUT/vllm/{graph.pt,...}  (vLLM prefill, tp=1, enforce_eager)
#   [3/4] turbo      → $OUT/turbo/{graph.pt,...} (TorchTurboEngine.build_model, ep=1/sp=1)
#   [4/4] compare    — read both graph.pt → CLI diff + $OUT/report.json
#
# IMPORTANT — NPU env this script assumes you have already exported (matches
# the manual flow that works on-box):
#   export TORCHTURBO_DEVICE=npu          # pick npu backend (auto-detected if unset)
#   export ASCEND_RT_VISIBLE_DEVICES=0    # which NPU to use (should match GPU_ID)
# These are NOT forced here so a pre-exported value always wins; if neither is
# set, compare_train_infer.py auto-detects torch.npu and uses gpu 0.
#
# Weight guard: gen_input/vllm/turbo all reuse the SAME --out_dir; if a prior
# gen_input produced input_spec.pt with a different seq_len, re-running won't
# silently mix lengths — pass SEQ_LEN consistently.

# deps:
# [images]            hcr.meta-wulan01.hw-wulan.local/antsys/vllm:AsystemHybirdEngine-6968dbd2-20260612130916
# [vllm]              pip install /mnt/sfs_turbo/torchturbo/stable/qwen3_5_deps/vllm-0.23.1rc1.dev752+g1f486d96a.d20260717.empty-cp311-cp311-linux_aarch64.whl
# [vllm-ascend]       pip install /mnt/sfs_turbo/torchturbo/stable/qwen3_5_deps/vllm_ascend-0.19.1rc2.dev934+g2e605e544.d20260719-cp311-cp311-linux_aarch64.whl
# [triton-ascend]     pip install /mnt/sfs_turbo/torchturbo/stable/qwen3_5_deps/triton_ascend-3.2.1-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl --no-deps --force-reinstall
# [peft]              pip install peft==0.19.1


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAIN="$SCRIPT_DIR/compare_train_infer.py"

MODEL="${MODEL:-/sfs_turbo/pretrained_models/Qwen3.5-4B}"
OUT="${OUT:-$SCRIPT_DIR/../tmp/vllm_compare}"
GPU_ID="${GPU_ID:-0}"
DTYPE="${DTYPE:-bfloat16}"
INPUT_MODE="${INPUT_MODE:-text}"
SEQ_LEN="${SEQ_LEN:-}"

# Default seq_len mirrors run_qwen3_5.sh: text=128, image_text=512.
if [ -z "$SEQ_LEN" ]; then
    case "$INPUT_MODE" in
        text)       SEQ_LEN=128 ;;
        image_text) SEQ_LEN=512 ;;
        *) echo "INPUT_MODE must be 'text' or 'image_text', got: $INPUT_MODE" >&2; exit 1 ;;
    esac
fi

# gen_input flags depend on input mode (image_text needs --fake_image or --image_path).
GEN_MODE_FLAGS=()
case "$INPUT_MODE" in
    text)       GEN_MODE_FLAGS=(--input_mode text) ;;
    image_text) GEN_MODE_FLAGS=(--input_mode image_text --fake_image) ;;
esac

mkdir -p "$OUT"

echo "==[1/4] gen_input (mode=$INPUT_MODE, seq_len=$SEQ_LEN)=="
python "$MAIN" --mode gen_input --model_path "$MODEL" --out_dir "$OUT" \
    --seq_len "$SEQ_LEN" "${GEN_MODE_FLAGS[@]}" ${GEN_EXTRA:-}

echo "==[2/4] vllm (NPU, tp=1, enforce_eager, mamba_cache_dtype=float32)=="
python "$MAIN" --mode vllm --model_path "$MODEL" --out_dir "$OUT" \
    --gpu_id "$GPU_ID" --dtype "$DTYPE" ${RUN_EXTRA:-}

echo "==[3/4] turbo (NPU, ep=1/sp=1, world=1)=="
python "$MAIN" --mode turbo --model_path "$MODEL" --out_dir "$OUT" \
    --gpu_id "$GPU_ID" --dtype "$DTYPE" ${RUN_EXTRA:-}

echo "==[4/4] compare (turbo vs vllm)=="
# --show_tree prints the full module relationship tree so the first internal
# onset box (+parent/siblings/children) and structural-diff boxes are readable.
python "$MAIN" --mode compare --out_dir "$OUT" \
    --show_tree --export_json "$OUT/report.json"

echo ""
echo "Done. CLI report -> $OUT/report.json"
echo "Browser viewer   -> scripts/rl/report.html (pick $OUT/turbo/ and $OUT/vllm/)"
echo ""
echo "To isolate one operator with raw element-wise diff, re-record just that subtree:"
echo "  RUN_EXTRA=\"--hook_filter 'layers.0.mlp'\" bash $0   # re-runs vllm+turbo scoped, then compare"
