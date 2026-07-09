#!/usr/bin/env bash
# One-command precision comparison for Qwen3.5-MoE:
#   torchturbo training-side modeling  vs  vLLM prefill (inference).
#
# Minimal parallel config (per requirement #4): turbo ep=1/sp=1, vLLM tp=1.
# Default input is a synthetic image+text prompt: a deterministic fake image
# (no dataset/image files needed) is generated into $OUT, and both turbo and
# vLLM read the *same* lossless PNG so the two sides see byte-identical pixels
# → the vision path (tower + merger + mrope) is exercised symmetrically.
#
# Usage:
#   MODEL=/input/asystem/xiyun.wc/ckpt/small_qwen3_5/ bash scripts/rl/run_qwen3_5.sh
#
# Env overrides:
#   MODEL             HF model path (required)
#   OUT               output dir (default /tmp/rl_compare_qwen3_5)
#   SEQ_LEN           token length (default 512 for image_text, 128 for text)
#   GPU_ID            cuda device (default 0)
#   TORCHTURBO_SRC    path to torchturbo src (default <repo>/src)
#   VLLM              vllm is expected importable in the active env
#   INPUT_MODE        text | image_text (default image_text)
#   FAKE_IMAGE        1 to use a synthetic image; 0 to require IMAGES= (default 1
#                     when INPUT_MODE=image_text)
#   FAKE_IMAGE_SIZE   edge length (px) of each fake image (default 448)
#   FAKE_IMAGE_COUNT  number of fake images (default 1)
#   IMAGES            real image path(s) PATH[,PATH] — used only when
#                     INPUT_MODE=image_text and FAKE_IMAGE=0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MAIN="$SCRIPT_DIR/compare_train_infer.py"

MODEL="${MODEL:?MODEL env var required, e.g. /path/to/Qwen3___5-35B-A3B}"
OUT="${OUT:-/tmp/rl_compare_qwen3_5}"
SEQ_LEN="${SEQ_LEN:-}"
GPU_ID="${GPU_ID:-0}"
TORCHTURBO_SRC="${TORCHTURBO_SRC:-$REPO_ROOT/src}"

# Append torchturbo src to PYTHONPATH so `import torchturbo` resolves.
export PYTHONPATH="$TORCHTURBO_SRC:${PYTHONPATH:-}"
# Minimal parallel config for the FSDP2 backend.
export TORCH_TURBO_SP_SIZE=1
export TORCH_TURBO_EP_SIZE=1
export TORCH_TURBO_BACKEND=fsdp2

mkdir -p "$OUT"
# Default to the vision path with a synthesised fake image. Set INPUT_MODE=text
# for a language-backbone-only run; set FAKE_IMAGE=0 + IMAGES=... to use real
# image files under image_text. turbo/vllm both read input_spec.pt (which
# references the fake-image PNGs that live under $OUT).
INPUT_MODE="${INPUT_MODE:-image_text}"
EXTRA_GEN_ARGS=()
case "$INPUT_MODE" in
    text)
        SEQ_LEN="${SEQ_LEN:-128}"
        EXTRA_GEN_ARGS=(--input_mode text)
        ;;
    image_text)
        FAKE_IMAGE="${FAKE_IMAGE:-1}"
        SEQ_LEN="${SEQ_LEN:-512}"
        if [ "$FAKE_IMAGE" = "1" ]; then
            EXTRA_GEN_ARGS=(--input_mode image_text --fake_image
                            --fake_image_size "${FAKE_IMAGE_SIZE:-448}"
                            --fake_image_count "${FAKE_IMAGE_COUNT:-1}")
        else
            IMAGES="${IMAGES:?INPUT_MODE=image_text + FAKE_IMAGE=0 requires IMAGES=PATH[,PATH]}"
            EXTRA_GEN_ARGS=(--input_mode image_text --image_path "$IMAGES")
        fi
        ;;
    *)
        echo "INPUT_MODE must be 'text' or 'image_text', got: $INPUT_MODE" >&2
        exit 1
        ;;
esac

if [ "$INPUT_MODE" = "image_text" ] && [ "${FAKE_IMAGE:-1}" = "1" ]; then
    GEN_LABEL="$INPUT_MODE (fake_image, ${FAKE_IMAGE_SIZE:-448}px×${FAKE_IMAGE_COUNT:-1})"
else
    GEN_LABEL="$INPUT_MODE"
fi
echo "==[1/4] gen_input (mode=$GEN_LABEL)=="
python "$MAIN" --mode gen_input --model_path "$MODEL" --out_dir "$OUT" \
    --seq_len "$SEQ_LEN" "${EXTRA_GEN_ARGS[@]}"

echo "==[2/4] turbo=="
python "$MAIN" --mode turbo --model_path "$MODEL" --out_dir "$OUT" \
    --gpu_id "$GPU_ID" --dtype bfloat16

echo "==[3/4] vllm=="
python "$MAIN" --mode vllm --model_path "$MODEL" --out_dir "$OUT" \
    --gpu_id "$GPU_ID" --dtype bfloat16

echo "==[4/4] compare=="
# --show_tree prints the full module relationship tree so, when logp doesn't
# align, you can read off the first internal-onset box (+ its parent/siblings/
# children) and the structural-diff boxes.
python "$MAIN" --mode compare --out_dir "$OUT" \
    --show_tree --export_json "$OUT/report.json"

echo ""
echo "Done. CLI report → $OUT/report.json"
echo ""
echo "If a specific operator is suspect, re-record raw tensors for just it and inspect:"
echo "  python $MAIN --mode turbo --model_path $MODEL --out_dir $OUT --hook_filter 'layers.0.self_attn'"
echo "  python $MAIN --mode vllm  --model_path $MODEL --out_dir $OUT --hook_filter 'layers.0.self_attn'"
echo "  python $MAIN --mode compare --out_dir $OUT --select layers.0.self_attn"