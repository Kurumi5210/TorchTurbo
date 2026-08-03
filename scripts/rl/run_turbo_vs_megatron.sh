#!/usr/bin/env bash
# One-command turbo-vs-megatron precision comparison on Ascend NPU.
#
# Diffs torchturbo's training-side modeling against Megatron-Core (via
# Megatron-Bridge / mbridge) running on MindSpeed's Ascend operator adapter
# (mindspeed.megatron_adaptor patches MCore's CUDA/TE ops → Ascend). Result:
# "MCore + MindSpeed" vs torchturbo — see the compare side_b=megatron.
#
# Pipeline (each stage a separate process sharing --out_dir):
#   [1/4] gen_input         → $OUT/input_spec.pt
#   [2/4] megatron          → $OUT/megatron/{graph.pt, loss_grad.json, ...}   (MCore via mbridge, mindspeed adapter)
#   [3/4] turbo             → $OUT/turbo/{graph.pt, loss_grad.json, ...}      (TorchTurboEngine, ep=1/sp=1)
#   [4/4] compare           → $OUT/report.json (turbo vs megatron forward logits)
#   [cmp] loss_grad_compare → $OUT/loss_grad_report.json (turbo vs megatron loss & grad_norm)
#
# [2/4] and [3/4] are MERGED: a SINGLE forward per side captures the module
# graph AND computes the loss (from those same logits) + backward() grad_norm —
# no second forward for loss/grad. The graph capture uses .detach().cpu().clone()
# copies, so graph.pt is byte-identical to the old infer-only run.
#
# Usage:
#   bash scripts/rl/run_turbo_vs_megatron.sh
#   bash scripts/rl/run_turbo_vs_megatron.sh --packed          # packed (3 sequences)
#   MODEL=/path/to/9B bash scripts/rl/run_turbo_vs_megatron.sh --packed
#
# Env overrides:
#   MODEL          HF model path (default /sfs_turbo/pretrained_models/Qwen3.5-4B)
#   OUT            output dir (default ../tmp/meg_compare)
#   GPU_ID         NPU device id (default 0)
#   DTYPE          compute dtype (default bfloat16)
#   INPUT_MODE     text | image_text (default text)
#   SEQ_LEN        token length (default 128 for text, 512 for image_text)
#   MINDSPEED_PATH Path to a MindSpeed checkout (dir holding the `mindspeed`
#                  package). If set, forwarded as --megatron_mindspeed_path to
#                  the megatron leg AND added to PYTHONPATH for both legs.
#                  If unset, an already-importable `mindspeed` is assumed.
#   MEGATRON_PATH  Path to a megatron-bridge/MCore checkout (dir holding
#                  `megatron.bridge`/`megatron.core`). Added to PYTHONPATH if set.
#   TORCHTURBO_SRC Path to torchturbo src (default <repo>/src).
#   --packed       Pack 3 short sequences into one (with cu_seqlens
#                  boundaries + reset position_ids). Tests packed-input
#                  consistency: loss/logp skip cross-sequence boundaries,
#                  grad_norm reflects the packed forward. Absent = single seq.
#   GEN_EXTRA      extra flags forwarded only to gen_input
#   RUN_EXTRA      extra flags forwarded to BOTH megatron and turbo
#
# NPU env (export before running; this script does NOT force them so a
# pre-exported value always wins — same convention as run_turbo_vs_vllm.sh):
#   export TORCHTURBO_DEVICE=npu
#   export ASCEND_RT_VISIBLE_DEVICES=0
#
# No `pip install` is performed here. MindSpeed (and optionally megatron-bridge)
# are consumed from your checkout via PYTHONPATH / --megatron_mindspeed_path.

# deps:
# [images]            hcr.meta-wulan01.hw-wulan.local/antsys/vllm:AsystemHybirdEngine-6968dbd2-20260612130916
# [triton-ascend]     pip install /mnt/sfs_turbo/torchturbo/stable/qwen3_5_deps/triton_ascend-3.2.1-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl --no-deps --force-reinstall
# [Megatron-LM]       git clone git@code.alipay.com:Theta/Megatron-LM.git -b hpu-dev/core_0.16.0_qwen_pack
# [Megatron-Bridge]   git clone git@code.alipay.com:Theta/Megatron-Bridge.git -b hpu_release_v1.4.0_qwen_mm
# [MindSpeed]         git clone git@code.alipay.com:Theta/mindspeed.git -b hpu-dev/core_r0.16.0-jy-cp-pack
# [peft]              pip install peft==0.19.1


set -euo pipefail

PACKED=0  # default; overridden by --packed CLI flag below

# Parse CLI flags: --packed sets PACKED=1; everything else is ignored.
for _arg in "$@"; do
    case "$_arg" in
        --packed) PACKED=1 ;;
        --help|-h) sed -n '2,/^$/p' "$0" | sed 's/^# //; s/^#//'; exit 0 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MAIN="$SCRIPT_DIR/compare_train_infer.py"

MODEL="${MODEL:-/sfs_turbo/pretrained_models/Qwen3.5-4B}"
OUT="${OUT:-$SCRIPT_DIR/../tmp/meg_compare}"
GPU_ID="${GPU_ID:-0}"
DTYPE="${DTYPE:-bfloat16}"
INPUT_MODE="${INPUT_MODE:-text}"
SEQ_LEN="${SEQ_LEN:-}"
MINDSPEED_PATH="${MINDSPEED_PATH:-}"
MEGATRON_PATH="${MEGATRON_PATH:-}"
TORCHTURBO_SRC="${TORCHTURBO_SRC:-$REPO_ROOT/src}"

# Repo layout keeps torchturbo, Megatron-Bridge and mindspeed as sibling dirs.
REPO_PARENT="$(cd "$REPO_ROOT/.." && pwd)"
[ -z "$MEGATRON_PATH" ]  && [ -d "$REPO_PARENT/Megatron-Bridge/src" ] && MEGATRON_PATH="$REPO_PARENT/Megatron-Bridge/src"
[ -z "$MINDSPEED_PATH" ] && [ -d "$REPO_PARENT/mindspeed" ]          && MINDSPEED_PATH="$REPO_PARENT/mindspeed"

if [ -z "$SEQ_LEN" ]; then
    case "$INPUT_MODE" in
        text)       SEQ_LEN=128 ;;
        image_text) SEQ_LEN=512 ;;
        *) echo "INPUT_MODE must be 'text' or 'image_text', got: $INPUT_MODE" >&2; exit 1 ;;
    esac
fi

# Build a PYTHONPATH that lets both mindspeed AND megatron-bridge resolve from
# your checkouts, without pip-installing. A pre-exported PYTHONPATH always wins
# (we prepend, existing entries remain).
PP="$TORCHTURBO_SRC"
[ -n "$MINDSPEED_PATH" ] && PP="$MINDSPEED_PATH:$PP"
[ -n "$MEGATRON_PATH" ]  && PP="$MEGATRON_PATH:$PP"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PP"

# gen_input flags depend on input mode (image_text needs --fake_image or --image_path).
# When PACKED=1, append --packed so gen_input builds 3 concatenated sequences
# with cu_seqlens boundaries + reset position_ids (text mode only).
GEN_MODE_FLAGS=()
case "$INPUT_MODE" in
    text)       GEN_MODE_FLAGS=(--input_mode text) ;;
    image_text) GEN_MODE_FLAGS=(--input_mode image_text --fake_image) ;;
esac
if [ "$PACKED" = "1" ] && [ "$INPUT_MODE" = "text" ]; then
    GEN_MODE_FLAGS+=(--packed)
    echo "(packed mode: 3 short sequences will be concatenated with cu_seqlens)"
fi

# mindspeed adapter flag for the megatron leg only (turbo picks mindspeed up
# via PYTHONPATH through torchturbo's npu_fused_ops import).
MS_FLAG=()
[ -n "$MINDSPEED_PATH" ] && MS_FLAG=(--megatron_mindspeed_path "$MINDSPEED_PATH")

mkdir -p "$OUT"

PY="python -u"

echo "==[1/4] gen_input (mode=$INPUT_MODE, seq_len=$SEQ_LEN, packed=$PACKED)=="
$PY "$MAIN" --mode gen_input --model_path "$MODEL" --out_dir "$OUT" \
    --seq_len "$SEQ_LEN" "${GEN_MODE_FLAGS[@]}" ${GEN_EXTRA:-}

echo "==[2/4] megatron (NPU, MCore via mbridge + mindspeed; one merged forward → graph + loss/grad)=="
$PY "$MAIN" --mode megatron --model_path "$MODEL" --out_dir "$OUT" \
    --gpu_id "$GPU_ID" --dtype "$DTYPE" "${MS_FLAG[@]}" ${RUN_EXTRA:-}

echo "==[3/4] turbo (NPU, ep=1/sp=1, world=1; one merged forward → graph + loss/grad)=="
$PY "$MAIN" --mode turbo --model_path "$MODEL" --out_dir "$OUT" \
    --gpu_id "$GPU_ID" --dtype "$DTYPE" ${RUN_EXTRA:-}

echo "==[4/4] compare (turbo vs megatron forward logits)=="
# --side_b megatron diffs $OUT/turbo/ vs $OUT/megatron/ (default sides are
# turbo/vllm). Continue even if compare exits non-zero (FAIL) so the report is
# still written and the user sees the verdict.
$PY "$MAIN" --mode compare --out_dir "$OUT" \
    --side_a turbo --side_b megatron \
    --show_tree --export_json "$OUT/report.json" || true

# ── loss / grad_norm comparison ──
echo "==[loss_grad_compare] (turbo vs megatron loss/grad, no model needed)=="
# Pure arithmetic over the two loss_grad.json — runs anywhere, can't fail hard.
$PY "$MAIN" --mode loss_grad_compare --out_dir "$OUT" || true

echo ""
echo "Done. CLI report -> $OUT/report.json"
echo "      loss/grad  -> $OUT/loss_grad_report.json  (turbo vs megatron loss & grad_norm)"
echo "      packed     = $PACKED  (0=single-sequence, 1=3-sequence-concatenated)"
echo "Browser viewer   -> scripts/rl/report.html (pick $OUT/turbo/ and $OUT/megatron/)"
echo ""
echo "Resolved checkouts (empty = not found, that side will fall back to PYTHONPATH):"
echo "  MEGATRON_PATH  = ${MEGATRON_PATH:-<unset>}"
echo "  MINDSPEED_PATH = ${MINDSPEED_PATH:-<unset>}"
echo ""
echo "If megatron.bridge still fails to import, the checkout was not at the sibling"
echo "position ($REPO_PARENT/Megatron-Bridge/src). Point at it explicitly — note the"
echo "/src suffix (megatron/ is a namespace package there, must be on the path):"
echo "  MEGATRON_PATH=/abs/path/to/Megatron-Bridge/src MINDSPEED_PATH=/abs/path/to/mindspeed bash $0"
