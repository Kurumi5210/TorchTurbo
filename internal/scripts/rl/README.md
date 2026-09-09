# train/infer precision comparison

`compare_train_infer.py` verifies a torchturbo model is numerically consistent
with the reference **vLLM prefill** path (so it's safe to plug into an RL
framework that runs training *and* inference), and — when it isn't — pinpoints
**which module / operator diverged** via an interactive module graph.

## Pipeline (four stages, separate processes)

```
gen_input → input_spec.pt   (single source of truth: token_ids +, for mm, pixel_values/grid)
turbo     → turbo/{graph.pt, logp.npy}     (reads input_spec.pt)
vllm      → vllm/{graph.pt, logp.npy}      (reads input_spec.pt)
compare   ← run AFTER both .pt exist
```

### Input construction (per-model, symmetric)

Both sides read the **same** `input_spec.pt`, built once by a per-arch
`InputBuilder`, so turbo and vLLM see identical tokens by construction. This
iteration ships a **qwen3.5** builder with two input modes:

- **text** — `--input_mode text` (default). Pure token ids; neither side passes
  any multimodal argument. Symmetric by construction.
- **image_text** — `--input_mode image_text --image_path a.jpg[,b.jpg]`. Uses the
  HF processor to expand image placeholders → token_ids, plus `pixel_values` /
  `image_grid_thw` for turbo and `multi_modal_data={"image": PIL}` for vLLM.
  `mm_token_type_ids` is **deliberately not passed** on either side — the qwen3.5
  model derives it from `input_ids` (image/video tokens), removing the old
  turbo-only asymmetry.

The hook root follows the mode automatically: **text** → language backbone;
**image_text** → full model (so the vision tower + merger I/O is captured too).
Override with `--hook_root`.

```bash
# text-only
python ... --mode gen_input --model_path $MODEL --input_mode text --seq_len 128
# image + text
python ... --mode gen_input --model_path $MODEL --input_mode image_text \
    --image_path /path/img.jpg --seq_len 512
```

- **turbo** — `TorchTurboEngine.build_model(...)` (the real training-side entry
  point), `ep=1 / sp=1`, **no extra autocast** so the precision environment
  matches online training.
- **vllm** — `vllm.LLM` prefill, `tp=1`, `enforce_eager`, `prompt_token_ids`
  (bypasses vLLM's tokenizer).
- We do **not** modify either modeling — only **hook** modules to record.

## What gets recorded (the module relationship graph)

A `forward_pre_hook` (input) + `forward_hook` (output) is registered on **every**
`nn.Module` under the hook root. During the forward pass the hooks only *record*
(no analysis). Per module ("box"), stored to a single `<side>/graph.pt`:

- **I/O**: every input/output tensor, path-labelled (`args.0`,
  `kwargs.hidden_states`, `out.1`, …), as **raw CPU fp32 tensor + scalar stats**
  (shape/dtype/norm/sum/mean/max_abs). Multiple tensors per I/O are kept (so e.g.
  `router_logits` from a MoE tuple is capturable).
- **parent** — the containing module.
- **children** — direct sub-modules.
- **exec_pre / exec_post** — global forward-execution order, from which
  **prev/next sibling** (data-flow predecessor/successor within the parent) is
  derived.

This yields a navigable box-diagram:

```
model (root)
├── embed_tokens
├── layers.0
│   ├── input_layernorm
│   ├── self_attn            ← turbo: q_proj/k_proj/v_proj/o_proj
│   │                        ← vllm:  qkv_proj / o_proj   (structural diff!)
│   ├── post_attention_layernorm
│   └── mlp                  ← turbo: SparseMoeBlock/experts
│                            ← vllm:  Qwen3NextSparseMoeBlock
├── layers.1 …
└── lm_head
```

All recorded tensors are moved to **CPU** inside the hooks (the GPU never
retains intermediates), and their **original dtype is preserved** (bf16/fp16/
fp32/…) — dtype differences are themselves a divergence source, so the recorded
tensor is never cast. **Every** tensor of every hooked module is recorded in
full; there is no per-tensor or cumulative size cap. Use `--hook_filter` to scope
which modules get hooked (and thus recorded) if you only care about a subtree.

## compare (CLI)

Reads both `graph.pt`. Boxes are paired across architectures by **attribute-name
suffix** (the architecture-specific `model` / `language_model` wrappers are
dropped), so `language_model.layers.0.self_attn` (turbo) aligns with
`model.layers.0.self_attn` (vllm). Unpaired boxes are themselves diagnostic —
they flag operator-level structural differences (e.g. `q_proj` vs `qkv_proj`).

It reports:
- **Structural check** — paired-count + boxes only on one side.
- **Divergence localization** — the first box whose *output* diverges while its
  *input* aligned (internal onset `◆`), with that box's **parent / prev-next
  sibling / children / exec order** printed so you know exactly where to look.
  Distinguishes internal (this box broke) vs inherited (upstream diff).
- **Top-K** boxes by output rel diff.
- `--show_tree` — full relationship tree with `◆/○/●` markers.
- `--select <box>` — print one box's full relationship + per-path tensor stats +
  element-wise raw diff for its input/output.
- logp comparison + PASS/FAIL (exit 0/1).

```bash
python internal/scripts/rl/compare_train_infer.py --mode compare --out_dir $OUT --show_tree
python internal/scripts/rl/compare_train_infer.py --mode compare --out_dir $OUT \
    --select layers.0.self_attn.o_proj
```

### Megatron side (`--mode megatron`) — turbo vs Megatron precision

In addition to vLLM, the harness can run the **real Megatron-Core** model via
**Megatron-Bridge (`mbridge`, `megatron.bridge.AutoBridge`)** — the same qwen3.5
provider (`Qwen35VLMoEModelProvider` / `Qwen3VLModel`) that swift's megatron
training uses. This diffs turbo's training-side modeling against Megatron's,
operator by operator, for both text and image+text inputs.

```bash
python internal/scripts/rl/compare_train_infer.py --mode megatron --model_path $MODEL \
    --out_dir $OUT --gpu_id 0 --dtype bfloat16
python internal/scripts/rl/compare_train_infer.py --mode compare --out_dir $OUT \
    --side_a turbo --side_b megatron --show_tree
```

- Builds the MCore `GPTModel` from the **HF checkpoint** in-process
  (`AutoBridge.from_hf_pretrained` → `to_megatron_provider(load_weights=True)` →
  `provide_distributed_model(wrap_with_ddp=False)`), the canonical standalone
  flow from `megatron.bridge.inference.text_generation.load_bridge_model`.
  TP/PP/EP default to **1** so a single-GPU prefill is identical-by-contract to
  the turbo run (same params, same bf16 compute). Override with
  `--megatron_tp/--megatron_pp/--megatron_ep`.
- `AutoBridge.from_hf_pretrained` auto-routes by the HF checkpoint's
  architecture: a **text-only** checkpoint (`Qwen3_5MoeForCausalLM`) →
  `Qwen35MoEBridge` → MCore `GPTModel`; a **VLM** checkpoint
  (`Qwen3_5MoeForConditionalGeneration`) → `Qwen35VLMoEBridge` →
  `Qwen35VLMoEModelProvider` → MCore `Qwen3VLModel` (vision encoder + merger +
  hybrid GDN/attention LM + mrope) — the same qwen3.5-VL mbridge provider swift
  launches Megatron training through. So `--mode megatron` supports **both**
  `--input_mode text` and `--input_mode image_text`.
- image_text: the megatron side feeds the exact same `input_ids` (HF-expanded
  image tokens), `pixel_values`, `image_grid_thw` from the shared `input_spec.pt`
  that turbo does, with `position_ids=None` so MCore computes its own mrope from
  `input_ids` + the visual grids — symmetric with turbo's `position_ids=None` +
  `mm_token_type_ids` path. `reorganize_inputs` derives the image mask from
  `input_ids == image_token_id`, so no `attention_mask`/`image_input_mask` is
  needed (flash/TE builds the causal mask internally, as in text mode).
- MTP (multi-token-prediction) head is disabled by default for the forward
  (`disable_mtp_for_inference`), so `--mode megatron` returns the main LM-head
  logits — matching the HF/turbo side. `--megatron_keep_mtp` keeps it on.
- `--side_a/--side_b` name the two `<side>/` subdirectories `--mode compare`
  diffs (default `turbo`/`vllm`); `megatron` writes to `$OUT/megatron/`, so
  `--side_b megatron` compares turbo against Megatron.
- Cross-arch module pairing: the `decoder.` MCore wrapper is also dropped when
  aligning FQNs, so MCore `decoder.layers.0` aligns with HF
  `model.language_model.layers.0` → both `layers.0`. Fine-grained sub-module
  names (`self_attention.linear_qproj` vs `self_attn.q_proj`, MCore MoE vs HF
  `SparseMoeBlock`) still won't suffix-pair and surface as structural diffs —
  use `--select` on the layer-level box for those.

## Interactive viewer (`internal/scripts/rl/report.html`)

`internal/scripts/rl/report.html` is a **standalone, self-contained HTML** — no server, no
build step. The recorder writes a browser-readable `graph.json` (original-dtype
base64 tensors + parent/children/exec relationships) next to each `graph.pt`;
the viewer loads it client-side and renders **turbo on the left, vllm on the
right** as two real nested module trees.

1. Open `internal/scripts/rl/report.html` directly in a browser.
2. Use the top-bar directory pickers to select the **turbo/** and **vllm/**
   output dirs (or pick their `graph.json` files directly if your browser blocks
   directory selection).
3. Both trees render — parent/children nesting, `exec#N` forward order,
   `prev`/`next` sibling tags (all from the recorded relationship graph).
4. Click a node to expand it (collapsing a node prunes its whole subtree), then
   click one of its `in`/`out` tensor buttons — that selects that tensor for
   that side.
5. The **bottom panel diffs the two selected tensors**: dtype-aware stats, a
   `⚠️ dtype 不同` warning when they differ (then both decode to fp32), and an
   element-wise diff table (signed mean/max, diff abs mean/max) + a log10 SVG
   histogram.

Tensors are stored in `graph.json` in their **original dtype** as base64 bytes;
the viewer decodes bf16/fp16/fp32 client-side, so a dtype difference is both
visible and numerically comparable (with a note that some diff may come from
dtype quantization itself).

> Browser sandbox: pages loaded from `file://` can't read arbitrary paths, which
> is why the viewer uses directory/file pickers instead of a path input.

### Converting old `graph.pt` dumps to `graph.json` (`pt2json`)

The viewer reads `graph.json`, which new runs write automatically. For **old**
dumps made before that (or to regenerate one), convert an existing `graph.pt`
(+ `logp.npy`) in place — no re-running the recorder:

```bash
# auto-detects logp.npy next to the .pt; writes graph.json alongside it
python internal/scripts/rl/compare_train_infer.py --mode pt2json \
    --pt /path/to/turbo/graph.pt

# explicit paths (also useful if you moved the files around)
python internal/scripts/rl/compare_train_infer.py --mode pt2json \
    --pt turbo/graph.pt --logp_npy turbo/logp.npy --out_json turbo/graph.json
```

Then open `report.html` and pick the directories.

## Shrinking the model to fit one GPU (qwen3.5)

The full 35B model doesn't fit a single GPU. `shrink_qwen3_5_model.py` produces a
self-contained reduced variant — **only the language-model depth is cut**
(default 40 → 4 transformer layers); the vision tower, MoE expert count, vocab,
rope config and dtype are unchanged. The result is a normal HF model dir usable
by both turbo and vLLM.

```bash
python internal/scripts/rl/shrink_qwen3_5_model.py \
    --src /path/to/Qwen3___5-35B-A3B \
    --dst /tmp/qwen3_5_4layer \
    --num_layers 4
MODEL=/tmp/qwen3_5_4layer bash internal/scripts/rl/run_qwen3_5.sh
```

What it does:
- `text_config.num_hidden_layers` → `--num_layers`; `layer_types` is rebuilt from
  `full_attention_interval` so the last kept layer is `full_attention` (for
  4 layers / interval 4 → `[linear, linear, linear, full_attention]`).
- `mtp_num_hidden_layers = 0` (drop the MTP head) unless `--keep_mtp`.
- Weight shards are subsetted: only the kept language layers (+ embeddings,
  lm_head, vision tower) are written; dropped layers + MTP are omitted and the
  safetensors index is rewritten. All tokenizer/processor/*.py assets are copied.
- `--skip_weights` writes only config+assets (use when you point a custom loader
  at the source weights).

A shrunk model has different absolute values than the full one, so its logp mean
diff is **not** the production number — it's for making the comparison pipeline
runnable on a single GPU. The per-module divergence localization is still
meaningful: a divergence that first appears at layer 3's self_attn in the shrunk
model points at the same operator in the full model.

## Memory controls

**Every** tensor of every hooked module is recorded in full (original dtype, on
CPU) — there is no size cap. For a 35B model this can be tens of GB per side, so
to keep recordings manageable, scope which modules are hooked (and thus
recorded):

- `--hook_root language_model` / `--hook_root model` — recurse from a subtree.
- `--hook_filter 'layers\.0'` — only modules whose FQN matches get hooked at all.

## Quick start (Qwen3.5-MoE, text-only, minimal config)

```bash
MODEL=/path/to/Qwen3___5-35B-A3B
bash internal/scripts/rl/run_qwen3_5.sh          # runs gen_input/turbo/vllm/compare
# then explore interactively — open the standalone viewer in a browser:
#   internal/scripts/rl/report.html   (pick the turbo/ and vllm/ output dirs)
```

`compare` exits 0 on PASS (logp mean ≤ `--logp_mean_threshold`) and 1 on FAIL.

## Options

| flag | default | meaning |
|---|---|---|
| `--seq_len` | 128 | max tokens (gen_input) |
| `--arch` | auto | input builder arch (e.g. `qwen3_5`); auto-detect from config model_type |
| `--input_mode` | text | `text` or `image_text` (gen_input) |
| `--text` / `--prompt` | built-in | prompt text (gen_input) |
| `--image_path` | — | image_text: comma-separated image paths (gen_input) |
| `--gpu_id` | 0 | CUDA device (turbo/vllm) |
| `--dtype` | bfloat16 | compute dtype (turbo build_model torch_dtype / vLLM dtype) |
| `--hook_root` | backbone | recurse from this attr path |
| `--hook_filter` | — | regex; only matching modules get hooked (full tensors recorded, original dtype) |
| `--logp_mean_threshold` | 1e-3 | PASS threshold (千分位) |
| `--onset_threshold` | 1e-3 | output rel diff triggering onset |
| `--top_k` | 20 | top-K boxes by output rel diff |
| `--show_tree` | off | print full module tree |
| `--select` | — | inspect one box: relationships + per-tensor diff |
| `--pt` | — | [pt2json] input `graph.pt` |
| `--logp_npy` | auto | [pt2json] companion `logp.npy` (auto-detected next to .pt) |
| `--out_json` | alongside .pt | [pt2json] output `graph.json` path |

## Design notes

- **MoE / TP:** minimal config (`ep=1, sp=1`, `tp=1`) removes parallel-sharding
  as a precision source.
- **Text-only:** the VLM vision tower is out of scope; hook root defaults to the
  language model.
- **CPU-only captures:** all hook tensors `.to('cpu', float32)` immediately.
- **Cross-arch pairing:** suffix alignment drops `model` / `language_model`; add
  more wrapper names to `_DROP_WRAP` if a model differs.

## Extending to other torchturbo models

The recursive hook needs no per-model code — it walks whatever `nn.Module` tree a
model exposes. `--hook_root` narrows the tree if attribute layout differs.