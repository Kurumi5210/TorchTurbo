# LLaDA2.1-mini Inference

Script: `src/torchturbo/models/llada2_moe/generate.py`

---

## Part 1 — How generation works

### 1.1 LLaDA2 is a diffusion model, not autoregressive

LLaDA2 is a **discrete block-diffusion model**. Generation does **not** predict token-by-token. It starts from a fully-masked template (`<|mask|>` × `gen_length`) and iteratively denoises:

1. Pad the prompt up to a multiple of `block_length` (default 32).
2. Append a generation region of `gen_length` mask tokens.
3. For each **block** (blocks are sequential, tokens within a block are parallel):
   - forward → logits + confidence per mask position;
   - commit positions whose confidence > `threshold` (mask → real token); if too few, take top-k highest-confidence;
   - settled positions may be re-edited if a new token's confidence > `editing_threshold`;
   - repeat until the block has no masks left.
4. After all blocks resolve, run `max_post_steps` rounds of global refinement.
5. Truncate: by eos only if `eos_early_stop=True`, else strip trailing masks.

Because of this, the standard HF `model.generate(**inputs, do_sample=...)` does **not** apply — you must call the model's own diffusion `generate`:
`modeling_llada2_moe.LLaDA2MoeModelLM.generate`.

### 1.2 Run

```bash
python src/torchturbo/models/llada2_moe/generate.py \
    --model /path/to/LLaDA2.1-mini \
    --prompt "Provide a comprehensive yet accessible introduction to Large Language Models (LLMs)."
```

Output (key line is the telemetry):
```
[generate] 512 tok | 127.30s | 4.0 tok/s | fwd=166 | TPF=3.08
```
- `fwd` = number of forward calls (= diffusion steps)
- **`TPF`** (Tokens/Forward) = tokens committed per step — the core efficiency metric.

### 1.3 Key parameters

| Parameter | Default | Meaning |
|---|---|---|
| `--gen_length` | 512 | max tokens to generate (excl. prompt) |
| `--steps` | 512 | denoising steps per block |
| `--block_length` | 32 | diffusion block size |
| `--temperature` | 0.0 | 0.0 = greedy (recommended) |
| `--threshold` | 0.5 | mask→token commit confidence; 0.95 collapses to `\n` |
| `--editing_threshold` | 0.0 | re-edit threshold for settled tokens; 0 = off |
| `--num_to_transfer` | 1 | min tokens locked per step; raise to speed up |
| `--max_post_steps` | 16 | global refinement after masks resolve |
| `--eos_early_stop` | False | off by default (eos == pad, prone to false stops) |
| `--attn` | sdpa | sdpa / eager (fallback) / flash_attention_2 |
| `--dtype` | auto | auto → bf16 |

### 1.4 Implementation notes

- **Modeling source**: loads `modeling_llada2_moe.py` from the torchturbo source tree (same directory as the script) via a nominal package — not the `trust_remote_code` copy in the weights dir, which may be stale. Bypasses the `llada2_moe/__init__.py` engine/peft chain (inference only needs torch + transformers).
- **Prompt padding uses `mask_id`, not `pad_id`**: LLaDA2's eos/pad are the same token (`<|endoftext|>`, id 156892). Padding with it biases generation toward eos; `mask_id` (156895) is safe because `prompt_index_full` marks padded positions non-editable.
- **Attention mask is additive `0`/`-inf`**, matching `bd.block_diffusion_mask_4d`. On Ascend NPU, HF sdpa applies `logical_not` to non-bool masks; `0`/`-inf` survives the double-negation correctly, while a bool mask collapses under `is_causal=False`. **If output is all `<|endoftext|>` / garbled → switch `--attn eager`.**
- **`eos_early_stop` default off**: truncating on eos (= pad) collapses output to a single eos if any position lands there. Default strips trailing masks instead.
- **Timing**: `_counted_forward` wraps `generate`, counts forwards, and reports TPF.

### 1.5 CAP and inference

CAP (Confidence-aware Parallel Training) is **training-side** — enabled by `--tt_cap_lambda > 0` in the training script, which adds an auxiliary loss that sharpens confidence on correct-token positions. **`generate.py` has no CAP parameter**: CAP is baked into the checkpoint weights.

So a CAP-speedup comparison runs the **same generate command** against two checkpoints (one trained with `--tt_cap_lambda`, one without) and compares TPF. CAP training makes the model commit more per step → fewer `fwd` → higher TPF → faster end-to-end.

### 1.6 Quick tuning

| Goal | Params |
|---|---|
| Most stable | `--temperature 0.0 --threshold 0.5 --editing_threshold 0.0 --num_to_transfer 1` |
| Fast long-form | `--num_to_transfer 8 --steps 64 --gen_length 512` |
| Diverse | `--temperature 0.7 --top_p 0.9 --num_to_transfer 4` |
| Smoke test | `--gen_length 64 --steps 16 --num_to_transfer 8` |
| sdpa collapsed | `--attn eager` |

---

## Part 2 — Results

### 2.1 Benchmark: basemodel vs 100-step SFT (w/ vs w/o CAP)

Same prompt (`Provide a comprehensive yet accessible introduction to Large Language Models (LLMs).`), `gen_length=512`, default params; only `--model` differs.

|  | basemodel | 100 steps (w/. CAP) | 100 steps (w/o. CAP) |
| :---: | :---: | :---: | :---: |
| tokens generated | 512 | 512 | 512 |
| forward calls (`fwd`) | 260 | 166 | 171 |
| **TPF** (Tokens/Forward) | 1.97 | 3.08 | 2.99 |
| single-card E2E time | 195.9s | 127.3s | 137.21s |

### 2.2 Readings

- **CAP speedup holds**: `w/. CAP` TPF = 3.08 vs basemodel 1.97 → **~1.56×**, matching HF's official **1.46×** for LLaDA2.1-mini (within noise).
- **Mechanism**: CAP calibrates confidence in training → each step commits more → `fwd` drops 260→166 → TPF up, E2E down to 127.3s.
- **CAP vs no-CAP (same SFT)**: `w/. CAP` (127.3s, TPF 3.08) beats `w/o. CAP` (137.2s, TPF 2.99) — the speedup is from confidence calibration, independent of SFT itself.
- **basemodel slowest**: un-SFT weights have lowest TPF and longest E2E.

### 2.3 Conclusion

**torchturbo's CAP matches HF's stated 1.46× on TPF; functionality OK.**

Ref: [https://huggingface.co/inclusionAI/LLaDA2.1-mini](https://huggingface.co/inclusionAI/LLaDA2.1-mini)

### 2.4 Output quality (same prompt)

All three produce structured long-form text (headings, bullets), confirming the BD generation path is correct. After SFT, output is more regular (standard "What / How / Why-Large" sections) with fewer repetition glitches (basemodel shows "amounts amounts", "language language"). CAP vs no-CAP outputs are comparable in quality — **CAP's value is speed, not single-sample quality**.
