#!/usr/bin/env python
# ruff: noqa: E501
"""Shrink a Qwen3.5-MoE VLM checkpoint to a single-GPU-fitting variant.

Reduces the language model depth (default 40 → 4 transformer layers); optionally
also truncates the vision tower (``--num_vision_layers``). Everything else (MoE
expert count, vocab, rope, dtype) is kept as-is. The result is a self-contained
HF model dir you can point both ``compare_train_infer.py`` (turbo + vLLM) and HF
directly at.

Design notes
------------
* The default reduction keeps the qwen3.5 attention pattern. ``layer_types`` is
  rebuilt from ``full_attention_interval`` for the new depth — for 4 layers with
  interval 4 that's ``[linear, linear, linear, full_attention]``. If you ask for
  a depth whose last layer is not ``full_attention`` we keep the ORIGINAL
  ``layer_types`` prefix (so the relative pattern is preserved) rather than
  truncating arbitrarily.
* Vision tower (``--num_vision_layers``): the tower is a plain stack of
  independent ``Qwen3_5MoeVisionBlock``s, so dropping trailing blocks is
  structurally safe. We set ``vision_config.depth`` and keep
  ``model.visual.blocks.0..n-1`` (+ the shared ``patch_embed`` / ``pos_embed``
  / ``rotary_pos_emb`` / ``merger``), dropping ``blocks.n..``.
* MTP is disabled (``mtp_num_hidden_layers = 0``) to save memory and to match the
  compare script's default training/infer forward (MTP is training-only here).
* Weights: only the LM layers we keep are copied; MTP + dropped language layers
  + their MoE expert weights are omitted. The model therefore loads cleanly from
  the shrunken dir on either torchturbo or vLLM without sharded-prefix mismatch
  on the kept tensors. Parameters for the kept layers are copied by REFERENCE
  (link) where possible; else copied.

Usage
-----
    python scripts/rl/shrink_qwen3_5_model.py \
        --src /path/to/Qwen3___5-35B-A3B \
        --dst /tmp/qwen3_5_4layer \
        --num_layers 4

    # then use it with the compare script:
    MODEL=/tmp/qwen3_5_4layer bash scripts/rl/run_qwen3_5.sh

Options
-------
    --src                source HF model dir (required)
    --dst                output dir (required)
    --num_layers         number of language-model layers to keep (default 4)
    --num_vision_layers  number of vision-tower blocks to keep (default: keep all)
    --keep_mtp           keep MTP head (default: drop it)
    --copy_files         comma-separated extra files to copy verbatim (tokenizer,
                         chat_template, preprocessor config, etc. are auto-copied)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys


# Non-weight assets are copied verbatim. Rather than maintain a fragile allow-list
# of patterns (VLMs keep adding new preprocessor configs, e.g.
# ``video_preprocessor_config.json``), we copy everything EXCEPT weight shards,
# their index files, and ``config.json`` (which we rewrite). New asset types are
# picked up automatically.
_EXCLUDE_EXACT = {
    "config.json",                       # rewritten by this script
    "model.safetensors.index.json",      # regenerated from the subsetted weights
    "pytorch_model.bin.index.json",
    "model.index.json",
}
_EXCLUDE_SUFFIXES = (
    ".safetensors", ".bin", ".pt", ".pth", ".h5", ".msgpack", ".ot",
    # tmp/lock files that sometimes sit next to shards
    ".lock",
)


def _is_weight_or_excluded(fn: str) -> bool:
    if fn in _EXCLUDE_EXACT:
        return True
    return any(fn.endswith(suf) for suf in _EXCLUDE_SUFFIXES)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Shrink a Qwen3.5-MoE model to a single-GPU-fitting variant "
                    "(reduce language-model depth only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--src", type=str, required=True, help="source HF model dir")
    p.add_argument("--dst", type=str, required=True, help="output shrunken model dir")
    p.add_argument("--num_layers", type=int, default=4,
                   help="number of language-model transformer layers to keep")
    p.add_argument("--num_vision_layers", type=int, default=None,
                   help="number of vision-tower blocks to keep (default: keep all; "
                        "e.g. --num_vision_layers 4 trims the tower's depth). "
                        "Trailing blocks are dropped: blocks.0..n-1 survive.")
    p.add_argument("--keep_mtp", action="store_true",
                   help="keep the MTP head (default: drop it to save memory)")
    p.add_argument("--copy_files", type=str, default="",
                   help="comma-separated extra files to copy verbatim")
    p.add_argument("--skip_weights", action="store_true",
                   help="only write the reduced config + assets, skip weight "
                        "subsetting (use when you'll point at --src weights via "
                        "a custom loader)")
    return p.parse_args()


def _load_config(src: str) -> dict:
    import torch  # noqa: F401  (safetensors may need it)
    # Prefer a plain JSON config.json so we don't depend on the exact
    # transformers/qwen3.5 modeling being importable in this process.
    with open(os.path.join(src, "config.json")) as f:
        return json.load(f)


def _rebuild_layer_types(text_cfg: dict, num_layers: int) -> list[str]:
    """Layer-type pattern for the reduced depth.

    If the original config lists ``layer_types``, take its prefix of length
    ``num_layers`` only when that prefix already ends on a full-attention layer
    (keeps the model's recurrent vs full-attention structure meaningful).
    Otherwise rebuild from ``full_attention_interval``.
    """
    orig = text_cfg.get("layer_types")
    interval = text_cfg.get("full_attention_interval", 4) or 4
    if isinstance(orig, list) and len(orig) >= num_layers and len(orig) > 0:
        prefix = orig[:num_layers]
        # Make sure the kept prefix ends on a full-attention layer so the final
        # residual stream has attended globally before the lm_head.
        if prefix[-1] == "full_attention":
            return list(prefix)
    full = []
    for i in range(num_layers):
        full.append("linear_attention" if (i + 1) % interval != 0 else "full_attention")
    if full and full[-1] != "full_attention":
        full[-1] = "full_attention"
    return full


def shrink_config(cfg: dict, num_layers: int, keep_mtp: bool,
                  num_vision_layers: int | None = None) -> dict:
    tc = dict(cfg.get("text_config", {}))
    tc["num_hidden_layers"] = num_layers
    tc["layer_types"] = _rebuild_layer_types(tc, num_layers)
    if not keep_mtp:
        tc["mtp_num_hidden_layers"] = 0
    cfg = dict(cfg)
    cfg["text_config"] = tc
    # Optionally truncate the vision tower. The vision model is
    # ``nn.ModuleList([VisionBlock(config) for _ in range(config.depth)])``, so we
    # just lower ``depth``; weight subsetting drops the trailing ``blocks.n..``.
    if num_vision_layers is not None:
        vc = dict(cfg.get("vision_config", {}))
        vc["depth"] = num_vision_layers
        cfg["vision_config"] = vc
    # ensure rope/attention-pattern fields are preserved as-is
    return cfg


def _kept_layer_indices(num_layers: int) -> list[int]:
    return list(range(num_layers))


def _iter_shard_files(src: str):
    idx_path = os.path.join(src, "model.safetensors.index.json")
    bin_idx = os.path.join(src, "pytorch_model.bin.index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            wb = json.load(f).get("weight_map", {})
        return "safetensors", wb
    if os.path.exists(bin_idx):
        with open(bin_idx) as f:
            wb = json.load(f).get("weight_map", {})
        return "bin", wb
    # single-file
    for fn in os.listdir(src):
        if fn.endswith(".safetensors"):
            return "safetensors", None
        if fn.endswith(".bin"):
            return "bin", None
    return None, None


def topic_for(name: str) -> str:
    """Which part of the model a parameter belongs to: 'language', 'vision',
    'mtp', or 'other'."""
    if name.startswith("model.language_model.layers.") or name.startswith("language_model.layers."):
        return "language"
    # vision-tower blocks: model.visual.blocks.<i>.* (some ckpts use visual.blocks)
    if ".visual.blocks." in name or name.startswith("visual.blocks.") or name.startswith("model.visual.blocks."):
        return "vision"
    if name.startswith("mtp.") or name.startswith("model.mtp."):
        return "mtp"
    return "other"


def _layer_index_of(name: str) -> int | None:
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _vision_block_index_of(name: str) -> int | None:
    m = re.search(r"visual\.blocks\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def subset_weights(src: str, dst: str, kept_layers: list[int], keep_mtp: bool,
                   fmt: str, weight_map: dict | None,
                   num_vision_layers: int | None = None) -> None:
    """Rewrite weight shards keeping only ``kept_layers`` language layers
    (+ non-language weights, optionally MTP, and vision blocks truncated to
    ``num_vision_layers`` if given).

    Strategy: group the (tensor → src file) map, then for each output shard
    filename load only the needed tensors from the source shard and re-save.
    For safetensors this avoids loading the full model at once.
    """
    import torch
    if fmt == "safetensors":
        from safetensors.torch import load_file, save_file
    else:
        load_file = lambda p: torch.load(p, map_location="cpu")
        save_file = None

    # Build candidate kept-parameter set by inspecting the weight map keys.
    if weight_map is None:
        # single-file: load everything, then filter
        files = [f for f in os.listdir(src)
                 if f.endswith(f".{ 'safetensors' if fmt == 'safetensors' else 'bin'}")]
        weight_map = {}
        # We can't enumerate single-file keys without loading; fall through to
        # the per-file load below which filters by name via the loaded dict.
        per_file = {f: None for f in files}
    else:
        per_file = {}
        for name, fn in weight_map.items():
            per_file.setdefault(fn, None)

    def _keep(name: str) -> bool:
        bloc = topic_for(name)
        if bloc == "language":
            li = _layer_index_of(name)
            return li is not None and li in kept_layers
        if bloc == "vision":
            # Keep shared vision params (patch_embed/pos_embed/merger/...); only
            # the per-block tensors carry a blocks.<i>. index, and those beyond
            # num_vision_layers are dropped.
            if num_vision_layers is None:
                return True
            bi = _vision_block_index_of(name)
            if bi is None:
                return True  # shared vision param, not a block
            return bi < num_vision_layers
        if bloc == "mtp":
            return keep_mtp
        return True  # embeddings, lm_head, norms, router-of-nothing etc.

    total_kept = total_dropped = 0
    new_weight_map: dict[str, str] = {}
    for fn in sorted(per_file.keys()):
        src_path = os.path.join(src, fn)
        if not os.path.exists(src_path):
            continue
        tensors = load_file(src_path)
        kept = {n: t for n, t in tensors.items() if _keep(n)}
        dropped = len(tensors) - len(kept)
        total_kept += len(kept)
        total_dropped += dropped
        del tensors
        if not kept:
            print(f"  {fn}: 0 kept, {dropped} dropped → skipped (file not written)")
            continue
        if fmt == "safetensors":
            save_file(kept, os.path.join(dst, fn), metadata={"format": "pt"})
        else:
            torch.save(kept, os.path.join(dst, fn))
        for n in kept:
            new_weight_map[n] = fn
        print(f"  {fn}: {len(kept)} kept, {dropped} dropped")

    # Write a fresh index for safetensors (single-file → no index needed).
    if fmt == "safetensors" and (len(per_file) > 1 or weight_map is not None):
        idx_path = os.path.join(dst, "model.safetensors.index.json")
        with open(idx_path, "w") as f:
            json.dump({"metadata": {"total_size": -1}, "weight_map": new_weight_map}, f)
        print(f"  wrote index ({len(new_weight_map)} tensors)")
    elif weight_map is not None and fmt == "bin":
        idx_path = os.path.join(dst, "pytorch_model.bin.index.json")
        with open(idx_path, "w") as f:
            json.dump({"metadata": {"total_size": -1}, "weight_map": new_weight_map}, f)
        print(f"  wrote index ({len(new_weight_map)} tensors)")

    print(f"\n[weights] kept {total_kept} tensors, dropped {total_dropped}")


def copy_assets(src: str, dst: str, extra: list[str]) -> None:
    """Copy every non-weight asset from src → dst.

    Weight shards/indexes are excluded (they're subsetted separately), and
    ``config.json`` is excluded (rewritten by this script). Everything else —
    tokenizer, processor + image/video preprocessor configs, chat templates,
    generation_config, README, *.py remote-code, … — is copied verbatim so no
    asset type is silently dropped when a model adds a new one.
    """
    copied = set()
    for fn in os.listdir(src):
        if _is_weight_or_excluded(fn):
            continue
        s = os.path.join(src, fn)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(dst, fn))
            copied.add(fn)
        # (subdirectories, e.g. onnx/, are not copied — none needed for inference.)
    for fn in extra:
        fn = fn.strip()
        if not fn:
            continue
        s = os.path.join(src, fn)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(dst, fn))
            copied.add(fn)
    print(f"[assets] copied {len(copied)} file(s): {sorted(copied)}")


def main() -> None:
    args = parse_args()
    if not os.path.isdir(args.src):
        sys.exit(f"--src not a dir: {args.src}")
    if args.num_layers <= 0:
        sys.exit("--num_layers must be > 0")
    nvl = args.num_vision_layers
    if nvl is not None and nvl <= 0:
        sys.exit("--num_vision_layers must be > 0")

    os.makedirs(args.dst, exist_ok=True)

    cfg = _load_config(args.src)
    text_cfg = cfg.get("text_config", {})
    orig_n = text_cfg.get("num_hidden_layers", "?")
    orig_vd = cfg.get("vision_config", {}).get("depth", "?")
    cfg = shrink_config(cfg, args.num_layers, args.keep_mtp, num_vision_layers=nvl)
    print(f"[config] text_config.num_hidden_layers: {orig_n} → {args.num_layers}")
    print(f"[config] layer_types (kept): {cfg['text_config']['layer_types']}")
    print(f"[config] mtp: {'kept' if args.keep_mtp else 'dropped (mtp_num_hidden_layers=0)'}")
    if nvl is not None:
        print(f"[config] vision_config.depth: {orig_vd} → {nvl}")
    else:
        print(f"[config] vision_config: unchanged (depth={orig_vd})")

    with open(os.path.join(args.dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"[config] wrote {args.dst}/config.json")

    copy_assets(args.src, args.dst, [s for s in args.copy_files.split(",") if s])

    if not args.skip_weights:
        fmt, weight_map = _iter_shard_files(args.src)
        if fmt is None:
            print("[weights] no .safetensors/.bin index found in src — skipping "
                  "weight subsetting. Copy weights manually or pass --skip_weights.")
            return
        kept_layers = _kept_layer_indices(args.num_layers)
        print(f"\n[weights] format={fmt}; keeping language layers {kept_layers}"
              + (f"; vision blocks 0..{nvl-1}" if nvl is not None else ""))
        subset_weights(args.src, args.dst, kept_layers, args.keep_mtp, fmt,
                       weight_map, num_vision_layers=nvl)
    else:
        print("[weights] --skip_weights: not copying weights")

    print(f"\nDone. Shrunken model → {args.dst}")
    print("Use it with:")
    print(f"  MODEL={args.dst} bash scripts/rl/run_qwen3_5.sh")


if __name__ == "__main__":
    main()