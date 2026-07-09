#!/usr/bin/env python
# ruff: noqa: E501
"""Compare torchturbo training-side modeling vs vLLM prefill (inference) precision.

Pipeline
--------
1. ``--mode gen_input``  tokenize a fixed prompt → ``input.npy`` (shared tokens)
2. ``--mode turbo``      run the **real** turbo model (TorchTurboEngine.build_model,
                         ep=1/sp=1, no extra autocast) and record the module
                         relationship-graph + raw I/O tensors → ``turbo.pt``
3. ``--mode vllm``       run vLLM prefill (tp=1, enforce_eager, prompt_token_ids)
                         and record the module graph + raw I/O → ``vllm.pt``
4. ``--mode compare``    read both .pt, CLI diff + divergence localization
5. ``--mode pt2json``    convert an existing ``<side>/graph.pt`` (+ ``logp.npy``)
                         → browser-loadable ``graph.json`` (for old dumps).

Recording model
---------------
A forward hook is registered on **every** ``nn.Module`` under the hook root.
During the forward pass the hooks do **only** recording (no analysis):

  * capture every input/output tensor (with a path label like ``args.0`` /
    ``kwargs.hidden_states`` / ``out.1``) → moved to **CPU fp32** immediately;
    the GPU never retains captured intermediates;
  * store, per module ("box": its FQN):
      - I/O raw tensors + scalar stats (shape/dtype/norm/sum/mean/max_abs)
      - ``parent`` / ``children``            — module-tree relationships
      - ``exec_pre`` / ``exec_post``         — global forward order, from which
        prev/next-sibling and data-flow predecessor are derived
  * persist everything to a single ``<side>.pt`` file.

Because turbo and vLLM differ in operator order / fused ops inside attn / MoE,
recursion reaches leaf projections so a human can pick any two boxes and diff
them. Boxes that don't pair across architectures are themselves diagnostic (e.g.
turbo ``q_proj`` vs vLLM ``qkv_proj``).

Target: logp mean abs diff at the **千分位 (~1e-3)** level.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from typing import Any


# ─────────────────────────────── CLI ────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="torchturbo vs vLLM train/infer precision comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["gen_input", "turbo", "vllm", "compare",
                                  "pt2json"],
                   required=True, help="Run mode.")
    p.add_argument("--model_path", type=str, default=None,
                   help="HF model path (required for gen_input / turbo / vllm).")
    p.add_argument("--out_dir", type=str, default="/tmp/rl_compare",
                   help="Directory for the .pt files and report.")
    p.add_argument("--seq_len", type=int, default=128, help="Max tokens (gen_input).")
    # ── input construction (per-model builder; this iteration: qwen3.5) ──
    p.add_argument("--arch", type=str, default=None,
                   help="[gen_input] Input builder arch. Default = auto-detect from config "
                        "model_type. Currently supported: qwen3_5.")
    p.add_argument("--input_mode", choices=["text", "image_text"], default="text",
                   help="[gen_input] text-only or image+text mixed input.")
    p.add_argument("--text", type=str, default=None,
                   help="[gen_input] Prompt text. Defaults to a built-in paragraph (text mode) "
                        "or a short caption (image_text mode).")
    p.add_argument("--image_path", type=str, default=None,
                   help="[gen_input] image_text mode: comma-separated image file paths.")
    p.add_argument("--fake_image", action="store_true",
                   help="[gen_input] image_text mode: synthesize a deterministic fake image "
                        "(no --image_path needed) so turbo and vLLM share one synthetic input. "
                        "Mutually exclusive with --image_path.")
    p.add_argument("--fake_image_size", type=int, default=448,
                   help="[gen_input] edge length (px) of each synthesized fake image. Must keep "
                        "the resulting patch count within the processor's min/max_pixels.")
    p.add_argument("--fake_image_count", type=int, default=1,
                   help="[gen_input] number of fake images to synthesize in image_text mode.")
    p.add_argument("--prompt", type=str, default=None,
                   help="[gen_input] (alias of --text) override prompt text.")

    p.add_argument("--gpu_id", type=int, default=0, help="CUDA device id (turbo/vllm).")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"], help="Compute dtype.")
    # ── hook scope / recording (turbo / vllm) ──
    p.add_argument("--hook_root", type=str, default=None,
                   help="[turbo/vllm] Attribute path (relative to the model) to recurse "
                        "from. Default = language backbone.")
    p.add_argument("--hook_filter", type=str, default=None,
                   help="[turbo/vllm] Regex; when set, only modules whose FQN matches get "
                        "hooked (others skipped entirely). Default = hook the whole tree. "
                        "All tensors of hooked modules are recorded in full (original dtype).")
    # NOTE: every matching tensor is stored in full (original dtype, on CPU);
    # there is no per-tensor or cumulative size cap — record everything.
    # ── compare ──
    p.add_argument("--logp_mean_threshold", type=float, default=1e-3,
                   help="Pass threshold for logp mean abs diff (千分位).")
    p.add_argument("--onset_threshold", type=float, default=1e-3,
                   help="Output rel diff above which a node is a divergence onset.")
    p.add_argument("--export_json", type=str, default=None, help="[compare] JSON report path.")
    p.add_argument("--top_k", type=int, default=20,
                   help="[compare/html] top-K nodes by output rel diff.")
    p.add_argument("--show_tree", action="store_true",
                   help="[compare] print the full module relationship tree.")
    p.add_argument("--select", type=str, default=None,
                   help="[compare] inspect one node relationship: '<fqn>' (prints the node's "
                        "parent / prev-next sibling / children + per-tensor stats).")
    # ── pt2json ── convert an existing graph.pt (+ logp.npy) to browser-loadable
    # graph.json, so old dumps made before graph.json existed can be viewed in the
    # HTML without re-running the recorder.
    p.add_argument("--pt", type=str, default=None,
                   help="[pt2json] input <side>/graph.pt path.")
    p.add_argument("--logp_npy", type=str, default=None,
                   help="[pt2json] companion <side>/logp.npy (auto-detected if omitted "
                        "and sits next to the .pt).")
    p.add_argument("--out_json", type=str, default=None,
                   help="[pt2json] output graph.json path (default: alongside the .pt).")
    p.add_argument("--max_raw_bytes", type=int, default=262144,
                   help="[turbo/vllm/pt2json] Max raw-bytes (original dtype) to embed per "
                        "tensor in graph.json as base64. Tensors larger than this keep only "
                        "their stats (shape/norm/sum/mean/max_abs); the HTML viewer then "
                        "shows a stats-only comparison for that slot. 0 = embed no raw bytes "
                        "(meta/structure only). Keeps graph.json small enough for the browser "
                        "to JSON.parse (an 800 MB graph.json fails with 'Unexpected end of "
                        "JSON input'). For per-element diffs of dropped tensors use compare mode.")
    return p.parse_args()


DEFAULT_PROMPT = (
    "The quick brown fox jumps over the lazy dog. In a quiet village by the river, "
    "farmers gather each morning to plan the work ahead, exchanging news, weather notes "
    "and small stories of the night. Mathematics, much like farming, rewards patience: "
    "the gradient of a smooth function near its minimum is small, and a careful step size "
    "keeps the iterate from overshooting. Reinforcement learning compounds this idea over "
    "thousands of collected rollouts, where every token decision carries both reward and risk. "
    "Consistency between the training forward pass and the inference prefill is therefore not "
    "a luxury but a prerequisite for stable policy optimisation."
)


# ─────────────────────── per-model input builder ─────────────────────


class InputSpec:
    """One source of truth for what both turbo and vLLM see.

    Derived once from the model's processor/tokenizer, then each side reads its
    own representation from it — this is what keeps the two paths *symmetric*:
    we never build inputs twice in two different places.

    Attributes
    ----------
    token_ids : list[int]      shared token sequence (image placeholders already
                               expanded for VLMs in image_text mode) — used by
                               the turbo side (HF-style forward).
    text_prompt : str | None   for image_text: the PRE-expansion chat-templated
                               prompt (a single ``<|image_pad|>`` placeholder per
                               image). vLLM needs this UN-expanded form plus the
                               PIL image(s); it expands the placeholder itself,
                               matching the HF processor's patch math (same image
                               processor/config) so both sides end up the same
                               length. Passing the already-expanded token_ids to
                               vLLM *and* the image causes vLLM to re-expand →
                               mismatched lengths.
    mode : "text" | "image_text"
    pixel_values : torch.Tensor | None     (HF-processed; for turbo)
    image_grid_thw : torch.Tensor | None   (for turbo)
    image_paths : list[str]                (for vLLM to reload PIL)
    meta : dict
    """

    def __init__(self, token_ids, mode="text", pixel_values=None,
                 image_grid_thw=None, image_paths=None, meta=None,
                 text_prompt=None):
        self.token_ids = list(token_ids)
        self.mode = mode
        self.pixel_values = pixel_values
        self.image_grid_thw = image_grid_thw
        self.image_paths = image_paths or []
        self.text_prompt = text_prompt
        self.meta = meta or {}

    def save(self, path: str) -> None:
        import torch
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "token_ids": self.token_ids, "mode": self.mode,
            "pixel_values": self.pixel_values, "image_grid_thw": self.image_grid_thw,
            "image_paths": self.image_paths, "meta": self.meta,
            "text_prompt": self.text_prompt,
        }, path)

    @classmethod
    def load(cls, path: str) -> "InputSpec":
        import torch
        d = torch.load(path, map_location="cpu", weights_only=False)
        return cls(d["token_ids"], d.get("mode", "text"), d.get("pixel_values"),
                   d.get("image_grid_thw"), d.get("image_paths", []), d.get("meta", {}),
                   text_prompt=d.get("text_prompt"))


class InputBuilder:
    """Abstract base. Subclasses per model arch produce an InputSpec and know how
    to turn it into (a) turbo forward kwargs and (b) a vLLM prompt object, plus
    the default hook_root (language backbone vs full model incl. vision tower)."""

    name = "base"
    # default hook_root: language backbone for text; full model for mm (vision
    # tower included so its I/O is captured too).
    default_hook_root_text = None      # resolved per-arch
    default_hook_root_mm = None

    def __init__(self, model_path: str):
        self.model_path = model_path

    def build(self, text: str, image_paths: list[str], seq_len: int,
              fake_image: bool = False, fake_image_size: int = 448,
              fake_image_count: int = 1, out_dir: str = ".") -> InputSpec:
        raise NotImplementedError

    def turbo_kwargs(self, spec: InputSpec, device, dtype) -> dict:
        """Forward kwargs for the turbo model (transformers-style). Text mode passes
        a 1D arange position_ids; image_text mode passes position_ids=None plus
        mm_token_type_ids so the model computes its own 3D mrope positions,
        matching online inference and vLLM."""
        raise NotImplementedError

    def vllm_prompt(self, spec: InputSpec):
        """A vLLM prompt object (list[int] for text, dict for mm)."""
        raise NotImplementedError

    def default_hook_root(self, spec: InputSpec) -> str | None:
        return self.default_hook_root_mm if spec.mode == "image_text" \
            else self.default_hook_root_text


# ── qwen3.5 input builder (text + image_text) ──

class Qwen3_5InputBuilder(InputBuilder):
    name = "qwen3_5"
    # turbo: top class ...ForConditionalGeneration → .model.language_model
    default_hook_root_text = "model.language_model"
    # mm: hook the whole model so the vision tower + merger are captured too
    default_hook_root_mm = ""  # "" = the model root itself

    def __init__(self, model_path: str):
        super().__init__(model_path)
        from transformers import AutoTokenizer, AutoProcessor, AutoConfig
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        try:
            self.proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            self.proc = None

    def build(self, text: str, image_paths: list[str], seq_len: int,
              fake_image: bool = False, fake_image_size: int = 448,
              fake_image_count: int = 1, out_dir: str = ".") -> InputSpec:
        if image_paths and fake_image:
            sys.exit("[qwen3_5] --image_path and --fake_image are mutually exclusive.")
        if fake_image:
            image_paths = self._make_fake_images(out_dir, fake_image_size, fake_image_count)
        if image_paths:
            return self._build_mm(text, image_paths, seq_len)
        return self._build_text(text, seq_len)

    @staticmethod
    def _make_fake_images(out_dir: str, size: int, count: int) -> list[str]:
        """Synthesize deterministic fake images as PNGs on disk.

        Each image is a per-RGB-channel linear gradient so the pixel data is
        fully reproducible across runs (no RNG, no seed). Output dimensions are
        ``size x size`` — callers should keep the resulting patch count inside
        the processor's min/max_pixels (448 is a safe default). Both turbo (via
        the HF processor at gen_input time) and vLLM (via its own image
        processor at run time) read the *same* lossless PNGs, so the two sides
        see byte-identical pixels.
        """
        from PIL import Image
        import numpy as np
        os.makedirs(out_dir, exist_ok=True)
        size = int(size)
        count = max(1, int(count))
        # A base gradient ramp across the spatial axis; shift it per image so
        # multiple fake images are (deterministically) distinct.
        ramp = np.linspace(0, 255, size, dtype=np.uint8)
        row = np.tile(ramp, (size, 1))                       # (size, size)
        paths = []
        for i in range(count):
            shift = (i * 37) % 256                            # deterministic, distinct per image
            r = (row + shift).astype(np.uint8)
            g = (row.T + shift * 2).astype(np.uint8)
            b = (row // 2 + shift).astype(np.uint8)
            arr = np.stack([r, g, b], axis=-1)                # (size, size, 3)
            p = os.path.join(out_dir, f"fake_image_{i}.png")
            Image.fromarray(arr, mode="RGB").save(p)          # PNG == lossless
            paths.append(p)
        return paths

    def _build_text(self, text: str, seq_len: int) -> InputSpec:
        ids = self.tok(text, add_special_tokens=True, return_tensors=None)["input_ids"]
        if len(ids) > seq_len:
            ids = ids[:seq_len]
        return InputSpec(ids, mode="text", meta={"text": text})

    def _build_mm(self, text: str, image_paths: list[str], seq_len: int) -> InputSpec:
        if self.proc is None:
            sys.exit("[qwen3_5] image_text mode requires an HF AutoProcessor; "
                     "got None. Check the model path / transformers version.")
        from PIL import Image
        pils = [Image.open(p).convert("RGB") for p in image_paths]
        # Build the chat-templated prompt with one image entry per supplied image.
        content = []
        for _ in pils:
            content.append({"type": "image"})
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]
        # The chat template normally lives on the *tokenizer*, not the processor
        # (the processor may report "does not have a chat template"). Prefer the
        # tokenizer; fall back to the processor only if the tokenizer lacks one.
        templated = None
        for tmpl_holder in (self.tok, self.proc):
            try:
                templated = tmpl_holder.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                break
            except (ValueError, AttributeError) as e:
                last_err = e
        if templated is None:
            sys.exit("[qwen3_5] neither the tokenizer nor the processor exposes a "
                     f"chat template; cannot build image_text prompt. ({last_err})")
        inputs = self.proc(
            text=[templated], images=pils, return_tensors="pt", add_special_tokens=False)
        token_ids = inputs["input_ids"][0].tolist()
        if len(token_ids) > seq_len:
            sys.exit(f"[qwen3_5] mm token length {len(token_ids)} > --seq_len {seq_len}; "
                     "raise --seq_len.")
        import torch
        return InputSpec(
            token_ids, mode="image_text",
            pixel_values=inputs.get("pixel_values"),       # may be a tensor or a nested structure
            image_grid_thw=inputs.get("image_grid_thw"),
            image_paths=image_paths,
            # PRE-expansion templated text (one <|image_pad|> per image). vLLM
            # needs this un-expanded form plus the PIL image; passing the already
            # expanded token_ids to vLLM *and* the image double-expands. See
            # InputSpec docstring.
            text_prompt=templated,
            meta={"text": text, "n_images": len(pils),
                  "templated_len": len(token_ids)},
        )

    def turbo_kwargs(self, spec: InputSpec, device, dtype) -> dict:
        import torch
        ids = torch.as_tensor(spec.token_ids, dtype=torch.long, device=device).unsqueeze(0)
        seq_len = ids.shape[1]
        attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
        if spec.mode == "image_text":
            # Let the model compute its own 3D mrope position ids (text / temporal
            # / height / width) from input_ids + image_grid_thw, exactly as it does
            # in online training/inference. Passing a plain 1D arange here would
            # broadcast to all 4 dims and silently drop the vision 3D RoPE — making
            # turbo and vLLM diverge for the very reason this harness exists. So we
            # pass position_ids=None (→ compute_3d_position_ids → get_rope_index)
            # and give it mm_token_type_ids so the mrope branch is actually taken.
            # vLLM derives analogous positions internally from the same token ids,
            # keeping the two sides symmetric.
            image_token_id = getattr(self.cfg, "image_token_id", None)
            video_token_id = getattr(self.cfg, "video_token_id", None)
            mm_token_type_ids = torch.zeros_like(ids)
            if image_token_id is not None:
                mm_token_type_ids[ids == image_token_id] = 1
            if video_token_id is not None:
                mm_token_type_ids[ids == video_token_id] = 2
            kw = {
                "input_ids": ids,
                "attention_mask": attention_mask,
                "position_ids": None,
                "mm_token_type_ids": mm_token_type_ids,
                "labels": None,
            }
            if spec.pixel_values is not None:
                kw["pixel_values"] = _to_device(spec.pixel_values, device)
            if spec.image_grid_thw is not None:
                kw["image_grid_thw"] = _to_device(spec.image_grid_thw, device)
        else:
            # text-only: a plain 1D arange is correct (the text model expands
            # [1, seq] → [4, seq] internally and pure-text uses 1D positions).
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            kw = {
                "input_ids": ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "labels": None,
            }
        return kw

    def vllm_prompt(self, spec: InputSpec):
        if spec.mode == "text":
            return spec.token_ids
        # image_text: let vLLM do its own image-token expansion. We pass the
        # PRE-expansion chat-templated text (one <|image_pad|> per image) plus the
        # PIL image(s); vLLM tokenizes the text and expands the placeholder from
        # the image's patch grid — the same image_processor/config as the HF
        # side, so the expanded length matches turbo's []. Passing the already
        # HF-expanded token_ids here (in addition to the image) would make vLLM
        # re-expand and produce a mismatched length (216 vs 411).
        from PIL import Image
        pils = [Image.open(p).convert("RGB") for p in spec.image_paths]
        prompt_text = getattr(spec, "text_prompt", None)
        if prompt_text is None:
            # Legacy spec without text_prompt: fall back to the expanded token_ids
            # (longer/noisy, but at least runs). Regenerate the input spec to fix.
            print("[vllm_prompt] WARNING: InputSpec has no text_prompt; falling "
                  "back to prompt_token_ids (lengths may mismatch turbo). "
                  "Re-run gen_input to produce an input_spec with text_prompt.")
            return {
                "prompt_token_ids": spec.token_ids,
                "multi_modal_data": {"image": pils if len(pils) > 1 else pils[0]},
            }
        return {
            "prompt": prompt_text,
            "multi_modal_data": {"image": pils if len(pils) > 1 else pils[0]},
        }


def _to_device(t, device):
    import torch
    if torch.is_tensor(t):
        return t.to(device)
    if isinstance(t, (list, tuple)):
        return [_to_device(e, device) for e in t]
    if isinstance(t, dict):
        return {k: _to_device(v, device) for k, v in t.items()}
    return t


# Arch → builder class. model_type auto-detection maps known qwen3.5 types here.
_INPUT_BUILDERS = {"qwen3_5": Qwen3_5InputBuilder}
_QWEN3_5_MODEL_TYPES = {"qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text"}


def _resolve_builder(model_path: str, arch: str | None) -> InputBuilder:
    """Pick an InputBuilder by explicit --arch or by auto-detecting config model_type."""
    from transformers import AutoConfig
    if arch:
        cls = _INPUT_BUILDERS.get(arch)
        if cls is None:
            sys.exit(f"[input] unknown --arch {arch!r}; available: {list(_INPUT_BUILDERS)}")
        return cls(model_path)
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    mt = getattr(cfg, "model_type", "")
    if mt in _QWEN3_5_MODEL_TYPES:
        return Qwen3_5InputBuilder(model_path)
    # text_config for VLM configs
    tc = getattr(cfg, "text_config", None)
    if tc is not None and getattr(tc, "model_type", "") in _QWEN3_5_MODEL_TYPES:
        return Qwen3_5InputBuilder(model_path)
    sys.exit(f"[input] no input builder for model_type {mt!r}; pass --arch qwen3_5 "
             "or add a builder.")


# ─────────────────────────── shared helpers ─────────────────────────


def _ensure_torchturbo_on_path() -> None:
    try:
        import torchturbo  # noqa: F401
        return
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(here))
    src_dir = os.path.join(repo_root, "src")
    if os.path.isdir(os.path.join(src_dir, "torchturbo")):
        sys.path.insert(0, src_dir)


def _rel_diff(a, b) -> tuple[float, float]:
    eps = 1e-12
    abs_d = abs(a - b)
    return abs_d, abs_d / max(abs(a), abs(b), eps)


def _tensor_stats(t) -> dict | None:
    import torch
    if t is None or not torch.is_tensor(t) or t.ndim == 0 or t.numel() == 0:
        return None
    tf = t.detach().to("cpu", torch.float32)
    absv = tf.abs()
    return {"shape": list(tf.shape), "dtype": str(t.dtype),
            "norm": float(tf.norm().item()), "sum": float(tf.sum().item()),
            "mean": float(tf.mean().item()), "max_abs": float(absv.max().item())}


def _extract_tensors(obj, prefix: str) -> list[tuple[str, Any]]:
    """Flatten an arbitrary hook input/output into (path, tensor) pairs.

    Paths look like ``args.0`` / ``kwargs.hidden_states`` / ``out.1`` / nested.
    Tensors only — non-tensor leaves are ignored.
    """
    import torch
    found: list[tuple[str, Any]] = []

    def scan(o, p):
        if o is None:
            return
        if torch.is_tensor(o):
            if o.ndim == 0 or o.numel() == 0:
                return
            found.append((p, o))
        elif isinstance(o, (list, tuple)):
            for i, e in enumerate(o):
                scan(e, f"{p}.{i}")
        elif isinstance(o, dict):
            for k, v in o.items():
                scan(v, f"{p}.{k}")

    scan(obj, prefix)
    return found


# ─────────────────────── recursive graph recorder ───────────────────


class GraphRecorder:
    """Register hooks on every module under a subtree and build a relationship
    graph: per-module I/O (raw + stats), tree parent/children, and forward
    execution order (exec_pre/exec_post) — all on CPU, persisted to one .pt.

    Relationship model — RUNTIME STACK
    -----------------------------------
    The box-diagram parent/child relationship is captured by a runtime call
    stack, NOT by string-parsing FQNs at finalize time:

      * a ``pre`` hook PUSHES the module's FQN onto ``_stack`` on entry; the
        FQN just below it on the stack is its **parent** (the box that's
        currently executing whose forward directly called this one);
      * a ``post`` hook POPs the FQN on exit.

    This is correct even when the FQN's dotted path doesn't reflect the real
    nesting (reused sub-modules, ModuleList wrappers, weights shared across
    parents, reentrant calls): the parent is whatever was actually on top of
    the stack when the module was entered. ``exec_pre``/``exec_post`` are the
    global forward ticks of push/pop, recorded only for the longest-seq (prefill)
    pass so spurious early wakes can't invert the parent-before-child ordering.

    Every input/output tensor of every hooked module is captured and **cloned
    to CPU fp32-preserved (original dtype) inside the hook**, so the GPU never
    retains intermediates. Rich per-node info (class, parent, children,
    push/pop ticks, call depth, sibling order, I/O tensors) is preserved for the
    interactive viewer.
    """

    def __init__(self, hook_filter: str | None):
        import torch
        self.torch = torch
        self.nodes: dict[str, dict] = {}
        self._hooks: list = []
        self._tick = 0                       # monotonic global forward counter
        self._stack: list[str] = []          # runtime call stack of FQNs (push on pre, pop on post)
        self._filter = re.compile(hook_filter) if hook_filter else None
        self._root_name = ""

    # ── node helper ──
    def _node(self, fqn: str, cls: str) -> dict:
        n = self.nodes.get(fqn)
        if n is None:
            n = {
                "class": cls,
                "parent": None,            # set by the stack at push time
                "children": [],            # filled as children push while this is on top
                "exec_pre": None,          # push tick (longest-seq pass)
                "exec_post": None,         # pop tick (longest-seq pass)
                "call_depth": None,        # stack depth at push (= nesting level)
                "push_count": 0,           # how many times this module was entered
                "in": {}, "out": {},
                "_exec_pre_seq": (-1, None),   # (seq, tick) — keep longest-seq
                "_exec_post_seq": (-1, None),
            }
            self.nodes[fqn] = n
        else:
            n["class"] = n["class"] or cls
        return n

    # ── capture one tensor into a node slot ──
    def _capture_slot(self, fqn: str, slot: str, path: str, t) -> None:
        stats = _tensor_stats(t)
        if stats is None:
            return
        store = self.nodes[fqn][slot]
        cur = store.get(path)

        # keep the longest-seq (prefill) capture per (slot,path); spurious early
        # wakes with smaller seq are overwritten. This is what ties exec order
        # and tensors to the SAME prefill pass.
        def _seqdim(s):
            return s["shape"][0] if s and s.get("shape") else 0

        keep = cur is None or stats["shape"][0] > _seqdim(cur["stats"])
        # clone to CPU preserving ORIGINAL dtype — dtype is itself a divergence
        # source, so we never cast. No size cap: record everything.
        raw = t.detach().to("cpu").clone() if self.torch.is_tensor(t) else None
        entry = {"path": path, "stats": stats, "raw": raw, "dropped": raw is None}
        if keep:
            store[path] = entry

    def _capture_many(self, fqn, slot, items):
        for path, t in items:
            self._capture_slot(fqn, slot, path, t)

    @staticmethod
    def _seq_of(args, kwargs, output=None):
        """The prompt-seq dim from whichever tensor is handy — identifies the
        prefill pass so we keep only that pass's exec order + tensors."""
        for v in list((args or []) if args is not None else []) \
                 + list((kwargs or {}).values() if kwargs else []):
            if True and v is not None and hasattr(v, "ndim") and v.ndim >= 2:
                return v.shape[-2] if v.shape[0] == 1 else v.shape[0]
        return 0

    # ── hooks ──
    def register(self, root_module, root_name: str = "") -> None:
        self._root_name = root_name
        for name, mod in root_module.named_modules():
            fqn = name if root_name == "" else (f"{root_name}.{name}" if name else root_name)
            if self._filter is not None and not self._filter.search(fqn):
                continue

            def _make_pre(fqn, mod):
                cls = type(mod).__name__

                def pre(_m, args, kwargs):
                    self._tick += 1
                    n = self._node(fqn, cls)
                    n["push_count"] += 1
                    # ── STACK-PARENT: whoever's currently on top of the stack
                    # is this module's runtime parent (the box that's executing
                    # whose forward directly called this one). This nails the
                    # box-diagram nesting even for reused/shared reentrant subs.
                    if self._stack and self._stack[-1] != fqn:
                        p = self._stack[-1]
                        n["parent"] = p
                        pn = self._node(p, self.nodes[p]["class"]) if p in self.nodes else None
                    else:
                        n["parent"] = None
                    n["call_depth"] = len(self._stack)
                    self._stack.append(fqn)
                    # register parent→child link (dedup) — children list reflects
                    # the order they were FIRST invoked under this parent
                    p = n["parent"]
                    if p is not None:
                        pn = self._node(p, "")
                        if fqn not in pn["children"]:
                            pn["children"].append(fqn)
                    # keep exec_pre from the longest-seq pass (prefill)
                    seq = self._seq_of(args, kwargs)
                    if n["push_count"] == 1 or seq > n["_exec_pre_seq"][0]:
                        if n["push_count"] > 1:
                            pass
                        if seq >= n["_exec_pre_seq"][0]:
                            n["_exec_pre_seq"] = (seq, self._tick)
                    in_items = [(f"args.{i}", a) for i, a in enumerate(args or [])]
                    in_items += [(f"kwargs.{k}", v) for k, v in (kwargs or {}).items()]
                    in_items = [(p2, t) for p2, t in in_items
                                if self.torch.is_tensor(t)]
                    self._capture_many(fqn, "in", in_items)
                return pre

            def _make_post(fqn, mod):
                cls = type(mod).__name__

                def post(_m, _a, _kw, output):
                    self._tick += 1
                    n = self._node(fqn, cls)
                    # pop our own entry (guard against unbalanced pops from
                    # reentrant/again calls by popping the deepest matching frame)
                    if fqn in self._stack:
                        # pop until fqn is removed (handles interleaved pushes)
                        while self._stack and self._stack[-1] != fqn:
                            self._stack.pop()
                        if self._stack and self._stack[-1] == fqn:
                            self._stack.pop()
                    out_items = [(p, t) for p, t in _extract_tensors(output, "out")]
                    self._capture_many(fqn, "out", out_items)
                    seq = 0
                    for _, t in out_items:
                        if self.torch.is_tensor(t) and t.ndim >= 2:
                            seq = t.shape[-2] if t.shape[0] == 1 else t.shape[0]
                            break
                    if seq >= n["_exec_post_seq"][0]:
                        n["_exec_post_seq"] = (seq, self._tick)
                return post

            self._hooks.append(mod.register_forward_pre_hook(
                _make_pre(fqn, mod), with_kwargs=True))
            self._hooks.append(mod.register_forward_hook(
                _make_post(fqn, mod), with_kwargs=True))

    def remove(self) -> None:
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()

    # ── finalize: resolve exec order into a dense monotonic prefill trace ──
    def finalize(self) -> dict:
        """Resolve the kept-seq push/pop ticks into ``exec_pre``/``exec_post``
        and re-number them into a dense, monotonic forward order (parent always
        before child, sibling order = real forward call order). Drop the
        bookkeeping fields used during recording.
        """
        for n in self.nodes.values():
            n["exec_pre"] = n["_exec_pre_seq"][1]
            n["exec_post"] = n["_exec_post_seq"][1]

        # dense re-rank of pre ticks (only nodes that fired in the kept pass)
        pre_ticks = sorted({n["exec_pre"] for n in self.nodes.values()
                            if n["exec_pre"] is not None})
        pre_rank = {t: i + 1 for i, t in enumerate(pre_ticks)}
        post_ticks = sorted({n["exec_post"] for n in self.nodes.values()
                             if n["exec_post"] is not None})
        post_rank = {t: i + 1 for i, t in enumerate(post_ticks)}
        offset = len(pre_rank)
        for n in self.nodes.values():
            if n["exec_pre"] is not None:
                n["exec_pre"] = pre_rank[n["exec_pre"]]
            if n["exec_post"] is not None:
                n["exec_post"] = offset + post_rank[n["exec_post"]]
            n.pop("_exec_pre_seq", None)
            n.pop("_exec_post_seq", None)
        # sort children by exec_pre so a parent's subtree reads in forward order
        for n in self.nodes.values():
            n["children"].sort(key=lambda c: self.nodes[c]["exec_pre"]
                               if c in self.nodes and self.nodes[c]["exec_pre"] is not None
                               else 1e9)
        return {"nodes": self.nodes, "root": self._root_name or ""}

    def save(self, path: str, meta: dict) -> None:
        graph = self.finalize()
        graph["meta"] = meta
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.torch.save(graph, path)
        n_raw = sum(1 for n in self.nodes.values()
                    for s in (n["in"], n["out"]) for e in s.values()
                    if e.get("raw") is not None)
        n_drop = sum(1 for n in self.nodes.values()
                     for s in (n["in"], n["out"]) for e in s.values()
                     if e.get("dropped"))
        stack_driven = sum(1 for n in self.nodes.values() if n.get("parent"))
        print(f"  saved {len(self.nodes)} nodes, {n_raw} raw tensors "
              f"({n_drop} non-tensor slots), {stack_driven} stack-assigned parents → {path}")


def _tensor_raw_bytes(t):
    """Return (raw_bytes, shape, dtype_str) for a CPU tensor — original dtype,
    row-major, stride-agnostic (works for bfloat16 and non-contiguous views).

    The returned bytes are exactly ``numel * element_size`` long; the HTML viewer
    decodes them client-side (it already knows how to read fp16/bf16/fp32/int*).
    We flatten to a 1-D contiguous tensor first whose only stride is 1 element,
    so ``view(torch.uint8)`` is always legal (same idiom transformers uses for
    fp8 packing) and the copy is a single C-level memcpy — never the per-byte
    Python iteration ``bytes(UntypedStorage)`` would do.
    """
    if t is None:
        return None, None, None
    flat = t.detach().reshape(-1).contiguous()
    nbytes = flat.numel() * flat.element_size()
    try:
        b = flat.numpy().tobytes()                  # fp16/fp32/int* / bool
    except (TypeError, RuntimeError):
        # bfloat16 / other numpy-unsupported dtypes → 1-D contiguous ⇒ legal uint8 view
        import torch
        b = flat.view(torch.uint8)[:nbytes].numpy().tobytes()
    return b, list(t.shape), str(flat.dtype)


def write_graph_json(graph: dict, path: str, logp=None,
                     max_raw_bytes: int | None = None) -> None:
    """Write a browser-loadable ``graph.json`` + a ``tensors/`` dir of per-tensor
    ``.bin`` files (original dtype, row-major) next to it.

    Instead of inlining every tensor's base64 bytes into graph.json (which blew
    it up to ~800 MB and made the browser fail with 'Unexpected end of JSON
    input'), each captured tensor is sharded into its own small file
    ``tensors/<id>.bin``. graph.json then carries only metadata + stats + a
    ``file`` pointer, so it's a few hundred KB and trivially parseable. The HTML
    viewer reads individual ``.bin`` files ON DEMAND (only the tensors you click
    on) via the File System Access API — it never loads them all at once.

    ``max_raw_bytes`` is kept for API compatibility but currently unused
    (sharding per tensor already bounds what the browser pulls in one go).
    """
    out_dir = os.path.dirname(path) or "."
    tensors_dir = os.path.join(out_dir, "tensors")
    os.makedirs(tensors_dir, exist_ok=True)

    out = {"meta": graph.get("meta", {}), "root": graph.get("root", ""),
           "nodes": {}, "logp": None, "tensors_dir": "tensors"}
    tensor_id = 0
    for fqn, node in graph["nodes"].items():
        jnode = {
            "class": node.get("class"),
            "parent": node.get("parent"),       # runtime stack-assigned parent
            "children": list(node.get("children", [])),
            "exec_pre": node.get("exec_pre"),   # push tick (prefill pass)
            "exec_post": node.get("exec_post"), # pop tick
            "call_depth": node.get("call_depth"),  # nesting level = stack depth at push
            "push_count": node.get("push_count"),  # times entered (reused submodules >1)
            "in": {}, "out": {},
        }
        for slot in ("in", "out"):
            for path_, entry in node.get(slot, {}).items():
                raw_t = entry.get("raw")
                raw_meta = None
                try:
                    b, shape, dtype_str = _tensor_raw_bytes(raw_t)
                except Exception:
                    b = None
                if b is not None:
                    fname = f"{tensor_id:06d}.bin"
                    with open(os.path.join(tensors_dir, fname), "wb") as f:
                        f.write(b)
                    raw_meta = {"shape": shape, "dtype": dtype_str,
                                "file": f"tensors/{fname}",
                                "bytes": len(b)}
                    tensor_id += 1
                jnode[slot][path_] = {
                    "stats": entry.get("stats"),
                    "raw": raw_meta,
                }
        out["nodes"][fqn] = jnode
    if logp is not None:
        # logp is stored as a tensor file too (keeps graph.json tiny); the HTML
        # summary reads it on demand like any other tensor.
        import numpy as np
        lp = np.asarray(logp, dtype=np.float32)
        with open(os.path.join(tensors_dir, "logp.bin"), "wb") as f:
            f.write(lp.tobytes())
        out["logp"] = {"shape": list(lp.shape), "dtype": "torch.float32",
                       "file": "tensors/logp.bin", "bytes": lp.nbytes}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(_sanitize_for_json(out), f, allow_nan=False)


def _sanitize_for_json(obj):
    """Replace NaN/Infinity floats with None so the output is STRICT JSON.

    ``json.dump`` (default) emits bare ``NaN``/``Infinity``, which Python
    accepts on read-back but the browser ``JSON.parse`` rejects (the spec has
    no NaN/Infinity). Mixed-precision inference routinely produces NaN stats,
    so this is hit in practice. We walk the structure and swap non-finite
    floats for None, leaving every numeric value valid JSON.
    """
    import math
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _load_graph(path: str) -> dict:
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


# ─────────────────────────── gen_input ──────────────────────────────


def mode_gen_input(args: argparse.Namespace) -> None:
    import numpy as np
    if not args.model_path:
        sys.exit("--model_path is required for gen_input")
    os.makedirs(args.out_dir, exist_ok=True)

    builder = _resolve_builder(args.model_path, args.arch)
    text = args.text or args.prompt
    if text is None:
        text = DEFAULT_PROMPT if args.input_mode == "text" else "Describe this image in one short sentence."
    image_paths = []
    fake_image = bool(getattr(args, "fake_image", False))
    if args.input_mode == "image_text" and not fake_image:
        if not args.image_path:
            sys.exit("[gen_input] --input_mode image_text requires --image_path PATH[,PATH] "
                     "(or pass --fake_image to synthesize one).")
        image_paths = [p for p in args.image_path.split(",") if p]

    spec = builder.build(
        text=text, image_paths=image_paths, seq_len=args.seq_len,
        fake_image=fake_image,
        fake_image_size=getattr(args, "fake_image_size", 448),
        fake_image_count=getattr(args, "fake_image_count", 1),
        out_dir=args.out_dir,
    )
    spec.meta.update({"model_path": args.model_path, "arch": builder.name,
                      "seq_len": len(spec.token_ids), "builder_mode": args.input_mode,
                      "fake_image": fake_image})

    # Save the single source of truth + a plain numpy copy for back-compat.
    spec.save(os.path.join(args.out_dir, "input_spec.pt"))
    np.save(os.path.join(args.out_dir, "input.npy"),
            np.asarray(spec.token_ids, dtype=np.int64))
    with open(os.path.join(args.out_dir, "input_meta.json"), "w") as f:
        json.dump({**spec.meta, "token_ids": spec.token_ids}, f, indent=2,
                  ensure_ascii=False, default=str)
    kind = f"{spec.mode} ({len(spec.token_ids)} tokens"
    if spec.mode == "image_text":
        kind += f", {len(spec.image_paths)} image(s)"
    kind += ")"
    print(f"[gen_input] {builder.name} :: {kind} → {args.out_dir}/input_spec.pt")
    print(f"[gen_input] first 16 ids: {spec.token_ids[:16]}")


# ─────────────────────────── turbo runner ───────────────────────────


def _init_dist_single_gpu() -> None:
    import torch
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29555")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://", rank=0, world_size=1)


def _attr_path(obj, path: str):
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def _resolve_hook_root(model, args, builder=None, spec=None):
    """Pick the module to hook from. Precedence:
      1. explicit --hook_root
      2. builder.default_hook_root(spec)  (mm → full model incl. vision tower)
      3. fallback heuristics over language_backbone / model / root
    """
    import torch
    if args.hook_root:
        return _attr_path(model, args.hook_root), args.hook_root
    if builder is not None and spec is not None:
        dr = builder.default_hook_root(spec)
        if dr is not None:
            try:
                m = model if dr == "" else _attr_path(model, dr)
                if isinstance(m, torch.nn.Module):
                    return m, dr
            except Exception:
                pass
    for path in ("model.language_model", "model", "language_model", ""):
        try:
            m = _attr_path(model, path) if path else model
            if isinstance(m, torch.nn.Module):
                return m, path
        except Exception:
            continue
    return model, ""


def mode_turbo(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    if not args.model_path:
        sys.exit("--model_path is required for turbo mode")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ["TORCH_TURBO_SP_SIZE"] = "1"
    os.environ["TORCH_TURBO_EP_SIZE"] = "1"
    os.environ.setdefault("TORCH_TURBO_BACKEND", "fsdp2")

    _ensure_torchturbo_on_path()
    import torchturbo.models  # noqa: F401
    from torchturbo.engine.core import TorchTurboEngine

    _init_dist_single_gpu()
    torch.cuda.set_device(0)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}
    compute_dtype = dtype_map[args.dtype]

    engine = TorchTurboEngine()
    print("[turbo] building model (ep=1, sp=1, world=1) via TorchTurboEngine …")
    model = engine.build_model(args.model_path, torch_dtype=compute_dtype,
                               trust_remote_code=True, load_weights=True)
    model.eval()

    # Load the shared input spec + builder first so the hook root can follow the
    # builder's default (mm → full model incl. vision tower; text → backbone).
    spec = InputSpec.load(os.path.join(args.out_dir, "input_spec.pt"))
    builder = _resolve_builder(args.model_path, args.arch)
    hook_root_mod, hook_root_name = _resolve_hook_root(model, args, builder, spec)
    print(f"[turbo] hook root = {hook_root_name or '<root>'}")

    # Build forward kwargs from the one shared spec → turbo/vllm symmetric.
    fwd_kwargs = builder.turbo_kwargs(spec, device="cuda", dtype=compute_dtype)
    ids = fwd_kwargs["input_ids"]
    seq_len = ids.shape[1]
    print(f"[turbo] input mode={spec.mode} seq_len={seq_len}"
          + (f" n_images={len(spec.image_paths)}" if spec.mode == "image_text" else ""))

    rec = GraphRecorder(hook_filter=args.hook_filter)
    rec.register(hook_root_mod, root_name=hook_root_name)

    # NOTE: no extra autocast — run the model with its own dtype / forward semantics
    # so the precision environment matches online training. For image_text the
    # builder passes position_ids=None + mm_token_type_ids so the model computes
    # its 3D mrope positions itself (matching online inference / vLLM).
    with torch.no_grad():
        out = model(**fwd_kwargs)
    rec.remove()
    logits = out.logits if hasattr(out, "logits") and out.logits is not None else out[0]
    print(f"[turbo] logits {tuple(logits.shape)}; captured {len(rec.nodes)} nodes")

    logits_f = logits.float()
    log_probs = torch.log_softmax(logits_f, dim=-1)
    tgt = ids[0, 1:]
    gathered = log_probs[0, :-1, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    logp = torch.zeros(seq_len, dtype=torch.float32)
    logp[1:] = gathered.cpu()

    out_dir = os.path.join(args.out_dir, "turbo")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "logp.npy"), logp.numpy())
    param_dtype = next((str(p.dtype) for _, p in model.named_parameters()), None)
    rec.save(os.path.join(out_dir, "graph.pt"), {
        "backend": "torchturbo-fsdp2", "seq_len": int(seq_len),
        "hook_root": hook_root_name, "param_dtype_sample": param_dtype,
        "compute_dtype": str(compute_dtype), "hook_filter": args.hook_filter,
    })
    # browser-loadable twin of graph.pt (passes logp for the HTML summary)
    write_graph_json(_load_graph(os.path.join(out_dir, "graph.pt")),
                     os.path.join(out_dir, "graph.json"), logp=logp.numpy(),
                     max_raw_bytes=args.max_raw_bytes)


# ─────────────────────────── vllm runner ────────────────────────────


def mode_vllm(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from vllm import LLM, SamplingParams
    if not args.model_path:
        sys.exit("--model_path is required for vllm mode")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    dtype_map = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float"}
    vllm_dtype = dtype_map[args.dtype]

    print("[vllm] building LLM (tp=1, enforce_eager) …")
    llm = LLM(model=args.model_path, tensor_parallel_size=1, dtype=vllm_dtype,
              enforce_eager=True, disable_custom_all_reduce=True, seed=0,
              gpu_memory_utilization=0.7, trust_remote_code=True,
              max_model_len=max(args.seq_len + 16, 1024))

    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    if not isinstance(model, torch.nn.Module):
        inner = getattr(model, "model", None)
        if inner is not None:
            model = inner

    hook_root_mod, hook_root_name = _resolve_hook_root(model, args)
    print(f"[vllm] hook root = {hook_root_name or '<root>'}")

    # Load the shared input spec + builder; build the vLLM prompt symmetrically
    # with turbo (both derived from one InputSpec). Re-resolve hook root with the
    # builder so mm mode captures the vision tower here too.
    spec = InputSpec.load(os.path.join(args.out_dir, "input_spec.pt"))
    builder = _resolve_builder(args.model_path, args.arch)
    hook_root_mod, hook_root_name = _resolve_hook_root(model, args, builder, spec)
    print(f"[vllm] hook root = {hook_root_name or '<root>'} (resolved)")
    prompt = builder.vllm_prompt(spec)
    seq_len = len(spec.token_ids)
    print(f"[vllm] input mode={spec.mode} seq_len={seq_len}"
          + (f" n_images={len(spec.image_paths)}" if spec.mode == "image_text" else ""))

    rec = GraphRecorder(hook_filter=args.hook_filter)
    rec.register(hook_root_mod, root_name=hook_root_name)

    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    print("[vllm] running prefill …")
    outputs = llm.generate(prompts=[prompt], sampling_params=sp, use_tqdm=False)
    rec.remove()
    print(f"[vllm] captured {len(rec.nodes)} nodes")

    out0 = outputs[0]
    pl = out0.prompt_logprobs
    # Prefer vLLM's actually-tokenized prompt token ids (it tokenizes/expands
    # multimodal prompts itself), so per-position logp alignment is exact. Fall
    # back to the shared spec token_ids only for text mode where they coincide.
    vllm_tok = getattr(out0, "prompt_token_ids", None) or spec.token_ids
    vllm_len = len(vllm_tok)
    if vllm_len != seq_len:
        print(f"[vllm] NOTE: vLLM prompt length {vllm_len} differs from spec "
              f"token_ids {seq_len} (expected for image_text — vLLM expanded the "
              f"image placeholder itself). Using vLLM's length for logp capture.")
    seq_len = vllm_len
    tok_ids = vllm_tok
    logp = np.zeros(seq_len, dtype=np.float32)
    n = 0
    for j in range(1, seq_len):
        d = pl[j] if j < len(pl) else None
        if not d:
            continue
        lp = d.get(tok_ids[j]) or next(iter(d.values()))
        val = getattr(lp, "logprob", None)
        if val is None and isinstance(lp, dict):
            val = lp.get("logprob")
        logp[j] = float(val)
        n += 1
    print(f"[vllm] collected {n} prompt logprobs")

    out_dir = os.path.join(args.out_dir, "vllm")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "logp.npy"), logp)
    param_dtype = next((str(p.dtype) for _, p in model.named_parameters()
                        if p.numel() > 0), None)
    rec.save(os.path.join(out_dir, "graph.pt"), {
        "backend": "vllm-prefill", "seq_len": int(seq_len),
        "hook_root": hook_root_name, "param_dtype_sample": param_dtype,
        "compute_dtype": vllm_dtype, "hook_filter": args.hook_filter,
    })
    write_graph_json(_load_graph(os.path.join(out_dir, "graph.pt")),
                     os.path.join(out_dir, "graph.json"), logp=logp,
                     max_raw_bytes=args.max_raw_bytes)


# ─────────────────────────── alignment ──────────────────────────────


_DROP_WRAP = {"model", "language_model"}


def _suffix(fqn: str) -> str:
    return ".".join(s for s in fqn.split(".") if s not in _DROP_WRAP)


def _node_sort_key(fqn: str):
    segs = []
    for s in _suffix(fqn).split("."):
        segs.append((1, int(s), "") if s.isdigit() else (0, 0, s))
    return segs


def _build_pairs(t_nodes: dict, v_nodes: dict):
    """Pair nodes by aligned suffix. Returns (pairs, only_t, only_v).

    pairs: list of (turbo_fqn, vllm_fqn, suffix)."""
    t_by = {}
    for f in t_nodes:
        t_by.setdefault(_suffix(f), f)
    v_by = {}
    for f in v_nodes:
        v_by.setdefault(_suffix(f), f)
    common = set(t_by) & set(v_by)
    pairs = sorted(((t_by[s], v_by[s], s) for s in common), key=lambda t: _node_sort_key(t[2]))
    used_t = {p[0] for p in pairs}
    used_v = {p[1] for p in pairs}
    only_t = sorted([f for f in t_nodes if f not in used_t], key=_node_sort_key)
    only_v = sorted([f for f in v_nodes if f not in used_v], key=_node_sort_key)
    return pairs, only_t, only_v


def _slot_rel(t_entry: dict | None, v_entry: dict | None) -> float | None:
    """max(norm_rel, sum_rel) between two tensor stat entries (any path)."""
    if not t_entry or not v_entry:
        return None
    a, b = t_entry["stats"], v_entry["stats"]
    _, nr = _rel_diff(a.get("norm"), b.get("norm"))
    _, sr = _rel_diff(a.get("sum"), b.get("sum"))
    return max(nr, sr)


def _slot_rel_by_path(node_a: dict, node_b: dict, slot: str) -> dict[str, float | None]:
    """Per-path rel diffs (matched by path label) for a slot across two nodes."""
    a_slots = node_a.get(slot, {})
    b_slots = node_b.get(slot, {})
    out = {}
    for path in sorted(set(a_slots) | set(b_slots)):
        out[path] = _slot_rel(a_slots.get(path), b_slots.get(path))
    return out


def _relationships(t_nodes: dict, v_nodes: dict, fqn: str) -> dict:
    """parent / children / prev-next sibling (data-flow) for a node, derived from
    exec_pre within the tree."""
    node = t_nodes.get(fqn) or v_nodes.get(fqn)
    parent = node.get("parent")
    siblings = []
    if parent and parent in (t_nodes | v_nodes):
        pnode = t_nodes.get(parent) or v_nodes.get(parent)
        siblings = [c for c in pnode.get("children", []) if c != fqn]
    siblings_sorted = sorted(siblings, key=lambda c: (t_nodes.get(c, {})
                             .get("exec_pre") or v_nodes.get(c, {})
                             .get("exec_pre") or 1 << 30))
    prev = _pre_sibling(siblings_sorted, fqn, t_nodes, v_nodes) if siblings_sorted else None
    nxt = _post_sibling(siblings_sorted, fqn, t_nodes, v_nodes) if siblings_sorted else None
    return {"parent": parent, "children": list(node.get("children", [])),
            "prev_sibling": prev, "next_sibling": nxt,
            "exec_pre": node.get("exec_pre"), "exec_post": node.get("exec_post")}


def _pre_sibling(sorted_sibs, fqn, t_nodes, v_nodes):
    n_seq = (t_nodes.get(fqn, {}).get("exec_pre") or v_nodes.get(fqn, {}).get("exec_pre"))
    cand = [c for c in sorted_sibs
            if ((t_nodes.get(c, {}).get("exec_pre") or v_nodes.get(c, {}).get("exec_pre")
                 or 1 << 30) < (n_seq or 1 << 30))]
    return cand[-1] if cand else None


def _post_sibling(sorted_sibs, fqn, t_nodes, v_nodes):
    n_seq = (t_nodes.get(fqn, {}).get("exec_pre") or v_nodes.get(fqn, {}).get("exec_pre")
             or 1 << 30)
    cand = [c for c in sorted_sibs
            if ((t_nodes.get(c, {}).get("exec_pre") or v_nodes.get(c, {}).get("exec_pre")
                 or 1 << 30) > n_seq)]
    return cand[0] if cand else None


# ─────────────────────────── compare (CLI) ──────────────────────────


def _fmt(v, w=12):
    if v is None:
        return "N/A".rjust(w)
    if isinstance(v, float):
        return f"{v:.3e}".rjust(w) if math.isfinite(v) else "nan".rjust(w)
    return str(v)[:w].rjust(w)


def mode_compare(args: argparse.Namespace) -> None:
    import numpy as np
    tdir = os.path.join(args.out_dir, "turbo")
    vdir = os.path.join(args.out_dir, "vllm")
    for d in (tdir, vdir):
        if not os.path.exists(os.path.join(d, "graph.pt")):
            sys.exit(f"[compare] missing {d}/graph.pt — run --mode turbo/--mode vllm first")
    tg = _load_graph(os.path.join(tdir, "graph.pt"))
    vg = _load_graph(os.path.join(vdir, "graph.pt"))
    t_nodes, v_nodes = tg["nodes"], vg["nodes"]
    t_meta, v_meta = tg.get("meta", {}), vg.get("meta", {})
    t_logp = np.load(os.path.join(tdir, "logp.npy"))
    v_logp = np.load(os.path.join(vdir, "logp.npy"))

    pairs, only_t, only_v = _build_pairs(t_nodes, v_nodes)

    report: dict[str, Any] = {"meta": {"turbo": t_meta, "vllm": v_meta},
                              "struct": {}, "pairs": [], "logp": {}}

    print("\n" + "=" * 96)
    print("  STRUCTURAL CHECK (paired vs structural-diff boxes)")
    print("=" * 96)
    print(f"  turbo nodes={len(t_nodes)}  vllm nodes={len(v_nodes)}  paired={len(pairs)}")
    report["struct"] = {"turbo_n": len(t_nodes), "vllm_n": len(v_nodes),
                        "paired": len(pairs), "only_turbo": only_t, "only_vllm": only_v}
    if only_t:
        print(f"  ⚠️  {len(only_t)} boxes only in turbo (operator exists only on turbo side):")
        for f in only_t[:15]:
            print(f"        turbo: {f}  [{t_nodes[f].get('class')}]")
    if only_v:
        print(f"  ⚠️  {len(only_v)} boxes only in vllm (operator exists only on vLLM side):")
        for f in only_v[:15]:
            print(f"        vllm:  {f}  [{v_nodes[f].get('class')}]")

    # ── per-pair diff + onset ──
    pair_rows = []
    first_internal = None
    first_any = None
    dtype_mismatches: list[tuple[str, str, str, str]] = []  # (suf, slot, path, "ta vs vb")
    for t_f, v_f, suf in pairs:
        tn, vn = t_nodes[t_f], v_nodes[v_f]
        # aggregate over paths: worst across in/out paths
        in_rels = _slot_rel_by_path(tn, vn, "in")
        out_rels = _slot_rel_by_path(tn, vn, "out")
        in_rel = max((r for r in in_rels.values() if r is not None), default=None)
        out_rel = max((r for r in out_rels.values() if r is not None), default=None)
        # record dtype mismatches per (slot, path) — original dtypes preserved in
        # the recorded raw tensors / stats; dtype diff is a divergence source.
        for slot, rels in (("in", in_rels), ("out", out_rels)):
            tslot, vslot = tn.get(slot, {}), vn.get(slot, {})
            for path in set(tslot) | set(vslot):
                te, ve = tslot.get(path), vslot.get(path)
                da = te["stats"]["dtype"] if te else None
                db = ve["stats"]["dtype"] if ve else None
                if da and db and da != db:
                    dtype_mismatches.append((suf, slot, path, f"{da} vs {db}"))
        internal = (out_rel is not None and out_rel > args.onset_threshold
                    and (in_rel is None or in_rel <= args.onset_threshold))
        row = {"turbo": t_f, "vllm": v_f, "suffix": suf, "class": tn.get("class"),
               "in_rel": in_rel, "out_rel": out_rel,
               "in_paths": in_rels, "out_paths": out_rels,
               "exec_pre": tn.get("exec_pre"), "internal_onset": bool(internal)}
        pair_rows.append(row)
        report["pairs"].append({k: row[k] for k in
                                ("turbo", "vllm", "suffix", "class", "in_rel", "out_rel",
                                 "internal_onset", "exec_pre")})
        if out_rel is not None and out_rel > args.onset_threshold and first_any is None:
            first_any = row
        if internal and first_internal is None:
            first_internal = row

    if dtype_mismatches:
        report["dtype_mismatches"] = [list(m) for m in dtype_mismatches]
        print(f"  ⚠️  {len(dtype_mismatches)} tensor(s) with differing turbo/vllm dtype "
              f"(dtype is itself a divergence source):")
        for suf, slot, path, d in dtype_mismatches[:15]:
            print(f"        {suf} :: {slot} :: {path}  ({d})")

    print("\n" + "=" * 96)
    print("  DIVERGENCE LOCALIZATION")
    print("=" * 96)
    if first_internal is None and first_any is None:
        print(f"  ✅ No paired box's output exceeds onset threshold ({args.onset_threshold:.0e}).")
    elif first_internal is not None:
        r = first_internal
        print(f"  📍 First INTERNAL divergence (this box broke; its input aligned):")
        print(f"     turbo : {r['turbo']}  [{r['class']}]")
        print(f"     vllm  : {r['vllm']}")
        print(f"     input  rel : {_fmt(r['in_rel'])}    output rel : {_fmt(r['out_rel'])}")
        rel = _relationships(t_nodes, v_nodes, r["turbo"])
        print(f"     parent        : {rel['parent']}")
        print(f"     prev sibling  : {rel['prev_sibling']}")
        print(f"     next sibling  : {rel['next_sibling']}")
        print(f"     children      : {', '.join(rel['children']) or '(none)'}")
        print(f"     exec_pre/post : {rel['exec_pre']} / {rel['exec_post']}")
        print(f"     → investigate this box's operators / op order.")
        report["first_internal"] = {k: r[k] for k in ("turbo", "vllm", "class",
                                                        "in_rel", "out_rel")}
        report["first_internal"].update(rel)
    else:
        r = first_any
        print(f"  🌊 First divergence is INHERITED (input already differs):")
        print(f"     turbo : {r['turbo']}  [{r['class']}]")
        print(f"     vllm  : {r['vllm']}")
        print(f"     input  rel : {_fmt(r['in_rel'])}    output rel : {_fmt(r['out_rel'])}")
        print(f"     → diff is upstream; walk earlier boxes.")

    # ── top-K ──
    if args.top_k > 0 and pair_rows:
        ranked = sorted([r for r in pair_rows if r["out_rel"] is not None],
                        key=lambda r: r["out_rel"], reverse=True)
        print("\n" + "=" * 96)
        print(f"  TOP-{min(args.top_k, len(ranked))} PAIRS BY OUTPUT REL DIFF")
        print("=" * 96)
        print(f"  {'suffix':<42}{'in_rel':>11}{'out_rel':>11}  class")
        for r in ranked[: args.top_k]:
            print(f"  {r['suffix']:<42}{_fmt(r['in_rel']):>11}{_fmt(r['out_rel']):>11}"
                  f"  {r['class']}")

    # ── explicit select: print one node's full relationship + per-tensor stats ──
    if args.select:
        _cli_select(args.select, tdir, vdir, t_nodes, v_nodes, pairs)

    # ── tree ──
    if args.show_tree:
        _cli_tree(t_nodes, v_nodes, pair_rows, args)

    # ── logp ──
    print("\n" + "=" * 96)
    print("  PER-TOKEN LOGP COMPARISON")
    print("=" * 96)
    n = min(len(t_logp), len(v_logp))
    a, b = t_logp[1:n], v_logp[1:n]
    ad = np.abs(a - b)
    mean_abs = float(ad.mean())
    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.where(denom < 1e-12, 1e-12, denom)
    within = float((ad <= args.logp_mean_threshold).mean())
    print(f"  positions compared: {len(a)} (1..{n-1})")
    print(f"  turbo logp[1:6]: {t_logp[1:6].tolist()}")
    print(f"  vllm  logp[1:6]: {v_logp[1:6].tolist()}")
    print(f"  mean abs logp diff : {mean_abs:.6e}")
    print(f"  max  abs logp diff : {float(ad.max()):.6e}")
    print(f"  frac positions ≤ {args.logp_mean_threshold:.0e}: {within*100:.1f}%")
    passed = mean_abs <= args.logp_mean_threshold
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  threshold (mean ≤ {args.logp_mean_threshold:.0e}): {status}")
    report["logp"] = {"n_positions": int(len(a)), "mean_abs": mean_abs,
                      "max_abs": float(ad.max()), "within_frac": within,
                      "passed": bool(passed)}

    print("\n" + "=" * 96)
    print("  FINAL VERDICT")
    print("=" * 96)
    fi = report.get("first_internal")
    print(f"  logp mean abs diff      : {mean_abs:.6e}  ({status})")
    print(f"  first internal divergence: {fi['turbo'] if fi else 'none'}")
    overall = passed
    report["overall_pass"] = bool(overall)
    print(f"  Overall: {'✅ PASS' if overall else '❌ FAIL'}")
    print("=" * 96)
    if not passed and fi is None and not only_t and not only_v:
        print("  ⚠️  logp diverges but no internal onset & no structural diff.")
        print("      Likely in lm_head/log_softmax/gather or an unpaired operator.")
        print("      Use --select <box> or --html to inspect per-tensor diffs.")

    if args.export_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.export_json)), exist_ok=True)
        with open(args.export_json, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  📄 Report → {args.export_json}")
    sys.exit(0 if overall else 1)


def _cli_select(spec: str, tdir: str, vdir: str, t_nodes: dict, v_nodes: dict,
                pairs: list) -> None:
    import numpy as np
    import torch
    # resolve turbo fqn (accept full or suffix)
    target = spec.strip()
    t_f = target if target in t_nodes else _find_by_suffix(t_nodes, target)
    v_f = None
    if t_f:
        for tf, vf, _ in pairs:
            if tf == t_f:
                v_f = vf
                break
    if v_f is None:  # try vllm side
        v_f = target if target in v_nodes else _find_by_suffix(v_nodes, target)
        t_f = next((tf for tf, vf, _ in pairs if vf == v_f), None)
    print("\n" + "=" * 96)
    print(f"  SELECTED BOX")
    print("=" * 96)
    print(f"  turbo : {t_f}  [{type_name(t_nodes.get(t_f, {}).get('class'))}]")
    print(f"  vllm  : {v_f}  [{type_name(v_nodes.get(v_f, {}).get('class'))}]")
    if t_f:
        rel = _relationships(t_nodes, v_nodes, t_f)
        print(f"  parent={rel['parent']}  prev_sib={rel['prev_sibling']}  "
              f"next_sib={rel['next_sibling']}")
        print(f"  children={rel['children']}  exec_pre/post={rel['exec_pre']}/{rel['exec_post']}")
    # per-tensor detail + raw element-wise diff
    _print_slots(t_nodes.get(t_f, {}), v_nodes.get(v_f, {}), "in")
    _print_slots(t_nodes.get(t_f, {}), v_nodes.get(v_f, {}), "out")


def _print_slots(tn: dict, vn: dict, slot: str) -> None:
    ts, vs = tn.get(slot, {}), vn.get(slot, {})
    print(f"\n  ── {slot} tensors ──")
    for path in sorted(set(ts) | set(vs)):
        te, ve = ts.get(path), vs.get(path)
        print(f"   [{path}]")
        print(f"     turbo: {te['stats'] if te else '—'}"
              f"  (raw {'yes' if te and te.get('raw') is not None else 'no'})")
        print(f"     vllm : {ve['stats'] if ve else '—'}"
              f"  (raw {'yes' if ve and ve.get('raw') is not None else 'no'})")
        if te and ve:
            r = _slot_rel(te, ve)
            print(f"     rel diff: {_fmt(r)}")
            # ── dtype-aware element-wise diff ──
            # recorded raw tensors keep their ORIGINAL dtype; dtype differences
            # are a divergence source on their own, so surface them explicitly:
            #   same dtype  → direct compare
            #   diff dtype  → warn, then cast both to fp32 and compare numerically
            if te.get("raw") is not None and ve.get("raw") is not None:
                a, b = te["raw"], ve["raw"]
                if a.shape != b.shape:
                    print(f"     shape mismatch: {list(a.shape)} vs {list(b.shape)}")
                    continue
                dtype_a, dtype_b = str(a.dtype), str(b.dtype)
                if dtype_a == dtype_b:
                    print(f"     dtype 相同 ({dtype_a}) → 直接对比")
                    af, bf = a.float(), b.float()
                else:
                    print(f"     ⚠️ dtype 不同: turbo={dtype_a}  vllm={dtype_b} "
                          f"→ 转 fp32 对比 (注意：部分差异可能源自 dtype 量化本身)")
                    af, bf = a.float(), b.float()
                d = (af - bf).abs()
                print(f"     element-wise: mean_abs={d.mean():.3e} max_abs={d.max():.3e}")
            elif te and ve and te["stats"]["dtype"] != ve["stats"]["dtype"]:
                print(f"     ⚠️ dtype 不同: turbo={te['stats']['dtype']} "
                      f"vllm={ve['stats']['dtype']} (无 raw，仅统计可见)")


def _cli_tree(t_nodes, v_nodes, pair_rows, args) -> None:
    by_t = {p["turbo"]: p for p in pair_rows}
    print("\n" + "=" * 96)
    print("  MODULE RELATIONSHIP TREE (turbo, with aligned vLLM box + output rel)")
    print("  ● aligned  ◆ internal onset  ○ inherited / structural-only")
    print("=" * 96)
    for fqn in sorted(t_nodes.keys(), key=_node_sort_key):
        n = t_nodes[fqn]
        depth = _depth(fqn)
        indent = "  " + "  " * min(depth, 24)
        pr = by_t.get(fqn)
        if pr is None:
            mark, vq, d = "○", "(no vllm pair)", ""
        elif pr["internal_onset"]:
            mark, vq = "◆", pr["vllm"]
            d = f" out_rel={_fmt(pr['out_rel'],10)} in_rel={_fmt(pr['in_rel'],10)}"
        else:
            ok = pr["out_rel"] is None or pr["out_rel"] <= args.onset_threshold
            mark = "●" if ok else "○"
            vq, d = pr["vllm"], (f" out_rel={_fmt(pr['out_rel'],10)}"
                                 if pr["out_rel"] is not None else "")
        print(f"  {mark} {indent}{fqn or '<root>'}  [{n.get('class')}]")
        print(f"        ↳ vllm: {vq}{d}")


def _depth(fqn: str) -> int:
    return fqn.count(".")


def _find_by_suffix(nodes: dict, target: str) -> str | None:
    tgt = _suffix(target)
    for f in nodes:
        if _suffix(f) == tgt:
            return f
    best = None
    for f in nodes:
        s = _suffix(f)
        if s.endswith("." + tgt) or tgt.endswith("." + s):
            best = f
    return best


def type_name(c):
    return c or "?"




def mode_pt2json(args: argparse.Namespace) -> None:
    """Convert an existing ``graph.pt`` (+ ``logp.npy``) → browser-loadable
    ``graph.json``.

    For dumps produced before the recorder wrote ``graph.json`` automatically, or
    when you want to regenerate it. Reads the .pt, maps every recorded tensor to
    original-dtype base64 + stats + relationships, and (optionally) folds in the
    per-token logp from the companion ``logp.npy``.

    Usage::

        python compare_train_infer.py --mode pt2json --pt turbo/graph.pt
        python compare_train_infer.py --mode pt2json --pt turbo/graph.pt \\
            --logp_npy turbo/logp.npy --out_json turbo/graph.json
    """
    if not args.pt:
        sys.exit("[pt2json] --pt <side>/graph.pt is required")
    if not os.path.exists(args.pt):
        sys.exit(f"[pt2json] --pt not found: {args.pt}")

    print(f"[pt2json] loading {args.pt} …")
    graph = _load_graph(args.pt)
    n_nodes = len(graph.get("nodes", {}))
    n_raw = sum(1 for n in graph["nodes"].values()
                for s in (n.get("in", {}), n.get("out", {}))
                for e in s.values() if e.get("raw") is not None)
    print(f"[pt2json] {n_nodes} nodes, {n_raw} raw tensors")

    # logp: explicit --logp_npy, else look for logp.npy next to the .pt.
    logp = None
    logp_path = args.logp_npy
    if logp_path is None:
        cand = os.path.join(os.path.dirname(args.pt), "logp.npy")
        if os.path.exists(cand):
            logp_path = cand
    if logp_path is not None and os.path.exists(logp_path):
        import numpy as np
        logp = np.load(logp_path)
        print(f"[pt2json] logp loaded from {logp_path} ({len(logp)} positions)")

    out_json = args.out_json or os.path.join(os.path.dirname(args.pt), "graph.json")
    write_graph_json(graph, out_json, logp=logp, max_raw_bytes=args.max_raw_bytes)
    size_mb = os.path.getsize(out_json) / 1e6
    tensors_dir = os.path.join(os.path.dirname(out_json) or ".", "tensors")
    n_bins = len(os.listdir(tensors_dir)) if os.path.isdir(tensors_dir) else 0
    print(f"[pt2json] wrote {out_json} ({size_mb:.1f} MB) + tensors/ ({n_bins} .bin files)")
    print("[pt2json] now: open scripts/rl/report.html in a browser and click "
          "'turbo 文件夹' / 'vllm 文件夹', selecting the dir that contains "
          "graph.json + tensors/. (Use the File System Access–enabled folder "
          "picker so .bin tensors load on demand.)")





# ─────────────────────────────── main ───────────────────────────────


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.mode == "gen_input":
        mode_gen_input(args)
    elif args.mode == "turbo":
        mode_turbo(args)
    elif args.mode == "vllm":
        mode_vllm(args)
    elif args.mode == "compare":
        mode_compare(args)
    elif args.mode == "pt2json":
        mode_pt2json(args)
    else:
        sys.exit(f"unknown mode {args.mode}")


if __name__ == "__main__":
    main()