#!/usr/bin/env python
"""Fine-grained diagnostics: vanilla LlamaFactory vs LlamaFactory + torchturbo.

Checks six levels of consistency on the very first training step:
  1. **Parameters** — per-named-parameter norm and sum after model loading
  2. **Buffers** — per-named-buffer norm and sum
  3. **Inputs** — the exact batch tensors entering the first forward pass
  4. **Layer I/O** — per-module input *and* output tensors from forward hooks,
     with divergence classification (match / internal / inherited)
  5. **Loss** — the scalar training loss
  6. **Saved weights** — post-training parameter tensors (bitwise comparison)

Usage:
    # Prerequisites — LlamaFactory must be importable. Either:
    #   Option A — editable install:
    pip install -e /path/to/LLaMA-Factory
    #   Option B — set PYTHONPATH:
    export PYTHONPATH=/path/to/LLaMA-Factory/src:$PYTHONPATH
    #
    # The dataset_dir (containing dataset_info.json) is auto-detected from
    # the llamafactory package location. To override: --dataset_dir /path/to/LLaMA-Factory/data

    # Step 1 — baseline (vanilla LlamaFactory)
    TORCH_TURBO_DISABLE=1 torchrun --nproc_per_node=NUM_GPUS scripts/compare_diagnostics.py \\
        --mode baseline \\
        --model_name_or_path /path/to/Qwen3-MoE \\
        --dataset identity \\
        --output_dir /tmp/compare_diagnostics \\
        --dump_params --dump_inputs --dump_layer_io

    # Step 2 — torchturbo
    torchrun --nproc_per_node=NUM_GPUS scripts/compare_diagnostics.py \\
        --mode torchturbo \\
        --model_name_or_path /path/to/Qwen3-MoE \\
        --dataset identity \\
        --output_dir /tmp/compare_diagnostics \\
        --dump_params --dump_inputs --dump_layer_io

    # Step 3 — compare
    python scripts/compare_diagnostics.py --mode compare \\
        --output_dir /tmp/compare_diagnostics \\
        --verbose_compare

    # Step 3 (alt) — compare with relaxed bf16 thresholds
    python scripts/compare_diagnostics.py --mode compare \\
        --output_dir /tmp/compare_diagnostics \\
        --input_threshold 1e-3 --output_threshold 1e-2

    # Narrow scope: only hook layer 0, depth ≤ 5
    torchrun --nproc_per_node=1 scripts/compare_diagnostics.py \\
        --mode baseline \\
        --model_name_or_path /path/to/Qwen3-MoE \\
        --output_dir /tmp/compare_diagnostics \\
        --hook_filter 'layers\\.0\\.' --max_hook_depth 5

Arguments:
    --mode              Required. One of: baseline, torchturbo, compare.
    --model_name_or_path  Path to model (required for baseline/torchturbo).
    --dataset           Dataset name for LlamaFactory (default: identity).
    --template          Chat template (default: qwen3).
    --output_dir        Directory for snapshots and reports (default: /tmp/compare_diagnostics).
    --cutoff_len        Max sequence length (default: 512).
    --learning_rate     Learning rate (default: 1e-5).
    --per_device_train_batch_size  Batch size per device (default: 1).
    --bf16              Enable bf16 training (default: True).
    --neat_packing      Enable neat packing (default: False).
    --seed              Random seed for reproducibility (default: 42).
    --flash_attn        Flash attention backend (default: fa2).
    --rank              Rank(s) to capture diagnostics on (default: [0]).
                        e.g. --rank 0 2 4. Each rank's artifacts are stored in
                        a standalone rank_<N>/ subdirectory under --output_dir.
                        In compare mode, comparison runs for each specified rank.
    --stage             LlamaFactory training stage: sft or pt (default: sft).
    --trainer_class     Fully qualified trainer class to monkey-patch.
                        Defaults to the stage-specific LlamaFactory trainer.
    --extra_lf_args     Extra LlamaFactory args as key-value pairs,
                        e.g. --extra_lf_args --gradient_checkpointing True
    --dataset_dir       Path to LlamaFactory data directory containing dataset_info.json.
                        If not set, auto-detected from the llamafactory package location
                        (works with both pip install -e and PYTHONPATH).

  Capture-mode only (baseline / torchturbo):
    --max_hook_depth    Max module depth for forward hooks (default: None = unlimited).
                        e.g. --max_hook_depth 3 hooks modules up to 3 levels deep.
    --hook_filter       Regex filter on module FQN (default: None = all modules).
                        e.g. --hook_filter 'layers\\.0\\.' to hook only layer 0.
    --dump_inputs       Save first-batch tensors as .pt files for bitwise comparison (default: False).
    --dump_params       Save raw parameter tensors as .pt files for bitwise comparison (default: False).
                        Enables element-wise param diff in compare mode.
    --dump_layer_io     Save raw layer I/O tensors as .pt files for element-wise diff (default: False).
                        Enables per-element max diff and histogram in compare mode.
    --dump_weights      Save model weights after the training step as .pt files (default: False).
                        Enables post-training weight comparison in compare mode.

  Compare-mode only:
    --top_k_layers      Show only top-K layers with largest diffs (default: 0 = all).
    --verbose_compare   Show per-tensor input/output stats for every mismatched module (default: False).
    --show_matched      List matched (non-divergent) modules in the divergence report (default: True).
    --input_threshold   Max input rel diff to consider 'same input' (default: 1e-6).
                        Lower = stricter. Modules with input diff above this AND output
                        diff above output_threshold are classified as 'inherited'.
    --output_threshold  Max output rel diff to consider 'match' (default: 1e-6).
                        Lower = stricter.
    --param_threshold   Max rel diff for params/buffers to consider 'match' (default: 1e-5).
    --export_json PATH  Export the full comparison report as JSON for post-processing (default: None).
                        e.g. --export_json /tmp/compare_diagnostics/report.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import sys
from collections import OrderedDict


@dataclasses.dataclass
class TensorComparisonResult:
    """Per-tensor comparison stats between baseline and torchturbo snapshots."""

    key: str  # e.g. "model.layers.0.self_attn|in|args.0"
    bl_stats: dict | None  # {shape, dtype, norm, sum, mean, max_abs, min_abs}
    tt_stats: dict | None
    norm_abs_diff: float
    norm_rel_diff: float
    sum_abs_diff: float
    sum_rel_diff: float
    shape_match: bool
    dtype_match: bool
    # Element-wise diff stats (populated only when raw tensors are available)
    max_elem_abs_diff: float | None = None
    max_elem_abs_diff_index: list | None = None
    diff_histogram: dict | None = None  # {bucket_label: count}


@dataclasses.dataclass
class ModuleDivergenceResult:
    """Per-module divergence classification and per-tensor comparison details."""

    fqn: str
    classification: str  # "match", "internal_divergence", "inherited_divergence"
    max_input_rel_diff: float
    max_output_rel_diff: float
    per_tensor_inputs: list[TensorComparisonResult]
    per_tensor_outputs: list[TensorComparisonResult]
    # RNG state comparison (None when rng_states not available)
    rng_cpu_match: bool | None = None
    rng_device_match: bool | None = None


DEFAULT_TRAINER_CLASSES = {
    "sft": "llamafactory.train.sft.trainer.CustomSeq2SeqTrainer",
    "pt": "llamafactory.train.pt.workflow.CustomTrainer",
}


# ───────────────────────── CLI ─────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-grained LlamaFactory training comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["baseline", "torchturbo", "compare"],
        required=True,
        help="Run mode",
    )
    p.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="HuggingFace model path (required for baseline/torchturbo)",
    )
    p.add_argument(
        "--dataset", type=str, default="identity", help="LlamaFactory dataset name"
    )
    p.add_argument(
        "--stage",
        choices=["sft", "pt"],
        default="sft",
        help="LlamaFactory training stage",
    )
    p.add_argument(
        "--template", type=str, default="qwen3", help="LlamaFactory chat template"
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/compare_diagnostics",
        help="Directory for snapshots and comparison output",
    )
    p.add_argument("--cutoff_len", type=int, default=512, help="Max sequence length")
    p.add_argument(
        "--learning_rate", type=float, default=1e-5, help="Training learning rate"
    )
    p.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size per device",
    )
    p.add_argument(
        "--bf16", action="store_true", default=True, help="Use bf16 mixed precision"
    )
    p.add_argument(
        "--neat_packing",
        action="store_true",
        default=False,
        help="Enable neat packing in LlamaFactory",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    p.add_argument(
        "--flash_attn", type=str, default="fa2", help="Flash attention backend"
    )
    p.add_argument(
        "--top_k_layers",
        type=int,
        default=0,
        help="If >0, only show top-K layers with largest diffs in compare mode",
    )
    p.add_argument(
        "--dump_inputs",
        action="store_true",
        help="Save first-batch tensors as .pt files for bitwise comparison",
    )
    p.add_argument(
        "--dump_params",
        action="store_true",
        help="Save raw parameter tensors as .pt files for bitwise comparison",
    )
    p.add_argument(
        "--extra_lf_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra LlamaFactory args, passed through as-is. "
        "Must be the LAST argument on the command line. "
        "e.g. --extra_lf_args --my_flag --gradient_checkpointing True",
    )
    p.add_argument(
        "--max_hook_depth",
        type=int,
        default=None,
        help="Max module depth for hooks (None=unlimited)",
    )
    p.add_argument(
        "--hook_filter",
        type=str,
        default=None,
        help="Regex filter on module FQN, e.g. 'layers\\.0\\.'",
    )
    p.add_argument(
        "--verbose_compare",
        action="store_true",
        help="Show per-tensor input/output stats for every mismatched module",
    )
    p.add_argument(
        "--input_threshold",
        type=float,
        default=1e-6,
        help="Max input rel diff to consider 'same input'",
    )
    p.add_argument(
        "--output_threshold",
        type=float,
        default=1e-6,
        help="Max output rel diff to consider 'match'",
    )
    p.add_argument(
        "--dump_layer_io",
        action="store_true",
        help="Save raw layer I/O tensors as .pt for element-wise diff (capture modes)",
    )
    p.add_argument(
        "--export_json",
        type=str,
        default=None,
        help="Export comparison report as JSON to this path (compare mode)",
    )
    p.add_argument(
        "--param_threshold",
        type=float,
        default=1e-5,
        help="Max rel diff for params/buffers to consider 'match'",
    )
    p.add_argument(
        "--show_matched",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="List matched (non-divergent) modules in the divergence report",
    )

    p.add_argument(
        "--rank",
        type=int,
        nargs="+",
        default=[0],
        help="Rank(s) to capture diagnostics on (e.g. --rank 0 2 4). "
        "Each rank's artifacts are stored in a standalone "
        "rank_<N>/ subdirectory under --output_dir. In compare "
        "mode, comparison runs for each specified rank. "
        "Defaults to rank 0.",
    )
    p.add_argument(
        "--trainer_class",
        type=str,
        default=None,
        help="Fully qualified trainer class to monkey-patch, "
        "e.g. 'llamafactory.train.sft.trainer.CustomSeq2SeqTrainer'. "
        "If omitted, the default is selected from --stage.",
    )
    p.add_argument(
        "--no_save_model",
        action="store_true",
        help="Skip saving the trained model/checkpoint after the run",
    )
    p.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Path to LlamaFactory data directory containing dataset_info.json. "
        "If not set, auto-detected from the llamafactory package location.",
    )
    return p.parse_args()


def resolve_trainer_class(args) -> str:
    """Return the trainer class to patch for the requested LF stage."""
    if args.trainer_class:
        return args.trainer_class
    try:
        return DEFAULT_TRAINER_CLASSES[args.stage]
    except KeyError as exc:
        raise ValueError(f"Unsupported LlamaFactory stage: {args.stage!r}") from exc


def get_rank():
    """Return the current process rank from the RANK env var (default 0)."""
    return int(os.environ.get("RANK", 0))


# ───────────────────── Diagnostic Tracker ──────────────────────


class DiagnosticTracker:
    """Captures parameter stats, input tensors, and per-layer forward outputs."""

    def __init__(self):
        self.param_stats: OrderedDict[str, dict] = OrderedDict()
        self.input_stats: OrderedDict[str, dict] = OrderedDict()
        self.layer_inputs: OrderedDict[str, dict] = OrderedDict()
        self.layer_outputs: OrderedDict[str, dict] = OrderedDict()
        self.buffer_stats: OrderedDict[str, dict] = OrderedDict()
        self.sampler_stats: OrderedDict[str, object] = OrderedDict()
        self.precision_stats: OrderedDict[str, object] = OrderedDict()
        self.rng_states: OrderedDict[str, dict] = OrderedDict()
        self.loss: float | None = None
        self._hooks: list = []

    # ── 1. Parameter snapshot ──

    @staticmethod
    def _unwrap_tensor(tensor):
        """Unwrap FSDP2 DTensor to get the full/local data for fair comparison.

        Tries full_tensor() first (gathers across ranks), falls back to
        _local_tensor, then plain .data.
        """
        data = tensor
        if hasattr(tensor, "full_tensor"):
            try:
                data = tensor.full_tensor()
            except Exception:
                if hasattr(tensor, "_local_tensor"):
                    data = tensor._local_tensor
                else:
                    data = tensor.data
        elif hasattr(tensor, "_local_tensor"):
            data = tensor._local_tensor
        return data

    @staticmethod
    def _tensor_stats(data) -> dict:
        """Compute norm / sum / mean / shape / dtype stats from an unwrapped tensor."""
        data_f = data.detach().cpu().float()
        return {
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "norm": data_f.norm().item(),
            "sum": data_f.sum().item(),
            "mean": data_f.mean().item() if data_f.numel() > 0 else None,
            "numel": data_f.numel(),
        }

    def snapshot_params(self, model):
        """Record norm / sum / shape / dtype for every named parameter."""
        for name, param in model.named_parameters():
            data = self._unwrap_tensor(param)
            self.param_stats[name] = self._tensor_stats(data)

    def snapshot_buffers(self, model):
        """Record norm / sum / shape / dtype for every named buffer."""
        for name, buf in model.named_buffers():
            data = self._unwrap_tensor(buf)
            self.buffer_stats[name] = self._tensor_stats(data)

    # ── 2. Input snapshot ──

    def snapshot_inputs(self, batch: dict):
        """Record stats (and partial content) of the first training batch."""
        import torch
        import hashlib

        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                val_cpu = val.detach().cpu()
                val_f = val_cpu.float()  # CPU for deterministic reduction
                entry = {
                    "shape": list(val.shape),
                    "dtype": str(val.dtype),
                    "norm": val_f.norm().item(),
                    "sum": val_f.sum().item(),
                    "mean": val_f.mean().item() if val_f.numel() > 0 else None,
                    "numel": val_f.numel(),
                }
                # Content hash for exact comparison (catches differences that
                # norm/sum miss, e.g. different token arrangements with same sum)
                # .numpy() doesn't support bf16 — view as raw bytes via storage
                entry["content_hash"] = hashlib.md5(
                    val_cpu.contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest()
                # Save first few values for spot-checking
                flat = val_cpu.reshape(-1)
                entry["first_16"] = flat[:16].tolist()
                entry["last_16"] = flat[-16:].tolist()
                self.input_stats[key] = entry
            elif val is not None:
                self.input_stats[key] = {
                    "type": type(val).__name__,
                    "repr": repr(val)[:200],
                }

    def snapshot_sampler(self, trainer):
        """Record sampler and dataloader state from the trainer."""
        try:
            dl = trainer.get_train_dataloader()
        except Exception:
            return
        sampler = getattr(dl, "sampler", None)
        self.sampler_stats["dataloader_workers"] = getattr(dl, "num_workers", -1)
        self.sampler_stats["batch_size"] = getattr(dl, "batch_size", -1)
        self.sampler_stats["sampler_type"] = (
            type(sampler).__name__ if sampler else "None"
        )
        # SequentialSampler uses `data_source`, DistributedSampler uses `dataset`
        ds = getattr(sampler, "dataset", None) or getattr(sampler, "data_source", None)
        if ds is not None:
            try:
                self.sampler_stats["dataset_size"] = len(ds)
            except Exception:
                pass
        for attr in ["rank", "world_size", "num_replicas", "seed", "epoch", "shuffle"]:
            if hasattr(sampler, attr):
                self.sampler_stats[f"sampler_{attr}"] = getattr(sampler, attr)

    def snapshot_precision(self, model, trainer=None):
        """Record mixed precision / FSDP precision state.

        Note: autocast_enabled / autocast_dtype are captured inside the first
        forward hook fire (where autocast is actually active), not here.
        This method records the static trainer config and param metadata.
        """
        import torch

        # 1. Detect device type (also used by the hook for autocast query)
        device_type = "cuda"
        try:
            from llamafactory.extras.misc import is_npu_available

            npu = is_npu_available() if callable(is_npu_available) else is_npu_available
            if npu:
                device_type = "npu"
        except ImportError:
            pass
        self.precision_stats["device_type"] = device_type

        # 2. HF Trainer mixed precision config
        if trainer is not None:
            args = getattr(trainer, "args", None)
            if args is not None:
                self.precision_stats["trainer_bf16"] = getattr(args, "bf16", None)
                self.precision_stats["trainer_fp16"] = getattr(args, "fp16", None)
                self.precision_stats["trainer_bf16_full_eval"] = getattr(
                    args, "bf16_full_eval", None
                )
                fsdp = getattr(args, "fsdp", None)
                self.precision_stats["trainer_fsdp"] = str(fsdp) if fsdp else None

        # 3. Sample param dtypes (first 5 unique)
        dtypes = set()
        for name, p in model.named_parameters():
            dtypes.add(str(p.dtype))
            if len(dtypes) >= 5:
                break
        self.precision_stats["param_dtypes"] = sorted(dtypes)

        # 4. FSDP2 MixedPrecisionPolicy (if model is FSDP-wrapped)
        try:
            from torch.distributed.fsdp import FSDPModule

            if isinstance(model, FSDPModule):
                mp = getattr(model, "_mp_policy", None)
                if mp is not None:
                    self.precision_stats["fsdp2_param_dtype"] = str(
                        getattr(mp, "param_dtype", None)
                    )
                    self.precision_stats["fsdp2_reduce_dtype"] = str(
                        getattr(mp, "reduce_dtype", None)
                    )
                    self.precision_stats["fsdp2_output_dtype"] = str(
                        getattr(mp, "output_dtype", None)
                    )
                else:
                    self.precision_stats["fsdp2_mp_policy"] = "None"
            else:
                self.precision_stats["fsdp2_wrapped"] = False
        except ImportError:
            self.precision_stats["fsdp2_available"] = False

        # 5. Check if any module is a DTensor-wrapped param
        self.precision_stats["has_dtensor_params"] = False
        for name, p in model.named_parameters():
            if type(p).__name__ == "DTensor" or hasattr(p, "_local_tensor"):
                self.precision_stats["has_dtensor_params"] = True
                self.precision_stats["dtensor_sample_param"] = name
                break

        # 6. Gradient checkpointing state
        self.precision_stats["gradient_checkpointing"] = getattr(
            model, "is_gradient_checkpointing", False
        )

    def dump_params(self, model, output_dir: str, label: str):
        """Save all parameters as a single .pt file keyed by param name.

        Using a single dict avoids filesystem naming issues (dots, slashes,
        FSDP prefixes) and makes bitwise comparison straightforward.

        IMPORTANT: Must be called on ALL ranks because _unwrap_tensor may call
        full_tensor() which is a collective op. The caller must gate the call
        or pass only on the rank that should write.
        """
        import torch

        state = {}
        for name, param in model.named_parameters():
            data = self._unwrap_tensor(param)
            state[name] = data.detach().cpu()
        dump_path = os.path.join(output_dir, f"{label}_params.pt")
        torch.save(state, dump_path)
        print(f"[{label}] Dumped {len(state)} params → {dump_path}")

    def dump_batch(self, batch: dict, output_dir: str, label: str):
        """Save every tensor in the batch as a .pt file for bitwise comparison.

        Unlike dump_params, batch tensors are local (not DTensors) so there's
        no collective hang risk. The caller must gate the call to the desired rank.
        """
        import torch

        os.makedirs(output_dir, exist_ok=True)
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                path = os.path.join(output_dir, f"{label}_{key}.pt")
                torch.save(val.detach().cpu(), path)
                print(f"[{label}] Dumped '{key}' {list(val.shape)} → {path}")

    # ── 3. Forward hooks ──

    def register_hooks(self, model, max_depth=None, filter_pattern=None, dump_dir=None):
        """Register forward hooks on modules, optionally filtered by depth/pattern.

        Args:
            model: The nn.Module to hook.
            max_depth: If None, hook all non-root modules. If set, hook only
                modules whose FQN depth (number of dot-separated segments) is
                ≤ max_depth.
            filter_pattern: If not None, compiled as a regex; only modules whose
                FQN matches (via ``re.search``) are hooked.
            dump_dir: If not None, save raw layer I/O tensors as .pt files
                for element-wise diff comparison.

        Both filters are conjunctive — a module must pass both to be hooked.
        """
        import torch

        # Guard: remove any existing hooks before registering new ones
        if self._hooks:
            self.remove_hooks()

        compiled_pattern = (
            re.compile(filter_pattern) if filter_pattern is not None else None
        )

        def _extract_tensors(item, prefix, storage):
            if item is None:
                return
            if isinstance(item, torch.Tensor):
                # Unwrap DTensor to local shard — avoids collective ops inside
                # hooks and ensures shape/stats are consistent.  We compare
                # local shards (same rank, same mesh) so this is correct for
                # baseline-vs-torchturbo comparison.
                is_dtensor = hasattr(item, "_local_tensor")
                if is_dtensor:
                    local = item._local_tensor
                else:
                    local = item
                tf = local.detach().cpu().float()
                abs_tf = tf.abs()
                entry = {
                    "shape": list(local.shape),
                    "dtype": str(local.dtype),
                    "norm": tf.norm().item(),
                    "sum": tf.sum().item(),
                    "mean": tf.mean().item() if tf.numel() > 0 else None,
                    "max_abs": abs_tf.max().item() if tf.numel() > 0 else None,
                    "min_abs": abs_tf.min().item() if tf.numel() > 0 else None,
                }
                if is_dtensor:
                    entry["global_shape"] = list(item.shape)
                    entry["placements"] = str(item.placements)
                storage[prefix] = entry
                # Save raw tensor for element-wise comparison
                if dump_dir is not None:
                    safe_name = prefix.replace("|", "_").replace("/", "_")
                    pt_path = os.path.join(dump_dir, f"{safe_name}.pt")
                    torch.save(tf, pt_path)
            elif isinstance(item, (tuple, list)):
                for i, val in enumerate(item):
                    _extract_tensors(val, f"{prefix}.{i}", storage)
            elif isinstance(item, dict):
                for k, val in item.items():
                    _extract_tensors(val, f"{prefix}.{k}", storage)

        # Track which modules have already fired so gradient-checkpointing
        # recomputation doesn't overwrite the original forward-pass capture.
        _captured: set[str] = set()
        _pre_captured: set[str] = set()

        def _make_pre_hook(fqn: str):
            def pre_hook(_module, args, kwargs):
                """Capture RNG state BEFORE the module's forward pass."""
                if fqn in _pre_captured:
                    return
                _pre_captured.add(fqn)

                # Capture autocast state on the very first hook fire — this runs
                # inside the actual forward pass where autocast is active.
                if not self.precision_stats.get("_autocast_captured"):
                    import torch as _torch

                    device_type = self.precision_stats.get("device_type", "cuda")
                    try:
                        self.precision_stats["autocast_enabled"] = (
                            _torch.is_autocast_enabled(device_type)
                        )
                    except TypeError:
                        self.precision_stats["autocast_enabled"] = (
                            _torch.is_autocast_enabled()
                        )
                    try:
                        self.precision_stats["autocast_dtype"] = str(
                            _torch.get_autocast_dtype(device_type)
                        )
                    except (TypeError, RuntimeError):
                        try:
                            self.precision_stats["autocast_dtype"] = str(
                                _torch.get_autocast_gpu_dtype()
                            )
                        except Exception:
                            pass
                    self.precision_stats["_autocast_captured"] = True

                # Capture RNG state hash BEFORE forward for accurate divergence
                # detection.  Previously this was in the post-hook, which
                # captured the state AFTER the module ran — masking divergences
                # when the module itself consumed RNG.
                import torch as _t
                import hashlib as _hl

                rng_entry = {
                    "cpu": _hl.md5(
                        _t.random.get_rng_state().numpy().tobytes()
                    ).hexdigest(),
                }
                device_type = self.precision_stats.get("device_type", "cuda")
                try:
                    if device_type == "npu":
                        rng_entry["device"] = _hl.md5(
                            _t.npu.get_rng_state().numpy().tobytes()
                        ).hexdigest()
                    elif _t.cuda.is_available():
                        rng_entry["device"] = _hl.md5(
                            _t.cuda.get_rng_state().numpy().tobytes()
                        ).hexdigest()
                except Exception:
                    pass
                self.rng_states[fqn] = rng_entry

            return pre_hook

        def _make_hook(fqn: str):
            def hook(_module, args, kwargs, output):
                """Captures ALL input (positional/keyword) and output tensors."""
                # Skip recomputation passes (gradient checkpointing fires hooks twice)
                if fqn in _captured:
                    return
                _captured.add(fqn)

                _extract_tensors(args, f"{fqn}|in|args", self.layer_inputs)
                _extract_tensors(kwargs, f"{fqn}|in|kwargs", self.layer_inputs)
                _extract_tensors(output, f"{fqn}|out|", self.layer_outputs)

            return hook

        for name, child in model.named_modules():
            if not name:  # skip root module itself
                continue
            depth = len(name.split("."))
            # Depth filter: if max_depth is set, skip modules deeper than max_depth
            if max_depth is not None and depth > max_depth:
                continue
            # Regex filter: if filter_pattern is set, skip non-matching FQNs
            if compiled_pattern is not None and not compiled_pattern.search(name):
                continue
            # Pre-hook: capture RNG state before forward
            ph = child.register_forward_pre_hook(_make_pre_hook(name), with_kwargs=True)
            self._hooks.append(ph)
            # Post-hook: capture input/output tensors after forward
            h = child.register_forward_hook(_make_hook(name), with_kwargs=True)
            self._hooks.append(h)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ── Serialization ──

    def to_dict(self) -> dict:
        return {
            "param_stats": dict(self.param_stats),
            "buffer_stats": dict(self.buffer_stats),
            "input_stats": dict(self.input_stats),
            "layer_inputs": dict(self.layer_inputs),
            "layer_outputs": dict(self.layer_outputs),
            "sampler_stats": dict(self.sampler_stats),
            "precision_stats": {
                k: v for k, v in self.precision_stats.items() if not k.startswith("_")
            },
            "rng_states": dict(self.rng_states),
            "loss": self.loss,
        }

    def save(self, path: str):
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ───────────────────── Divergence Analyzer ─────────────────────


class DivergenceAnalyzer:
    """Classify each module's divergence type and produce per-tensor comparison data."""

    EPSILON = 1e-12

    # Histogram bucket boundaries for element-wise absolute diffs
    _HIST_BOUNDS = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0]
    _HIST_LABELS = ["==0", "<1e-8", "<1e-6", "<1e-4", "<1e-2", "<1e-1", "<1.0", "≥1.0"]

    def __init__(
        self,
        input_threshold: float = 1e-6,
        output_threshold: float = 1e-6,
        output_dir: str | None = None,
    ):
        self.input_threshold = input_threshold
        self.output_threshold = output_threshold
        self.output_dir = output_dir

    # ── helpers ──

    @staticmethod
    def _rel_diff(a: float | None, b: float | None) -> tuple[float, float]:
        """Return (abs_diff, rel_diff) using abs(a-b)/max(abs(a), abs(b), epsilon).

        Using max of both values prevents false-positive divergence when both
        values are near-zero (e.g. both ~1e-15).
        """
        if a is None or b is None:
            return (float("nan"), float("nan"))
        abs_d = abs(a - b)
        rel_d = abs_d / max(abs(a), abs(b), DivergenceAnalyzer.EPSILON)
        return abs_d, rel_d

    # ── classification ──

    def _classify(self, max_in_rd: float, max_out_rd: float) -> str:
        """Classify a module's divergence based on input/output relative diffs."""
        if max_out_rd <= self.output_threshold:
            return "match"
        if max_in_rd <= self.input_threshold:
            return "internal_divergence"
        return "inherited_divergence"

    # ── per-tensor comparison ──

    def _compare_tensor_group(
        self, bl_data: dict, tt_data: dict, key_prefix: str
    ) -> list[TensorComparisonResult]:
        """Compare tensors whose keys start with *key_prefix* across both dicts."""
        bl_keys = {k for k in bl_data if k.startswith(key_prefix)}
        tt_keys = {k for k in tt_data if k.startswith(key_prefix)}
        all_keys = sorted(bl_keys | tt_keys)

        results: list[TensorComparisonResult] = []
        for key in all_keys:
            bl_stats = bl_data.get(key)
            tt_stats = tt_data.get(key)

            if bl_stats is None or tt_stats is None:
                results.append(
                    TensorComparisonResult(
                        key=key,
                        bl_stats=bl_stats,
                        tt_stats=tt_stats,
                        norm_abs_diff=float("nan"),
                        norm_rel_diff=float("nan"),
                        sum_abs_diff=float("nan"),
                        sum_rel_diff=float("nan"),
                        shape_match=False,
                        dtype_match=False,
                    )
                )
                continue

            norm_ad, norm_rd = self._rel_diff(
                bl_stats.get("norm"), tt_stats.get("norm")
            )
            sum_ad, sum_rd = self._rel_diff(bl_stats.get("sum"), tt_stats.get("sum"))
            shape_match = bl_stats.get("shape") == tt_stats.get("shape")
            dtype_match = bl_stats.get("dtype") == tt_stats.get("dtype")

            # Element-wise comparison from dumped .pt files
            max_elem_abs_diff = None
            max_elem_abs_diff_index = None
            diff_histogram = None

            if self.output_dir is not None and shape_match:
                safe_name = key.replace("|", "_").replace("/", "_")
                bl_pt = os.path.join(
                    self.output_dir, f"baseline_layer_io/{safe_name}.pt"
                )
                tt_pt = os.path.join(
                    self.output_dir, f"torchturbo_layer_io/{safe_name}.pt"
                )
                if os.path.exists(bl_pt) and os.path.exists(tt_pt):
                    try:
                        import torch

                        bl_t = torch.load(bl_pt, map_location="cpu", weights_only=True)
                        tt_t = torch.load(tt_pt, map_location="cpu", weights_only=True)
                        if bl_t.shape == tt_t.shape:
                            abs_diff = (bl_t - tt_t).abs()
                            max_val, max_idx_flat = abs_diff.reshape(-1).max(dim=0)
                            max_elem_abs_diff = max_val.item()
                            # Convert flat index to multi-dim index
                            max_elem_abs_diff_index = list(
                                torch.unravel_index(max_idx_flat, abs_diff.shape)
                            )
                            max_elem_abs_diff_index = [
                                i.item() for i in max_elem_abs_diff_index
                            ]
                            # Histogram — separate exact zeros from nonzero diffs
                            flat_diff = abs_diff.reshape(-1)
                            hist = {}
                            exact_zero = int((flat_diff == 0).sum().item())
                            hist[self._HIST_LABELS[0]] = exact_zero  # "==0"
                            nonzero_diff = flat_diff[flat_diff > 0]
                            prev = 0.0
                            for bound, label in zip(
                                self._HIST_BOUNDS, self._HIST_LABELS[1:-1]
                            ):
                                count = int(
                                    ((nonzero_diff > prev) & (nonzero_diff <= bound))
                                    .sum()
                                    .item()
                                )
                                hist[label] = count
                                prev = bound
                            hist[self._HIST_LABELS[-1]] = int(
                                (nonzero_diff > self._HIST_BOUNDS[-1]).sum().item()
                            )
                            diff_histogram = hist
                    except Exception:
                        pass

            results.append(
                TensorComparisonResult(
                    key=key,
                    bl_stats=bl_stats,
                    tt_stats=tt_stats,
                    norm_abs_diff=norm_ad,
                    norm_rel_diff=norm_rd,
                    sum_abs_diff=sum_ad,
                    sum_rel_diff=sum_rd,
                    shape_match=shape_match,
                    dtype_match=dtype_match,
                    max_elem_abs_diff=max_elem_abs_diff,
                    max_elem_abs_diff_index=max_elem_abs_diff_index,
                    diff_histogram=diff_histogram,
                )
            )
        return results

    # ── main entry point ──

    def analyze(
        self,
        bl_inputs: dict,
        tt_inputs: dict,
        bl_outputs: dict,
        tt_outputs: dict,
        bl_rng: dict | None = None,
        tt_rng: dict | None = None,
    ) -> list[ModuleDivergenceResult]:
        """Analyze all modules and classify divergence type per-module."""
        # Extract unique FQNs from all four dicts.
        # Keys use "|in|" / "|out|" as separators (pipe chars never appear in
        # module FQNs), so splitting is unambiguous even for modules named
        # "input_layernorm" etc.
        all_keys = (
            list(bl_inputs.keys())
            + list(tt_inputs.keys())
            + list(bl_outputs.keys())
            + list(tt_outputs.keys())
        )
        fqns: set[str] = set()
        for k in all_keys:
            for sep in ("|in|", "|out|"):
                idx = k.find(sep)
                if idx != -1:
                    fqns.add(k[:idx])
                    break

        # Track per-side FQN sets for mismatch reporting
        bl_fqns: set[str] = set()
        tt_fqns: set[str] = set()
        for k in list(bl_inputs.keys()) + list(bl_outputs.keys()):
            for sep in ("|in|", "|out|"):
                idx = k.find(sep)
                if idx != -1:
                    bl_fqns.add(k[:idx])
                    break
        for k in list(tt_inputs.keys()) + list(tt_outputs.keys()):
            for sep in ("|in|", "|out|"):
                idx = k.find(sep)
                if idx != -1:
                    tt_fqns.add(k[:idx])
                    break
        self._only_in_bl = sorted(bl_fqns - tt_fqns)
        self._only_in_tt = sorted(tt_fqns - bl_fqns)

        # Preserve baseline input key order (= forward pass order) for reporting
        self._bl_layer_in_keys = list(bl_inputs.keys())

        # Build FQN → forward-order index map (hooks fire in forward pass order)
        self._fqn_forward_order: dict[str, int] = {}
        _idx = 0
        for k in self._bl_layer_in_keys:
            for sep in ("|in|", "|out|"):
                pos = k.find(sep)
                if pos != -1:
                    fqn = k[:pos]
                    if fqn not in self._fqn_forward_order:
                        self._fqn_forward_order[fqn] = _idx
                        _idx += 1
                    break

        def _max_rd(tensors: list[TensorComparisonResult]) -> float:
            """Max of norm_rel_diff and sum_rel_diff across all tensors (NaN-safe)."""
            vals = []
            for r in tensors:
                if r.norm_rel_diff == r.norm_rel_diff:  # skip NaN
                    vals.append(r.norm_rel_diff)
                if r.sum_rel_diff == r.sum_rel_diff:
                    vals.append(r.sum_rel_diff)
            return max(vals, default=0.0)

        def _analyze_one_fqn(fqn: str) -> ModuleDivergenceResult:
            in_prefix = f"{fqn}|in|"
            out_prefix = f"{fqn}|out|"

            per_in = self._compare_tensor_group(bl_inputs, tt_inputs, in_prefix)
            per_out = self._compare_tensor_group(bl_outputs, tt_outputs, out_prefix)

            max_in_rd = _max_rd(per_in)
            max_out_rd = _max_rd(per_out)

            classification = self._classify(max_in_rd, max_out_rd)
            # RNG state comparison for this module
            rng_cpu_match = None
            rng_device_match = None
            if bl_rng is not None and tt_rng is not None:
                br = bl_rng.get(fqn, {})
                tr = tt_rng.get(fqn, {})
                if br or tr:
                    rng_cpu_match = br.get("cpu") == tr.get("cpu")
                    # Only compare device RNG when at least one side captured it;
                    # if neither has a "device" key the result is unknown (None).
                    if "device" in br or "device" in tr:
                        rng_device_match = br.get("device") == tr.get("device")
                    else:
                        rng_device_match = None
            return ModuleDivergenceResult(
                fqn=fqn,
                classification=classification,
                max_input_rel_diff=max_in_rd,
                max_output_rel_diff=max_out_rd,
                per_tensor_inputs=per_in,
                per_tensor_outputs=per_out,
                rng_cpu_match=rng_cpu_match,
                rng_device_match=rng_device_match,
            )

        fqn_list = sorted(fqns)
        results = [_analyze_one_fqn(fqn) for fqn in fqn_list]

        # Sort by max_output_rel_diff descending
        results.sort(key=lambda r: r.max_output_rel_diff, reverse=True)
        return results

    @staticmethod
    def find_first_divergent(
        results: list[ModuleDivergenceResult],
        fqn_forward_order: dict[str, int] | None = None,
    ) -> ModuleDivergenceResult | None:
        """Find the first module with internal divergence in forward order.

        Among all modules classified as internal_divergence, containers whose
        FQN is a strict prefix of another internal divergence are excluded
        (they diverge only because a child diverged).  From the remaining
        candidates the one that fires earliest in the forward pass is the
        most likely root cause — this handles VLM / multi-branch architectures
        where depth alone is misleading.

        Falls back to natural numeric FQN order when forward order is unavailable.
        If all internal divergences are containers, returns the earliest container.
        """
        internal = [r for r in results if r.classification == "internal_divergence"]
        if not internal:
            return None

        # Exclude containers: a module is a container if another internal
        # divergence's FQN starts with "<fqn>."
        internal_fqns = {r.fqn for r in internal}
        non_containers = [
            r
            for r in internal
            if not any(f.startswith(r.fqn + ".") for f in internal_fqns)
        ]
        candidates = non_containers if non_containers else internal

        if fqn_forward_order:
            candidates.sort(key=lambda r: fqn_forward_order.get(r.fqn, float("inf")))
        else:
            # Fallback: natural numeric FQN order
            candidates.sort(
                key=lambda r: (
                    [(int(s) if s.isdigit() else s) for s in r.fqn.split(".")],
                )
            )
        return candidates[0]

    # ── reporting ──

    def print_report(
        self,
        results: list[ModuleDivergenceResult],
        top_k: int = 0,
        verbose: bool = False,
        show_matched: bool = False,
    ) -> None:
        """Print formatted divergence report with per-tensor detail.

        Args:
            results: List of per-module divergence results from analyze().
            top_k: If >0, limit to top_k mismatched modules. 0 means show all.
            verbose: If True, print per-tensor baseline vs torchturbo stats.
            show_matched: If True, list all matched (non-divergent) modules.
        """
        total = len(results)
        matched = sum(1 for r in results if r.classification == "match")
        internal = sum(1 for r in results if r.classification == "internal_divergence")
        inherited = sum(
            1 for r in results if r.classification == "inherited_divergence"
        )

        print("\n" + "=" * 80)
        print("  DIVERGENCE ANALYSIS REPORT")
        print("=" * 80)

        only_bl = getattr(self, "_only_in_bl", [])
        only_tt = getattr(self, "_only_in_tt", [])
        bl_count = total - len(only_tt)  # modules in baseline
        tt_count = total - len(only_bl)  # modules in torchturbo

        print(f"  Baseline modules:       {bl_count}")
        print(f"  TorchTurbo modules:     {tt_count}")
        print(f"  Total (union):          {total}")
        print(f"  Matched:                {matched}")
        print(f"  Internal divergences:   {internal}")
        print(f"  Inherited divergences:  {inherited}")
        print(
            f"  Thresholds:             input={self.input_threshold:.0e}  output={self.output_threshold:.0e}"
        )
        print(
            f"  (in_rd/out_rd = max relative diff across all input/output tensors per module)"
        )
        print()
        print(f"  Classification guide:")
        print(
            f"    match               — in_rd ≤ threshold AND out_rd ≤ threshold (module behaves the same)"
        )
        print(
            f"    internal_divergence — in_rd ≤ threshold BUT out_rd > threshold (this module causes the diff)"
        )
        print(
            f"    inherited_divergence— in_rd > threshold (inputs already differ from upstream)"
        )

        # RNG summary across all analyzed modules
        rng_checked = [r for r in results if r.rng_cpu_match is not None]
        if rng_checked:
            rng_cpu_mismatches = sum(1 for r in rng_checked if r.rng_cpu_match is False)
            rng_dev_mismatches = sum(
                1 for r in rng_checked if r.rng_device_match is False
            )
            rng_any_mismatch = sum(
                1
                for r in rng_checked
                if r.rng_cpu_match is False or r.rng_device_match is False
            )
            if rng_any_mismatch:
                print(
                    f"  RNG mismatches:         {rng_any_mismatch}/{len(rng_checked)} modules "
                    f"(cpu: {rng_cpu_mismatches}, device: {rng_dev_mismatches})"
                )
            else:
                print(
                    f"  RNG states:             ✅ all {len(rng_checked)} modules match"
                )

        # Show matched modules when requested (in forward order)
        if show_matched:
            matched_results = [r for r in results if r.classification == "match"]
            if matched_results:
                fwd = getattr(self, "_fqn_forward_order", None) or {}
                matched_results.sort(key=lambda r: fwd.get(r.fqn, float("inf")))

                def _is_bitwise_equal(
                    per_tensors: list[TensorComparisonResult],
                ) -> bool | None:
                    """True if all tensors are bitwise equal, False if any differ, None if no dump data."""
                    has_dump = any(t.max_elem_abs_diff is not None for t in per_tensors)
                    if not has_dump:
                        return None
                    return all(
                        t.max_elem_abs_diff == 0.0
                        for t in per_tensors
                        if t.max_elem_abs_diff is not None
                    )

                def _short_key(key: str) -> str:
                    """Extract the short tensor name from a full key like 'fqn|in|args.0'."""
                    for sep in ("|in|", "|out|"):
                        idx = key.find(sep)
                        if idx != -1:
                            return key[idx + len(sep) :]
                    return key

                def _print_tensor_detail(
                    t: TensorComparisonResult, indent: str = "          "
                ) -> None:
                    """Print per-tensor comparison detail."""
                    bl = t.bl_stats or {}
                    tt = t.tt_stats or {}
                    name = _short_key(t.key)
                    shape_str = str(bl.get("shape", "?"))
                    dtype_str = bl.get("dtype", "?")

                    def _fmtv(v):
                        return f"{v:.6e}" if isinstance(v, (int, float)) else str(v)

                    print(f"{indent}{name}  shape={shape_str}  dtype={dtype_str}")
                    print(
                        f"{indent}  norm: bl={_fmtv(bl.get('norm', 'N/A'))}  tt={_fmtv(tt.get('norm', 'N/A'))}  rel_diff={t.norm_rel_diff:.2e}"
                    )
                    print(
                        f"{indent}  sum:  bl={_fmtv(bl.get('sum', 'N/A'))}  tt={_fmtv(tt.get('sum', 'N/A'))}  rel_diff={t.sum_rel_diff:.2e}"
                    )
                    if t.max_elem_abs_diff is not None:
                        if t.max_elem_abs_diff == 0.0:
                            print(f"{indent}  element-wise: bitwise identical")
                        else:
                            print(
                                f"{indent}  element-wise: max_abs_diff={t.max_elem_abs_diff:.2e} at index {t.max_elem_abs_diff_index}"
                            )
                            if t.diff_histogram:
                                parts = [
                                    f"{k}:{v}"
                                    for k, v in t.diff_histogram.items()
                                    if v > 0
                                ]
                                print(f"{indent}  histogram: {', '.join(parts)}")

                # Check if dump_layer_io data is available for any matched module
                has_any_dump = any(
                    any(
                        t.max_elem_abs_diff is not None
                        for t in r.per_tensor_inputs + r.per_tensor_outputs
                    )
                    for r in matched_results
                )

                if has_any_dump:
                    # Split into bitwise-identical vs within-threshold
                    bitwise_eq = []
                    within_thresh = []
                    for r in matched_results:
                        in_bw = _is_bitwise_equal(r.per_tensor_inputs)
                        out_bw = _is_bitwise_equal(r.per_tensor_outputs)
                        if (in_bw is True or in_bw is None) and (
                            out_bw is True or out_bw is None
                        ):
                            bitwise_eq.append(r)
                        else:
                            within_thresh.append(r)

                    if bitwise_eq:
                        print(f"\n  ✅ Bitwise identical modules ({len(bitwise_eq)}):")
                        print(
                            f"      All input and output tensors are exactly equal (zero diff)."
                        )
                        for r in bitwise_eq:
                            n_in = len(r.per_tensor_inputs)
                            n_out = len(r.per_tensor_outputs)
                            rng_tag = ""
                            if r.rng_cpu_match is not None:
                                rng_ok = r.rng_cpu_match and (
                                    r.rng_device_match is None or r.rng_device_match
                                )
                                rng_tag = "  rng=✅" if rng_ok else "  rng=❌"
                            print(f"      {r.fqn}  ({n_in} in, {n_out} out){rng_tag}")

                    if within_thresh:
                        print(
                            f"\n  ⚠️  Within threshold but not bitwise identical ({len(within_thresh)}):"
                        )
                        print(
                            f"      Relative diff ≤ {self.output_threshold:.0e} but element-wise differences exist."
                        )
                        print(
                            f"      These are numerically close but not exact — typically from floating-point"
                        )
                        print(
                            f"      non-determinism (reductions, different op order, etc.)."
                        )
                        for r in within_thresh:
                            n_in = len(r.per_tensor_inputs)
                            n_out = len(r.per_tensor_outputs)
                            rng_tag = ""
                            if r.rng_cpu_match is not None:
                                rng_ok = r.rng_cpu_match and (
                                    r.rng_device_match is None or r.rng_device_match
                                )
                                rng_tag = "  rng=✅" if rng_ok else "  rng=❌"
                            print(
                                f"\n      {r.fqn}  in_rd={r.max_input_rel_diff:.2e}  out_rd={r.max_output_rel_diff:.2e}{rng_tag}"
                            )
                            if r.per_tensor_inputs:
                                print(f"        Inputs ({n_in} tensor(s)):")
                                for t in r.per_tensor_inputs:
                                    _print_tensor_detail(t)
                            if r.per_tensor_outputs:
                                print(f"        Outputs ({n_out} tensor(s)):")
                                for t in r.per_tensor_outputs:
                                    _print_tensor_detail(t)
                else:
                    # No dump data — show as threshold-matched with summary stats
                    print(f"\n  ✅ Matched modules ({matched}, in forward order):")
                    print(
                        f"      Relative diff ≤ threshold (no dump data for bitwise check)."
                    )
                    for r in matched_results:
                        n_in = len(r.per_tensor_inputs)
                        n_out = len(r.per_tensor_outputs)
                        in_rd = (
                            f"in_rd={r.max_input_rel_diff:.2e}"
                            if r.max_input_rel_diff > 0
                            else "in_rd=0"
                        )
                        out_rd = (
                            f"out_rd={r.max_output_rel_diff:.2e}"
                            if r.max_output_rel_diff > 0
                            else "out_rd=0"
                        )
                        rng_tag = ""
                        if r.rng_cpu_match is not None:
                            rng_ok = r.rng_cpu_match and (
                                r.rng_device_match is None or r.rng_device_match
                            )
                            rng_tag = "  rng=✅" if rng_ok else "  rng=❌"
                        print(
                            f"      {r.fqn}  {in_rd} ({n_in} tensors)  {out_rd} ({n_out} tensors){rng_tag}"
                        )

        # Report modules only in one snapshot
        if only_bl:
            print(f"\n  ⚠️  {len(only_bl)} module(s) only in baseline:")
            for fqn in only_bl[:10]:
                print(f"      {fqn}")
            if len(only_bl) > 10:
                print(f"      ... and {len(only_bl) - 10} more")
        if only_tt:
            print(f"\n  ⚠️  {len(only_tt)} module(s) only in torchturbo:")
            for fqn in only_tt[:10]:
                print(f"      {fqn}")
            if len(only_tt) > 10:
                print(f"      ... and {len(only_tt) - 10} more")

        # Report per-module tensor count mismatches (input/output arity differs)
        tensor_mismatches = []
        for r in results:
            bl_in = sum(1 for t in r.per_tensor_inputs if t.bl_stats is not None)
            tt_in = sum(1 for t in r.per_tensor_inputs if t.tt_stats is not None)
            bl_out = sum(1 for t in r.per_tensor_outputs if t.bl_stats is not None)
            tt_out = sum(1 for t in r.per_tensor_outputs if t.tt_stats is not None)
            if bl_in != tt_in or bl_out != tt_out:
                tensor_mismatches.append((r, bl_in, tt_in, bl_out, tt_out))
        if tensor_mismatches:
            print(
                f"\n  ⚠️  {len(tensor_mismatches)} module(s) with different tensor counts:"
            )
            for r, bi, ti, bo, to_ in tensor_mismatches[:10]:
                print(f"      {r.fqn}  inputs: {bi} vs {ti}  outputs: {bo} vs {to_}")
            if len(tensor_mismatches) > 10:
                print(f"      ... and {len(tensor_mismatches) - 10} more")
            # Show which specific keys are missing (only for mismatched modules)
            for r, *_ in tensor_mismatches:
                only_bl_in = [
                    t.key
                    for t in r.per_tensor_inputs
                    if t.bl_stats is not None and t.tt_stats is None
                ]
                only_tt_in = [
                    t.key
                    for t in r.per_tensor_inputs
                    if t.tt_stats is not None and t.bl_stats is None
                ]
                only_bl_out = [
                    t.key
                    for t in r.per_tensor_outputs
                    if t.bl_stats is not None and t.tt_stats is None
                ]
                only_tt_out = [
                    t.key
                    for t in r.per_tensor_outputs
                    if t.tt_stats is not None and t.bl_stats is None
                ]
                if not (only_bl_in or only_tt_in or only_bl_out or only_tt_out):
                    continue
                print(f"      {r.fqn}:")
                for tag, keys in [
                    ("input only in baseline", only_bl_in),
                    ("input only in torchturbo", only_tt_in),
                    ("output only in baseline", only_bl_out),
                    ("output only in torchturbo", only_tt_out),
                ]:
                    for k in keys:
                        print(f"        {tag}: {k}")

        # Separate mismatched into internal-first, then inherited
        internal_results = [
            r for r in results if r.classification == "internal_divergence"
        ]
        inherited_results = [
            r for r in results if r.classification == "inherited_divergence"
        ]
        mismatched = internal_results + inherited_results

        if not mismatched:
            print("\n  ✅ All modules match!")

        # Show near-miss modules (matched but with nonzero diff) so small
        # divergences don't silently vanish from the report.
        near_miss = [
            r
            for r in results
            if r.classification == "match"
            and (r.max_output_rel_diff > 0 or r.max_input_rel_diff > 0)
        ]
        if near_miss:
            fwd = getattr(self, "_fqn_forward_order", None) or {}
            near_miss.sort(key=lambda r: fwd.get(r.fqn, float("inf")))
            show = near_miss[:10]
            print(
                f"\n  ℹ️  {len(near_miss)} module(s) matched but with nonzero diff (top {len(show)}):"
            )
            for r in show:
                print(
                    f"      {r.fqn}  in_rd={r.max_input_rel_diff:.2e}  out_rd={r.max_output_rel_diff:.2e}"
                )

        if not mismatched:
            return

        # Most specific divergent layer (earliest internal divergence in forward order = root cause)
        fqn_forward_order = getattr(self, "_fqn_forward_order", None)
        first_div = self.find_first_divergent(
            results, fqn_forward_order=fqn_forward_order
        )
        if first_div is not None:
            print(f"\n  🔍 First divergent layer (forward order): {first_div.fqn}")
            print(f"     Max input rel diff:  {first_div.max_input_rel_diff:.6e}")
            print(f"     Max output rel diff: {first_div.max_output_rel_diff:.6e}")
            if first_div.rng_cpu_match is not None:
                cpu_s = "✅" if first_div.rng_cpu_match else "❌"
                dev_s = (
                    "✅"
                    if first_div.rng_device_match
                    else ("❌" if first_div.rng_device_match is False else "—")
                )
                print(f"     RNG state:           cpu={cpu_s}  device={dev_s}")

        # Sort mismatched by forward order
        fwd = fqn_forward_order or {}
        mismatched.sort(key=lambda r: fwd.get(r.fqn, float("inf")))

        # Apply top_k limit
        if top_k > 0:
            mismatched = mismatched[:top_k]

        print(f"\n  Showing {len(mismatched)} mismatched module(s):")
        print("  " + "-" * 76)

        for r in mismatched:
            label = (
                "💥 INTERNAL"
                if r.classification == "internal_divergence"
                else "🌊 INHERITED"
            )
            root_cause_tag = ""
            if first_div is not None and r.fqn == first_div.fqn:
                root_cause_tag = "  ⚠️  LIKELY ROOT CAUSE"

            print(f"\n  [{label}] {r.fqn}{root_cause_tag}")
            print(f"    Max input  rel diff: {r.max_input_rel_diff:.6e}")
            print(f"    Max output rel diff: {r.max_output_rel_diff:.6e}")
            if r.rng_cpu_match is not None:
                cpu_s = "✅" if r.rng_cpu_match else "❌"
                dev_s = (
                    "✅"
                    if r.rng_device_match
                    else ("❌" if r.rng_device_match is False else "—")
                )
                print(f"    RNG state:  cpu={cpu_s}  device={dev_s}")

            if verbose:
                # Print per-tensor input stats
                if r.per_tensor_inputs:
                    print(f"    ── Inputs ({len(r.per_tensor_inputs)} tensor(s)) ──")
                    for t in r.per_tensor_inputs:
                        print(f"      {t.key}")
                        self._print_tensor_stats(t)
                # Print per-tensor output stats
                if r.per_tensor_outputs:
                    print(f"    ── Outputs ({len(r.per_tensor_outputs)} tensor(s)) ──")
                    for t in r.per_tensor_outputs:
                        print(f"      {t.key}")
                        self._print_tensor_stats(t)

    @staticmethod
    def _print_tensor_stats(t: TensorComparisonResult) -> None:
        """Print baseline vs torchturbo stats for a single tensor comparison."""
        bl = t.bl_stats or {}
        tt = t.tt_stats or {}

        def _val(stats, key):
            v = stats.get(key)
            if v is None:
                return "N/A"
            if isinstance(v, float):
                return f"{v:.6e}"
            return str(v)

        print(f"        {'':>10} {'Baseline':>16} {'TorchTurbo':>16}")
        print(
            f"        {'shape':>10} {str(bl.get('shape', 'N/A')):>16} {str(tt.get('shape', 'N/A')):>16}"
        )
        print(f"        {'dtype':>10} {_val(bl, 'dtype'):>16} {_val(tt, 'dtype'):>16}")
        print(f"        {'norm':>10} {_val(bl, 'norm'):>16} {_val(tt, 'norm'):>16}")
        print(f"        {'sum':>10} {_val(bl, 'sum'):>16} {_val(tt, 'sum'):>16}")
        print(f"        {'mean':>10} {_val(bl, 'mean'):>16} {_val(tt, 'mean'):>16}")
        print(
            f"        {'max_abs':>10} {_val(bl, 'max_abs'):>16} {_val(tt, 'max_abs'):>16}"
        )
        print(
            f"        norm_rel_diff: {t.norm_rel_diff:.6e}  sum_rel_diff: {t.sum_rel_diff:.6e}"
        )
        # Element-wise diff stats (only when raw tensors were dumped)
        if t.max_elem_abs_diff is not None:
            print(
                f"        max_elem_abs_diff: {t.max_elem_abs_diff:.6e}  at index: {t.max_elem_abs_diff_index}"
            )
        if t.diff_histogram is not None:
            parts = [f"{k}:{v}" for k, v in t.diff_histogram.items() if v > 0]
            print(f"        diff histogram: {', '.join(parts)}")

    def export_json(
        self,
        results: list[ModuleDivergenceResult],
        path: str,
        bl_rng: dict | None = None,
        tt_rng: dict | None = None,
    ) -> None:
        """Export comparison results as JSON for post-processing."""
        data = []
        for r in results:
            entry = {
                "fqn": r.fqn,
                "classification": r.classification,
                "max_input_rel_diff": r.max_input_rel_diff,
                "max_output_rel_diff": r.max_output_rel_diff,
                "rng_cpu_match": r.rng_cpu_match,
                "rng_device_match": r.rng_device_match,
                "per_tensor_inputs": [
                    self._tensor_result_to_dict(t) for t in r.per_tensor_inputs
                ],
                "per_tensor_outputs": [
                    self._tensor_result_to_dict(t) for t in r.per_tensor_outputs
                ],
            }
            data.append(entry)

        first_div = self.find_first_divergent(results)
        export = {
            "summary": {
                "total": len(results),
                "matched": sum(1 for r in results if r.classification == "match"),
                "internal_divergences": sum(
                    1 for r in results if r.classification == "internal_divergence"
                ),
                "inherited_divergences": sum(
                    1 for r in results if r.classification == "inherited_divergence"
                ),
                "only_in_baseline": getattr(self, "_only_in_bl", []),
                "only_in_torchturbo": getattr(self, "_only_in_tt", []),
                "first_divergent_fqn": first_div.fqn if first_div else None,
                "thresholds": {
                    "input_threshold": self.input_threshold,
                    "output_threshold": self.output_threshold,
                },
            },
            "modules": data,
        }
        # Include RNG state comparison if available
        if bl_rng and tt_rng:
            rng_data = []
            all_fqns = list(bl_rng.keys())
            for fqn in tt_rng:
                if fqn not in bl_rng:
                    all_fqns.append(fqn)
            for fqn in all_fqns:
                br = bl_rng.get(fqn, {})
                tr = tt_rng.get(fqn, {})
                dev_match = None
                if "device" in br or "device" in tr:
                    dev_match = br.get("device") == tr.get("device")
                rng_data.append(
                    {
                        "fqn": fqn,
                        "cpu_match": br.get("cpu") == tr.get("cpu"),
                        "device_match": dev_match,
                    }
                )
            export["rng_states"] = rng_data
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(path, "w") as f:
            json.dump(export, f, indent=2, default=str)
        print(f"\n  📄 Comparison report exported to {path}")

    @staticmethod
    def _tensor_result_to_dict(t: TensorComparisonResult) -> dict:

        def _safe(v):
            """Convert NaN/Inf floats to None for valid JSON."""
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v

        d = {
            "key": t.key,
            "norm_abs_diff": _safe(t.norm_abs_diff),
            "norm_rel_diff": _safe(t.norm_rel_diff),
            "sum_abs_diff": _safe(t.sum_abs_diff),
            "sum_rel_diff": _safe(t.sum_rel_diff),
            "shape_match": t.shape_match,
            "dtype_match": t.dtype_match,
        }
        if t.bl_stats:
            d["bl_stats"] = t.bl_stats
        if t.tt_stats:
            d["tt_stats"] = t.tt_stats
        if t.max_elem_abs_diff is not None:
            d["max_elem_abs_diff"] = t.max_elem_abs_diff
            d["max_elem_abs_diff_index"] = t.max_elem_abs_diff_index
        if t.diff_histogram is not None:
            d["diff_histogram"] = t.diff_histogram
        return d


# ───────────────────── Training runner ─────────────────────


def seed_all(seed=42):
    import os
    import random
    import numpy as np
    import torch
    from llamafactory.extras.misc import (
        is_torch_cuda_available,
        is_torch_npu_available,
    )

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if is_torch_cuda_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    elif is_torch_npu_available():
        import torch_npu

        os.environ["HCCL_DETERMINISTIC"] = str(True)
        os.environ["LCCL_DETERMINISTIC"] = str(1)
        os.environ["CLOSE_MATMUL_K_SHIFT"] = str(1)
        os.environ["ATB_MATMUL_SHUFFLE_K_ENABLE"] = "0"
        os.environ["ATB_LLM_LCOC_ENABLE"] = "0"
        torch.use_deterministic_algorithms(True)
        torch_npu.npu.manual_seed_all(seed)
        torch_npu.npu.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_training(args, label: str):
    """Run LlamaFactory training for 1 step with full diagnostics."""
    import torch
    from transformers import TrainerCallback

    # Determine which ranks should dump and where.
    # Each rank gets its own rank_<N>/ subdirectory.
    current_rank = get_rank()
    target_ranks = set(args.rank)
    should_dump = current_rank in target_ranks
    dump_dir_root = os.path.join(args.output_dir, f"rank_{current_rank}")

    if should_dump:
        os.makedirs(dump_dir_root, exist_ok=True)

    # Strict seeding for deterministic data loading
    print(f"[{label}] Setting seed to {args.seed}")
    seed_all(args.seed)

    tracker = DiagnosticTracker()

    class DiagnosticCallback(TrainerCallback):
        """Capture inputs, register hooks before first step, capture loss, then stop."""

        def __init__(self):
            self._hooked = False

        def on_train_begin(self, _args, state, control, model=None, **kwargs):
            # HF may pass model via kwargs rather than as a positional arg
            if model is None:
                model = kwargs.get("model")
            if model is not None:
                # Re-seed after model loading to align RNG state.
                # Baseline (from_pretrained) and torchturbo (meta device + load)
                # consume different amounts of RNG during model init, so the
                # RNG state diverges before the first forward pass.  Re-seeding
                # here ensures both paths start the forward pass with identical
                # RNG state.
                print(f"[{label}] Re-seeding to {args.seed} after model load")
                seed_all(args.seed)

                tracker.snapshot_params(model)
                print(f"[{label}] Snapshotted {len(tracker.param_stats)} parameters")
                tracker.snapshot_buffers(model)
                print(f"[{label}] Snapshotted {len(tracker.buffer_stats)} buffers")
                if getattr(args, "dump_params", False):
                    # _unwrap_tensor may call full_tensor() (collective) so
                    # all ranks must participate.  Only target ranks write.
                    if should_dump:
                        tracker.dump_params(model, dump_dir_root, label)
                    else:
                        # Participate in collective without writing
                        for _, param in model.named_parameters():
                            tracker._unwrap_tensor(param)
                print(f"[{label}] Registering forward hooks...")
                dump_dir = None
                if getattr(args, "dump_layer_io", False) and should_dump:
                    dump_dir = os.path.join(dump_dir_root, f"{label}_layer_io")
                    os.makedirs(dump_dir, exist_ok=True)
                tracker.register_hooks(
                    model,
                    max_depth=args.max_hook_depth,
                    filter_pattern=args.hook_filter,
                    dump_dir=dump_dir,
                )
                self._hooked = True

        def on_step_begin(self, _args, state, control, **kwargs):
            pass  # inputs are captured in training_step patch below

        def on_log(self, _args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs and tracker.loss is None:
                tracker.loss = logs["loss"]
                print(f"\n[{label}] Step {state.global_step} loss = {tracker.loss}")
                control.should_training_stop = True

        def on_train_end(self, *a, **kw):
            if self._hooked:
                tracker.remove_hooks()
            # Dump post-training weights for bitwise comparison
            model = kw.get("model")
            if model is not None and getattr(args, "dump_weights", False):
                if should_dump:
                    tracker.dump_params(model, dump_dir_root, f"{label}_post")
                else:
                    # Participate in collective without writing
                    for _, param in model.named_parameters():
                        tracker._unwrap_tensor(param)

    diag_cb = DiagnosticCallback()

    # Monkey-patch Trainer.training_step to capture inputs
    trainer_class = resolve_trainer_class(args)
    module_path, class_name = trainer_class.rsplit(".", 1)
    import importlib

    trainer_module = importlib.import_module(module_path)
    trainer = getattr(trainer_module, class_name)
    print(f"[{label}] Using trainer: {trainer_class}")
    orig_training_step = trainer.training_step

    _input_captured = False

    def _patched_training_step(self_trainer, model, inputs, num_items_in_batch):
        nonlocal _input_captured
        if not _input_captured:
            print(f"[{label}] Capturing input batch...")
            tracker.snapshot_inputs(inputs)
            tracker.snapshot_sampler(self_trainer)
            print(f"[{label}] Capturing precision state...")
            tracker.snapshot_precision(model, trainer=self_trainer)
            if args.dump_inputs and should_dump:
                tracker.dump_batch(inputs, dump_dir_root, label)
            _input_captured = True
        return orig_training_step(self_trainer, model, inputs, num_items_in_batch)

    trainer.training_step = _patched_training_step

    # Build LF args
    lf_args = {
        "stage": args.stage,
        "do_train": True,
        "model_name_or_path": args.model_name_or_path,
        "dataset": args.dataset,
        "template": args.template,
        "cutoff_len": args.cutoff_len,
        "output_dir": os.path.join(args.output_dir, f"{label}_output"),
        "overwrite_output_dir": True,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "max_steps": 1,
        "learning_rate": args.learning_rate,
        "logging_steps": 1,
        "save_strategy": "no",
        "bf16": args.bf16,
        "seed": args.seed,
        "flash_attn": args.flash_attn,
        "report_to": "none",
        "gradient_checkpointing": False,
        "disable_gradient_checkpointing": True,
        "disable_shuffling": True,  # ensure same data order
        "dataloader_num_workers": 0,  # eliminate multiprocessing jitter
        "val_size": 0,
        "finetuning_type": "full",
        "lr_scheduler_type": "cosine",
        "overwrite_cache": True,
        "warmup_ratio": 0.0,
        "plot_loss": True,
        "ddp_timeout": 1800000000,
    }
    # Resolve dataset_dir so LF can find dataset_info.json when running
    # from a directory other than the LlamaFactory repo root.
    dataset_dir = args.dataset_dir
    if dataset_dir is None:
        import llamafactory

        # Walk up from the package __file__ to find the repo root containing
        # data/dataset_info.json.  Works for both flat (llamafactory/) and
        # src-layout (src/llamafactory/) repos, whether installed via
        # pip install -e or added to PYTHONPATH.
        pkg_dir = os.path.dirname(llamafactory.__file__)
        for _ in range(4):
            pkg_dir = os.path.dirname(pkg_dir)
            candidate = os.path.join(pkg_dir, "data")
            if os.path.isfile(os.path.join(candidate, "dataset_info.json")):
                dataset_dir = candidate
                print(f"[{label}] Auto-detected dataset_dir: {dataset_dir}")
                break
    if dataset_dir is not None:
        lf_args["dataset_dir"] = dataset_dir

    if args.neat_packing:
        lf_args["neat_packing"] = True

    if getattr(args, "no_save_model", False):
        lf_args["plot_loss"] = False
        lf_args["should_save"] = False

    print(f"\n[{label}] LlamaFactory args:")
    for k, v in sorted(lf_args.items()):
        print(f"  {k}: {v}")
    if args.extra_lf_args:
        print(f"  [extra]: {' '.join(args.extra_lf_args)}")

    # Build a CLI-style argument list instead of a dict so that
    # LlamaFactory's _parse_args uses parse_args_into_dataclasses()
    # rather than parse_dict().  This ensures `remaining` is a string
    # list ["--key", "value"] that downstream argparse-based parsers
    # (e.g. get_my_args) can handle correctly.
    lf_argv = []
    for k, v in lf_args.items():
        lf_argv.append(f"--{k}")
        lf_argv.append(str(v))

    # Append extra LF args as raw strings so that store_true flags
    # (e.g. --my_flag without a value) pass through correctly.
    lf_argv.extend(args.extra_lf_args)

    try:
        from llamafactory.train.tuner import run_exp

        run_exp(args=lf_argv, callbacks=[diag_cb])
    finally:
        # Restore original training_step
        trainer.training_step = orig_training_step

    # Save snapshot into the rank-specific directory (or root for legacy).
    if should_dump:
        snapshot_path = os.path.join(dump_dir_root, f"{label}_snapshot.json")
        tracker.save(snapshot_path)
        print(f"\n[{label}] Snapshot saved to {snapshot_path}")
        print(f"  Parameters captured: {len(tracker.param_stats)}")
        print(f"  Buffers captured: {len(tracker.buffer_stats)}")
        print(f"  Input keys captured: {list(tracker.input_stats.keys())}")
        print(f"  Layer outputs captured: {len(tracker.layer_outputs)}")
        if tracker.sampler_stats:
            print(f"  Sampler: {tracker.sampler_stats.get('sampler_type', '?')}")
            print(f"  Dataset size: {tracker.sampler_stats.get('dataset_size', '?')}")


# ───────────────────── Comparison ─────────────────────


def _diff_scalar(a: float | None, b: float | None) -> tuple[float, float]:
    """Return (abs_diff, rel_diff). Delegates to DivergenceAnalyzer._rel_diff."""
    return DivergenceAnalyzer._rel_diff(a, b)


def _fmt(val, width=14):
    if val is None:
        return "N/A".rjust(width)
    if isinstance(val, float):
        return f"{val:.6e}".rjust(width)
    return str(val).rjust(width)


def compare_results(
    output_dir: str,
    top_k: int = 0,
    verbose: bool = False,
    export_json_path: str | None = None,
    input_threshold: float = 1e-6,
    output_threshold: float = 1e-6,
    param_threshold: float = 1e-5,
    show_matched: bool = False,
    ranks: list[int] | None = None,
):
    """Compare baseline vs torchturbo snapshots.

    Each rank's artifacts live in ``<output_dir>/rank_<N>/``.
    Comparison runs independently per rank.
    """
    if ranks is None:
        ranks = [0]
    all_ok = True
    for r in sorted(ranks):
        rank_dir = os.path.join(output_dir, f"rank_{r}")
        print(f"\n{'#' * 80}")
        print(f"  RANK {r}  ({rank_dir})")
        print(f"{'#' * 80}")
        ok = _compare_single_rank(
            rank_dir,
            top_k=top_k,
            verbose=verbose,
            export_json_path=export_json_path,
            input_threshold=input_threshold,
            output_threshold=output_threshold,
            param_threshold=param_threshold,
            show_matched=show_matched,
            rank_label=r,
        )
        if not ok:
            all_ok = False
    # Final cross-rank summary
    print(f"\n{'=' * 80}")
    if all_ok:
        print(f"  ✅ ALL RANKS PASSED — ranks {sorted(ranks)}")
    else:
        print(f"  ❌ DIFFERENCES DETECTED on one or more ranks — see above")
    print("=" * 80)


def _compare_single_rank(
    output_dir: str,
    top_k: int = 0,
    verbose: bool = False,
    export_json_path: str | None = None,
    input_threshold: float = 1e-6,
    output_threshold: float = 1e-6,
    param_threshold: float = 1e-5,
    show_matched: bool = False,
    rank_label: int | None = None,
) -> bool:
    """Compare a single rank's snapshots. Returns True if all checks pass."""
    bl_path = os.path.join(output_dir, "baseline_snapshot.json")
    tt_path = os.path.join(output_dir, "torchturbo_snapshot.json")

    for p, name in [(bl_path, "baseline"), (tt_path, "torchturbo")]:
        if not os.path.exists(p):
            print(f"ERROR: {name} snapshot not found at {p}")
            print(f"Run with --mode {name} first.")
            sys.exit(1)

    with open(bl_path) as f:
        bl = json.load(f)
    with open(tt_path) as f:
        tt = json.load(f)

    all_ok = True

    if rank_label is not None:
        print(f"\n  Comparing rank {rank_label} snapshots")
    # ── 0. Loss ──
    print("\n" + "=" * 80)
    print("  LOSS COMPARISON")
    print("=" * 80)
    bl_loss, tt_loss = bl.get("loss"), tt.get("loss")
    abs_d, rel_d = _diff_scalar(bl_loss, tt_loss)
    loss_ok = (not math.isnan(abs_d)) and rel_d <= output_threshold
    status = "✅" if loss_ok else "❌"
    if not loss_ok:
        all_ok = False
    print(f"  Baseline loss:   {_fmt(bl_loss)}")
    print(f"  TorchTurbo loss: {_fmt(tt_loss)}")
    print(f"  Abs diff:        {_fmt(abs_d)}")
    print(f"  Rel diff:        {_fmt(rel_d)}")
    print(f"  Threshold:       rel_diff <= {output_threshold:.0e}")
    print(f"  Status:          {status}")

    # ── 1. Parameters ──
    print("\n" + "=" * 80)
    print("  PARAMETER COMPARISON")
    print("=" * 80)

    bl_params = bl.get("param_stats", {})
    tt_params = tt.get("param_stats", {})
    all_param_names = sorted(set(bl_params.keys()) | set(tt_params.keys()))

    only_bl = [n for n in all_param_names if n not in tt_params]
    only_tt = [n for n in all_param_names if n not in bl_params]
    if only_bl:
        print(f"\n  ⚠️  {len(only_bl)} params only in baseline:")
        for n in only_bl[:10]:
            print(f"      {n}")
        if len(only_bl) > 10:
            print(f"      ... and {len(only_bl) - 10} more")
        all_ok = False
    if only_tt:
        print(f"\n  ⚠️  {len(only_tt)} params only in torchturbo:")
        for n in only_tt[:10]:
            print(f"      {n}")
        if len(only_tt) > 10:
            print(f"      ... and {len(only_tt) - 10} more")
        all_ok = False

    common = [n for n in all_param_names if n in bl_params and n in tt_params]
    param_diffs = []
    for name in common:
        bp, tp = bl_params[name], tt_params[name]
        norm_ad, norm_rd = _diff_scalar(bp.get("norm"), tp.get("norm"))
        sum_ad, sum_rd = _diff_scalar(bp.get("sum"), tp.get("sum"))
        shape_match = bp.get("shape") == tp.get("shape")
        dtype_match = bp.get("dtype") == tp.get("dtype")
        max_rd = max(norm_rd, sum_rd)
        param_diffs.append(
            {
                "name": name,
                "bl_shape": bp.get("shape"),
                "tt_shape": tp.get("shape"),
                "bl_dtype": bp.get("dtype"),
                "tt_dtype": tp.get("dtype"),
                "shape_match": shape_match,
                "dtype_match": dtype_match,
                "norm_abs_diff": norm_ad,
                "norm_rel_diff": norm_rd,
                "sum_abs_diff": sum_ad,
                "sum_rel_diff": sum_rd,
                "max_rel_diff": max_rd,
                "bl_norm": bp.get("norm"),
                "tt_norm": tp.get("norm"),
                "bl_sum": bp.get("sum"),
                "tt_sum": tp.get("sum"),
            }
        )

    mismatched_params = [
        d
        for d in param_diffs
        if d["max_rel_diff"] > param_threshold
        or not d["shape_match"]
        or not d["dtype_match"]
    ]
    matched_params = len(param_diffs) - len(mismatched_params)

    print(f"\n  Total common params: {len(common)}")
    print(
        f"  Matched:  {matched_params}  (threshold: rel_diff <= {param_threshold:.0e})"
    )
    print(f"  Mismatch: {len(mismatched_params)}")

    if mismatched_params:
        all_ok = False
        # Sort by diff magnitude
        mismatched_params.sort(key=lambda d: d["max_rel_diff"], reverse=True)
        show = mismatched_params[:20] if top_k == 0 else mismatched_params[:top_k]
        print(f"\n  Top mismatched parameters (showing {len(show)}):")
        print(
            f"  {'Name':<55} {'Shape?':>6} {'Type?':>6} {'NormDiff':>14} {'SumDiff':>14}"
        )
        print("  " + "-" * 95)
        for d in show:
            sm = "✅" if d["shape_match"] else "❌"
            dm = "✅" if d["dtype_match"] else "❌"
            print(
                f"  {d['name']:<55} {sm:>6} {dm:>6} {_fmt(d['norm_abs_diff'])} {_fmt(d['sum_abs_diff'])}"
            )
            if not d["shape_match"]:
                print(
                    f"    baseline shape: {d['bl_shape']}  torchturbo shape: {d['tt_shape']}"
                )
    else:
        print("  ✅ All parameters match!")

    # ── 1a-bis. Bitwise param comparison from dumped .pt files ──
    bl_param_pt = os.path.join(output_dir, "baseline_params.pt")
    tt_param_pt = os.path.join(output_dir, "torchturbo_params.pt")
    if os.path.exists(bl_param_pt) and os.path.exists(tt_param_pt):
        import torch

        print("\n  BITWISE PARAMETER COMPARISON (.pt files)")
        bl_state = torch.load(bl_param_pt, map_location="cpu", weights_only=True)
        tt_state = torch.load(tt_param_pt, map_location="cpu", weights_only=True)
        bl_names = set(bl_state.keys())
        tt_names = set(tt_state.keys())
        common_pt = sorted(bl_names & tt_names)
        only_bl_pt = sorted(bl_names - tt_names)
        only_tt_pt = sorted(tt_names - bl_names)
        if only_bl_pt:
            print(f"    ⚠️  {len(only_bl_pt)} param(s) only in baseline dump")
        if only_tt_pt:
            print(f"    ⚠️  {len(only_tt_pt)} param(s) only in torchturbo dump")
        bitwise_match = 0
        bitwise_mismatch = []
        for pname in common_pt:
            bl_t = bl_state[pname]
            tt_t = tt_state[pname]
            if bl_t.shape != tt_t.shape:
                bitwise_mismatch.append(
                    (
                        pname,
                        "shape",
                        f"bl={list(bl_t.shape)} tt={list(tt_t.shape)}",
                        None,
                    )
                )
                continue
            if bl_t.dtype != tt_t.dtype:
                bitwise_mismatch.append(
                    (pname, "dtype", f"bl={bl_t.dtype} tt={tt_t.dtype}", None)
                )
                continue
            if torch.equal(bl_t, tt_t):
                bitwise_match += 1
            else:
                diff = (bl_t.float() - tt_t.float()).abs()
                max_diff = diff.max().item()
                diff_count = int((bl_t != tt_t).sum().item())
                bitwise_mismatch.append(
                    (
                        pname,
                        "values",
                        f"{diff_count}/{bl_t.numel()} elements differ, max_abs_diff={max_diff:.6e}",
                        bl_t.dtype,
                    )
                )
        print(f"    Bitwise identical: {bitwise_match}/{len(common_pt)}")
        if bitwise_mismatch:
            all_ok = False
            print(f"    Bitwise mismatch:  {len(bitwise_mismatch)}")
            for pname, kind, detail, dtype in bitwise_mismatch[:20]:
                dtype_str = f" ({dtype})" if dtype else ""
                print(f"      ❌ {pname}{dtype_str}: {kind} — {detail}")
            if len(bitwise_mismatch) > 20:
                print(f"      ... and {len(bitwise_mismatch) - 20} more")
        else:
            print("    ✅ All parameters bitwise identical!")
        del bl_state, tt_state

    # ── 1b. Buffers ──
    print("\n" + "=" * 80)
    print("  BUFFER COMPARISON")
    print("=" * 80)

    bl_bufs = bl.get("buffer_stats", {})
    tt_bufs = tt.get("buffer_stats", {})
    all_buf_names = sorted(set(bl_bufs.keys()) | set(tt_bufs.keys()))

    only_bl_b = [n for n in all_buf_names if n not in tt_bufs]
    only_tt_b = [n for n in all_buf_names if n not in bl_bufs]
    if only_bl_b:
        print(f"\n  ⚠️  {len(only_bl_b)} buffers only in baseline")
        all_ok = False
    if only_tt_b:
        print(f"\n  ⚠️  {len(only_tt_b)} buffers only in torchturbo")
        all_ok = False

    common_bufs = [n for n in all_buf_names if n in bl_bufs and n in tt_bufs]
    buf_diffs = []
    for name in common_bufs:
        bb, tb = bl_bufs[name], tt_bufs[name]
        norm_ad, norm_rd = _diff_scalar(bb.get("norm"), tb.get("norm"))
        sum_ad, sum_rd = _diff_scalar(bb.get("sum"), tb.get("sum"))
        shape_match = bb.get("shape") == tb.get("shape")
        dtype_match = bb.get("dtype") == tb.get("dtype")
        max_rd = max(norm_rd, sum_rd)
        buf_diffs.append(
            {
                "name": name,
                "max_rel_diff": max_rd,
                "shape_match": shape_match,
                "dtype_match": dtype_match,
                "norm_abs_diff": norm_ad,
                "sum_abs_diff": sum_ad,
                "bl_shape": bb.get("shape"),
                "tt_shape": tb.get("shape"),
            }
        )

    mismatched_bufs = [
        d
        for d in buf_diffs
        if d["max_rel_diff"] > param_threshold
        or not d["shape_match"]
        or not d["dtype_match"]
    ]
    matched_bufs = len(buf_diffs) - len(mismatched_bufs)

    print(f"\n  Total common buffers: {len(common_bufs)}")
    print(f"  Matched:  {matched_bufs}  (threshold: rel_diff <= {param_threshold:.0e})")
    print(f"  Mismatch: {len(mismatched_bufs)}")

    if mismatched_bufs:
        all_ok = False
        mismatched_bufs.sort(key=lambda d: d["max_rel_diff"], reverse=True)
        show = mismatched_bufs[:20] if top_k == 0 else mismatched_bufs[:top_k]
        print(f"\n  Top mismatched buffers (showing {len(show)}):")
        print(
            f"  {'Name':<55} {'Shape?':>6} {'Type?':>6} {'NormDiff':>14} {'SumDiff':>14}"
        )
        print("  " + "-" * 95)
        for d in show:
            sm = "✅" if d["shape_match"] else "❌"
            dm = "✅" if d["dtype_match"] else "❌"
            print(
                f"  {d['name']:<55} {sm:>6} {dm:>6} {_fmt(d['norm_abs_diff'])} {_fmt(d['sum_abs_diff'])}"
            )
            if not d["shape_match"]:
                print(
                    f"    baseline shape: {d['bl_shape']}  torchturbo shape: {d['tt_shape']}"
                )
    else:
        print("  ✅ All buffers match!")

    # ── 2. Inputs ──
    print("\n" + "=" * 80)
    print("  INPUT TENSOR COMPARISON")
    print("=" * 80)

    bl_inputs = bl.get("input_stats", {})
    tt_inputs = tt.get("input_stats", {})
    input_keys = sorted(set(bl_inputs.keys()) | set(tt_inputs.keys()))

    # ── 2a. Sampler / Dataloader State ──
    bl_sampler = bl.get("sampler_stats", {})
    tt_sampler = tt.get("sampler_stats", {})
    if bl_sampler or tt_sampler:
        print("\n  SAMPLER / DATALOADER STATE")
        all_skeys = sorted(set(bl_sampler.keys()) | set(tt_sampler.keys()))
        for sk in all_skeys:
            bv = bl_sampler.get(sk)
            tv = tt_sampler.get(sk)
            match = "✅" if bv == tv else "❌"
            if bv != tv:
                all_ok = False
            print(
                f"    {sk:<25}  baseline={str(bv):<15}  torchturbo={str(tv):<15}  {match}"
            )

    # ── 2a-bis. Precision / Mixed Precision State ──
    bl_prec = bl.get("precision_stats", {})
    tt_prec = tt.get("precision_stats", {})
    if bl_prec or tt_prec:
        print("\n  PRECISION / MIXED PRECISION STATE")
        all_pkeys = sorted(set(bl_prec.keys()) | set(tt_prec.keys()))
        for pk in all_pkeys:
            bv = bl_prec.get(pk)
            tv = tt_prec.get(pk)
            match = "✅" if bv == tv else "❌"
            if bv != tv:
                all_ok = False
            print(
                f"    {pk:<30}  baseline={str(bv):<20}  torchturbo={str(tv):<20}  {match}"
            )

    # ── 2a-ter. RNG State Comparison ──
    bl_rng = bl.get("rng_states", {})
    tt_rng = tt.get("rng_states", {})
    if bl_rng or tt_rng:
        if not bl_rng:
            print(
                f"\n  ⚠️  RNG states only in torchturbo ({len(tt_rng)} modules), none in baseline"
            )
        elif not tt_rng:
            print(
                f"\n  ⚠️  RNG states only in baseline ({len(bl_rng)} modules), none in torchturbo"
            )
        else:
            # Use baseline insertion order (= hook execution order) for correct
            # "first divergence" detection, then append any torchturbo-only FQNs.
            all_rng_fqns = list(bl_rng.keys())
            for fqn in tt_rng:
                if fqn not in bl_rng:
                    all_rng_fqns.append(fqn)
            rng_mismatches = []
            first_rng_diverge = None
            for fqn in all_rng_fqns:
                br = bl_rng.get(fqn, {})
                tr = tt_rng.get(fqn, {})
                cpu_match = br.get("cpu") == tr.get("cpu")
                # Only compare device RNG when at least one side captured it
                dev_match = True
                if "device" in br or "device" in tr:
                    dev_match = br.get("device") == tr.get("device")
                if not cpu_match or not dev_match:
                    rng_mismatches.append(fqn)
                    if first_rng_diverge is None:
                        first_rng_diverge = fqn
            if rng_mismatches:
                print(
                    f"\n  ⚠️  RNG STATE DIVERGENCE ({len(rng_mismatches)}/{len(all_rng_fqns)} modules)"
                )
                print(f"  🔍 First RNG divergence at: {first_rng_diverge}")
                print(f"     This means a random op (dropout, etc.) consumed different")
                print(
                    f"     RNG values before this module between baseline and torchturbo."
                )
                for fqn in rng_mismatches[:10]:
                    br = bl_rng.get(fqn, {})
                    tr = tt_rng.get(fqn, {})
                    tags = []
                    if br.get("cpu") != tr.get("cpu"):
                        tags.append("cpu")
                    if ("device" in br or "device" in tr) and br.get(
                        "device"
                    ) != tr.get("device"):
                        tags.append("device")
                    print(f"      {fqn}  [{','.join(tags)}]")
                if len(rng_mismatches) > 10:
                    print(f"      ... and {len(rng_mismatches) - 10} more")
            else:
                print(
                    f"\n  ✅ RNG states identical across all {len(all_rng_fqns)} hooked modules"
                )

    # ── 2b. Dumped .pt file comparison ──
    # Auto-discover dump keys from files on disk instead of hardcoding
    import glob as _glob

    bl_dump_files = _glob.glob(os.path.join(output_dir, "baseline_*.pt"))
    # Exclude param dumps (handled separately above)
    dump_keys = set()
    for fp in bl_dump_files:
        fname = os.path.basename(fp)  # e.g. "baseline_input_ids.pt"
        if fname == "baseline_params.pt":
            continue
        key = fname.removeprefix("baseline_").removesuffix(".pt")
        dump_keys.add(key)
    # Also check torchturbo side for keys only present there
    tt_dump_files = _glob.glob(os.path.join(output_dir, "torchturbo_*.pt"))
    for fp in tt_dump_files:
        fname = os.path.basename(fp)
        if fname == "torchturbo_params.pt":
            continue
        key = fname.removeprefix("torchturbo_").removesuffix(".pt")
        dump_keys.add(key)
    dump_keys = sorted(dump_keys)
    found_dumps = False
    # Track keys verified by element-wise .pt comparison (ground truth)
    dump_verified: dict[str, bool] = {}
    for dk in dump_keys:
        bl_pt = os.path.join(output_dir, f"baseline_{dk}.pt")
        tt_pt = os.path.join(output_dir, f"torchturbo_{dk}.pt")
        if os.path.exists(bl_pt) and os.path.exists(tt_pt):
            import torch

            b = torch.load(bl_pt, map_location="cpu", weights_only=True)
            t = torch.load(tt_pt, map_location="cpu", weights_only=True)
            if not found_dumps:
                print("\n  DUMPED TENSOR COMPARISON (.pt files)")
                found_dumps = True
            if b.shape != t.shape:
                print(
                    f"    ❌ '{dk}': shape mismatch  baseline={list(b.shape)}  torchturbo={list(t.shape)}"
                )
                dump_verified[dk] = False
                all_ok = False
            else:
                equal = (b == t).all().item()
                if equal:
                    print(f"    ✅ '{dk}': bitwise identical  shape={list(b.shape)}")
                    dump_verified[dk] = True
                else:
                    diff_count = (b != t).sum().item()
                    total = b.numel()
                    first_diff = (b != t).nonzero()[0]
                    print(
                        f"    ❌ '{dk}': {diff_count}/{total} elements differ  shape={list(b.shape)}"
                    )
                    print(
                        f"       First diff at index {first_diff.tolist()}:  baseline={b[tuple(first_diff)].item()}  torchturbo={t[tuple(first_diff)].item()}"
                    )
                    dump_verified[dk] = False
                    all_ok = False

    for key in input_keys:
        bi = bl_inputs.get(key, {})
        ti = tt_inputs.get(key, {})

        if key not in bl_inputs:
            print(f"\n  ⚠️  '{key}' only in torchturbo inputs")
            all_ok = False
            continue
        if key not in tt_inputs:
            print(f"\n  ⚠️  '{key}' only in baseline inputs")
            all_ok = False
            continue

        # Non-tensor entries
        if "norm" not in bi:
            print(f"\n  '{key}': {bi.get('type', '?')} (non-tensor, skipped)")
            continue

        shape_match = bi.get("shape") == ti.get("shape")
        norm_ad, norm_rd = _diff_scalar(bi.get("norm"), ti.get("norm"))
        sum_ad, sum_rd = _diff_scalar(bi.get("sum"), ti.get("sum"))

        bl_hash = bi.get("content_hash")
        tt_hash = ti.get("content_hash")

        # If this key was already verified by element-wise .pt comparison,
        # use that as ground truth (it's authoritative).
        if key in dump_verified:
            status = "✅" if dump_verified[key] else "❌"
        elif bl_hash is not None and tt_hash is not None:
            # Use content hash for exact comparison when available (catches
            # differences that norm/sum miss).
            hash_match = bl_hash == tt_hash
            status = "✅" if hash_match and shape_match else "❌"
        else:
            # Legacy snapshots without hash — use dtype-aware comparison
            dtype_str = bi.get("dtype", "")
            is_integer = any(t in dtype_str for t in ("int", "long", "bool"))
            if is_integer:
                first_match = bi.get("first_16") == ti.get("first_16")
                last_match = bi.get("last_16") == ti.get("last_16")
                values_ok = (
                    norm_ad == 0
                    and sum_ad == 0
                    and first_match
                    and last_match
                    and not math.isnan(norm_ad)
                )
                status = "✅" if values_ok and shape_match else "❌"
            else:
                status = (
                    "✅"
                    if norm_rd <= input_threshold
                    and sum_rd <= input_threshold
                    and shape_match
                    else "❌"
                )
        if status == "❌":
            all_ok = False

        print(f"\n  {status} '{key}':")
        print(
            f"    Shape:  baseline={bi.get('shape')}  torchturbo={ti.get('shape')}  match={shape_match}"
        )
        print(f"    Dtype:  baseline={bi.get('dtype')}  torchturbo={ti.get('dtype')}")
        print(
            f"    Norm:   baseline={_fmt(bi.get('norm'))}  torchturbo={_fmt(ti.get('norm'))}  diff={_fmt(norm_ad)}"
        )
        print(
            f"    Sum:    baseline={_fmt(bi.get('sum'))}  torchturbo={_fmt(ti.get('sum'))}  diff={_fmt(sum_ad)}"
        )
        if bl_hash is not None and tt_hash is not None:
            hash_status = "✅" if bl_hash == tt_hash else "❌"
            print(
                f"    Hash:   {hash_status}  baseline={bl_hash[:12]}…  torchturbo={tt_hash[:12]}…"
            )

        # Spot-check first/last values
        bf16 = bi.get("first_16", [])
        tf16 = ti.get("first_16", [])
        bl16 = bi.get("last_16", [])
        tl16 = ti.get("last_16", [])
        if bf16 and tf16 and bf16 != tf16:
            print(f"    First16 baseline:   {bf16}")
            print(f"    First16 torchturbo: {tf16}")
        if bl16 and tl16 and bl16 != tl16:
            print(f"    Last16 baseline:    {bl16}")
            print(f"    Last16 torchturbo:  {tl16}")

    # ── 3. Saved weights (post-training) ──
    bl_weights_pt = os.path.join(output_dir, "baseline_post_params.pt")
    tt_weights_pt = os.path.join(output_dir, "torchturbo_post_params.pt")
    if os.path.exists(bl_weights_pt) and os.path.exists(tt_weights_pt):
        import torch

        print("\n" + "=" * 80)
        print("  SAVED WEIGHT COMPARISON (post-training)")
        print("=" * 80)
        bl_wstate = torch.load(bl_weights_pt, map_location="cpu", weights_only=True)
        tt_wstate = torch.load(tt_weights_pt, map_location="cpu", weights_only=True)
        bl_wnames = set(bl_wstate.keys())
        tt_wnames = set(tt_wstate.keys())
        common_w = sorted(bl_wnames & tt_wnames)
        only_bl_w = sorted(bl_wnames - tt_wnames)
        only_tt_w = sorted(tt_wnames - bl_wnames)
        if only_bl_w:
            print(f"    ⚠️  {len(only_bl_w)} weight(s) only in baseline")
            all_ok = False
        if only_tt_w:
            print(f"    ⚠️  {len(only_tt_w)} weight(s) only in torchturbo")
            all_ok = False
        w_match = 0
        w_mismatch = []
        for wname in common_w:
            bl_t = bl_wstate[wname]
            tt_t = tt_wstate[wname]
            if bl_t.shape != tt_t.shape:
                w_mismatch.append(
                    (
                        wname,
                        "shape",
                        f"bl={list(bl_t.shape)} tt={list(tt_t.shape)}",
                        None,
                    )
                )
                continue
            if bl_t.dtype != tt_t.dtype:
                w_mismatch.append(
                    (wname, "dtype", f"bl={bl_t.dtype} tt={tt_t.dtype}", None)
                )
                continue
            if torch.equal(bl_t, tt_t):
                w_match += 1
            else:
                diff = (bl_t.float() - tt_t.float()).abs()
                max_diff = diff.max().item()
                diff_count = int((bl_t != tt_t).sum().item())
                w_mismatch.append(
                    (
                        wname,
                        "values",
                        f"{diff_count}/{bl_t.numel()} elements differ, max_abs_diff={max_diff:.6e}",
                        bl_t.dtype,
                    )
                )
        print(f"    Bitwise identical: {w_match}/{len(common_w)}")
        if w_mismatch:
            all_ok = False
            print(f"    Bitwise mismatch:  {len(w_mismatch)}")
            for wname, kind, detail, dtype in w_mismatch[:20]:
                dtype_str = f" ({dtype})" if dtype else ""
                print(f"      ❌ {wname}{dtype_str}: {kind} — {detail}")
            if len(w_mismatch) > 20:
                print(f"      ... and {len(w_mismatch) - 20} more")
        else:
            print("    ✅ All post-training weights bitwise identical!")
        del bl_wstate, tt_wstate

    # ── 4. Layer input/output (via DivergenceAnalyzer) ──
    bl_layer_in = bl.get("layer_inputs", {})
    tt_layer_in = tt.get("layer_inputs", {})
    bl_layer_out = bl.get("layer_outputs", {})
    tt_layer_out = tt.get("layer_outputs", {})

    analyzer = DivergenceAnalyzer(
        input_threshold=input_threshold,
        output_threshold=output_threshold,
        output_dir=output_dir,
    )
    divergence_results = analyzer.analyze(
        bl_layer_in,
        tt_layer_in,
        bl_layer_out,
        tt_layer_out,
        bl_rng=bl.get("rng_states", {}),
        tt_rng=tt.get("rng_states", {}),
    )
    analyzer.print_report(
        divergence_results, top_k=top_k, verbose=verbose, show_matched=show_matched
    )

    if export_json_path:
        analyzer.export_json(
            divergence_results,
            export_json_path,
            bl_rng=bl.get("rng_states", {}),
            tt_rng=tt.get("rng_states", {}),
        )

    if any(r.classification != "match" for r in divergence_results):
        all_ok = False

    # ── Summary ──
    print("\n" + "=" * 80)
    if all_ok:
        print(
            "  ✅ ALL CHECKS PASSED — baseline and torchturbo produce identical results"
        )
    else:
        print("  ❌ DIFFERENCES DETECTED — see detailed output above")
        print("\n  Debugging tips:")
        print(
            "    • If parameters differ → check LoadModelPatch / init_kwargs forwarding"
        )
        print("    • If buffers differ → check buffer registration or FSDP sharding")
        print("    • If inputs differ → check data collation / attention mask handling")
        print(
            "    • If layer I/O shows INTERNAL divergence → the module itself introduces error"
        )
        print(
            "    • If layer I/O shows only INHERITED divergence → error propagated from upstream"
        )
        print(
            "    • If all layers differ → check model config patching (attn_impl, rope, etc.)"
        )
        print("    • Use --verbose_compare for per-tensor stats on mismatched modules")
        print("    • Use --hook_filter / --max_hook_depth to narrow the scope")
    print("=" * 80)

    return all_ok


# ───────────────────── main ─────────────────────


def main():
    args = parse_args()

    if args.mode == "compare":
        compare_results(
            args.output_dir,
            top_k=args.top_k_layers,
            verbose=args.verbose_compare,
            export_json_path=args.export_json,
            input_threshold=args.input_threshold,
            output_threshold=args.output_threshold,
            param_threshold=args.param_threshold,
            show_matched=args.show_matched,
            ranks=args.rank,
        )
        return

    if args.model_name_or_path is None:
        print("ERROR: --model_name_or_path is required for baseline/torchturbo modes")
        sys.exit(1)

    if args.mode == "baseline":
        os.environ["TORCH_TURBO_DISABLE"] = "1"
        print("\n" + "=" * 80)
        print("  BASELINE (vanilla LlamaFactory, torchturbo disabled)")
        print("=" * 80)
        run_training(args, label="baseline")

    elif args.mode == "torchturbo":
        os.environ.pop("TORCH_TURBO_DISABLE", None)
        import torchturbo  # noqa: F401

        print("\n" + "=" * 80)
        print("  TORCHTURBO (LlamaFactory + torchturbo)")
        print("=" * 80)
        run_training(args, label="torchturbo")


if __name__ == "__main__":
    main()
