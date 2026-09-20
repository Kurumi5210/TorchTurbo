#!/usr/bin/env python3
"""Offline dataset tokenization for the ms-swift path (CPU-only, no engine, no GPU).

Thin, correctness-preserving wrapper around swift's native
``swift export --to_cached_dataset true``
(``swift/pipelines/export/cached_dataset.py``; verified identical in ms-swift 4.4.2
— the production-pinned version). It differs from the stock CLI in exactly two ways:

1. Activates the msswift adapter (``torchturbo.activate("msswift")``) FIRST, so
   ``turbo_<name>`` model types and torchturbo swift templates are registered
   before args are parsed. (Activation swaps the ``SwiftSft`` module global,
   but swift's ``ExportCachedDataset`` binds the original class via
   ``from ..train import SwiftSft`` at import time, so the export pipeline is
   unaffected.)
2. Forces ``--to_cached_dataset true`` and sanity-checks the result.

Usage — mirrors the training command, minus ``--tt_*``/trainer knobs::

    # Step 1 (once, CPU-only):
    python scripts/offline_tokenizer/swift_tokenize.py \
        --model_type turbo_qwen3_5_moe --model /path/to/model \
        --template qwen3_5 --dataset /data/my_sft.jsonl \
        --max_length 262144 \
        --output_dir /shared_fs/tokenized/my_sft_v1

    # Step 2 (training): point --cached_dataset at the saved dir
    torchrun ... scripts/runs/run_swift.py \
        --model_type turbo_qwen3_5_moe --model /path/to/model \
        --template qwen3_5 --packing true --packing_length 262144 \
        --cached_dataset /shared_fs/tokenized/my_sft_v1/train ...

What is saved (swift semantics): the *raw* rows plus a precomputed
``lengths`` column (``AddLengthPreprocessor``). Actual ``template.encode``
runs lazily at fetch time in both online and cached paths
(``LazyLLMDataset``), so cached training consumes byte-identical encodings
WHILE SKIPPING the full-dataset startup pass that causes tokenizer timeouts.

Precision contract: the cached artifact is only valid for the SAME template /
max_length / truncation_strategy / loss_scale args used at export. This script
writes ``<output_dir>/torchturbo_tokenize_args.json`` capturing them;
``verify_tokenized.py swift`` re-encodes samples from both paths and compares.
(``--packing`` is rejected by swift's export on purpose: packing is applied at
TRAIN time over the cached rows, so it is not part of the artifact contract.)

Known upstream limitation: for MULTIMODAL datasets the cached_dataset flow
re-encodes from raw rows and can drop media fields (ms-swift issue #6598).
Text SFT/CPT is unaffected. Do not use this for MLLM data until that is fixed.
"""

from __future__ import annotations

import json
import os
import sys

from _common import SWIFT_CONTRACT_ARGS, SWIFT_CONTRACT_FILENAME, ensure_repo_src

ensure_repo_src()

# Backward-compatible alias (older verify tooling imports this name).
CONTRACT_ARGS = SWIFT_CONTRACT_ARGS


def _patch_export_no_weights() -> None:
    """Force ``load_model=False`` on the export pipeline.

    Offline tokenization needs only the tokenizer/processor, never weights.
    But ``ExportCachedDataset.__init__`` calls ``_prepare_model_tokenizer()``
    with no kwargs (``load_model`` defaults True), and for ``turbo_*`` model
    types the loader is torchturbo's ``TurboModelMixin.get_model``, which
    requires a live engine. Forcing ``load_model=False`` makes
    ``get_model_processor`` return ``(None, processor)`` without touching the
    engine (same path ``verify_tokenized.py`` uses).
    """
    from swift.pipelines.export.cached_dataset import ExportCachedDataset
    from swift.pipelines.train.sft import SwiftSft

    orig_prepare = SwiftSft._prepare_model_tokenizer

    def _prepare_model_tokenizer_no_weights(self, **kwargs):
        kwargs.setdefault("load_model", False)
        return orig_prepare(self, **kwargs)

    ExportCachedDataset._prepare_model_tokenizer = _prepare_model_tokenizer_no_weights


def main() -> None:
    argv = sys.argv[1:]

    from torchturbo.engine.args import parse_torchturbo_args

    tt_args, argv = parse_torchturbo_args(argv)

    if "--output_dir" not in argv and not any(a.startswith("--output_dir=") for a in argv):
        raise SystemExit("swift_tokenize.py requires --output_dir <dir> (artifact root).")

    # Force the export mode.
    if "--to_cached_dataset" not in argv:
        argv += ["--to_cached_dataset", "true"]

    # Register turbo models/templates before swift parses args.
    import torchturbo

    torchturbo.activate("msswift")

    # No engine here: build a standalone PluginManager and consume --tt_plugin into the now-active msswift adapter.
    if tt_args.plugins:
        from torchturbo.adapters.msswift.plugins import install_plugins
        from torchturbo.engine.plugins import PluginManager

        install_plugins(PluginManager(list(tt_args.plugins)))

    _patch_export_no_weights()

    # Parse ONCE here through swift's parser (constructing the dataclass with a
    # raw list would assign it to the first field instead of parsing it).
    # (Re-running with an existing output_dir raises FileExistsError unless
    # --exist_ok true is passed.)
    from swift.arguments import ExportArguments
    from swift.pipelines.export.cached_dataset import export_cached_dataset
    from swift.utils import parse_args

    args, _remaining = parse_args(ExportArguments, argv)

    # Persist the tokenization contract next to the artifact BEFORE export.
    contract = {name: getattr(args, name, None) for name in SWIFT_CONTRACT_ARGS}
    contract_path = os.path.join(args.output_dir, SWIFT_CONTRACT_FILENAME)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(contract_path, "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2, ensure_ascii=False, default=str)

    export_cached_dataset(args)

    print(f"[swift_tokenize] contract written to {contract_path}")
    print(f"[swift_tokenize] train with: --cached_dataset {os.path.join(args.output_dir, 'train')}")


if __name__ == "__main__":
    main()
