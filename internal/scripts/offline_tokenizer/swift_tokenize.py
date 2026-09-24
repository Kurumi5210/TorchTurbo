#!/usr/bin/env python3
"""Offline dataset tokenization for the ms-swift path (CPU-only, no engine).

When a plugin (e.g. mdata) provides ``--dataset`` as a complete dataset, drives
the plugin's ``build_dataset`` directly and writes the prepared cache to
``<output_dir>/train/`` (single-process). Otherwise falls back to swift's native
``export_cached_dataset`` (raw rows + lengths). In both cases the result is
read at train time via ``--cached_dataset <output_dir>/train``.

Activates the msswift adapter first so turbo model templates are registered
before args are parsed. ``ExportCachedDataset._prepare_model_tokenizer`` is
patched to ``load_model=False`` (offline tokenization needs no weights).
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
    """Patch ``ExportCachedDataset`` to skip model loading (offline needs only the tokenizer)."""
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

    # plugin drives its own build_dataset; otherwise native swift export.
    from torchturbo.adapters.msswift.plugins import SwiftPluginHost
    from torchturbo.engine.plugins import PluginManager

    host = SwiftPluginHost()
    if tt_args.plugins:
        host.install_plugins(PluginManager(list(tt_args.plugins)))

    _patch_export_no_weights()

    from swift.arguments import ExportArguments
    from swift.pipelines.export.cached_dataset import export_cached_dataset
    from swift.utils import parse_args

    args, _remaining = parse_args(ExportArguments, argv)

    contract = {name: getattr(args, name, None) for name in SWIFT_CONTRACT_ARGS}
    contract_path = os.path.join(args.output_dir, SWIFT_CONTRACT_FILENAME)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(contract_path, "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2, ensure_ascii=False, default=str)

    # plugin owns --dataset -> drive build_dataset directly into the export cache
    route = host.dataset_bridge.resolve_dataset(args)
    if route is not None:
        name, plugin = route
        from swift.pipelines.export.cached_dataset import ExportCachedDataset

        exp = ExportCachedDataset(args)
        cached_dir = os.path.join(args.output_dir, "train")
        ctx = host.dataset_bridge.make_ctx(
            plugin.dataset_info(name), exp.args, exp.template, cache_path=cached_dir)
        plugin.build_dataset(name, ctx)
        print(f"[swift_tokenize] activate plugin '{plugin.plugin_name}': cache at {cached_dir}")
        print(f"[swift_tokenize] contract written to {contract_path}")
        print(f"[swift_tokenize] train with: --cached_dataset {cached_dir}")
        return

    # native swift offline tokenize
    export_cached_dataset(args)
    print(f"[swift_tokenize] contract written to {contract_path}")
    print(f"[swift_tokenize] train with: --cached_dataset {os.path.join(args.output_dir, 'train')}")


if __name__ == "__main__":
    main()
