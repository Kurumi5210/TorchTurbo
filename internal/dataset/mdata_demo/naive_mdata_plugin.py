"""Naive mdata: reads raw jsonl, concatenates subsets; swift native pipeline does the rest."""

from __future__ import annotations

import json
import os
from typing import Any

from torchturbo.engine.plugins import DatasetPlugin, register_plugin


@register_plugin("mdata-naive")
class NaiveMdataPlugin(DatasetPlugin):
    """Supplies raw rows (reads + concatenates mdata jsonl); the framework's native
    pipeline tokenizes and packs."""
    _DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets.json")

    def __init__(self, config: str | None = None) -> None:
        super().__init__(config)
        path = self.config or self._DEFAULT_CONFIG
        with open(path, encoding="utf-8") as handle:
            self._entries: dict[str, dict[str, Any]] = json.load(handle)

    def dataset_names(self) -> list[str]:
        return sorted(self._entries)

    def dataset_info(self, name: str) -> dict[str, Any]:
        raw = self._entries[name]
        return {
            "uri": raw["uri"],
            "columns": dict(raw.get("columns") or {}),
            "tags": dict(raw.get("tags") or {}),
            "formatting": raw.get("formatting"),
        }

    def samples(self, name: str, cache_dir: str | None = None) -> Any:
        import glob as _glob

        from datasets import concatenate_datasets, load_dataset

        local_paths = self._mount_and_get_paths(self._entries[name]["uri"])
        if not local_paths:
            raise RuntimeError(f"mdata URI resolved to no datasets: {self._entries[name]['uri']!r}")

        ordered = sorted(local_paths, key=lambda p: os.path.basename(os.path.normpath(p)))
        datasets = []
        for path in ordered:
            files = sorted(_glob.glob(os.path.join(path, "*.jsonl")))
            if not files:
                continue
            datasets.append(load_dataset("json", data_files=files, split="train", cache_dir=cache_dir))
        if not datasets:
            raise RuntimeError(f"mdata dataset {name!r} has no jsonl across all subsets")
        return datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)

    @staticmethod
    def _mount_and_get_paths(uri: str) -> list[str]:
        try:
            from mdatasets import MdataHelper
        except ImportError as exc:
            raise ImportError("mdata plugin needs the internal 'mdatasets' package.") from exc
        helper = MdataHelper(uri, local_path=None)
        helper.mount_data()
        info = helper.get_datasets()
        paths = []
        for entry in info:
            weight = float(entry["trainWeight"]) if entry.get("trainWeight") not in (None, "") else 1.0
            if weight == 0:
                continue
            path = entry.get("localPath") or entry.get("local_path") or ""
            if path:
                paths.append(path)
        return paths


__all__ = ["NaiveMdataPlugin"]