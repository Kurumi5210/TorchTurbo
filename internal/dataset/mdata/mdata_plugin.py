"""mdata dataset plugin (internal): one framework-agnostic class owning datasets.json
config and the mount/blend/sample pipeline. Exposes ``dataset_info`` (metadata) and
``samples`` (the data) per the public DatasetPlugin API."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Iterable

from torchturbo.engine.plugins import DatasetPlugin, PluginKind, register_plugin

logger = logging.getLogger("mdata")


@dataclass(frozen=True)
class MdataSpec:
    """Parsed mdata spec: active dataset paths, blend weights, sampling method/seed."""

    local_paths: tuple[str, ...]
    weights: tuple[float, ...]
    sampling_method: str
    seed: int


@register_plugin("mdata", PluginKind.DATASET)
class MdataPlugin(DatasetPlugin):
    """Loads ``datasets.json`` (``self.config`` or colocated default), mounts URIs,
    blends by ``trainWeight``, samples per ``samplingMethod``/seed."""

    _DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets.json")

    def __init__(self, config: str | None = None) -> None:
        super().__init__(config)
        path = self.config or self._DEFAULT_CONFIG
        with open(path, encoding="utf-8") as handle:
            self._entries: dict[str, dict[str, Any]] = json.load(handle)

    # --- metadata: this mdata provider's schema (consumers read these explicit keys) ---
    def dataset_info(self) -> dict[str, dict[str, Any]]:
        """mdata schema per dataset: ``uri``, ``columns``, ``tags``, ``formatting``
        (mirrors ``datasets.json``). Consumers (swift/LF) read these keys explicitly."""
        # Return the parsed datasets.json verbatim; the on-disk file is the schema source.
        return {
            name: {
                "uri": raw["uri"],
                "columns": dict(raw.get("columns") or {}),
                "tags": dict(raw.get("tags") or {}),
                "formatting": raw.get("formatting"),
                "samplingMethod": raw.get("samplingMethod"),
            }
            for name, raw in self._entries.items()
        }

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def names(self) -> list[str]:
        return list(self._entries)

    # --- data: the actual sample stream for a dataset name ---
    def samples(self, name: str, cache_dir: str | None = None) -> Iterable:
        entry = self._entries[name]
        uri = entry["uri"]
        # datasets.json entry wins: samplingMethod + samplingSeed override per-shard values.
        # Defaults when the entry omits them: "random" (our own model) + a fresh random seed.
        sampling_method = entry.get("samplingMethod") or "random"
        sampling_seed = entry.get("samplingSeed")
        if sampling_seed is None:
            import secrets
            sampling_seed = secrets.randbelow(2**31)
        spec = self._mount_and_parse(uri, sampling_method=sampling_method, sampling_seed=sampling_seed)
        return self._load_shards(spec, cache_dir)

    @classmethod
    def _mount_and_parse(cls, uri: str, *, sampling_method: str, sampling_seed: int) -> MdataSpec:
        _, datasets_info = cls._mount(uri)
        if not datasets_info:
            raise RuntimeError(f"mdata URI resolved to no datasets: {uri!r}")
        return cls._parse(datasets_info, sampling_method=sampling_method, sampling_seed=sampling_seed)

    @staticmethod
    def _mount(uri: str, local_path: str | None = None) -> tuple[str, list[dict]]:
        try:
            from mdatasets import MdataHelper
        except ImportError as exc:
            raise ImportError("mdata plugin needs the internal 'mdatasets' package.") from exc
        helper = MdataHelper(uri, local_path=local_path)
        return helper.mount_data(), helper.get_datasets()

    @staticmethod
    def _parse(
        datasets_info: list[dict],
        *,
        sampling_method: str,
        sampling_seed: int,
    ) -> MdataSpec:
        """Build a spec from ``MdataHelper.get_datasets()`` output. ``sampling_method``
        and ``sampling_seed`` are dataset-wide (resolved by :meth:`samples` from the
        datasets.json entry, defaulting to ``"random"`` and a fresh random seed)."""
        active: list[tuple[str, float]] = []
        for info in datasets_info:
            path = info.get("localPath") or info.get("local_path") or ""
            weight = float(info["trainWeight"]) if info.get("trainWeight") not in (None, "") else 1.0
            if weight == 0:
                continue
            active.append((path, weight))

        if not active:
            raise RuntimeError("mdata URI resolved to no datasets with non-zero trainWeight")

        weights = [w for _, w in active]
        if len(set(weights)) > 1:
            logger.info(
                "mdata datasets have different trainWeight: %s; blending with normalized weights",
                weights,
            )

        # Method is dataset-wide (datasets.json entry), not per-shard.
        method = sampling_method.lower()

        return MdataSpec(
            local_paths=tuple(p for p, _ in active),
            weights=tuple(w for _, w in active),
            sampling_method=method,
            seed=sampling_seed,
        )

    @staticmethod
    def _load_shards(spec: MdataSpec, cache_dir: str | None = None):
        """Load jsonl shards and dispatch to the per-method ``_assemble_*`` builder."""
        from datasets import load_dataset

        raw = [
            load_dataset("json", data_files=os.path.join(p, "*.jsonl"), split="train", cache_dir=cache_dir)
            for p in spec.local_paths
        ]

        assemblers = {
            "antllm_random": MdataPlugin._assemble_antllm_random,
            "random": MdataPlugin._assemble_random,
            "head": MdataPlugin._assemble_head,
            "tail": MdataPlugin._assemble_tail,
        }
        assemble = assemblers.get(spec.sampling_method)
        if assemble is None:
            raise ValueError(
                f"Unknown samplingMethod {spec.sampling_method!r}; expected one of {sorted(assemblers)}"
            )
        return assemble(raw, spec.weights, spec.seed)

    @staticmethod
    def _assemble_antllm_random(raw: list, weights: tuple[float, ...], seed: int):
        """Antllm model: per-shard index sampler with ``ratio=weight`` (random pick via
        ``torch.Generator().manual_seed(seed)``), end-to-end concatenation (no re-blending)."""
        import math

        import torch
        from datasets import concatenate_datasets

        def _index_sampler(data_size: int, ratio: float) -> list[int]:
            total = math.floor(data_size * ratio)
            integer_repeats = math.floor(ratio)
            generator = torch.Generator().manual_seed(seed)
            parts: list[torch.Tensor] = []
            if integer_repeats > 0:
                parts.append(
                    torch.cat([torch.randperm(data_size, generator=generator) for _ in range(integer_repeats)])
                )
            fractional = total - sum(int(t.numel()) for t in parts)
            if fractional > 0:
                parts.append(torch.randint(0, data_size, (fractional,), generator=generator))
            return torch.cat(parts).tolist() if parts else []

        sampled = [
            ds.select(_index_sampler(len(ds), float(w)))
            for ds, w in zip(raw, weights)
            if len(ds) > 0 and float(w) > 0
        ]
        if not sampled:
            raise RuntimeError("antllm_random: every shard sampled to empty (zero weight or empty shard)")
        return sampled[0] if len(sampled) == 1 else concatenate_datasets(sampled)

    @staticmethod
    def _assemble_random(raw: list, weights: tuple[float, ...], seed: int):
        """Our model: interleave shards by ``weight`` as mix probability, then globally shuffle.

        ``interleave_datasets`` and ``shuffle`` draw independent sub-seeds from one
        ``random.Random(seed)`` stream (no seed reuse), still reproducible from the root seed."""
        import random as stdrandom

        from datasets import interleave_datasets

        rng = stdrandom.Random(seed)
        interleave_seed = rng.randrange(2**31)
        shuffle_seed = rng.randrange(2**31)

        if len(raw) == 1:
            blended = raw[0]
        else:
            total = float(sum(weights)) or 1.0
            blended = interleave_datasets(
                raw,
                probabilities=[w / total for w in weights],
                seed=interleave_seed,
                stopping_strategy="all_exhausted",
            )
        return blended.shuffle(seed=shuffle_seed)

    @staticmethod
    def _assemble_head(raw: list, weights: tuple[float, ...], seed: int):
        """Our model: interleave shards (kept in natural order) by ``weight`` as mix probability."""
        if len(raw) == 1:
            return raw[0]
        from datasets import interleave_datasets

        total = float(sum(weights)) or 1.0
        return interleave_datasets(
            raw,
            probabilities=[w / total for w in weights],
            seed=seed,
            stopping_strategy="all_exhausted",
        )

    @staticmethod
    def _assemble_tail(raw: list, weights: tuple[float, ...], seed: int):
        """Our model: reverse each shard, then interleave by ``weight`` as mix probability."""
        reversed_shards = [ds.select(range(len(ds) - 1, -1, -1)) for ds in raw]
        if len(reversed_shards) == 1:
            return reversed_shards[0]
        from datasets import interleave_datasets

        total = float(sum(weights)) or 1.0
        return interleave_datasets(
            reversed_shards,
            probabilities=[w / total for w in weights],
            seed=seed,
            stopping_strategy="all_exhausted",
        )


__all__ = ["MdataPlugin", "MdataSpec"]