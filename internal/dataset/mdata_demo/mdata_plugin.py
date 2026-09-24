"""mdata packed-dataset plugin (internal): antllm-aligned end-to-end packing plan.

Per sub-dataset: read jsonl, drive the framework's NATIVE ``MessagesPreprocessor`` +
``template.encode``, pack records with an inlined order-preserving greedy plan (antllm
``find_pack_groups`` Next-Fit + ``max_pack_sample_num`` cap + oversize drop), apply the
per-subset ``trainWeight`` ratio at the pack-row level (antllm ``SFTSampledDataset``), and
concatenate subsets in a stable name order. Returns a :class:`PackedIndexDataset` that is a
drop-in for the framework's native ``PackingDataset`` -- ``__getitem__`` returns a
``list[encoded dict]`` consumed by ``template.packing_row``/collator/``inputs_bridging``.

Per-subset ``sampling_method``/``sampling_seed`` come from the platform's mdata data info
(per shard, via ``MdataHelper.get_datasets()``).

Helpers are static/class methods on :class:`MdataPlugin`.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from torchturbo.engine.plugins import DatasetPlugin, DatasetPrepareContext, register_plugin

logger = logging.getLogger("mdata")

SeqItem = tuple[int, int]


@dataclass(frozen=True)
class MdataSpec:
    """Per-subset local paths, weights, sampling method/seed."""

    local_paths: tuple[str, ...]
    weights: tuple[float, ...]
    sampling_methods: tuple[str, ...]
    sampling_seeds: tuple[int, ...]


@register_plugin("mdata")
class MdataPlugin(DatasetPlugin):
    """Custom packed mdata dataset: :class:`PackedIndexDataset` is a drop-in for the
    framework's native ``PackingDataset``."""

    _DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets.json")
    _DEFAULT_MAX_PACK_SAMPLE_NUM = 300
    _CACHE_KIND = "mdata-prepack-cache-v1"

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

    def provides_dataset(self, name: str) -> bool:
        return True

    def build_dataset(self, name: str, ctx: DatasetPrepareContext) -> tuple:
        if ctx.cache_path:
            cache_dir = os.path.join(ctx.cache_path, f"{name}.{self._CACHE_KIND}")
            shared = True
        else:
            default_root = os.environ.get(
                "HF_DATASETS_CACHE",
                os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets"),
            )
            cache_dir = os.path.join(default_root, "mdata_prepack", f"{name}.{self._CACHE_KIND}")
            shared = False

        cached = self._read_cache(cache_dir)
        if cached is None:
            cached = self._build_and_cache(cache_dir, name, ctx, shared)
        else:
            logger.info("mdata dataset %r: using prepared-subsets cache at %r", name, cache_dir)
        return self._finalize(PackedIndexDataset(cached), None, ctx)

    @staticmethod
    def _finalize(train_dataset, val_dataset, ctx: DatasetPrepareContext):
        """Log per-pack token-length stats via the framework-injected ``ctx.show_dataset``."""
        if ctx.show_dataset is not None:
            try:
                ctx.show_dataset(train_dataset, val_dataset)
            except Exception:
                logger.debug("mdata: ctx.show_dataset logging failed (non-fatal)", exc_info=True)
        return train_dataset, val_dataset

    def _build_subsets(self, name: str, ctx: DatasetPrepareContext) -> list:
        entry = self._entries[name]
        spec = self._mount_and_parse(entry["uri"])
        max_length = int(ctx.max_length)
        max_pack = self._env_max_pack()
        ordered = sorted(
            zip(spec.local_paths, spec.weights, spec.sampling_methods, spec.sampling_seeds),
            key=lambda pw: self._dsname(pw[0]),
        )
        subsets: list[tuple[list[dict], list[list]]] = []
        for local_path, weight, method, seed in ordered:
            raw = self._load_jsonl(local_path)
            normed = ctx.preprocess(raw)
            cache = self._encode_and_cache(normed, ctx.encode_fn, ctx.num_proc)
            packs = self._pack_subset(cache, max_length, max_pack)
            ratio_packs = self._resample_packs(packs, weight, method, seed)
            subsets.append((cache, ratio_packs))
        if not subsets:
            raise RuntimeError(f"mdata dataset {name!r} resolved to no sub-datasets")
        return subsets

    def _build_and_cache(self, cache_dir: str, name: str, ctx: DatasetPrepareContext, shared: bool):
        """Build + write the cache, then read it back. ``shared`` True -> global rank0;
        False -> each machine's local_rank==0. Single-process paths just build+write+return."""
        import torch

        dist_ok = torch.distributed.is_available() and torch.distributed.is_initialized()
        if not dist_ok:
            subsets = self._build_subsets(name, ctx)
            self._export_cache(cache_dir, subsets, ctx.num_proc)
            logger.info("mdata dataset %r: wrote prepared-subsets cache to %r", name, cache_dir)
            return subsets

        if shared:
            i_build = torch.distributed.get_rank() == 0
            who = "rank0"
        else:
            i_build = int(os.environ.get("LOCAL_RANK", "0")) == 0
            who = "local_rank0"

        if i_build:
            subsets = self._build_subsets(name, ctx)
            self._export_cache(cache_dir, subsets, ctx.num_proc)
            logger.info("mdata dataset %r: %s wrote prepared-subsets cache to %r", name, who, cache_dir)
        torch.distributed.barrier()
        return self._read_cache(cache_dir)

    def _pack_subset(self, cache, max_length: int, max_pack: int) -> list[list]:
        if hasattr(cache, "column_names") and "lengths" in cache.column_names:
            lengths = [MdataPlugin._flatten_len(l) for l in cache["lengths"]]
        else:
            lengths = [self._flatten_len(c.get("lengths") or [len(c["input_ids"])]) for c in cache]
        sequences = list(enumerate(lengths))
        packs, _ = self._pack_groups(sequences, max_length, max_pack)
        return [list(g) for g in packs]

    @classmethod
    def _env_max_pack(cls) -> int:
        env = os.environ.get("MAX_PACK_SAMPLE_NUM", "").strip()
        return int(env) if env else cls._DEFAULT_MAX_PACK_SAMPLE_NUM

    @staticmethod
    def _pack_groups(
        sequences: Iterable[SeqItem],
        max_length: int,
        max_pack_sample_num: int | None,
        *,
        is_finished: bool = True,
    ) -> tuple[list[list[SeqItem]], list[SeqItem]]:
        """Order-preserving greedy first-fit (antllm ``find_pack_groups`` Next-Fit). Oversize
        records (>max_length) are dropped (flush first); a pack flushes when the next record
        would overflow or when ``max_pack_sample_num`` is reached."""
        cap = max_pack_sample_num
        packs: list[list[SeqItem]] = []
        cur: list[SeqItem] = []
        cur_len = 0
        cur_count = 0
        skipped = 0

        def _flush() -> None:
            nonlocal cur, cur_len, cur_count
            if cur:
                packs.append(cur)
            cur, cur_len, cur_count = [], 0, 0

        for idx, length in sequences:
            if length > max_length:
                skipped += 1
                _flush()
                continue
            if cur and (cur_len + length > max_length or (cap is not None and cur_count >= cap)):
                _flush()
            cur.append((idx, length))
            cur_len += length
            cur_count += 1

        if skipped:
            logger.info(
                "mdata: dropped %d oversize sequences (>max_length=%d) while packing",
                skipped, max_length)
        if is_finished:
            _flush()
            return packs, []
        return packs, cur

    @staticmethod
    def _norm_method(method: str | None) -> str:
        method = (method or "head").lower()
        return method if method in ("head", "tail", "random") else "head"

    @staticmethod
    def _dsname(local_path: str) -> str:
        return os.path.basename(os.path.normpath(local_path))

    @staticmethod
    def _flatten_len(lengths: Any) -> int:
        if isinstance(lengths, (list, tuple)):
            return int(sum(int(x) for x in lengths))
        return int(lengths)

    @staticmethod
    def _create_index_sampler(size: int, ratio: float, method: str, seed: int) -> list[int]:
        """antllm ``SFTSampledDataset.create_index_sampler`` at the pack-row level. Returns
        ``floor(size*ratio)`` indices: head/tail repeat arange, random repeats randperm + fractional identity."""
        if size <= 0:
            return []
        total = math.floor(size * ratio)
        if total <= 0:
            return []

        def _arange() -> list[int]:
            return list(range(size))

        def _reverse() -> list[int]:
            return list(range(size))[::-1]

        if method == "tail":
            full, frac_pick = _reverse, lambda n: _reverse()[:n]
        elif method == "random":
            import torch

            integer_gen = torch.Generator().manual_seed(seed)

            def full() -> list[int]:
                return torch.randperm(size, generator=integer_gen).tolist()

            def frac_pick(n: int) -> list[int]:
                return torch.randint(0, size, (n,), generator=torch.Generator().manual_seed(seed)).tolist()
        else:
            full, frac_pick = _arange, lambda n: _arange()[:n]

        integer = math.floor(ratio)
        frac = total - integer * size
        parts: list[int] = []
        for _ in range(integer):
            parts += full()
        if frac > 0:
            parts += frac_pick(frac)
        return parts[:total]

    @staticmethod
    def _resample_packs(packs: list[list], weight: float, method: str, seed: int) -> list[list]:
        idx = MdataPlugin._create_index_sampler(len(packs), float(weight), MdataPlugin._norm_method(method), int(seed))
        return [packs[i] for i in idx]

    @staticmethod
    def _load_jsonl(local_path: str):
        """Load jsonl rows in antllm prepack-v2 order: sorted filenames, each file in line order."""
        import glob as _glob

        from datasets import load_dataset

        files = sorted(_glob.glob(os.path.join(local_path, "*.jsonl")))
        if not files:
            raise RuntimeError(f"mdata sub-dataset dir has no jsonl: {local_path!r}")
        return load_dataset("json", data_files=files, split="train")

    @staticmethod
    def _encode_and_cache(normed, encode_fn: Callable[[Any], dict], num_proc: int = 1):
        """Drive ``template.encode`` per record via HF ``datasets.map(batched=True, num_proc=N)``.
        Returns the encoded HF ``Dataset`` (Arrow-backed). Per-row errors are skipped."""
        def _batch_encode(batch):
            cols = list(batch.keys())
            n = len(batch[cols[0]]) if cols else 0
            out_columns: dict[str, list] = {}
            for i in range(n):
                row = {k: batch[k][i] for k in cols}
                try:
                    enc = encode_fn(row)
                except Exception:
                    continue
                if enc is None:
                    continue
                for k, v in enc.items():
                    out_columns.setdefault(k, []).append(v)
            return out_columns

        return normed.map(
            _batch_encode,
            batched=True,
            batch_size=1000,
            num_proc=num_proc,
            remove_columns=list(normed.column_names),
            desc=f"mdata encode (num_proc={num_proc}, batch=1000)",
        )

    @staticmethod
    def _read_cache(cache_dir: str):
        """Load the prepared-subsets cache via HF ``load_from_disk`` (Arrow mmap-backed).
        Returns ``(encoded_ds, ratio_packs)`` per subset, or ``None`` if missing/unrecognized."""
        import glob

        if not os.path.isdir(cache_dir):
            return None
        import torch
        from datasets import load_from_disk

        meta_files = sorted(
            glob.glob(os.path.join(cache_dir, "subset_*_meta.pt")),
            key=lambda f: int(os.path.basename(f).split("_")[1]),
        )
        if not meta_files:
            return None
        subsets: list = []
        for meta_path in meta_files:
            meta = torch.load(meta_path, weights_only=False)
            if not (isinstance(meta, dict) and meta.get("kind") == "mdata-prepack-cache-v1"):
                logger.warning("mdata cache meta %r: unrecognized; rebuilding", meta_path)
                return None
            subset_id = meta["subset_id"]
            ratio_packs = meta["ratio_packs"]
            encoded_ds = load_from_disk(os.path.join(cache_dir, f"subset_{subset_id:05d}_data"))
            subsets.append((encoded_ds, ratio_packs))
        return subsets

    @staticmethod
    def _export_cache(cache_dir: str, subsets, num_proc: int = 1) -> None:
        """Per subset: ``save_to_disk`` (HF Arrow auto-sharded, matching swift/LF) + a small
        ``ratio_packs`` ``.pt`` meta file. ``num_proc > 1`` parallelises the Arrow write."""
        import torch

        os.makedirs(cache_dir, exist_ok=True)
        kind = "mdata-prepack-cache-v1"
        for i, (encoded_ds, ratio_packs) in enumerate(subsets):
            subset_dir = os.path.join(cache_dir, f"subset_{i:05d}_data")
            encoded_ds.save_to_disk(subset_dir, num_proc=num_proc)
            torch.save(
                {"kind": kind, "subset_id": i, "ratio_packs": ratio_packs, "n_records": len(encoded_ds)},
                os.path.join(cache_dir, f"subset_{i:05d}_meta.pt"),
            )
            logger.info("mdata subset %d: saved %d records to %s", i, len(encoded_ds), subset_dir)

    @classmethod
    def _mount_and_parse(cls, uri: str) -> MdataSpec:
        _, datasets_info = cls._mount(uri)
        if not datasets_info:
            raise RuntimeError(f"mdata URI resolved to no datasets: {uri!r}")
        return cls._parse(datasets_info)

    @staticmethod
    def _mount(uri: str, local_path: str | None = None) -> tuple[str, list[dict]]:
        try:
            from mdatasets import MdataHelper
        except ImportError as exc:
            raise ImportError("mdata plugin needs the internal 'mdatasets' package.") from exc
        helper = MdataHelper(uri, local_path=local_path)
        return helper.mount_data(), helper.get_datasets()

    @staticmethod
    def _parse(datasets_info: list[dict]) -> MdataSpec:
        """Build a spec from ``MdataHelper.get_datasets()``. Per-shard sampling comes from
        each shard's data info (not datasets.json); defaults head / 42."""
        local_paths: list[str] = []
        weights: list[float] = []
        methods: list[str] = []
        seeds: list[int] = []
        for info in datasets_info:
            path = info.get("localPath") or info.get("local_path") or ""
            weight = float(info["trainWeight"]) if info.get("trainWeight") else 1.0
            local_paths.append(path)
            weights.append(weight)
            methods.append(MdataPlugin._norm_method(info.get("samplingMethod")))
            seeds.append(int(info.get("samplingSeed") or 42))
        if not local_paths:
            raise RuntimeError("mdata URI resolved to no datasets")
        return MdataSpec(
            local_paths=tuple(local_paths),
            weights=tuple(weights),
            sampling_methods=tuple(methods),
            sampling_seeds=tuple(seeds),
        )


class PackedIndexDataset:
    """Drop-in replacement for the framework's native ``PackingDataset``.

    ``__getitem__(pack_id)`` returns the ``list[encoded dict]`` for that pack, consumed by
    ``template.packing_row``/collator/``inputs_bridging`` downstream.
    """

    def __init__(self, subsets: Sequence[tuple[list[dict], list[list]]]) -> None:
        self._caches: list[list[dict]] = [s[0] for s in subsets]
        self._packs: list[tuple[int, list[int]]] = []
        # Per-pack total token length, parallel to ``_packs``; mirrors swift
        # ``PackingDataset.packed_length`` for ``_stat_dataset`` logging.
        self.packed_length: list[int] = []
        for j, (_, ratio_packs) in enumerate(subsets):
            for grp in ratio_packs:
                self._packs.append((j, [idx for idx, _len in grp]))
                self.packed_length.append(int(sum(_len for _idx, _len in grp) if grp else 0))

    def __len__(self) -> int:
        return len(self._packs)

    def __getitem__(self, pack_id: int) -> list[dict]:
        j, idxs = self._packs[pack_id]
        cache = self._caches[j]
        return [cache[i] for i in idxs]


__all__ = ["MdataPlugin", "MdataSpec", "PackedIndexDataset"]