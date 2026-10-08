"""antllm pre-packed SFT reader, called by the pack plugin after the pack step.

The reader half of the antllm pre-pack flow. The pack plugin
(``antllm_pack_plugin.py``, loaded via ``--tt_plugin``) runs antllm's
``SFTPrePackProcesserV2`` into ``save_path`` (if not already packed), then imports
:func:`AntllmPrePackedDataset` and :func:`collate_prepacked` from this module, builds
the dataset over the packed shards, and serves ``collate_prepacked`` via its
``DatasetPlugin.get_collate_fn`` so the trainer hook uses it (one packed row per step).
This module is therefore a library imported by the pack plugin, not a ``--tt_plugin``
entry point of its own; the single pack-plugin run (pack-if-needed -> read -> train)
is the whole flow.

This module memmaps antllm ``sft_pre_pack`` shards directly (no antllm import at read
time), expands each packed dataset directory through antllm's ratio sampler,
concatenates the directories into one global index space, and rewrites each packed
row into the engine's packed-input contract (single dict + ``cu_seq_lens_q/k``).

Labels are emitted HF-aligned (``labels.shape == input_ids.shape``): the FSDP2
backend applies the left-shift itself once, globally, in
``FSDP2Backend._prepare_loss_labels`` and does not mask per ``cu_seq_lens`` boundaries,
so the reader undoes antllm's Megatron pre-shift and sets the first position of
every sub-sequence to -100 to mask the single cross-boundary pair the global shift
creates.
"""
from __future__ import annotations

import bisect
import functools
import json
import math
import pickle
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from torchturbo.engine.utils import assert_packed_inputs
from torchturbo.logging import get_logger

logger = get_logger("torchturbo.antllm_prepacked_reader")

IGNORE_INDEX = -100


_READ_FIELDS = ("input_ids", "labels", "cu_seqlens")

_PACK_CONFIG_FILE = "meta.json"
_PACK_INDEX_FILE = "pack_meta.json"
_PACK_DIR_NAME = "pack_data"
_SHARD_INDEX_FILE = "meta.pkl"
_SHARD_DATA_FILE = "data.{dtype}.npy"

_MDATA_SCHEME = "mdata://"

# antllm's ``SFTConcatSampledDataset`` default for a dataset directory that no
# ``data_info`` entry covers. Used only when the pack records no mdata snapshot,
# i.e. when there is nothing left to recover the mixture from.
_DEFAULT_DATA_INFO = {"ratio": 1.0, "sampling_method": "head", "sampling_seed": 42}

# Mirrors antllm's retry_func_until_timeout (antllm/data/sft_pre_pack/utils.py):
# a shard a peer node just packed may take some seconds to appear on this reader's
# node over the shared filesystem, so file open/load is retried while waiting.
_RETRY_MAX_DURATION = 240  # seconds
_RETRY_SLEEP = 5           # seconds between tries


def _retry_until_timeout(func, *, what, max_duration=_RETRY_MAX_DURATION, sleep_time=_RETRY_SLEEP):
    """Retry a shared-FS file open/load until it succeeds or ``max_duration`` elapses.

    Mirrors antllm's ``retry_func_until_timeout``: ``open(meta.pkl)`` and
    ``np.load(data.<dtype>.npy, mmap_mode='r')`` are both wrapped this way in antllm's
    own shard reader (``np_memory_map_dataset.py``). Without it a reader on a node that
    did not write a shard fails fast on cross-node FS propagation lag instead of
    waiting for the file to become visible.
    """
    start = time.time()
    err: Exception | None = None
    while time.time() - start < max_duration:
        try:
            return func()
        except Exception as e:
            err = e
            logger.warning_rank0(
                "antllm_prepacked_reader: %s not ready yet: %r; retry in %ds (%.0fs left)",
                what, err, sleep_time, max_duration - (time.time() - start),
            )
            time.sleep(sleep_time)
    raise TimeoutError(
        f"antllm_prepacked_reader: {what} timed out after {max_duration}s"
    ) from err


def shard_dir_name(shard_id: int, shard_num: int) -> str:
    """Return antllm's shard directory name (``antllm sft_pre_pack.utils``)."""
    return f"shard_{shard_id:05d}_{shard_num:05d}"


def create_index_sampler(
    data_size: int,
    ratio: float,
    method: str,
    seed: int | None = None,
) -> torch.Tensor:
    """Expand one dataset to ``ratio`` copies of itself, antllm's way.
    """
    if data_size <= 0:
        raise ValueError(f"data_size must be positive, got {data_size}")
    if ratio < 0:
        raise ValueError(f"ratio must be non-negative, got {ratio}")
    if method not in ("head", "tail", "random"):
        raise ValueError(f"method must be 'head', 'tail' or 'random', got {method!r}")

    total_samples = math.floor(data_size * ratio)
    integer_repeats = math.floor(ratio)
    if integer_repeats > 0:
        if method == "head":
            integer_part = torch.arange(data_size).repeat(integer_repeats)
        elif method == "tail":
            integer_part = torch.arange(data_size - 1, -1, -1).repeat(integer_repeats)
        else:
            generator = torch.Generator().manual_seed(seed) if seed is not None else None
            integer_part = torch.cat(
                [torch.randperm(data_size, generator=generator) for _ in range(integer_repeats)]
            )
    else:
        integer_part = torch.empty(0, dtype=torch.long)

    fractional_samples = total_samples - integer_part.size(0)
    if fractional_samples == 0:
        fractional_part = torch.empty(0, dtype=torch.long)
    elif method == "head":
        fractional_part = torch.arange(fractional_samples)
    elif method == "tail":
        fractional_part = torch.arange(data_size - 1, data_size - fractional_samples - 1, -1)
    else:
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        fractional_part = torch.randint(0, data_size, (fractional_samples,), generator=generator)

    return torch.cat([integer_part, fractional_part])


def _field_dtype(field_meta: dict[str, dict[str, Any]], name: str) -> str:
    """Return the dtype bucket holding *name* in one record's field index."""
    for dtype, fields in field_meta.items():
        if name in fields:
            return dtype
    raise KeyError(f"field {name!r} not found in pack shard; have {sorted(field_meta)}")


class _MemoryMapShard:
    """One antllm shard: per-dtype memmapped ``.npy`` files plus a slice index.

    antllm's per-record index (``meta.pkl``) is kept only long enough to
    extract the slice bounds of :data:`_READ_FIELDS`. Its nested dicts cost a
    few KB of Python objects per record -- hundreds of MB for a large pack, in
    the parent *and* in every forked dataloader worker -- so they are replaced
    by two int64 columns.
    """

    def __init__(self, shard_dir: Path) -> None:
        meta_path = shard_dir / _SHARD_INDEX_FILE
        with _retry_until_timeout(
            functools.partial(open, meta_path, "rb"), what=f"open {meta_path}"
        ) as f:
            index = pickle.load(f)
        records = index["data_meta"]
        self.size = len(records)

        # A packed record holds exactly one sample; anything else means this is
        # an encode-stage shard, not a pack-stage one.
        for record in records:
            if record["sample_size"] != 1:
                raise ValueError(
                    f"{shard_dir}: expected packed records with sample_size 1, got "
                    f"{record['sample_size']}; point save_path at the directory "
                    "holding meta.json, not at encode_data"
                )

        # antllm writes every record through the same code path, so the dtype
        # and rank of a field are fixed across the shard; read them off record 0.
        first = records[0]["sample_meta"][0]
        self.field_dtypes = tuple(_field_dtype(first, name) for name in _READ_FIELDS)
        for name, dtype in zip(_READ_FIELDS, self.field_dtypes):
            info = first[dtype][name]
            if tuple(info["shape"]) != (info["len"],):
                raise ValueError(f"{shard_dir}: expected 1-D {name}, got shape {tuple(info['shape'])}")

        self.starts = np.empty((self.size, len(_READ_FIELDS)), dtype=np.int64)
        self.lengths = np.empty((self.size, len(_READ_FIELDS)), dtype=np.int64)
        for row, record in enumerate(records):
            field_meta = record["sample_meta"][0]
            for col, (name, dtype) in enumerate(zip(_READ_FIELDS, self.field_dtypes)):
                info = field_meta[dtype][name]
                self.starts[row, col] = info["start"]
                self.lengths[row, col] = info["len"]

        self.arrays = {}
        for dtype in set(self.field_dtypes):
            data_file = shard_dir / _SHARD_DATA_FILE.format(dtype=dtype)
            self.arrays[dtype] = _retry_until_timeout(
                functools.partial(np.load, data_file, mmap_mode="r"),
                what=f"np.load {data_file}",
            )

    def __len__(self) -> int:
        return self.size

    def read(self, index: int) -> tuple[np.ndarray, ...]:
        """Return the :data:`_READ_FIELDS` slices of one packed sample."""
        return tuple(
            self.arrays[dtype][start : start + length]
            for dtype, start, length in zip(
                self.field_dtypes, self.starts[index], self.lengths[index]
            )
        )


class _PackedDir:
    """One dataset directory under ``pack_save_path``: shards plus sampled indices.

    ``indices`` is antllm's ratio-expanded index table: position ``i`` names the
    sample this dataset contributes at step ``i``.
    """

    def __init__(
        self,
        name: str,
        data_dir: Path,
        shard_num: int,
        ratio: float,
        sampling_method: str,
        sampling_seed: int,
    ) -> None:
        self.name = name
        self.shards = [
            _MemoryMapShard(data_dir / shard_dir_name(shard_id, shard_num)) for shard_id in range(shard_num)
        ]
        self.shard_ends: list[int] = []
        total = 0
        for shard in self.shards:
            total += len(shard)
            self.shard_ends.append(total)

        self.raw_size = total
        self.indices = create_index_sampler(total, ratio, sampling_method, sampling_seed)

    def __len__(self) -> int:
        return int(self.indices.numel())

    def read(self, index: int) -> tuple[np.ndarray, ...]:
        """Read the ``index``-th sampled sample of this dataset."""
        data_idx = int(self.indices[index].item())
        shard_idx = bisect.bisect_right(self.shard_ends, data_idx)
        base = self.shard_ends[shard_idx - 1] if shard_idx > 0 else 0
        return self.shards[shard_idx].read(data_idx - base)


def mdataset_parse(data_path: str) -> list[dict[str, Any]]:
    """Return the datasets one ``mdata://`` URI resolves to.

    The counterpart of antllm's ``mdataset_parse`` (``antllm.data.data_utils``),
    reduced to what this reader needs: antllm also returns the local mount path
    because it goes on to read the data, while here the data was packed long ago
    and only the snapshot's settings are wanted.

    ``mount_data()`` runs before ``get_datasets()``, the order the internal mdata
    dataset plugin uses, for two reasons. mdata folds the snapshot
    ``trainConfig`` into ``get_datasets()`` only once the snapshot is
    materialized, so an unmounted call reports no ``trainWeight``. And for a
    ``label_snapshot`` URI ``get_datasets()`` builds a per-dataset path by
    appending the dataset name to the helper's ``local_path``, which is left at
    mdata's own default here and stays unset until ``mount_data()`` resolves it.

    Only rank 0 mounts: the mount is a round trip against the mdata service,
    every rank would ask for the same snapshot just to read a handful of ratios
    off it, and nothing here touches the mounted files. The other ranks wait in
    the broadcast that hands them rank 0's answer.

    Args:
        data_path: the ``mdata://`` URI to resolve.
    """
    try:
        from mdatasets import MdataHelper
    except ImportError as exc:
        raise ImportError(
            "the pack recorded an mdata data_path, so reading its sampling ratios "
            "needs the internal 'mdatasets' package; install it, or re-pack with "
            "the ratios already applied"
        ) from exc

    payload: list[list[dict[str, Any]] | None] = [None]
    if not dist.is_initialized() or dist.get_rank() == 0:
        helper = MdataHelper(data_path, local_path=None)
        logger.info_rank0("Mounting the mdata snapshot %s to read its sampling ratios", data_path)
        helper.mount_data()
        payload[0] = helper.get_datasets()
    if dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)

    datasets_info = payload[0]
    if not datasets_info:
        raise RuntimeError(f"mdata URI {data_path!r} resolved to no datasets")
    return datasets_info


def rename_mdata_info_for_sft(mdata_info: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Re-key ``mdataset_parse`` output the way antllm's dataset classes read it.

    Ports ``rename_mdata_info_for_sft`` (``antllm.utils.data_utils``): one entry
    per dataset, keyed by the dataset directory name. antllm derives that name as
    ``datasetUk.split("/")[1].split(".")[0]``; mdata hands over the same string
    directly as ``datasetUniqueName``, which is also the name of the directory
    the snapshot mounts to and the pack copies. The camelCase
    ``trainWeight`` / ``samplingMethod`` / ``samplingSeed`` become the snake_case
    keys antllm's ``SFTConcatSampledDataset`` reads, and a dataset the snapshot
    records no settings for keeps antllm's defaults -- ratio 1.0, ``head``, seed
    42 -- rather than turning a missing value into ``int(None)``.
    """
    table: dict[str, dict[str, Any]] = {}
    for entry in mdata_info:
        # mdata only puts the three sampling settings on an entry once the
        # snapshot's trainConfig has been folded in, so all three may be absent.
        train_weight = entry.get("trainWeight")
        sampling_seed = entry.get("samplingSeed")
        table[entry["datasetUniqueName"]] = {
            "ratio": float(train_weight) if train_weight else 1.0,
            "sampling_method": entry.get("samplingMethod") or "head",
            "sampling_seed": int(sampling_seed) if sampling_seed else 42,
            "dataset_id": entry.get("datasetId"),
            "dataset_uk": entry.get("datasetUk"),
            "version": entry.get("datasetVersion"),
        }
    return table


def _resolve_data_info(
    config: dict[str, Any],
    dataset_names: list[str],
    data_path_override: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Map every packed dataset directory to its ratio / sampling settings.

    The settings come from the mdata snapshot recorded in the pack's
    ``meta.json``, because antllm writes no ratio anywhere in the pack itself. A
    pack whose record is a plain directory has no snapshot to consult -- antllm
    writes the mounted path it packed from -- so every directory falls back to
    one pass in on-disk order, antllm's own default when it is given no
    ``data_info``, with a warning: a multi-dataset pack then trains the wrong
    mixture.

    antllm logs an error and carries on when the snapshot and the packed
    directories do not line up. This raises instead, because that mismatch is
    exactly how a run ends up training a wrong mixture at ratio 1.0. Datasets
    the snapshot trains on at ratio 0 are the one legitimate absence: antllm
    skips them, so they were never packed.

    ``data_path_override``, when set, replaces the recorded ``data_path``
    outright, including in the messages below, so the logs always name the
    snapshot the ratios actually came from.
    """
    data_path = data_path_override or config["data_path"]
    if data_path != config["data_path"]:
        logger.warning_rank0(
            "reading sampling ratios from the overridden mdata URI %s; the "
            "pre-pack run under %s recorded data_path=%s",
            data_path,
            config["save_path"],
            config["data_path"],
        )

    if not data_path.startswith(_MDATA_SCHEME):
        logger.warning_rank0(
            "the pre-pack run under %s recorded data_path=%s, which is not an "
            "mdata URI, so its per-dataset sampling ratios cannot be recovered; "
            "training every dataset once at ratio=1.0 method=head seed=42. Re-pack "
            "from the mdata URI to train the intended mixture",
            config["save_path"],
            data_path,
        )
        return {name: dict(_DEFAULT_DATA_INFO) for name in dataset_names}

    table = rename_mdata_info_for_sft(mdataset_parse(data_path))
    skipped = sorted(name for name, entry in table.items() if entry["ratio"] == 0)
    table = {name: entry for name, entry in table.items() if entry["ratio"] > 0}
    if skipped:
        logger.info_rank0(
            "mdata snapshot %s: %d datasets at ratio 0, not packed: %s",
            data_path,
            len(skipped),
            skipped,
        )
    missing = sorted(set(dataset_names) - set(table))
    unknown = sorted(set(table) - set(dataset_names))
    if missing or unknown:
        raise ValueError(
            f"the mdata snapshot {data_path} does not match the packed datasets "
            f"under {config['save_path']}: missing entries for {missing}, "
            f"unknown datasets {unknown}; packed datasets are {sorted(dataset_names)}"
        )
    # The mixture is assembled from three layers (mdata snapshot -> pack
    # directories -> this table), so print what mdata actually reported before
    # the ratios above are applied to it. ``json.dumps`` keeps the whole table on
    # one line and shows exactly which snapshot each number came from.
    logger.info_rank0(
        "mdata data_info for %s (dataset -> ratio / sampling_method / sampling_seed): %s",
        data_path,
        json.dumps(
            {
                name: {
                    "ratio": entry["ratio"],
                    "sampling_method": entry["sampling_method"],
                    "sampling_seed": entry["sampling_seed"],
                    "version": entry["version"],
                }
                for name, entry in table.items()
            },
            sort_keys=True,
        ),
    )
    return table


class AntllmPrePackedDataset(Dataset):
    """antllm pre-packed SFT data in the engine's packed-input contract.

    One item is one packed sequence of ``max_length`` tokens holding many
    sub-sequences (conversations) back to back, with a padded tail.

    Four fields are rewritten on read because antllm targets Megatron-LM:

    * ``labels`` -- antllm stores the Megatron convention (``labels[i]`` targets
      ``input_ids[i + 1]``, i.e. already shifted left by one). The FSDP2 backend
      applies that shift itself, so the shift is undone here to HF alignment.
      Position 0 is always a prompt token, so its label is always ignored and
      nothing is lost by shifting back. Every supervised target sits inside the
      real token prefix -- the last real token is the one supervised to predict
      the EOS that fills the padded tail -- so the tail stays fully ignored.
    * ``position_ids`` -- antllm's increase monotonically across the whole pack
      and are ``(2, L)`` (megatron 1d + block ids). ``assert_packed_inputs``
      requires a zero restart at every ``cu_seq_lens_q`` boundary, so they are
      rebuilt as ``(L,)`` restarting at 0 per sub-sequence. This is not a
      numerical deviation: antllm's positions are the same ramps plus a
      per-sub-sequence constant offset, and RoPE reads only differences within a
      sub-sequence, which the block-diagonal mask keeps separate.
    * ``cu_seq_lens_q`` / ``cu_seq_lens_k`` -- antllm stores the sub-sequence
      boundaries in ``cu_seqlens`` (real tokens only, padding tail excluded);
      they are forwarded as int32 so the engine can rebuild the integer segment
      mask the models read. ``cu_seqlens_padded`` is *not* used: it differs only
      in its last element (``max_length``) and would fold the padding tail into
      the last sub-sequence.
    * ``attention_mask`` -- not emitted. The packed-input contract forbids one
      and derives it from the boundaries instead.

    Only ``input_ids``, ``labels`` and ``cu_seqlens`` are read off disk, and
    only ``context_parallel_size=1`` packs are accepted; see :meth:`__init__`.
    """

    def __init__(self, save_path: str, data_path_override: str | None = None) -> None:
        """
        Args:
            save_path: antllm ``sft_pre_pack`` output directory, the one holding
                ``meta.json``.
            data_path_override: optional ``mdata://`` URI to read the per-dataset
                sampling ratios from, instead of the ``data_path`` the pack
                recorded in ``meta.json``. Empty means "use meta.json".
        """
        with open(Path(save_path) / _PACK_CONFIG_FILE) as f:
            config = json.load(f)
        pack_root = Path(save_path) / _PACK_DIR_NAME
        with open(pack_root / _PACK_INDEX_FILE) as f:
            pack_index = json.load(f)
        # ``meta.json`` is provenance: ``pack_save_path`` is the pack job's own
        # temp directory, so it points at a machine and a job that are gone as
        # soon as the pack is copied anywhere. antllm reads the pack from its
        # ``save_path`` (``SFTConcatSampledDataset`` lists ``<save_path>`` and
        # iterates its datasets), and so does every other path here.
        if str(pack_root) != config["pack_save_path"]:
            logger.info_rank0(
                "%s: reading the pack from %s; meta.json records pack_save_path=%s, "
                "which is where the pack job wrote it and is ignored",
                save_path,
                pack_root,
                config["pack_save_path"],
            )

        self.max_length: int = config["max_length"]
        context_parallel_size = config["context_parallel_size"]
        if context_parallel_size != 1:
            raise ValueError(
                f"the pre-pack run under {save_path} used context_parallel_size="
                f"{context_parallel_size}, which pads every sub-sequence up to a "
                "multiple of 2*cp inside the packed sequence. turbo does its own "
                "Ulysses sequence-parallel padding, so that padding is both wasted "
                "and indistinguishable from a sub-sequence boundary here; re-run "
                "the pre-pack with context_parallel_size=1"
            )
        # Any other rotary type stores position semantics this reader does not
        # reinterpret, and training on them would be a silent wrong-RoPE run.
        rotary_type = config.get("rotary_type", "full-1d")
        if rotary_type != "full-1d":
            raise ValueError(
                f"the pre-pack run under {save_path} used rotary_type={rotary_type!r}; "
                "this reader only rebuilds position_ids for 'full-1d'"
            )

        dataset_names = sorted(entry["relative_path"] for entry in pack_index)
        shard_nums = {entry["relative_path"]: entry["shard_num"] for entry in pack_index}
        table = _resolve_data_info(config, dataset_names, data_path_override)
        # antllm records shard_num=0 for a dataset whose encoded samples produced
        # no pack group at all, and writes no shard directory for it. antllm's own
        # reader discovers data by globbing shard_*, so such a dataset silently
        # never appears; this one reads pack_meta.json and has to skip it here.
        empty_names = [name for name in dataset_names if shard_nums[name] == 0]
        if empty_names:
            logger.warning_rank0(
                "%s: %d of %d packed datasets hold zero shards and contribute no "
                "samples: %s. The pre-pack run packed nothing for them, so the "
                "mixture trained here is missing them entirely",
                pack_root,
                len(empty_names),
                len(dataset_names),
                empty_names,
            )
        self.dirs = [
            _PackedDir(
                name=name,
                data_dir=pack_root / name,
                shard_num=shard_nums[name],
                ratio=table[name]["ratio"],
                sampling_method=table[name]["sampling_method"],
                sampling_seed=table[name]["sampling_seed"],
            )
            for name in dataset_names
            if shard_nums[name] > 0
        ]

        # Cumulative sample count after each dataset directory, for bisect lookup.
        self.cumulative_sizes: list[int] = []
        expanded_total = 0
        raw_total = 0
        for packed_dir in self.dirs:
            raw_total += packed_dir.raw_size
            expanded_total += len(packed_dir)
            self.cumulative_sizes.append(expanded_total)

        # total_pack_data_size counts the samples on disk, before the sampling
        # ratios expand (or shrink) each dataset.
        expected = config["total_pack_data_size"]
        if raw_total != expected:
            raise ValueError(
                f"antllm pre-packed data under {pack_root} holds {raw_total} packed "
                f"samples but meta.json reports total_pack_data_size={expected}; the "
                "pack run is incomplete or the directory was modified"
            )

        logger.info_rank0(
            "Loaded antllm pre-packed dataset: %d packed samples of %d tokens across "
            "%d shards in %d dataset dirs from %s",
            raw_total,
            self.max_length,
            sum(len(packed_dir.shards) for packed_dir in self.dirs),
            len(self.dirs),
            pack_root,
        )
        # The mixture is the one thing a wrong config changes silently, so print
        # what was actually built: raw samples -> expanded samples -> share.
        for packed_dir in self.dirs:
            entry = table[packed_dir.name]
            logger.info_rank0(
                "  %s: %d packed samples, ratio=%.4f method=%s seed=%d -> %d samples (%.2f%%)",
                packed_dir.name,
                packed_dir.raw_size,
                entry["ratio"],
                entry["sampling_method"],
                entry["sampling_seed"],
                len(packed_dir),
                100.0 * len(packed_dir) / expanded_total,
            )

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def _shard_starts(self) -> Iterator[int]:
        """Global index of the first sample of every shard, plus the last sample.

        The flat global index space hides the shard layout, and index 0 always
        resolves to the first shard -- so reading it never exercises the
        cumulative-offset arithmetic that maps an index into a later shard. The
        self-check walks these indices instead, one per shard.
        """
        base = 0
        for packed_dir in self.dirs:
            yield base
            for end in packed_dir.shard_ends[:-1]:
                yield base + end
            base += len(packed_dir)
        yield len(self) - 1

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        dir_idx = bisect.bisect_right(self.cumulative_sizes, index)
        base = self.cumulative_sizes[dir_idx - 1] if dir_idx > 0 else 0
        packed_input_ids, packed_labels, boundaries = self.dirs[dir_idx].read(index - base)

        # Copy out of the memmap: torch.from_numpy must never see a read-only
        # buffer, and the labels below are rebuilt anyway.
        input_ids = np.array(packed_input_ids, dtype=np.int64)
        seq_len = input_ids.shape[0]

        # Undo antllm's Megatron pre-shift; then mask every sub-sequence start
        # so the cross-boundary prediction pair the FSDP2 backend's global
        # left-shift creates is ignored (the class docstring documents this).
        labels = np.empty(seq_len, dtype=np.int64)
        labels[1:] = packed_labels[:-1]
        labels[np.asarray(boundaries[:-1], dtype=np.intp)] = IGNORE_INDEX

        # ``cu_seqlens`` holds the cumulative sub-sequence boundaries of the real
        # tokens; everything past its last entry is the padded tail.
        position_ids = np.zeros(seq_len, dtype=np.int64)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            position_ids[start:end] = np.arange(end - start)

        cu_seq_lens = torch.from_numpy(np.array(boundaries, dtype=np.int32))
        return {
            "input_ids": torch.from_numpy(input_ids),
            "labels": torch.from_numpy(labels),
            "position_ids": torch.from_numpy(position_ids),
            "cu_seq_lens_q": cu_seq_lens,
            "cu_seq_lens_k": cu_seq_lens,
        }


def collate_prepacked(features: list[dict[str, torch.Tensor]], **kwargs: Any) -> dict[str, Any]:
    """Turn one pre-packed sample into the engine's single-row packed batch.

    Every sample already has the same length and carries its own sub-sequence
    boundaries, so there is nothing to pad, truncate or re-pack. The packed
    input contract describes one row per step, so the dataloader must yield one
    sample; a larger global batch belongs to ``--tt_gbs`` and the engine's
    gradient accumulation. ``kwargs`` absorbs swift's ``padding_to`` keyword,
    which does not apply here.
    """
    if len(features) != 1:
        raise ValueError(
            "pre-packed data is trained one packed sequence per step, got a batch "
            f"of {len(features)}; use --per_device_train_batch_size 1 and --tt_gbs "
            "for a larger global batch"
        )

    sample = features[0]
    cu_seq_lens = sample["cu_seq_lens_q"]
    batch: dict[str, Any] = {
        key: sample[key].unsqueeze(0) for key in ("input_ids", "labels", "position_ids")
    }
    batch["cu_seq_lens_q"] = cu_seq_lens
    batch["cu_seq_lens_k"] = cu_seq_lens
    max_length = int(cu_seq_lens.diff().max())
    batch["max_length_q"] = max_length
    batch["max_length_k"] = max_length
    assert_packed_inputs(batch)
    return batch


def _check(save_path: str) -> None:
    """Load a pack directory and check it end to end.

    Runs on the training cluster where the pack data lives::

        python internal/dataset/antllm_pack/antllm_prepacked_reader.py <save_path>

    Prints the mixture table during construction, then reads one sample from
    every shard (plus the last sample) through :func:`collate_prepacked`, which
    validates the packed-input contract. One sample alone would only ever touch
    the first shard, see :meth:`AntllmPrePackedDataset._shard_starts`.
    """
    from torchturbo.logging import LogLevel, logging_configure

    logging_configure(LogLevel.INFO)
    dataset = AntllmPrePackedDataset(save_path)

    indices = sorted(set(dataset._shard_starts()))
    batches = [collate_prepacked([dataset[index]]) for index in indices]
    logger.info_rank0(
        "self-check ok: %d samples validated across %d shards in %d dataset dirs; "
        "last one input_ids %s, longest %d tokens",
        len(indices),
        sum(len(packed_dir.shards) for packed_dir in dataset.dirs),
        len(dataset.dirs),
        tuple(batches[-1]["input_ids"].shape),
        batches[-1]["max_length_q"],
    )


__all__ = [
    "AntllmPrePackedDataset",
    "collate_prepacked",
]


if __name__ == "__main__":
    import sys

    _check(sys.argv[1])
