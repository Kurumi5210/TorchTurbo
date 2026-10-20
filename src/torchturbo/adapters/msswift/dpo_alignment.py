"""Deterministic DPO sampling and logical input traces for Megatron comparison.

This module only imports PyTorch and the standard library at import time, so
the GPU trace plugin can load it without activating TorchTurbo.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler

TRACE_KEY = "_turbo_dpo_trace"
TRACE_SCHEMA = 1


class MegatronDPOSampler(Sampler[int]):
    """Match Swift's Megatron random sampler for complete optimizer steps.

    The reference is swift/megatron/trainers/batch_sampler.py at acb9f4cf.
    Its permutation seed is the epoch, independently of the dataset seed.
    Training stops after floor(num_packs / global_batch_size) steps for the
    one-epoch debug run. Do not truncate before constructing the permutation.
    """

    def __init__(
        self,
        total_samples: int,
        *,
        dp_rank: int,
        dp_size: int,
        micro_batch_size: int,
        gradient_accumulation_steps: int,
        shuffle: bool = True,
        data_sharding: bool = False,
        skip_batches: int = 0,
    ) -> None:
        if min(total_samples, dp_size, micro_batch_size, gradient_accumulation_steps) <= 0:
            raise ValueError("DPO dataset and batch dimensions must be positive.")
        if not 0 <= dp_rank < dp_size:
            raise ValueError("dp_rank must be in [0, dp_size).")
        self.total_samples = total_samples
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.micro_batch_size = micro_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.shuffle = shuffle
        self.data_sharding = data_sharding
        self.epoch = 0
        self.global_batch_size = dp_size * micro_batch_size * gradient_accumulation_steps
        self.num_steps = total_samples // self.global_batch_size
        self.num_samples = self.num_steps * gradient_accumulation_steps * micro_batch_size
        if self.num_steps == 0:
            raise ValueError("The DPO dataset cannot fill one global batch.")
        if not 0 <= skip_batches <= self.num_samples // micro_batch_size:
            raise ValueError("skip_batches is outside this epoch.")
        self.skip_batches = skip_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples - self.skip_batches * self.micro_batch_size

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.epoch)
        if self.shuffle and self.data_sharding:
            bucket_size = self.total_samples // (self.micro_batch_size * self.dp_size) * self.micro_batch_size
            order = torch.randperm(bucket_size, generator=generator).tolist()
            indices = [self.dp_rank * bucket_size + i for i in order]
        else:
            bucket_size = self.total_samples // self.micro_batch_size * self.micro_batch_size
            order = torch.randperm(bucket_size, generator=generator).tolist() if self.shuffle else range(bucket_size)
            indices = order[self.dp_rank::self.dp_size]
        start = self.skip_batches * self.micro_batch_size
        return iter(indices[start:self.num_samples])


@dataclass
class IndexedDPOItem:
    index: int
    data: dict[str, Any] | list[dict[str, Any]]


class IndexedDPODataset(Dataset):
    """Carry pack indices to the collator without changing encoded samples."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> IndexedDPOItem:
        return IndexedDPOItem(index, self.dataset[index])


def token_digest(values: Sequence[int] | torch.Tensor) -> str:
    """Hash signed int64 values in a device- and byte-order-independent format."""
    tensor = torch.as_tensor(values, dtype=torch.int64).detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().astype("<i8", copy=False).tobytes()).hexdigest()


def sequence_signature(
    input_ids: Sequence[int] | torch.Tensor,
    labels: Sequence[int] | torch.Tensor,
    positions: Sequence[int] | torch.Tensor,
) -> dict[str, Any]:
    labels = torch.as_tensor(labels, dtype=torch.int64)
    return {
        "length": len(input_ids),
        "supervised_tokens": int((labels != -100).sum()),
        "input_ids": token_digest(input_ids),
        "labels": token_digest(labels),
        "position_ids": token_digest(positions),
    }


@dataclass
class DPOTraceCollator:
    """Fingerprint encoded pairs and collated tensors before device transfer.

    CP pads individual GPU sequences; TorchTurbo pads the flattened tail for
    SP. Compare the real text tokens and their positions, excluding these
    physical padding slots. Labels here are unshifted on both backends.
    """

    collator: Callable[[list], dict[str, Any]]

    def __call__(self, batch: list[IndexedDPOItem]) -> dict[str, Any]:
        packs = [item.data if isinstance(item.data, list) else [item.data] for item in batch]
        samples = [sample for pack in packs for sample in pack]
        encoded = []
        lengths = []
        for prefix in ("chosen_", "rejected_"):
            for sample in samples:
                ids, labels = sample[prefix + "input_ids"], sample[prefix + "labels"]
                lengths.append(len(ids))
                encoded.append(sequence_signature(ids, labels, range(len(ids))))

        inputs = self.collator([item.data for item in batch])
        ids, labels = inputs["input_ids"], inputs["labels"]
        if ids.device.type != "cpu" or ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("DPO alignment tracing requires packed CPU text inputs with shape [1, tokens].")
        positions = inputs.get("text_position_ids", inputs["position_ids"])
        positions = positions.reshape(-1, ids.shape[-1])[0]
        starts = (positions == 0).nonzero().flatten().tolist()
        if len(starts) < len(lengths):
            raise ValueError("Collated DPO positions lost sample boundaries.")
        collated = [
            sequence_signature(ids[0, start:start + length], labels[0, start:start + length],
                               positions[start:start + length])
            for start, length in zip(starts, lengths)
        ]
        inputs[TRACE_KEY] = {
            "pack_indices": [item.index for item in batch],
            "pairs_per_pack": [len(pack) for pack in packs],
            "encoded": encoded,
            "collated": collated,
        }
        return inputs


def describe_dataset(dataset: Any) -> dict[str, Any] | None:
    """Describe the filtered dataset and exact pack membership, without encoding twice."""
    from swift.dataset import PackingDataset

    if dataset is None:
        return None
    if not isinstance(dataset, PackingDataset):
        raise ValueError("DPO alignment requires Swift's non-streaming PackingDataset.")
    return {
        "num_pairs": len(dataset.dataset),
        "num_packs": len(dataset),
        "packing_length": dataset.packing_length,
        "packing_indices": dataset.packed_idx,
        "packed_lengths": dataset.packed_length,
    }


def save_dataset_manifest(directory: str, train_dataset: Any, val_dataset: Any) -> None:
    """Save pack membership once; refuse a changed dataset when resuming a trace."""
    path = Path(directory) / "dataset.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": TRACE_SCHEMA,
        "train": describe_dataset(train_dataset),
        "validation": describe_dataset(val_dataset),
    }
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError(f"DPO trace dataset changed: {path}")
        return
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream)
        stream.write("\n")


class DPOTraceWriter:
    """Write only batches consumed by training, never worker-prefetched batches."""

    def __init__(
        self, directory: str, *, dp_rank: int, dp_size: int,
        gradient_accumulation_steps: int, micro_batch_size: int,
        start_step: int, max_steps: int,
    ) -> None:
        self.dp_rank = dp_rank
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.last_step = start_step
        self.micro_step = 0
        self.path = Path(directory) / f"dp-{dp_rank:05d}-from-{start_step:08d}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "kind": "metadata", "schema": TRACE_SCHEMA, "dp_size": dp_size,
            "dp_rank": dp_rank, "gradient_accumulation_steps": gradient_accumulation_steps,
            "micro_batch_size": micro_batch_size, "start_step": start_step, "max_steps": max_steps,
        }
        # A fresh attempt or a resume at the same step needs a new trace directory.
        # Silently appending would make duplicate steps look like data drift.
        with self.path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(header) + "\n")

    def write(self, completed_steps: int, trace: dict[str, Any]) -> None:
        if completed_steps != self.last_step:
            if completed_steps != self.last_step + 1 or self.micro_step != self.gradient_accumulation_steps:
                raise ValueError("DPO trace step or gradient accumulation is discontinuous.")
            self.last_step = completed_steps
            self.micro_step = 0
        if self.micro_step >= self.gradient_accumulation_steps:
            raise ValueError("More DPO microbatches than configured gradient accumulation steps.")
        record = {
            "kind": "microbatch", "step": completed_steps + 1,
            "micro_step": self.micro_step, "dp_rank": self.dp_rank, **trace,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        self.micro_step += 1
