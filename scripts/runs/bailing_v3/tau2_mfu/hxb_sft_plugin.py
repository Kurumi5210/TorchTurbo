"""Use prepared HXB tokens and pack boundaries without Swift re-tokenization."""

from __future__ import annotations

import json
from itertools import chain
from pathlib import Path
from typing import Any, Callable

import torch

from torchturbo.adapters.msswift.inputs_bridging import bridge_packed_inputs
from torchturbo.engine.plugins import DatasetPlugin, DatasetPrepareContext, register_plugin


class HxbPackedDataset:
    """One item is a group of independently masked, pre-tokenized cases."""

    def __init__(self, samples: list[dict[str, Any]], packs: list[list[int]]) -> None:
        self.samples = samples
        self.packs = packs

    def __len__(self) -> int:
        return len(self.packs)

    def __getitem__(self, index: int) -> list[dict[str, Any]]:
        return [
            {key: self.samples[sample][key] for key in ("input_ids", "labels", "lengths")}
            for sample in self.packs[index]
        ]


def collate_hxb_packs(batch: list[list[dict[str, Any]]], **kwargs: Any) -> dict[str, Any]:
    """Preserve token labels and reset positions at every isolated case boundary."""
    samples = list(chain.from_iterable(batch))
    lengths = [len(sample["input_ids"]) for sample in samples]
    inputs = {
        key: torch.tensor(list(chain.from_iterable(sample[key] for sample in samples)), dtype=torch.long).unsqueeze(0)
        for key in ("input_ids", "labels")
    }
    inputs["position_ids"] = torch.tensor(
        list(chain.from_iterable(range(length) for length in lengths)), dtype=torch.long
    ).unsqueeze(0)
    return bridge_packed_inputs(inputs, sequence_lengths=lengths)


@register_plugin("hxb_sft")
class HxbSftPlugin(DatasetPlugin):
    """Read the prepared manifest passed through --tt_plugin plugin.py:manifest.json."""

    def dataset_names(self) -> list[str]:
        return ["hxb_sft"]

    def dataset_info(self, name: str) -> dict[str, Any]:
        return {}

    def provides_dataset(self, name: str) -> bool:
        return name == "hxb_sft"

    def get_collate_fn(self) -> Callable:
        return collate_hxb_packs

    def build_dataset(self, name: str, ctx: DatasetPrepareContext) -> tuple[HxbPackedDataset, None]:
        if self.config is None:
            raise ValueError("Bind hxb_sft_plugin.py to the prepared manifest.json")
        directory = Path(self.config).parent
        manifest = json.loads(Path(self.config).read_text(encoding="utf-8"))
        plan = json.loads((directory / "packing.json").read_text(encoding="utf-8"))
        if plan["max_length"] != ctx.max_length:
            raise ValueError("Prepared pack length differs from the training max_length")
        samples = [json.loads(line) for line in (directory / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(samples) != manifest["train"]["retained"]:
            raise ValueError("Prepared sample count differs from the manifest")
        return HxbPackedDataset(samples, plan["packs"]), None
