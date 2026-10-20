"""Compare complete DPO optimizer steps, pack membership, tokens and labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def first_difference(left: Any, right: Any, path: str = "") -> str | None:
    """Return a concise field path for the first mismatch."""
    if type(left) is not type(right):
        return f"{path}: types differ"
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return f"{path}: keys differ"
        for key in left:
            difference = first_difference(left[key], right[key], f"{path}.{key}")
            if difference:
                return difference
    elif isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: lengths {len(left)} != {len(right)}"
        for i, (a, b) in enumerate(zip(left, right)):
            difference = first_difference(a, b, f"{path}[{i}]")
            if difference:
                return difference
    elif left != right:
        return f"{path}: {left!r} != {right!r}"
    return None


def load_trace(directory: Path) -> tuple[dict, dict[tuple[int, int, int], dict]]:
    """Read fresh and resumed trace files, rejecting duplicate batch keys."""
    records = {}
    config = None
    paths = sorted(directory.glob("dp-*.jsonl"))
    if not paths:
        raise ValueError(f"No DPO trace files in {directory}")
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            header = json.loads(next(stream))
            if header["kind"] != "metadata" or header["schema"] != 1:
                raise ValueError(f"Unsupported DPO trace schema: {path}")
            current = {key: header[key] for key in (
                "dp_size", "gradient_accumulation_steps", "micro_batch_size", "max_steps",
            )}
            if config is not None and config != current:
                raise ValueError(f"Trace configuration changed within {directory}")
            config = current
            for line in stream:
                record = json.loads(line)
                key = (record["step"], record["dp_rank"], record["micro_step"])
                if record["kind"] != "microbatch" or record["dp_rank"] != header["dp_rank"]:
                    raise ValueError(f"Invalid microbatch metadata: {path}")
                if key in records:
                    raise ValueError(f"Duplicate DPO batch {key} in {directory}")
                records[key] = record
    return config, records


def compare_traces(
    left: Path, right: Path, *, start_step: int = 1, steps: int | None = None,
) -> int:
    """Compare an explicit complete step range, or the entire configured run."""
    if start_step < 1 or (steps is not None and steps < 1):
        raise ValueError("Step range must be positive (steps are one-based).")
    difference = first_difference(
        json.loads((left / "dataset.json").read_text()),
        json.loads((right / "dataset.json").read_text()), "dataset",
    )
    if difference:
        raise ValueError(difference)
    left_config, a = load_trace(left)
    right_config, b = load_trace(right)
    keys = ("dp_size", "gradient_accumulation_steps", "micro_batch_size")
    if steps is None:
        keys += ("max_steps",)
    for key in keys:
        if left_config[key] != right_config[key]:
            raise ValueError(f"{key}: {left_config[key]} != {right_config[key]}")
    end_step = left_config["max_steps"] if steps is None else start_step + steps - 1
    if end_step < start_step:
        raise ValueError("No optimizer steps selected.")
    count = 0
    for step in range(start_step, end_step + 1):
        for rank in range(left_config["dp_size"]):
            for micro in range(left_config["gradient_accumulation_steps"]):
                key = (step, rank, micro)
                if key not in a or key not in b:
                    raise ValueError(f"Missing batch step={step}, dp={rank}, micro={micro}")
                for name, record in (("left", a[key]), ("right", b[key])):
                    difference = first_difference(record["encoded"], record["collated"], "encoded_vs_collated")
                    if difference:
                        raise ValueError(f"{name} batch {key}: {difference}")
                difference = first_difference(a[key], b[key], f"batch{key}")
                if difference:
                    raise ValueError(difference)
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--start-step", type=int, default=1)
    parser.add_argument("--steps", type=int, help="Number of complete optimizer steps to compare")
    args = parser.parse_args()
    try:
        count = compare_traces(args.left, args.right, start_step=args.start_step, steps=args.steps)
    except (ValueError, OSError, KeyError, StopIteration) as error:
        parser.exit(1, f"FAIL: {error}\n")
    print(f"PASS: {count} microbatches match (packs, encoded tokens, labels and collated text positions).")


if __name__ == "__main__":
    main()
