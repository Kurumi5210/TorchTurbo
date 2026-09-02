#!/usr/bin/env python3
"""Precision verifier for offline-tokenized datasets (both frameworks).

Compares the OFFLINE artifact against a fresh ONLINE tokenization of the same
raw data with the same args — row by row — and reports the first divergence.

LLaMA-Factory mode (``lf``):
    python verify_tokenized.py lf train_sft.yaml \
        --tokenized_path /shared_fs/tokenized/my_sft_v1 [--num_samples 200]

    online  = get_dataset(...) with tokenized_path=None (full re-tokenize)
    offline = datasets.load_from_disk(tokenized_path)
    PASS iff every compared row has identical input_ids (+ labels when present).

ms-swift mode (``swift``):
    python verify_tokenized.py swift \
        --model_type turbo_qwen3_5_moe --model /path/to/model \
        --template qwen3_5 --max_length 262144 \
        --cached_dataset /shared_fs/tokenized/my_sft_v1/train [--num_samples 200]

    online  = template.encode(raw_row) for each cached row (the cached artifact
              stores raw rows + precomputed `lengths`; encoding happens lazily
              at fetch time in BOTH online and cached training paths)
    checks  = (1) export-time contract args (torchturbo_tokenize_args.json)
              match the current CLI args; (2) template.encode succeeds on the
              sampled rows and its token count matches the stored `lengths`.
"""

from __future__ import annotations

import json
import os
import random
import sys

from _common import SWIFT_CONTRACT_ARGS, SWIFT_CONTRACT_FILENAME, ensure_repo_src, pop_value

ensure_repo_src()


def _row_signature(row: dict, keys=("input_ids", "labels")) -> dict:
    return {k: list(row[k]) for k in keys if k in row and row[k] is not None}


def _shuffled_sample_indices(n: int, num_samples: int) -> list[int]:
    idxs = list(range(n))
    random.Random(0).shuffle(idxs)
    return idxs[: min(num_samples, n)]


# --------------------------------------------------------------------------- lf

def verify_lf(argv: list[str]) -> int:
    argv, tokenized_path = pop_value(argv, "--tokenized_path")
    argv, num_samples = pop_value(argv, "--num_samples")
    num_samples = int(num_samples or 200)
    if not tokenized_path:
        raise SystemExit("lf mode requires --tokenized_path <saved_dir>")

    import torchturbo

    torchturbo.activate("llamafactory_v0")

    from datasets import DatasetDict, load_from_disk
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args, read_args
    from llamafactory.model import load_tokenizer

    raw_args = read_args(argv)
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(raw_args)
    stage = finetuning_args.stage

    # Force a REAL re-tokenization: without this, HF datasets' map cache would
    # make "online" trivially equal to whatever was computed before.
    data_args.overwrite_cache = True

    if stage == "pt":
        from torchturbo.adapters.llamafactory_v0.patches import PretrainProcessorPatch

        PretrainProcessorPatch().patch(engine=None)

    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)

    online = get_dataset(
        template=template,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        stage=stage,
        tokenizer=tokenizer_module["tokenizer"],
        processor=tokenizer_module.get("processor"),
    )["train_dataset"]

    offline_obj = load_from_disk(tokenized_path)
    offline = offline_obj["train"] if isinstance(offline_obj, DatasetDict) else offline_obj

    n_online, n_offline = len(online), len(offline)
    if n_online != n_offline:
        print(f"FAIL: row count differs online={n_online} offline={n_offline}")
        return 1

    bad = 0
    idxs = _shuffled_sample_indices(n_online, num_samples)
    for i in idxs:
        sig_on = _row_signature(online[i])
        sig_off = _row_signature(offline[int(i)])
        if sig_on != sig_off:
            bad += 1
            if bad <= 3:
                for k in sig_on:
                    if sig_on[k] != sig_off.get(k):
                        print(f"row {i} key {k}: online len={len(sig_on[k])} offline len={len(sig_off.get(k) or [])}")
                        print(f"  online [:16] = {sig_on[k][:16]}")
                        print(f"  offline[:16] = {(sig_off.get(k) or [])[:16]}")
    if bad:
        print(f"FAIL: {bad}/{len(idxs)} sampled rows differ")
        return 1
    print(f"PASS: {len(idxs)}/{n_online} rows identical (input_ids/labels) between online and offline")
    return 0


# ----------------------------------------------------------------------- swift

def _check_swift_contract(cached: str, args) -> bool:
    """Compare the export-time contract file against the current CLI args."""
    contract_path = os.path.join(os.path.dirname(os.path.normpath(cached)), SWIFT_CONTRACT_FILENAME)
    if not os.path.isfile(contract_path):
        print(f"WARN: no contract file at {contract_path} (artifact predates this tooling?)")
        return True
    with open(contract_path, encoding="utf-8") as f:
        contract = json.load(f)
    mismatches = {}
    for name in SWIFT_CONTRACT_ARGS:
        old = contract.get(name)
        new = getattr(args, name, None)
        if str(old) != str(new) and old is not None:
            mismatches[name] = (old, new)
    for k, (old, new) in mismatches.items():
        print(f"FAIL: arg {k}: exported with {old!r}, training with {new!r}")
    return not mismatches


def verify_swift(argv: list[str]) -> int:
    import torchturbo.adapters.msswift.plugin  # noqa: F401  (register turbo models/templates)

    argv, cached = pop_value(argv, "--cached_dataset")
    argv, num_samples = pop_value(argv, "--num_samples")
    num_samples = int(num_samples or 200)
    if not cached:
        raise SystemExit("swift mode requires --cached_dataset <saved_train_dir>")

    from swift.arguments import ExportArguments
    from swift.utils import parse_args

    args, _remaining = parse_args(ExportArguments, argv)

    # (1) contract check
    if not _check_swift_contract(cached, args):
        return 1

    # (2) re-encode sample rows and compare token counts with stored lengths
    from datasets import load_from_disk

    dataset = load_from_disk(cached)
    if hasattr(dataset, "keys") and "train" in getattr(dataset, "keys", lambda: [])():
        dataset = dataset["train"]

    # Build processor + template through swift's own args helpers — the exact
    # path SwiftSft._prepare_model_tokenizer/_prepare_template use.
    _model, processor = args.get_model_processor(load_model=False)
    template = args.get_template(processor)
    template.set_mode("train")

    n = len(dataset)
    bad = 0
    idxs = _shuffled_sample_indices(n, num_samples)
    for i in idxs:
        row = dataset[int(i)]
        encoded = template.encode(row, return_length=True)
        # determinism: encoding the same row twice must be bit-identical
        encoded2 = template.encode(row, return_length=True)
        if encoded.get("input_ids") != encoded2.get("input_ids") or encoded.get("labels") != encoded2.get("labels"):
            bad += 1
            if bad <= 3:
                print(f"row {i}: template.encode is NON-DETERMINISTIC (input_ids differ across calls)")
            continue
        got = encoded.get("lengths") or [len(encoded["input_ids"])]
        want = row.get("lengths") or ([row["length"]] if "length" in row else None)
        if want is not None and list(got) != list(want):
            bad += 1
            if bad <= 3:
                print(f"row {i}: stored lengths={want} re-encoded lengths={got}")
    if bad:
        print(f"FAIL: {bad}/{len(idxs)} rows failed determinism/length checks")
        return 1
    print(f"PASS: contract OK; {len(idxs)}/{n} rows encode deterministically to stored lengths")
    return 0


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"lf", "swift"}:
        raise SystemExit(__doc__)
    mode = sys.argv[1]
    sys.exit(verify_lf(sys.argv[2:]) if mode == "lf" else verify_swift(sys.argv[2:]))


if __name__ == "__main__":
    main()
