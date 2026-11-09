#!/usr/bin/env python3
"""Encode HXB cases with the checkpoint template and prepare complete SFT batches."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def encode_case(case: dict[str, Any], tokenizer: Any, max_length: int) -> dict[str, Any]:
    """Mask the complete prompt, including historical assistant and tool turns."""
    prompt, completion = case["prompt"], case["completion"]
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("prompt must be a nonempty message list")
    if not isinstance(completion, list) or len(completion) != 1:
        raise ValueError("completion must contain exactly one assistant message")
    if completion[0]["role"] != "assistant":
        raise ValueError("completion must be an assistant message")
    if prompt[-1]["role"] not in {"user", "tool"}:
        raise ValueError("prompt must end in a user or tool message")
    if not (completion[0].get("content") or "").strip() and not completion[0].get("tool_calls"):
        raise ValueError("completion must contain text or a tool call")
    template_kwargs = {"tools": case.get("tools", []), "enable_thinking": False, "tokenize": False}
    prompt_text = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, **template_kwargs)
    full_text = tokenizer.apply_chat_template(prompt + completion, add_generation_prompt=False, **template_kwargs)
    if not full_text.startswith(prompt_text):
        raise ValueError("The rendered completion does not extend the generation prompt")
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    input_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if input_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Tokenization changed the prompt/completion boundary")
    if len(input_ids) > max_length:
        raise ValueError(f"Case has {len(input_ids)} tokens, exceeding max_length={max_length}")
    if len(input_ids) <= len(prompt_ids):
        raise ValueError("completion has no supervised tokens")
    return {
        "id": case["id"],
        "input_ids": input_ids,
        "labels": [-100] * len(prompt_ids) + input_ids[len(prompt_ids):],
        "lengths": len(input_ids),
    }


def encode_file(source: Path, destination: Path, tokenizer: Any, max_length: int,
                expected_count: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Encode the supplied cases and stop if quarantine changes the comparison set."""
    raw = source.read_bytes()
    output, rejected, ids = [], [], set()
    kinds: Counter[str] = Counter()
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
            encoded = encode_case(case, tokenizer, max_length)
            case_id = encoded["id"]
            if not isinstance(case_id, str) or not case_id or case_id in ids:
                raise ValueError("Case IDs must be nonempty and unique")
            ids.add(case_id)
            output.append(encoded)
            kinds["tool_call" if case["completion"][0].get("tool_calls") else "text"] += 1
        except (KeyError, TypeError, ValueError) as error:
            rejected.append({"line": line_number, "error": str(error)})

    destination.parent.mkdir(parents=True, exist_ok=True)
    quarantine = destination.with_suffix(".quarantine.json")
    quarantine.write_text(json.dumps(rejected, indent=2) + "\n", encoding="utf-8")
    if len(output) != expected_count:
        raise ValueError(
            f"{source}: retained {len(output)} cases, expected {expected_count}; "
            f"inspect {quarantine} before changing the comparison dataset"
        )
    with destination.open("w", encoding="utf-8") as stream:
        for row in output:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    lengths = [row["lengths"] for row in output]
    supervised = [sum(label != -100 for label in row["labels"]) for row in output]
    return {
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "encoded_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "retained": len(output),
        "quarantined": len(rejected),
        "target_kinds": dict(kinds),
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "total_tokens": sum(lengths),
        "min_target_tokens": min(supervised),
        "max_target_tokens": max(supervised),
        "supervised_tokens_per_epoch": sum(supervised),
    }, output


def complete_final_batch(packs: list[list[int]], lengths: list[int], batch_size: int) -> list[list[int]]:
    """Split only the final optimizer batch so FSDP's drop_last loses no cases.

    Splitting at case boundaries preserves isolated attention and every target.
    No case is repeated, and earlier optimizer batches retain their membership.
    """
    remainder = len(packs) % batch_size
    if not remainder:
        return packs
    full_batches = packs[:-remainder]
    tail = [list(pack) for pack in packs[-remainder:]]
    if sum(map(len, tail)) < batch_size:
        raise ValueError("The last batch has fewer cases than data-parallel workers")
    while len(tail) < batch_size:
        candidates = [index for index, pack in enumerate(tail) if len(pack) > 1]
        index = max(candidates, key=lambda i: sum(lengths[j] for j in tail[i]))
        pack = tail[index]
        total = sum(lengths[j] for j in pack)
        prefix_lengths = []
        running = 0
        for sample in pack[:-1]:
            running += lengths[sample]
            prefix_lengths.append(running)
        split = min(range(1, len(pack)), key=lambda k: abs(total - 2 * prefix_lengths[k - 1]))
        tail[index:index + 1] = [pack[:split], pack[split:]]
    return full_batches + tail


def inspect_model(model: Path) -> dict[str, Any]:
    """Check the architecture and attention adapter inventory before allocation."""
    config_path = model / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "bailing_hybrid":
        raise ValueError("Expected model_type=bailing_hybrid")
    if "BailingMoeV3ForCausalLM" not in config.get("architectures", []):
        raise ValueError("Expected architecture BailingMoeV3ForCausalLM")
    index_path = model / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    targets = {"q_proj", "k_proj", "v_proj", "o_proj", "kv_a_proj_with_mqa", "kv_b_proj"}
    modules = []
    for name in index["weight_map"]:
        parts = name.split(".")
        if (len(parts) >= 5 and parts[:2] == ["model", "layers"]
                and int(parts[2]) < config["num_hidden_layers"]
                and parts[-1] == "weight" and parts[-2] in targets):
            modules.append(name.removesuffix(".weight"))
    if len(modules) != 161:
        raise ValueError(f"Expected 161 attention adapter modules, found {len(modules)}")
    return {
        "path": str(model.resolve()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "weight_identity_verified": False,
        "adapter_modules": sorted(modules),
        "tokenizer_sha256": {
            name: hashlib.sha256((model / name).read_bytes()).hexdigest()
            for name in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tt-config", type=Path, help="Read the global pack batch size from engine_config.gbs")
    parser.add_argument("--expected-train", type=int, default=421)
    parser.add_argument("--expected-valid", type=int, default=50)
    args = parser.parse_args()
    if args.tt_config:
        import yaml

        config = yaml.safe_load(args.tt_config.read_text(encoding="utf-8"))
        args.batch_size = int(config["engine_config"]["gbs"])
    if args.max_length <= 0 or args.batch_size <= 0:
        parser.error("max-length and batch-size must be positive")

    from transformers import AutoTokenizer
    import binpacking

    manifest: dict[str, Any] = {
        "contract": "hxb-trajectory-v2", "supervision": "completion_only",
        "model": inspect_model(args.model),
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    train_info, train = encode_file(args.train, args.output_dir / "train.jsonl", tokenizer,
                                    args.max_length, args.expected_train)
    valid_info, valid = encode_file(args.valid, args.output_dir / "valid.jsonl", tokenizer,
                                    args.max_length, args.expected_valid)
    if {row["id"] for row in train} & {row["id"] for row in valid}:
        raise ValueError("Training and validation case IDs overlap")

    lengths = [row["lengths"] for row in train]
    bins = binpacking.to_constant_volume(list(enumerate(lengths)), args.max_length, weight_pos=1)
    packs = complete_final_batch([[item[0] for item in pack] for pack in bins], lengths, args.batch_size)
    consumed = [index for pack in packs for index in pack]
    if sorted(consumed) != list(range(len(train))):
        raise ValueError("Pack plan must consume every training case exactly once")
    packing = {
        "strategy": "binpacking_with_complete_final_batch",
        "max_length": args.max_length,
        "global_batch_size": args.batch_size,
        "num_packs": len(packs),
        "steps_per_epoch": len(packs) // args.batch_size,
        "packs": packs,
    }
    (args.output_dir / "packing.json").write_text(json.dumps(packing, indent=2) + "\n", encoding="utf-8")
    manifest.update(train=train_info, valid=valid_info, packing={key: value for key, value in packing.items() if key != "packs"})
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train": train_info, "valid": valid_info, "packing": manifest["packing"],
                      "manifest": str(manifest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
