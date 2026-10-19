"""Verify a DPO debug branch against real logged arrays using CPU Swift."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

from verify_logged_inputs_cpu import compare_branch, contiguous_ranges, extract_sequences, read_records


def load_template_module(path: Path) -> ModuleType:
    """Execute the module with only its package logging import substituted.

    The template factory and register function run unchanged. This avoids
    importing the accelerator engine in the CPU-only verification process.
    """
    tree = ast.parse(path.read_text())
    logging_imports = [
        node for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.level == 3 and node.module == "logging"
    ]
    assert len(logging_imports) == 1
    tree.body.remove(logging_imports[0])
    module = ModuleType("dpo_debug_template_cpu")
    module.__file__ = str(path)
    logger = logging.getLogger(module.__name__)
    module.get_logger = lambda name: SimpleNamespace(info_rank0=logger.info, debug_rank0=logger.debug)
    sys.modules[module.__name__] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def recover_messages(input_ids: list[int], tokenizer: Any) -> list[dict[str, str]]:
    """Recover text and tool messages from the printed ChatML token arrays."""
    starts = [i for i, value in enumerate(input_ids) if value == 248045]
    messages = []
    for start, end in zip(starts, starts[1:] + [len(input_ids)]):
        assert input_ids[start + 2] == 198
        assert input_ids[end - 2:end] == [248046, 198]
        role = tokenizer.decode(input_ids[start + 1:start + 2])
        content = tokenizer.decode(input_ids[start + 3:end - 2])
        if role == "user" and content.startswith("<tool_response>\n"):
            parts = re.findall(r"<tool_response>\n(.*?)\n</tool_response>", content, re.DOTALL)
            assert "\n".join(f"<tool_response>\n{part}\n</tool_response>" for part in parts) == content
            messages.extend({"role": "tool", "content": part} for part in parts)
        else:
            messages.append({"role": role, "content": content})
    return messages


def summarize(encoded: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for branch in ("chosen", "rejected"):
        labels = encoded[f"{branch}_labels"]
        positions = [i for i, label in enumerate(labels) if label != -100]
        result[branch] = {
            "input_tokens": len(encoded[f"{branch}_input_ids"]),
            "supervised_tokens": len(positions),
            "supervised_ranges": contiguous_ranges(positions),
        }
    return result


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--swift-source", type=Path)
    parser.add_argument("--gpu-log", type=Path, default=root / "gpu_kubemaker_log.csv")
    parser.add_argument("--npu-log", type=Path, default=root / "log-0.txt")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["USE_HF"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.swift_source:
        sys.path.insert(0, str(args.swift_source.resolve()))

    import dill
    import swift
    from swift.model import get_model_processor
    from swift.template import TEMPLATE_MAPPING, get_template
    from swift.template.utils import get_last_user_round

    assert swift.__version__ == "4.4.2", "This comparison is pinned to the logged NPU Swift 4.4.2."
    model, processor = get_model_processor(
        str(args.tokenizer.resolve()), load_model=False, download_model=False,
        model_type="qwen3_5", device_map="cpu",
    )
    assert model is None
    settings = dict(max_length=128000, loss_scale="last_round+ignore_empty_think", padding_free=True,
                    add_non_thinking_prefix=True)
    baseline = get_template(processor, template_type="qwen3_5", **settings)
    baseline.set_mode("rlhf")
    source = root.parent / "src/torchturbo/models/qwen3_5/swift_template.py"
    module = load_template_module(source)
    experiment = module.DPO_DEBUG_EXPERIMENT
    assert experiment in {"history_think", "last_tool_mask"}
    module.register()
    registered_class = TEMPLATE_MAPPING["qwen3_5"].template_cls
    assert registered_class.__turbo_dpo_debug__ == experiment
    module.register()
    assert TEMPLATE_MAPPING["qwen3_5"].template_cls is registered_class
    debug = get_template(processor, template_type="qwen3_5", **settings)
    debug.set_mode("rlhf")
    sequences = {
        "gpu": extract_sequences(read_records(args.gpu_log, True)),
        "npu": extract_sequences(read_records(args.npu_log, False)),
    }
    results = {"experiment": experiment, "swift_version": swift.__version__, "device": "cpu",
               "model_weights_loaded": False, "engine_imported": False,
               "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "real_sample": {}}
    for origin, logged in sequences.items():
        row = {"messages": recover_messages(logged["chosen"].input_ids, processor.tokenizer),
               "rejected_messages": recover_messages(logged["rejected"].input_ids, processor.tokenizer)}
        saved = copy.deepcopy(row)
        original = baseline.encode(row)
        actual = debug.encode(row)
        assert row == saved
        for branch in ("chosen", "rejected"):
            npu = sequences["npu"][branch]
            gpu = sequences["gpu"][branch]
            assert original[f"{branch}_input_ids"] == npu.input_ids
            assert original[f"{branch}_labels"] == npu.labels
            comparison, extra = compare_branch(gpu, npu)
            if experiment == "history_think":
                expected_labels = npu.labels.copy()
                for block in comparison["removed_gpu_empty_think_blocks"]:
                    start, end = block["gpu_range"]
                    expected_labels[start:start] = [-100] * (end - start)
                assert actual[f"{branch}_input_ids"] == gpu.input_ids
                assert actual[f"{branch}_labels"] == expected_labels
            else:
                expected_labels = npu.labels.copy()
                for index in extra:
                    expected_labels[index] = -100
                assert actual[f"{branch}_input_ids"] == npu.input_ids
                assert actual[f"{branch}_labels"] == expected_labels
        results["real_sample"][origin + "_rendered"] = summarize(actual)

    first_call = "<tool_call>\n<function=lookup>\n<parameter=id>\nONE\n</parameter>\n</function>\n</tool_call>"
    second_call = first_call.replace("ONE", "TWO")
    history = [{"role": "user", "content": "Old question"},
               {"role": "assistant", "content": "<think>Old reasoning</think>Old answer"}]
    trajectory = [{"role": "assistant", "content": first_call}, {"role": "tool", "content": "First result"},
                  {"role": "assistant", "content": second_call}, {"role": "tool", "content": "Second result"}]
    messages = history + [{"role": "user", "content": "New question"}] + trajectory
    row = {"messages": messages + [{"role": "assistant", "content": "FINAL_GOOD"}],
           "rejected_messages": messages + [{"role": "assistant", "content": "FINAL_BAD"}]}
    original = baseline.encode(copy.deepcopy(row))
    actual = debug.encode(copy.deepcopy(row))
    for branch, answer in (("chosen", "FINAL_GOOD"), ("rejected", "FINAL_BAD")):
        labels = actual[f"{branch}_labels"]
        supervised_text = processor.tokenizer.decode([token for token in labels if token != -100])
        assert answer in supervised_text
        if experiment == "history_think":
            assert "Old reasoning" in processor.tokenizer.decode(actual[f"{branch}_input_ids"])
            assert [token for token in labels if token != -100] == [
                token for token in original[f"{branch}_labels"] if token != -100
            ]
        else:
            assert actual[f"{branch}_input_ids"] == original[f"{branch}_input_ids"]
            assert "ONE" not in supervised_text and "TWO" not in supervised_text
    restored = dill.loads(dill.dumps(debug))
    assert restored.encode(copy.deepcopy(row)) == actual

    # Check both rejected-response forms without changing their construction.
    if experiment == "last_tool_mask":
        for response in ("FINAL_BAD", trajectory + [{"role": "assistant", "content": "FINAL_BAD"}]):
            alternative = {"messages": row["messages"], "rejected_response": response}
            original_response = baseline.encode(copy.deepcopy(alternative))
            debug_response = debug.encode(copy.deepcopy(alternative))
            for branch in ("chosen", "rejected"):
                assert debug_response[f"{branch}_input_ids"] == original_response[f"{branch}_input_ids"]
        explicit = copy.deepcopy(row)
        for field in ("messages", "rejected_messages"):
            explicit[field][3]["loss"] = True
        explicit_encoded = debug.encode(explicit)
        supervised_text = processor.tokenizer.decode([
            token for token in explicit_encoded["chosen_labels"] if token != -100
        ])
        assert "ONE" in supervised_text and "TWO" not in supervised_text

    # SFT, inference, and non-last-round strategies keep their existing behavior.
    for mode in ("train", "transformers"):
        baseline.set_mode(mode)
        debug.set_mode(mode)
        assert baseline.encode(copy.deepcopy(row)) == debug.encode(copy.deepcopy(row))
    baseline.set_mode("rlhf")
    debug.set_mode("rlhf")
    baseline._loss_scale = debug._loss_scale = "default+ignore_empty_think"
    assert baseline.encode(copy.deepcopy(row)) == debug.encode(copy.deepcopy(row))
    assert get_last_user_round(row["messages"]) == 2
    results["checks"] = ["complete_logged_arrays", "registration_idempotent", "two_tool_hops",
                          "dill_roundtrip", "sft_and_inference_unchanged", "other_loss_strategy_unchanged",
                          "shared_boundary_function_unchanged"]
    if experiment == "last_tool_mask":
        results["checks"] += ["rejected_response_inputs_unchanged", "explicit_loss_override_preserved"]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
