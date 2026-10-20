"""Exercise the scoped backport with the actual Swift 4.4.2 text encoder."""

from copy import deepcopy
import pickle

import pytest
import torch

swift = pytest.importorskip("swift")
if swift.__version__ != "4.4.2":
    pytest.skip("Qwen3.8 backport targets Swift 4.4.2", allow_module_level=True)

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3_5Config
from swift.model import ModelInfo, MODEL_MAPPING
from swift.template import get_template, TEMPLATE_MAPPING, get_last_user_round
from swift.template.template_inputs import StdTemplateInputs
from torchturbo.models.qwen3_5.swift_template import register


@pytest.fixture
def tokenizer():
    vocab = {word: i for i, word in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.add_special_tokens(["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>"])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="<|im_end|>", pad_token="<|endoftext|>",
    )
    tokenizer.model_info = ModelInfo(
        "qwen3_5", ".", torch.bfloat16, 128000, None, 0,
        config=Qwen3_5Config(), task_type="causal_lm",
    )
    tokenizer.model_meta = MODEL_MAPPING["qwen3_5"]
    register()
    return tokenizer


def _template(tokenizer, name="qwen3_8", **kwargs):
    template = get_template(
        tokenizer, template_type=name, loss_scale="last_round+ignore_empty_think", **kwargs,
    )
    template.set_mode("train")
    return template


def _conversation():
    return [
        {"role": "user", "content": "old-query"},
        {"role": "assistant", "content": "old-call"},
        {"role": "tool", "content": "old-observation"},
        {"role": "assistant", "content": "old-answer"},
        {"role": "user", "content": "new-query"},
        {"role": "assistant", "content": "new-call"},
        {"role": "tool", "content": "new-observation"},
        {"role": "assistant", "content": "new-answer"},
    ]


def _texts(template, messages):
    encoded = template.encode({"messages": messages})
    return (
        template.tokenizer.decode(encoded["input_ids"]),
        template.tokenizer.decode([v for v in encoded["labels"] if v != -100]),
    )


def test_qwen38_history_prefixes_and_last_tool_mask(tokenizer):
    messages = _conversation()
    original = deepcopy(messages)
    text, targets = _texts(_template(tokenizer), messages)
    assert text.count("<think>\n\n</think>\n\n") == 4
    assert "new-answer" in targets
    assert "new-call" not in targets
    assert "old-answer" not in targets
    assert "<think>" not in targets
    assert messages == original


def test_qwen35_and_global_last_round_behavior_are_unchanged(tokenizer):
    messages = _conversation()
    _texts(_template(tokenizer), messages)
    text, targets = _texts(_template(tokenizer, "qwen3_5"), messages)
    assert text.count("<think>\n\n</think>\n\n") == 2
    assert "new-call" in targets and "new-answer" in targets
    assert "old-answer" not in targets
    assert get_last_user_round(messages) == 4


def test_explicit_loss_overrides_last_round(tokenizer):
    messages = _conversation()
    messages[1]["loss"] = True
    messages[-1]["loss"] = False
    _, targets = _texts(_template(tokenizer), messages)
    assert "old-call" in targets
    assert "new-answer" not in targets
    assert "new-call" not in targets


@pytest.mark.parametrize("preserve", [None, False, True])
def test_history_thinking_override_and_no_duplicate_prefix(tokenizer, preserve):
    messages = _conversation()
    messages[3]["content"] = "<think>old-reason</think>old-answer"
    messages[-1]["content"] = "<think>\n\n</think>\n\nnew-answer"
    text, targets = _texts(_template(tokenizer, preserve_thinking=preserve), messages)
    assert ("old-reason" in text) is (preserve is not False)
    assert text.count("<think>\n\n</think>\n\nnew-answer") == 1
    assert "new-answer" in targets
    assert "old-reason" not in targets


def test_list_content_keeps_explicit_per_fragment_loss(tokenizer):
    template = _template(tokenizer)
    messages = _conversation()
    messages[5]["content"] = ["new-call", "extra-call"]
    messages[5]["loss"] = [True, False]
    inputs = StdTemplateInputs(messages=deepcopy(messages))
    template._add_non_thinking_prefix(inputs)
    assert inputs.messages[5]["content"] == ["<think>\n\n</think>\n\nnew-call", "extra-call"]
    from swift.template import ContextType

    contexts, scales = template.loss_scale(
        [messages[i]["content"] for i in (1, 3, 5, 7)],
        [ContextType.RESPONSE] * 4, messages,
    )
    included = [context for context, scale in zip(contexts, scales) if scale]
    assert included == ["new-call", "new-answer"]


def test_reasoning_instruction_is_inactive_unless_enabled(tokenizer):
    inputs = StdTemplateInputs(messages=[{"role": "user", "content": "query"}])
    assert _template(tokenizer)._get_reasoning_instructions(inputs) == ""
    template = _template(tokenizer, enable_thinking=True)
    assert template._get_reasoning_instructions(inputs).startswith("Reasoning effort is set to xhigh.")
    inputs.chat_template_kwargs["reasoning_effort"] = "low"
    assert template._get_reasoning_instructions(inputs).startswith("Reasoning effort is set to low.")


def test_registration_and_worker_pickle(tokenizer):
    cls = TEMPLATE_MAPPING["qwen3_8"].template_cls
    register()
    assert TEMPLATE_MAPPING["qwen3_8"].template_cls is cls
    template = _template(tokenizer)
    before = _texts(template, _conversation())
    restored = pickle.loads(pickle.dumps(template))
    assert _texts(restored, _conversation()) == before


def test_real_dpo_collator_keeps_tokens_labels_and_mrope_across_cp_padding(tokenizer):
    from functools import partial
    from types import SimpleNamespace
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    from torchturbo.adapters.msswift.dpo_alignment import DPOTraceCollator, IndexedDPOItem, TRACE_KEY
    from torchturbo.adapters.msswift.inputs_bridging import SwiftInputCollator

    template = _template(tokenizer, padding_free=True)
    template.set_mode("rlhf")
    # PackingDataset sets this flag before the trainer builds its collator.
    template.packing = True
    model = SimpleNamespace(config=tokenizer.model_info.config)
    model.get_rope_index = Qwen3_5Model.get_rope_index.__get__(model)
    template.model = model
    samples = [template.encode({"messages": _conversation(), "rejected_response": answer})
               for answer in ("bad-answer", "another-bad-answer")]

    npu = DPOTraceCollator(SwiftInputCollator(template.data_collator, template))(
        [IndexedDPOItem(0, deepcopy(samples))],
    )
    template.use_megatron = True
    template.sequence_parallel_size = 4
    gpu = DPOTraceCollator(partial(template.data_collator, padding_to=2048))(
        [IndexedDPOItem(0, deepcopy(samples))],
    )
    assert gpu["input_ids"].shape[-1] % 2048 == 0
    assert gpu["input_ids"].shape[-1] > npu["input_ids"].shape[-1]
    assert gpu[TRACE_KEY] == npu[TRACE_KEY]
    assert gpu[TRACE_KEY]["encoded"] == gpu[TRACE_KEY]["collated"]
    for batch in (gpu, npu):
        positions = batch["text_position_ids"][0]
        starts = (positions == 0).nonzero().flatten().tolist()
        for start, signature in zip(starts, batch[TRACE_KEY]["collated"]):
            end = start + signature["length"]
            assert torch.equal(batch["position_ids"][:, 0, start:end], positions[start:end].expand(3, -1))


@pytest.mark.parametrize("tool_role", ["tool", "tool_response"])
def test_rejected_response_uses_the_same_raw_boundary_as_megatron(tokenizer, tool_role):
    messages = _conversation()
    for message in messages:
        if message["role"] == "tool":
            message["role"] = tool_role
    template = _template(tokenizer)
    template.set_mode("rlhf")
    encoded = template.encode({"messages": messages, "rejected_response": "bad-answer"})
    rejected = tokenizer.decode(encoded["rejected_input_ids"])
    assert ("new-observation" in rejected) is (tool_role == "tool")
    assert "bad-answer" in rejected


def test_tool_json_literals_match_new_swift_without_changing_qwen35(tokenizer):
    messages = [
        {"role": "user", "content": "query"},
        {"role": "tool_call", "content": {"name": "lookup", "arguments": {
            "yes": True, "no": False, "empty": None, "literal": "True",
            "nested": {"flag": True}, "items": [None, False],
        }}},
        {"role": "tool_response", "content": "result"},
        {"role": "assistant", "content": "answer"},
    ]
    text, _ = _texts(_template(tokenizer), messages)
    for name, value in (("yes", "true"), ("no", "false"), ("empty", "null"), ("literal", "True")):
        assert f"<parameter={name}>\n{value}\n</parameter>" in text
    assert '{"flag": true}' in text and '[null, false]' in text
    native, _ = _texts(_template(tokenizer, "qwen3_5"), messages)
    assert "<parameter=yes>\nTrue\n</parameter>" in native
    assert "<parameter=empty>\nNone\n</parameter>" in native
