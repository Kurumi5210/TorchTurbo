"""Check native Swift encoding and Turbo's model-input bridge without weights."""

from copy import deepcopy
import json

import pytest
import torch

pytest.importorskip("swift")

from swift.template.register import TEMPLATE_MAPPING
from swift.template.templates import qwen

from torchturbo.models.qwen3_5.swift_template import register


@pytest.fixture
def native_templates(monkeypatch):
    """Restore native classes even if another test activated the adapter."""
    templates = {}
    for name, class_name in (
        ("qwen3_5", "Qwen3_5Template"),
        ("qwen3_8", "Qwen3_8Template"),
    ):
        if name not in TEMPLATE_MAPPING:
            continue
        native = deepcopy(TEMPLATE_MAPPING[name])
        native.template_cls = vars(qwen)[class_name]
        monkeypatch.setitem(TEMPLATE_MAPPING, name, native)
        templates[name] = native
    return templates


@pytest.fixture(params=["qwen3_5", "qwen3_8"])
def native_template(request, native_templates):
    if request.param not in native_templates:
        pytest.skip(f"Installed Swift does not provide {request.param}.")
    return native_templates[request.param]


@pytest.fixture
def processor(monkeypatch):
    """Use a lossless byte tokenizer so native encoding needs no downloads."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PretrainedConfig, PreTrainedTokenizerFast
    from swift.model.model_meta import ModelInfo, ModelMeta

    special_tokens = [
        "<unk>", "<|endoftext|>", "<|im_start|>", "<|im_end|>",
        "<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>",
        "<think>", "</think>",
    ]
    tokens = special_tokens + sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(models.BPE({token: i for i, token in enumerate(tokens)}, []))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=special_tokens[2:],
        clean_up_tokenization_spaces=False,
        model_max_length=4096,
    )
    tokenizer.model_info = ModelInfo(
        model_type="qwen3_5", model_dir="qwen-template-test",
        torch_dtype=torch.float32, max_model_len=4096,
        quant_method=None, quant_bits=None, config=PretrainedConfig(),
        task_type="causal_lm", is_multimodal=True,
    )
    tokenizer.model_meta = ModelMeta(
        model_type="qwen3_5", model_groups=[], is_multimodal=True,
        task_type="causal_lm",
    )
    # Swift resolves this shared list to tokenizer-specific IDs in place.
    monkeypatch.setattr(qwen.Qwen2VLTemplate, "placeholder_tokens", [
        "<|image_pad|>", "<|video_pad|>",
    ])
    return tokenizer


def test_registration_preserves_native_metadata_and_is_idempotent(native_templates):
    register()
    wrapped_templates = {name: TEMPLATE_MAPPING[name] for name in native_templates}
    for name, native in native_templates.items():
        wrapped = wrapped_templates[name]
        assert wrapped is not native
        assert wrapped.template_cls is not native.template_cls
        assert issubclass(wrapped.template_cls, native.template_cls)
        metadata = deepcopy(wrapped)
        metadata.template_cls = native.template_cls
        assert vars(metadata) == vars(native)

    register()
    for name, wrapped in wrapped_templates.items():
        assert TEMPLATE_MAPPING[name] is wrapped


def test_older_swift_keeps_qwen3_5_without_inventing_qwen3_8(native_templates, monkeypatch):
    monkeypatch.delitem(TEMPLATE_MAPPING, "qwen3_8", raising=False)

    register()

    assert "qwen3_8" not in TEMPLATE_MAPPING
    assert issubclass(TEMPLATE_MAPPING["qwen3_5"].template_cls, qwen.Qwen3_5Template)
    assert TEMPLATE_MAPPING["qwen3_5"].template_cls is not qwen.Qwen3_5Template


@pytest.mark.parametrize("replace_input_ids", [False, True], ids=["dropped", "replaced"])
def test_native_hook_keeps_mtp_ids_and_merges_positions(
    native_template, monkeypatch, replace_input_ids,
):
    input_ids = torch.tensor([[10, 11, 12], [20, 21, 22]])
    replacement = input_ids + 1
    text_positions = torch.tensor([[0, 1, 2], [0, 1, 2]])
    rope_positions = torch.arange(18).reshape(3, 2, 3)
    embeddings = torch.randn(2, 3, 4)
    model = torch.nn.Identity()
    args = ("original",)
    calls = []

    def native_hook(self, received_model, received_args, kwargs):
        calls.append((self, received_model, received_args))
        result = dict(kwargs)
        result.pop("input_ids")
        result["inputs_embeds"] = embeddings
        if replace_input_ids:
            result["input_ids"] = replacement
        return ("native",), result

    monkeypatch.setattr(native_template.template_cls, "pre_forward_hook", native_hook)
    register()
    template = object.__new__(TEMPLATE_MAPPING[native_template.template_type].template_cls)

    returned_args, result = template.pre_forward_hook(model, args, {
        "input_ids": input_ids,
        "text_position_ids": text_positions,
        "position_ids": rope_positions,
    })

    assert calls == [(template, model, args)]
    assert returned_args == ("native",)
    assert result["input_ids"] is (replacement if replace_input_ids else input_ids)
    assert result["inputs_embeds"] is embeddings
    assert result["position_ids"].shape == (4, 2, 3)
    torch.testing.assert_close(result["position_ids"][0], text_positions)
    torch.testing.assert_close(result["position_ids"][1:], rope_positions)
    assert "text_position_ids" not in result


CONVERSATIONS = [
    [{"role": "user", "content": " question "},
     {"role": "assistant", "content": " answer "}],
    [{"role": "system", "content": " system text "},
     {"role": "user", "content": "question"},
     {"role": "assistant", "content": "<think> reasoning </think>answer"}],
    [{"role": "user", "content": "first question"},
     {"role": "assistant", "content": "<think>old reasoning</think>first answer"},
     {"role": "user", "content": "second question"},
     {"role": "assistant", "content": "second answer"}],
    [{"role": "user", "content": "lookup"},
     {"role": "assistant", "content": "<think>check the source</think>"},
     {"role": "tool_call", "content": json.dumps({"name": "lookup", "arguments": {"key": "value"}})},
     {"role": "tool", "content": " observation "},
     {"role": "assistant", "content": "final answer"}],
]


@pytest.mark.parametrize("messages", CONVERSATIONS, ids=["plain", "thinking", "history", "tool"])
@pytest.mark.parametrize("enable_thinking", [False, True])
def test_sft_encoding_and_labels_match_native(
    native_template, processor, messages, enable_thinking,
):
    native = native_template.template_cls(processor, native_template, enable_thinking=enable_thinking)
    native.set_mode("train")
    expected = native.encode({"messages": deepcopy(messages)})

    register()
    metadata = TEMPLATE_MAPPING[native_template.template_type]
    wrapped = metadata.template_cls(processor, metadata, enable_thinking=enable_thinking)
    wrapped.set_mode("train")
    actual = wrapped.encode({"messages": deepcopy(messages)})

    assert actual["input_ids"] == expected["input_ids"]
    assert actual["labels"] == expected["labels"]
    assert any(token != -100 for token in actual["labels"])


@pytest.mark.parametrize("enable_thinking,effort", [(False, "xhigh"), (True, "xhigh"), (True, "low"), (True, "medium")])
def test_qwen3_8_retains_reasoning_policy(native_templates, processor, enable_thinking, effort):
    if "qwen3_8" not in native_templates:
        pytest.skip("Installed Swift does not provide qwen3_8.")
    register()
    metadata = TEMPLATE_MAPPING["qwen3_8"]
    template = metadata.template_cls(processor, metadata, enable_thinking=enable_thinking)
    messages = deepcopy(CONVERSATIONS[2][:-1])

    encoded = template.encode({
        "messages": messages,
        "chat_template_kwargs": {"reasoning_effort": effort},
    })
    text = processor.decode(encoded["input_ids"])

    assert metadata.preserve_thinking is True
    assert "old reasoning" in text
    assert (f"Reasoning effort is set to {effort}." in text) is (enable_thinking and effort != "medium")
    if not enable_thinking:
        assert text.endswith("<think>\n\n</think>\n\n")
