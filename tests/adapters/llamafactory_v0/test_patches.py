from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

import torch

from torchturbo.adapters.llamafactory_v0.patches import PackedFA2CollatorPatch
from torchturbo.models.qwen3_5_moe.lf_template import Qwen35DummyMediaPatch


def _tokenizer(pad_token_id: int = 99):
    return SimpleNamespace(
        pad_token_id=pad_token_id,
        convert_tokens_to_ids=lambda _token: -1,
    )


def test_fsdp2_patch_routes_lf095_packed_mask_through_transformers_varlen(
    monkeypatch,
):
    import transformers.modeling_flash_attention_utils as fa_utils

    from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
    from torchturbo.utils.flash_attn_utils import get_unpad_data

    attention_mask = torch.tensor([[1, 1, 2, 2, 2, 0, 0, 0]])

    # Ensure this test observes the assignment made by FSDP2 rather than a
    # replacement left behind by another test or backend instance.
    monkeypatch.setattr(fa_utils, "_get_unpad_data", lambda _mask: None)
    FSDP2Backend._patch_transformers(None)
    assert fa_utils._get_unpad_data is get_unpad_data

    states = torch.arange(64, dtype=torch.float32).reshape(1, 8, 2, 4)

    def unexpected_unpad(*_args, **_kwargs):
        raise AssertionError("query/key lengths match; fallback unpad must not run")

    unpacked = fa_utils._upad_input(
        states,
        states,
        states,
        attention_mask,
        query_length=8,
        unpad_input_func=unexpected_unpad,
    )
    query, key, value, indices, (cu_q, cu_k), (max_q, max_k) = unpacked

    assert query.shape == key.shape == value.shape == (5, 2, 4)
    assert indices.tolist() == [0, 1, 2, 3, 4]
    assert cu_q.tolist() == cu_k.tolist() == [0, 2, 5]
    assert int(max_q) == int(max_k) == 3

def _install_fake_llamafactory(monkeypatch):
    collator_module = ModuleType("llamafactory.data.collator")

    class MultiModalDataCollatorForSeq2Seq:
        def __call__(self, features):
            self.rope_seen_by_lf = self.get_rope_func
            self.plugin_tokens_seen_by_lf = (
                self.template.mm_plugin.image_token,
                self.template.mm_plugin.audio_token,
            )
            return {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in self.parent_batch.items()
            }

    class SFTDataCollatorWith4DAttentionMask(MultiModalDataCollatorForSeq2Seq):
        @staticmethod
        def _unpad_packed_features(result):
            mask = result["attention_mask"]
            valid = torch.nonzero(mask[0] != 0, as_tuple=False).flatten()
            seq_len = mask.size(-1)
            for key, value in list(result.items()):
                if not torch.is_tensor(value):
                    continue
                is_sequence_field = (
                    key == "position_ids" or value.dim() == 2
                ) and value.size(-1) == seq_len
                if is_sequence_field:
                    result[key] = value.index_select(-1, valid)

        def __call__(self, features):
            result = super().__call__(features)
            if self.neat_packing and self.attn_implementation == "flash_attention_2":
                self._unpad_packed_features(result)
                result["attention_mask"] = None
            return result

    collator_module.MultiModalDataCollatorForSeq2Seq = MultiModalDataCollatorForSeq2Seq
    collator_module.SFTDataCollatorWith4DAttentionMask = (
        SFTDataCollatorWith4DAttentionMask
    )
    collator_module.MROPE_MODELS = {"qwen3_5", "qwen3_5_moe"}

    lf_module = ModuleType("llamafactory")
    lf_module.__path__ = []
    data_module = ModuleType("llamafactory.data")
    data_module.__path__ = []
    data_module.collator = collator_module
    lf_module.data = data_module
    monkeypatch.setitem(sys.modules, "llamafactory", lf_module)
    monkeypatch.setitem(sys.modules, "llamafactory.data", data_module)
    monkeypatch.setitem(sys.modules, "llamafactory.data.collator", collator_module)
    return (
        collator_module,
        MultiModalDataCollatorForSeq2Seq,
        SFTDataCollatorWith4DAttentionMask,
    )


def test_lf095_path_preserves_native_mrope_padding_and_mask(monkeypatch):
    module, parent_cls, sft_cls = _install_fake_llamafactory(monkeypatch)
    original_parent_call = parent_cls.__call__
    original_sft_call = sft_cls.__call__
    qwen_patch = Qwen35DummyMediaPatch()
    packed_patch = PackedFA2CollatorPatch()
    qwen_patch.patch()
    packed_patch.patch()

    rope_func = object()
    plugin = SimpleNamespace(image_token="<image>", audio_token="<audio>")
    collator = sft_cls()
    collator.model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5_moe"))
    collator.tokenizer = _tokenizer()
    collator.template = SimpleNamespace(mm_plugin=plugin)
    collator.get_rope_func = rope_func
    collator.block_diag_attn = True
    collator.neat_packing = True
    collator.attn_implementation = "flash_attention_2"
    position_ids = torch.zeros(4, 1, 8, dtype=torch.long)
    position_ids[0, 0, :5] = torch.tensor([0, 1, 0, 1, 2])
    collator.parent_batch = {
        "input_ids": torch.arange(8).reshape(1, -1),
        "labels": torch.tensor([[0, 1, 2, 3, 4, -100, -100, -100]]),
        "position_ids": position_ids,
        "token_type_ids": torch.ones(1, 8, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 2, 2, 2, 0, 0, 0]]),
    }

    result = collator([{"images": None, "videos": None, "audios": None}])

    assert collator.rope_seen_by_lf is rope_func
    assert collator.plugin_tokens_seen_by_lf == (None, None)
    assert plugin.image_token == "<image>"
    assert plugin.audio_token == "<audio>"
    assert module.MROPE_MODELS == {"qwen3_5", "qwen3_5_moe"}
    assert result["input_ids"].shape == (1, 8)
    assert result["position_ids"].shape == (4, 1, 8)
    assert result["labels"][0, 5:].tolist() == [-100, -100, -100]
    assert result["attention_mask"].tolist() == [[1, 1, 2, 2, 2, 0, 0, 0]]
    assert collator.neat_packing is True

    packed_patch.revert()
    qwen_patch.revert()
    assert parent_cls.__call__ is original_parent_call
    assert sft_cls.__call__ is original_sft_call


def test_lf095_kimi_packed_fa2_preserves_mask(monkeypatch):
    _, _, sft_cls = _install_fake_llamafactory(monkeypatch)
    patch = PackedFA2CollatorPatch()
    patch.patch()

    collator = sft_cls()
    collator.model = SimpleNamespace(config=SimpleNamespace(model_type="kimi_k25"))
    collator.tokenizer = _tokenizer()
    collator.template = SimpleNamespace(
        mm_plugin=SimpleNamespace(image_token=None, audio_token=None)
    )
    collator.get_rope_func = None
    collator.block_diag_attn = True
    collator.neat_packing = True
    collator.attn_implementation = "flash_attention_2"
    collator.parent_batch = {
        "input_ids": torch.arange(8).reshape(1, -1),
        "labels": torch.tensor([[0, 1, 2, 3, 4, -100, -100, -100]]),
        "position_ids": torch.tensor([[0, 1, 0, 1, 2, 0, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 2, 2, 2, 0, 0, 0]]),
    }

    result = collator([{}])

    assert result["input_ids"].shape == (1, 8)
    assert result["labels"][0, 5:].tolist() == [-100, -100, -100]
    assert result["position_ids"].tolist() == [[0, 1, 0, 1, 2, 0, 0, 0]]
    assert result["attention_mask"].tolist() == [[1, 1, 2, 2, 2, 0, 0, 0]]
    assert collator.neat_packing is True
    patch.revert()


def test_lf095_non_fa2_batch_is_unchanged(monkeypatch):
    _, _, sft_cls = _install_fake_llamafactory(monkeypatch)
    patch = PackedFA2CollatorPatch()
    patch.patch()

    collator = sft_cls()
    collator.model = SimpleNamespace(config=SimpleNamespace(model_type="bailing"))
    collator.tokenizer = _tokenizer()
    collator.template = SimpleNamespace(
        mm_plugin=SimpleNamespace(image_token=None, audio_token=None)
    )
    collator.get_rope_func = None
    collator.block_diag_attn = True
    collator.neat_packing = True
    collator.attn_implementation = "eager"
    collator.parent_batch = {
        "input_ids": torch.arange(4).reshape(1, -1),
        "labels": torch.arange(4).reshape(1, -1),
        "position_ids": torch.tensor([[0, 1, 0, 1]]),
        "attention_mask": torch.tensor([[1, 1, 2, 2]]),
    }

    result = collator([{}])

    assert result["input_ids"].shape == (1, 4)
    assert result["attention_mask"].tolist() == [[1, 1, 2, 2]]
    patch.revert()


def test_lf095_packed_fa2_preserves_padding_for_other_models(monkeypatch):
    _, _, sft_cls = _install_fake_llamafactory(monkeypatch)
    patch = PackedFA2CollatorPatch()
    patch.patch()

    collator = sft_cls()
    collator.model = SimpleNamespace(config=SimpleNamespace(model_type="llada2_moe"))
    collator.tokenizer = _tokenizer()
    collator.template = SimpleNamespace(
        mm_plugin=SimpleNamespace(image_token=None, audio_token=None)
    )
    collator.get_rope_func = None
    collator.block_diag_attn = True
    collator.neat_packing = True
    collator.attn_implementation = "flash_attention_2"
    collator.parent_batch = {
        "input_ids": torch.arange(8).reshape(1, -1),
        "labels": torch.tensor([[0, 1, 2, 3, 4, -100, -100, -100]]),
        "position_ids": torch.tensor([[0, 1, 0, 1, 2, 0, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 2, 2, 2, 0, 0, 0]]),
    }

    result = collator([{}])

    assert result["input_ids"].shape == (1, 8)
    assert result["labels"][0, 5:].tolist() == [-100, -100, -100]
    assert result["position_ids"].shape == (1, 8)
    assert result["attention_mask"].tolist() == [[1, 1, 2, 2, 2, 0, 0, 0]]
    assert collator.neat_packing is True
    patch.revert()


def test_adapter_discovers_and_registers_model_lf_patches(monkeypatch):
    from torchturbo.adapters.llamafactory_v0 import LlamaFactoryV0Adapter
    from torchturbo.models import registry
    from torchturbo.registry.adapter import BaseAdapter

    monkeypatch.setattr(registry, "get_all_models", lambda: {"qwen3_5_moe": object})

    adapter = object.__new__(LlamaFactoryV0Adapter)
    BaseAdapter.__init__(adapter)

    adapter._register_lf_templates()
    assert [p.name for p in adapter._patches] == ["qwen3_5_dummy_media"]
