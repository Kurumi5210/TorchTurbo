from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

import torch
import pytest

from torchturbo.adapters.llamafactory_v0.patches import (
    AcceleratorPatch,
    PackedFA2CollatorPatch,
)
from torchturbo.models.qwen3_5_moe.lf_template import Qwen35DummyMediaPatch


def _tokenizer(pad_token_id: int = 99):
    return SimpleNamespace(
        pad_token_id=pad_token_id,
        padding_side="right",
        convert_tokens_to_ids=lambda _token: -1,
    )


def test_accelerator_patch_keeps_engine_optimizer_state_placement(monkeypatch):
    calls = []

    class Accelerator:
        def prepare(self, *args, device_placement=None):
            return args

        def prepare_model(self, model, device_placement=None, evaluation_mode=False):
            return model

        def prepare_optimizer(self, optimizer, device_placement=None):
            calls.append((optimizer, device_placement))
            return optimizer

        def clip_grad_norm_(self, parameters, max_norm, norm_type=2):
            return None

        def backward(self, loss, **kwargs):
            return None

    accelerate = ModuleType("accelerate")
    accelerate.Accelerator = Accelerator
    monkeypatch.setitem(sys.modules, "accelerate", accelerate)

    raw_optimizer = object()
    engine = SimpleNamespace(
        optimizer=raw_optimizer,
        backend=SimpleNamespace(clip_grad_norm_=lambda *_args: None),
    )
    patch = AcceleratorPatch()
    patch.patch(engine)
    try:
        accelerator = Accelerator()
        assert accelerator.prepare_optimizer(raw_optimizer) is raw_optimizer
        assert calls == [(raw_optimizer, False)]
    finally:
        patch.revert()


def test_engine_patch_routes_lf095_packed_mask_through_transformers_varlen(
    monkeypatch,
):
    import transformers.integrations.flash_attention as flash_integration
    import transformers.modeling_flash_attention_utils as fa_utils

    from torchturbo.engine import TorchTurboEngine
    from torchturbo.utils.flash_attn_utils import get_unpad_data

    attention_mask = torch.tensor([[1, 1, 2, 2, 2, 0, 0, 0]])

    # Verify the engine installs packing support for sequence-ID masks.
    monkeypatch.setattr(
        flash_integration, "_flash_attention_forward", flash_integration._flash_attention_forward,
    )
    monkeypatch.setattr(fa_utils, "_get_unpad_data", lambda _mask: None)
    engine = object.__new__(TorchTurboEngine)
    engine._patch_transformers()
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
        compute_dtype = torch.float32

        def __call__(self, features):
            self.rope_seen_by_lf = self.get_rope_func
            self.plugin_tokens_seen_by_lf = (
                self.template.mm_plugin.image_token,
                self.template.mm_plugin.audio_token,
            )
            for feature in features:
                feature.pop("packing_params", None)
            fake_input_ids = self.__dict__.get("fake_input_ids", [])
            if self.tokenizer.padding_side == "right":
                features[0]["input_ids"] += fake_input_ids
            else:
                features[0]["input_ids"] = fake_input_ids + features[0]["input_ids"]
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
                assert result["input_ids"].shape[0] == 1
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


def test_lf095_path_preserves_native_mrope_and_compacts_packed_tokens(monkeypatch):
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

    result = collator([{
        "input_ids": list(range(8)), "images": None, "videos": None, "audios": None,
        "packing_params": {"sequence_boundaries": [0, 2, 5, 8]},
    }])

    assert collator.rope_seen_by_lf is rope_func
    assert collator.plugin_tokens_seen_by_lf == (None, None)
    assert plugin.image_token == "<image>"
    assert plugin.audio_token == "<audio>"
    assert module.MROPE_MODELS == {"qwen3_5", "qwen3_5_moe"}
    assert result["input_ids"].tolist() == [[0, 1, 2, 3, 4]]
    torch.testing.assert_close(result["position_ids"], position_ids[:, :, :5])
    assert result["labels"].tolist() == [[-100, 1, 2, 3, 4]]
    assert result["token_type_ids"].shape == (1, 5)
    assert "attention_mask" not in result
    assert result["cu_seq_lens_q"].tolist() == result["cu_seq_lens_k"].tolist() == [0, 2, 5]
    assert result["max_length_q"] == result["max_length_k"] == 3
    assert collator.neat_packing is True

    packed_patch.revert()
    qwen_patch.revert()
    assert parent_cls.__call__ is original_parent_call
    assert sft_cls.__call__ is original_sft_call


def test_lf095_kimi_packed_fa2_uses_cu_metadata(monkeypatch):
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

    result = collator([{
        "input_ids": list(range(8)), "packing_params": {"sequence_boundaries": [0, 2, 5, 8]},
    }])

    assert result["input_ids"].tolist() == [[0, 1, 2, 3, 4]]
    assert result["labels"].tolist() == [[-100, 1, 2, 3, 4]]
    assert result["position_ids"].tolist() == [[0, 1, 0, 1, 2]]
    assert "attention_mask" not in result
    assert result["cu_seq_lens_q"].tolist() == [0, 2, 5]
    assert result["cu_seq_lens_q"].dtype == torch.int32
    assert collator.neat_packing is True
    patch.revert()


@pytest.mark.parametrize("neat_packing,attn_implementation", [(True, "eager"), (False, "flash_attention_2")])
def test_lf095_other_collator_paths_are_unchanged(monkeypatch, neat_packing, attn_implementation):
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
    collator.neat_packing = neat_packing
    collator.attn_implementation = attn_implementation
    collator.parent_batch = {
        "input_ids": torch.arange(4).reshape(1, -1),
        "labels": torch.arange(4).reshape(1, -1),
        "position_ids": torch.tensor([[0, 1, 0, 1]]),
        "attention_mask": torch.tensor([[1, 1, 2, 2]]),
    }

    result = collator([{"input_ids": list(range(4))}])

    assert result["input_ids"].shape == (1, 4)
    assert result["attention_mask"].tolist() == [[1, 1, 2, 2]]
    assert "cu_seq_lens_q" not in result
    patch.revert()


def test_lf095_packed_fa2_compacts_other_models(monkeypatch):
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

    result = collator([{
        "input_ids": list(range(8)), "packing_params": {"sequence_boundaries": [0, 2, 5, 8]},
    }])

    assert result["input_ids"].shape == (1, 5)
    assert result["position_ids"].shape == (1, 5)
    assert "attention_mask" not in result
    assert result["cu_seq_lens_q"].tolist() == [0, 2, 5]
    assert collator.neat_packing is True
    patch.revert()


def _packed_collator(monkeypatch):
    _, _, sft_cls = _install_fake_llamafactory(monkeypatch)
    patch = PackedFA2CollatorPatch()
    patch.patch()
    collator = sft_cls()
    collator.tokenizer = _tokenizer()
    collator.template = SimpleNamespace(
        mm_plugin=SimpleNamespace(image_token="<image>", audio_token="<audio>")
    )
    collator.get_rope_func = None
    collator.neat_packing = True
    collator.attn_implementation = "flash_attention_2"
    return collator, patch


def test_lf095_neat_config_keeps_unpacked_generation_batches(monkeypatch):
    collator, patch = _packed_collator(monkeypatch)
    collator.tokenizer.padding_side = "left"
    collator.compute_dtype = torch.bfloat16
    collator.parent_batch = {
        "input_ids": torch.tensor([[99, 99, 99, 10], [99, 20, 21, 22]]),
        "labels": torch.tensor([[-100, -100, -100, -100, 11, 12], [-100, -100, 23, 24, 25, 26]]),
        "attention_mask": torch.tensor([[0, 0, 0, 1], [0, 1, 1, 1]]),
        "pixel_values": torch.tensor([[1.0, 2.0]]),
    }
    try:
        result = collator([
            {"input_ids": [10], "labels": [11, 12]},
            {"input_ids": [20, 21, 22], "labels": [23, 24, 25, 26]},
        ])
    finally:
        patch.revert()

    for key in ("input_ids", "labels", "attention_mask"):
        torch.testing.assert_close(result[key], collator.parent_batch[key])
    assert result["pixel_values"].dtype == torch.bfloat16
    assert "position_ids" not in result
    assert "cu_seq_lens_q" not in result


def test_lf095_packed_fa2_flattens_rows_without_cross_row_loss(monkeypatch):
    collator, patch = _packed_collator(monkeypatch)
    collator.compute_dtype = torch.bfloat16
    positions = torch.arange(40).reshape(4, 2, 5)
    collator.parent_batch = {
        "input_ids": torch.tensor([[10, 11, 12, 99, 99], [20, 21, 22, 23, 99]]),
        "labels": torch.tensor([[10, 11, 12, -100, -100], [20, 21, 22, 23, -100]]),
        "attention_mask": torch.tensor([[1, 1, 2, 0, 0], [1, 2, 2, 2, 0]]),
        "position_ids": positions,
        "token_type_ids": torch.arange(10).reshape(2, 5),
        "mm_token_type_ids": torch.arange(10).reshape(2, 5),
        "cross_attention_mask": torch.arange(10).reshape(2, 5, 1, 1),
        "pixel_values": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "image_grid_thw": torch.tensor([[1, 2, 3], [1, 4, 5]]),
    }
    try:
        result = collator([
            {"input_ids": [10, 11, 12, 99, 99], "packing_params": {"sequence_boundaries": [0, 2, 3, 5]}},
            {"input_ids": [20, 21, 22, 23, 99], "packing_params": {"sequence_boundaries": [0, 1, 4, 5]}},
        ])
    finally:
        patch.revert()

    assert result["input_ids"].tolist() == [[10, 11, 12, 20, 21, 22, 23]]
    # Intra-row sample starts keep LF's labels; only original row starts change.
    assert result["labels"].tolist() == [[-100, 11, 12, -100, 21, 22, 23]]
    assert result["cu_seq_lens_q"].tolist() == [0, 2, 3, 4, 7]
    assert result["max_length_q"] == result["max_length_k"] == 3
    expected_positions = torch.cat([positions[:, :1, :3], positions[:, 1:, :4]], dim=-1)
    torch.testing.assert_close(result["position_ids"], expected_positions)
    for key in ("token_type_ids", "mm_token_type_ids"):
        assert result[key].tolist() == [[0, 1, 2, 5, 6, 7, 8]]
    assert result["cross_attention_mask"].flatten().tolist() == [0, 1, 2, 5, 6, 7, 8]
    assert result["pixel_values"].dtype == torch.bfloat16
    assert result["pixel_values"].tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert result["image_grid_thw"].tolist() == [[1, 2, 3], [1, 4, 5]]
    assert "attention_mask" not in result

    logits = torch.randn(2, 5, 32, generator=torch.Generator().manual_seed(0))
    original_loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 32),
        collator.parent_batch["labels"][:, 1:].reshape(-1),
        reduction="sum",
    )
    packed_logits = logits.flatten(0, 1)[[0, 1, 2, 5, 6, 7, 8]]
    packed_loss = torch.nn.functional.cross_entropy(
        packed_logits[:-1], result["labels"][0, 1:], reduction="sum",
    )
    torch.testing.assert_close(packed_loss, original_loss)


@pytest.mark.parametrize("mrope", [False, True])
def test_lf095_dummy_media_stays_at_masked_tail(monkeypatch, mrope):
    collator, patch = _packed_collator(monkeypatch)
    collator.fake_input_ids = [70, 71]
    attention_mask = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0]])
    positions = torch.tensor([[0, 1, 0, 0], [0, 1, 2, 0]])
    if mrope:
        positions = torch.arange(64).reshape(4, 2, 8)
        # LF appends an extra zero mask tail in its dummy-image mRoPE branch.
        attention_mask = torch.nn.functional.pad(attention_mask, (0, 2))
    collator.parent_batch = {
        "input_ids": torch.tensor([[10, 11, 99, 99, 70, 71, 99, 99], [20, 21, 22, 99, 99, 99, 99, 99]]),
        "labels": torch.tensor([[10, 11, -100, -100, -100, -100, -100, -100], [20, 21, 22, -100, -100, -100, -100, -100]]),
        "attention_mask": attention_mask,
        "position_ids": positions,
        "has_dummy_image": mrope,
        "pixel_values": torch.tensor([[7.0, 8.0]]),
    }
    try:
        result = collator([
            {"input_ids": [10, 11, 99, 99], "packing_params": {"sequence_boundaries": [0, 2, 4]}},
            {"input_ids": [20, 21, 22, 99], "packing_params": {"sequence_boundaries": [0, 3, 4]}},
        ])
    finally:
        patch.revert()

    assert result["input_ids"].tolist() == [[10, 11, 20, 21, 22, 70, 71]]
    assert result["labels"].tolist() == [[-100, 11, -100, 21, 22, -100, -100]]
    assert result["cu_seq_lens_q"].tolist() == [0, 2, 5]
    assert result["max_length_q"] == 3
    if mrope:
        expected = torch.cat([positions[:, :1, :2], positions[:, 1:, :3], positions[:, :1, 4:6]], dim=-1)
        torch.testing.assert_close(result["position_ids"], expected)
    else:
        assert result["position_ids"].tolist() == [[0, 1, 0, 1, 2, 0, 0]]
    assert result["pixel_values"].tolist() == [[7.0, 8.0]]
    assert "has_dummy_image" not in result
    assert "attention_mask" not in result


def test_lf095_mismatched_text_positions_rejected(monkeypatch):
    collator, patch = _packed_collator(monkeypatch)
    collator.parent_batch = {
        "input_ids": torch.tensor([[10, 11, 12, 99]]),
        "labels": torch.tensor([[10, 11, 12, -100]]),
        "attention_mask": torch.tensor([[1, 1, 2, 0]]),
        "position_ids": torch.tensor([[0, 1, 2, 0]]),
    }
    try:
        with pytest.raises(ValueError, match="position resets"):
            collator([{
                "input_ids": [10, 11, 12, 99], "packing_params": {"sequence_boundaries": [0, 2, 3, 4]},
            }])
    finally:
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
