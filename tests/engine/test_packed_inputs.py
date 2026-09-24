"""Packed SFT inputs are converted before backend preparation and model forward."""

from collections import UserDict
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from torchturbo.engine.core import TorchTurboEngine
from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.utils.flash_attn_utils import cu_seqlens_to_attention_mask


class _NoScalarReads(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        assert func != torch.ops.aten._local_scalar_dense.default
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("num_tokens", [5, 7])
def test_cu_mask_uses_true_lengths_without_scalar_reads(num_tokens):
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    with _NoScalarReads():
        mask = cu_seqlens_to_attention_mask(cu, num_tokens)
    assert mask.tolist() == [[1, 1, 2, 2, 2] + [0] * (num_tokens - 5)]


@pytest.mark.parametrize(("physical_length", "sp_size"), [(5, 1), (5, 2), (7, 4)])
@pytest.mark.parametrize("batch_type", [dict, UserDict])
def test_compute_loss_builds_mask_before_sp_and_moe(physical_length, sp_size, batch_type):
    calls = []
    expected_mask = [1, 1, 2, 2, 2] + [0] * ((-physical_length) % sp_size + physical_length - 5)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(pad_token_id=99)

        def forward(self, input_ids, position_ids, labels, attention_mask):
            calls.append("model")
            assert attention_mask.tolist() == [expected_mask]
            assert input_ids.shape == labels.shape == position_ids.shape == attention_mask.shape
            return torch.tensor(0.0)

    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend.get_sp_info = lambda: {"sp_size": sp_size}
    original_prepare = backend.prepare_inputs

    def prepare(model, inputs):
        calls.append("prepare")
        assert inputs["attention_mask"].tolist() == [[1, 1, 2, 2, 2] + [0] * (physical_length - 5)]
        return original_prepare(model, inputs)

    backend.prepare_inputs = prepare
    backend.compute_loss = lambda model, inputs, *args, **kwargs: model(**inputs)
    engine = TorchTurboEngine.__new__(TorchTurboEngine)
    engine.backend = backend

    def prepare_moe(model, inputs, gradient_accumulation_steps):
        calls.append("moe")
        assert inputs["attention_mask"].tolist() == [expected_mask]

    engine._prepare_moe_runtime = prepare_moe
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    batch = batch_type({
        "input_ids": torch.arange(physical_length).unsqueeze(0),
        "position_ids": torch.tensor([[0, 1, 0, 1, 2] + [0] * (physical_length - 5)]),
        "labels": torch.full((1, physical_length), -100),
        "cu_seq_lens_q": cu,
        "cu_seq_lens_k": cu,
        "max_length_q": 3,
        "max_length_k": 3,
    })

    engine.compute_loss(Model().eval(), batch)

    assert calls == ["prepare", "moe", "model"]
    assert "attention_mask" not in batch
    assert batch["cu_seq_lens_q"].tolist() == [0, 2, 5]


def test_nonpacked_native_mask_and_kwargs_are_preserved():
    batch = {
        "attention_mask": torch.tensor([[1, 1, 0]]),
        "cu_seq_lens_q": torch.tensor([0, 2], dtype=torch.int32),
    }
    engine = TorchTurboEngine.__new__(TorchTurboEngine)
    engine.backend = SimpleNamespace(
        prepare_inputs=lambda model, inputs: inputs,
        compute_loss=lambda model, inputs, *args, **kwargs: inputs,
    )
    engine._prepare_moe_runtime = lambda *args: None

    assert engine.compute_loss(torch.nn.Identity().eval(), batch) is batch
