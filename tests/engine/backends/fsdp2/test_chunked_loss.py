"""Chunked-loss model contract tests for the FSDP2 backend."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.config import LossChunkConfig
from torchturbo.models.base import SequenceParallelInputMode, TurboModel
from torchturbo.ops import get_or_create_ops
from torchturbo.runtime.cpu import CpuRuntime


@pytest.fixture(autouse=True)
def _process_ops(monkeypatch):
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    get_or_create_ops(device_runtime=CpuRuntime())
    yield


def _backend(*, enabled=True, mtp=False, scaling=0.2):
    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend.rank = 0
    backend.config = SimpleNamespace(
        loss_chunk=LossChunkConfig(enabled=enabled),
        mtp=SimpleNamespace(
            enabled=mtp,
            loss_scaling_factor=scaling,
            log_loss=False,
        ),
    )
    backend.get_sp_info = lambda: {
        "sp_size": 1,
        "sp_group": None,
        "sp_rank": 0,
    }
    return backend


def _batch():
    input_ids = torch.tensor([[1, 2, 3, 4]])
    return {"input_ids": input_ids, "labels": input_ids.clone()}


class _CountingLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0

    def forward(self, hidden_states):
        self.calls += 1
        return super().forward(hidden_states)


@dataclass
class _ContractOutput(ModelOutput):
    logits: torch.Tensor | None = None
    mtp_logits: list[torch.Tensor] | None = None


class _ContractModel(TurboModel, nn.Module):
    def __init__(
        self,
        *,
        fail_forward=False,
        mtp=False,
        upcast_logits=False,
        dtype=torch.float32,
    ):
        super().__init__()
        torch.manual_seed(3)
        self.hidden = nn.Parameter(torch.randn(1, 4, 3, dtype=dtype))
        self.lm_head = _CountingLinear(3, 7, bias=False, dtype=dtype)
        self.fail_forward = fail_forward
        self.with_mtp = mtp
        self.upcast_logits = upcast_logits
        self.forward_calls = []

    def get_sequence_parallel_input_mode(self):
        return SequenceParallelInputMode.POST_EMBEDDING

    def get_lm_head(self):
        return self.lm_head

    def forward(
        self,
        input_ids,
        position_ids,
        use_cache,
        return_dict=None,
        logits_to_keep=None,
    ):
        self.forward_calls.append((return_dict, logits_to_keep))
        hidden = self.hidden.expand(input_ids.shape[0], -1, -1)
        logits = self.lm_head(hidden)
        if self.upcast_logits:
            logits = logits.float()
        if self.fail_forward:
            raise RuntimeError("model forward failed")
        mtp_hidden_states = [hidden * 0.5] if self.with_mtp else None
        return _ContractOutput(
            logits=logits,
            mtp_logits=(
                [
                    self.lm_head(state).float()
                    if self.upcast_logits
                    else self.lm_head(state)
                    for state in mtp_hidden_states
                ]
                if mtp_hidden_states
                else None
            ),
        )


def test_chunked_loss_restores_lm_head_forward_after_model_error():
    model = _ContractModel(fail_forward=True)
    real_lm_head = model.lm_head

    assert "lm_head" not in vars(model)
    with pytest.raises(RuntimeError, match="model forward failed"):
        _backend().compute_loss(model, _batch())

    assert "lm_head" not in vars(model)
    assert model._modules["lm_head"] is real_lm_head
    assert model.lm_head is real_lm_head
    assert model.lm_head(model.hidden).shape == (1, 4, 7)


def test_chunked_loss_bypass_preserves_lm_head_registration():
    class InspectModel(_ContractModel):
        def forward(self, *args, **kwargs):
            assert self._modules["lm_head"] is real_lm_head
            assert isinstance(self.lm_head, nn.Identity)
            assert (
                tuple(name for name, _ in self.named_parameters())
                == parameter_names
            )
            assert tuple(self.state_dict()) == state_dict_keys
            return super().forward(*args, **kwargs)

    model = InspectModel()
    backend = _backend()
    real_lm_head = model.lm_head
    parameter_names = tuple(name for name, _ in model.named_parameters())
    state_dict_keys = tuple(model.state_dict())

    outputs, hidden_states = backend._forward_for_lm_loss(
        model,
        {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "position_ids": torch.arange(4).unsqueeze(0),
            "use_cache": False,
        },
        use_chunked_loss=True,
    )

    assert outputs.logits is None
    assert len(hidden_states) == 1
    assert hidden_states[0] is model.hidden
    assert "lm_head" not in vars(model)
    assert model._modules["lm_head"] is real_lm_head
    assert model.lm_head is real_lm_head


def test_chunked_loss_noops_nested_lm_head():
    class NestedModel(TurboModel, nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = nn.Module()
            self.language_model.lm_head = nn.Linear(3, 7, bias=False)

        def get_lm_head(self):
            return self.language_model.lm_head

        def forward(self, hidden_states, return_dict=None):
            return _ContractOutput(
                logits=self.language_model.lm_head(hidden_states)
            )

    model = NestedModel()
    hidden_states = torch.randn(1, 4, 3)

    outputs, captured = _backend()._forward_for_lm_loss(
        model,
        {"hidden_states": hidden_states},
        use_chunked_loss=True,
    )

    assert outputs.logits is None
    assert len(captured) == 1
    assert captured[0] is hidden_states


def test_chunked_loss_uses_head_input_before_logits_upcast():
    model = _ContractModel(upcast_logits=True, dtype=torch.bfloat16)

    total, outputs = _backend().compute_loss(
        model, _batch(), return_outputs=True
    )

    labels = _batch()["labels"]
    shifted_labels = F.pad(labels, (0, 1), value=-100)[..., 1:]
    reference = F.cross_entropy(
        model.lm_head(model.hidden).float().flatten(0, 1),
        shifted_labels.flatten(),
        reduction="sum",
    ) / (shifted_labels != -100).sum()

    assert outputs.logits is None
    assert torch.allclose(total, reference)


def test_chunked_loss_rejects_diffusion_training():
    with pytest.raises(NotImplementedError, match="diffusion training"):
        _backend().compute_diffusion_loss(None, {})


def test_eval_uses_regular_logits_path():
    model = _ContractModel()
    model.eval()

    loss, outputs = _backend().compute_loss(model, _batch(), return_outputs=True)

    assert loss.ndim == 0
    assert outputs.logits is not None
    assert model.forward_calls == [(None, None)]
    assert model.lm_head.calls == 1


def test_main_and_mtp_losses_are_aggregated_on_chunked_path():
    model = _ContractModel(mtp=True)
    backend = _backend(mtp=True, scaling=0.2)

    total, outputs = backend.compute_loss(model, _batch(), return_outputs=True)
    assert model.forward_calls == [(True, None)]
    assert model.lm_head.calls == 0
    assert outputs.logits is None
    assert outputs.mtp_logits is None

    labels = _batch()["labels"]
    main_labels = F.pad(labels, (0, 1), value=-100)[..., 1:]
    mtp_labels = F.pad(labels, (0, 2), value=-100)[..., 2:]
    main_ref = F.cross_entropy(
        model.lm_head(model.hidden).float().flatten(0, 1),
        main_labels.flatten(),
        reduction="sum",
    ) / (main_labels != -100).sum()
    mtp_ref = F.cross_entropy(
        model.lm_head(model.hidden * 0.5).float().flatten(0, 1),
        mtp_labels.flatten(),
        reduction="sum",
    ) / (mtp_labels != -100).sum()

    assert len(outputs.mtp_loss) == 1
    assert torch.allclose(outputs.main_loss, main_ref)
    assert torch.allclose(outputs.mtp_loss[0], mtp_ref)
    assert torch.allclose(total, main_ref + mtp_ref * 0.2)

    total.backward()
    assert model.hidden.grad is not None
    assert model.lm_head.weight.grad is not None


def test_main_and_mtp_losses_are_aggregated_on_regular_path():
    model = _ContractModel(mtp=True)
    model.eval()
    backend = _backend(mtp=True, scaling=0.2)

    total, outputs = backend.compute_loss(model, _batch(), return_outputs=True)

    labels = _batch()["labels"]
    main_labels = F.pad(labels, (0, 1), value=-100)[..., 1:]
    mtp_labels = F.pad(labels, (0, 2), value=-100)[..., 2:]
    main_ref = F.cross_entropy(
        model.lm_head(model.hidden).float().flatten(0, 1),
        main_labels.flatten(),
        reduction="sum",
    ) / (main_labels != -100).sum()
    mtp_ref = F.cross_entropy(
        model.lm_head(model.hidden * 0.5).float().flatten(0, 1),
        mtp_labels.flatten(),
        reduction="sum",
    ) / (mtp_labels != -100).sum()

    assert model.forward_calls == [(None, None)]
    assert outputs.logits is not None
    assert len(outputs.mtp_loss) == 1
    assert torch.allclose(outputs.main_loss, main_ref)
    assert torch.allclose(outputs.mtp_loss[0], mtp_ref)
    assert torch.allclose(total, main_ref + mtp_ref * 0.2)

    total.backward()
    assert model.hidden.grad is not None
    assert model.lm_head.weight.grad is not None


def test_zero_mtp_scale_keeps_diagnostics_but_excludes_mtp_from_total():
    model = _ContractModel(mtp=True)

    total, outputs = _backend(mtp=True, scaling=0.0).compute_loss(
        model, _batch(), return_outputs=True
    )

    assert len(outputs.mtp_loss) == 1
    assert outputs.mtp_loss_scaled.item() == 0.0
    assert torch.allclose(total, outputs.main_loss)


def test_chunk_loss_lm_head_detached_fallback_receives_positional_input(monkeypatch):
    import torchturbo.engine.backends.fsdp2.backend as backend_module

    class FakeFSDPLinear(nn.Linear):
        def set_reshard_after_forward(self, value, recurse=False):
            self.reshard_after_forward = value

    lm_head = FakeFSDPLinear(3, 7, bias=False)
    model = SimpleNamespace(
        get_lm_head=lambda: lm_head,
    )
    reference = torch.randn(1, 4, 3)
    calls = []

    def fake_register(module, method_name):
        original = getattr(module, method_name)

        def wrapped(*args, **kwargs):
            calls.append(args)
            return original(*args, **kwargs)

        setattr(module, method_name, wrapped)

    monkeypatch.setattr(backend_module, "FSDPModule", FakeFSDPLinear)
    monkeypatch.setattr(
        backend_module, "register_fsdp_forward_method", fake_register
    )

    weight, prepared_reference, anchor = _backend()._prepare_chunk_loss_lm_head(
        model, reference
    )

    assert calls == [(reference,)]
    assert weight is lm_head.weight
    assert prepared_reference is reference
    assert anchor.shape == ()
    assert anchor.requires_grad is lm_head.weight.requires_grad


def test_chunk_loss_lm_head_differentiable_input_uses_no_fallback_anchor(monkeypatch):
    import torchturbo.engine.backends.fsdp2.backend as backend_module

    class FakeFSDPLinear(nn.Linear):
        def set_reshard_after_forward(self, value, recurse=False):
            self.reshard_after_forward = value

    lm_head = FakeFSDPLinear(3, 7, bias=False)
    model = SimpleNamespace(
        get_lm_head=lambda: lm_head,
    )
    reference = torch.randn(1, 4, 3, requires_grad=True)
    calls = []

    def fake_register(module, method_name):
        original = getattr(module, method_name)

        def wrapped(*args, **kwargs):
            calls.append(args)
            return original(*args, **kwargs)

        setattr(module, method_name, wrapped)

    monkeypatch.setattr(backend_module, "FSDPModule", FakeFSDPLinear)
    monkeypatch.setattr(
        backend_module, "register_fsdp_forward_method", fake_register
    )

    weight, prepared_reference, anchor = _backend()._prepare_chunk_loss_lm_head(
        model, reference
    )

    assert calls == [(reference,)]
    assert weight is lm_head.weight
    assert prepared_reference is reference
    assert anchor.shape == ()
    assert not anchor.requires_grad


def test_chunk_loss_lm_head_post_backward_follows_weight_gradient(monkeypatch):
    import torchturbo.engine.backends.fsdp2.backend as backend_module

    events = []
    lm_head = None

    class RegisterPostBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, reference):
            return reference

        @staticmethod
        def backward(ctx, grad):
            events.append(lm_head.weight.grad is not None)
            return grad

    class FakeFSDPLinear(nn.Linear):
        def set_reshard_after_forward(self, value, recurse=False):
            self.reshard_after_forward = value

    lm_head = FakeFSDPLinear(3, 7, bias=False)

    class Model(_ContractModel):
        def __init__(self):
            super().__init__(mtp=True)
            self.lm_head = lm_head

    def fake_register(module, method_name):
        original = getattr(module, method_name)

        def wrapped(reference):
            return original(RegisterPostBackward.apply(reference))

        setattr(module, method_name, wrapped)

    monkeypatch.setattr(backend_module, "FSDPModule", FakeFSDPLinear)
    monkeypatch.setattr(
        backend_module, "register_fsdp_forward_method", fake_register
    )

    model = Model()
    loss, _ = _backend(mtp=True).compute_loss(
        model, _batch(), return_outputs=True
    )
    loss.backward()

    assert events == [True]
    assert model.hidden.grad is not None
    assert lm_head.weight.grad is not None
