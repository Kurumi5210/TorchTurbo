"""Chunked-loss model contract tests for the FSDP2 backend."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.config import LossChunkConfig
from torchturbo.models.base import SequenceParallelInputMode
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


class _ContractModel(nn.Module):
    def __init__(self, *, ignore_skip=False, omit_hidden=False, mtp=False):
        super().__init__()
        torch.manual_seed(3)
        self.hidden = nn.Parameter(torch.randn(1, 4, 3))
        self.lm_head = nn.Linear(3, 7, bias=False)
        self.ignore_skip = ignore_skip
        self.omit_hidden = omit_hidden
        self.with_mtp = mtp
        self.skip_calls = []

    def get_sequence_parallel_input_mode(self):
        return SequenceParallelInputMode.POST_EMBEDDING

    def get_lm_head(self):
        return self.lm_head

    def forward(
        self,
        input_ids,
        position_ids,
        use_cache,
        skip_lm_head=False,
    ):
        self.skip_calls.append(skip_lm_head)
        hidden = self.hidden.expand(input_ids.shape[0], -1, -1)
        logits = self.lm_head(hidden) if not skip_lm_head or self.ignore_skip else None
        mtp_hidden_states = [hidden * 0.5] if self.with_mtp else None
        return SimpleNamespace(
            logits=logits,
            last_hidden_state=None if self.omit_hidden else hidden,
            mtp_hidden_states=mtp_hidden_states,
            mtp_logits=(
                [self.lm_head(state) for state in mtp_hidden_states]
                if mtp_hidden_states and not skip_lm_head
                else None
            ),
        )


def test_enabled_chunked_loss_rejects_ignored_flag():
    model = _ContractModel(ignore_skip=True)

    with pytest.raises(RuntimeError, match="ignored skip_lm_head"):
        _backend().compute_loss(model, _batch())


def test_enabled_chunked_loss_requires_hidden_states():
    model = _ContractModel(omit_hidden=True)

    with pytest.raises(RuntimeError, match="last_hidden_state"):
        _backend().compute_loss(model, _batch())


def test_eval_uses_regular_logits_path():
    model = _ContractModel()
    model.eval()

    loss, outputs = _backend().compute_loss(model, _batch(), return_outputs=True)

    assert loss.ndim == 0
    assert outputs.logits is not None
    assert model.skip_calls == [False]


def test_main_and_mtp_losses_are_aggregated_on_chunked_path():
    model = _ContractModel(mtp=True)
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

    assert model.skip_calls == [True]
    assert outputs.logits is None
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

    assert model.skip_calls == [False]
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


def test_chunk_loss_lm_head_forward_receives_positional_input(monkeypatch):
    import torchturbo.engine.backends.fsdp2.backend as backend_module

    class FakeFSDPLinear(nn.Linear):
        def set_reshard_after_forward(self, value, recurse=False):
            self.reshard_after_forward = value

    lm_head = FakeFSDPLinear(3, 7, bias=False)
    model = SimpleNamespace(get_lm_head=lambda: lm_head)
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


def test_chunk_loss_lm_head_post_backward_follows_weight_gradient(monkeypatch):
    import torchturbo.engine.backends.fsdp2.backend as backend_module

    events = []

    class FakeFSDPLinear(nn.Linear):
        def set_reshard_after_forward(self, value, recurse=False):
            self.reshard_after_forward = value

    lm_head = FakeFSDPLinear(3, 7, bias=False)

    class RegisterPostBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, reference):
            return reference

        @staticmethod
        def backward(ctx, grad):
            events.append(lm_head.weight.grad is not None)
            return grad

    class Model(_ContractModel):
        def __init__(self):
            super().__init__()
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
    loss, _ = _backend().compute_loss(model, _batch(), return_outputs=True)
    loss.backward()

    assert events == [True]
    assert model.hidden.grad is not None
    assert lm_head.weight.grad is not None
