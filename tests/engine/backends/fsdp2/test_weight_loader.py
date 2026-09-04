"""Unit tests for FSDP2 weight loading behavior."""

import os
import sys
import types

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

if "grouped_gemm" not in sys.modules:  # pragma: no cover - env shim
    _gg = types.ModuleType("grouped_gemm")
    _gg.ops = types.SimpleNamespace()
    sys.modules["grouped_gemm"] = _gg

import torch
import torch.nn as nn

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.backends.fsdp2.config import CPUOffloadMode
import torchturbo.engine.backends.fsdp2.weight_loader as weight_loader_mod
from torchturbo.engine.backends.fsdp2.weight_loader import DistParamLoader
from torchturbo.runtime import TurboDevice


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = types.SimpleNamespace()
        self.weight = nn.Parameter(torch.ones(2))
        self.register_buffer("buf", torch.ones(2))
        self.tie_called = False

    def tie_weights(self):
        self.tie_called = True


def test_load_weights_cpu_offload_assigns_and_restores_buffers(monkeypatch):
    model = TinyModel()

    loader = DistParamLoader.__new__(DistParamLoader)
    loader.model = model
    loader.device = torch.device("cuda:0")
    loader.is_cpu_offload = True
    loader.rank = 0
    loader._sync_log_rank0 = lambda *args, **kwargs: None
    loader._load_local_shards = lambda: {}
    loader._broadcast_buffers = lambda sharded_sd: sharded_sd.update(
        {"buf": torch.tensor([2.0, 2.0])}
    )
    loader._broadcast_params = lambda sharded_sd: sharded_sd.update(
        {"weight": torch.tensor([3.0, 3.0])}
    )
    loader._cleanup = lambda: None

    load_calls = {}

    def fake_load_state_dict(state_dict, assign=False, strict=True):
        load_calls["assign"] = assign
        load_calls["strict"] = strict
        load_calls["devices"] = {k: v.device.type for k, v in state_dict.items()}
        load_calls["keys"] = tuple(state_dict.keys())
        return None

    model.load_state_dict = fake_load_state_dict

    loader.load_weights()

    assert load_calls["assign"] is True
    assert load_calls["strict"] is False
    assert load_calls["devices"] == {"buf": "cpu", "weight": "cpu"}
    assert load_calls["keys"] == ("buf", "weight")
    assert model.tie_called is True
    assert torch.equal(model.buf, torch.ones(2))


def test_load_model_materializes_model_when_cpu_offload(monkeypatch):
    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend.fsdp_config = types.SimpleNamespace(
        cpu_offload=types.SimpleNamespace(
            enabled=True,
            mode=CPUOffloadMode.FULL,
        )
    )
    backend._device_runtime = types.SimpleNamespace(
        device=torch.device("cuda"),
        turbo_device=TurboDevice.CUDA,
    )
    backend._mesh = None
    backend._register_no_shard_params_grad_sync = lambda model: None

    model = TinyModel()
    to_empty_calls = []

    def fake_to_empty(device):
        to_empty_calls.append(torch.device(device))
        return model

    model.to_empty = fake_to_empty

    helper_calls = []
    backend._keep_non_offloaded_on_device = lambda m: helper_calls.append(
        (m, backend.device)
    )

    backend.load_model(model, "unused", load_weights=False)

    assert to_empty_calls == [torch.device("cpu")]
    assert helper_calls == [(model, torch.device("cuda"))]


def test_broadcast_buffers_writes_directly_into_module(monkeypatch):
    model = TinyModel()
    loader = DistParamLoader.__new__(DistParamLoader)
    loader.model = model
    loader.device = torch.device("cpu")
    loader.is_cpu_offload = False
    loader.rank = 0
    loader.param_converter = types.SimpleNamespace(
        is_persist_buffer=lambda name: name == "buf",
        convert_persist_buffer=lambda name: "hf_buf",
        get_default_hf_dtype=lambda name: torch.float32,
    )
    loader.param_to_filename = {"hf_buf": "file.safetensors"}
    loader.param_to_rank = {"hf_buf": 0}
    loader.local_sd = {"hf_buf": torch.tensor([7.0, 8.0])}

    broadcast_calls = []

    def fake_broadcast(tensor, src, async_op=False, group=None):
        broadcast_calls.append((tensor.device.type, src, async_op))
        return None

    monkeypatch.setattr(weight_loader_mod.dist, "broadcast", fake_broadcast)

    sharded_sd = {}
    loader._broadcast_buffers(sharded_sd)

    assert broadcast_calls == [("cpu", 0, False)]
    assert torch.equal(sharded_sd["buf"], torch.tensor([7.0, 8.0]))
    assert torch.equal(model.buf, torch.ones(2))
