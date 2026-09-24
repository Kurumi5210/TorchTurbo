"""Compatibility tests for FSDP2 optimizer checkpoints."""

import os

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

import torch
from torch import nn
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.backends.fsdp2.config import FSDPConfig
from torchturbo.engine.backends.fsdp2.dcp import DcpCheckpoint
from torchturbo.engine.optimizers import OptimizerClassPlan
from torchturbo.models.weight_converter import BaseWeightConverter


def test_load_adamw_checkpoint_without_parameter_names(tmp_path, monkeypatch):
    # DCP supports single-process CPU save/load without a process group.
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    torch.manual_seed(7)
    reference_model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2))
    reference_optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in reference_model.parameters() if p.ndim > 1],
                "weight_decay": 0.1,
            },
            {
                "params": [p for p in reference_model.parameters() if p.ndim == 1],
                "weight_decay": 0.0,
            },
        ],
        lr=0.002,
        weight_decay=0.1,
    )
    for step in range(3):
        for index, param in enumerate(reference_model.parameters()):
            param.grad = torch.full_like(param, (step + 1) * (index + 1) / 10)
        reference_optimizer.step()
        reference_optimizer.zero_grad()

    checkpoint = DcpCheckpoint(
        rank=0,
        weight_converter=BaseWeightConverter.__new__(BaseWeightConverter),
    )
    parallel_context = {"ep": {"ep_rank": 0}}
    checkpoint_dir = str(tmp_path)
    checkpoint.save_model(reference_model, checkpoint_dir, parallel_context)
    checkpoint.save_optimizer(
        reference_optimizer, reference_model, checkpoint_dir, parallel_context
    )

    resumed_model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2))
    plan = OptimizerClassPlan(
        resumed_model,
        torch.optim.AdamW,
        {"lr": 0.002, "weight_decay": 0.1},
    )
    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend.fsdp_config = FSDPConfig()
    backend._mesh = {"edp": object()}
    resumed_optimizer = backend.prepare_optimizer(plan)

    checkpoint.load_model(resumed_model, checkpoint_dir, parallel_context)
    checkpoint.load_optimizer(
        resumed_optimizer, resumed_model, checkpoint_dir, parallel_context
    )
    torch.testing.assert_close(
        get_optimizer_state_dict(resumed_model, resumed_optimizer)["state"],
        get_optimizer_state_dict(reference_model, reference_optimizer)["state"],
        rtol=0,
        atol=0,
    )

    for model, optimizer in (
        (reference_model, reference_optimizer),
        (resumed_model, resumed_optimizer),
    ):
        for index, param in enumerate(model.parameters()):
            param.grad = torch.full_like(param, -(index + 1) / 5)
        optimizer.step()

    torch.testing.assert_close(
        resumed_model.state_dict(), reference_model.state_dict(), rtol=0, atol=0
    )
    resumed_state = get_optimizer_state_dict(resumed_model, resumed_optimizer)
    reference_state = get_optimizer_state_dict(reference_model, reference_optimizer)
    assert resumed_state["param_groups"] == reference_state["param_groups"]
    torch.testing.assert_close(
        resumed_state["state"], reference_state["state"], rtol=0, atol=0
    )
