"""Pure TP contracts: configuration, loss targets, checkpoint layouts, and expert views."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch.distributed.tensor.parallel import ColwiseParallel

from torchturbo.engine.args import TorchTurboArgs, parse_torchturbo_args
from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.checkpoint.metadata import validate_parallel_layout
from torchturbo.engine.config import build_from_args
from torchturbo.engine.core import TorchTurboEngine
from torchturbo.models.modules.loss import prepare_loss_targets
from torchturbo.models.weight_converter import BaseWeightConverter
from torchturbo.parallel import FSDPParallelPlan, TensorParallelPlanSpec


def test_tensor_degrees_follow_cli_over_environment_over_seed(monkeypatch):
    monkeypatch.setenv("TORCH_TURBO_TP_SIZE", "2")
    monkeypatch.setenv("TORCH_TURBO_ETP_SIZE", "2")
    plan = build_from_args(TorchTurboArgs(tp_size=4)).parallel_plan
    assert (plan.tp_size, plan.etp_size) == (2, 2)
    args, remaining = parse_torchturbo_args(["--tt_tp_size", "8", "--other"])
    assert remaining == ["--other"]
    plan = build_from_args(args).parallel_plan
    assert (plan.tp_size, plan.etp_size) == (8, 2)


def test_plan_serialization_keeps_tensor_parallel_spec():
    plan = FSDPParallelPlan(tp_size=2, etp_size=2)
    assert FSDPParallelPlan.from_dict(json.loads(json.dumps(plan.to_dict()))) == plan
    styled = FSDPParallelPlan(tensor_parallel_plan=TensorParallelPlanSpec(plan={"up": ColwiseParallel()}))
    restored = FSDPParallelPlan.from_dict(styled.to_dict())
    assert isinstance(restored.tensor_parallel_plan.plan["up"], ColwiseParallel)


def test_loss_targets_shift_within_packed_samples():
    labels = torch.tensor([[1, 2, 3, 4, 5, 6]])
    boundaries = torch.tensor([0, 3, 6])
    assert prepare_loss_targets(labels, boundaries).tolist() == [[2, 3, -100, 5, 6, -100]]
    assert prepare_loss_targets(labels, boundaries, shift=2).tolist() == [[3, -100, -100, 6, -100, -100]]


def test_tp_loss_normalizer_divides_step_targets_over_dense_ranks():
    labels, logits = torch.tensor([[1, 2, 3, 4]]), [torch.zeros(1, 4, 7)]
    options = dict(sp_size=1, sp_group=None, sp_rank=0, dense_size=4)
    _, normalizers = FSDP2Backend._prepare_loss_labels(logits, labels, num_items_in_batch=12, **options)
    assert normalizers == [3.0]
    for count in (None, 0):
        with pytest.raises(ValueError):
            FSDP2Backend._prepare_loss_labels(logits, labels, num_items_in_batch=count, **options)


def _layout(tp_size=2):
    return {
        "topology": dict(world_size=4, sp_size=1, tp_size=tp_size, ep_size=1, etp_size=1),
        "layout_schema_version": 1, "expert_layout_version": "packed_2d_v1", "parameters": [],
    }


def test_native_restore_requires_the_saved_tp_layout():
    saved = {"schema_version": 2, "parallel_layout": _layout()}
    validate_parallel_layout(saved, _layout())
    validate_parallel_layout(None, None)
    validate_parallel_layout({"schema_version": 1}, None)
    with pytest.raises(ValueError, match="schema 2"):
        validate_parallel_layout({"schema_version": 1}, _layout())
    with pytest.raises(ValueError, match="saved TP configuration"):
        validate_parallel_layout(saved, None)
    with pytest.raises(ValueError, match="topology"):
        validate_parallel_layout(saved, _layout(tp_size=4))


def test_gradient_accumulation_counts_logical_data_replicas():
    assert TorchTurboEngine._ga_from_gbs(64, 4) == 16
    with pytest.raises(ValueError, match="divisible"):
        TorchTurboEngine._ga_from_gbs(6, 4)


def test_etp_expert_weights_convert_through_packed_hf_views():
    converter = SimpleNamespace(
        etp_size=2, ep_size=2, moe_config=SimpleNamespace(num_moe_experts=4),
        config=SimpleNamespace(hidden_size=3, moe_intermediate_size=2),
        is_expert_weight1=lambda name: name.endswith("weight1"),
        is_expert_weight2=lambda name: name.endswith("weight2"),
    )
    view = BaseWeightConverter._packed_expert_view
    gate_up = torch.arange(24.).view(6, 2, 2)
    packed = view(converter, "experts.weight1", gate_up)
    # Gate features precede up features in the packed HF-facing layout.
    torch.testing.assert_close(packed, torch.cat([gate_up[:, 0], gate_up[:, 1]], dim=-1))
    assert view(converter, "experts.weight2", torch.zeros(2, 2, 3)).shape == (4, 3)
    with pytest.raises(ValueError, match="logical shape"):
        view(converter, "experts.weight1", packed)
    converter.etp_size = 1
    assert view(converter, "experts.weight1", packed) is packed
