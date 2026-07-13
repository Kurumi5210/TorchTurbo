"""Unit tests for FSDP parallel plan spec semantics."""

import os
import sys
import types

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

if "grouped_gemm" not in sys.modules:  # pragma: no cover - env shim
    _gg = types.ModuleType("grouped_gemm")
    _gg.ops = types.SimpleNamespace()
    sys.modules["grouped_gemm"] = _gg

import pytest

from torchturbo.parallel.fsdp import (
    FSDPParallelPlan,
    FSDPPlacementSpec,
    FSDPPlanSpec,
)


def test_standalone_world_shard_spec_is_valid():
    spec = FSDPPlanSpec(standalone_world_shard=True)

    assert spec.standalone_world_shard is True
    assert spec.no_shard is False
    assert spec.placement is None


def test_no_shard_conflicts_with_standalone_world_shard():
    with pytest.raises(ValueError, match="mutually exclusive"):
        FSDPPlanSpec(no_shard=True, standalone_world_shard=True)


def test_no_shard_conflicts_with_placement():
    with pytest.raises(ValueError, match="mutually exclusive"):
        FSDPPlanSpec(no_shard=True, placement=FSDPPlacementSpec(object(), "world"))


def test_standalone_world_shard_conflicts_with_placement():
    with pytest.raises(ValueError, match="redundant"):
        FSDPPlanSpec(
            standalone_world_shard=True,
            placement=FSDPPlacementSpec(object(), "edp"),
        )


def test_parallel_plan_still_normalizes_tuple_placement_specs():
    plan = FSDPParallelPlan(plan={"layers.*.mlp.experts": (object(), "edp")})

    spec = plan.plan["layers.*.mlp.experts"]
    assert isinstance(spec, FSDPPlanSpec)
    assert isinstance(spec.placement, FSDPPlacementSpec)
    assert spec.placement.mesh_dim == "edp"
