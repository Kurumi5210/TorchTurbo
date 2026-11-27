"""Chunked AdamW updates and joint workspace calibration on CPU-backed storage."""

import copy
import gc
import os
import sys
import types
import weakref

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")
if "grouped_gemm" not in sys.modules:  # pragma: no cover - env shim
    _gg = types.ModuleType("grouped_gemm")
    _gg.ops = types.SimpleNamespace()
    sys.modules["grouped_gemm"] = _gg

import pytest
import torch
import torch.distributed as dist

from torchturbo.engine.backends.fsdp2.optimizers.adamw import (
    AdamWSwap,
    DeviceWorkspace,
    WorkspacePlan,
    bind_parameters,
)
from torchturbo.runtime import TurboDevice
from torchturbo.runtime.cpu import CpuRuntime


class WorkspaceRuntime(CpuRuntime):
    """Expose accelerator memory statistics while allocating CPU tensors."""

    turbo_device = TurboDevice.NPU
    device = torch.device("cpu")

    def __init__(self, free=4096, reserved=0, allocated=0):
        self.free = free
        self.reserved = reserved
        self.allocated = allocated
        self.cache_clears = 0
        self.host_cache_clears = 0

    def memory_info(self):
        return self.free, self.free + self.reserved

    def memory_reserved(self):
        return self.reserved

    def memory_allocated(self):
        return self.allocated

    def empty_cache(self):
        self.cache_clears += 1

    def empty_host_cache(self):
        self.host_cache_clears += 1


class AsyncWorkspaceRuntime(WorkspaceRuntime):
    supports_async_streams = True


def _optimizer(sizes=(4, 4, 4, 4), runtime=None, **kwargs):
    params = [torch.nn.Parameter(torch.linspace(0.1, 1.0, size)) for size in sizes]
    optimizer = AdamWSwap(
        params, device_runtime=CpuRuntime(), optimizer_memory_fraction=0, **kwargs
    )
    optimizer.optimizer_memory_fraction = None
    optimizer._AUTO_WORKSPACE_FRACTION = 0.5
    optimizer.device_runtime = runtime or WorkspaceRuntime()
    for index, param in enumerate(params):
        param.grad = torch.full_like(param, 0.2 + index)
    return optimizer, params


def _select_plan(optimizer, plan):
    optimizer._tuned_workspace_plan = plan


@pytest.mark.parametrize("fraction", [0.5, 0.95, 1.0])
def test_explicit_fraction_uses_free_memory_without_trials(monkeypatch, cpu_adamw, fraction):
    optimizer, _ = _optimizer((512,), WorkspaceRuntime(free=1024, reserved=4096))
    optimizer.optimizer_memory_fraction = fraction
    monkeypatch.setattr(optimizer, "_autotune_workspace", lambda *a, **k: pytest.fail("unexpected trial"))
    for free in (1024, 768, 64):
        optimizer.device_runtime.free = free
        capacity = int(free * fraction) // 32 * 32
        assert optimizer.autotune_workspace() == capacity
        optimizer.step()
        assert optimizer._workspace_plan == WorkspacePlan(1, capacity)


def test_omitted_fraction_probes_to_90_percent(cpu_adamw):
    optimizer, _ = _optimizer((4,), WorkspaceRuntime(free=1024, reserved=4096))
    del optimizer._AUTO_WORKSPACE_FRACTION
    optimizer.autotune_workspace()
    # int(1024 * 0.9) = 921, aligned down to 32 bytes.
    assert optimizer._workspace_calibration_stats["probe_budget_bytes"] == 896


def _distributed(monkeypatch, all_reduce):
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", all_reduce)


def _run_cpu_adamw(self, params, grads, states, steps, group):
    """Consume pre-incremented counters without changing them, as fused AdamW does."""
    views = [param.detach() for param in params]
    native = torch.optim.AdamW(
        views,
        **{
            key: group[key]
            for key in ("lr", "betas", "eps", "weight_decay", "maximize")
        },
        fused=False,
        foreach=False,
    )
    for param, grad, state, step in zip(views, grads, states, steps):
        param.grad = grad
        # Native AdamW advances only this independent scalar to the current step.
        native.state[param] = {"step": step.detach().clone() - 1, **state}
    native.step()


@pytest.fixture
def cpu_adamw(monkeypatch):
    """Simulate fused math and pinned memory; accelerator kernels are not tested."""
    empty, empty_like = torch.empty, torch.empty_like

    def unpinned_empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return empty(*args, **kwargs)

    def unpinned_empty_like(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return empty_like(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", unpinned_empty)
    monkeypatch.setattr(torch, "empty_like", unpinned_empty_like)
    monkeypatch.setattr(AdamWSwap, "_run_adamw", _run_cpu_adamw)


@pytest.fixture
def disable_cyclic_gc():
    """Check that the retry path releases storage before cyclic collection."""
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def _snapshot(optimizer, params):
    return (
        [param.detach().clone() for param in params],
        [None if param.grad is None else param.grad.clone() for param in params],
        copy.deepcopy(optimizer.state_dict()),
    )


def _assert_unchanged(optimizer, params, saved):
    values, grads, state = saved
    for param, value, grad in zip(params, values, grads):
        torch.testing.assert_close(param, value, rtol=0, atol=0)
        torch.testing.assert_close(param.grad, grad, rtol=0, atol=0)
    actual = optimizer.state_dict()
    assert actual["param_groups"] == state["param_groups"]
    for param_id, entry in state["state"].items():
        for key, value in entry.items():
            torch.testing.assert_close(
                actual["state"][param_id][key], value, rtol=0, atol=0
            )


def _trial_inputs(optimizer):
    bound = bind_parameters(
        optimizer._active_parameters(), optimizer.param_to_cpu_states_map
    )
    optimizer._acquire_workspace(bound)
    saved = [param.local_param.detach().clone() for param in bound]
    sink = torch.empty(optimizer.workspace.capacity_bytes, dtype=torch.uint8)
    return bound, torch.ones(3, dtype=torch.int64), saved, sink


def test_bounds_count_cache_active_demand_and_allow_parameter_splitting():
    runtime = WorkspaceRuntime(free=0, reserved=320, allocated=64)
    optimizer, params = _optimizer((257, 4, 4096), runtime)
    params[-1].grad = None
    bound = bind_parameters(
        optimizer._active_parameters(), optimizer.param_to_cpu_states_map
    )
    assert optimizer._workspace_bounds(bound) == (32, 128)
    runtime.free, runtime.reserved = 256, 64
    assert optimizer._workspace_bounds(bound) == (32, 128)
    _select_plan(optimizer, WorkspacePlan(1, 128))
    assert optimizer._acquire_workspace().capacity_bytes == 128
    optimizer.release_workspace()


def test_local_bounds_require_only_one_aligned_moment_pair(
    monkeypatch,
):
    param = torch.nn.Parameter(torch.ones(5, 4))
    optimizer = AdamWSwap(
        [param], device_runtime=CpuRuntime(), optimizer_memory_fraction=0
    )
    optimizer.device_runtime = WorkspaceRuntime()
    optimizer.optimizer_memory_fraction = 0.5
    param.grad = torch.ones_like(param)
    _distributed(
        monkeypatch, lambda *args, **kwargs: pytest.fail("bounds must remain local")
    )
    bound = bind_parameters(
        [(param, optimizer.param_groups[0])], optimizer.param_to_cpu_states_map
    )
    assert optimizer._workspace_bounds(bound) == (
        32,
        2048,
    )


def test_budget_failure_is_reported_before_counters_or_values_change():
    optimizer, params = _optimizer((33,), WorkspaceRuntime(free=32))
    optimizer.param_to_name_map[params[0]] = "model.proj.weight"
    saved = _snapshot(optimizer, params)
    with pytest.raises(RuntimeError, match="before any update") as failure:
        optimizer.step()
    detail = str(failure.value.__cause__)
    assert "budget is 0 bytes" in detail
    assert "at least 32 bytes" in detail
    assert "model.proj.weight" not in detail
    _assert_unchanged(optimizer, params, saved)
    assert optimizer.workspace is None


def test_empty_rank_participates_in_allocation_and_update_status(monkeypatch):
    optimizer, params = _optimizer()
    for param in params:
        param.grad = None
    _select_plan(optimizer, WorkspacePlan(1, 0))
    calls = []
    _distributed(monkeypatch, lambda value, op, group: calls.append(value.numel()))
    monkeypatch.setattr(
        DeviceWorkspace,
        "__init__",
        lambda *_: pytest.fail("empty rank allocated a slab"),
    )
    optimizer.step()
    assert calls == [1, 1]
    assert optimizer.workspace is None
    assert all(step.item() == 0 for step in optimizer.param_to_step_map.values())


def test_global_empty_step_does_not_benchmark_or_allocate(monkeypatch):
    optimizer, params = _optimizer()
    for param in params:
        param.grad = None
    monkeypatch.setattr(
        DeviceWorkspace,
        "__init__",
        lambda *_: pytest.fail("empty step allocated a slab"),
    )
    monkeypatch.setattr(
        optimizer,
        "_benchmark_workspace",
        lambda *_: pytest.fail("empty step benchmarked"),
    )
    optimizer.step()
    assert optimizer._tuned_workspace_plan is None


def test_global_empty_first_step_remains_eligible_for_calibration(
    monkeypatch, cpu_adamw
):
    optimizer, params = _optimizer((33,))
    params[0].grad = None
    calibrations = []
    calibrate = optimizer._autotune_workspace

    def record(*args, **kwargs):
        calibrations.append(True)
        return calibrate(*args, **kwargs)

    monkeypatch.setattr(optimizer, "_autotune_workspace", record)
    optimizer.step()
    assert optimizer._tuned_workspace_plan is None
    assert optimizer.param_to_step_map[params[0]].item() == 0
    params[0].grad = torch.ones_like(params[0])
    optimizer.step()
    assert len(calibrations) == 2
    assert optimizer._tuned_workspace_plan.capacity_bytes > 0
    assert optimizer.param_to_step_map[params[0]].item() == 1


def test_candidates_probe_aligned_shards_and_both_budget_endpoints():
    assert AdamWSwap._workspace_candidates(65, 510, True) == tuple(
        WorkspacePlan(slots, size)
        for slots, size in [
            (1, 96),
            (2, 96),
            (1, 192),
            (2, 192),
            (2, 224),
            (1, 384),
            (1, 480),
        ]
    )
    assert AdamWSwap._workspace_candidates(96, 128, True) == (
        WorkspacePlan(1, 96),
        WorkspacePlan(1, 128),
    )
    with pytest.raises(RuntimeError, match="cannot probe M"):
        AdamWSwap._workspace_candidates(97, 96, False)


def test_candidates_stop_at_one_chunk_and_need_two_chunks_for_overlap():
    # One slot stops at the single-chunk slot; two slots stop one step below it.
    assert AdamWSwap._workspace_candidates(65, 510, True, full_slot=192) == (
        WorkspacePlan(1, 96),
        WorkspacePlan(2, 96),
        WorkspacePlan(2, 160),
        WorkspacePlan(1, 192),
    )
    # A single parameter never splits at its own slot, so two slots are skipped.
    assert AdamWSwap._workspace_candidates(65, 4096, True, full_slot=96) == (
        WorkspacePlan(1, 96),
    )


@pytest.mark.parametrize(
    "sizes,dtypes",
    [((33, 17, 4), (torch.float32,) * 3), ((7, 1, 30), (torch.bfloat16, torch.float32, torch.float16))],
)
def test_single_chunk_slot_is_the_smallest_one_chunk_slot(sizes, dtypes):
    from torchturbo.engine.backends.fsdp2.optimizers.adamw_workspace import (
        BoundParameter,
        iter_chunks,
        single_chunk_slot_bytes,
    )

    bound = []
    for size, dtype in zip(sizes, dtypes):
        tensor = torch.zeros(size, dtype=dtype)
        bound.append(
            BoundParameter(tensor, tensor, tensor, {}, tensor.clone(), tensor.clone())
        )
    full = single_chunk_slot_bytes(bound)
    assert full % 32 == 0
    assert len(list(iter_chunks(bound, full))) == 1
    assert len(list(iter_chunks(bound, full - 32))) == 2


def test_nonzero_calibration_releases_pinned_host_cache_once(cpu_adamw):
    optimizer, _ = _optimizer((17, 9))
    optimizer.autotune_workspace()
    assert optimizer.device_runtime.host_cache_clears == 0
    for state in optimizer.param_to_cpu_states_map.values():
        state["exp_avg"].fill_(0.1)
    optimizer.autotune_workspace()
    assert optimizer.device_runtime.host_cache_clears == 1
    optimizer.step()
    assert optimizer.device_runtime.host_cache_clears == 1


def test_calibration_ignores_warmup_and_outliers_and_prefers_near_fastest(
    monkeypatch, cpu_adamw
):
    optimizer, _ = _optimizer()
    plans = [WorkspacePlan(1, value) for value in (128, 64, 32)]
    samples = {
        128: iter([100, 1, 90, 1]),
        64: iter([100, 1.02, 90, 1.02]),
        32: iter([0, 1.04, 0.001, 1.04]),
    }
    monkeypatch.setattr(optimizer, "_workspace_candidates", lambda *_: plans)
    monkeypatch.setattr(
        optimizer,
        "_benchmark_workspace",
        lambda *_: next(samples[optimizer.workspace.capacity_bytes]),
    )
    assert optimizer.autotune_workspace() == 64
    assert optimizer._tuned_workspace_plan == WorkspacePlan(1, 64)
    assert all(next(values, None) is None for values in samples.values())


def test_calibration_uses_slowest_rank_time(monkeypatch, cpu_adamw):
    optimizer, _ = _optimizer()
    monkeypatch.setattr(optimizer, "_benchmark_workspace", lambda *_: 1.0)
    timings = []

    def all_reduce(value, op, group):
        if value.dtype is torch.float32:
            assert op is dist.ReduceOp.MAX
            capacity = optimizer.workspace.capacity_bytes
            timings.append(capacity)
            value.fill_(1.0 if capacity == 128 else 2.0)

    _distributed(monkeypatch, all_reduce)
    assert optimizer.autotune_workspace() == 128
    assert timings == [size for size in (32, 64, 128) for _ in range(4)]


@pytest.mark.parametrize("at_completion", [False, True], ids=["enqueue", "completion"])
def test_timing_collective_oom_aborts_after_smaller_candidate_succeeded(
    monkeypatch, cpu_adamw, at_completion
):
    optimizer, params = _optimizer()
    saved = _snapshot(optimizer, params)
    timed_capacities = []
    pending_error = False

    def all_reduce(value, op, group):
        nonlocal pending_error
        if value.dtype == torch.float32 and op == dist.ReduceOp.MAX:
            capacity = optimizer.workspace.capacity_bytes
            timed_capacities.append(capacity)
            if capacity == 64:
                if not at_completion:
                    raise torch.OutOfMemoryError("timing collective failed at enqueue")
                pending_error = True

    def synchronize():
        nonlocal pending_error
        if pending_error:
            pending_error = False
            raise torch.OutOfMemoryError("timing collective failed at completion")

    _distributed(monkeypatch, all_reduce)
    monkeypatch.setattr(optimizer.device_runtime, "synchronize", synchronize)
    with pytest.raises(torch.OutOfMemoryError, match="timing collective failed"):
        optimizer.autotune_workspace()

    assert timed_capacities == [32] * 4 + [64]
    assert optimizer._tuned_workspace_plan is None
    assert "out_of_memory_capacity_bytes" not in optimizer._workspace_calibration_stats
    _assert_unchanged(optimizer, params, saved)


def test_equal_capacity_serial_and_pipeline_plans_are_both_measured(
    monkeypatch, cpu_adamw
):
    optimizer, _ = _optimizer(runtime=AsyncWorkspaceRuntime(free=256))
    elapsed = {
        WorkspacePlan(1, 128): 0.9,
        WorkspacePlan(1, 32): 1.3,
        WorkspacePlan(1, 64): 1.1,
        WorkspacePlan(2, 32): 0.98,
        WorkspacePlan(2, 64): 0.7,
    }
    measured = []

    def benchmark(*args):
        measured.append(optimizer._workspace_plan)
        return elapsed[optimizer._workspace_plan]

    monkeypatch.setattr(optimizer, "_get_pipeline", lambda _: None)
    monkeypatch.setattr(optimizer, "_benchmark_workspace", benchmark)
    assert optimizer.autotune_workspace() == 128
    assert optimizer._tuned_workspace_plan == WorkspacePlan(2, 64)
    assert measured == [
        plan
        for plan in sorted(elapsed, key=lambda p: (p.slot_bytes, p.num_slots))
        for _ in range(4)
    ]


def test_calibration_uses_driver_free_not_allocator_cache(cpu_adamw):
    optimizer, params = _optimizer((32,), WorkspaceRuntime(free=128, reserved=4096))
    saved = _snapshot(optimizer, params)
    with pytest.raises(RuntimeError, match="cannot probe M=128"):
        optimizer.autotune_workspace()
    _assert_unchanged(optimizer, params, saved)


def test_nonzero_calibration_reuses_snapshot_and_sink_then_releases_views(
    monkeypatch, cpu_adamw
):
    optimizer, _ = _optimizer((17, 9))
    for state in optimizer.param_to_cpu_states_map.values():
        state["exp_avg"].fill_(0.1)
    references, snapshots = [], []
    benchmark, view_like = optimizer._benchmark_workspace, DeviceWorkspace.view_like

    def track_view(workspace, offset, tensor):
        result = view_like(workspace, offset, tensor)
        references.append(weakref.ref(result))
        return result

    def record_trial(bound, status, saved, sink):
        assert sink.numel() == 2 * (80 + 48)
        snapshots.append((tuple(id(param) for param in saved), id(sink)))
        references.extend(weakref.ref(param) for param in saved)
        references.append(weakref.ref(sink))
        return benchmark(bound, status, saved, sink)

    monkeypatch.setattr(DeviceWorkspace, "view_like", track_view)
    monkeypatch.setattr(optimizer, "_benchmark_workspace", record_trial)
    optimizer.autotune_workspace()
    assert len(set(snapshots)) == 1
    assert all(reference() is None for reference in references)
    assert optimizer.device_runtime.cache_clears == 0
    optimizer.step()
    assert optimizer.device_runtime.cache_clears == 0


@pytest.mark.parametrize("dirty_key", [None, "exp_avg", "exp_avg_sq"])
def test_every_trial_restores_verified_zero_or_nonzero_step_zero_state(
    monkeypatch, cpu_adamw, dirty_key
):
    from torchturbo.engine.backends.fsdp2.optimizers import adamw as adamw_module

    optimizer, params = _optimizer((33, 17))
    if dirty_key is not None:
        optimizer.param_to_cpu_states_map[params[0]][dirty_key].fill_(0.125)
    saved = _snapshot(optimizer, params)
    benchmark = optimizer._benchmark_workspace
    trials = []
    resets = []
    reset = adamw_module.reset_workspace_moments

    def record_reset(*args):
        resets.append(True)
        return reset(*args)

    def inspect_trial(bound, status, saved_params, sink):
        assert (sink is None) == (dirty_key is None)
        _assert_unchanged(optimizer, params, saved)
        result = benchmark(bound, status, saved_params, sink)
        _assert_unchanged(optimizer, params, saved)
        trials.append(optimizer._workspace_plan)
        return result

    monkeypatch.setattr(optimizer, "_benchmark_workspace", inspect_trial)
    monkeypatch.setattr(adamw_module, "reset_workspace_moments", record_reset)
    optimizer.autotune_workspace()
    assert len(trials) >= 8
    assert len(set(trials)) >= 2
    assert len(resets) == (len(trials) if dirty_key is None else 0)
    stats = optimizer._workspace_calibration_stats
    assert stats["output_mode"] == ("scratch" if dirty_key else "real_zero_moments")
    assert (stats["output_bytes"] == 0) == (dirty_key is None)
    assert stats["parameter_restore_seconds"] >= 0
    assert stats["moment_reset_seconds"] >= 0
    assert all(value >= 0 for key, value in stats.items() if key.endswith("_seconds"))
    _assert_unchanged(optimizer, params, saved)


def test_zero_check_only_considers_active_moments(monkeypatch, cpu_adamw):
    optimizer, params = _optimizer((33, 17))
    params[1].grad = None
    optimizer.param_to_cpu_states_map[params[1]]["exp_avg"].fill_(2)
    saved = _snapshot(optimizer, params)
    optimizer.autotune_workspace()
    assert optimizer._workspace_calibration_stats["output_mode"] == "real_zero_moments"
    _assert_unchanged(optimizer, params, saved)


def test_zero_detection_error_fails_in_snapshot_preparation(monkeypatch, cpu_adamw):
    optimizer, params = _optimizer((33,))
    saved = _snapshot(optimizer, params)

    def fail_zero_check(*args):
        raise RuntimeError("injected moment inspection failure")

    monkeypatch.setattr(torch, "count_nonzero", fail_zero_check)
    with pytest.raises(RuntimeError, match="snapshot preparation failed"):
        optimizer.autotune_workspace()
    _assert_unchanged(optimizer, params, saved)
    assert not optimizer._workspace_restore_failed


@pytest.mark.parametrize("growth,expected_clears", [(0, 0), (192, 0), (224, 1)])
def test_cleanup_only_clears_growth_beyond_selected_plan_and_margin(
    monkeypatch, cpu_adamw, growth, expected_clears
):
    runtime = WorkspaceRuntime(reserved=1024)
    optimizer, _ = _optimizer((33,), runtime)

    def measure(*args):
        runtime.reserved = 1024 + growth
        return 1.0

    monkeypatch.setattr(optimizer, "_WORKSPACE_CACHE_MARGIN_BYTES", 32)
    monkeypatch.setattr(optimizer, "_benchmark_workspace", measure)
    assert optimizer.autotune_workspace() == 160
    assert runtime.cache_clears == expected_clears
    stats = optimizer._workspace_calibration_stats
    assert stats["reserved_before_bytes"] == 1024
    assert stats["reserved_after_trials_bytes"] == 1024 + growth
    assert stats["cleanup_cleared_cache"] == bool(expected_clears)


@pytest.mark.parametrize("peer_failure", [False, True])
def test_workspace_oom_aborts_all_ranks_without_retry(monkeypatch, peer_failure):
    optimizer, params = _optimizer()
    saved, allocations = _snapshot(optimizer, params), []
    reductions = 0
    initialize = DeviceWorkspace.__init__

    def allocate(workspace, runtime, capacity):
        allocations.append(capacity)
        if not peer_failure and capacity > 64:
            raise torch.OutOfMemoryError("injected workspace failure")
        initialize(workspace, runtime, capacity)

    def all_reduce(value, op, group):
        nonlocal reductions
        reductions += 1
        if peer_failure:
            value.zero_()

    monkeypatch.setattr(DeviceWorkspace, "__init__", allocate)
    _distributed(monkeypatch, all_reduce)
    with pytest.raises(RuntimeError, match="before any update"):
        optimizer._reserve_workspace(256, status=torch.ones(1))
    assert reductions == 1
    assert allocations == [256]
    assert optimizer.workspace is None
    assert optimizer.device_runtime.cache_clears == 0
    _assert_unchanged(optimizer, params, saved)
    optimizer.release_workspace()


@pytest.mark.parametrize("oom", [False, True])
def test_allocation_failure_terminates_before_any_update(monkeypatch, oom):
    optimizer, params = _optimizer()
    saved, attempts = _snapshot(optimizer, params), []

    def allocate(workspace, runtime, capacity):
        attempts.append(capacity)
        if oom:
            raise torch.OutOfMemoryError("injected workspace failure")
        raise RuntimeError("injected invalid device")

    monkeypatch.setattr(DeviceWorkspace, "__init__", allocate)
    with pytest.raises(RuntimeError, match="before any update"):
        optimizer._reserve_workspace(128, status=torch.ones(1))
    assert attempts == [128]
    _assert_unchanged(optimizer, params, saved)


def test_tuned_plan_exceeding_live_budget_falls_back_to_measured_plan(
    monkeypatch, cpu_adamw
):
    optimizer, params = _optimizer((33,), WorkspaceRuntime(free=128))
    _select_plan(optimizer, WorkspacePlan(1, 128))
    optimizer._workspace_measurements = [
        (WorkspacePlan(1, 128), 1.0, 128),
        (WorkspacePlan(1, 64), 1.5, 64),
        (WorkspacePlan(2, 32), 1.6, 64),
    ]
    allocations, initialize = [], DeviceWorkspace.__init__

    def allocate(workspace, runtime, capacity):
        allocations.append(capacity)
        initialize(workspace, runtime, capacity)

    monkeypatch.setattr(DeviceWorkspace, "__init__", allocate)
    optimizer.step()
    # Budget 64 bytes, minimum slot 32: both smaller plans fit, and the faster
    # single slot wins; the 128-byte plan is no longer a candidate.
    assert allocations == [64]
    assert optimizer._tuned_workspace_plan == WorkspacePlan(1, 64)
    assert optimizer.param_to_step_map[params[0]].item() == 1
    optimizer.step()
    assert allocations == [64, 64]


def test_fallback_uses_peer_budget_so_every_rank_selects_one_plan(
    monkeypatch, cpu_adamw
):
    optimizer, _ = _optimizer((33,), WorkspaceRuntime(free=4096))
    _select_plan(optimizer, WorkspacePlan(1, 256))
    optimizer._workspace_measurements = [
        (WorkspacePlan(1, 256), 1.0, 512),
        (WorkspacePlan(1, 128), 1.2, 256),
        (WorkspacePlan(1, 64), 1.5, 128),
    ]
    reductions = []

    def all_reduce(value, op, group):
        reductions.append(value)
        if len(reductions) == 1:
            value.fill_(2)  # A peer requests a refit.
        elif len(reductions) == 2:
            value.fill_(min(value.item(), 160))  # The peer can hold 160 bytes.

    _distributed(monkeypatch, all_reduce)
    optimizer.step()
    assert optimizer._tuned_workspace_plan == WorkspacePlan(1, 128)
    # Reuse the allocation-status scalar for the shared budget and the retry.
    assert len(reductions) == 4
    assert reductions[0] is reductions[1] is reductions[2]
    assert reductions[1].numel() == 1 and reductions[1].dtype == torch.int64


def test_tuned_plan_exceeding_live_budget_is_rejected():
    optimizer, _ = _optimizer((65,), WorkspaceRuntime(free=128))
    _select_plan(optimizer, WorkspacePlan(1, 128))
    with pytest.raises(RuntimeError, match="no longer fits"):
        optimizer.step()
    assert optimizer.workspace is None


def test_previously_empty_rank_requires_explicit_calibration(cpu_adamw):
    optimizer, params = _optimizer((33,))
    _select_plan(optimizer, WorkspacePlan(1, 0))
    saved = _snapshot(optimizer, params)
    with pytest.raises(RuntimeError, match="no calibrated AdamW workspace"):
        optimizer.step()
    _assert_unchanged(optimizer, params, saved)
    optimizer.autotune_workspace()
    optimizer.step()
    assert optimizer.param_to_step_map[params[0]].item() == 1


def test_pipeline_allocation_oom_does_not_switch_to_serial(monkeypatch):
    optimizer, _ = _optimizer((33,), AsyncWorkspaceRuntime())
    _select_plan(optimizer, WorkspacePlan(2, 64))
    initialize = DeviceWorkspace.__init__

    def allocate(workspace, runtime, capacity):
        if capacity > 32:
            raise torch.OutOfMemoryError("injected two-slot allocation failure")
        initialize(workspace, runtime, capacity)

    monkeypatch.setattr(optimizer, "_get_pipeline", lambda _: None)
    monkeypatch.setattr(DeviceWorkspace, "__init__", allocate)
    with pytest.raises(RuntimeError, match="injected two-slot allocation failure"):
        optimizer.step()
    assert optimizer.workspace is None


@pytest.mark.parametrize("slot_bytes", [32, 128])
def test_whole_trial_has_three_collectives_independent_of_chunk_count(
    monkeypatch, cpu_adamw, slot_bytes
):
    optimizer, params = _optimizer((33, 17))
    _select_plan(optimizer, WorkspacePlan(1, slot_bytes))
    arguments, saved = _trial_inputs(optimizer), _snapshot(optimizer, params)
    calls = []
    _distributed(
        monkeypatch, lambda value, op, group: calls.append((op, value.numel()))
    )
    with torch.no_grad():
        optimizer._benchmark_workspace(*arguments)
    assert calls == [(dist.ReduceOp.MIN, 3)] * 3
    _assert_unchanged(optimizer, params, saved)
    optimizer.release_workspace()


@pytest.mark.parametrize("phase", ["prepare", "execute", "restore"])
def test_peer_trial_failure_restores_local_parameters(monkeypatch, cpu_adamw, phase):
    optimizer, params = _optimizer((33, 17))
    _select_plan(optimizer, WorkspacePlan(1, 32))
    arguments, saved = _trial_inputs(optimizer), _snapshot(optimizer, params)
    phases = []

    def all_reduce(value, op, group):
        current = ("prepare", "execute", "restore")[len(phases)]
        phases.append(current)
        if current == phase:
            value.copy_(torch.tensor([0, 0, phase != "restore"], dtype=value.dtype))

    _distributed(monkeypatch, all_reduce)
    with torch.no_grad(), pytest.raises(RuntimeError):
        optimizer._benchmark_workspace(*arguments)
    assert phases == (
        ["prepare"] if phase == "prepare" else ["prepare", "execute", "restore"]
    )
    _assert_unchanged(optimizer, params, saved)
    optimizer.release_workspace()


def test_partial_calibration_failure_restores_all_previous_chunks(
    monkeypatch, cpu_adamw
):
    optimizer, params = _optimizer((33, 17))
    saved, updates = _snapshot(optimizer, params), []

    def fail_after_write(params, grads, states, steps, group):
        updates.append(True)
        params[0].add_(10)
        states[0]["exp_avg"].add_(5)
        if len(updates) == 3:
            assert any(
                torch.count_nonzero(state["exp_avg"]).item()
                for state in optimizer.param_to_cpu_states_map.values()
            )
            raise RuntimeError("injected partial kernel failure")

    monkeypatch.setattr(
        optimizer, "_workspace_candidates", lambda *_: [WorkspacePlan(1, 32)]
    )
    monkeypatch.setattr(optimizer, "_run_adamw", fail_after_write)
    with pytest.raises(RuntimeError, match="execute failed"):
        optimizer.autotune_workspace()
    assert len(updates) == 3
    _assert_unchanged(optimizer, params, saved)


def test_trial_oom_restores_values_and_aborts_calibration(monkeypatch, cpu_adamw):
    optimizer, params = _optimizer((33,))
    saved, run = _snapshot(optimizer, params), optimizer._run_adamw

    def fail_large(params, grads, states, steps, group):
        if optimizer.workspace.capacity_bytes > 32:
            params[0].add_(10)
            raise torch.OutOfMemoryError("injected kernel allocation failure")
        run(params, grads, states, steps, group)

    monkeypatch.setattr(optimizer, "_run_adamw", fail_large)
    with pytest.raises(torch.OutOfMemoryError, match="execute failed"):
        optimizer.autotune_workspace()
    assert optimizer._tuned_workspace_plan is None
    _assert_unchanged(optimizer, params, saved)
    assert optimizer.device_runtime.cache_clears == 0
    assert not optimizer._workspace_calibration_stats["cleanup_cleared_cache"]


def _reference_adamw(optimizer, params):
    """Native AdamW over copies of the current values, gradients and state."""
    reference_params = [torch.nn.Parameter(param.detach().clone()) for param in params]
    reference = torch.optim.AdamW(reference_params, fused=False, foreach=False)
    for param, reference_param in zip(params, reference_params):
        reference_param.grad = param.grad.clone()
        reference.state[reference_param] = {
            key: tensor.detach().clone() for key, tensor in optimizer.state[param].items()
        }
    return reference, reference_params


def _assert_matches_reference(optimizer, params, reference, reference_params):
    for param, reference_param in zip(params, reference_params):
        torch.testing.assert_close(param, reference_param, rtol=0, atol=0)
        for key, value in reference.state[reference_param].items():
            torch.testing.assert_close(optimizer.state[param][key], value, rtol=0, atol=0)


def test_calibration_oom_skips_that_capacity_and_larger_but_keeps_smaller(
    monkeypatch, cpu_adamw
):
    optimizer, params = _optimizer(runtime=AsyncWorkspaceRuntime())
    saved, initialize, measured = _snapshot(optimizer, params), DeviceWorkspace.__init__, []

    def allocate(workspace, runtime, capacity):
        if capacity >= 192:
            raise torch.OutOfMemoryError("injected workspace allocation failure")
        initialize(workspace, runtime, capacity)

    def benchmark(*args):
        measured.append(optimizer._workspace_plan)
        return 0.5 if optimizer._workspace_plan == WorkspacePlan(1, 128) else 1.0

    monkeypatch.setattr(optimizer, "_get_pipeline", lambda _: None)
    monkeypatch.setattr(DeviceWorkspace, "__init__", allocate)
    monkeypatch.setattr(optimizer, "_benchmark_workspace", benchmark)
    assert optimizer.autotune_workspace() == 128
    # (2, 96) runs out of memory at 192 bytes. (1, 128) follows it in slot order
    # but needs less memory, so it is still measured and wins.
    plans = [
        WorkspacePlan(1, 32),
        WorkspacePlan(2, 32),
        WorkspacePlan(1, 64),
        WorkspacePlan(2, 64),
        WorkspacePlan(1, 128),
    ]
    assert measured == [plan for plan in plans for _ in range(4)]
    assert [entry[0] for entry in optimizer._workspace_measurements] == plans
    assert optimizer._tuned_workspace_plan == WorkspacePlan(1, 128)
    assert optimizer._workspace_calibration_stats["out_of_memory_capacity_bytes"] == 192
    _assert_unchanged(optimizer, params, saved)


@pytest.mark.parametrize("dirty", [False, True], ids=["zero-moments", "nonzero-moments"])
def test_trial_oom_after_partial_update_restores_state_for_the_next_step(
    monkeypatch, cpu_adamw, dirty, disable_cyclic_gc
):
    optimizer, params = _optimizer((33, 17))
    if dirty:
        for state in optimizer.param_to_cpu_states_map.values():
            state["exp_avg"].fill_(0.25)
            state["exp_avg_sq"].fill_(0.5)
    saved = _snapshot(optimizer, params)
    reference, reference_params = _reference_adamw(optimizer, params)
    run, large_calls = optimizer._run_adamw, 0
    empty = torch.empty
    sinks = []

    def record_sink(*args, **kwargs):
        tensor = empty(*args, **kwargs)
        if kwargs.get("pin_memory") and tensor.dtype == torch.uint8:
            sinks.append(weakref.ref(tensor))
        return tensor

    def clear_host_cache():
        # The allocator can only recycle blocks whose last tensor has gone away.
        assert len(sinks) == 1 and sinks[0]() is None
        optimizer.device_runtime.host_cache_clears += 1

    def fail_second_large_chunk(local_params, grads, states, steps, group):
        nonlocal large_calls
        run(local_params, grads, states, steps, group)
        if optimizer.workspace.capacity_bytes >= 320:
            large_calls += 1
            if large_calls == 2:
                # The first chunk already updated values and wrote its moments.
                local_params[0].add_(10)
                raise torch.OutOfMemoryError("injected kernel workspace failure")

    monkeypatch.setattr(optimizer, "_run_adamw", fail_second_large_chunk)
    monkeypatch.setattr(torch, "empty", record_sink)
    monkeypatch.setattr(optimizer.device_runtime, "empty_host_cache", clear_host_cache)
    assert optimizer.autotune_workspace() == 160
    assert large_calls == 2
    assert optimizer.device_runtime.host_cache_clears == int(dirty)
    assert optimizer._tuned_workspace_plan == WorkspacePlan(1, 160)
    assert [entry[0] for entry in optimizer._workspace_measurements] == [
        WorkspacePlan(1, 160)
    ]
    assert not optimizer._workspace_restore_failed
    _assert_unchanged(optimizer, params, saved)
    monkeypatch.setattr(optimizer, "_run_adamw", run)
    optimizer.step()
    reference.step()
    _assert_matches_reference(optimizer, params, reference, reference_params)


@pytest.mark.parametrize(
    "local_oom,peer_code,expected_oom",
    [(True, 0, False), (True, 3, True), (False, 1, True)],
    ids=["peer-error", "local-oom", "peer-oom"],
)
def test_allocation_raises_oom_only_when_every_failure_is_oom(
    monkeypatch, local_oom, peer_code, expected_oom
):
    optimizer, _ = _optimizer()
    initialize = DeviceWorkspace.__init__

    def allocate(workspace, runtime, capacity):
        if local_oom:
            raise torch.OutOfMemoryError("injected workspace failure")
        initialize(workspace, runtime, capacity)

    def all_reduce(value, op, group):
        value.fill_(min(int(value.item()), peer_code))

    monkeypatch.setattr(DeviceWorkspace, "__init__", allocate)
    _distributed(monkeypatch, all_reduce)
    with pytest.raises(RuntimeError, match="before any update") as info:
        optimizer._reserve_workspace(64, status=torch.ones(1))
    assert isinstance(info.value, torch.OutOfMemoryError) == expected_oom
    assert optimizer.workspace is None


def test_trial_oom_with_peer_non_oom_failure_aborts_calibration(monkeypatch, cpu_adamw):
    optimizer, params = _optimizer((33, 17))
    saved, run = _snapshot(optimizer, params), optimizer._run_adamw

    def fail_large(local_params, grads, states, steps, group):
        if optimizer.workspace.capacity_bytes >= 320:
            local_params[0].add_(10)
            raise torch.OutOfMemoryError("injected kernel workspace failure")
        run(local_params, grads, states, steps, group)

    def all_reduce(value, op, group):
        if value.numel() == 3 and not value[0]:
            value[1] = 0  # A peer failed in the same phase without running out of memory.

    monkeypatch.setattr(optimizer, "_run_adamw", fail_large)
    _distributed(monkeypatch, all_reduce)
    with pytest.raises(RuntimeError, match="execute failed") as info:
        optimizer.autotune_workspace()
    assert not isinstance(info.value, torch.OutOfMemoryError)
    assert optimizer._tuned_workspace_plan is None
    _assert_unchanged(optimizer, params, saved)


def test_trial_oom_with_failed_restore_is_sticky(monkeypatch, cpu_adamw):
    optimizer, params = _optimizer((33, 17))
    run, copy_tensor = optimizer._run_adamw, torch.Tensor.copy_

    def fail_large(local_params, grads, states, steps, group):
        if optimizer.workspace.capacity_bytes >= 320:
            local_params[0].add_(10)
            raise torch.OutOfMemoryError("injected kernel workspace failure")
        run(local_params, grads, states, steps, group)

    def restore(tensor, source, *args, **kwargs):
        workspace = optimizer.workspace
        if tensor is params[0] and workspace is not None and workspace.capacity_bytes >= 320:
            raise RuntimeError("injected parameter restore failure")
        return copy_tensor(tensor, source, *args, **kwargs)

    monkeypatch.setattr(optimizer, "_run_adamw", fail_large)
    monkeypatch.setattr(torch.Tensor, "copy_", restore)
    with pytest.raises(RuntimeError, match="reload the model checkpoint") as failure:
        optimizer.autotune_workspace()
    assert str(failure.value.__cause__) == "injected parameter restore failure"
    monkeypatch.setattr(torch.Tensor, "copy_", copy_tensor)
    assert optimizer._workspace_restore_failed
    assert optimizer._tuned_workspace_plan is None
    for operation in (optimizer.step, optimizer.autotune_workspace):
        with pytest.raises(RuntimeError, match="could not restore"):
            operation()


def test_step_allocation_oom_retries_smaller_measured_plans_before_any_update(
    monkeypatch, cpu_adamw, disable_cyclic_gc
):
    optimizer, params = _optimizer((33,))
    _select_plan(optimizer, WorkspacePlan(1, 256))
    optimizer._workspace_measurements = [
        (WorkspacePlan(1, 256), 1.0, 256),
        (WorkspacePlan(1, 128), 1.2, 128),
        (WorkspacePlan(1, 64), 1.5, 64),
    ]
    reference, reference_params = _reference_adamw(optimizer, params)
    allocations, initialize = [], DeviceWorkspace.__init__
    storage = []

    def allocate(workspace, runtime, capacity):
        allocations.append(capacity)
        if capacity >= 128:
            raise torch.OutOfMemoryError("injected workspace allocation failure")
        initialize(workspace, runtime, capacity)
        storage.append(weakref.ref(workspace._storage))

    monkeypatch.setattr(DeviceWorkspace, "__init__", allocate)
    optimizer.step()
    assert allocations == [256, 128, 64]
    assert len(storage) == 1 and storage[0]() is None
    assert optimizer._tuned_workspace_plan == WorkspacePlan(1, 64)
    reference.step()
    _assert_matches_reference(optimizer, params, reference, reference_params)


def test_step_allocation_oom_without_smaller_measured_plan_fails_before_any_update(
    monkeypatch, cpu_adamw
):
    optimizer, params = _optimizer((33,))
    _select_plan(optimizer, WorkspacePlan(1, 256))
    optimizer._workspace_measurements = [(WorkspacePlan(1, 256), 1.0, 256)]
    saved = _snapshot(optimizer, params)

    def allocate(workspace, runtime, capacity):
        raise torch.OutOfMemoryError("injected workspace allocation failure")

    monkeypatch.setattr(DeviceWorkspace, "__init__", allocate)
    with pytest.raises(RuntimeError, match="ran out of memory") as info:
        optimizer.step()
    assert isinstance(info.value.__cause__, torch.OutOfMemoryError)
    assert "injected workspace allocation failure" in str(info.value.__cause__)
    assert not optimizer._workspace_update_failed
    _assert_unchanged(optimizer, params, saved)


def test_execution_failure_then_cpu_zero_failure_blocks_future_updates(
    monkeypatch, cpu_adamw
):
    optimizer, params = _optimizer((33,))
    destination = optimizer.param_to_cpu_states_map[params[0]]["exp_avg"]
    zero_tensor = torch.Tensor.zero_
    run = optimizer._run_adamw
    calls = []

    def fail_second_chunk(*args):
        calls.append(True)
        run(*args)
        if len(calls) == 2:
            raise RuntimeError("injected execution failure after a completed chunk")

    def fail_zero(tensor, *args, **kwargs):
        if tensor is destination:
            raise RuntimeError("injected moment restoration failure")
        return zero_tensor(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "zero_", fail_zero)
    monkeypatch.setattr(optimizer, "_run_adamw", fail_second_chunk)
    with pytest.raises(RuntimeError, match="could not restore"):
        optimizer.autotune_workspace()
    assert optimizer._workspace_restore_failed
    assert len(calls) == 2
    assert optimizer.param_to_step_map[params[0]].item() == 0
    for operation in (optimizer.step, optimizer.autotune_workspace):
        with pytest.raises(RuntimeError, match="could not restore"):
            operation()


@pytest.mark.parametrize("failure", ["zero", "d2h"])
def test_device_moment_reset_failure_is_sticky(monkeypatch, cpu_adamw, failure):
    from torchturbo.engine.backends.fsdp2.optimizers import adamw as adamw_module

    optimizer, params = _optimizer((33,))
    original_value = params[0].detach().clone()
    reset = adamw_module.reset_workspace_moments
    zero_tensor, copy_tensor = torch.Tensor.zero_, torch.Tensor.copy_
    inside_reset = False
    failures = []

    def record_reset(*args):
        nonlocal inside_reset
        inside_reset = True
        try:
            return reset(*args)
        finally:
            inside_reset = False

    def fail_zero(tensor, *args, **kwargs):
        if inside_reset and failure == "zero":
            assert tensor.dtype is torch.uint8
            failures.append(True)
            raise RuntimeError("injected workspace zero failure")
        return zero_tensor(tensor, *args, **kwargs)

    def fail_copy(tensor, source, *args, **kwargs):
        if inside_reset and failure == "d2h":
            assert kwargs["non_blocking"]
            failures.append(True)
            raise RuntimeError("injected moment D2H failure")
        return copy_tensor(tensor, source, *args, **kwargs)

    monkeypatch.setattr(adamw_module, "reset_workspace_moments", record_reset)
    monkeypatch.setattr(torch.Tensor, "zero_", fail_zero)
    monkeypatch.setattr(torch.Tensor, "copy_", fail_copy)
    with pytest.raises(RuntimeError, match="could not restore"):
        optimizer.autotune_workspace()
    assert failures == [True]
    assert optimizer._workspace_restore_failed
    assert optimizer.workspace is None
    assert optimizer.param_to_step_map[params[0]].item() == 0
    torch.testing.assert_close(params[0], original_value, rtol=0, atol=0)
    for operation in (optimizer.step, optimizer.autotune_workspace):
        with pytest.raises(RuntimeError, match="could not restore"):
            operation()


def test_failed_transfer_drain_prevents_rewriting_cpu_moments(monkeypatch, cpu_adamw):
    optimizer, params = _optimizer((33,))
    moments = [
        optimizer.param_to_cpu_states_map[params[0]][key]
        for key in optimizer.state_keys
    ]
    run = optimizer._run_adamw
    zero_tensor = torch.Tensor.zero_
    calls = 0
    fail_sync = False
    restored_moments = []

    def fail_second_chunk(params, grads, states, steps, group):
        nonlocal calls, fail_sync
        calls += 1
        if calls == 2:
            fail_sync = True
            raise RuntimeError("injected partial submission failure")
        run(params, grads, states, steps, group)

    def synchronize():
        nonlocal fail_sync
        if fail_sync:
            fail_sync = False
            raise RuntimeError("injected stream drain failure")

    def record_zero(tensor, *args, **kwargs):
        if any(tensor is moment for moment in moments):
            restored_moments.append(tensor)
        return zero_tensor(tensor, *args, **kwargs)

    monkeypatch.setattr(optimizer, "_run_adamw", fail_second_chunk)
    monkeypatch.setattr(optimizer.device_runtime, "synchronize", synchronize)
    monkeypatch.setattr(torch.Tensor, "zero_", record_zero)
    with pytest.raises(RuntimeError, match="could not restore"):
        optimizer.autotune_workspace()
    assert optimizer._workspace_restore_failed
    assert not restored_moments
    assert torch.count_nonzero(moments[0]).item() > 0


def test_each_step_binds_once_and_observes_replaced_gradients(monkeypatch, cpu_adamw):
    from torchturbo.engine.backends.fsdp2.optimizers import adamw as adamw_module

    optimizer, params = _optimizer((33,))
    bind = adamw_module.bind_parameters
    calls = []

    def record_binding(active, state):
        calls.append([param.grad for param, _ in active])
        return bind(active, state)

    monkeypatch.setattr(adamw_module, "bind_parameters", record_binding)
    for iteration in range(2):
        grad = torch.full_like(params[0], 0.125 * (iteration + 1))
        params[0].grad = grad
        optimizer.step()
        assert len(calls) == iteration + 1
        assert calls[-1][0] is grad


@pytest.mark.parametrize("fail_update", [False, True])
def test_completed_step_skips_release_sync_but_failed_step_keeps_it(
    monkeypatch, cpu_adamw, fail_update
):
    optimizer, _ = _optimizer((33,))
    _select_plan(optimizer, WorkspacePlan(1, 32))
    synchronizations = []
    monkeypatch.setattr(
        optimizer.device_runtime,
        "synchronize",
        lambda: synchronizations.append(optimizer.workspace is not None),
    )
    if fail_update:

        def fail(*args):
            raise RuntimeError("injected update failure")

        monkeypatch.setattr(optimizer, "_run_adamw", fail)
        with pytest.raises(RuntimeError, match="after it started"):
            optimizer.step()
    else:
        optimizer.step()
    assert synchronizations == ([False, True] if fail_update else [False])
    assert optimizer.workspace is None


def test_public_release_still_synchronizes_before_dropping_storage(monkeypatch):
    optimizer, _ = _optimizer((33,))
    _select_plan(optimizer, WorkspacePlan(1, 32))
    optimizer._acquire_workspace()
    synchronizations = []
    monkeypatch.setattr(
        optimizer.device_runtime,
        "synchronize",
        lambda: synchronizations.append(optimizer.workspace is not None),
    )
    optimizer.release_workspace()
    assert synchronizations == [True]
    assert optimizer.workspace is None


def test_failed_restore_prevents_future_updates_and_calibration(monkeypatch, cpu_adamw):
    optimizer, params = _optimizer((33,))
    copy_tensor = torch.Tensor.copy_

    def fail_restore(tensor, source, *args, **kwargs):
        if tensor is params[0]:
            raise RuntimeError("injected restore failure")
        return copy_tensor(tensor, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", fail_restore)
    with pytest.raises(RuntimeError, match="could not restore"):
        optimizer.autotune_workspace()
    assert optimizer._workspace_restore_failed
    monkeypatch.setattr(torch.Tensor, "copy_", copy_tensor)
    saved = _snapshot(optimizer, params)
    for operation in (optimizer.step, optimizer.autotune_workspace):
        with pytest.raises(RuntimeError, match="could not restore"):
            operation()
        _assert_unchanged(optimizer, params, saved)


def test_actual_partial_update_failure_is_sticky(monkeypatch, cpu_adamw):
    optimizer, params = _optimizer((33,))
    _select_plan(optimizer, WorkspacePlan(1, 32))

    def fail_after_write(params, grads, states, steps, group):
        params[0].add_(10)
        raise RuntimeError("injected actual update failure")

    monkeypatch.setattr(optimizer, "_run_adamw", fail_after_write)
    with pytest.raises(RuntimeError, match="after it started"):
        optimizer.step()
    assert optimizer._workspace_update_failed
    saved = _snapshot(optimizer, params)
    with pytest.raises(RuntimeError, match="after it started"):
        optimizer.step()
    _assert_unchanged(optimizer, params, saved)


@pytest.mark.parametrize("slot_bytes", [32, 96])
def test_chunks_match_adamw_for_tail_mixed_groups_dtypes_zero_and_replaced_gradients(
    cpu_adamw, slot_bytes
):
    params = [
        torch.nn.Parameter(torch.linspace(0.1, 1, 33)),
        torch.nn.Parameter(torch.linspace(0.1, 1, 17, dtype=torch.float64)),
        torch.nn.Parameter(torch.linspace(0.1, 1, 9, dtype=torch.bfloat16)),
        torch.nn.Parameter(torch.empty(0)),
    ]
    references = [torch.nn.Parameter(param.detach().clone()) for param in params]

    def groups(values):
        return [
            {"params": values[:2], "lr": 0.003, "weight_decay": 0.1},
            {"params": values[2:], "lr": 0.001, "betas": (0.8, 0.95)},
        ]

    optimizer = AdamWSwap(
        groups(params), device_runtime=CpuRuntime(), optimizer_memory_fraction=0
    )
    optimizer.device_runtime = WorkspaceRuntime()
    optimizer.optimizer_memory_fraction = None
    _select_plan(optimizer, WorkspacePlan(1, slot_bytes))
    reference = torch.optim.AdamW(groups(references), fused=False, foreach=False)
    for iteration in range(1, 4):
        gradients = []
        for index, (param, expected) in enumerate(zip(params, references)):
            grad = (
                None
                if iteration == 1 and index == 1
                else torch.full_like(param, 0.1 * iteration + index)
            )
            param.grad, expected.grad = grad, None if grad is None else grad.clone()
            gradients.append(grad)
        optimizer.step()
        reference.step()
        for index, (param, expected) in enumerate(zip(params, references)):
            assert param.grad is gradients[index]
            # CPU scalar/vector kernels can round differently at chunk tails.
            tolerance = 2 * torch.finfo(param.dtype).eps
            torch.testing.assert_close(param, expected, rtol=tolerance, atol=0)
            expected_step = iteration - (index == 1)
            assert optimizer.param_to_step_map[param].item() == expected_step
            if expected_step:
                for key, value in reference.state[expected].items():
                    torch.testing.assert_close(
                        optimizer.state[param][key], value, rtol=tolerance, atol=0
                    )


@pytest.mark.parametrize("operation", ["step", "autotune_workspace"])
def test_transposed_parameter_fails_before_mutating_optimizer(operation):
    param = torch.nn.Parameter(torch.ones(5, 4).t())
    optimizer = AdamWSwap(
        [param], device_runtime=CpuRuntime(), optimizer_memory_fraction=0
    )
    optimizer.optimizer_memory_fraction = 0.5
    optimizer.device_runtime = WorkspaceRuntime()
    param.grad = torch.ones_like(param)
    _select_plan(optimizer, WorkspacePlan(1, 160))
    saved = _snapshot(optimizer, [param])
    with pytest.raises(RuntimeError, match="requires contiguous parameter;"):
        if operation == "step":
            optimizer.step()
        else:
            optimizer.autotune_workspace()
    _assert_unchanged(optimizer, [param], saved)
    assert optimizer.workspace is None


def test_empty_tensor_with_gradient_advances_once_without_workspace():
    optimizer, params = _optimizer((0,))
    optimizer.step()
    assert optimizer.param_to_step_map[params[0]].item() == 1
    assert optimizer.workspace is None
    assert optimizer._tuned_workspace_plan is None


def test_amp_skip_never_calibrates_or_advances_counters(monkeypatch):
    optimizer, params = _optimizer((17,))
    optimizer.zero_grad()
    scaler = torch.amp.GradScaler("cpu", init_scale=8)
    scaler.scale(params[0].sum()).backward()
    params[0].grad.fill_(float("inf"))
    value = params[0].detach().clone()
    monkeypatch.setattr(
        optimizer, "autotune_workspace", lambda: pytest.fail("AMP skip calibrated")
    )
    scaler.step(optimizer)
    scaler.update()
    torch.testing.assert_close(params[0], value, rtol=0, atol=0)
    assert optimizer.param_to_step_map[params[0]].item() == 0
    assert optimizer.workspace is None


def test_first_step_calibrates_once_and_checkpoint_resume_updates_once(
    monkeypatch, cpu_adamw
):
    optimizer, params = _optimizer((33, 17, 1))
    references = [torch.nn.Parameter(param.detach().clone()) for param in params]
    reference = torch.optim.AdamW(references, fused=False, foreach=False)
    calls, calibrate = [], optimizer._autotune_workspace

    def record_calibration(*args, **kwargs):
        calls.append(True)
        return calibrate(*args, **kwargs)

    monkeypatch.setattr(optimizer, "_autotune_workspace", record_calibration)
    for _ in range(2):
        for param, expected in zip(params, references):
            expected.grad = param.grad.clone()
        optimizer.step()
        reference.step()
    assert len(calls) == 1
    saved = copy.deepcopy(optimizer.state_dict())
    resumed, resumed_params = _optimizer((33, 17, 1))
    with torch.no_grad():
        for param, previous in zip(resumed_params, params):
            param.copy_(previous)
    resumed.load_state_dict(saved)
    assert resumed._tuned_workspace_plan is None
    resumed.step()
    reference.step()
    for param, expected in zip(resumed_params, references):
        torch.testing.assert_close(param, expected, rtol=0, atol=0)
        for key, value in reference.state[expected].items():
            torch.testing.assert_close(resumed.state[param][key], value, rtol=0, atol=0)
        assert resumed.param_to_step_map[param].item() == 3
