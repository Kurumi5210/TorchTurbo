"""Real Gloo coordination for rank-local workspace calibration on CPU storage."""

import os
import sys
from contextlib import contextmanager
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .test_adamw_workspace import (
    AdamWSwap,
    DeviceWorkspace,
    WorkspacePlan,
    WorkspaceRuntime,
    _assert_matches_reference,
    _assert_unchanged,
    _optimizer,
    _reference_adamw,
    _run_cpu_adamw,
    _select_plan,
    _snapshot,
    bind_parameters,
)

_HAS_GLOO = dist.is_available() and dist.is_gloo_available()


@contextmanager
def _cpu_calibration():
    empty_like, empty = torch.empty_like, torch.empty

    def unpinned_empty_like(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return empty_like(*args, **kwargs)

    def unpinned_empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return empty(*args, **kwargs)

    with (
        patch.object(AdamWSwap, "_run_adamw", _run_cpu_adamw),
        patch.object(torch, "empty_like", unpinned_empty_like),
        patch.object(torch, "empty", unpinned_empty),
    ):
        yield


def _run_workspace_coordination(rank, init_file, empty_peer):
    os.environ.setdefault(
        "GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo"
    )
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        runtime = WorkspaceRuntime(free=1280 if rank == 0 else 640)
        optimizer, params = _optimizer((33, 17, 4, 4) if rank == 0 else (4,), runtime)
        if rank == 1 and empty_peer:
            params[0].grad = None
        expected_bounds = (
            (32, 640) if rank == 0 else ((0, 0) if empty_peer else (32, 320))
        )
        bound = bind_parameters(
            optimizer._active_parameters(), optimizer.param_to_cpu_states_map
        )
        assert optimizer._workspace_bounds(bound) == expected_bounds
        with patch.object(optimizer, "optimizer_memory_fraction", 0.5):
            expected_fixed = (640 if rank == 0 else 0) if empty_peer else 320
            assert optimizer.autotune_workspace() == expected_fixed
            optimizer._acquire_workspace(bound)
            assert optimizer._workspace_plan.capacity_bytes == expected_fixed
            optimizer.release_workspace()

        # A larger rank-local slab need not fit inside another rank's budget.
        # Its injected allocation failure must not reallocate the peer's slab.
        attempts = []
        initialize = DeviceWorkspace.__init__

        def allocate(workspace, device_runtime, capacity):
            attempts.append(capacity)
            if rank == 0 and capacity > 64:
                raise torch.OutOfMemoryError("injected allocation failure on rank 0")
            initialize(workspace, device_runtime, capacity)

        with (
            patch.object(DeviceWorkspace, "__init__", allocate),
            pytest.raises(RuntimeError, match="before any update"),
        ):
            optimizer._reserve_workspace(expected_bounds[1], status=torch.ones(1))
        assert attempts == ([640] if rank == 0 else ([] if empty_peer else [320]))
        assert optimizer.workspace is None
        runtime.cache_clears = 0

        empty_like, empty = torch.empty_like, torch.empty

        def unpinned_empty_like(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return empty_like(*args, **kwargs)

        def unpinned_empty(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return empty(*args, **kwargs)

        measured_capacities = []
        benchmark = optimizer._benchmark_workspace

        def measure(bound, status, saved_params, cpu_sink):
            capacity = (
                optimizer.workspace.capacity_bytes
                if optimizer.workspace is not None
                else 0
            )
            measured_capacities.append(capacity)
            benchmark(bound, status, saved_params, cpu_sink)
            # Keep selection deterministic while executing the real transfer,
            # backup, update and restore coordination in every measured slot.
            return {160: 1.2, 320: 1.1, 512: 1.0}[capacity] if rank == 0 else 0.5

        saved = _snapshot(optimizer, params)
        with (
            patch.object(AdamWSwap, "_run_adamw", _run_cpu_adamw),
            patch.object(torch, "empty_like", unpinned_empty_like),
            patch.object(torch, "empty", unpinned_empty),
            patch.object(optimizer, "_benchmark_workspace", measure),
        ):
            chosen = optimizer.autotune_workspace()
            assert chosen == ((512 if rank == 0 else 0) if empty_peer else 320)
            # The sweep stops at rank 0's single-chunk slot (2 * 256 bytes).
            assert optimizer._tuned_workspace_plan == WorkspacePlan(
                1, 512 if empty_peer else 320
            )
            expected_measurements = [160] * 4 + [320] * 4
            if empty_peer:
                expected_measurements += [512] * 4
                if rank == 1:
                    expected_measurements = [0] * 12
            assert measured_capacities == expected_measurements
            _assert_unchanged(optimizer, params, saved)
            assert optimizer.workspace is None
            assert runtime.cache_clears == 0

            reference_params = [
                torch.nn.Parameter(param.detach().clone()) for param in params
            ]
            reference = torch.optim.AdamW(reference_params, fused=False, foreach=False)
            for param, reference_param in zip(params, reference_params):
                reference_param.grad = (
                    None if param.grad is None else param.grad.clone()
                )
            optimizer.step()
            reference.step()
            for param, reference_param in zip(params, reference_params):
                torch.testing.assert_close(param, reference_param, rtol=0, atol=0)
                expected_step = 0 if param.grad is None else 1
                assert optimizer.param_to_step_map[param].item() == expected_step
                if expected_step:
                    for key, value in reference.state[reference_param].items():
                        torch.testing.assert_close(
                            optimizer.state[param][key], value, rtol=0, atol=0
                        )
            assert optimizer.workspace is None
            assert runtime.cache_clears == 0
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not _HAS_GLOO, reason="requires torch.distributed with Gloo support"
)
@pytest.mark.parametrize(
    "empty_peer", [False, True], ids=["single-capacity-peer", "empty-peer"]
)
def test_workspace_calibration_with_different_rank_demands(tmp_path, empty_peer):
    mp.spawn(
        _run_workspace_coordination,
        args=(str(tmp_path / "workspace"), empty_peer),
        nprocs=2,
        join=True,
    )


def _run_calibration_phase_failure(rank, init_file, phase):
    os.environ.setdefault(
        "GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo"
    )
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        optimizer, params = _optimizer((33, 17))
        saved = _snapshot(optimizer, params)
        empty_like, empty = torch.empty_like, torch.empty
        copy_tensor = torch.Tensor.copy_
        to_tensor = torch.Tensor.to
        kernel_calls = 0

        def prepare(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return empty(*args, **kwargs)

        def snapshot(tensor, *args, **kwargs):
            if (
                rank == 1
                and phase == "prepare"
                and kwargs.get("device") == "cpu"
                and kwargs.get("copy")
            ):
                raise RuntimeError("injected rank 1 backup allocation failure")
            return to_tensor(tensor, *args, **kwargs)

        def unpinned_empty_like(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return empty_like(*args, **kwargs)

        def execute(self, local_params, grads, states, steps, group):
            nonlocal kernel_calls
            kernel_calls += 1
            if rank == 1 and phase == "update":
                local_params[0].add_(10)
                raise RuntimeError("injected rank 1 partial update failure")
            result = _run_cpu_adamw(self, local_params, grads, states, steps, group)
            if rank == 1 and phase == "execute" and kernel_calls == 2:
                # The first chunk has already written nonzero canonical moments.
                local_params[0].add_(10)
                raise RuntimeError("injected rank 1 partial update failure")
            return result

        def restore(tensor, source, *args, **kwargs):
            if rank == 1 and phase == "restore" and tensor is params[0]:
                raise RuntimeError("injected rank 1 parameter restore failure")
            return copy_tensor(tensor, source, *args, **kwargs)

        with (
            patch.object(AdamWSwap, "_run_adamw", execute),
            patch.object(torch, "empty", prepare),
            patch.object(torch, "empty_like", unpinned_empty_like),
            patch.object(torch.Tensor, "copy_", restore),
            patch.object(torch.Tensor, "to", snapshot),
        ):
            message = (
                "could not restore"
                if phase == "restore"
                else "failed on a training rank"
            )
            if phase == "update":
                _select_plan(optimizer, WorkspacePlan(1, 32))
                message = "after it started"
            with pytest.raises(RuntimeError, match=message):
                optimizer.step()
        assert optimizer.workspace is None
        assert all(
            step.item() == (1 if phase == "update" else 0)
            for step in optimizer.param_to_step_map.values()
        )
        if phase != "update" and (rank == 0 or phase != "restore"):
            _assert_unchanged(optimizer, params, saved)
        if phase in ("restore", "update"):
            assert (
                optimizer._workspace_restore_failed
                if phase == "restore"
                else optimizer._workspace_update_failed
            )
            for operation in (optimizer.step, optimizer.autotune_workspace):
                with pytest.raises(
                    RuntimeError,
                    match="could not restore"
                    if phase == "restore"
                    else "after it started",
                ):
                    operation()
        else:
            assert not optimizer._workspace_restore_failed
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not _HAS_GLOO, reason="requires torch.distributed with Gloo support"
)
@pytest.mark.parametrize("phase", ["prepare", "execute", "restore", "update"])
def test_workspace_calibration_failure_reaches_every_rank(tmp_path, phase):
    mp.spawn(
        _run_calibration_phase_failure,
        args=(str(tmp_path / "failure"), phase),
        nprocs=2,
        join=True,
    )


def _run_empty_then_active(rank, init_file):
    os.environ.setdefault(
        "GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo"
    )
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        optimizer, params = _optimizer((33, 17), WorkspaceRuntime(free=512))
        for param in params:
            param.grad = None
        saved = _snapshot(optimizer, params)
        calibrations = []
        measurements = []
        calibrate = optimizer._autotune_workspace
        benchmark = optimizer._benchmark_workspace

        def record_calibration(bound, *, error=None):
            calibrations.append(len(bound))
            return calibrate(bound, error=error)

        def record_measurement(bound, status, saved_params, cpu_sink):
            measurements.append(len(bound))
            benchmark(bound, status, saved_params, cpu_sink)
            return 1.0

        with (
            _cpu_calibration(),
            patch.object(optimizer, "_autotune_workspace", record_calibration),
            patch.object(optimizer, "_benchmark_workspace", record_measurement),
        ):
            optimizer.step()
            assert optimizer._tuned_workspace_plan is None
            assert optimizer._workspace_plan.capacity_bytes == 0
            assert measurements == []
            _assert_unchanged(optimizer, params, saved)
            num_empty_calibrations = len(calibrations)

            reference_params = [
                torch.nn.Parameter(param.detach().clone()) for param in params
            ]
            reference = torch.optim.AdamW(reference_params, fused=False, foreach=False)
            if rank == 0:
                for index, (param, reference_param) in enumerate(
                    zip(params, reference_params)
                ):
                    param.grad = torch.full_like(param, 0.2 + index)
                    reference_param.grad = param.grad.clone()

            for expected_step in (1, 2):
                optimizer.step()
                reference.step()
                assert len(calibrations) == num_empty_calibrations + 1
                assert calibrations[-1] == (len(params) if rank == 0 else 0)
                assert optimizer._tuned_workspace_plan == WorkspacePlan(1, 160)
                assert optimizer._workspace_plan.capacity_bytes == (160 if rank == 0 else 0)
                for param, reference_param in zip(params, reference_params):
                    torch.testing.assert_close(param, reference_param, rtol=0, atol=0)
                    step = optimizer.param_to_step_map[param].item()
                    assert step == (expected_step if rank == 0 else 0)
                    if rank == 0:
                        for key, value in reference.state[reference_param].items():
                            # CPU kernels can round whole tensors and small
                            # chunks differently after the first update.
                            torch.testing.assert_close(
                                optimizer.state[param][key],
                                value,
                                rtol=2 * torch.finfo(value.dtype).eps,
                                atol=0,
                            )
                assert optimizer.workspace is None
                assert optimizer.device_runtime.cache_clears == 0

            assert measurements and set(measurements) == {2 if rank == 0 else 0}
            counts = [None, None]
            dist.all_gather_object(counts, len(measurements))
            assert counts[0] == counts[1]
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not _HAS_GLOO, reason="requires torch.distributed with Gloo support"
)
def test_globally_empty_step_does_not_freeze_later_rank_local_calibration(tmp_path):
    mp.spawn(
        _run_empty_then_active,
        args=(str(tmp_path / "empty-then-active"),),
        nprocs=2,
        join=True,
    )


def _run_mixed_initial_moments(rank, init_file):
    os.environ.setdefault(
        "GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo"
    )
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        optimizer, params = _optimizer((33, 17), WorkspaceRuntime(free=512))
        if rank == 1:
            for param in params:
                optimizer.state[param]["exp_avg"].fill_(0.25)
                optimizer.state[param]["exp_avg_sq"].fill_(0.5)
        assert all(optimizer.param_to_step_map[param].item() == 0 for param in params)
        saved = _snapshot(optimizer, params)
        reference_params = [
            torch.nn.Parameter(param.detach().clone()) for param in params
        ]
        reference = torch.optim.AdamW(reference_params, fused=False, foreach=False)
        for param, reference_param in zip(params, reference_params):
            reference_param.grad = param.grad.clone()
            reference.state[reference_param] = {
                key: tensor.detach().clone()
                for key, tensor in optimizer.state[param].items()
            }
        sink_modes = []
        benchmark = optimizer._benchmark_workspace

        def record_measurement(bound, status, saved_params, cpu_sink):
            sink_modes.append(cpu_sink is not None)
            _assert_unchanged(optimizer, params, saved)
            benchmark(bound, status, saved_params, cpu_sink)
            # Every trial must restore moments, even when the real counter is zero.
            _assert_unchanged(optimizer, params, saved)
            return 1.0

        with (
            _cpu_calibration(),
            patch.object(optimizer, "_benchmark_workspace", record_measurement),
        ):
            assert optimizer.autotune_workspace() == 160
            assert len(sink_modes) > 4
            assert set(sink_modes) == {rank == 1}
            assert optimizer._workspace_calibration_stats["output_mode"] == (
                "scratch" if rank == 1 else "real_zero_moments"
            )
            _assert_unchanged(optimizer, params, saved)
            optimizer.step()
            reference.step()
            for param, reference_param in zip(params, reference_params):
                torch.testing.assert_close(param, reference_param, rtol=0, atol=0)
                for key, value in reference.state[reference_param].items():
                    torch.testing.assert_close(
                        optimizer.state[param][key], value, rtol=0, atol=0
                    )
            assert optimizer.workspace is None
            assert optimizer.device_runtime.cache_clears == 0
            observations = [None, None]
            dist.all_gather_object(observations, (len(sink_modes), sink_modes[0]))
            assert observations[0][0] == observations[1][0]
            assert [entry[1] for entry in observations] == [False, True]
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not _HAS_GLOO, reason="requires torch.distributed with Gloo support"
)
def test_zero_and_nonzero_moment_ranks_calibrate_without_mutating_step_zero_state(
    tmp_path,
):
    mp.spawn(
        _run_mixed_initial_moments,
        args=(str(tmp_path / "mixed-moments"),),
        nprocs=2,
        join=True,
    )


def _run_one_rank_trial_oom(rank, init_file, dirty_rank):
    os.environ.setdefault(
        "GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo"
    )
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        optimizer, params = _optimizer((33, 17))
        if rank == dirty_rank:
            for param in params:
                optimizer.state[param]["exp_avg"].fill_(0.25)
                optimizer.state[param]["exp_avg_sq"].fill_(0.5)
        saved = _snapshot(optimizer, params)
        reference_params = [
            torch.nn.Parameter(param.detach().clone()) for param in params
        ]
        reference = torch.optim.AdamW(reference_params, fused=False, foreach=False)
        for param, reference_param in zip(params, reference_params):
            reference_param.grad = param.grad.clone()
            reference.state[reference_param] = {
                key: tensor.detach().clone()
                for key, tensor in optimizer.state[param].items()
            }
        large_calls, measured = 0, []
        benchmark = optimizer._benchmark_workspace

        def execute(self, local_params, grads, states, steps, group):
            nonlocal large_calls
            _run_cpu_adamw(self, local_params, grads, states, steps, group)
            if rank == 1 and self.workspace.capacity_bytes >= 320:
                large_calls += 1
                if large_calls == 2:
                    # Rank 1 fails after its first chunk; rank 0 completes the trial.
                    local_params[0].add_(10)
                    raise torch.OutOfMemoryError("injected rank 1 kernel failure")

        def measure(bound, status, saved_params, cpu_sink):
            measured.append(optimizer.workspace.capacity_bytes)
            return benchmark(bound, status, saved_params, cpu_sink)

        with _cpu_calibration():
            with (
                patch.object(AdamWSwap, "_run_adamw", execute),
                patch.object(optimizer, "_benchmark_workspace", measure),
            ):
                assert optimizer.autotune_workspace() == 160
            assert measured == [160] * 4 + [320]
            assert optimizer._tuned_workspace_plan == WorkspacePlan(1, 160)
            stats = optimizer._workspace_calibration_stats
            assert stats["out_of_memory_capacity_bytes"] == 320
            assert stats["output_mode"] == (
                "scratch" if rank == dirty_rank else "real_zero_moments"
            )
            assert not optimizer._workspace_restore_failed
            _assert_unchanged(optimizer, params, saved)
            optimizer.step()
        reference.step()
        for param, reference_param in zip(params, reference_params):
            torch.testing.assert_close(param, reference_param, rtol=0, atol=0)
            for key, value in reference.state[reference_param].items():
                torch.testing.assert_close(
                    optimizer.state[param][key], value, rtol=0, atol=0
                )
        plans = [None, None]
        dist.all_gather_object(plans, optimizer._tuned_workspace_plan)
        assert plans[0] == plans[1]
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not _HAS_GLOO, reason="requires torch.distributed with Gloo support"
)
@pytest.mark.parametrize(
    "dirty_rank", [0, 1], ids=["failing-rank-zero-moments", "failing-rank-nonzero-moments"]
)
def test_trial_oom_on_one_rank_shrinks_calibration_on_every_rank(tmp_path, dirty_rank):
    mp.spawn(
        _run_one_rank_trial_oom,
        args=(str(tmp_path / "one-rank-oom"), dirty_rank),
        nprocs=2,
        join=True,
    )


def _run_timing_failure(rank, init_file, operation, oom):
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2,
        timeout=timedelta(seconds=15),
    )
    try:
        optimizer, params = _optimizer((33, 17))
        saved = _snapshot(optimizer, params)
        reference, reference_params = _reference_adamw(optimizer, params)
        benchmark = optimizer._benchmark_workspace
        zeros = torch.zeros
        completed = 0
        injected = False
        # The second candidate warms up on trial 5 and records its first sample on trial 6.
        fault_trial = 5 if operation == "fill_" else 6

        def run_trial(*args):
            nonlocal completed
            elapsed = benchmark(*args)
            completed += 1
            return elapsed

        def allocate(*args, **kwargs):
            buffer = zeros(*args, **kwargs)
            if rank == 1 and args == ((),) and kwargs.get("dtype") is torch.float32:
                # Replace only the selected operation on the preallocated timing scalar.
                original = getattr(buffer, operation)

                def fail_at_trial(*values):
                    nonlocal injected
                    if completed == fault_trial and not injected:
                        injected = True
                        error = torch.OutOfMemoryError if oom else RuntimeError
                        raise error("injected rank-local timing failure")
                    return original(*values)

                setattr(buffer, operation, fail_at_trial)
            return buffer

        with _cpu_calibration():
            with patch.object(optimizer, "_benchmark_workspace", run_trial), patch.object(torch, "zeros", allocate):
                if oom:
                    assert optimizer.autotune_workspace() == 160
                    assert optimizer._workspace_calibration_stats["out_of_memory_capacity_bytes"] == 320
                else:
                    with pytest.raises(RuntimeError, match="timing"):
                        optimizer.autotune_workspace()
                    assert optimizer._tuned_workspace_plan is None
            assert completed == fault_trial
            assert injected is (rank == 1)
            _assert_unchanged(optimizer, params, saved)
            if oom:
                assert optimizer._tuned_workspace_plan == WorkspacePlan(1, 160)
                optimizer.step()
                reference.step()
                _assert_matches_reference(optimizer, params, reference, reference_params)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not _HAS_GLOO, reason="requires torch.distributed with Gloo support")
@pytest.mark.parametrize(
    ("operation", "oom"), [("fill_", True), ("item", True), ("fill_", False)],
    ids=["write-oom", "read-oom", "write-error"],
)
def test_rank_local_timing_failure_is_coordinated(tmp_path, operation, oom):
    mp.spawn(_run_timing_failure, args=(str(tmp_path / "timing"), operation, oom), nprocs=2)
