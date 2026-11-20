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
    _assert_unchanged,
    _optimizer,
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
        patch.object(AdamWSwap, "_WORKSPACE_TARGET_BYTES", 128),
        patch.object(AdamWSwap, "_WORKSPACE_CHUNK_BYTES", 32),
        patch.object(AdamWSwap, "_WORKSPACE_PIPELINE_BYTES", 64),
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
        runtime = WorkspaceRuntime(free=256 if rank == 0 else 64)
        optimizer, params = _optimizer((33, 17, 4, 4) if rank == 0 else (4,), runtime)
        if rank == 1 and empty_peer:
            params[0].grad = None
        expected_bounds = (
            (32, 128) if rank == 0 else ((0, 0) if empty_peer else (32, 32))
        )
        bound = bind_parameters(
            optimizer._active_parameters(), optimizer.param_to_cpu_states_map
        )
        assert optimizer._workspace_bounds(bound) == expected_bounds

        # A larger rank-local slab need not fit inside another rank's budget.
        # Its injected allocation failure must not reallocate the peer's slab.
        attempts = []
        initialize = DeviceWorkspace.__init__

        def allocate(workspace, device_runtime, capacity):
            attempts.append(capacity)
            if rank == 0 and capacity > 64:
                raise torch.OutOfMemoryError("injected allocation failure on rank 0")
            initialize(workspace, device_runtime, capacity)

        with patch.object(DeviceWorkspace, "__init__", allocate):
            workspace = optimizer._reserve_workspace(
                expected_bounds[1], expected_bounds[0]
            )
        if rank == 0:
            assert attempts == [128, 64]
            assert workspace.capacity_bytes == 64
        else:
            assert attempts == ([] if empty_peer else [32])
            if empty_peer:
                assert workspace is None
            else:
                assert workspace.capacity_bytes == 32
        del workspace
        optimizer.release_workspace()
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
            return {32: 1.2, 64: 1.1, 128: 1.0}[capacity] if rank == 0 else 0.5

        saved = _snapshot(optimizer, params)
        with (
            patch.object(AdamWSwap, "_WORKSPACE_TARGET_BYTES", 128),
            patch.object(AdamWSwap, "_WORKSPACE_CHUNK_BYTES", 32),
            patch.object(AdamWSwap, "_WORKSPACE_PIPELINE_BYTES", 64),
            patch.object(AdamWSwap, "_run_adamw", _run_cpu_adamw),
            patch.object(torch, "empty_like", unpinned_empty_like),
            patch.object(torch, "empty", unpinned_empty),
            patch.object(optimizer, "_benchmark_workspace", measure),
        ):
            chosen = optimizer.autotune_workspace()
            assert chosen == (128 if rank == 0 else (0 if empty_peer else 32))
            expected_measurements = [128] * 4 + [32] * 4 + [64] * 4
            if rank == 1:
                expected_measurements = [0 if empty_peer else 32] * 12
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
            patch.object(AdamWSwap, "_WORKSPACE_TARGET_BYTES", 128),
            patch.object(AdamWSwap, "_WORKSPACE_CHUNK_BYTES", 32),
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
            assert optimizer._tuned_workspace_bytes is None
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
                assert optimizer._tuned_workspace_plan is not None
                assert optimizer._tuned_workspace_bytes == (32 if rank == 0 else 0)
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
            assert optimizer.autotune_workspace() == 32
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
