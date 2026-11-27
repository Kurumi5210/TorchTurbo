"""AdamW construction and optimizer-state CPU offload for FSDP2."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import chain, groupby
from statistics import median
from time import perf_counter
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer

from .....logging import get_logger
from .....runtime import DeviceRuntime, TurboDevice
from ....optimizers.plan import ParamGroup
from ..config import CPUOffloadConfig, CPUOffloadMode
from .adamw_workspace import (
    BoundParameter,
    DeviceWorkspace,
    PipelineStreams,
    WorkspacePlan,
    bind_parameters,
    calibration_sink_bytes,
    reset_workspace_moments,
    run_workspace,
    single_chunk_slot_bytes,
)
from .state import wrap_cpu_state

__all__ = [
    "AdamWSwap",
    "prepare_fsdp_adamw_optimizer",
]

logger = get_logger(__name__)


class _WorkspaceRestoreError(RuntimeError):
    """Calibration could not restore the parameter values it saved."""

    def __init__(self, detail: str = "") -> None:
        super().__init__(
            f"AdamW calibration could not restore parameter values on a training rank{detail}; "
            "reload the model checkpoint and recreate the optimizer before continuing"
        )


class _CoordinatedWorkspaceOOM(torch.OutOfMemoryError):
    """Every rank agreed that a local workspace operation ran out of memory."""


def prepare_fsdp_adamw_optimizer(
    groups: Sequence[ParamGroup],
    optimizer_kwargs: Mapping[str, Any],
    *,
    device_runtime: DeviceRuntime,
    cpu_offload: CPUOffloadConfig,
    split_param_groups: Callable[
        [Sequence[Mapping[str, Any]]],
        list[dict[str, Any]],
    ],
) -> Optimizer:
    """Build native AdamW, or AdamWSwap under optimizer-state offload.

    Native AdamW plans and the Muon fallback both use this function, so they
    apply the same offload policy and FSDP topology split.
    """
    param_groups = split_param_groups(
        tuple(group.get_optimizer_param_group() for group in groups)
    )
    options = dict(optimizer_kwargs)
    if cpu_offload.enabled and cpu_offload.mode is CPUOffloadMode.OPTIMIZER:
        # AdamWSwap always runs the fused kernel and has no fused option.
        options.pop("fused", None)
        return AdamWSwap(
            param_groups,
            **options,
            device_runtime=device_runtime,
            optimizer_memory_fraction=cpu_offload.optimizer_memory_fraction,
            param_names={
                param: name
                for group in groups
                for name, param in group.params_by_fqn.items()
            },
        )
    return torch.optim.AdamW(param_groups, **options)


class AdamWSwapMixin:
    """Run AdamW with CPU-resident optimizer state."""

    state_keys = ("exp_avg", "exp_avg_sq")
    _AUTO_WORKSPACE_FRACTION = 0.9
    _WORKSPACE_CACHE_MARGIN_BYTES = 64 << 20

    def __init__(
        self,
        device_runtime: DeviceRuntime,
        optimizer_memory_fraction: float | None,
        param_names: Mapping[torch.Tensor, str] | None = None,
    ) -> None:
        self.device_runtime = device_runtime
        self.optimizer_memory_fraction = optimizer_memory_fraction
        self.param_to_name_map = dict(param_names) if param_names is not None else {}
        self.param_to_cpu_states_map = {}
        self.param_to_step_map = {}
        self.workspace: DeviceWorkspace | None = None
        self._tuned_workspace_plan: WorkspacePlan | None = None
        # (shared plan, slowest-rank median seconds, total HBM over ranks); identical on every rank.
        self._workspace_measurements: list[tuple[WorkspacePlan, float, int]] = []
        self._workspace_plan = WorkspacePlan(1, 0)
        self._pipeline: PipelineStreams | None = None
        self._workspace_update_failed = False
        self._workspace_restore_failed = False
        self._workspace_adamw_logged = False
        self._workspace_calibration_stats: dict[str, Any] = {}
        self._checkpoint_meshes: dict[DeviceMesh, DeviceMesh] = {}
        self._opt_states_initialization()

    @staticmethod
    def _reduce_workspace_value(tensor: torch.Tensor, op) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=op, group=dist.group.WORLD)

    def _active_parameters(self):
        return [
            (param, group)
            for group in self.param_groups
            for param in group["params"]
            if param.grad is not None
        ]

    def _workspace_bounds(
        self, bound: list[BoundParameter], *, calibration: bool = False
    ) -> tuple[int, int]:
        """Return aligned bounds; only real updates may reuse allocator cache."""
        minimum = max((param.minimum_slot_bytes for param in bound), default=0)
        if not minimum:
            return 0, 0
        alignment = 2 * DeviceWorkspace._ALIGNMENT_BYTES
        runtime = self.device_runtime
        free, _ = runtime.memory_info()
        fraction = self.optimizer_memory_fraction
        if fraction is None and not calibration:
            free += runtime.memory_reserved() - runtime.memory_allocated()
        fraction = self._AUTO_WORKSPACE_FRACTION if fraction is None else fraction
        budget = int(free * fraction)
        budget = budget // alignment * alignment
        if minimum > budget:
            raise RuntimeError(
                f"AdamW workspace budget is {budget} bytes after {alignment}-byte alignment, "
                f"but at least {minimum} bytes are required for one aligned pair of moments "
                f"(available={free} bytes, fraction={fraction})"
            )
        return minimum, budget

    def _get_pipeline(self, plan: WorkspacePlan) -> PipelineStreams | None:
        if plan.num_slots == 1 or not plan.capacity_bytes:
            return None
        if self._pipeline is None:
            self._pipeline = PipelineStreams(self.device_runtime, num_slots=2)
        return self._pipeline

    def _reserve_workspace(
        self, capacity: int, *, status: torch.Tensor, error=None, refit: bool = False
    ) -> DeviceWorkspace | None:
        """Allocate once; no rank proceeds if another rank fails.

        Status codes are reduced with MIN: 0 error, 1 allocation out of memory,
        2 refit, 3 allocated. ``refit`` reports that this rank cannot use the
        tuned plan; every rank then leaves ``status`` at 2 and returns without a
        workspace. ``torch.OutOfMemoryError`` is raised on every rank only when
        every failing rank ran out of memory in this allocation. Nothing has
        been updated at that point, so callers may retry a smaller capacity.
        """
        workspace = None
        out_of_memory = False
        if error is None and capacity and not refit:
            try:
                workspace = DeviceWorkspace(self.device_runtime, capacity)
            except Exception as allocation_error:  # noqa: BLE001 - Coordinate all ranks.
                error = allocation_error.with_traceback(None)
                out_of_memory = self._is_out_of_memory(error)
        if error is None:
            status.fill_(2 if refit else 3)
        else:
            status.fill_(1 if out_of_memory else 0)
        self._reduce_workspace_value(status, dist.ReduceOp.MIN)
        code = status.item()
        if code == 2:
            return None
        if code < 2:
            workspace = None
            detail = f": {error}" if error is not None else ""
            message = (
                "AdamW workspace allocation failed on a training rank "
                f"before any update (local capacity={capacity} bytes){detail}"
            )
            if code == 1:
                raise _CoordinatedWorkspaceOOM(message) from error
            raise RuntimeError(message) from error
        self.workspace = workspace
        return workspace

    @staticmethod
    def _is_out_of_memory(error: RuntimeError) -> bool:
        return isinstance(error, torch.OutOfMemoryError) or (
            "out of memory" in str(error).lower()
        )

    def _check_workspace_restore(self) -> None:
        if self._workspace_update_failed:
            raise RuntimeError(
                "AdamW update failed after it started on a training rank; "
                "reload the model checkpoint and recreate the optimizer before continuing"
            )
        if self._workspace_restore_failed:
            raise _WorkspaceRestoreError()

    def _fixed_workspace_plan(self, bound, error) -> WorkspacePlan:
        capacity = 0
        inactive = torch.iinfo(torch.int64).max
        try:
            if error is None:
                _, capacity = self._workspace_bounds(bound)
        except Exception as exc:
            error = exc.with_traceback(None)
        shared = torch.tensor(
            -1 if error is not None else (capacity or inactive),
            device=self.device_runtime.device, dtype=torch.int64,
        )
        self._reduce_workspace_value(shared, dist.ReduceOp.MIN)
        if shared.item() < 0:
            detail = f": {error}" if error is not None else ""
            raise RuntimeError(f"AdamW fixed workspace validation failed before any update{detail}") from error
        return WorkspacePlan(1, shared.item() if capacity else 0)

    def _acquire_workspace(
        self, bound=None, *, error: Exception | None = None
    ) -> DeviceWorkspace | None:
        """Allocate the per-step workspace for AdamW state staging."""
        self.release_workspace()
        self.device_runtime.synchronize()
        # Reserve the status scalar before allocation; a retry reuses it for a byte budget.
        status = torch.ones(1, dtype=torch.int64, device=self.device_runtime.device)
        plan = WorkspacePlan(1, 0)
        try:
            if error is not None:
                raise error
            if bound is None:
                bound = bind_parameters(
                    self._active_parameters(), self.param_to_cpu_states_map
                )
        except Exception as binding_error:
            error = binding_error.with_traceback(None)
        if self.optimizer_memory_fraction is not None:
            plan = self._fixed_workspace_plan(bound, error)
            self._workspace_plan = plan
            return self._reserve_workspace(plan.capacity_bytes, status=status)
        maximum = 0
        refit = False
        try:
            if error is not None:
                raise error
            _, maximum = self._workspace_bounds(bound)
            target = self._tuned_workspace_plan
            if maximum:
                if target is None or target.capacity_bytes == 0:
                    raise RuntimeError(
                        "active rank has no calibrated AdamW workspace; recalibrate on all ranks"
                    )
                refit = target.capacity_bytes > maximum
                if not refit:
                    plan = target
            self._get_pipeline(plan)
        except Exception as budget_error:  # noqa: BLE001 - Propagate errors to every rank.
            error = budget_error.with_traceback(None)
        shared = self._tuned_workspace_plan
        cause = None
        while True:
            try:
                workspace = self._reserve_workspace(
                    plan.capacity_bytes, status=status, error=error, refit=refit
                )
            except _CoordinatedWorkspaceOOM as out_of_memory:
                # Allocation precedes every counter or value change, so every
                # rank may retry a measured plan below the capacity that failed.
                if shared is None or not self._workspace_measurements:
                    raise
                maximum = min(maximum, shared.capacity_bytes - 1) if maximum else 0
                cause = out_of_memory.with_traceback(None)
            else:
                if status.item() != 2:
                    break
            # Rare path: every rank reaches it and selects the same shared plan.
            shared = self._refit_workspace_plan(maximum, status, cause)
            cause = None
            plan = shared if maximum else WorkspacePlan(1, 0)
            refit, error = False, None
            try:
                self._get_pipeline(plan)
            except Exception as pipeline_error:  # noqa: BLE001 - Propagate errors to every rank.
                error = pipeline_error.with_traceback(None)
        self._workspace_plan = plan
        return workspace

    def _refit_workspace_plan(
        self, maximum: int, status: torch.Tensor, cause: Exception | None = None
    ) -> WorkspacePlan:
        """Select a measured plan that fits every active rank, or fail on all ranks."""
        status.fill_(maximum or torch.iinfo(torch.int64).max)
        self._reduce_workspace_value(status, dist.ReduceOp.MIN)
        shared_maximum = status.item()
        fitting = [
            entry
            for entry in self._workspace_measurements
            if entry[0].capacity_bytes <= shared_maximum
        ]
        reason = "ran out of memory" if cause is not None else "no longer fits"
        if not fitting:
            raise RuntimeError(
                f"selected AdamW workspace {self._tuned_workspace_plan} {reason} "
                f"(maximum total={shared_maximum} bytes) and no "
                "measured candidate fits; recalibrate on all ranks"
            ) from cause
        selected = self._select_measurement(fitting)
        logger.warning_rank0(
            "AdamW workspace %s %s; switching to measured plan %s",
            self._tuned_workspace_plan,
            reason,
            selected[0],
        )
        self._tuned_workspace_plan = selected[0]
        return selected[0]

    @staticmethod
    def _select_measurement(measurements):
        """Least total HBM within 3% of the fastest median."""
        fastest = min(elapsed for _, elapsed, _ in measurements)
        return min(
            (entry for entry in measurements if entry[1] <= fastest * 1.03),
            key=lambda entry: (entry[2], entry[1]),
        )

    @staticmethod
    def _workspace_candidates(
        shard_bytes: int, budget: int, overlap: bool, full_slot: int | None = None
    ) -> tuple[WorkspacePlan, ...]:
        """Double the slot size from the largest active shard to each aligned endpoint.

        ``full_slot`` is the largest rank's single-chunk slot. Larger slots only
        add HBM, and two slots overlap nothing unless that rank has two chunks.
        """
        alignment = 2 * DeviceWorkspace._ALIGNMENT_BYTES
        start = (shard_bytes + alignment - 1) // alignment * alignment
        if start <= 0 or start > budget:
            raise RuntimeError(
                f"AdamW workspace cannot probe M={shard_bytes} bytes (aligned={start}, budget={budget})"
            )
        plans = []
        for slots in (1, 2) if overlap else (1,):
            endpoint = budget // slots // alignment * alignment
            if full_slot is not None:
                limit = max(full_slot, start) if slots == 1 else full_slot - alignment
                endpoint = min(endpoint, limit)
            if endpoint < start:
                continue
            size = start
            while size < endpoint:
                plans.append(WorkspacePlan(slots, size))
                size *= 2
            plans.append(WorkspacePlan(slots, endpoint))
        return tuple(sorted(plans, key=lambda plan: (plan.slot_bytes, plan.num_slots)))

    @torch.no_grad()
    def autotune_workspace(self) -> int:
        """Calibrate chunk size and overlap after a representative backward.

        Call on every training rank, or let the first device step call it.
        Complete trial updates are timed, then parameter values are restored
        from one reusable CPU snapshot. Verified zero moments use their real CPU
        destinations and are reset after every trial. Nonzero moments use an
        independent output ring. Gradients and real counters stay unchanged.
        A retained autograd graph must not be reused afterwards.
        A candidate that runs out of device memory on any rank is skipped with
        every larger capacity once all ranks have restored their state.
        Returns the selected total workspace capacity in bytes on this rank.
        """
        self._check_workspace_restore()
        if self.optimizer_memory_fraction == 0:
            return 0
        bound = []
        error = None
        try:
            bound = bind_parameters(
                self._active_parameters(), self.param_to_cpu_states_map
            )
        except Exception as binding_error:  # noqa: BLE001 - Propagate errors to every rank.
            error = binding_error.with_traceback(None)
        if self.optimizer_memory_fraction is not None:
            return self._fixed_workspace_plan(bound, error).capacity_bytes
        return self._autotune_workspace(bound, error=error)

    def _autotune_workspace(
        self, bound: list[BoundParameter], *, error: Exception | None = None
    ) -> int:
        """Calibrate using the same bindings as the upcoming update."""
        started = perf_counter()
        self._check_workspace_restore()
        runtime = self.device_runtime
        if runtime.turbo_device is TurboDevice.CPU:
            raise RuntimeError(
                "workspace is unavailable; calibration requires CUDA/NPU"
            )
        self.release_workspace()
        runtime.synchronize()
        status = torch.ones(3, dtype=torch.int64, device=runtime.device)
        allocation_status = torch.ones(1, dtype=torch.int64, device=runtime.device)
        control = torch.zeros(1, dtype=torch.int64, device=runtime.device)
        # Largest active shard and single-chunk slot, reduced together.
        shard = torch.zeros(2, dtype=torch.int64, device=runtime.device)
        limits = torch.ones(2, dtype=torch.int64, device=runtime.device)
        timing = torch.zeros((), dtype=torch.float32, device=runtime.device)
        stats = self._workspace_calibration_stats = {
            "reserved_before_bytes": runtime.memory_reserved(),
            "cleanup_cleared_cache": False,
            "output_mode": "none",
        }
        minimum = maximum = 0
        try:
            if error is not None:
                raise error
            minimum, maximum = self._workspace_bounds(bound, calibration=True)
            shard[0] = max((param.local_param.nbytes for param in bound), default=0)
            shard[1] = single_chunk_slot_bytes(bound)
            limits[0] = maximum if minimum else torch.iinfo(torch.int64).max
            limits[1] = not minimum or runtime.supports_async_streams
        except Exception as budget_error:  # noqa: BLE001 - Propagate errors to every rank.
            error = budget_error.with_traceback(None)
        self._check_benchmark_phase(
            status, error, "budget validation before any update"
        )
        self._reduce_workspace_value(shard, dist.ReduceOp.MAX)
        max_shard, full_slot = shard.tolist()
        stats["setup_seconds"] = perf_counter() - started
        if not max_shard:
            stats["total_seconds"] = perf_counter() - started
            return 0
        self._reduce_workspace_value(limits, dist.ReduceOp.MIN)
        budget, overlap = limits.tolist()
        shared_candidates = self._workspace_candidates(
            max_shard, budget, bool(overlap), full_slot
        )
        stats.update(
            max_shard_bytes=max_shard,
            single_chunk_slot_bytes=full_slot,
            probe_budget_bytes=budget,
        )
        # Empty ranks run every trial without allocating, but keep the shared plan.
        candidates = (
            shared_candidates
            if minimum
            else (WorkspacePlan(1, 0),) * len(shared_candidates)
        )

        measurements = []
        saved_params = []
        cpu_sink = None
        selected_capacity = 0
        trials_started = None
        try:
            preparation_started = perf_counter()
            error = None
            try:
                saved_params = [
                    param.local_param.detach().to(device="cpu", copy=True)
                    for param in bound
                ]
                stats["snapshot_parameter_bytes"] = sum(
                    param.nbytes for param in saved_params
                )
                zero_check_started = perf_counter()
                zero_moments = all(
                    torch.count_nonzero(moment).item() == 0
                    for param in bound
                    for moment in (param.exp_avg, param.exp_avg_sq)
                )
                stats["zero_check_seconds"] = perf_counter() - zero_check_started
                if zero_moments:
                    stats["output_mode"] = "real_zero_moments"
                else:
                    sink_bytes = calibration_sink_bytes(bound, max(plan.capacity_bytes for plan in candidates))
                    cpu_sink = torch.empty(
                        sink_bytes, dtype=torch.uint8, pin_memory=True
                    )
                    stats["output_mode"] = "scratch"
                stats["output_bytes"] = cpu_sink.nbytes if cpu_sink is not None else 0
                for plan in candidates:
                    self._get_pipeline(plan)
                runtime.synchronize()
            except Exception as preparation_error:  # noqa: BLE001 - Propagate errors to every rank.
                error = preparation_error.with_traceback(None)
            self._check_benchmark_phase(status, error, "snapshot preparation")
            stats["snapshot_seconds"] = perf_counter() - preparation_started
            trials_started = perf_counter()
            ceiling = None
            for plan, shared_plan in zip(candidates, shared_candidates):
                # Candidates are ordered by slot size, not capacity.
                if ceiling is not None and shared_plan.capacity_bytes >= ceiling:
                    continue
                self.release_workspace()
                try:
                    self._reserve_workspace(
                        plan.capacity_bytes, status=allocation_status
                    )
                    self._workspace_plan = plan
                    samples = []
                    for trial in range(4):
                        elapsed = self._benchmark_workspace(
                            bound, status, saved_params, cpu_sink
                        )
                        self._record_trial_time(timing, elapsed, status, samples, warmup=trial == 0)
                except _CoordinatedWorkspaceOOM as error:
                    # Only a completed status reduction can authorize a shared retry.
                    ceiling = shared_plan.capacity_bytes
                    stats["out_of_memory_capacity_bytes"] = ceiling
                    # The first candidate is the smallest. Keep the exception
                    # only on failure; trial tracebacks reference the CPU sink.
                    if not measurements:
                        raise torch.OutOfMemoryError(
                            "AdamW workspace calibration ran out of memory at the smallest "
                            f"candidate ({ceiling} bytes): {error}"
                        ) from error.with_traceback(None)
                    logger.warning_rank0(
                        "AdamW workspace calibration ran out of memory on a training "
                        "rank at capacity=%d bytes; skipping it and larger candidates",
                        ceiling,
                    )
                    continue
                control.fill_(plan.capacity_bytes)
                self._reduce_workspace_value(control, dist.ReduceOp.SUM)
                elapsed = median(samples)
                measurements.append((shared_plan, elapsed, control.item()))
                logger.info_rank0(
                    "AdamW workspace calibration: rank0 slots=%d, slot=%d bytes, "
                    "capacity=%d bytes, step=%.3f ms",
                    plan.num_slots,
                    plan.slot_bytes,
                    plan.capacity_bytes,
                    elapsed * 1000,
                )
            selected = self._select_measurement(measurements)
            self._workspace_measurements = measurements
            self._tuned_workspace_plan = selected[0]
            selected_capacity = selected[0].capacity_bytes if minimum else 0
            logger.info_rank0(
                "AdamW workspace selected: rank0 slots=%d, slot=%d bytes, "
                "capacity=%d bytes, step=%.3f ms, output=%s "
                "(least total HBM within 3%% of fastest)",
                selected[0].num_slots,
                selected[0].slot_bytes,
                selected_capacity,
                selected[1] * 1000,
                stats["output_mode"],
            )
            return selected_capacity
        finally:
            if trials_started is not None:
                stats["trials_seconds"] = perf_counter() - trials_started
            cleanup_started = perf_counter()
            saved_params.clear()
            if cpu_sink is not None:
                # The caching host allocator would otherwise keep the pinned
                # sink until exit. Every trial has synchronized its copies.
                cpu_sink = None
                runtime.empty_host_cache()
            stats["release_host_seconds"] = perf_counter() - cleanup_started
            cleanup_started = perf_counter()
            self.release_workspace()
            stats["release_workspace_seconds"] = perf_counter() - cleanup_started
            reserved = runtime.memory_reserved()
            stats["reserved_after_trials_bytes"] = reserved
            # Retain the selected slab and allocator rounding, and preserve
            # cached forward/backward storage when trials reused that storage.
            allowance = selected_capacity + self._WORKSPACE_CACHE_MARGIN_BYTES
            cleanup_started = perf_counter()
            if reserved - stats["reserved_before_bytes"] > allowance:
                runtime.empty_cache()
                stats["cleanup_cleared_cache"] = True
            stats["empty_cache_seconds"] = perf_counter() - cleanup_started
            stats["reserved_after_cleanup_bytes"] = runtime.memory_reserved()
            stats["total_seconds"] = perf_counter() - started
            logger.info_rank0(
                "AdamW workspace calibration: rank0 total=%.3f s, setup=%.3f s, "
                "snapshot=%.3f s, trials=%.3f s, release_host=%.3f s, "
                "release_workspace=%.3f s, empty_cache=%.3f s, cleanup_cleared_cache=%s",
                stats["total_seconds"],
                stats["setup_seconds"],
                stats.get("snapshot_seconds", 0.0),
                stats.get("trials_seconds", 0.0),
                stats["release_host_seconds"],
                stats["release_workspace_seconds"],
                stats["empty_cache_seconds"],
                stats["cleanup_cleared_cache"],
            )

    def release_workspace(self) -> None:
        """Synchronize and release the workspace."""
        if self.workspace is None:
            return
        self.device_runtime.synchronize()
        self._drop_workspace()

    def _drop_workspace(self) -> None:
        """Release storage after the caller has finished all workspace operations."""
        assert self.workspace is not None
        capacity_mib = self.workspace.capacity_bytes // (1024 * 1024)
        self.workspace = None
        logger.debug(
            "FSDP2 workspace released: capacity=%d MiB",
            capacity_mib,
        )

    def _opt_states_initialization(self) -> None:
        """Zero the swap state and publish it under ``self.state``, to avoid
        the helper faking a zero-grad / lr-0 ``step()``, which would decay the
        CPU moments and bump the step counters on every checkpoint save/load.
        """
        for group in self.param_groups:
            for param in group["params"]:
                local_param = param.to_local() if isinstance(param, DTensor) else param
                cpu_state = {
                    key: torch.empty_like(
                        local_param,
                        device="cpu",
                        pin_memory=self.optimizer_memory_fraction != 0,
                    ).zero_()
                    for key in self.state_keys
                }
                step = torch.zeros(
                    (),
                    dtype=torch.float32,
                    device=(
                        "cpu"
                        if self.optimizer_memory_fraction == 0
                        else local_param.device
                    ),
                )
                state = {"step": step, **cpu_state}
                self.state[param] = state
                self.param_to_cpu_states_map[param] = state
                self.param_to_step_map[param] = step

    def _run_adamw(
        self,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        states: list[dict[str, torch.Tensor]],
        steps: list[torch.Tensor],
        group: dict,
    ) -> None:
        beta1, beta2 = group["betas"]
        torch._fused_adamw_(
            params,
            grads,
            [state["exp_avg"] for state in states],
            [state["exp_avg_sq"] for state in states],
            [],
            steps,
            lr=group["lr"],
            beta1=beta1,
            beta2=beta2,
            weight_decay=group["weight_decay"],
            eps=group["eps"],
            amsgrad=False,
            maximize=group["maximize"],
        )

    def _advance_steps(self, params_with_groups) -> None:
        steps = [self.param_to_step_map[param] for param, _ in params_with_groups]
        if steps:
            torch._foreach_add_(steps, 1)

    def _step_on_device(self, params_with_groups) -> None:
        if self.device_runtime.turbo_device is TurboDevice.CPU:
            raise RuntimeError(
                "workspace is unavailable; AdamW device update requires a CUDA/NPU runtime"
            )
        bound = []
        error = None
        try:
            bound = bind_parameters(params_with_groups, self.param_to_cpu_states_map)
        except Exception as binding_error:  # noqa: BLE001 - Propagate errors to every rank.
            error = binding_error.with_traceback(None)
        if self.optimizer_memory_fraction is None and self._tuned_workspace_plan is None:
            self._autotune_workspace(bound, error=error)
        status = torch.ones(1, dtype=torch.int32, device=self.device_runtime.device)
        self._acquire_workspace(bound, error=error)
        # Allocation and stream creation must finish before counters or values change.
        error = None
        try:
            self._advance_steps(params_with_groups)
            if self.workspace is not None:
                self._step_on_device_workspace(bound)
                self._drop_workspace()
        except Exception as update_error:  # noqa: BLE001 - Propagate errors to every rank.
            error = update_error.with_traceback(None)
            status.zero_()
        self._reduce_workspace_value(status, dist.ReduceOp.MIN)
        if not status.item():
            self._workspace_update_failed = True
            raise RuntimeError(
                "AdamW update failed after it started on a training rank; "
                "reload the model checkpoint and recreate the optimizer before continuing"
            ) from error

    def _parameter_name(self, param: torch.Tensor) -> str:
        return self.param_to_name_map.get(param, "<unnamed parameter>")

    def _wrap_checkpoint_tensor(
        self, tensor: torch.Tensor, param: torch.Tensor
    ) -> torch.Tensor:
        return wrap_cpu_state(tensor, param, self._checkpoint_meshes)

    def _check_benchmark_phase(
        self,
        status: torch.Tensor,
        error: Exception | None,
        phase: str,
    ) -> None:
        """Finish a phase together, preserving the failure type across ranks."""
        started = perf_counter()
        status.fill_(1)
        if error is not None:
            status[0] = 0
            status[1] = self._is_out_of_memory(error)
            status[2] = phase != "restore"
        self._reduce_workspace_value(status, dist.ReduceOp.MIN)
        succeeded, out_of_memory, restored = status.tolist()
        stats = self._workspace_calibration_stats
        stats["coordination_seconds"] = stats.get("coordination_seconds", 0.0) + (
            perf_counter() - started
        )
        detail = f": {error}" if error is not None else ""
        if not restored:
            self._workspace_restore_failed = True
            raise _WorkspaceRestoreError(detail) from error
        if not succeeded:
            message = (
                f"AdamW workspace calibration {phase} failed on a training rank"
                + detail
            )
            if out_of_memory:
                raise _CoordinatedWorkspaceOOM(message) from error
            raise RuntimeError(message) from error

    def _record_trial_time(
        self,
        timing: torch.Tensor,
        elapsed: float,
        status: torch.Tensor,
        samples: list[float],
        *,
        warmup: bool,
    ) -> None:
        """Agree that local timing operations succeeded before advancing together."""
        error = None
        try:
            timing.fill_(elapsed)
            self.device_runtime.synchronize()
        except Exception as exc:  # noqa: BLE001 - Report the local write to every rank.
            error = exc.with_traceback(None)
        self._check_benchmark_phase(status, error, "timing write")

        # A transport error can surface at synchronization. Let it leave the sweep.
        self._reduce_workspace_value(timing, dist.ReduceOp.MAX)
        self.device_runtime.synchronize()
        if warmup:
            return
        try:
            samples.append(timing.item())
        except Exception as exc:  # noqa: BLE001 - Report the local read to every rank.
            error = exc.with_traceback(None)
        self._check_benchmark_phase(status, error, "timing read")

    def _benchmark_workspace(
        self,
        bound: list[BoundParameter],
        status: torch.Tensor,
        saved_params: list[torch.Tensor],
        cpu_sink: torch.Tensor | None,
    ) -> float:
        """Time one complete fill/compute/drain, with no per-chunk collectives.

        A missing sink requires verified zero moments, restored after the trial.
        """
        runtime = self.device_runtime
        stats = self._workspace_calibration_stats
        preparation_started = perf_counter()
        steps = {}
        preparation_error = None
        try:
            if bound:
                step_values = torch.stack(
                    [self.param_to_step_map[param.original] for param in bound]
                ).add_(1)
                steps = {
                    param.original: step_values[index]
                    for index, param in enumerate(bound)
                }
            runtime.synchronize()
        except Exception as error:  # noqa: BLE001 - Propagate errors to every rank.
            preparation_error = error.with_traceback(None)
        self._check_benchmark_phase(status, preparation_error, "prepare")
        stats["prepare_seconds"] = stats.get("prepare_seconds", 0.0) + (
            perf_counter() - preparation_started
        )
        execution_error = None
        elapsed = 0.0
        try:
            try:
                started = perf_counter()
                if self.workspace is not None:
                    run_workspace(
                        bound,
                        self._workspace_plan,
                        self.workspace,
                        steps,
                        self._run_adamw,
                        pipeline=self._get_pipeline(self._workspace_plan),
                        cpu_sink=cpu_sink,
                    )
                elapsed = perf_counter() - started
                stats["execute_seconds"] = stats.get("execute_seconds", 0.0) + elapsed
            except Exception as error:  # noqa: BLE001 - Propagate errors to every rank.
                execution_error = error.with_traceback(None)
            self._check_benchmark_phase(status, execution_error, "execute")
        finally:
            restore_started = perf_counter()
            restore_error = None
            drained = True
            if execution_error is not None:
                # Do not rewrite host inputs while a failed submission might
                # still be transferring them on a device stream.
                try:
                    runtime.synchronize()
                except Exception as error:  # noqa: BLE001 - Propagate errors to every rank.
                    restore_error = error.with_traceback(None)
                    drained = False
            parameter_restore_started = perf_counter()
            for param, saved in zip(bound, saved_params):
                try:
                    param.local_param.copy_(saved)
                except Exception as error:  # noqa: BLE001 - Propagate errors to every rank.
                    restore_error = error.with_traceback(None)
            stats["parameter_restore_seconds"] = stats.get(
                "parameter_restore_seconds", 0.0
            ) + (perf_counter() - parameter_restore_started)
            moment_reset_started = perf_counter()
            if cpu_sink is None and drained:
                if execution_error is None and self.workspace is not None:
                    try:
                        reset_workspace_moments(
                            bound, self._workspace_plan, self.workspace
                        )
                    except Exception as error:  # noqa: BLE001 - Propagate errors to every rank.
                        restore_error = error.with_traceback(None)
                else:
                    for param in bound:
                        for moment in (param.exp_avg, param.exp_avg_sq):
                            try:
                                moment.zero_()
                            except Exception as error:  # noqa: BLE001 - Propagate errors to every rank.
                                restore_error = error.with_traceback(None)
            stats["moment_reset_seconds"] = stats.get("moment_reset_seconds", 0.0) + (
                perf_counter() - moment_reset_started
            )
            try:
                runtime.synchronize()
            except Exception as error:  # noqa: BLE001 - Propagate errors to every rank.
                restore_error = error.with_traceback(None)
            stats["restore_seconds"] = stats.get("restore_seconds", 0.0) + (
                perf_counter() - restore_started
            )
            self._check_benchmark_phase(status, restore_error, "restore")
        return elapsed

    def _step_on_device_workspace(self, bound: list[BoundParameter]) -> None:
        """Update fresh parameter views with the selected chunk and stream plan."""
        assert self.workspace is not None
        if not self._workspace_adamw_logged:
            logger.info_rank0(
                "FSDP2 workspace: AdamW slots=%d, slot=%d MiB, capacity=%d MiB, "
                "parameters=%d",
                self._workspace_plan.num_slots,
                self._workspace_plan.slot_bytes // (1024 * 1024),
                self._workspace_plan.capacity_bytes // (1024 * 1024),
                len(bound),
            )
            self._workspace_adamw_logged = True
        run_workspace(
            bound,
            self._workspace_plan,
            self.workspace,
            self.param_to_step_map,
            self._run_adamw,
            pipeline=self._get_pipeline(self._workspace_plan),
        )

    def _step_on_host(self, params_with_groups) -> None:
        for _, group_entries in groupby(
            params_with_groups, key=lambda entry: id(entry[1])
        ):
            batch = list(group_entries)
            group = batch[0][1]
            local_params = [
                param.to_local() if isinstance(param, DTensor) else param
                for param, _ in batch
            ]
            local_grads = [
                param.grad.to_local() if isinstance(param.grad, DTensor) else param.grad
                for param, _ in batch
            ]
            cpu_params = [
                torch.empty_like(param, device="cpu", pin_memory=True)
                for param in local_params
            ]
            cpu_grads = [
                torch.empty_like(grad, device="cpu", pin_memory=True)
                for grad in local_grads
            ]
            for cpu_param, local_param in zip(cpu_params, local_params):
                cpu_param.copy_(local_param.detach())
            for cpu_grad, local_grad in zip(cpu_grads, local_grads):
                cpu_grad.copy_(local_grad)
            self._advance_steps(batch)
            self._run_adamw(
                cpu_params,
                cpu_grads,
                [self.param_to_cpu_states_map[param] for param, _ in batch],
                [self.param_to_step_map[param] for param, _ in batch],
                group,
            )
            for local_param, cpu_param in zip(local_params, cpu_params):
                local_param.copy_(cpu_param)


class AdamWSwap(torch.optim.AdamW, AdamWSwapMixin):
    """Fused AdamW that keeps optimizer state on CPU between steps."""

    def __init__(
        self,
        params,
        lr: float | Tensor = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        *,
        maximize: bool = False,
        device_runtime: DeviceRuntime,
        optimizer_memory_fraction: float | None = None,
        param_names: Mapping[torch.Tensor, str] | None = None,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=False,
            foreach=False,
            maximize=maximize,
            capturable=False,
            differentiable=False,
            fused=True,
        )
        # Let GradScaler unscale and check overflow before the low-level fused call.
        self._step_supports_amp_scaling = False
        AdamWSwapMixin.__init__(
            self, device_runtime, optimizer_memory_fraction, param_names
        )

    def state_dict(self, *args, **kwargs):
        """Return a standard AdamW-format checkpoint view of the swap state.

        Sharded moments are wrapped in metadata-only CPU DTensors while the
        state and param_groups keys remain standard packed integer IDs.
        """
        state_dict = super().state_dict(*args, **kwargs)
        params = chain.from_iterable(group["params"] for group in self.param_groups)
        for (param_id, entry), param in zip(state_dict["state"].items(), params):
            checkpoint_entry = dict(entry)
            for key in self.state_keys:
                checkpoint_entry[key] = self._wrap_checkpoint_tensor(entry[key], param)
            state_dict["state"][param_id] = checkpoint_entry
        return state_dict

    def load_state_dict(self, state_dict):
        """Restore swap state from a standard AdamW-format state dict.

        Unlike the base implementation, the loaded tensors are copied into the
        existing buffers instead of being adopted.
        """
        state_dict = state_dict.copy()
        for hook in self._optimizer_load_state_dict_pre_hooks.values():
            replacement = hook(self, state_dict)
            if replacement is not None:
                state_dict = replacement
        saved_groups = state_dict["param_groups"]
        groups = self.param_groups
        if len(groups) != len(saved_groups):
            raise ValueError(
                f"loaded state dict has {len(saved_groups)} parameter groups; "
                f"optimizer has {len(groups)}"
            )
        if [len(group["params"]) for group in groups] != [
            len(group["params"]) for group in saved_groups
        ]:
            raise ValueError(
                "loaded state dict has a different number of parameters per "
                "group than the optimizer"
            )

        for group, saved in zip(groups, saved_groups):
            for key, value in saved.items():
                if key != "params":
                    group[key] = value

        current_params = [p for group in groups for p in group["params"]]
        saved_ids = [p for group in saved_groups for p in group["params"]]
        param_map = dict(zip(saved_ids, current_params))

        for saved_id, saved_state in state_dict["state"].items():
            param = param_map.get(saved_id)
            if param is None:
                raise ValueError(
                    "loaded state dict contains state for unknown parameter "
                    f"id {saved_id!r}"
                )
            cpu_state = self.param_to_cpu_states_map.get(param)
            if cpu_state is None:
                raise ValueError(
                    "loaded state dict contains state for a parameter without "
                    "swap state"
                )
            for key, value in saved_state.items():
                if isinstance(value, DTensor):
                    value = value.to_local()
                if key == "step":
                    step = self.param_to_step_map[param]
                    if not torch.is_tensor(value):
                        value = torch.tensor(value)
                    self._validate_state_shape(param, key, value, step)
                    step.copy_(value)
                elif key in self.state_keys:
                    self._validate_state_shape(param, key, value, cpu_state[key])
                    cpu_state[key].copy_(value)
                # Other keys (e.g. amsgrad ``max_exp_avg_sq``) have no swap
                # counterpart and are ignored.
        for hook in self._optimizer_load_state_dict_post_hooks.values():
            hook(self)

    def _validate_state_shape(
        self,
        param: torch.Tensor,
        key: str,
        value: torch.Tensor,
        destination: torch.Tensor,
    ) -> None:
        if value.shape != destination.shape:
            raise ValueError(
                f"loaded optimizer state '{key}' for parameter "
                f"'{self._parameter_name(param)}' has shape "
                f"{tuple(value.shape)}, expected {tuple(destination.shape)}"
            )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        try:
            self._check_workspace_restore()
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            params_with_groups = self._active_parameters()
            if self.optimizer_memory_fraction == 0:
                self._step_on_host(params_with_groups)
            else:
                self._step_on_device(params_with_groups)
        finally:
            self.release_workspace()
        return loss

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True):
        """Release a stale workspace when AMP skips the optimizer update."""
        self.release_workspace()
        return super().zero_grad(set_to_none=set_to_none)
