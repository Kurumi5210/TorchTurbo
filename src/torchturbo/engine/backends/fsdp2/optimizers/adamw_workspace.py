"""Chunked AdamW state staging and transfer-stream execution."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.distributed.tensor import DTensor

from .....runtime import DeviceRuntime, TurboDevice


class DeviceWorkspace:
    """Byte-backed temporary device storage that yields typed tensor views."""

    _ALIGNMENT_BYTES = 16

    def __init__(
        self,
        device_runtime: DeviceRuntime,
        capacity_bytes: int,
    ) -> None:
        """Reserve the requested temporary device storage."""
        self.device_runtime = device_runtime
        self._storage: torch.Tensor | None = None
        if device_runtime.turbo_device is TurboDevice.CPU:
            return

        device = device_runtime.device
        if capacity_bytes < self._ALIGNMENT_BYTES:
            raise RuntimeError(
                f"workspace capacity {capacity_bytes} bytes is too small"
            )
        self._storage = torch.empty(capacity_bytes, dtype=torch.uint8, device=device)

    @property
    def is_available(self) -> bool:
        return self._storage is not None

    def _require_storage(self) -> torch.Tensor:
        if self._storage is None:
            raise RuntimeError("workspace is unavailable")
        return self._storage

    @property
    def device(self) -> torch.device:
        return self._require_storage().device

    @property
    def capacity_bytes(self) -> int:
        return self._require_storage().numel()

    @classmethod
    def align(cls, offset: int) -> int:
        alignment = cls._ALIGNMENT_BYTES
        return (offset + alignment - 1) // alignment * alignment

    def view_like(self, offset: int, tensor: torch.Tensor) -> torch.Tensor:
        nbytes = tensor.numel() * tensor.element_size()
        if offset % tensor.element_size() != 0:
            raise ValueError("workspace offset is not aligned for tensor dtype")
        if offset + nbytes > self.capacity_bytes:
            raise RuntimeError("workspace capacity exceeded")
        raw = self._require_storage().narrow(0, offset, nbytes)
        return raw.view(tensor.dtype).view(tensor.shape)


@dataclass(frozen=True)
class WorkspacePlan:
    """The number and byte capacity of reusable moment-pair slots."""

    num_slots: int
    slot_bytes: int

    def __post_init__(self) -> None:
        if self.num_slots not in (1, 2):
            raise ValueError("AdamW workspace requires one or two slots")
        if self.slot_bytes < 0 or self.slot_bytes % 32:
            raise ValueError("AdamW slot capacity must be a nonnegative multiple of 32")

    @property
    def capacity_bytes(self) -> int:
        return self.num_slots * self.slot_bytes


@dataclass(frozen=True)
class BoundParameter:
    """Current local parameter, gradient, and canonical CPU moment tensors."""

    original: torch.Tensor
    local_param: torch.Tensor
    local_grad: torch.Tensor
    group: dict[str, Any]
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor

    @property
    def numel(self) -> int:
        return self.local_param.numel()

    @property
    def minimum_slot_bytes(self) -> int:
        return 2 * DeviceWorkspace._ALIGNMENT_BYTES if self.numel else 0


def bind_parameters(
    params_with_groups: Sequence[tuple[torch.Tensor, dict[str, Any]]],
    cpu_state_map: Mapping[torch.Tensor, Mapping[str, torch.Tensor]],
) -> list[BoundParameter]:
    """Bind views for this step without retaining gradients across steps."""
    bound = []
    for original, group in params_with_groups:
        grad = original.grad
        if grad is None:
            continue
        local_param = original.to_local() if isinstance(original, DTensor) else original
        local_grad = grad.to_local() if isinstance(grad, DTensor) else grad
        state = cpu_state_map[original]
        for name, tensor in (
            ("parameter", local_param),
            ("gradient", local_grad),
            ("exp_avg", state["exp_avg"]),
            ("exp_avg_sq", state["exp_avg_sq"]),
        ):
            if not tensor.is_contiguous():
                raise RuntimeError(
                    f"AdamW workspace requires contiguous {name}; "
                    f"got shape={tuple(tensor.shape)}, stride={tensor.stride()}"
                )
        bound.append(
            BoundParameter(
                original,
                local_param,
                local_grad,
                group,
                state["exp_avg"],
                state["exp_avg_sq"],
            )
        )
    return bound


@dataclass(frozen=True)
class ChunkSpan:
    """One shared-storage parameter range and its offset in each moment half."""

    parameter: BoundParameter
    param: torch.Tensor
    grad: torch.Tensor
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor
    moment_offset: int
    start: int
    numel: int


def iter_chunks(
    bound: Sequence[BoundParameter],
    slot_bytes: int,
) -> Iterator[tuple[ChunkSpan, ...]]:
    """Pack contiguous parameter slices into slots."""
    if slot_bytes < 0 or slot_bytes % 32:
        raise ValueError("AdamW slot capacity must be a nonnegative multiple of 32")
    for parameter in bound:
        if parameter.minimum_slot_bytes > slot_bytes:
            raise RuntimeError(
                "AdamW parameter requires a workspace slot of "
                f"{parameter.minimum_slot_bytes} bytes, got {slot_bytes} bytes"
            )
    half_bytes = slot_bytes // 2
    chunk: list[ChunkSpan] = []
    moment_offset = 0
    for parameter in bound:
        start = 0
        tensors = (
            parameter.local_param,
            parameter.local_grad,
            parameter.exp_avg,
            parameter.exp_avg_sq,
        )
        moment_element_bytes = max(
            parameter.exp_avg.element_size(), parameter.exp_avg_sq.element_size()
        )
        while start < parameter.numel:
            moment_offset = DeviceWorkspace.align(moment_offset)
            available = (half_bytes - moment_offset) // moment_element_bytes
            remaining = parameter.numel - start
            if available == 0:
                yield tuple(chunk)
                chunk = []
                moment_offset = 0
                continue
            numel = min(remaining, available)
            views = tuple(tensor.view(-1).narrow(0, start, numel) for tensor in tensors)
            chunk.append(ChunkSpan(parameter, *views, moment_offset, start, numel))
            moment_offset += numel * moment_element_bytes
            start += numel
    if chunk:
        yield tuple(chunk)


@dataclass(frozen=True)
class _SlotEvents:
    input_ready: Any
    compute_done: Any
    finished: Any


class PipelineStreams:
    """Reusable streams and events, without ownership of parameter or slab views."""

    def __init__(self, runtime: DeviceRuntime, num_slots: int = 2) -> None:
        if num_slots not in (1, 2):
            raise ValueError("AdamW pipeline requires one or two slots")
        self.runtime = runtime
        self.num_slots = num_slots
        self.h2d = runtime.new_stream()
        self.d2h = runtime.new_stream()
        if self.h2d is None or self.d2h is None:
            raise RuntimeError("AdamW overlap requires asynchronous device streams")
        self.caller_ready = runtime.new_event()
        self.slots = tuple(
            _SlotEvents(runtime.new_event(), runtime.new_event(), runtime.new_event())
            for _ in range(num_slots)
        )


def single_chunk_slot_bytes(bound: Sequence[BoundParameter]) -> int:
    """Return the smallest slot that holds every active moment pair in one chunk.

    Each half packs one moment per parameter at 16-byte aligned offsets, as
    ``iter_chunks`` does. Any larger slot yields the same single chunk.
    """
    half = 0
    for param in bound:
        element_bytes = max(param.exp_avg.element_size(), param.exp_avg_sq.element_size())
        half += DeviceWorkspace.align(param.numel * element_bytes)
    return 2 * half


def calibration_sink_bytes(bound: Sequence[BoundParameter], capacity: int) -> int:
    """Bound the output ring by the aligned size of all active moment pairs."""
    return min(capacity, single_chunk_slot_bytes(bound))


def _sink_view(
    storage: torch.Tensor,
    offset: int,
    tensor: torch.Tensor,
) -> torch.Tensor:
    nbytes = tensor.numel() * tensor.element_size()
    return storage.narrow(0, offset, nbytes).view(tensor.dtype).view(tensor.shape)


def _stage_views(
    chunk: tuple[ChunkSpan, ...],
    plan: WorkspacePlan,
    slot_index: int,
    workspace: DeviceWorkspace,
    cpu_sink: torch.Tensor | None,
) -> list[tuple[ChunkSpan, dict[str, torch.Tensor], dict[str, torch.Tensor]]]:
    base = slot_index * plan.slot_bytes
    entries = []
    for span in chunk:
        state = {}
        output = {}
        for key, moment, half_offset in (
            ("exp_avg", span.exp_avg, 0),
            ("exp_avg_sq", span.exp_avg_sq, plan.slot_bytes // 2),
        ):
            offset = base + half_offset + span.moment_offset
            state[key] = workspace.view_like(offset, moment)
            if cpu_sink is None:
                output[key] = moment
            else:
                # The sink packs exp_avg for every slot in its first half and
                # exp_avg_sq in its second half. Chunks fill slots in order, so
                # slot k starts at k * half_slot of the unchunked packing and
                # never ends beyond single_chunk_slot_bytes(bound) // 2.
                sink_offset = base // 2 + span.moment_offset
                if key == "exp_avg_sq":
                    sink_offset += cpu_sink.numel() // 2
                output[key] = _sink_view(cpu_sink, sink_offset, moment)
        entries.append((span, state, output))
    return entries


def _copy_in(entries: list) -> None:
    for span, state, _ in entries:
        state["exp_avg"].copy_(span.exp_avg, non_blocking=True)
        state["exp_avg_sq"].copy_(span.exp_avg_sq, non_blocking=True)


def _compute(
    entries: list,
    steps: Mapping[torch.Tensor, torch.Tensor],
    run_adamw: Callable[..., None],
) -> None:
    batches: dict[tuple[int, torch.device, torch.dtype], list] = {}
    for entry in entries:
        span = entry[0]
        key = (id(span.parameter.group), span.param.device, span.param.dtype)
        batches.setdefault(key, []).append(entry)
    for batch in batches.values():
        run_adamw(
            [span.param for span, _, _ in batch],
            [span.grad for span, _, _ in batch],
            [state for _, state, _ in batch],
            [steps[span.parameter.original] for span, _, _ in batch],
            batch[0][0].parameter.group,
        )


def _copy_out(entries: list) -> None:
    for _, state, output in entries:
        for key in ("exp_avg", "exp_avg_sq"):
            output[key].copy_(state[key], non_blocking=True)


@torch.no_grad()
def reset_workspace_moments(
    bound: Sequence[BoundParameter],
    plan: WorkspacePlan,
    workspace: DeviceWorkspace,
) -> None:
    """Restore verified zero CPU moments using the existing device slot.

    The caller must finish the trial before invoking this restore. The slot
    remains zero and read-only throughout the copies, so every chunk can reuse
    slot zero without additional storage or per-chunk synchronization.
    """
    if workspace.capacity_bytes < plan.capacity_bytes:
        raise ValueError("AdamW workspace is smaller than the selected plan")
    runtime = workspace.device_runtime
    caller = runtime.current_stream()
    try:
        with runtime.stream(caller):
            workspace._require_storage().narrow(0, 0, plan.slot_bytes).zero_()
            for chunk in iter_chunks(bound, plan.slot_bytes):
                entries = _stage_views(chunk, plan, 0, workspace, cpu_sink=None)
                _copy_out(entries)
        runtime.synchronize_stream(caller)
    except BaseException:
        try:
            runtime.synchronize_stream(caller)
        except Exception:  # noqa: BLE001, S110 - Preserve the restore error.
            pass
        raise


@torch.no_grad()
def run_workspace(
    bound: Sequence[BoundParameter],
    plan: WorkspacePlan,
    workspace: DeviceWorkspace,
    steps: Mapping[torch.Tensor, torch.Tensor],
    run_adamw: Callable[..., None],
    *,
    pipeline: PipelineStreams | None = None,
    cpu_sink: torch.Tensor | None = None,
) -> None:
    """Execute chunks and finish host writes before returning.

    The caller increments each original parameter's step exactly once. An
    optional pinned byte tensor receives calibration output without modifying
    the canonical CPU moments. Its two halves omit unused device capacity.
    """
    if workspace.capacity_bytes < plan.capacity_bytes:
        raise ValueError("AdamW workspace is smaller than the selected plan")
    if cpu_sink is not None and (
        cpu_sink.device.type != "cpu"
        or cpu_sink.dtype is not torch.uint8
        or cpu_sink.ndim != 1
        or cpu_sink.numel() % 32
        or cpu_sink.numel() < calibration_sink_bytes(bound, plan.capacity_bytes)
    ):
        raise ValueError("AdamW calibration sink must cover the active aligned moment pairs as CPU bytes")
    if plan.num_slots == 2 and pipeline is None:
        raise ValueError("two-slot AdamW workspace requires a pipeline")
    if pipeline is not None and pipeline.num_slots < plan.num_slots:
        raise ValueError("AdamW pipeline has fewer events than the selected plan")

    runtime = workspace.device_runtime
    caller = runtime.current_stream()
    if pipeline is None:
        try:
            for chunk in iter_chunks(bound, plan.slot_bytes):
                entries = _stage_views(chunk, plan, 0, workspace, cpu_sink)
                _copy_in(entries)
                _compute(entries, steps, run_adamw)
                _copy_out(entries)
        except BaseException:
            try:
                runtime.synchronize_stream(caller)
            except Exception:  # noqa: BLE001, S110 - Preserve the submission error.
                pass
            raise
        runtime.synchronize_stream(caller)
        return

    if caller is None:
        raise RuntimeError("AdamW overlap requires a current device stream")
    storage = workspace._require_storage()
    for stream in (caller, pipeline.h2d, pipeline.d2h):
        runtime.record_stream(storage, stream)
    pipeline.caller_ready.record(caller)
    pipeline.h2d.wait_event(pipeline.caller_ready)
    occupied = [False] * plan.num_slots
    try:
        for index, chunk in enumerate(iter_chunks(bound, plan.slot_bytes)):
            slot_index = index % plan.num_slots
            events = pipeline.slots[slot_index]
            entries = _stage_views(chunk, plan, slot_index, workspace, cpu_sink)
            with runtime.stream(pipeline.h2d):
                if occupied[slot_index]:
                    pipeline.h2d.wait_event(events.finished)
                _copy_in(entries)
                events.input_ready.record(pipeline.h2d)
            with runtime.stream(caller):
                caller.wait_event(events.input_ready)
                _compute(entries, steps, run_adamw)
                events.compute_done.record(caller)
            with runtime.stream(pipeline.d2h):
                pipeline.d2h.wait_event(events.compute_done)
                _copy_out(entries)
                events.finished.record(pipeline.d2h)
            occupied[slot_index] = True
        # D2H completion includes every preceding copy and compute dependency.
        runtime.synchronize_stream(pipeline.d2h)
    except BaseException:
        # A partially submitted chunk may not have recorded its next event.
        # Drain each producer instead of introducing waits on missing events.
        for stream in (pipeline.h2d, caller, pipeline.d2h):
            try:
                runtime.synchronize_stream(stream)
            except Exception:  # noqa: BLE001, S110 - Preserve the submission error.
                pass
        raise
