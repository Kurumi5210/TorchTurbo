"""Chunk layout and asynchronous slot ownership without accelerator hardware."""

import os
import sys
import types
import weakref
from contextlib import contextmanager
from dataclasses import dataclass

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

if "grouped_gemm" not in sys.modules:  # pragma: no cover - env shim
    _gg = types.ModuleType("grouped_gemm")
    _gg.ops = types.SimpleNamespace()
    sys.modules["grouped_gemm"] = _gg

import pytest
import torch

from torchturbo.engine.backends.fsdp2.optimizers.adamw_workspace import (
    DeviceWorkspace,
    PipelineStreams,
    WorkspacePlan,
    bind_parameters,
    iter_chunks,
    reset_workspace_moments,
    run_workspace,
)
from torchturbo.runtime import TurboDevice
from torchturbo.runtime.cpu import CpuRuntime


@dataclass(eq=False)
class _Task:
    stream: object
    label: str
    action: object
    dependencies: tuple
    ordinal: int
    done: bool = False


class _QueuedStream:
    def __init__(self, runtime, name, priority):
        self.runtime = runtime
        self.name = name
        self.priority = priority
        self.tail = None

    def enqueue(self, label, action=None, dependencies=()):
        previous = () if self.tail is None else (self.tail,)
        task = _Task(
            self, label, action, previous + tuple(dependencies), len(self.runtime.tasks)
        )
        self.runtime.tasks.append(task)
        self.tail = task
        return task

    def wait_event(self, event):
        assert event.latest is not None, "wait submitted before an event was recorded"
        # Device waits capture the current record, not a later re-recording.
        generation = event.generation
        event.waits.append((self.name, generation))
        self.enqueue(f"wait:{event.name}:{generation}", dependencies=(event.latest,))

    def synchronize(self):
        self.runtime.synchronize_stream(self)


class _QueuedEvent:
    def __init__(self, name):
        self.name = name
        self.generation = 0
        self.latest = None
        self.waits = []

    def record(self, stream):
        self.generation += 1
        self.latest = stream.enqueue(f"record:{self.name}:{self.generation}")


class QueuedRuntime(CpuRuntime):
    """Execute H2D eagerly relative to compute and D2H to expose unsafe reuse."""

    turbo_device = TurboDevice.NPU
    device = torch.device("cpu")

    def __init__(self):
        self.tasks = []
        self.trace = []
        self.events = []
        self.streams = []
        self.synchronizations = []
        self.recorded_storage = []
        self.global_synchronizations = 0
        self.caller = _QueuedStream(self, "compute", 1)
        self.active = self.caller

    def new_stream(self, priority=0):
        index = len(self.streams)
        stream = _QueuedStream(self, "h2d" if index == 0 else "d2h", index * 2)
        self.streams.append(stream)
        return stream

    def new_event(self):
        event = _QueuedEvent(f"event{len(self.events)}")
        self.events.append(event)
        return event

    def current_stream(self):
        return self.active

    @contextmanager
    def stream(self, stream=None):
        previous = self.active
        self.active = self.caller if stream is None else stream
        try:
            yield self.active
        finally:
            self.active = previous

    def enqueue(self, label, action):
        self.active.enqueue(label, action)

    def record_stream(self, tensor, stream):
        self.recorded_storage.append((tensor.data_ptr(), stream.name))

    def synchronize_stream(self, stream):
        self.synchronizations.append(stream.name)
        target = stream.tail
        while target is not None and not target.done:
            ready = [
                task
                for task in self.tasks
                if not task.done
                and all(dependency.done for dependency in task.dependencies)
            ]
            assert ready, "event dependency cycle or an unrecorded producer"
            task = min(ready, key=lambda item: (item.stream.priority, item.ordinal))
            self.trace.append((task.stream.name, task.label))
            try:
                if task.action is not None:
                    task.action()
            finally:
                task.done = True
                # Completed device operations no longer own tensor views.
                task.action = None

    def synchronize(self):
        self.global_synchronizations += 1
        for stream in (self.caller, *self.streams):
            self.synchronize_stream(stream)


def enqueue_copies(monkeypatch, runtime):
    """Defer actual tensor copies, retaining source views until execution."""
    copy_tensor = torch.Tensor.copy_

    def copy_(destination, source, *args, **kwargs):
        if kwargs.get("non_blocking"):
            runtime.enqueue(
                "copy", lambda: copy_tensor(destination, source, *args, **kwargs)
            )
            return destination
        return copy_tensor(destination, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", copy_)


def _bindings(specs=((25, torch.float32), (17, torch.float16), (11, torch.bfloat16))):
    groups = [{"lr": 0.25}, {"lr": 0.5}]
    params = []
    states = {}
    steps = {}
    params_with_groups = []
    for index, (numel, dtype) in enumerate(specs):
        param = torch.nn.Parameter(torch.arange(numel, dtype=dtype).mul_(0.25))
        param.grad = torch.full_like(param, 0.125 * (index + 1))
        states[param] = {
            "exp_avg": torch.full_like(param, index + 0.25),
            "exp_avg_sq": torch.full_like(param, index + 0.5),
        }
        steps[param] = torch.tensor(float(index + 5))
        params.append(param)
        params_with_groups.append((param, groups[index % len(groups)]))
    return params, states, steps, bind_parameters(params_with_groups, states)


def _queued_update(runtime, submissions, fail_on=None):
    def update(params, grads, states, steps, group):
        assert len({param.dtype for param in params}) == 1
        submissions.append((len(params), group["lr"]))

        def compute():
            for param, grad, state in zip(params, grads, states):
                state["exp_avg"].add_(grad)
                state["exp_avg_sq"].add_(grad.square())
                param.add_(state["exp_avg"], alpha=-group["lr"])

        runtime.enqueue("kernel", compute)
        if len(submissions) == fail_on:
            raise RuntimeError("injected failure before recording compute completion")

    return update


def _expected(bound):
    result = []
    for parameter in bound:
        avg = parameter.exp_avg + parameter.local_grad
        square = parameter.exp_avg_sq + parameter.local_grad.square()
        param = parameter.local_param.detach().clone()
        param.add_(avg, alpha=-parameter.group["lr"])
        result.append((param, avg, square))
    return result


def test_optimizer_drops_drained_pipeline_without_another_global_sync(monkeypatch):
    from .test_adamw_workspace import _optimizer, _select_plan

    runtime = QueuedRuntime()
    monkeypatch.setattr(runtime, "memory_info", lambda: (4096, 4096))
    monkeypatch.setattr(runtime, "memory_reserved", lambda: 0)
    monkeypatch.setattr(runtime, "memory_allocated", lambda: 0)
    enqueue_copies(monkeypatch, runtime)
    optimizer, params = _optimizer((33, 17), runtime)
    _select_plan(optimizer, WorkspacePlan(2, 32))
    bound = bind_parameters(
        optimizer._active_parameters(), optimizer.param_to_cpu_states_map
    )
    expected = _expected(bound)
    storages = []
    initialize = DeviceWorkspace.__init__

    def allocate(workspace, device_runtime, capacity):
        initialize(workspace, device_runtime, capacity)
        storages.append(weakref.ref(workspace._require_storage()))

    monkeypatch.setattr(DeviceWorkspace, "__init__", allocate)
    monkeypatch.setattr(optimizer, "_run_adamw", _queued_update(runtime, []))
    optimizer.step()

    assert runtime.global_synchronizations == 1
    assert all(task.done for task in runtime.tasks)
    assert optimizer.workspace is None
    assert all(storage() is None for storage in storages)
    for original, parameter, (param, avg, square) in zip(params, bound, expected):
        torch.testing.assert_close(parameter.local_param, param, rtol=0, atol=0)
        torch.testing.assert_close(parameter.exp_avg, avg, rtol=0, atol=0)
        torch.testing.assert_close(parameter.exp_avg_sq, square, rtol=0, atol=0)
        assert optimizer.param_to_step_map[original].item() == 1


@pytest.mark.parametrize("overlap", [False, True])
def test_reused_slots_preserve_data_and_event_generations(monkeypatch, overlap):
    runtime = QueuedRuntime()
    enqueue_copies(monkeypatch, runtime)
    params, _, steps, bound = _bindings(
        ((257, torch.float32), (131, torch.float16), (67, torch.bfloat16))
    )
    plan = WorkspacePlan(2 if overlap else 1, 64)
    workspace = DeviceWorkspace(runtime, plan.capacity_bytes)
    storage = weakref.ref(workspace._require_storage())
    pipeline = PipelineStreams(runtime) if overlap else None
    submissions = []
    update = _queued_update(runtime, submissions)
    saved_steps = [step.clone() for step in steps.values()]
    num_chunks = len(list(iter_chunks(bound, plan.slot_bytes)))
    assert num_chunks > 6 * plan.num_slots

    for _ in range(2):
        expected = _expected(bound)
        # The slab can have a preceding producer on the caller's stream.
        runtime.enqueue("producer", workspace._require_storage().zero_)
        run_workspace(bound, plan, workspace, steps, update, pipeline=pipeline)
        assert all(task.done for task in runtime.tasks)
        for parameter, (param, avg, square) in zip(bound, expected):
            torch.testing.assert_close(parameter.local_param, param, rtol=0, atol=0)
            torch.testing.assert_close(parameter.exp_avg, avg, rtol=0, atol=0)
            torch.testing.assert_close(parameter.exp_avg_sq, square, rtol=0, atol=0)
        for step, original in zip(steps.values(), saved_steps):
            torch.testing.assert_close(step, original, rtol=0, atol=0)
        for param in params:
            param.grad.add_(0.125)

    assert runtime.global_synchronizations == 0
    assert runtime.synchronizations == (["d2h"] * 2 if overlap else ["compute"] * 2)
    if overlap:
        assert {name for _, name in runtime.recorded_storage} == {
            "compute",
            "h2d",
            "d2h",
        }
        assert pipeline.caller_ready.generation == 2
        for events in pipeline.slots:
            assert events.finished.generation >= 4
            assert ("h2d", 1) in events.finished.waits
            assert ("compute", 1) in events.input_ready.waits
            assert ("d2h", 1) in events.compute_done.waits
        assert runtime.trace.index(("compute", "producer")) < runtime.trace.index(
            ("h2d", "copy")
        )
    del workspace
    assert storage() is None


def test_chunk_tail_and_mixed_dtype_views_cover_original_storage():
    _, _, _, bound = _bindings()
    chunks = list(iter_chunks(bound, 64))
    assert all(chunks)
    for parameter in bound:
        spans = [
            span for chunk in chunks for span in chunk if span.parameter is parameter
        ]
        cursor = 0
        for span in spans:
            assert span.start == cursor
            assert span.moment_offset % 16 == 0
            assert (
                span.moment_offset + span.exp_avg.numel() * span.exp_avg.element_size()
                <= 32
            )
            for view, original in (
                (span.param, parameter.local_param),
                (span.grad, parameter.local_grad),
                (span.exp_avg, parameter.exp_avg),
                (span.exp_avg_sq, parameter.exp_avg_sq),
            ):
                assert (
                    view.untyped_storage().data_ptr()
                    == original.untyped_storage().data_ptr()
                )
            cursor += span.numel
        assert cursor == parameter.numel
    first_spans = [
        span for chunk in chunks for span in chunk if span.parameter is bound[0]
    ]
    assert [span.numel for span in first_spans] == [8, 8, 8, 1]


@pytest.mark.parametrize("field", ["parameter", "gradient", "exp_avg", "exp_avg_sq"])
def test_binding_rejects_noncontiguous_workspace_inputs(field):
    transposed = torch.zeros(4, 3).T
    param = torch.nn.Parameter(
        transposed if field == "parameter" else torch.zeros(3, 4)
    )
    param.grad = (
        transposed if field in ("parameter", "gradient") else torch.ones_like(param)
    )
    states = {
        param: {
            key: transposed if key == field else torch.zeros_like(param)
            for key in ("exp_avg", "exp_avg_sq")
        }
    }
    with pytest.raises(RuntimeError, match=f"requires contiguous {field};"):
        bind_parameters([(param, {})], states)


def test_binding_rejects_strided_slice_with_holes():
    param = torch.nn.Parameter(torch.zeros(3, 8)[:, ::2])
    param.grad = torch.zeros(3, 8)[:, ::2]
    states = {
        param: {key: torch.zeros_like(param) for key in ("exp_avg", "exp_avg_sq")}
    }
    with pytest.raises(RuntimeError, match="requires contiguous parameter;"):
        bind_parameters([(param, {})], states)


def test_calibration_sink_keeps_canonical_moments_unchanged(monkeypatch):
    runtime = QueuedRuntime()
    enqueue_copies(monkeypatch, runtime)
    _, states, steps, bound = _bindings()
    plan = WorkspacePlan(2, 64)
    workspace = DeviceWorkspace(runtime, plan.capacity_bytes)
    pipeline = PipelineStreams(runtime)
    saved = {
        param: {key: tensor.clone() for key, tensor in state.items()}
        for param, state in states.items()
    }
    expected = _expected(bound)
    sink = torch.zeros(plan.capacity_bytes, dtype=torch.uint8)

    run_workspace(
        bound,
        plan,
        workspace,
        steps,
        _queued_update(runtime, []),
        pipeline=pipeline,
        cpu_sink=sink,
    )

    assert sink.count_nonzero().item() > 0
    for parameter, (param, _, _) in zip(bound, expected):
        torch.testing.assert_close(parameter.local_param, param, rtol=0, atol=0)
    for param, state in states.items():
        for key, tensor in state.items():
            torch.testing.assert_close(tensor, saved[param][key], rtol=0, atol=0)
    assert all(task.done for task in runtime.tasks)


def test_two_slot_plan_without_streams_does_not_execute_serially():
    runtime = QueuedRuntime()
    _, _, steps, bound = _bindings(((41, torch.float32),))
    plan = WorkspacePlan(2, 64)
    with pytest.raises(ValueError, match="requires a pipeline"):
        run_workspace(
            bound,
            plan,
            DeviceWorkspace(runtime, plan.capacity_bytes),
            steps,
            _queued_update(runtime, []),
        )
    assert not runtime.tasks


def test_partial_submission_failure_drains_all_streams_without_missing_events(
    monkeypatch,
):
    runtime = QueuedRuntime()
    enqueue_copies(monkeypatch, runtime)
    _, _, steps, bound = _bindings(((41, torch.float32),))
    plan = WorkspacePlan(2, 64)
    workspace = DeviceWorkspace(runtime, plan.capacity_bytes)
    storage = weakref.ref(workspace._require_storage())
    pipeline = PipelineStreams(runtime)
    submissions = []

    with pytest.raises(RuntimeError, match="injected failure"):
        run_workspace(
            bound,
            plan,
            workspace,
            steps,
            _queued_update(runtime, submissions, fail_on=3),
            pipeline=pipeline,
        )

    assert len(submissions) == 3
    assert runtime.synchronizations == ["h2d", "compute", "d2h"]
    assert all(task.done for task in runtime.tasks)
    assert runtime.global_synchronizations == 0
    del workspace
    assert storage() is None


def test_zero_size_parameters_and_missing_gradients_produce_no_chunks():
    empty = torch.nn.Parameter(torch.empty(0))
    empty.grad = torch.empty(0)
    skipped = torch.nn.Parameter(torch.ones(4))
    states = {empty: {"exp_avg": torch.empty(0), "exp_avg_sq": torch.empty(0)}}
    bound = bind_parameters([(empty, {}), (skipped, {})], states)
    assert len(bound) == 1 and bound[0].minimum_slot_bytes == 0
    assert list(iter_chunks(bound, 0)) == []


def _enqueue_workspace_zero(monkeypatch, runtime, workspace):
    zero_tensor = torch.Tensor.zero_
    pointer = workspace._require_storage().untyped_storage().data_ptr()
    writes = []

    def zero_(tensor):
        assert tensor.untyped_storage().data_ptr() == pointer
        writes.append((tensor.dtype, tensor.storage_offset(), tensor.numel()))
        runtime.enqueue("zero", lambda: zero_tensor(tensor))
        return tensor

    monkeypatch.setattr(torch.Tensor, "zero_", zero_)
    return writes


def test_reset_moments_reuses_read_only_slot_zero_and_finishes_cpu_writes(monkeypatch):
    runtime = QueuedRuntime()
    enqueue_copies(monkeypatch, runtime)
    params, states, steps, bound = _bindings()
    saved = [
        (param.detach().clone(), param.grad.clone(), steps[param].clone())
        for param in params
    ]
    plan = WorkspacePlan(2, 96)
    assert len(list(iter_chunks(bound, plan.slot_bytes))) > 2
    workspace = DeviceWorkspace(runtime, plan.capacity_bytes)
    storage = workspace._require_storage()
    storage.fill_(173)
    runtime.enqueue("preceding-write", lambda: storage[: plan.slot_bytes].fill_(91))
    writes = _enqueue_workspace_zero(monkeypatch, runtime, workspace)
    enqueue = runtime.enqueue
    reads = []

    def checked_enqueue(label, action):
        if label == "copy":

            def checked_copy():
                # Every queued D2H sees the same zero source. Other slots are
                # neither cleared nor used as temporary storage by the restore.
                assert torch.count_nonzero(storage[: plan.slot_bytes]).item() == 0
                assert torch.all(storage[plan.slot_bytes :] == 173).item()
                reads.append(True)
                action()

            enqueue(label, checked_copy)
        else:
            enqueue(label, action)

    def unexpected_allocation(*args, **kwargs):
        pytest.fail("moment reset must reuse the existing slab without allocations")

    monkeypatch.setattr(runtime, "enqueue", checked_enqueue)
    with monkeypatch.context() as allocation_guard:
        allocation_guard.setattr(DeviceWorkspace, "__init__", unexpected_allocation)
        for name in ("empty", "empty_like", "zeros", "zeros_like"):
            allocation_guard.setattr(torch, name, unexpected_allocation)
        reset_workspace_moments(bound, plan, workspace)

    assert writes == [(torch.uint8, 0, plan.slot_bytes)]
    assert len(reads) > 4
    assert runtime.trace[:2] == [
        ("compute", "preceding-write"),
        ("compute", "zero"),
    ]
    assert all(label == "copy" for _, label in runtime.trace[2:])
    assert workspace._require_storage() is storage
    assert runtime.synchronizations == ["compute"]
    assert runtime.global_synchronizations == 0
    assert all(task.done for task in runtime.tasks)
    for param, (value, grad, step) in zip(params, saved):
        torch.testing.assert_close(param, value, rtol=0, atol=0)
        torch.testing.assert_close(param.grad, grad, rtol=0, atol=0)
        torch.testing.assert_close(steps[param], step, rtol=0, atol=0)
        for moment in states[param].values():
            assert torch.count_nonzero(moment).item() == 0


def test_reset_moments_submission_failure_drains_already_queued_writes(monkeypatch):
    runtime = QueuedRuntime()
    enqueue_copies(monkeypatch, runtime)
    params, _, steps, bound = _bindings(((41, torch.float32),))
    saved = (
        params[0].detach().clone(),
        params[0].grad.clone(),
        steps[params[0]].clone(),
    )
    plan = WorkspacePlan(2, 64)
    workspace = DeviceWorkspace(runtime, plan.capacity_bytes)
    workspace._require_storage().fill_(173)
    storage = weakref.ref(workspace._require_storage())
    writes = _enqueue_workspace_zero(monkeypatch, runtime, workspace)
    queued_copy = torch.Tensor.copy_
    num_copies = 0

    def fail_after_submission(tensor, source, *args, **kwargs):
        nonlocal num_copies
        result = queued_copy(tensor, source, *args, **kwargs)
        num_copies += 1
        if num_copies == 3:
            raise RuntimeError("injected moment restore submission failure")
        return result

    monkeypatch.setattr(torch.Tensor, "copy_", fail_after_submission)
    with pytest.raises(
        RuntimeError, match="injected moment restore submission failure"
    ):
        reset_workspace_moments(bound, plan, workspace)

    assert writes == [(torch.uint8, 0, plan.slot_bytes)]
    assert num_copies == 3
    assert all(task.done for task in runtime.tasks)
    assert runtime.synchronizations == ["compute"]
    assert runtime.global_synchronizations == 0
    torch.testing.assert_close(params[0], saved[0], rtol=0, atol=0)
    torch.testing.assert_close(params[0].grad, saved[1], rtol=0, atol=0)
    torch.testing.assert_close(steps[params[0]], saved[2], rtol=0, atol=0)
    del workspace
    assert storage() is None
