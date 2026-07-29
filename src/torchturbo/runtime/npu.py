from __future__ import annotations

import os
from contextlib import nullcontext

import torch

from .base import BaseProfiler, DeviceRuntime, TurboDevice


class _NpuProfiler(BaseProfiler):
    def __init__(
        self,
        output_dir: str,
        worker_name: str,
        profiling_enabled: bool,
        memory_snapshot_steps: int | None,
    ) -> None:
        self._profiler = None
        if profiling_enabled:
            import torch_npu

            experimental_config = torch_npu.profiler._ExperimentalConfig(
                export_type=[torch_npu.profiler.ExportType.Text],
                profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
                mstx=False,
                aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
                l2_cache=False,
                op_attr=False,
                data_simplification=False,
                record_op_args=False,
                gc_detect_threshold=None,
            )

            self._profiler = torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                schedule=torch_npu.profiler.schedule(
                    wait=0, warmup=0, active=1, repeat=1, skip_first=1
                ),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    output_dir, worker_name=worker_name
                ),
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
                with_modules=False,
                with_flops=False,
                experimental_config=experimental_config,
            )
        self._snapshot_path = os.path.join(
            os.path.dirname(output_dir),
            "memory_snapshot",
            f"{worker_name}.pickle",
        )
        self._memory_snapshot_steps = memory_snapshot_steps
        self._memory_snapshot_active = False
        self._step = 0

    def start(self) -> None:
        if self._profiler is not None:
            self._profiler.start()
        if self._memory_snapshot_steps is not None:
            torch.npu.memory._record_memory_history()
            self._memory_snapshot_active = True

    def step(self) -> None:
        self._step += 1
        if self._memory_snapshot_active and self._step == self._memory_snapshot_steps:
            self._memory_snapshot_active = False
            try:
                torch.npu.memory._dump_snapshot(self._snapshot_path)
            finally:
                torch.npu.memory._record_memory_history(enabled=None)
        if self._profiler is not None:
            self._profiler.step()

    def stop(self) -> None:
        if self._memory_snapshot_active:
            self._memory_snapshot_active = False
            torch.npu.memory._record_memory_history(enabled=None)
        if self._profiler is not None:
            self._profiler.stop()


class NpuRuntime(DeviceRuntime):
    turbo_device = TurboDevice.NPU
    dist_backend = "hccl"

    def __init__(self) -> None:
        self._initialized = False

    def available(self) -> bool:
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            return False
        return hasattr(torch, "npu") and torch.npu.is_available()

    def initialize(self) -> None:
        if self._initialized:
            return
        import torch_npu  # noqa: F401
        from torch_npu.contrib import transfer_to_npu as _  # noqa: F401

        self._initialized = True

    @property
    def _npu(self):
        return torch.npu

    def current_device(self):
        return self._npu.current_device()

    def set_device(self, device: int) -> None:
        self._npu.set_device(device)

    def synchronize(self) -> None:
        if self.available():
            self._npu.synchronize()

    def empty_cache(self) -> None:
        if self.available():
            self._npu.empty_cache()

    def memory_info(self) -> tuple[int, int]:
        if not hasattr(self._npu, "mem_get_info"):
            raise NotImplementedError(
                f"{type(self).__name__} does not implement memory_info()."
            )
        return self._npu.mem_get_info()

    def stream(self, stream=None):
        if stream is None:
            return nullcontext()
        return self._npu.stream(stream)

    def current_stream(self):
        if not self.available():
            return None
        return self._npu.current_stream()

    def new_stream(self, priority: int = 0):
        try:
            return self._npu.Stream(priority=priority)
        except TypeError:
            return self._npu.Stream()

    def manual_seed_all(self, seed: int) -> None:
        self._npu.manual_seed_all(seed)

    def profiler(
        self,
        *,
        output_dir: str,
        worker_name: str,
        profiling_enabled: bool,
        memory_snapshot_steps: int | None,
    ) -> BaseProfiler:
        return _NpuProfiler(
            output_dir,
            worker_name,
            profiling_enabled,
            memory_snapshot_steps,
        )
