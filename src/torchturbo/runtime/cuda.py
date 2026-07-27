from __future__ import annotations

from contextlib import nullcontext

import torch

from .base import DeviceRuntime, ProfilerHandle, TurboDevice


class CudaRuntime(DeviceRuntime):
    turbo_device = TurboDevice.CUDA
    dist_backend = "nccl"

    def initialize(self) -> None:
        return None

    def available(self) -> bool:
        return torch.cuda.is_available()

    def current_device(self):
        return torch.cuda.current_device()

    def set_device(self, device: int) -> None:
        torch.cuda.set_device(device)

    def synchronize(self) -> None:
        if self.available():
            torch.cuda.synchronize()

    def empty_cache(self) -> None:
        if self.available():
            torch.cuda.empty_cache()

    def memory_info(self) -> tuple[int, int]:
        return torch.cuda.mem_get_info()

    def stream(self, stream=None):
        if stream is None:
            return nullcontext()
        return torch.cuda.stream(stream)

    def current_stream(self):
        if not self.available():
            return None
        return torch.cuda.current_stream()

    def new_stream(self, priority: int = 0):
        return torch.cuda.Stream(priority=priority)

    def manual_seed_all(self, seed: int) -> None:
        torch.cuda.manual_seed_all(seed)

    def profiler(
        self,
        *,
        output_dir: str,
        worker_name: str,
    ) -> ProfilerHandle:
        return torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=0, warmup=0, active=1, repeat=1, skip_first=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                output_dir, worker_name=worker_name
            ),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_modules=False,
            with_flops=False,
        )
