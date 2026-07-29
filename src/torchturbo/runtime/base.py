from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum

import torch


class BaseProfiler(ABC):
    @abstractmethod
    def __init__(
        self,
        output_dir: str,
        worker_name: str,
        profiling_enabled: bool,
        memory_snapshot_steps: int | None,
    ) -> None:
        ...

    @abstractmethod
    def start(self) -> None:
        ...

    @abstractmethod
    def step(self) -> None:
        ...

    @abstractmethod
    def stop(self) -> None:
        ...


class TurboDevice(StrEnum):
    """Supported TorchTurbo runtime devices."""

    CUDA = "cuda"
    NPU = "npu"
    CPU = "cpu"


class DeviceRuntime(ABC):
    # Concrete runtimes must set these class attributes.
    turbo_device: TurboDevice
    dist_backend: str

    @property
    def device(self) -> torch.device:
        """Device passed to Torch APIs."""
        return torch.device(self.turbo_device.value)

    @abstractmethod
    def available(self) -> bool:
        ...

    @abstractmethod
    def initialize(self) -> None:
        ...

    @abstractmethod
    def current_device(self):
        ...

    @abstractmethod
    def set_device(self, device: int) -> None:
        ...

    @abstractmethod
    def synchronize(self) -> None:
        ...

    @abstractmethod
    def empty_cache(self) -> None:
        ...

    @abstractmethod
    def memory_info(self) -> tuple[int, int]:
        ...

    @abstractmethod
    def stream(self, stream=None):
        ...

    @abstractmethod
    def current_stream(self):
        ...

    @abstractmethod
    def new_stream(self, priority: int = 0):
        ...

    def synchronize_stream(self, stream) -> None:
        if stream is not None:
            stream.synchronize()

    @abstractmethod
    def manual_seed_all(self, seed: int) -> None:
        ...

    @abstractmethod
    def profiler(
        self,
        *,
        output_dir: str,
        worker_name: str,
        profiling_enabled: bool,
        memory_snapshot_steps: int | None,
    ) -> BaseProfiler:
        ...
