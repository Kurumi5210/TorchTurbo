from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Protocol

import torch


class ProfilerHandle(Protocol):
    def start(self) -> None: ...

    def step(self) -> None: ...

    def stop(self) -> None: ...


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
    def profiler(self, *, output_dir: str, worker_name: str) -> ProfilerHandle:
        ...
