from __future__ import annotations

from .base import DeviceRuntime, ProfilerHandle, TurboDevice
from .runtime import (
    get_device_name,
    get_device_runtime,
    is_cuda_available,
    is_device_available,
    is_npu_available,
)


__all__ = [
    "DeviceRuntime",
    "get_device_name",
    "get_device_runtime",
    "is_cuda_available",
    "is_device_available",
    "is_npu_available",
    "ProfilerHandle",
    "TurboDevice",
]
