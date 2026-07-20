"""Process-group initialization for the torchturbo engine."""

from __future__ import annotations

from datetime import timedelta
import os

import torch
import torch.distributed as dist

from ..logging import get_logger
from ..utils.device import get_device_name, get_nccl_backend, get_torch_device

logger = get_logger(__name__)

DEFAULT_PROCESS_GROUP_TIMEOUT = timedelta(seconds=1800)


def init_process_group_if_needed(enable_cpu_backend: bool = False) -> bool:
    """Initialize the default process group if it is absent.

    Idempotent: returns immediately once a default group exists (so it is safe
    to call before the training framework, which will then reuse this group).

    Parameters
    ----------
    enable_cpu_backend:
        When ``True``, register a CPU ``gloo`` backend alongside the device
        backend (required for FSDP CPU-offload collectives).

    Returns
    -------
    ``True`` if this call created the process group, ``False`` otherwise
    (already initialized, or no distributed launch detected).
    """
    if dist.is_initialized():
        return False

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        logger.debug(
            "No distributed launch detected (RANK/WORLD_SIZE unset); "
            "skipping process-group initialization."
        )
        return False

    device_type = get_device_name()
    if device_type != "cpu":
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        try:
            get_torch_device().set_device(local_rank)
        except Exception:  # pragma: no cover - defensive, device-specific
            logger.warning(
                "Failed to set %s device to local rank %d before process-group init",
                device_type,
                local_rank,
                exc_info=True,
            )

    backend_map = getattr(torch.distributed.Backend, "default_device_backend_map", {})
    backend = backend_map.get(device_type)
    if backend is None:
        if device_type == "npu":
            backend = get_nccl_backend()
        elif device_type == "cpu":
            backend = "gloo"
        else:
            backend = "nccl"

    if enable_cpu_backend and device_type != "cpu":
        backend = f"{device_type}:{backend},cpu:gloo"

    logger.info(
        "Initializing torch.distributed (backend=%s, timeout=%s, cpu_offload=%s)",
        backend,
        DEFAULT_PROCESS_GROUP_TIMEOUT,
        enable_cpu_backend,
    )
    dist.init_process_group(
        backend=backend,
        timeout=DEFAULT_PROCESS_GROUP_TIMEOUT,
    )
    return True
