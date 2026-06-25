"""NPU/CUDA profiling utilities for torchturbo.

Usage: pass ``--tt_profiling`` to enable. The profiler runs skip_first=1,
active=1 then stops — capturing exactly one training step after warmup.
Output goes to ``{output_dir}/profiling/``.
"""

from __future__ import annotations

import os
from typing import Any

from ..logging import get_logger

logger = get_logger(__name__)

_profiler: Any = None


def start_profiling(output_dir: str) -> None:
    """Start the profiler. Call once before the training loop."""
    global _profiler

    profiling_dir = os.path.join(output_dir, "profiling")
    os.makedirs(profiling_dir, exist_ok=True)

    try:
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

        _profiler = torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            schedule=torch_npu.profiler.schedule(
                wait=0, warmup=0, active=1, repeat=1, skip_first=1
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_dir),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_modules=False,
            with_flops=False,
            experimental_config=experimental_config,
        )
        _profiler.start()
        logger.info("Profiling started, output dir: %s", profiling_dir)

    except ImportError:
        import torch

        _profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=0, warmup=0, active=1, repeat=1, skip_first=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profiling_dir),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_modules=False,
            with_flops=False,
        )
        _profiler.start()
        logger.info("Profiling started (CUDA fallback), output dir: %s", profiling_dir)


def step_profiling() -> None:
    """Advance the profiler by one step. Call after each training step."""
    if _profiler is not None:
        _profiler.step()


def stop_profiling() -> None:
    """Stop the profiler and flush traces. Call after the training loop."""
    global _profiler
    if _profiler is not None:
        _profiler.stop()
        logger.info("Profiling stopped, traces flushed.")
        _profiler = None
