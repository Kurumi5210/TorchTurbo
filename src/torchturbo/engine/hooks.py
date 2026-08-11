"""Training hooks and state — the extension mechanism for the training loop.

Adapters and users register :class:`TrainHook` subclasses to customize the
training loop at 12 hook points.  All hooks receive a mutable
:class:`TrainState` context so they can inspect or modify training state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from ..logging import get_logger


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# TrainState — mutable context shared across hooks
# ---------------------------------------------------------------------------


@dataclass
class TrainState:
    """Mutable context passed to every hook.

    Hooks can read and modify any field.  For example, a hook could
    replace ``loss`` before backward to apply custom scaling.
    """

    # Model / optimizer (set once at init)
    model: nn.Module | None = None
    optimizer: Any = None
    lr_scheduler: Any = None

    # Progress
    epoch: int = 0
    global_step: int = 0
    max_steps: int | None = None
    num_epochs: int = 1

    # Per-step values (reset each step)
    batch: Any = None
    loss: torch.Tensor | None = None
    outputs: Any = None
    grad_norm: float | None = None

    # Distributed / Parallel state
    sp_info: Any = None

    # Control flags (hooks can set these to alter loop behavior)
    should_stop: bool = False
    """Set to ``True`` in any hook to stop training after the current step."""

    skip_optimizer_step: bool = False
    """Set to ``True`` in ``before_optimizer_step`` to skip the step (e.g. NaN grads)."""

    # Metrics accumulator (hooks can append to this for logging)
    metrics: dict[str, Any] = field(default_factory=dict)
    """Arbitrary metrics dict — hooks can write here and loggers can read."""


# ---------------------------------------------------------------------------
# TrainHook — base class with no-op defaults
# ---------------------------------------------------------------------------


class TrainHook:
    """Base class for training hooks.

    Subclass this and override the methods you care about.  All methods
    receive the current :class:`TrainState` and can modify it in place.

    **Hook execution order** within a single step::

        on_train_begin
        ├── on_epoch_begin
        │   ├── on_step_begin
        │   │   ├── on_microbatch_begin
        │   │   │   ├── before_forward
        │   │   │   ├── after_forward
        │   │   │   ├── before_backward
        │   │   │   ├── after_backward
        │   │   └── on_microbatch_end
        │   │   ├── before_optimizer_step
        │   │   ├── after_optimizer_step
        │   │   └── on_step_end
        │   └── on_epoch_end
        └── on_train_end

    Example::

        class LoggingHook(TrainHook):
            def on_step_end(self, state: TrainState) -> None:
                if state.global_step % 10 == 0:
                    logger.info_rank0(
                        "Step %d, Loss: %s",
                        state.global_step, state.metrics.get('loss', '?'),
                    )
    """

    # pylint: disable=unused-argument

    def on_train_begin(self, state: TrainState) -> None:
        """Called once before the first epoch."""

    def on_train_end(self, state: TrainState) -> None:
        """Called once after the last epoch (or early stop)."""

    def on_epoch_begin(self, state: TrainState) -> None:
        """Called at the start of each epoch."""

    def on_epoch_end(self, state: TrainState) -> None:
        """Called at the end of each epoch."""

    def on_step_begin(self, state: TrainState) -> None:
        """Called at the start of each optimizer's step."""

    def on_step_end(self, state: TrainState) -> None:
        """Called at the end of each optimizer's step."""

    def on_microbatch_begin(self, state: TrainState) -> None:
        """Called at the start of each micro-batch."""

    def on_microbatch_end(self, state: TrainState) -> None:
        """Called at the end of each micro-batch."""

    def before_forward(self, state: TrainState) -> None:
        """Called just before model forward / compute_loss."""

    def after_forward(self, state: TrainState) -> None:
        """Called just after compute_loss (``state.loss`` is set)."""

    def before_backward(self, state: TrainState) -> None:
        """Called before ``loss.backward()``."""

    def after_backward(self, state: TrainState) -> None:
        """Called after ``loss.backward()``."""

    def before_optimizer_step(self, state: TrainState) -> None:
        """Called before ``optimizer.step()``.

        Set ``state.skip_optimizer_step = True`` to skip (e.g. NaN grads).
        """

    def after_optimizer_step(self, state: TrainState) -> None:
        """Called after ``optimizer.step()`` and ``lr_scheduler.step()``."""
