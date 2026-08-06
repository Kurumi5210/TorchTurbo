"""LlamaFactory-specific training hooks."""

from __future__ import annotations

from ...logging import get_logger
from typing import Any

from ...engine.hooks import TrainHook, TrainState


logger = get_logger(__name__)


class LlamaFactoryHook(TrainHook):
    """Training hook that bridges LlamaFactory v0's logging and checkpointing.

    Parameters
    ----------
    training_args:
        LlamaFactory's ``Seq2SeqTrainingArguments`` (or compatible dict).
    output_dir:
        Directory for saving checkpoints.
    log_interval:
        Log metrics every N steps.
    save_interval:
        Save checkpoint every N steps (0 = end of epoch only).
    """

    def __init__(
        self,
        training_args: Any = None,
        output_dir: str | None = None,
        log_interval: int = 10,
        save_interval: int = 0,
    ) -> None:
        self.training_args = training_args
        self.output_dir = output_dir or getattr(training_args, "output_dir", "./output")
        self.log_interval = log_interval
        self.save_interval = save_interval

    def on_train_begin(self, state: TrainState) -> None:
        logger.info_rank0(
            "LlamaFactory training started: %d epochs, max_steps=%s",
            state.num_epochs,
            state.max_steps,
        )
