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
        logger.info(
            "LlamaFactory training started: %d epochs, max_steps=%s",
            state.num_epochs,
            state.max_steps,
        )
        self._apply_ulysses_patch(state)

    def _apply_ulysses_patch(self, state: TrainState) -> None:
        """Apply Ulysses attention patch dynamically on the first microbatch."""
        if getattr(self, "_ulysses_patched", False):
            return

        self._ulysses_patched = True

        sp_group = state.sp_info["sp_group"]
        if not sp_group:
            return

        sp_size = state.sp_info["sp_size"]
        if sp_size <= 1:
            return

        try:
            import transformers.modeling_flash_attention_utils as flash_utils
            import transformers.integrations.flash_attention as flash_integration
        except ImportError:
            raise ImportError(
                "transformers.modeling_flash_attention_utils not found. "
                "Ulysses attention patch requires a newer version of transformers."
            )

        if not hasattr(flash_utils, "_flash_attention_forward"):
            raise AttributeError(
                "transformers.modeling_flash_attention_utils._flash_attention_forward not found. "
                "Ulysses attention patch requires a newer version of transformers."
            )

        from functools import partial
        from torchturbo.models.modules.attention import ulysses

        patched_forward = partial(
            ulysses.ulysses_flash_attention_forward, sp_group=sp_group, sp_size=sp_size
        )

        flash_utils._flash_attention_forward = patched_forward
        if hasattr(flash_integration, "_flash_attention_forward"):
            flash_integration._flash_attention_forward = patched_forward

        logger.info(
            "Patched transformers.modeling_flash_attention_utils._flash_attention_forward via LlamaFactoryHook for SP size=%d",
            sp_size,
        )
