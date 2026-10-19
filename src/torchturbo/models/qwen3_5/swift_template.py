"""ms-swift integration for the Qwen3.5 family: template wrap + input post-process.

Swift's ``pre_forward_hook`` fuses vision into ``inputs_embeds`` and drops the
original ``input_ids``. Turbo MTP needs those IDs to embed the shifted token.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

from ...logging import get_logger
logger = get_logger(__name__)

PRESERVE_KEYS = ("input_ids",)
DPO_DEBUG_EXPERIMENT = "history_think"


def _post_process_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Merge Swift text and mRoPE rows into Turbo's ``[4, B, S]`` layout."""
    tpos = inputs.get("text_position_ids")
    pos = inputs.get("position_ids")
    if pos is not None and tpos is not None and pos.dim() == 3 and pos.shape[0] == 3:
        inputs["position_ids"] = torch.cat(
            [tpos.reshape(1, *pos.shape[1:]), pos], dim=0
        )
        inputs.pop("text_position_ids", None)
    return inputs


def _make_kwarg_preserving_template_cls():
    """Subclass of swift's ``Qwen3_5Template`` that re-injects adapter kwargs
    Swift's ``pre_forward_hook`` drops, then merges the mRoPE rows."""
    # Deferred import: swift is not importable at module import time.
    from swift.template.templates.qwen import Qwen3_5Template

    preserve_keys = PRESERVE_KEYS

    class Qwen3_5KwargPreserveTemplate(Qwen3_5Template):
        __turbo_dpo_debug__ = DPO_DEBUG_EXPERIMENT

        def _get_preserve_thinking(self, inputs: Any = None) -> bool:
            # Match Qwen3.8 history handling without changing loss boundaries.
            if self.mode == "rlhf" and self.loss_scale.base_strategy == "last_round":
                return True
            return super()._get_preserve_thinking(inputs)

        def _is_add_non_thinking_round(
            self, messages: list[dict[str, Any]], i: int, start_idx: int
        ) -> bool:
            if self.mode == "rlhf" and self.loss_scale.base_strategy == "last_round":
                start_idx = -1
            return super()._is_add_non_thinking_round(messages, i, start_idx)

        def pre_forward_hook(self, model, args, kwargs):
            saved = {k: kwargs.get(k) for k in preserve_keys if k in kwargs}
            args, kwargs = super().pre_forward_hook(model, args, kwargs)
            for key, value in saved.items():
                if key in kwargs:
                    continue
                kwargs[key] = value
            return args, _post_process_inputs(kwargs)

    Qwen3_5KwargPreserveTemplate.__turbo_kwarg_preserve__ = True
    return Qwen3_5KwargPreserveTemplate


def register() -> None:
    """Wrap native ``qwen3_5`` so ``--template qwen3_5`` keeps MTP inputs."""
    from swift.template.register import TEMPLATE_MAPPING, register_template

    native = TEMPLATE_MAPPING.get("qwen3_5")
    if native is None:
        logger.debug_rank0(
            "swift_template: native 'qwen3_5' not registered; skipping turbo qwen3.5 template"
        )
        return

    # Idempotency: do not re-wrap a class we already wrapped.
    if getattr(native.template_cls, "__turbo_kwarg_preserve__", False):
        return

    wrapped = deepcopy(native)
    wrapped.template_cls = _make_kwarg_preserving_template_cls()
    register_template(wrapped, exist_ok=True)
    logger.info_rank0(
        "DPO debug template registered: %s (qwen3_5, mode=rlhf, last_round)",
        DPO_DEBUG_EXPERIMENT,
    )
    logger.debug_rank0("Wrapped ms-swift template 'qwen3_5' for turbo adapter kwargs")


__all__ = ["register"]
