"""ms-swift integration for Qwen3.5 and Qwen3.8 templates.

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


def _make_kwarg_preserving_template_cls(base_template_cls: type) -> type:
    """Preserve native template behavior while restoring Turbo model inputs."""
    preserve_keys = PRESERVE_KEYS

    class QwenKwargPreserveTemplate(base_template_cls):
        __turbo_kwarg_preserve__ = True

        def pre_forward_hook(
            self,
            model: torch.nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            saved = {k: kwargs.get(k) for k in preserve_keys if k in kwargs}
            args, kwargs = super().pre_forward_hook(model, args, kwargs)
            for key, value in saved.items():
                if key in kwargs:
                    continue
                kwargs[key] = value
            return args, _post_process_inputs(kwargs)

    return QwenKwargPreserveTemplate


def register() -> None:
    """Wrap registered Qwen templates without changing their encoding metadata."""
    from swift.template.register import TEMPLATE_MAPPING, register_template

    for template_type in ("qwen3_5", "qwen3_8"):
        native = TEMPLATE_MAPPING.get(template_type)
        if native is None:
            logger.debug_rank0(
                "swift_template: native %r not registered; skipping turbo template",
                template_type,
            )
            continue

        # Idempotency: do not re-wrap a class we already wrapped.
        if getattr(native.template_cls, "__turbo_kwarg_preserve__", False):
            continue

        wrapped = deepcopy(native)
        wrapped.template_cls = _make_kwarg_preserving_template_cls(native.template_cls)
        register_template(wrapped, exist_ok=True)
        logger.debug_rank0(
            "Wrapped ms-swift template %r for turbo adapter kwargs", template_type
        )


__all__ = ["register"]
