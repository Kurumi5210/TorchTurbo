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


class _KwargPreserveTemplateMixin:
    def pre_forward_hook(
        self, model: Any, args: tuple, kwargs: dict[str, Any]
    ) -> tuple[tuple, dict[str, Any]]:
        saved = {key: kwargs[key] for key in PRESERVE_KEYS if key in kwargs}
        args, kwargs = super().pre_forward_hook(model, args, kwargs)
        for key, value in saved.items():
            if key not in kwargs:
                kwargs[key] = value
        return args, _post_process_inputs(kwargs)


def _make_kwarg_preserving_template_cls():
    """Wrap the native Qwen3.5 template without importing Swift eagerly."""
    from swift.template.templates.qwen import Qwen3_5Template

    class Qwen3_5KwargPreserveTemplate(_KwargPreserveTemplateMixin, Qwen3_5Template):
        pass

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
    if not issubclass(native.template_cls, _KwargPreserveTemplateMixin):
        wrapped = deepcopy(native)
        wrapped.template_cls = _make_kwarg_preserving_template_cls()
        register_template(wrapped, exist_ok=True)
        logger.debug_rank0("Wrapped ms-swift template 'qwen3_5' for turbo adapter kwargs")

    from swift import __version__

    if __version__ == "4.4.2" and "qwen3_8" not in TEMPLATE_MAPPING:
        from .swift_qwen3_8 import register_qwen3_8

        register_qwen3_8()


__all__ = ["register"]
