"""ms-swift template hook that keeps adapter kwargs Swift's hook would drop.

Shared by both Qwen3.5 dense and MoE: the fix is adapter-specific, not
dense/MoE-specific. Swift's ``pre_forward_hook`` fuses vision into
``inputs_embeds`` then restores only a fixed allowlist from the original inputs
(ms-swift base.py), so adapter-injected kwargs vanish after multimodal encoding:

* ``skip_lm_head`` — turbo chunked loss asks the model to skip lm_head and
  return pre-head states.
* ``input_ids`` — turbo MTP re-embeds the shifted next token from raw
  ``input_ids`` every layer.
"""

from __future__ import annotations

from copy import deepcopy

from ...logging import get_logger

logger = get_logger(__name__)

# Adapter kwargs Swift's pre_forward_hook drops; re-injected when present in the
# original inputs. input_ids is re-added only once ``inputs_embeds`` is set
# (vision fused), so it is harmless for non-MTP / pure-text runs.
PRESERVE_KEYS = ("skip_lm_head", "input_ids")


def _make_kwarg_preserving_template_cls():
    """Subclass of swift's ``Qwen3_5Template`` that re-injects adapter kwargs
    Swift's ``pre_forward_hook`` drops."""
    # Deferred import: swift is not importable at module import time.
    from swift.template.templates.qwen import Qwen3_5Template

    preserve_keys = PRESERVE_KEYS

    class Qwen3_5KwargPreserveTemplate(Qwen3_5Template):
        def pre_forward_hook(self, model, args, kwargs):
            saved = {k: kwargs.get(k) for k in preserve_keys if k in kwargs}
            args, kwargs = super().pre_forward_hook(model, args, kwargs)
            for key, value in saved.items():
                if key in kwargs:
                    continue
                kwargs[key] = value
            return args, kwargs

    Qwen3_5KwargPreserveTemplate.__turbo_kwarg_preserve__ = True
    return Qwen3_5KwargPreserveTemplate


def register() -> None:
    """Wrap native ``qwen3_5`` so ``--template qwen3_5`` keeps ``skip_lm_head``
    (chunked loss) and ``input_ids`` (MTP)."""
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
    logger.debug_rank0("Wrapped ms-swift template 'qwen3_5' for turbo adapter kwargs")


__all__ = ["register"]