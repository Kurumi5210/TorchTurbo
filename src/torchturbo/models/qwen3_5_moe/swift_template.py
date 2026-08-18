"""ms-swift ``qwen3_5_mtp`` template — keeps raw input_ids so turbo MTP works.

swift's ``pre_forward_hook`` fuses vision into ``inputs_embeds`` then pops
``input_ids`` (ms-swift base.py). That starves turbo MTP, which re-embeds the
shifted next token from raw ``input_ids`` every layer. This template keeps
``_post_encode`` intact and only re-injects ``input_ids`` after the hook.

Opt in via ``--template qwen3_5_mtp`` when MTP is on; plain ``qwen3_5`` is
untouched otherwise. Loaded by the msswift adapter during activation.
"""

from __future__ import annotations

from copy import deepcopy

from ...logging import get_logger

logger = get_logger(__name__)


def _make_mtp_template_cls():
    # Deferred import: swift is not importable at module import time.
    from swift.template.templates.qwen import Qwen3_5Template

    class Qwen3_5MTPTemplate(Qwen3_5Template):
        def pre_forward_hook(self, model, args, kwargs):
            input_ids = kwargs.get("input_ids", None)
            args, kwargs = super().pre_forward_hook(model, args, kwargs)
            # Re-inject input_ids for the MTP heads (the turbo forward routes it
            # to None on the inner model, so no double vision fusion).
            if input_ids is not None and "inputs_embeds" in kwargs:
                kwargs["input_ids"] = input_ids
            return args, kwargs

    return Qwen3_5MTPTemplate


def register() -> None:
    """Register ``qwen3_5_mtp``: deepcopy of native ``qwen3_5`` meta with the
    MTP-aware template class (prompts/thinking/stop-words identical)."""
    from swift.template.register import TEMPLATE_MAPPING, register_template

    native = TEMPLATE_MAPPING.get("qwen3_5")
    if native is None:
        logger.debug_rank0(
            "swift_template: native 'qwen3_5' not registered; skipping 'qwen3_5_mtp'"
        )
        return

    if "qwen3_5_mtp" in TEMPLATE_MAPPING:
        return  # idempotent

    mtp_meta = deepcopy(native)
    mtp_meta.template_type = "qwen3_5_mtp"
    mtp_meta.template_cls = _make_mtp_template_cls()
    register_template(mtp_meta, exist_ok=True)
    logger.debug_rank0("Registered ms-swift template 'qwen3_5_mtp'")