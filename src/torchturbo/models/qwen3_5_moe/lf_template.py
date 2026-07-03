"""LlamaFactory template registration for Qwen3.5-MoE.

Called by the LF adapter's ``_register_lf_templates()`` during activation.
"""

from __future__ import annotations

from ...logging import get_logger

logger = get_logger(__name__)


def register() -> None:
    """Register the qwen3_5 chat templates with LlamaFactory."""
    from llamafactory.data.template import (
        register_template,
        TEMPLATES,
        ReasoningTemplate,
    )
    from llamafactory.data.formatter import StringFormatter
    from llamafactory.data.mm_plugin import get_mm_plugin

    if "qwen3_5" not in TEMPLATES:
        register_template(
            name="qwen3_5",
            format_user=StringFormatter(
                slots=[
                    "<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"
                ]
            ),
            format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
            format_system=StringFormatter(
                slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]
            ),
            stop_words=["<|im_end|>"],
            replace_eos=True,
            template_class=ReasoningTemplate,
            mm_plugin=get_mm_plugin(
                name="qwen3_vl",
                image_token="<|image_pad|>",
                video_token="<|video_pad|>",
            ),
        )
        logger.debug("Registered LF template 'qwen3_5'")

    if "qwen3_5_no_think" not in TEMPLATES:
        register_template(
            name="qwen3_5_no_think",
            format_user=StringFormatter(
                slots=[
                    "<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"
                ]
            ),
            format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
            format_system=StringFormatter(
                slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]
            ),
            stop_words=["<|im_end|>"],
            replace_eos=True,
            mm_plugin=get_mm_plugin(
                name="qwen3_vl",
                image_token="<|image_pad|>",
                video_token="<|video_pad|>",
            ),
        )
        logger.debug("Registered LF template 'qwen3_5_no_think'")
