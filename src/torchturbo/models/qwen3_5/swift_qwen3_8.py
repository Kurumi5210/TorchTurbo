# Copyright (c) ModelScope Contributors. All rights reserved.
"""Qwen3.8 template support for Swift releases without a native registration.

Reasoning instructions and rendering follow ms-swift v4.6.0, commit
ad818a2cdfafaf4305ac40c4acd892a211978bc2. Media processing stays in the
installed Swift Qwen3.5 template; history handling is local to this subclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from swift.template.template_inputs import StdTemplateInputs
from swift.template.templates.qwen import Qwen3_5Template, QwenTemplateMeta


@dataclass
class Qwen3_8TemplateMeta(QwenTemplateMeta):
    """Carry Qwen3.8's history policy without changing Swift's metadata class."""

    preserve_thinking: bool = True


class Qwen3_8Template(Qwen3_5Template):
    """Render Qwen3.8 messages with the installed Qwen3.5 media implementation."""

    reasoning_effort_instructions: dict[str, str] = {
        "xhigh": (
            "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
            "assumptions, consider plausible alternatives, and prioritize correctness, consistency, "
            "and clarity in the final answer."
        ),
        "medium": "",
        "low": (
            "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
            "conclusion without unnecessary elaboration."
        ),
    }
    default_reasoning_effort: str = "xhigh"

    def _get_preserve_thinking(self, inputs: StdTemplateInputs | None = None) -> bool:
        preserve = None if inputs is None else inputs.chat_template_kwargs.get("preserve_thinking")
        if preserve is None:
            preserve = self.preserve_thinking
        return True if preserve is None else preserve

    def _is_add_non_thinking_round(
        self, messages: list[dict[str, Any]], i: int, start_idx: int
    ) -> bool:
        # Qwen3.8 applies the prefix to all history, including last-round SFT.
        return super()._is_add_non_thinking_round(messages, i, -1)

    def _get_reasoning_instructions(self, inputs: StdTemplateInputs | None = None) -> str:
        if not self._get_enable_thinking(inputs):
            return ""
        effort = None if inputs is None else inputs.chat_template_kwargs.get("reasoning_effort")
        if effort is None:
            effort = self.chat_template_kwargs.get("reasoning_effort")
        if effort is None:
            effort = self.default_reasoning_effort
        if effort not in self.reasoning_effort_instructions:
            raise ValueError(
                f"Unexpected reasoning effort {effort}. Supported types are "
                f"{list(self.reasoning_effort_instructions.keys())}."
            )
        return self.reasoning_effort_instructions[effort]

    def _get_system(self, inputs: StdTemplateInputs) -> str | None:
        system = super()._get_system(inputs)
        instructions = self._get_reasoning_instructions(inputs)
        if not instructions:
            return system
        if system:
            return f"{instructions}\n\n{system}"
        return instructions

    def _jinja_encode(self, inputs: StdTemplateInputs) -> tuple[list[str], list[float], int]:
        for message in inputs.messages:
            content = message.get("content")
            if (
                message.get("role") != "assistant"
                or not isinstance(content, str)
                or message.get("reasoning_content") is not None
            ):
                continue
            if not content.startswith("<think>") or "</think>" not in content:
                continue
            reasoning, _, rest = content.partition("</think>")
            message["reasoning_content"] = reasoning[len("<think>"):].strip()
            message["content"] = rest.lstrip("\n")
        return super()._jinja_encode(inputs)
