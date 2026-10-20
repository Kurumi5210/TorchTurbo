"""Qwen3.8 text preprocessing for Swift 4.4.2.

Matches the Swift acb9f4cf template, cf6316ca last-tool loss boundary,
and ec8d1b01 JSON tool argument literals.
All overrides are scoped to the registered ``qwen3_8`` template.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any

from swift.agent_template import BaseAgentTemplate
from swift.agent_template.qwen3_coder import Qwen3_5AgentTemplate
from swift.loss_scale.base import LossScale
from swift.template import register_template
from swift.template.template_inputs import StdTemplateInputs, TemplateInputs
from swift.template.templates.qwen import Qwen3_5Template, QwenTemplateMeta

from .swift_template import _KwargPreserveTemplateMixin


def _last_user_or_tool(messages: list[dict[str, Any]]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] in ("user", "tool"):
            return i
    return -1


class _Qwen3_8AgentTemplate(Qwen3_5AgentTemplate):
    def _format_tool_calls(self, tool_call_messages: list[dict[str, Any]]) -> str:
        calls = []
        for message in tool_call_messages:
            call = self._parse_tool_call(message["content"])
            if call.get("arguments"):
                # The 4.4.2 formatter already handles dict/list JSON. Convert
                # scalar JSON literals before its str() branch, without
                # rewriting literal strings or nested JSON values.
                call = {**call, "arguments": {
                    name: json.dumps(value) if isinstance(value, bool) or value is None else value
                    for name, value in call["arguments"].items()
                }}
            calls.append({**message, "content": call})
        return super()._format_tool_calls(calls)


class _ToolBoundaryLossScale(LossScale):
    """Reuse Swift loss rules with tools treated as last-round boundaries."""

    def __init__(self, delegate: LossScale) -> None:
        super().__init__(delegate.base_strategy)
        self.delegate = delegate

    @property
    def is_binary_loss_scale(self) -> bool:
        return self.delegate.is_binary_loss_scale

    def get_loss_scale(self, context: str, **kwargs: Any) -> tuple[list, list[float]]:
        return self.delegate.get_loss_scale(context, **kwargs)

    def __call__(
        self, context_list: list, context_types: list, messages: list, **kwargs: Any
    ) -> tuple[list, list[float]]:
        # LossScale reads roles only to find the last query. Content and explicit
        # loss/loss_scale fields must remain intact, including list-valued fields.
        queries = [
            {**message, "role": "user"} if message["role"] == "tool" else message
            for message in messages
        ]
        return self.delegate(context_list, context_types, queries, **kwargs)


class _ToolBoundaryTemplateInputs(TemplateInputs):
    @staticmethod
    def _compat_rejected_response(inputs: dict[str, Any]) -> None:
        # TemplateInputs also imports get_last_user_round. Keep preference
        # construction scoped to this template, including raw role="tool".
        if "rejected_response" not in inputs:
            return
        messages = inputs["messages"]
        boundary = _last_user_or_tool(messages) + 1
        rejected = inputs.pop("rejected_response")
        if isinstance(rejected, str):
            rejected = [{"role": "assistant", "content": rejected}]
        elif not isinstance(rejected, list):
            raise ValueError("rejected_response must be a string or a list of messages.")
        if any(message["role"] == "user" for message in rejected):
            raise ValueError("rejected_response must not contain user messages.")
        if (len(messages[boundary:]) == 1 and len(rejected) == 1
                and rejected[0]["content"] == messages[boundary]["content"]):
            raise ValueError("The chosen and rejected responses must differ.")
        inputs["rejected_messages"] = deepcopy(messages[:boundary]) + rejected


@dataclass
class Qwen3_8TemplateMeta(QwenTemplateMeta):
    preserve_thinking: bool = True


class Qwen3_8Template(_KwargPreserveTemplateMixin, Qwen3_5Template):
    """Backport Qwen3.8 framing without changing Swift's global helpers."""

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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._version += "-turbo-qwen38-v2"

    def encode(
        self, inputs: Any, return_template_inputs: bool = False, return_length: bool = False,
    ) -> dict[str, Any]:
        if isinstance(inputs, dict) and "rejected_response" in inputs:
            inputs = _ToolBoundaryTemplateInputs.from_dict(inputs)
        return super().encode(inputs, return_template_inputs=return_template_inputs, return_length=return_length)

    @property
    def loss_scale(self) -> LossScale:
        loss_scale = super().loss_scale
        if not isinstance(loss_scale, _ToolBoundaryLossScale):
            loss_scale = _ToolBoundaryLossScale(loss_scale)
            self._loss_scale_cache[self._loss_scale] = loss_scale
        return loss_scale

    @property
    def agent_template(self) -> BaseAgentTemplate:
        agent = super().agent_template
        if self._agent_template == "qwen3_5" and not isinstance(agent, _Qwen3_8AgentTemplate):
            agent = _Qwen3_8AgentTemplate()
            self._agent_template_cache[self._agent_template] = agent
        return agent

    def _get_preserve_thinking(self, inputs: StdTemplateInputs | None = None) -> bool:
        preserve = None if inputs is None else inputs.chat_template_kwargs.get("preserve_thinking")
        if preserve is None:
            preserve = self.preserve_thinking
        if preserve is None:
            preserve = True if self.use_chat_template else super()._get_preserve_thinking(inputs)
        return preserve

    def _add_non_thinking_prefix(
        self, inputs: StdTemplateInputs, thinking_prefix: str = "<think>"
    ) -> None:
        prefix = self.template_meta.non_thinking_prefix
        if not prefix:
            return
        # Qwen3.8 metadata preserves history, so even last_round inserts the
        # non-thinking prefix in every assistant round, as in upstream Swift.
        for i, message in enumerate(inputs.messages):
            if not self._is_add_non_thinking_round(inputs.messages, i, -1):
                continue
            content = message["content"]
            if isinstance(content, list):
                if content and isinstance(content[0], str) and not content[0].startswith((thinking_prefix, prefix)):
                    content[0] = prefix + content[0]
            elif isinstance(content, str) and not content.startswith((thinking_prefix, prefix)):
                message["content"] = prefix + content

    def _remove_history_thinking(self, inputs: StdTemplateInputs) -> None:
        boundary = _last_user_or_tool(inputs.messages)
        for i, message in enumerate(inputs.messages):
            if i >= boundary or message["role"] != "assistant":
                continue
            content = message["content"]
            if isinstance(content, list) and content and isinstance(content[0], str):
                content[0] = self._remove_thinking_content(content[0])
            elif isinstance(content, str):
                message["content"] = self._remove_thinking_content(content)

    def _get_reasoning_instructions(self, inputs: StdTemplateInputs | None = None) -> str:
        if not self._get_enable_thinking(inputs):
            return ""
        effort = None if inputs is None else inputs.chat_template_kwargs.get("reasoning_effort")
        if effort is None:
            effort = self.chat_template_kwargs.get("reasoning_effort")
        if effort is None:
            effort = self.default_reasoning_effort
        if effort not in self.reasoning_effort_instructions:
            raise ValueError(f"Unsupported Qwen3.8 reasoning_effort: {effort!r}")
        return self.reasoning_effort_instructions[effort]

    def _get_system(self, inputs: StdTemplateInputs) -> str | None:
        system = super()._get_system(inputs)
        instruction = self._get_reasoning_instructions(inputs)
        if not instruction:
            return system
        return f"{instruction}\n\n{system}" if system else instruction

    def _jinja_encode(self, inputs: StdTemplateInputs) -> Any:
        for message in inputs.messages:
            content = message.get("content")
            if (message["role"] != "assistant" or not isinstance(content, str)
                    or message.get("reasoning_content") is not None):
                continue
            if content.startswith("<think>") and "</think>" in content:
                reasoning, _, rest = content.partition("</think>")
                message["reasoning_content"] = reasoning[len("<think>"):].strip()
                message["content"] = rest.lstrip("\n")
        return super()._jinja_encode(inputs)


def register_qwen3_8() -> None:
    """Register the compatibility template after Swift's native templates."""
    register_template(Qwen3_8TemplateMeta(
        "qwen3_8",
        template_cls=Qwen3_8Template,
        default_system=None,
        thinking_prefix="<think>\n",
        non_thinking_prefix="<think>\n\n</think>\n\n",
        agent_template="qwen3_5",
        is_thinking=True,
    ))
