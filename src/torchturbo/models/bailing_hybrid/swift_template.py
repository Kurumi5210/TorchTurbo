"""ms-swift chat template for Bailing V3.

Mirrors the Ling v3 jinja: ``<role>SYSTEM/HUMAN/ASSISTANT/OBSERVATION</role>``
closed by ``<|role_end|>``, hybrid thinking, a ``detailed thinking on/off``
system directive, and ``<function=...><parameter=...>`` tool calls with
```` ```result ... ``` ```` fenced observations.
"""

from __future__ import annotations

from ...logging import get_logger

logger = get_logger(__name__)

# Thinking markers (literal special tokens of the bailing tokenizer; verified
# byte-for-byte against ms-swift's ring2 template and antllm's chat_format).
THINK_START = '<think>'
THINK_END = '</think>'

# Role wrappers.
ROLE_END = '<|role_end|>'
SYS_OPEN = '<role>SYSTEM</role>'
HUMAN_OPEN = '<role>HUMAN</role>'
ASSISTANT_OPEN = '<role>ASSISTANT</role>'
OBSERVATION_OPEN = '<role>OBSERVATION</role>'

# ``detailed thinking`` directive options injected into the system prompt.
DETAILED_THINKING_ON = 'detailed thinking on'
DETAILED_THINKING_OFF = 'detailed thinking off'

# Fenced block used for tool/observation results.
RESULT_FENCE = '```result'

# Tool-call XML building blocks (kept split so the source stays unambiguous).
FN_OPEN = '<function='
FN_CLOSE = '</function>'
PARAM_OPEN = '<parameter='
PARAM_CLOSE = '</parameter>'

# The system-side tools preamble the Ling v3 jinja emits when tools are present.
LING_V3_TOOLS_PREAMBLE = (
    '# Tools\n\n'
    'You may call one or more functions to assist with the user query.\n\n'
    'You are provided with function signatures within <tools></tools> XML tags:\n<tools>'
)
LING_V3_TOOLS_SUFFIX = (
    '\n</tools>\n\n'
    'If none of the functions can be used, point it out. '
    'If the given question lacks the parameters required by the function, also point it out.\n'
    'If you need to use a function, for each function call, output the function name and arguments '
    'within the following XML format:\n'
    '<function=func_name>\n<parameter=arg_key>\narg_value\n</parameter>\n...\n</function>\n'
)


def _make_ling_v3_agent_template_cls():
    """Ling v3 agent template on swift's model-agnostic ``BaseAgentTemplate``."""
    import json
    import re
    from typing import List, Optional, Tuple, Union

    from swift.agent_template.base import BaseAgentTemplate
    from swift.infer_engine import Function
    from swift.template import Prompt

    class LingV3AgentTemplate(BaseAgentTemplate):

        @staticmethod
        def _parse_function_block(block: str) -> Optional[Function]:
            """Parse a single ``<function=...>...</function>`` block."""
            block = block.strip()
            if not block.startswith(FN_OPEN) or not block.endswith(FN_CLOSE):
                return None
            name_match = re.search(r'<function=([^>]+)>', block)
            if not name_match:
                return None
            func_name = name_match.group(1).strip()
            parameters = {}
            for param_name, param_value in re.findall(
                    r'<parameter=([^>]+)>\s*(.*?)\s*</parameter>', block, re.DOTALL):
                parameters[param_name.strip()] = param_value.strip()
            return Function(name=func_name, arguments=json.dumps(parameters, ensure_ascii=False))

        def get_toolcall(self, response: str) -> List[Function]:
            """Reverse of ``_format_tool_calls`` -- parse bare ``<function=...>`` blocks
            out of generated text into :class:`Function` objects (used at inference).
            Falls back to the ReAct parser when no block is found.
            """
            functions = []
            for match in re.finditer(r'<function=.*?</function>', response, re.DOTALL):
                fn = self._parse_function_block(match.group(0))
                if fn is not None:
                    functions.append(fn)
            if not functions:
                # Fall back to the ReAct-style parser for compatibility.
                functions = super().get_toolcall(response)
            return functions

        def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None,
                          user_message=None) -> str:
            """Render the tools preamble the jinja emits under ``{%- if tools %}``:

                # Tools
                You may call one or more functions ...
                <tools>
                {tool|tojson}
                ...
                </tools>
                If you need to use a function ...

            The ``detailed thinking`` directive is NOT added here -- it lives in
            :meth:`LingV3Template._get_system` so it lands *after* the tools block (per
            the jinja ordering). ``system`` (if given) is prepended as ``system\n``.
            """
            tool_lines = []
            for tool in tools:
                if isinstance(tool, dict) and tool.get('type') == 'function':
                    wrapped = tool
                else:
                    wrapped = self.wrap_tool(tool)
                tool_lines.append(json.dumps(wrapped, ensure_ascii=False))
            tools_block = LING_V3_TOOLS_PREAMBLE + '\n' + '\n'.join(tool_lines) + LING_V3_TOOLS_SUFFIX
            if system:
                return system.rstrip('\n') + '\n' + tools_block
            return tools_block

        def _format_tool_calls(self, tool_call_messages) -> str:
            """Render each tool call as bare XML (the jinja assistant-branch
            tool-call emit, no outer wrapper):

                <function={name}>
                <parameter={k}>
                {v}
                </parameter>
                ...
                </function>

            Multiple calls are separated by a single ``\\n``."""
            parts = []
            for idx, message in enumerate(tool_call_messages):
                tool_call = self._parse_tool_call(message['content'])
                parts.append(FN_OPEN + tool_call['name'] + '>\n')
                args = tool_call.get('arguments') or {}
                for arg_name, arg_value in args.items():
                    # jinja: ``v is string ? v : v | tojson``
                    if not isinstance(arg_value, str):
                        arg_value = json.dumps(arg_value, ensure_ascii=False)
                    parts.append(PARAM_OPEN + arg_name + '>\n' + arg_value + '\n' + PARAM_CLOSE + '\n')
                parts.append(FN_CLOSE + '\n')
                if idx != len(tool_call_messages) - 1:
                    parts.append('\n')
            return ''.join(parts)

        def _format_tool_responses(
            self,
            assistant_content: str,
            tool_messages,
        ) -> Tuple[str, 'Prompt']:
            """Render the jinja tool branch as the ``tool`` message payload:

                <role>OBSERVATION</role>
                ```result
                {content}
                ```
                ...
                <|role_end|>

            ``<role>OBSERVATION</role>`` is NOT a swift template slot -- a
            ``role=='tool'`` message's content is concatenated verbatim, so the wrapper
            is emitted here (cf. ``HermesAgentTemplate``). Preceding assistant content
            is returned unchanged.
            """
            res = [OBSERVATION_OPEN]
            for tool_message in tool_messages:
                res.append('\n' + RESULT_FENCE + '\n' + tool_message['content'] + '\n```')
            res.append(ROLE_END)
            return assistant_content, [''.join(res)]

        def _add_tool_call_prefix(self, tool_content: str, pre_message=None) -> str:
            """Insert one ``\n`` before a tool call when the preceding assistant turn
            has non-empty content, mirroring the jinja tool-call loop:

                {%- if (loop.first and content) or (not loop.first) %}
                {{ '\n' }}
                {%- endif %}

            where ``content`` is the assistant text after the think block.
            """
            if not pre_message or pre_message.get('role') != 'assistant':
                return tool_content
            content = pre_message.get('content', '')
            if not isinstance(content, str):
                return tool_content
            if THINK_END in content:
                effective = content.split(THINK_END)[-1]
            elif THINK_START in content:
                effective = content.split(THINK_START)[-1]
            else:
                effective = content
            if effective.strip():
                return '\n' + tool_content
            return tool_content

    return LingV3AgentTemplate


def _make_ling_v3_template_cls():
    """Ling v3 ``Template`` subclass (``detailed thinking`` system directive)."""
    from swift.template import Template

    class LingV3Template(Template):
        """Ling v3 chat template with hybrid thinking + detailed-thinking directive."""

        def _get_system(self, inputs):
            """Inject the ``detailed thinking {on|off}`` directive into the system
            (default ``on`` unless ``enable_thinking`` is False), preserved when already
            present. Placement follows the jinja ordering:

                no tools: system + '\n' + directive   (or directive if no system)
                tools:    system + '\n' + tools_block + directive
                          ^ directive AFTER tools (jinja ``{%- if tools %}`` branch)
            """
            template_meta = self.template_meta
            system = inputs.system
            tools = inputs.tools
            template_meta.check_system(system)
            if system is None:
                system = template_meta.default_system

            enable = self._get_enable_thinking(inputs)
            thinking_on = enable is not False
            directive = DETAILED_THINKING_ON if thinking_on else DETAILED_THINKING_OFF
            has_directive = isinstance(system, str) and (
                DETAILED_THINKING_ON in system or DETAILED_THINKING_OFF in system)

            if tools:
                inner = self.agent_template._format_tools(
                    tools, system if isinstance(system, str) else None, inputs.messages[0])
                if not has_directive:
                    inner = inner + directive
                return inner

            if isinstance(system, str) and system:
                return system if has_directive else system.rstrip('\n') + '\n' + directive
            return directive

    return LingV3Template


def _register_ling_v3_agent_template() -> None:
    from swift.agent_template import agent_template_map
    if 'ling_v3' in agent_template_map:
        return
    agent_template_map['ling_v3'] = _make_ling_v3_agent_template_cls()
    logger.debug_rank0("Registered ms-swift agent_template 'ling_v3'")


def register() -> None:
    """Register the ``ling_v3`` template + agent template and point
    ``turbo_bailing_hybrid`` at it."""
    from swift.model.model_meta import MODEL_MAPPING
    from swift.template.register import TEMPLATE_MAPPING, register_template
    from swift.template.template_meta import TemplateMeta

    _register_ling_v3_agent_template()

    if 'ling_v3' not in TEMPLATE_MAPPING:
        ling_v3_template_cls = _make_ling_v3_template_cls()
        meta = TemplateMeta(
            template_type='ling_v3',
            prefix=[],
            system_prefix=[SYS_OPEN + '\n{{SYSTEM}}' + ROLE_END + '\n\n'],
            prompt=[HUMAN_OPEN + '\n{{QUERY}}' + ROLE_END + '\n\n' + ASSISTANT_OPEN + '\n'],
            chat_sep=[ROLE_END + '\n\n'],
            suffix=[ROLE_END + '\n\n'],
            template_cls=ling_v3_template_cls,
            default_system=None,
            is_thinking=True,
            thinking_prefix=THINK_START,
            non_thinking_prefix=THINK_START + THINK_END,
            history_thinking_prefix=THINK_START + THINK_END,
            agent_template='ling_v3',
        )
        register_template(meta, exist_ok=True)
        logger.debug_rank0("Registered ms-swift template 'ling_v3'")

    # ``turbo_bailing_hybrid`` would otherwise inherit the V2.5 template: the
    # turbo name ``bailing_hybrid`` collides with swift's native V2.5
    # ``LLMModelType.bailing_hybrid``. architectures/requires/tags need no patch
    # (model_registry keeps architectures empty; V2.5 sets no requires/tags).
    turbo_meta = MODEL_MAPPING.get('turbo_bailing_hybrid')
    if turbo_meta is not None:
        turbo_meta.template = 'ling_v3'
        logger.debug_rank0("Patched turbo_bailing_hybrid ModelMeta -> template='ling_v3'")

__all__ = ['register']