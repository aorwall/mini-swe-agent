"""Anthropic Toolcall Model with Extended Thinking and Caching Support.

This model combines:
1. Tool calling via LiteLLM
2. Extended thinking with interleaved-thinking beta header
3. Anthropic prompt caching for cost optimization
"""

import json
import time
from typing import Literal

from jinja2 import StrictUndefined, Template

from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.cache_control import set_cache_control

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}

# Beta header required for extended thinking with tool use on Anthropic models
# See: https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking
ANTHROPIC_INTERLEAVED_THINKING_HEADER = "interleaved-thinking-2025-05-14"


class AnthropicToolcallModelConfig(LitellmModelConfig):
    format_error_template: str = "Unknown tool '{{tool_name}}'. Valid tools: {{valid_tools}}"
    set_cache_control: Literal["default_end"] | None = "default_end"
    """Set explicit cache control markers for Anthropic prompt caching"""


class AnthropicToolcallModel(LitellmModel):
    """Anthropic model with tool calling, extended thinking, and prompt caching.

    Features:
    - Tool calling via LiteLLM's function calling interface
    - Extended thinking support via interleaved-thinking beta header
    - Anthropic prompt caching via cache_control markers
    """

    def __init__(self, **kwargs):
        super().__init__(config_class=AnthropicToolcallModelConfig, **kwargs)

    def query(self, messages: list[dict], **kwargs) -> dict:
        """Query the model with cache control applied."""
        # Apply cache control markers for Anthropic prompt caching
        if self.config.set_cache_control:
            messages = set_cache_control(messages, mode=self.config.set_cache_control)
        return super().query(messages, **kwargs)

    def _query(self, messages: list[dict[str, str]], **kwargs):
        """Execute query with tools and interleaved thinking header."""
        extra_headers = kwargs.pop('extra_headers', {})

        # Add beta header for interleaved thinking with tool use
        # This is required for Anthropic models to continue producing thinking blocks
        # after tool calls. Without this header, thinking stops after the first response.
        if self.config.model_kwargs.get('reasoning_effort'):
            existing_beta = extra_headers.get('anthropic-beta', '')
            if ANTHROPIC_INTERLEAVED_THINKING_HEADER not in existing_beta:
                if existing_beta:
                    extra_headers['anthropic-beta'] = f"{existing_beta},{ANTHROPIC_INTERLEAVED_THINKING_HEADER}"
                else:
                    extra_headers['anthropic-beta'] = ANTHROPIC_INTERLEAVED_THINKING_HEADER

        return super()._query(messages, tools=[BASH_TOOL], extra_headers=extra_headers, **kwargs)

    def parse_actions(self, response) -> list[dict]:
        """Parse tool calls from the response. Raises FormatError if unknown tool."""
        tool_calls = response.choices[0].message.tool_calls or []
        actions = []
        for tool_call in tool_calls:
            error_msg = ""
            try:
                args = json.loads(tool_call.function.arguments)
            except Exception as e:
                error_msg = f"Error parsing tool call arguments: {e}."
            if tool_call.function.name != "bash":
                error_msg += f"Unknown tool '{tool_call.function.name}'"
            if error_msg:
                raise FormatError(
                    {
                        "role": "user",
                        "content": error_msg,
                        "extra": {
                            "interrupt_type": "FormatError",
                        },
                    }
                )
            actions.append({"command": args["command"], "tool_call_id": tool_call.id})
        return actions

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Format execution outputs into tool result messages."""
        results = []
        actions = message.get("extra", {}).get("actions", [])
        for action, output in zip(actions, outputs):
            content = Template(self.config.observation_template, undefined=StrictUndefined).render(
                output=output, **(template_vars or {})
            )
            results.append(
                self.format_message(
                    role="tool",
                    tool_call_id=action["tool_call_id"],
                    content=content,
                    extra={
                        "raw_output": output.get("output", ""),
                        "returncode": output.get("returncode"),
                        "timestamp": time.time(),
                        **(
                            {"exception_info": output["exception_info"]} | output.get("extra", {})
                            if output.get("exception_info")
                            else {}
                        ),
                    },
                )
            )
        return results
