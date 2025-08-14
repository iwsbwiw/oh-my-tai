"""LLM data structures and helpers for oh-my-tai."""
import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from .tools import Tool


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """Parsed LLM response."""
    content: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_message: dict = field(default_factory=dict)


def create_user_message(content: str) -> dict:
    """Create a user role message.

    Args:
        content: The user's message content.

    Returns:
        Message dict with role="user" and content.
    """
    return {"role": "user", "content": content}


def create_system_message(content: str) -> dict:
    """Create a system role message.

    Args:
        content: The system message content.

    Returns:
        Message dict with role="system" and content.
    """
    return {"role": "system", "content": content}


def create_assistant_message(content: Optional[str], tool_calls: list[ToolCall] = None) -> dict:
    """Create an assistant role message.

    Args:
        content: The assistant's text content (can be None if only tool calls).
        tool_calls: Optional list of ToolCall objects.

    Returns:
        Message dict with role="assistant", content, and optionally tool_calls.
    """
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}
            }
            for tc in tool_calls
        ]
    return msg


def create_tool_result_message(tool_call_id: str, result: str) -> dict:
    """Create a tool role message with execution result.

    Args:
        tool_call_id: The ID of the tool call this result corresponds to.
        result: The tool execution result as a string.

    Returns:
        Message dict with role="tool", tool_call_id, and content.
    """
    return {"role": "tool", "tool_call_id": tool_call_id, "content": result}


# Default maximum history size (per D-07)
DEFAULT_MAX_HISTORY = 20

# Maximum tool calling rounds (per D-10)
MAX_TOOL_ROUNDS = 5

# Type alias for execute_tool callback
ExecuteToolCallback = Callable[["ToolCall"], str]


def truncate_history(messages: list[dict], max_messages: int = DEFAULT_MAX_HISTORY) -> list[dict]:
    """Truncate message history while preserving system messages.

    Per D-06, D-08: Sliding window truncation that preserves system messages
    and keeps the most recent user/assistant/tool messages.

    Args:
        messages: List of message dicts with "role" keys.
        max_messages: Maximum number of messages to keep.

    Returns:
        Truncated list of messages, or original if under limit.
    """
    if len(messages) <= max_messages:
        return messages

    # Separate system messages (always at start) from others
    system_messages = [m for m in messages if m.get("role") == "system"]
    other_messages = [m for m in messages if m.get("role") != "system"]

    # Calculate how many non-system messages we can keep
    keep_count = max_messages - len(system_messages)

    # If system messages exceed max, just return them all (edge case)
    if keep_count <= 0:
        return system_messages

    # Return system messages + most recent other messages
    return system_messages + other_messages[-keep_count:]


class LLMClient:
    """HTTP client for OpenAI-compatible LLM APIs using urllib.

    Per D-01: Uses urllib (stdlib), zero external dependencies.
    Per D-02: Direct HTTP POST to /chat/completions endpoint.
    Per D-12: API errors exit directly (raise RuntimeError).
    Per D-13: Error includes HTTP status and response body.
    """

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def call(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        """Make a chat completion request with tools.

        Args:
            messages: OpenAI-format message list
            tools: OpenAI-format tools list (each wrapped in {"type": "function", "function": {...}})

        Returns:
            LLMResponse with content and parsed tool_calls

        Raises:
            RuntimeError: On HTTP error or network error
        """
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
                return self._parse_response(data)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise RuntimeError(f"LLM API error {e.code}: {error_body}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Network error: {e.reason}")

    def _parse_response(self, data: dict) -> LLMResponse:
        """Parse OpenAI response into LLMResponse."""
        message = data.get("choices", [{}])[0].get("message", {})
        content = message.get("content")
        tool_calls = []

        for tc in message.get("tool_calls", []):
            if tc.get("type") == "function":
                function = tc.get("function", {})
                tool_calls.append(ToolCall(
                    id=tc.get("id", ""),
                    name=function.get("name", ""),
                    arguments=json.loads(function.get("arguments", "{}")),
                ))

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            raw_message=message,
        )


def agentic_loop(
    client: LLMClient,
    messages: list[dict],
    tools: list[dict],
    execute_tool: ExecuteToolCallback,
) -> str:
    """Run agentic loop until completion or max rounds.

    Per D-09: Multiple tool_calls execute in parallel, results returned together.
    Per D-10: Max 5 rounds of tool calling loop.
    Per D-11: End loop when tool_calls empty or only text response.

    Args:
        client: LLMClient instance
        messages: Conversation history (will be modified in place)
        tools: OpenAI-format tools list
        execute_tool: Callback function to execute a ToolCall and return result string

    Returns:
        Final text response from LLM, or "Max tool calling rounds reached"
    """
    for round_num in range(MAX_TOOL_ROUNDS):
        # Truncate history before each call to prevent token overflow
        messages[:] = truncate_history(messages)

        response = client.call(messages, tools)

        # Build assistant message with tool_calls for history
        assistant_message = response.raw_message
        messages.append(assistant_message)

        # Check for tool calls
        if not response.tool_calls:
            # No more tools to call - return final text
            return response.content or ""

        # Execute tools and add results to history
        # Note: D-09 requires parallel execution, but that's Phase 4 concern
        # For now, execute sequentially (parallel execution in Phase 4)
        for tool_call in response.tool_calls:
            result = execute_tool(tool_call)
            messages.append(create_tool_result_message(tool_call.id, result))

    return "Max tool calling rounds reached"


def tools_to_openai_format(tools: list["Tool"]) -> list[dict]:
    """Convert Tool objects to OpenAI tools format.

    Per RESEARCH.md Pitfall 1: Tool.to_openai_schema() returns the inner
    function object, but the API expects {"type": "function", "function": {...}}.

    Args:
        tools: List of Tool objects from tool discovery.

    Returns:
        List of OpenAI-format tool dicts.
    """
    return [
        {"type": "function", "function": tool.to_openai_schema()}
        for tool in tools
    ]
