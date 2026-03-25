"""oh-my-tai 的 LLM 数据结构与辅助函数。"""
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
    """表示一次来自 LLM 的工具调用。"""
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """解析后的 LLM 响应。"""
    content: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_message: dict = field(default_factory=dict)


def create_user_message(content: str) -> dict:
    """创建一条 ``user`` 角色消息。

    Args:
        content: 用户消息内容。

    Returns:
        包含 ``role="user"`` 和 ``content`` 的消息字典。
    """
    return {"role": "user", "content": content}


def create_system_message(content: str) -> dict:
    """创建一条 ``system`` 角色消息。

    Args:
        content: system message 的内容。

    Returns:
        包含 ``role="system"`` 和 ``content`` 的消息字典。
    """
    return {"role": "system", "content": content}


def create_assistant_message(content: Optional[str], tool_calls: list[ToolCall] = None) -> dict:
    """创建一条 ``assistant`` 角色消息。

    Args:
        content: assistant 的文本内容；如果只有工具调用可以为 ``None``。
        tool_calls: 可选的 ``ToolCall`` 列表。

    Returns:
        包含 ``role="assistant"``、``content`` 以及可选 ``tool_calls`` 的消息字典。
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
    """创建一条携带执行结果的 ``tool`` 角色消息。

    Args:
        tool_call_id: 本结果对应的工具调用 ID。
        result: 字符串形式的工具执行结果。

    Returns:
        包含 ``role="tool"``、``tool_call_id`` 和 ``content`` 的消息字典。
    """
    return {"role": "tool", "tool_call_id": tool_call_id, "content": result}


# 默认最大历史消息数（对应 D-07）
DEFAULT_MAX_HISTORY = 20

# 最大工具调用轮数（对应 D-10）
MAX_TOOL_ROUNDS = 5

# execute_tool 回调的类型别名
ExecuteToolCallback = Callable[["ToolCall"], str]


def truncate_history(messages: list[dict], max_messages: int = DEFAULT_MAX_HISTORY) -> list[dict]:
    """裁剪消息历史，并保留 system message。

    对应 D-06、D-08：使用滑动窗口方式裁剪历史，
    保留 system message，同时保留最近的 user/assistant/tool 消息。

    Args:
        messages: 带有 ``role`` 字段的消息字典列表。
        max_messages: 允许保留的最大消息数。

    Returns:
        裁剪后的消息列表；如果未超限则返回原列表。
    """
    if len(messages) <= max_messages:
        return messages

    # 将 system message 与其他消息分开
    system_messages = [m for m in messages if m.get("role") == "system"]
    other_messages = [m for m in messages if m.get("role") != "system"]

    # 计算还能保留多少条非 system 消息
    keep_count = max_messages - len(system_messages)

    # 边界情况：如果 system message 已超过上限，直接全部返回
    if keep_count <= 0:
        return system_messages

    # 返回 system message 和最近的其他消息
    return system_messages + other_messages[-keep_count:]


class LLMClient:
    """基于 ``urllib`` 的 OpenAI 兼容 LLM HTTP 客户端。

    对应 D-01：使用标准库 ``urllib``，不引入额外依赖。
    对应 D-02：直接向 ``/chat/completions`` 发起 HTTP POST。
    对应 D-12：API 错误直接抛出 ``RuntimeError``。
    对应 D-13：错误信息包含 HTTP 状态码和响应体。
    """

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def call(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        """携带 tools 发起一次 chat completion 请求。

        Args:
            messages: OpenAI 格式的消息列表。
            tools: OpenAI 格式的 tools 列表，
                每项形如 ``{"type": "function", "function": {...}}``。

        Returns:
            包含文本内容和解析后 ``tool_calls`` 的 ``LLMResponse``。

        Raises:
            RuntimeError: 在 HTTP 错误或网络错误时抛出。
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
        """将 OpenAI 响应解析为 ``LLMResponse``。"""
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
    """运行 agent loop，直到完成或达到最大轮数。

    对应 D-09：多个 ``tool_calls`` 应并行执行，并统一返回结果。
    对应 D-10：最多进行 5 轮工具调用。
    对应 D-11：当 ``tool_calls`` 为空或仅返回文本时结束循环。

    Args:
        client: ``LLMClient`` 实例。
        messages: 对话历史，会被原地修改。
        tools: OpenAI 格式的 tools 列表。
        execute_tool: 用于执行 ``ToolCall`` 并返回结果字符串的回调。

    Returns:
        LLM 最终返回的文本；如果超出轮数限制则返回提示字符串。
    """
    for round_num in range(MAX_TOOL_ROUNDS):
        # 每轮调用前先裁剪历史，避免 token 过长
        messages[:] = truncate_history(messages)

        response = client.call(messages, tools)

        # 将包含 tool_calls 的 assistant 消息加入历史
        assistant_message = response.raw_message
        messages.append(assistant_message)

        # 检查本轮是否有工具调用
        if not response.tool_calls:
            # 没有更多工具调用，直接返回最终文本
            return response.content or ""

        # 执行工具，并把结果写回历史
        # 注意：D-09 要求并行执行，但那是后续阶段的工作
        # 当前实现仍按顺序执行
        for tool_call in response.tool_calls:
            result = execute_tool(tool_call)
            messages.append(create_tool_result_message(tool_call.id, result))

    return "Max tool calling rounds reached"


def tools_to_openai_format(tools: list["Tool"]) -> list[dict]:
    """将 ``Tool`` 对象转换为 OpenAI tools 格式。

    对应 ``RESEARCH.md`` 中的 Pitfall 1：``Tool.to_openai_schema()``
    返回的是内部 function 对象，但 API 期望的格式是
    ``{"type": "function", "function": {...}}``。

    Args:
        tools: 工具发现阶段得到的 ``Tool`` 对象列表。

    Returns:
        OpenAI 格式的工具字典列表。
    """
    return [
        {"type": "function", "function": tool.to_openai_schema()}
        for tool in tools
    ]
