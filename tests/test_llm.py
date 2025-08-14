"""Tests for llm module - Data structures and message helpers."""
import json
from pathlib import Path

import pytest


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_tool_call_has_required_fields(self):
        """ToolCall should have id, name, arguments fields."""
        from tai.llm import ToolCall

        tc = ToolCall(
            id="call_123",
            name="get_weather",
            arguments={"location": "Tokyo"}
        )
        assert tc.id == "call_123"
        assert tc.name == "get_weather"
        assert tc.arguments == {"location": "Tokyo"}

    def test_tool_call_arguments_is_dict(self):
        """ToolCall arguments should be a dict."""
        from tai.llm import ToolCall

        tc = ToolCall(
            id="call_456",
            name="search",
            arguments={"query": "test", "limit": 10}
        )
        assert isinstance(tc.arguments, dict)


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_llm_response_has_content_field(self):
        """LLMResponse should have content field."""
        from tai.llm import LLMResponse

        response = LLMResponse(content="Hello, world!")
        assert response.content == "Hello, world!"

    def test_llm_response_has_tool_calls_field(self):
        """LLMResponse should have tool_calls field."""
        from tai.llm import LLMResponse, ToolCall

        tc = ToolCall(id="call_1", name="test", arguments={})
        response = LLMResponse(tool_calls=[tc])
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "test"

    def test_llm_response_has_raw_message_field(self):
        """LLMResponse should have raw_message field."""
        from tai.llm import LLMResponse

        raw = {"role": "assistant", "content": "test"}
        response = LLMResponse(raw_message=raw)
        assert response.raw_message == raw

    def test_llm_response_defaults(self):
        """LLMResponse should have sensible defaults."""
        from tai.llm import LLMResponse

        response = LLMResponse()
        assert response.content is None
        assert response.tool_calls == []
        assert response.raw_message == {}


class TestCreateToolResultMessage:
    """Tests for create_tool_result_message helper."""

    def test_create_tool_result_message_structure(self):
        """create_tool_result_message should return correct structure."""
        from tai.llm import create_tool_result_message

        msg = create_tool_result_message("call_123", "Sunny, 25C")

        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_123"
        assert msg["content"] == "Sunny, 25C"

    def test_create_tool_result_message_with_json_result(self):
        """create_tool_result_message should handle JSON results."""
        from tai.llm import create_tool_result_message

        result = json.dumps({"temp": 25, "condition": "sunny"})
        msg = create_tool_result_message("call_456", result)

        assert msg["role"] == "tool"
        assert msg["content"] == result


class TestCreateUserMessage:
    """Tests for create_user_message helper."""

    def test_create_user_message_structure(self):
        """create_user_message should return correct structure."""
        from tai.llm import create_user_message

        msg = create_user_message("What's the weather?")

        assert msg["role"] == "user"
        assert msg["content"] == "What's the weather?"


class TestCreateSystemMessage:
    """Tests for create_system_message helper."""

    def test_create_system_message_structure(self):
        """create_system_message should return correct structure."""
        from tai.llm import create_system_message

        msg = create_system_message("You are a helpful assistant.")

        assert msg["role"] == "system"
        assert msg["content"] == "You are a helpful assistant."


class TestCreateAssistantMessage:
    """Tests for create_assistant_message helper."""

    def test_create_assistant_message_with_content_only(self):
        """create_assistant_message should work with content only."""
        from tai.llm import create_assistant_message

        msg = create_assistant_message("Hello!")

        assert msg["role"] == "assistant"
        assert msg["content"] == "Hello!"
        assert "tool_calls" not in msg

    def test_create_assistant_message_with_tool_calls(self):
        """create_assistant_message should include tool_calls when provided."""
        from tai.llm import create_assistant_message, ToolCall

        tool_calls = [
            ToolCall(id="call_1", name="get_weather", arguments={"city": "Tokyo"})
        ]
        msg = create_assistant_message("Let me check.", tool_calls)

        assert msg["role"] == "assistant"
        assert msg["content"] == "Let me check."
        assert "tool_calls" in msg
        assert len(msg["tool_calls"]) == 1

        tc = msg["tool_calls"][0]
        assert tc["id"] == "call_1"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        # Arguments should be JSON string
        assert tc["function"]["arguments"] == '{"city": "Tokyo"}'

    def test_create_assistant_message_with_none_content(self):
        """create_assistant_message should handle None content."""
        from tai.llm import create_assistant_message, ToolCall

        tool_calls = [ToolCall(id="call_1", name="test", arguments={})]
        msg = create_assistant_message(None, tool_calls)

        assert msg["role"] == "assistant"
        assert msg["content"] is None
        assert "tool_calls" in msg

    def test_create_assistant_message_empty_tool_calls(self):
        """create_assistant_message should not add tool_calls for empty list."""
        from tai.llm import create_assistant_message

        msg = create_assistant_message("Hello!", [])

        assert msg["role"] == "assistant"
        assert msg["content"] == "Hello!"
        assert "tool_calls" not in msg


class TestTruncateHistory:
    """Tests for truncate_history function."""

    def test_truncate_history_no_change_when_under_max(self):
        """truncate_history should return messages unchanged when len <= max."""
        from tai.llm import truncate_history

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = truncate_history(messages, 20)
        assert result == messages
        assert len(result) == 3

    def test_truncate_history_truncates_when_over_max(self):
        """truncate_history should truncate when messages exceed max."""
        from tai.llm import truncate_history

        messages = [
            {"role": "system", "content": "System"},
        ] + [
            {"role": "user", "content": f"Message {i}"}
            for i in range(15)
        ]
        # Total: 16 messages, max 10
        result = truncate_history(messages, 10)
        assert len(result) == 10

    def test_truncate_history_preserves_system_messages(self):
        """truncate_history should preserve all system messages."""
        from tai.llm import truncate_history

        messages = [
            {"role": "system", "content": "System 1"},
            {"role": "system", "content": "System 2"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = truncate_history(messages, 3)
        # 2 system + 2 other = 4, max 3 -> keep 2 system + 1 other
        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "system"
        assert result[2]["role"] == "assistant"  # Most recent non-system

    def test_truncate_history_keeps_most_recent_non_system(self):
        """truncate_history should keep most recent non-system messages."""
        from tai.llm import truncate_history

        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Old"},
            {"role": "assistant", "content": "Old reply"},
            {"role": "user", "content": "Recent"},
            {"role": "assistant", "content": "Recent reply"},
        ]
        result = truncate_history(messages, 3)
        # 1 system + 4 other = 5, max 3 -> keep 1 system + 2 other
        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[1]["content"] == "Recent"
        assert result[2]["content"] == "Recent reply"

    def test_truncate_history_only_system_messages_exceeds_max(self):
        """truncate_history should return all system messages even if exceeds max."""
        from tai.llm import truncate_history

        messages = [
            {"role": "system", "content": f"System {i}"}
            for i in range(5)
        ]
        result = truncate_history(messages, 3)
        # Edge case: all system messages, exceed max -> return all
        assert len(result) == 5
        assert all(m["role"] == "system" for m in result)

    def test_default_max_history_constant_exists(self):
        """DEFAULT_MAX_HISTORY constant should exist and equal 20."""
        from tai.llm import DEFAULT_MAX_HISTORY

        assert DEFAULT_MAX_HISTORY == 20

    def test_truncate_history_default_max(self):
        """truncate_history should use DEFAULT_MAX_HISTORY when max not specified."""
        from tai.llm import truncate_history, DEFAULT_MAX_HISTORY

        messages = [
            {"role": "system", "content": "System"},
        ] + [
            {"role": "user", "content": f"Message {i}"}
            for i in range(25)
        ]
        result = truncate_history(messages)
        assert len(result) == DEFAULT_MAX_HISTORY


class TestLLMClient:
    """Tests for LLMClient class."""

    def test_llm_client_init_stores_params(self):
        """LLMClient.__init__ should store all params correctly."""
        from tai.llm import LLMClient

        client = LLMClient(
            base_url="https://api.example.com/v1",
            api_key="sk-test-key",
            model="gpt-4o-mini"
        )
        assert client.base_url == "https://api.example.com/v1"
        assert client.api_key == "sk-test-key"
        assert client.model == "gpt-4o-mini"

    def test_llm_client_init_timeout_default(self):
        """LLMClient.__init__ should accept optional timeout param (default 60)."""
        from tai.llm import LLMClient

        client = LLMClient(
            base_url="https://api.example.com/v1",
            api_key="sk-test-key",
            model="gpt-4o-mini"
        )
        assert client.timeout == 60

        client_with_timeout = LLMClient(
            base_url="https://api.example.com/v1",
            api_key="sk-test-key",
            model="gpt-4o-mini",
            timeout=30
        )
        assert client_with_timeout.timeout == 30

    def test_llm_client_call_makes_post_request(self):
        """LLMClient.call() should make POST request to {base_url}/chat/completions."""
        from unittest.mock import MagicMock, patch
        from tai.llm import LLMClient

        client = LLMClient(
            base_url="https://api.example.com/v1",
            api_key="sk-test-key",
            model="gpt-4o-mini"
        )

        mock_response = MagicMock()
        mock_response.read.return_value = b'{"choices": [{"message": {"content": "Hello"}}]}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            with patch("urllib.request.Request") as mock_request:
                client.call([{"role": "user", "content": "Hi"}], [])

                # Verify Request was called with POST method
                call_args = mock_request.call_args
                assert call_args[1]["method"] == "POST"
                assert call_args[0][0] == "https://api.example.com/v1/chat/completions"

    def test_llm_client_call_sends_auth_header(self):
        """LLMClient.call() should send Authorization: Bearer {api_key} header."""
        from unittest.mock import MagicMock, patch
        from tai.llm import LLMClient

        client = LLMClient(
            base_url="https://api.example.com/v1",
            api_key="sk-test-key",
            model="gpt-4o-mini"
        )

        mock_response = MagicMock()
        mock_response.read.return_value = b'{"choices": [{"message": {"content": "Hello"}}]}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            with patch("urllib.request.Request") as mock_request:
                client.call([{"role": "user", "content": "Hi"}], [])

                call_args = mock_request.call_args
                headers = call_args[1]["headers"]
                assert headers["Authorization"] == "Bearer sk-test-key"

    def test_llm_client_call_sends_content_type_header(self):
        """LLMClient.call() should send Content-Type: application/json header."""
        from unittest.mock import MagicMock, patch
        from tai.llm import LLMClient

        client = LLMClient(
            base_url="https://api.example.com/v1",
            api_key="sk-test-key",
            model="gpt-4o-mini"
        )

        mock_response = MagicMock()
        mock_response.read.return_value = b'{"choices": [{"message": {"content": "Hello"}}]}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            with patch("urllib.request.Request") as mock_request:
                client.call([{"role": "user", "content": "Hi"}], [])

                call_args = mock_request.call_args
                headers = call_args[1]["headers"]
                assert headers["Content-Type"] == "application/json"

    def test_llm_client_call_payload_includes_model_messages_tools(self):
        """LLMClient.call() payload should include model, messages, tools, tool_choice."""
        import json
        from unittest.mock import MagicMock, patch
        from tai.llm import LLMClient

        client = LLMClient(
            base_url="https://api.example.com/v1",
            api_key="sk-test-key",
            model="gpt-4o-mini"
        )

        mock_response = MagicMock()
        mock_response.read.return_value = b'{"choices": [{"message": {"content": "Hello"}}]}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            with patch("urllib.request.Request") as mock_request:
                messages = [{"role": "user", "content": "Hi"}]
                tools = [{"type": "function", "function": {"name": "test"}}]
                client.call(messages, tools)

                call_args = mock_request.call_args
                payload = json.loads(call_args[1]["data"])
                assert payload["model"] == "gpt-4o-mini"
                assert payload["messages"] == messages
                assert payload["tools"] == tools
                assert payload["tool_choice"] == "auto"

    def test_llm_client_call_returns_llm_response(self):
        """LLMClient.call() should return LLMResponse with content and tool_calls parsed."""
        from unittest.mock import MagicMock, patch
        from tai.llm import LLMClient, LLMResponse, ToolCall

        client = LLMClient(
            base_url="https://api.example.com/v1",
            api_key="sk-test-key",
            model="gpt-4o-mini"
        )

        response_body = b'''{
            "choices": [{
                "message": {
                    "content": "Let me help you.",
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\\"city\\": \\"Tokyo\\"}"
                        }
                    }]
                }
            }]
        }'''

        mock_response = MagicMock()
        mock_response.read.return_value = response_body
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = client.call([{"role": "user", "content": "Hi"}], [])

        assert isinstance(result, LLMResponse)
        assert result.content == "Let me help you."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_123"
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments == {"city": "Tokyo"}

    def test_llm_client_call_http_error(self):
        """LLMClient.call() with HTTP 4xx/5xx should raise RuntimeError with status and body."""
        import io
        import urllib.error
        from unittest.mock import patch
        from tai.llm import LLMClient

        client = LLMClient(
            base_url="https://api.example.com/v1",
            api_key="sk-test-key",
            model="gpt-4o-mini"
        )

        # Create a file-like object for the error body
        error_body = io.BytesIO(b'{"error": "Invalid API key"}')
        mock_http_error = urllib.error.HTTPError(
            url="https://api.example.com/v1/chat/completions",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=error_body
        )

        with patch("urllib.request.urlopen", side_effect=mock_http_error):
            with pytest.raises(RuntimeError) as exc_info:
                client.call([{"role": "user", "content": "Hi"}], [])

        assert "LLM API error 401" in str(exc_info.value)
        assert "Invalid API key" in str(exc_info.value)

    def test_llm_client_call_network_error(self):
        """LLMClient.call() with network error should raise RuntimeError with reason."""
        import urllib.error
        from unittest.mock import patch
        from tai.llm import LLMClient

        client = LLMClient(
            base_url="https://api.example.com/v1",
            api_key="sk-test-key",
            model="gpt-4o-mini"
        )

        mock_url_error = urllib.error.URLError("Connection refused")

        with patch("urllib.request.urlopen", side_effect=mock_url_error):
            with pytest.raises(RuntimeError) as exc_info:
                client.call([{"role": "user", "content": "Hi"}], [])

        assert "Network error" in str(exc_info.value)
        assert "Connection refused" in str(exc_info.value)

    def test_llm_client_parse_tool_calls(self):
        """ToolCall parsing should extract id, function.name, parse function.arguments as JSON."""
        from unittest.mock import MagicMock, patch
        from tai.llm import LLMClient, ToolCall

        client = LLMClient(
            base_url="https://api.example.com/v1",
            api_key="sk-test-key",
            model="gpt-4o-mini"
        )

        response_data = {
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_001",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": '{"query": "test", "limit": 10}'
                            }
                        },
                        {
                            "id": "call_002",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Tokyo", "units": "celsius"}'
                            }
                        }
                    ]
                }
            }]
        }

        result = client._parse_response(response_data)

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].id == "call_001"
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].arguments == {"query": "test", "limit": 10}
        assert result.tool_calls[1].id == "call_002"
        assert result.tool_calls[1].name == "get_weather"
        assert result.tool_calls[1].arguments == {"city": "Tokyo", "units": "celsius"}


class TestAgenticLoop:
    """Tests for agentic_loop function."""

    def test_max_tool_rounds_constant_exists(self):
        """MAX_TOOL_ROUNDS constant should exist and equal 5."""
        from tai.llm import MAX_TOOL_ROUNDS

        assert MAX_TOOL_ROUNDS == 5

    def test_agentic_loop_calls_llm_client(self):
        """agentic_loop() should call LLMClient.call() with messages and tools."""
        from unittest.mock import MagicMock
        from tai.llm import LLMClient, LLMResponse, agentic_loop

        client = LLMClient(
            base_url="https://api.example.com/v1",
            api_key="sk-test-key",
            model="gpt-4o-mini"
        )
        # Mock the call method directly
        client.call = MagicMock(return_value=LLMResponse(content="Done!", tool_calls=[], raw_message={"role": "assistant", "content": "Done!"}))

        messages = [{"role": "user", "content": "Hi"}]
        tools = []
        execute_tool = MagicMock(return_value="tool result")

        result = agentic_loop(client, messages, tools, execute_tool)

        client.call.assert_called_once_with(messages, tools)
        assert result == "Done!"

    def test_agentic_loop_returns_content_when_no_tool_calls(self):
        """agentic_loop() should return LLM content immediately when no tool_calls."""
        from unittest.mock import MagicMock
        from tai.llm import LLMClient, LLMResponse, agentic_loop

        client = LLMClient(
            base_url="https://api.example.com/v1",
            api_key="sk-test-key",
            model="gpt-4o-mini"
        )
        client.call = MagicMock(return_value=LLMResponse(
            content="Final answer",
            tool_calls=[],
            raw_message={"role": "assistant", "content": "Final answer"}
        ))

        messages = [{"role": "user", "content": "Hi"}]
        execute_tool = MagicMock(return_value="result")

        result = agentic_loop(client, messages, [], execute_tool)

        assert result == "Final answer"

    def test_agentic_loop_continues_with_tool_calls(self):
        """agentic_loop() should continue loop when tool_calls present."""
        from unittest.mock import MagicMock
        from tai.llm import LLMClient, LLMResponse, ToolCall, agentic_loop

        client = LLMClient(
            base_url="https://api.example.com/v1",
            api_key="sk-test-key",
            model="gpt-4o-mini"
        )

        # First call returns tool_call, second call returns final answer
        call_count = [0]
        def mock_call(messages, tools):
            call_count[0] += 1
            if call_count[0] == 1:
                return LLMResponse(
                    content=None,
                    tool_calls=[ToolCall(id="call_1", name="test", arguments={})],
                    raw_message={
                        "role": "assistant",
                        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "test", "arguments": "{}"}}
                    ]}
                )
            else:
                return LLMResponse(
                    content="Done after tool",
                    tool_calls=[],
                    raw_message={"role": "assistant", "content": "Done after tool"}
                )

        client.call = MagicMock(side_effect=mock_call)

        messages = [{"role": "user", "content": "Hi"}]
        execute_tool = MagicMock(return_value="tool result")

        result = agentic_loop(client, messages, [], execute_tool)

        assert call_count[0] == 2  # Two LLM calls made
        assert result == "Done after tool"

    def test_agentic_loop_appends_assistant_message(self):
        """agentic_loop() should append assistant message with tool_calls to history."""
        from unittest.mock import MagicMock
        from tai.llm import LLMClient, LLMResponse, ToolCall, agentic_loop

        client = LLMClient(
            base_url="https://api.example.com/v1",
            api_key="sk-test-key",
            model="gpt-4o-mini"
        )

        call_count = [0]
        def mock_call(messages, tools):
            call_count[0] += 1
            if call_count[0] == 1:
                return LLMResponse(
                    content=None,
                    tool_calls=[ToolCall(id="call_1", name="test", arguments={"x": 1})],
                    raw_message={
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "test", "arguments": '{"x": 1}'}}]
                    }
                )
            return LLMResponse(content="Done", tool_calls=[], raw_message={"role": "assistant", "content": "Done"})

        client.call = MagicMock(side_effect=mock_call)

        messages = [{"role": "user", "content": "Hi"}]
        execute_tool = MagicMock(return_value="result")

        agentic_loop(client, messages, [], execute_tool)

        # Check that assistant message was appended
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        assert len(assistant_msgs) >= 1
        # First assistant message should have tool_calls
        assert "tool_calls" in assistant_msgs[0]

    def test_agentic_loop_appends_tool_result_messages(self):
        """agentic_loop() should append tool result messages after execution."""
        from unittest.mock import MagicMock
        from tai.llm import LLMClient, LLMResponse, ToolCall, agentic_loop

        client = LLMClient(
            base_url="https://api.example.com/v1",
            api_key="sk-test-key",
            model="gpt-4o-mini"
        )

        call_count = [0]
        def mock_call(messages, tools):
            call_count[0] += 1
            if call_count[0] == 1:
                return LLMResponse(
                    content=None,
                    tool_calls=[ToolCall(id="call_123", name="test_tool", arguments={"arg": "value"})],
                    raw_message={
                        "role": "assistant",
                        "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "test_tool", "arguments": '{"arg": "value"}'}}]
                    }
                )
            return LLMResponse(content="Done", tool_calls=[], raw_message={"role": "assistant", "content": "Done"})

        client.call = MagicMock(side_effect=mock_call)

        messages = [{"role": "user", "content": "Hi"}]
        execute_tool = MagicMock(return_value="tool executed successfully")

        agentic_loop(client, messages, [], execute_tool)

        # Check that tool result message was appended
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "call_123"
        assert tool_msgs[0]["content"] == "tool executed successfully"

    def test_agentic_loop_stops_after_max_rounds(self):
        """agentic_loop() should stop after MAX_TOOL_ROUNDS (5) and return message."""
        from unittest.mock import MagicMock
        from tai.llm import LLMClient, LLMResponse, ToolCall, agentic_loop, MAX_TOOL_ROUNDS

        client = LLMClient(
            base_url="https://api.example.com/v1",
            api_key="sk-test-key",
            model="gpt-4o-mini"
        )

        # Always return tool calls - simulates infinite loop
        def mock_call(messages, tools):
            return LLMResponse(
                content=None,
                tool_calls=[ToolCall(id=f"call_{len(messages)}", name="test", arguments={})],
                raw_message={
                    "role": "assistant",
                    "tool_calls": [{"id": f"call_{len(messages)}", "type": "function", "function": {"name": "test", "arguments": "{}"}}]
                }
            )

        client.call = MagicMock(side_effect=mock_call)

        messages = [{"role": "user", "content": "Hi"}]
        execute_tool = MagicMock(return_value="result")

        result = agentic_loop(client, messages, [], execute_tool)

        assert result == "Max tool calling rounds reached"
        # Should have called LLM exactly MAX_TOOL_ROUNDS times
        assert client.call.call_count == MAX_TOOL_ROUNDS

    def test_agentic_loop_accepts_execute_tool_callback(self):
        """agentic_loop() should accept execute_tool callback function."""
        from unittest.mock import MagicMock
        from tai.llm import LLMClient, LLMResponse, ToolCall, agentic_loop

        client = LLMClient(
            base_url="https://api.example.com/v1",
            api_key="sk-test-key",
            model="gpt-4o-mini"
        )

        call_count = [0]
        def mock_call(messages, tools):
            call_count[0] += 1
            if call_count[0] == 1:
                return LLMResponse(
                    content=None,
                    tool_calls=[ToolCall(id="call_1", name="my_tool", arguments={"x": 42})],
                    raw_message={
                        "role": "assistant",
                        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "my_tool", "arguments": '{"x": 42}'}}]
                    }
                )
            return LLMResponse(content="Done", tool_calls=[], raw_message={"role": "assistant", "content": "Done"})

        client.call = MagicMock(side_effect=mock_call)

        messages = [{"role": "user", "content": "Hi"}]
        execute_tool = MagicMock(return_value="callback result")

        agentic_loop(client, messages, [], execute_tool)

        # Verify callback was called with ToolCall
        execute_tool.assert_called_once()
        call_arg = execute_tool.call_args[0][0]
        assert call_arg.id == "call_1"
        assert call_arg.name == "my_tool"
        assert call_arg.arguments == {"x": 42}

    def test_agentic_loop_applies_truncate_history(self):
        """agentic_loop() should apply truncate_history before each LLM call."""
        from unittest.mock import MagicMock, patch
        from tai.llm import LLMClient, LLMResponse, agentic_loop

        client = LLMClient(
            base_url="https://api.example.com/v1",
            api_key="sk-test-key",
            model="gpt-4o-mini"
        )
        client.call = MagicMock(return_value=LLMResponse(
            content="Done",
            tool_calls=[],
            raw_message={"role": "assistant", "content": "Done"}
        ))

        messages = [{"role": "user", "content": "Hi"}]
        execute_tool = MagicMock(return_value="result")

        with patch("tai.llm.truncate_history") as mock_truncate:
            # Make truncate_history return messages unchanged
            mock_truncate.side_effect = lambda msgs: msgs

            agentic_loop(client, messages, [], execute_tool)

            # truncate_history should have been called
            mock_truncate.assert_called()


class TestToolsToOpenAIFormat:
    """Tests for tools_to_openai_format helper."""

    def test_empty_list_returns_empty_list(self):
        """tools_to_openai_format([]) should return []."""
        from tai.llm import tools_to_openai_format

        result = tools_to_openai_format([])
        assert result == []

    def test_converts_single_tool(self):
        """tools_to_openai_format([tool1]) should return list with 1 item."""
        from tai.llm import tools_to_openai_format
        from tai.tools import Tool, ToolParameter

        tool = Tool(
            name="get_weather",
            description="Get weather info",
            parameters=[ToolParameter(name="city", type="string", description="City name")],
            script_path=Path("/tmp/test.sh")
        )

        result = tools_to_openai_format([tool])

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"

    def test_converts_multiple_tools(self):
        """tools_to_openai_format([tool1, tool2]) should return list with 2 items."""
        from tai.llm import tools_to_openai_format
        from tai.tools import Tool, ToolParameter

        tools = [
            Tool(
                name="get_weather",
                description="Get weather",
                parameters=[],
                script_path=Path("/tmp/weather.sh")
            ),
            Tool(
                name="search",
                description="Search the web",
                parameters=[ToolParameter(name="query", type="string", description="Search query")],
                script_path=Path("/tmp/search.sh")
            )
        ]

        result = tools_to_openai_format(tools)

        assert len(result) == 2
        assert result[0]["type"] == "function"
        assert result[1]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[1]["function"]["name"] == "search"

    def test_item_has_correct_structure(self):
        """Each item should have structure {"type": "function", "function": {...}}."""
        from tai.llm import tools_to_openai_format
        from tai.tools import Tool, ToolParameter

        tool = Tool(
            name="test_tool",
            description="A test tool",
            parameters=[ToolParameter(name="arg1", type="string", description="First arg")],
            script_path=Path("/tmp/test.sh")
        )

        result = tools_to_openai_format([tool])

        assert "type" in result[0]
        assert "function" in result[0]
        assert result[0]["type"] == "function"
        assert isinstance(result[0]["function"], dict)

    def test_function_name_matches_tool_name(self):
        """function.name should match tool.name."""
        from tai.llm import tools_to_openai_format
        from tai.tools import Tool

        tool = Tool(
            name="my_custom_tool",
            description="Custom tool",
            parameters=[],
            script_path=Path("/tmp/custom.sh")
        )

        result = tools_to_openai_format([tool])

        assert result[0]["function"]["name"] == "my_custom_tool"

    def test_function_parameters_match_schema(self):
        """function.parameters should match tool.to_openai_schema()["parameters"]."""
        from tai.llm import tools_to_openai_format
        from tai.tools import Tool, ToolParameter

        tool = Tool(
            name="calculator",
            description="Do math",
            parameters=[
                ToolParameter(name="x", type="number", description="First number"),
                ToolParameter(name="y", type="number", description="Second number")
            ],
            script_path=Path("/tmp/calc.sh")
        )

        result = tools_to_openai_format([tool])

        expected_schema = tool.to_openai_schema()
        assert result[0]["function"]["parameters"] == expected_schema["parameters"]
