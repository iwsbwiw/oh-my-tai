"""oh-my-tai 的 CLI 入口。"""
import argparse
import sys
from typing import Callable

from .config import load_config, ensure_config_exists, get_tools_directory
from .tools import scan_tools_directory, Tool
from .llm import (
    LLMClient,
    agentic_loop,
    tools_to_openai_format,
    create_user_message,
    create_system_message,
    ToolCall,
)
from .executor import execute_tool, ExecutionResult

EXIT_COMMANDS = {"exit", "quit", "q"}

SYSTEM_PROMPT = """You are a helpful assistant that executes user commands by calling tools.
Analyze the user's request and call the appropriate tool(s) to fulfill it.
Be concise in your responses."""


def create_execute_tool_callback(tools: list[Tool]) -> Callable[[ToolCall], str]:
    """基于工具列表创建执行回调。

    Args:
        tools: ``scan_tools_directory`` 返回的 ``Tool`` 对象列表。

    Returns:
        与 ``agentic_loop`` 兼容的回调函数。
    """
    def execute_tool_callback(tool_call: ToolCall) -> str:
        result = execute_tool(tool_call, tools)
        return result.to_llm_content()

    return execute_tool_callback


def placeholder_execute_tool(tool_call: ToolCall) -> str:
    """已废弃：请改用 ``create_execute_tool_callback``。"""
    return f"[DEPRECATED] Would execute: {tool_call.name}({tool_call.arguments})"


def create_parser() -> argparse.ArgumentParser:
    """创建并配置命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="tai",
        description="LLM-powered script launcher - fast natural language tool execution",
        epilog="Examples:\n  tai 'list all python files'\n  tai -i",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command",
        nargs="?",
        help="Natural language command to execute",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Start interactive multi-turn conversation mode",
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )
    return parser


def run_single_shot(command: str) -> int:
    """执行单次命令并退出。"""
    config = load_config()
    tools = scan_tools_directory(get_tools_directory())

    # 读取 provider 配置
    provider = config.get_provider()

    # 检查 API key
    if not provider.api_key:
        print(f"Error: No API key configured for provider '{config.default_provider}'")
        print(f"Please add your API key to ~/.tai/config.toml")
        print(f"Reminder: Run 'chmod 600 ~/.tai/config.toml' to protect your credentials")
        return 1

    # 创建 LLM 客户端
    client = LLMClient(
        base_url=provider.base_url,
        api_key=provider.api_key,
        model=provider.model,
    )

    # 构造消息列表
    messages = [
        create_system_message(SYSTEM_PROMPT),
        create_user_message(command),
    ]

    # 转换为 OpenAI 兼容的 tools 格式
    openai_tools = tools_to_openai_format(tools)

    # 基于工具上下文创建执行回调
    execute_tool_callback = create_execute_tool_callback(tools)

    # 运行 agent loop
    try:
        response = agentic_loop(client, messages, openai_tools, execute_tool_callback)
        print(response)
        return 0
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1


def run_interactive_mode() -> int:
    """运行交互式 REPL 模式。"""
    config = load_config()
    tools = scan_tools_directory(get_tools_directory())

    # 读取 provider 配置
    provider = config.get_provider()

    # 检查 API key
    if not provider.api_key:
        print(f"Error: No API key configured for provider '{config.default_provider}'")
        print(f"Please add your API key to ~/.tai/config.toml")
        print(f"Reminder: Run 'chmod 600 ~/.tai/config.toml' to protect your credentials")
        return 1

    # 创建 LLM 客户端
    client = LLMClient(
        base_url=provider.base_url,
        api_key=provider.api_key,
        model=provider.model,
    )

    # 工具列表在会话期间不变，因此只需转换一次
    openai_tools = tools_to_openai_format(tools)

    # 基于工具上下文创建执行回调
    execute_tool_callback = create_execute_tool_callback(tools)

    # 使用 system message 初始化对话历史
    messages = [create_system_message(SYSTEM_PROMPT)]

    print(f"tai interactive mode (provider: {config.default_provider})")
    print(f"[Discovered {len(tools)} tools]")
    print("Type 'exit' or 'q' to quit.\n")

    try:
        while True:
            try:
                user_input = input("tai> ").strip()
                if user_input.lower() in EXIT_COMMANDS:
                    break
                if not user_input:
                    continue

                # 将用户输入加入历史
                messages.append(create_user_message(user_input))

                # 运行 agent loop
                try:
                    response = agentic_loop(client, messages, openai_tools, execute_tool_callback)
                    print(response)
                except RuntimeError as e:
                    print(f"Error: {e}")
                    # 交互模式下出错时不退出，允许继续会话
                    # 同时移除本次失败的用户消息，避免污染历史
                    if messages and messages[-1].get("role") == "user":
                        messages.pop()

            except EOFError:  # Ctrl+D
                print()
                break
    except KeyboardInterrupt:  # Ctrl+C
        print()

    print("Goodbye!")
    return 0


def main() -> int:
    """主入口函数。"""
    # 在执行任何逻辑前先确保配置存在
    ensure_config_exists()

    parser = create_parser()
    args = parser.parse_args()

    if args.interactive:
        return run_interactive_mode()
    elif args.command:
        return run_single_shot(args.command)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
