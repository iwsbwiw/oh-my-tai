"""CLI entry point for oh-my-tai."""
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
    """Create execute_tool callback with tools context.

    Args:
        tools: List of Tool objects from scan_tools_directory.

    Returns:
        Callback function compatible with agentic_loop.
    """
    def execute_tool_callback(tool_call: ToolCall) -> str:
        result = execute_tool(tool_call, tools)
        return result.to_llm_content()

    return execute_tool_callback


def placeholder_execute_tool(tool_call: ToolCall) -> str:
    """DEPRECATED: Use create_execute_tool_callback instead."""
    return f"[DEPRECATED] Would execute: {tool_call.name}({tool_call.arguments})"


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
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
    """Execute a single command and exit."""
    config = load_config()
    tools = scan_tools_directory(get_tools_directory())

    # Get provider config
    provider = config.get_provider()

    # Check for API key
    if not provider.api_key:
        print(f"Error: No API key configured for provider '{config.default_provider}'")
        print(f"Please add your API key to ~/.tai/config.toml")
        print(f"Reminder: Run 'chmod 600 ~/.tai/config.toml' to protect your credentials")
        return 1

    # Create LLM client
    client = LLMClient(
        base_url=provider.base_url,
        api_key=provider.api_key,
        model=provider.model,
    )

    # Build messages
    messages = [
        create_system_message(SYSTEM_PROMPT),
        create_user_message(command),
    ]

    # Convert tools to OpenAI format
    openai_tools = tools_to_openai_format(tools)

    # Create execute callback with tools context
    execute_tool_callback = create_execute_tool_callback(tools)

    # Run agentic loop
    try:
        response = agentic_loop(client, messages, openai_tools, execute_tool_callback)
        print(response)
        return 0
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1


def run_interactive_mode() -> int:
    """Run interactive REPL mode."""
    config = load_config()
    tools = scan_tools_directory(get_tools_directory())

    # Get provider config
    provider = config.get_provider()

    # Check for API key
    if not provider.api_key:
        print(f"Error: No API key configured for provider '{config.default_provider}'")
        print(f"Please add your API key to ~/.tai/config.toml")
        print(f"Reminder: Run 'chmod 600 ~/.tai/config.toml' to protect your credentials")
        return 1

    # Create LLM client
    client = LLMClient(
        base_url=provider.base_url,
        api_key=provider.api_key,
        model=provider.model,
    )

    # Convert tools to OpenAI format (done once, tools don't change during session)
    openai_tools = tools_to_openai_format(tools)

    # Create execute callback with tools context
    execute_tool_callback = create_execute_tool_callback(tools)

    # Initialize conversation history with system message
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

                # Add user message to history
                messages.append(create_user_message(user_input))

                # Run agentic loop
                try:
                    response = agentic_loop(client, messages, openai_tools, execute_tool_callback)
                    print(response)
                except RuntimeError as e:
                    print(f"Error: {e}")
                    # Don't exit on error in interactive mode, allow user to continue
                    # Remove the failed user message from history
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
    """Main entry point."""
    # Ensure config exists before doing anything
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
