"""Script execution with safety checks for oh-my-tai.

Per EXEC-01: Tool execution captures both stdout and stderr in structured result.
Per EXEC-02: Execution times out after configurable period (default 120s).
Per EXEC-03: Dangerous commands (rm -rf /, sudo, etc.) are blocked before execution.
"""
import subprocess
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .tools import Tool
from .llm import ToolCall


# Default timeout for script execution (per EXEC-02)
DEFAULT_TIMEOUT = 120  # seconds


@dataclass
class ExecutionResult:
    """Result of a tool execution.

    Per EXEC-01, EXEC-04: Structured result with stdout, stderr, return_code
    for LLM consumption.
    """
    success: bool
    stdout: str
    stderr: str
    return_code: int
    timed_out: bool = False
    timeout: int = DEFAULT_TIMEOUT  # For error message formatting

    def to_llm_content(self) -> str:
        """Format result for LLM consumption.

        Returns a human-readable string describing the execution result,
        suitable for feeding back to the LLM.
        """
        if self.timed_out:
            return f"Error: Command timed out after {self.timeout}s"

        if not self.success:
            return f"Error (exit code {self.return_code}):\n{self.stderr}"

        parts = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(f"[stderr]: {self.stderr}")

        return "\n".join(parts) if parts else "Command completed successfully (no output)"


# Dangerous command patterns (per EXEC-03, RESEARCH.md Pattern 2)
DANGEROUS_PATTERNS = [
    # Destructive filesystem operations
    r'\brm\s+-rf\s+/',
    r'\brm\s+-rf\s+~',
    r'\brm\s+-rf\s+\*',
    r'\brm\s+-fr\s+/',
    # Privilege escalation
    r'\bsudo\s+',
    r'\bsu\s+',
    # System modification
    r'\bdd\s+if=',
    r'\bmkfs\b',
    r'\bformat\b',
    # Network download and execute (remote code execution)
    r'\bcurl\s+.*\|\s*bash',
    r'\bwget\s+.*\|\s*bash',
    r'\bcurl\s+.*\|\s*sh',
    r'\bwget\s+.*\|\s*sh',
    # Fork bomb
    r'\b:()\s*\{.*&\}',
    r'\b:()\s*\{.*:\s*&\}',
]


def is_dangerous_command(command: str, args: list[str]) -> bool:
    """Check if a command matches dangerous patterns.

    Per EXEC-03: Blocks dangerous commands before execution.

    Args:
        command: The script path/command to check.
        args: List of arguments that will be passed.

    Returns:
        True if the command matches dangerous patterns, False otherwise.
    """
    # Combine command and args for pattern matching
    full_command = f"{command} {' '.join(args)}"

    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, full_command, re.IGNORECASE):
            return True
    return False


def execute_tool(tool_call: ToolCall, tools: list[Tool], timeout: int = DEFAULT_TIMEOUT) -> ExecutionResult:
    """Execute a tool by name with given arguments.

    Per EXEC-01: Captures stdout and stderr in structured result.
    Per EXEC-02: Times out after configurable period.
    Per EXEC-03: Blocks dangerous commands before execution.

    Args:
        tool_call: ToolCall object with name, arguments, and id.
        tools: List of available Tool objects to lookup by name.
        timeout: Maximum execution time in seconds.

    Returns:
        ExecutionResult with success status, stdout, stderr, return_code.
    """
    # Find the tool by name
    tool = None
    for t in tools:
        if t.name == tool_call.name:
            tool = t
            break

    if tool is None:
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=f"Tool not found: {tool_call.name}",
            return_code=1,
        )

    # Build args list from ToolCall.arguments dict
    # Arguments are key-value pairs, convert to command-line args
    args = []
    for key, value in tool_call.arguments.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        elif isinstance(value, (int, float)):
            args.append(f"--{key}")
            args.append(str(value))
        else:
            args.append(f"--{key}")
            args.append(str(value))

    # Check for dangerous commands
    if is_dangerous_command(str(tool.script_path), args):
        return ExecutionResult(
            success=False,
            stdout="",
            stderr="Blocked: dangerous command pattern detected",
            return_code=1,
        )

    try:
        result = subprocess.run(
            [str(tool.script_path)] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        return ExecutionResult(
            success=result.returncode == 0,
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
            timeout=timeout,
        )

    except subprocess.TimeoutExpired:
        return ExecutionResult(
            success=False,
            stdout="",
            stderr="",
            return_code=-1,
            timed_out=True,
            timeout=timeout,
        )

    except Exception as e:
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=f"Execution error: {e}",
            return_code=1,
        )
