"""oh-my-tai 的脚本执行与基础安全检查。

对应 EXEC-01：以结构化结果捕获 stdout 和 stderr。
对应 EXEC-02：执行在可配置时间后超时（默认 120 秒）。
对应 EXEC-03：在执行前拦截危险命令（如 rm -rf /、sudo 等）。
"""
import subprocess
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .tools import Tool
from .llm import ToolCall


# 默认执行超时时间（对应 EXEC-02）
DEFAULT_TIMEOUT = 120  # seconds


@dataclass
class ExecutionResult:
    """工具执行结果。

    对应 EXEC-01、EXEC-04：为 LLM 提供包含 stdout、stderr、
    return_code 的结构化结果。
    """
    success: bool
    stdout: str
    stderr: str
    return_code: int
    timed_out: bool = False
    timeout: int = DEFAULT_TIMEOUT  # For error message formatting

    def to_llm_content(self) -> str:
        """将结果格式化为适合提供给 LLM 的内容。

        返回一段人类可读的执行结果描述，可直接回填给 LLM。
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


# 危险命令模式（对应 EXEC-03、RESEARCH.md Pattern 2）
DANGEROUS_PATTERNS = [
    # 破坏性文件系统操作
    r'\brm\s+-rf\s+/',
    r'\brm\s+-rf\s+~',
    r'\brm\s+-rf\s+\*',
    r'\brm\s+-fr\s+/',
    # 权限提升
    r'\bsudo\s+',
    r'\bsu\s+',
    # 系统级修改
    r'\bdd\s+if=',
    r'\bmkfs\b',
    r'\bformat\b',
    # 下载后直接执行（远程代码执行）
    r'\bcurl\s+.*\|\s*bash',
    r'\bwget\s+.*\|\s*bash',
    r'\bcurl\s+.*\|\s*sh',
    r'\bwget\s+.*\|\s*sh',
    # Fork bomb
    r'\b:()\s*\{.*&\}',
    r'\b:()\s*\{.*:\s*&\}',
]


def is_dangerous_command(command: str, args: list[str]) -> bool:
    """检查命令是否命中危险模式。

    对应 EXEC-03：在执行前阻止危险命令。

    Args:
        command: 要检查的脚本路径或命令。
        args: 即将传入的参数列表。

    Returns:
        命中危险模式返回 ``True``，否则返回 ``False``。
    """
    # 将命令和参数拼接后统一做模式匹配
    full_command = f"{command} {' '.join(args)}"

    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, full_command, re.IGNORECASE):
            return True
    return False


def execute_tool(tool_call: ToolCall, tools: list[Tool], timeout: int = DEFAULT_TIMEOUT) -> ExecutionResult:
    """按名称执行工具，并传入给定参数。

    对应 EXEC-01：在结构化结果中捕获 stdout 和 stderr。
    对应 EXEC-02：按配置超时。
    对应 EXEC-03：执行前拦截危险命令。

    Args:
        tool_call: 包含名称、参数和 ID 的 ``ToolCall`` 对象。
        tools: 可用 ``Tool`` 列表，用于按名称查找。
        timeout: 最大执行时长，单位为秒。

    Returns:
        包含 success、stdout、stderr、return_code 的 ``ExecutionResult``。
    """
    # 按名称查找对应工具
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

    # 从 ToolCall.arguments 构造命令行参数
    # 键值对会被转换为 --key value 形式
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

    # 执行前先检查危险命令模式
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
