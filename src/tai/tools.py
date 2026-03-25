"""oh-my-tai 的工具发现与注解解析。"""
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ToolParameter:
    """表示工具的单个参数。"""
    name: str
    type: str = "string"  # string, number, boolean
    description: str = ""


@dataclass
class Tool:
    """表示扫描得到的工具及其元数据。"""
    name: str
    description: str = ""
    parameters: list[ToolParameter] = field(default_factory=list)
    script_path: Path = field(default_factory=Path)

    def to_openai_schema(self) -> dict:
        """生成兼容 OpenAI function calling 的 schema。"""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description
            }
            # 按当前项目约定，声明过的参数都视为必填
            required.append(param.name)

        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        }


def parse_script_annotations(script_path: Path) -> Optional[Tool]:
    """解析脚本中的 ``# @name``、``# @desc``、``# @param`` 注解。

    在以下情况返回 ``None``:
    - 文件无法读取（``IOError``、``UnicodeDecodeError``）
    - 没有找到 ``# @name`` 注解
    """
    name = None
    description = ""
    parameters = []

    # 匹配形式：# @key value 或 # @param name:type:description
    annotation_pattern = re.compile(r'#\s*@(\w+)\s*(.*)')
    param_pattern = re.compile(r'(\w+)(?::(\w+))?(?::(.*))?')

    try:
        with open(script_path, "r", encoding="utf-8") as f:
            # 只扫描文件顶部的注释区域
            for line in f:
                line = line.strip()
                if not line.startswith("#"):
                    # 遇到第一行非注释内容就停止
                    break

                match = annotation_pattern.match(line)
                if not match:
                    continue

                key, value = match.groups()
                value = value.strip()

                if key == "name":
                    name = value
                elif key == "desc":
                    description = value
                elif key == "param":
                    param_match = param_pattern.match(value)
                    if param_match:
                        pname, ptype, pdesc = param_match.groups()
                        # 当省略 type 时，去掉 description 前面的冒号
                        if ptype is None and pdesc and pdesc.startswith(":"):
                            pdesc = pdesc[1:].lstrip()
                        parameters.append(ToolParameter(
                            name=pname,
                            type=ptype or "string",  # 默认类型为 string
                            description=pdesc or ""
                        ))
    except (IOError, UnicodeDecodeError):
        # 无法读取的文件直接跳过
        return None

    if not name:
        return None  # 只有带 @name 的脚本才注册为工具

    return Tool(
        name=name,
        description=description,
        parameters=parameters,
        script_path=script_path
    )


def scan_tools_directory(tools_dir: Path) -> list[Tool]:
    """扫描目录，发现带 ``@name`` 注解的可执行脚本。

    会递归扫描给定目录中的可执行文件，解析其注解，并为
    带有 ``# @name`` 注解的脚本返回 ``Tool`` 对象列表。

    Args:
        tools_dir: 要扫描的工具目录路径。

    Returns:
        所有带 ``@name`` 注解的可执行脚本对应的 ``Tool`` 列表。
        如果目录不存在，返回空列表。
    """
    tools = []

    if not tools_dir.exists():
        return tools

    # 递归扫描目录中的所有文件
    for script_path in tools_dir.rglob("*"):
        if not script_path.is_file():
            continue
        if not os.access(script_path, os.X_OK):
            continue

        tool = parse_script_annotations(script_path)
        if tool:  # Only if @name found
            tools.append(tool)

    return tools
