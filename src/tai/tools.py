"""Tool discovery and annotation parsing for oh-my-tai."""
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ToolParameter:
    """Represents a single parameter for a tool."""
    name: str
    type: str = "string"  # string, number, boolean
    description: str = ""


@dataclass
class Tool:
    """Represents a discovered tool with its metadata."""
    name: str
    description: str = ""
    parameters: list[ToolParameter] = field(default_factory=list)
    script_path: Path = field(default_factory=Path)

    def to_openai_schema(self) -> dict:
        """Generate OpenAI function calling compatible schema."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description
            }
            # All declared params are required per user decision
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
    """Parse # @name, @desc, @param annotations from script.

    Returns None if:
    - File cannot be read (IOError, UnicodeDecodeError)
    - No # @name annotation found
    """
    name = None
    description = ""
    parameters = []

    # Pattern: # @key value or # @param name:type:description
    annotation_pattern = re.compile(r'#\s*@(\w+)\s*(.*)')
    param_pattern = re.compile(r'(\w+)(?::(\w+))?(?::(.*))?')

    try:
        with open(script_path, "r", encoding="utf-8") as f:
            # Only scan top of file for annotations
            for line in f:
                line = line.strip()
                if not line.startswith("#"):
                    # Stop at first non-comment line
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
                        # Strip leading colon from description when type is omitted
                        if ptype is None and pdesc and pdesc.startswith(":"):
                            pdesc = pdesc[1:].lstrip()
                        parameters.append(ToolParameter(
                            name=pname,
                            type=ptype or "string",  # Default to string
                            description=pdesc or ""
                        ))
    except (IOError, UnicodeDecodeError):
        # Silently skip files we can't read
        return None

    if not name:
        return None  # Only register scripts with @name

    return Tool(
        name=name,
        description=description,
        parameters=parameters,
        script_path=script_path
    )


def scan_tools_directory(tools_dir: Path) -> list[Tool]:
    """Scan directory for executable scripts with @name annotation.

    Recursively scans the given directory for executable files, parses
    their annotations, and returns a list of Tool objects for scripts
    that have a # @name annotation.

    Args:
        tools_dir: Path to the tools directory to scan.

    Returns:
        List of Tool objects for all executable scripts with @name annotation.
        Returns empty list if directory doesn't exist.
    """
    tools = []

    if not tools_dir.exists():
        return tools

    # Recursive scan for all files
    for script_path in tools_dir.rglob("*"):
        if not script_path.is_file():
            continue
        if not os.access(script_path, os.X_OK):
            continue

        tool = parse_script_annotations(script_path)
        if tool:  # Only if @name found
            tools.append(tool)

    return tools
