"""Tests for tools module - Tool and ToolParameter dataclasses."""
import os
import tempfile
from pathlib import Path

import pytest


class TestToolParameter:
    """Tests for ToolParameter dataclass."""

    def test_tool_parameter_defaults(self):
        """ToolParameter should have sensible defaults."""
        from tai.tools import ToolParameter

        param = ToolParameter(name="test_param")
        assert param.name == "test_param"
        assert param.type == "string"
        assert param.description == ""

    def test_tool_parameter_all_fields(self):
        """ToolParameter should accept all fields."""
        from tai.tools import ToolParameter

        param = ToolParameter(
            name="count",
            type="number",
            description="Number of items"
        )
        assert param.name == "count"
        assert param.type == "number"
        assert param.description == "Number of items"


class TestTool:
    """Tests for Tool dataclass."""

    def test_tool_defaults(self):
        """Tool should have sensible defaults."""
        from tai.tools import Tool

        tool = Tool(name="test_tool", script_path=Path("/tmp/test.sh"))
        assert tool.name == "test_tool"
        assert tool.description == ""
        assert tool.parameters == []
        assert tool.script_path == Path("/tmp/test.sh")

    def test_tool_all_fields(self):
        """Tool should accept all fields."""
        from tai.tools import Tool, ToolParameter

        params = [
            ToolParameter(name="count", type="number"),
            ToolParameter(name="verbose", type="boolean"),
        ]
        tool = Tool(
            name="my_tool",
            description="Does something useful",
            parameters=params,
            script_path=Path("/usr/local/bin/my_tool.sh")
        )
        assert tool.name == "my_tool"
        assert tool.description == "Does something useful"
        assert len(tool.parameters) == 2
        assert tool.parameters[0].name == "count"

    def test_tool_openai_schema_minimal(self):
        """Tool.to_openai_schema() should return valid schema for minimal tool."""
        from tai.tools import Tool

        tool = Tool(name="minimal_tool", script_path=Path("/tmp/min.sh"))
        schema = tool.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["name"] == "minimal_tool"
        assert schema["description"] == ""
        assert schema["parameters"]["type"] == "object"
        assert schema["parameters"]["properties"] == {}
        assert schema["parameters"]["required"] == []

    def test_tool_openai_schema_with_params(self):
        """Tool.to_openai_schema() should include all parameters."""
        from tai.tools import Tool, ToolParameter

        tool = Tool(
            name="search_files",
            description="Search for files matching a pattern",
            parameters=[
                ToolParameter(name="pattern", type="string", description="Regex pattern"),
                ToolParameter(name="case_sensitive", type="boolean", description="Case sensitive search"),
                ToolParameter(name="max_results", type="number", description="Max results to return"),
            ],
            script_path=Path("/home/user/.tai/tools/search.sh")
        )
        schema = tool.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["name"] == "search_files"
        assert schema["description"] == "Search for files matching a pattern"

        # Check parameters
        params = schema["parameters"]
        assert params["type"] == "object"

        props = params["properties"]
        assert "pattern" in props
        assert props["pattern"]["type"] == "string"
        assert props["pattern"]["description"] == "Regex pattern"

        assert "case_sensitive" in props
        assert props["case_sensitive"]["type"] == "boolean"

        assert "max_results" in props
        assert props["max_results"]["type"] == "number"

        # All declared params are required
        assert set(params["required"]) == {"pattern", "case_sensitive", "max_results"}


class TestParseScriptAnnotations:
    """Tests for parse_script_annotations function."""

    def test_parse_script_with_name_only(self, tmp_path):
        """Parse script with only @name annotation."""
        from tai.tools import parse_script_annotations

        script = tmp_path / "test.sh"
        script.write_text("#!/bin/bash\n# @name hello\n\necho hello\n")
        os.chmod(script, 0o755)

        tool = parse_script_annotations(script)
        assert tool is not None
        assert tool.name == "hello"
        assert tool.description == ""
        assert tool.parameters == []
        assert tool.script_path == script

    def test_parse_script_with_full_annotations(self, tmp_path):
        """Parse script with @name, @desc, and @param annotations."""
        from tai.tools import parse_script_annotations

        script = tmp_path / "test.sh"
        script.write_text(
            "#!/bin/bash\n"
            "# @name test_tool\n"
            "# @desc A test\n"
            "# @param count:number:How many\n"
            "# @param verbose:boolean:Verbose output\n"
            "\necho test\n"
        )
        os.chmod(script, 0o755)

        tool = parse_script_annotations(script)
        assert tool is not None
        assert tool.name == "test_tool"
        assert tool.description == "A test"
        assert len(tool.parameters) == 2
        assert tool.parameters[0].name == "count"
        assert tool.parameters[0].type == "number"
        assert tool.parameters[0].description == "How many"
        assert tool.parameters[1].name == "verbose"
        assert tool.parameters[1].type == "boolean"
        assert tool.parameters[1].description == "Verbose output"

    def test_parse_script_without_name_returns_none(self, tmp_path):
        """Script without @name annotation should return None."""
        from tai.tools import parse_script_annotations

        script = tmp_path / "test.sh"
        script.write_text("#!/bin/bash\n# @desc No name here\n\necho test\n")
        os.chmod(script, 0o755)

        tool = parse_script_annotations(script)
        assert tool is None

    def test_parse_param_defaults_type_to_string(self, tmp_path):
        """@param without type should default to string."""
        from tai.tools import parse_script_annotations

        script = tmp_path / "test.sh"
        script.write_text(
            "#!/bin/bash\n"
            "# @name test\n"
            "# @param path::Path to file\n"  # type omitted
            "\necho test\n"
        )
        os.chmod(script, 0o755)

        tool = parse_script_annotations(script)
        assert tool is not None
        assert tool.parameters[0].name == "path"
        assert tool.parameters[0].type == "string"
        assert tool.parameters[0].description == "Path to file"

    def test_parse_stops_at_first_non_comment(self, tmp_path):
        """Parser should stop at first non-comment line."""
        from tai.tools import parse_script_annotations

        script = tmp_path / "test.sh"
        script.write_text(
            "#!/bin/bash\n"
            "# @name test\n"
            "\n"  # empty line stops parsing
            "# @desc This should be ignored\n"
            "\necho test\n"
        )
        os.chmod(script, 0o755)

        tool = parse_script_annotations(script)
        assert tool is not None
        assert tool.name == "test"
        assert tool.description == ""  # not parsed

    def test_parse_handles_unicode_decode_error(self, tmp_path):
        """Parser should return None on UnicodeDecodeError."""
        from tai.tools import parse_script_annotations

        script = tmp_path / "binary"
        # Write binary data that's not valid UTF-8
        script.write_bytes(b"\x00\x01\x02\xff\xfe")

        tool = parse_script_annotations(script)
        assert tool is None

    def test_parse_handles_io_error(self):
        """Parser should return None on IOError for non-existent file."""
        from tai.tools import parse_script_annotations

        tool = parse_script_annotations(Path("/nonexistent/file.sh"))
        assert tool is None

    def test_parse_flexible_whitespace(self, tmp_path):
        """Parser should handle flexible whitespace in annotations."""
        from tai.tools import parse_script_annotations

        script = tmp_path / "test.sh"
        script.write_text(
            "#!/bin/bash\n"
            "#  @name  test_tool  \n"  # extra spaces
            "#@desc Minimal spacing\n"  # no space after #
            "\necho test\n"
        )
        os.chmod(script, 0o755)

        tool = parse_script_annotations(script)
        assert tool is not None
        assert tool.name == "test_tool"
        assert tool.description == "Minimal spacing"


class TestScanToolsDirectory:
    """Tests for scan_tools_directory function."""

    def test_scan_finds_executable_script(self, tmp_path):
        """Scan should find executable scripts with @name."""
        from tai.tools import scan_tools_directory

        script = tmp_path / "test.sh"
        script.write_text("#!/bin/bash\n# @name hello\n\necho hello\n")
        os.chmod(script, 0o755)

        tools = scan_tools_directory(tmp_path)
        assert len(tools) == 1
        assert tools[0].name == "hello"

    def test_scan_ignores_non_executable(self, tmp_path):
        """Scan should ignore non-executable files."""
        from tai.tools import scan_tools_directory

        script = tmp_path / "test.sh"
        script.write_text("#!/bin/bash\n# @name hello\n\necho hello\n")
        # No chmod - not executable

        tools = scan_tools_directory(tmp_path)
        assert len(tools) == 0

    def test_scan_ignores_scripts_without_name(self, tmp_path):
        """Scan should ignore scripts without @name annotation."""
        from tai.tools import scan_tools_directory

        script = tmp_path / "test.sh"
        script.write_text("#!/bin/bash\n# @desc No name\n\necho hello\n")
        os.chmod(script, 0o755)

        tools = scan_tools_directory(tmp_path)
        assert len(tools) == 0

    def test_scan_returns_empty_for_nonexistent_directory(self):
        """Scan should return empty list for non-existent directory."""
        from tai.tools import scan_tools_directory

        tools = scan_tools_directory(Path("/nonexistent/directory"))
        assert tools == []

    def test_scan_recursively(self, tmp_path):
        """Scan should find scripts in subdirectories."""
        from tai.tools import scan_tools_directory

        # Create nested directory structure
        subdir = tmp_path / "subdir" / "nested"
        subdir.mkdir(parents=True)

        script1 = tmp_path / "tool1.sh"
        script1.write_text("#!/bin/bash\n# @name tool1\n\necho 1\n")
        os.chmod(script1, 0o755)

        script2 = subdir / "tool2.sh"
        script2.write_text("#!/bin/bash\n# @name tool2\n\necho 2\n")
        os.chmod(script2, 0o755)

        tools = scan_tools_directory(tmp_path)
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"tool1", "tool2"}

    def test_scan_only_processes_files(self, tmp_path):
        """Scan should only process regular files, not directories."""
        from tai.tools import scan_tools_directory

        # Create a directory with executable permissions
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        os.chmod(subdir, 0o755)

        tools = scan_tools_directory(tmp_path)
        assert len(tools) == 0

    def test_scan_multiple_scripts(self, tmp_path):
        """Scan should find multiple scripts."""
        from tai.tools import scan_tools_directory

        for i in range(3):
            script = tmp_path / f"tool{i}.sh"
            script.write_text(f"#!/bin/bash\n# @name tool{i}\n\necho {i}\n")
            os.chmod(script, 0o755)

        tools = scan_tools_directory(tmp_path)
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert names == {"tool0", "tool1", "tool2"}
