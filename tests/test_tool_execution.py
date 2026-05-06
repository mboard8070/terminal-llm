"""
Tests for tool execution — core file/shell tools via maude_core.execute_tool.
"""

import os

from maude_core import execute_tool


class TestReadFile:
    def test_read_existing_file(self, tmp_file):
        result = execute_tool("read_file", {"path": str(tmp_file)})
        assert "line one" in result
        assert "line two" in result

    def test_read_with_line_range(self, tmp_file):
        result = execute_tool("read_file", {"path": str(tmp_file), "start_line": 2, "end_line": 2})
        assert "line two" in result
        assert "line one" not in result

    def test_read_nonexistent_file(self):
        result = execute_tool("read_file", {"path": "/nonexistent/file.txt"})
        assert "Error" in result


class TestWriteFile:
    def test_write_new_file(self, tmp_path):
        path = str(tmp_path / "new.txt")
        result = execute_tool("write_file", {"path": path, "content": "hello world"})
        assert "Successfully wrote" in result
        assert os.path.exists(path)
        with open(path) as f:
            assert f.read() == "hello world"

    def test_write_creates_directories(self, tmp_path):
        path = str(tmp_path / "sub" / "dir" / "file.txt")
        result = execute_tool("write_file", {"path": path, "content": "nested"})
        assert "Successfully wrote" in result
        assert os.path.exists(path)


class TestEditFile:
    def test_edit_replaces_lines(self, tmp_file):
        result = execute_tool(
            "edit_file", {"path": str(tmp_file), "start_line": 2, "end_line": 2, "new_content": "replaced line two"}
        )
        assert "dited" in result  # matches "Edited" or "Successfully edited"
        content = tmp_file.read_text()
        assert "replaced line two" in content
        assert "line one" in content


class TestListDirectory:
    def test_list_directory(self, tmp_dir_with_files):
        result = execute_tool("list_directory", {"path": str(tmp_dir_with_files)})
        assert "a.txt" in result
        assert "b.py" in result
        assert "subdir" in result

    def test_list_nonexistent(self):
        result = execute_tool("list_directory", {"path": "/nonexistent/path"})
        assert "Error" in result


class TestRunCommand:
    def test_run_echo(self):
        result = execute_tool("run_command", {"command": "echo hello"})
        assert "hello" in result

    def test_run_command_returns_output(self):
        result = execute_tool("run_command", {"command": "echo test123"})
        assert "test123" in result


class TestUnknownTool:
    def test_unknown_tool_returns_error(self):
        result = execute_tool("nonexistent_tool", {})
        assert "Error" in result or "Unknown" in result


class TestPreFlight:
    """Test that pre-flight checks work."""

    def test_read_file_always_ready(self):
        from health import check_tool_ready

        ready, reason = check_tool_ready("read_file")
        assert ready is True
        assert reason == ""

    def test_unknown_tool_assumed_ready(self):
        from health import check_tool_ready

        ready, _reason = check_tool_ready("totally_fake_tool")
        assert ready is True
