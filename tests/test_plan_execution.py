"""
Tests for execute_plan tool — multi-stage parallel tool execution.
"""

import json
import time
from unittest.mock import patch, MagicMock

import pytest

from maude_core.tools_plan import (
    PARALLEL_SAFE,
    execute_plan,
    _resolve_refs,
    _resolve_args,
)


class TestParallelSafeSet:
    """Verify the shared PARALLEL_SAFE set is coherent."""

    def test_is_frozenset(self):
        assert isinstance(PARALLEL_SAFE, frozenset)

    def test_contains_read_only_tools(self):
        for name in ("read_file", "list_directory", "search_file",
                      "web_search", "gmail_list", "recall_memory"):
            assert name in PARALLEL_SAFE

    def test_excludes_mutating_tools(self):
        for name in ("write_file", "edit_file", "run_command",
                      "change_directory", "gmail_send", "save_memory"):
            assert name not in PARALLEL_SAFE

    def test_nonempty(self):
        assert len(PARALLEL_SAFE) > 30


class TestRefResolution:
    """Test $N.M reference syntax."""

    def test_simple_ref(self):
        results = {0: ["hello", "world"]}
        assert _resolve_refs("$0.0", results) == "hello"
        assert _resolve_refs("$0.1", results) == "world"

    def test_ref_in_sentence(self):
        results = {0: ["/path/to/file.py"]}
        val = _resolve_refs("read the file at $0.0 now", results)
        assert val == "read the file at /path/to/file.py now"

    def test_multi_stage_refs(self):
        results = {0: ["first"], 1: ["second"]}
        assert _resolve_refs("$0.0 and $1.0", results) == "first and second"

    def test_missing_ref_left_unchanged(self):
        results = {0: ["only one"]}
        assert _resolve_refs("$0.5", results) == "$0.5"
        assert _resolve_refs("$3.0", results) == "$3.0"

    def test_no_refs_passthrough(self):
        results = {0: ["x"]}
        assert _resolve_refs("no refs here", results) == "no refs here"

    def test_non_string_passthrough(self):
        assert _resolve_refs(42, {}) == 42
        assert _resolve_refs(None, {}) is None

    def test_resolve_args_dict(self):
        results = {0: ["found.py"]}
        args = {"path": "$0.0", "pattern": "TODO"}
        resolved = _resolve_args(args, results)
        assert resolved["path"] == "found.py"
        assert resolved["pattern"] == "TODO"

    def test_resolve_args_list_values(self):
        results = {0: ["a.py", "b.py"]}
        args = {"files": ["$0.0", "$0.1", "literal"]}
        resolved = _resolve_args(args, results)
        assert resolved["files"] == ["a.py", "b.py", "literal"]


class TestExecutePlan:
    """Test the execute_plan function end-to-end."""

    @patch("maude_core.execute.execute_tool")
    def test_single_stage_single_tool(self, mock_exec):
        mock_exec.return_value = "file contents here"
        result = execute_plan([
            [{"name": "read_file", "args": {"path": "test.py"}}]
        ])
        assert "1 stages" in result
        assert "1 tools" in result
        assert "file contents here" in result
        mock_exec.assert_called_once_with("read_file", {"path": "test.py"})

    @patch("maude_core.execute.execute_tool")
    def test_single_stage_parallel(self, mock_exec):
        """Multiple parallel-safe tools in one stage should all execute."""
        mock_exec.side_effect = lambda name, args: f"result:{args.get('path', '')}"
        result = execute_plan([
            [
                {"name": "read_file", "args": {"path": "a.py"}},
                {"name": "read_file", "args": {"path": "b.py"}},
                {"name": "list_directory", "args": {"path": "/tmp"}},
            ]
        ])
        assert mock_exec.call_count == 3
        assert "result:a.py" in result
        assert "result:b.py" in result
        assert "3 tools" in result

    @patch("maude_core.execute.execute_tool")
    def test_two_stages_sequential(self, mock_exec):
        """Stage 1 should run after stage 0."""
        call_order = []
        def side_effect(name, args):
            call_order.append(name)
            return f"done:{name}"
        mock_exec.side_effect = side_effect

        result = execute_plan([
            [{"name": "read_file", "args": {"path": "x"}}],
            [{"name": "run_command", "args": {"command": "echo hi"}}],
        ])
        assert call_order == ["read_file", "run_command"]
        assert "2 stages" in result
        assert "done:read_file" in result
        assert "done:run_command" in result

    @patch("maude_core.execute.execute_tool")
    def test_ref_resolution_across_stages(self, mock_exec):
        """$0.0 in stage 1 should resolve to stage 0's first result."""
        def side_effect(name, args):
            if name == "search_directory":
                return "src/main.py:10: def main():"
            elif name == "read_file":
                return f"reading {args['path']}"
            return ""
        mock_exec.side_effect = side_effect

        result = execute_plan([
            [{"name": "search_directory", "args": {"directory": "src", "pattern": "main"}}],
            [{"name": "read_file", "args": {"path": "$0.0"}}],
        ])
        # Stage 1 should have resolved $0.0 to the search result
        calls = mock_exec.call_args_list
        stage1_args = calls[1][0][1]  # second call, positional arg 1
        assert "src/main.py:10: def main():" in stage1_args["path"]

    @patch("maude_core.execute.execute_tool")
    def test_mutating_tools_run_sequentially(self, mock_exec):
        """write_file and run_command should not run in parallel."""
        call_times = []
        def side_effect(name, args):
            call_times.append((name, time.time()))
            time.sleep(0.01)  # Small delay to verify ordering
            return f"ok:{name}"
        mock_exec.side_effect = side_effect

        execute_plan([
            [
                {"name": "write_file", "args": {"path": "a", "content": "x"}},
                {"name": "run_command", "args": {"command": "ls"}},
            ]
        ])
        # Both are mutating, so sequential — second starts after first
        assert call_times[0][0] == "write_file"
        assert call_times[1][0] == "run_command"
        assert call_times[1][1] >= call_times[0][1]

    @patch("maude_core.execute.execute_tool")
    def test_mixed_parallel_and_sequential(self, mock_exec):
        """Stage with both read-only and mutating tools."""
        mock_exec.return_value = "ok"
        execute_plan([
            [
                {"name": "read_file", "args": {"path": "a"}},
                {"name": "list_directory", "args": {"path": "/tmp"}},
                {"name": "run_command", "args": {"command": "echo"}},
            ]
        ])
        assert mock_exec.call_count == 3

    @patch("maude_core.execute.execute_tool")
    def test_tool_error_captured(self, mock_exec):
        mock_exec.side_effect = RuntimeError("boom")
        result = execute_plan([
            [{"name": "read_file", "args": {"path": "x"}}]
        ])
        assert "Error: boom" in result

    @patch("maude_core.execute.execute_tool")
    def test_result_truncation(self, mock_exec):
        mock_exec.return_value = "x" * 5000
        result = execute_plan([
            [{"name": "read_file", "args": {"path": "big.txt"}}]
        ])
        assert "truncated" in result

    def test_empty_stages(self):
        result = execute_plan([])
        assert "0 stages" in result

    def test_empty_stage(self):
        result = execute_plan([[]])
        assert "1 stages" in result
        assert "0 tools" in result

    @patch("maude_core.execute.execute_tool")
    def test_invalid_stage_type(self, mock_exec):
        result = execute_plan(["not a list"])
        assert "ERROR" in result


class TestToolRegistration:
    """Verify execute_plan is properly registered."""

    def test_handler_registered(self):
        from tool_registry import get_handler
        h = get_handler("execute_plan")
        assert h is not None

    def test_in_tool_definitions(self):
        from maude_core import TOOLS
        names = [t["function"]["name"] for t in TOOLS]
        assert "execute_plan" in names

    def test_in_core_tools(self):
        from maude_core import _CORE_TOOL_NAMES
        assert "execute_plan" in _CORE_TOOL_NAMES

    def test_schema_has_stages_param(self):
        from maude_core import TOOLS
        tool = next(t for t in TOOLS if t["function"]["name"] == "execute_plan")
        params = tool["function"]["parameters"]
        assert "stages" in params["properties"]
        assert params["required"] == ["stages"]
