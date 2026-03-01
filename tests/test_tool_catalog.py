"""
Tests for tool_catalog.py — catalog structure, filtering, execution targets.
"""

import pytest
from tool_catalog import get_catalog, get_filtered_tools, execute_server_tool, _LOCAL_TOOLS, _CLIENT_ONLY_TOOLS


class TestCatalogStructure:
    """Test that the catalog is well-formed."""

    def test_catalog_has_required_keys(self):
        catalog = get_catalog()
        assert "tools" in catalog
        assert "groups" in catalog
        assert "core_tools" in catalog
        assert "execution_targets" in catalog

    def test_all_tools_have_valid_schema(self):
        catalog = get_catalog()
        for tool in catalog["tools"]:
            assert tool.get("type") == "function", f"Tool missing type: {tool}"
            func = tool.get("function", {})
            assert "name" in func, f"Tool missing name: {tool}"
            assert "description" in func, f"Tool missing description: {func.get('name')}"
            assert "parameters" in func, f"Tool missing parameters: {func.get('name')}"
            params = func["parameters"]
            assert params.get("type") == "object", f"Parameters not object: {func['name']}"

    def test_no_duplicate_tool_names(self):
        catalog = get_catalog()
        names = [t["function"]["name"] for t in catalog["tools"]]
        assert len(names) == len(set(names)), f"Duplicate tool names: {[n for n in names if names.count(n) > 1]}"

    def test_every_tool_has_execution_target(self):
        catalog = get_catalog()
        targets = catalog["execution_targets"]
        for tool in catalog["tools"]:
            name = tool["function"]["name"]
            assert name in targets, f"Tool {name} missing execution target"
            assert targets[name] in ("local", "server"), f"Invalid target for {name}: {targets[name]}"

    def test_core_tools_nonempty(self):
        catalog = get_catalog()
        assert len(catalog["core_tools"]) >= 5

    def test_groups_have_keywords_and_tools(self):
        catalog = get_catalog()
        for gname, group in catalog["groups"].items():
            assert "keywords" in group, f"Group {gname} missing keywords"
            assert "tools" in group, f"Group {gname} missing tools"
            assert len(group["keywords"]) > 0, f"Group {gname} has no keywords"
            assert len(group["tools"]) > 0, f"Group {gname} has no tools"


class TestCatalogFiltering:
    """Test keyword filtering."""

    def test_core_tools_always_included(self):
        tools = get_filtered_tools("tell me a joke")
        names = {t["function"]["name"] for t in tools}
        assert "read_file" in names
        assert "web_search" in names

    def test_email_includes_gmail(self):
        tools = get_filtered_tools("check my email inbox")
        names = {t["function"]["name"] for t in tools}
        assert "gmail_list" in names

    def test_no_keywords_returns_core_only(self):
        tools = get_filtered_tools("hello world")
        names = {t["function"]["name"] for t in tools}
        # Should NOT include gmail, drive, calendar, etc.
        assert "gmail_list" not in names
        assert "drive_list" not in names

    def test_calendar_keyword(self):
        tools = get_filtered_tools("what's on my calendar today")
        names = {t["function"]["name"] for t in tools}
        assert "calendar_list_events" in names

    def test_image_keyword(self):
        tools = get_filtered_tools("generate an image of a cat")
        names = {t["function"]["name"] for t in tools}
        assert "generate_image" in names


class TestExecutionTargets:
    """Test that execution targets are correctly assigned."""

    def test_local_tools_are_local(self):
        catalog = get_catalog()
        targets = catalog["execution_targets"]
        for name in _LOCAL_TOOLS:
            if name in targets:
                assert targets[name] == "local", f"{name} should be local"

    def test_server_tools_are_server(self):
        catalog = get_catalog()
        targets = catalog["execution_targets"]
        assert targets.get("web_search") == "server"
        assert targets.get("gmail_list") == "server"


class TestExecuteServerTool:
    """Test execute_server_tool rejection and dispatch."""

    def test_rejects_local_tool(self):
        result = execute_server_tool("read_file", {"path": "/tmp/test"})
        assert result["error"] is not None
        assert "local tool" in result["error"]

    def test_rejects_client_only_tool(self):
        result = execute_server_tool("upload_to_server", {"local_path": "/tmp/test"})
        assert result["error"] is not None
        assert "local tool" in result["error"]
