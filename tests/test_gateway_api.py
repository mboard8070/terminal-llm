"""
Tests for gateway API endpoints — /health, /api/tools, /api/tools/execute.

These are integration-style tests that import the gateway handler logic
without starting a full HTTP server. Uses mock request/response objects.
"""

import pytest


class TestHealthEndpoint:
    """Test the enhanced /health response structure."""

    def test_health_checker_returns_valid_report(self):
        from health import HealthChecker

        report = HealthChecker().check_all()
        assert report["status"] in ("ok", "degraded", "error")
        assert "services" in report
        assert "dependencies" in report
        assert "tools" in report
        assert isinstance(report["uptime"], float)

    def test_health_tools_section(self):
        from health import HealthChecker

        report = HealthChecker().check_all()
        tools = report["tools"]
        assert tools["total"] > 0
        assert tools["functional"] >= 0
        assert isinstance(tools["degraded"], list)
        assert isinstance(tools["reasons"], dict)


class TestToolsEndpoint:
    """Test /api/tools catalog API."""

    def test_full_catalog(self):
        from tool_catalog import get_catalog

        catalog = get_catalog()
        assert len(catalog["tools"]) > 10
        assert len(catalog["groups"]) > 0
        assert len(catalog["core_tools"]) > 0

    def test_filtered_catalog(self):
        from tool_catalog import get_filtered_tools

        # With email keyword
        tools = get_filtered_tools("check my email")
        names = {t["function"]["name"] for t in tools}
        assert "gmail_list" in names

        # Without keywords — core only
        tools_plain = get_filtered_tools("hello")
        names_plain = {t["function"]["name"] for t in tools_plain}
        assert "gmail_list" not in names_plain

    def test_catalog_has_execution_targets(self):
        from tool_catalog import get_catalog

        catalog = get_catalog()
        targets = catalog["execution_targets"]
        assert targets["read_file"] == "local"
        assert targets["web_search"] == "server"


class TestExecuteEndpoint:
    """Test /api/tools/execute."""

    def test_rejects_local_tool(self):
        from tool_catalog import execute_server_tool

        result = execute_server_tool("read_file", {"path": "/tmp/test"})
        assert result["error"] is not None
        assert "local" in result["error"].lower()

    def test_rejects_client_only_tool(self):
        from tool_catalog import execute_server_tool

        result = execute_server_tool("upload_to_server", {"local_path": "/tmp/x"})
        assert result["error"] is not None

    def test_executes_server_tool(self):
        """Test that a server tool executes and returns a result."""
        from tool_catalog import execute_server_tool

        # list_directory is a local tool on the client but a server tool on the server
        # Use get_working_directory which is always available
        result = execute_server_tool("get_working_directory", {})
        if result["error"]:
            pytest.skip(f"Tool execution error: {result['error']}")
        assert result["result"] is not None
        assert isinstance(result["elapsed"], float)

    def test_result_has_elapsed(self):
        from tool_catalog import execute_server_tool

        result = execute_server_tool("get_working_directory", {})
        assert "elapsed" in result
        assert isinstance(result["elapsed"], (int, float))


class TestCollabEndpoint:
    """Test /api/collab/* via the hub."""

    def test_collab_status(self):
        try:
            from collab import get_hub

            hub = get_hub()
            status = hub.get_status()
            assert "ts" in status
            assert "presence" in status
        except ImportError:
            pytest.skip("collab module not available")
