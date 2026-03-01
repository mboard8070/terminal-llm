"""
Tests for client tool_router.py — ToolRouter fetch, cache, route.

These tests run without needing a real gateway by mocking HTTP calls.
"""

import time
import json
import pytest
from unittest.mock import patch, MagicMock

# Add maude-client to path
import sys
from pathlib import Path
CLIENT_DIR = Path(__file__).parent.parent / "maude-client"
if str(CLIENT_DIR) not in sys.path:
    sys.path.insert(0, str(CLIENT_DIR))

from maude_client.tool_router import (
    ToolRouter, CLIENT_ONLY_TOOLS, _LOCAL_TOOL_NAMES,
    _OFFLINE_TOOLS, CACHE_TTL,
)


@pytest.fixture
def mock_catalog():
    """A minimal catalog response from the gateway."""
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                }
            },
        ],
        "groups": {
            "gmail": {
                "keywords": ["gmail", "email", "inbox"],
                "tools": ["gmail_list", "gmail_read"],
            }
        },
        "core_tools": ["read_file", "web_search"],
        "execution_targets": {
            "web_search": "server",
            "read_file": "local",
        },
    }


class TestToolRouterFetch:
    def test_offline_fallback(self):
        """When gateway is unreachable, use offline tools."""
        router = ToolRouter(gateway_url="http://192.0.2.1:39999")
        catalog = router.fetch_catalog()
        assert not router.is_online
        assert len(catalog["tools"]) > 0
        # Should have local tools
        names = {t["function"]["name"] for t in catalog["tools"]}
        assert "read_file" in names

    @patch("maude_client.tool_router.requests.get")
    def test_online_fetch(self, mock_get, mock_catalog):
        """When gateway responds, use server catalog."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_catalog
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        router = ToolRouter(gateway_url="https://fake:30000")
        catalog = router.fetch_catalog()

        assert router.is_online
        # Should have server tools + client-only tools merged
        names = {t["function"]["name"] for t in router._all_tools}
        assert "web_search" in names
        # Client-only tools should be merged in
        for ct in CLIENT_ONLY_TOOLS:
            assert ct["function"]["name"] in names

    @patch("maude_client.tool_router.requests.get")
    def test_cache_expires(self, mock_get, mock_catalog):
        """Catalog is re-fetched after CACHE_TTL."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_catalog
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        router = ToolRouter(gateway_url="https://fake:30000")
        router.fetch_catalog()
        assert mock_get.call_count == 1

        # Second call within TTL — should use cache
        router.fetch_catalog()
        assert mock_get.call_count == 1

        # Expire the cache
        router._catalog_ts = time.time() - CACHE_TTL - 1
        router.fetch_catalog()
        assert mock_get.call_count == 2


class TestToolRouterFiltering:
    @patch("maude_client.tool_router.requests.get")
    def test_get_tools_for_message(self, mock_get, mock_catalog):
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_catalog
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        router = ToolRouter(gateway_url="https://fake:30000")
        router.fetch_catalog()

        # Plain message — should get core + local tools
        tools = router.get_tools_for_message("hello")
        names = {t["function"]["name"] for t in tools}
        assert "read_file" in names


class TestToolRouterExecution:
    def test_local_tool_dispatch(self, tmp_path):
        """Local tools should dispatch to client_tools.execute_tool."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        router = ToolRouter(gateway_url="http://192.0.2.1:39999")
        result = router.execute("read_file", {"path": str(test_file)})
        assert "hello world" in result

    @patch("maude_client.tool_router.requests.post")
    def test_remote_tool_dispatch(self, mock_post):
        """Server tools should POST to gateway."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"result": "search results here", "elapsed": 0.5, "error": None}
        mock_post.return_value = mock_resp

        router = ToolRouter(gateway_url="https://fake:30000")
        result = router.execute("web_search", {"query": "test"})
        assert "search results" in result
        mock_post.assert_called_once()

    def test_remote_tool_offline_error(self):
        """When gateway is down, server tools return error."""
        router = ToolRouter(gateway_url="http://192.0.2.1:39999")
        result = router.execute("web_search", {"query": "test"})
        assert "Error" in result


class TestFastDispatch:
    def test_list_directory(self, tmp_path):
        (tmp_path / "file.txt").write_text("hi")

        router = ToolRouter(gateway_url="http://192.0.2.1:39999")
        # Need to be in the tmp_path for list_directory to work with "."
        import os
        orig = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = router.fast_dispatch("list files in directory")
            if result:
                name, args, output = result
                assert name == "list_directory"
        finally:
            os.chdir(orig)

    def test_no_match(self):
        router = ToolRouter(gateway_url="http://192.0.2.1:39999")
        result = router.fast_dispatch("tell me a joke")
        assert result is None


class TestHealthWarnings:
    def test_offline_warning(self):
        router = ToolRouter(gateway_url="http://192.0.2.1:39999")
        router.fetch_catalog()
        warnings = router.health_warnings()
        assert any("unreachable" in w.lower() or "local" in w.lower() for w in warnings)

    @patch("maude_client.tool_router.requests.get")
    def test_no_warnings_when_online(self, mock_get, mock_catalog):
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_catalog
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        router = ToolRouter(gateway_url="https://fake:30000")
        router.fetch_catalog()
        warnings = router.health_warnings()
        # Should not have "unreachable" warning
        assert not any("unreachable" in w.lower() for w in warnings)
