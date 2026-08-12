"""Tests for tool tiering, session activation, and lazy domain schemas."""

import json

import pytest

from maude_core.tool_groups import (
    _CORE_TOOL_NAMES,
    activate_domain,
    clear_session_domains,
    get_session_domains,
    get_tools_for_message,
    list_domain_catalog,
    payload_stats,
)
from tool_catalog import get_catalog, get_filtered_tools


@pytest.fixture(autouse=True)
def _clean_session():
    clear_session_domains("test")
    clear_session_domains("default")
    yield
    clear_session_domains("test")
    clear_session_domains("default")


def _full_names(tools):
    return {
        t["function"]["name"]
        for t in tools
        if not (t["function"].get("description") or "").startswith("[lazy")
        and not (t["function"].get("name") or "").startswith("domain_")
    }


def _lazy_names(tools):
    return {
        t["function"]["name"]
        for t in tools
        if (t["function"].get("description") or "").startswith("[lazy")
        or (t["function"].get("name") or "").startswith("domain_")
    }


class TestCoreShrink:
    def test_core_is_small(self):
        # Always-on set should be ~file/shell/search/memory, not browser suite
        assert len(_CORE_TOOL_NAMES) <= 25
        assert "read_file" in _CORE_TOOL_NAMES
        assert "run_command" in _CORE_TOOL_NAMES
        assert "web_search" in _CORE_TOOL_NAMES
        assert "save_memory" in _CORE_TOOL_NAMES
        assert "browser_open" not in _CORE_TOOL_NAMES
        assert "social_post" not in _CORE_TOOL_NAMES
        assert "sandbox_exec" not in _CORE_TOOL_NAMES

    def test_default_turn_payload_much_smaller_than_full_catalog(self):
        from maude_core.tool_defs import TOOLS

        default = get_tools_for_message("hello how are you", session_id="test")
        stats = payload_stats(default)
        full_chars = len(json.dumps(TOOLS))
        # Default payload should be dramatically smaller than dumping everything
        assert stats["payload_chars"] < full_chars * 0.35
        assert stats["full_schema_count"] <= 30
        # Browser not fully expanded on plain hello
        assert "browser_open" not in _full_names(default)


class TestSessionActivation:
    def test_browser_keywords_activate_full_browser_schemas(self):
        tools = get_tools_for_message(
            "open the browser and login to twitter", session_id="test"
        )
        names = _full_names(tools)
        assert "browser_open" in names
        assert "browser_click" in names
        assert "browser_login" in names

    def test_browser_stays_sticky_for_session(self):
        get_tools_for_message("open the browser and navigate to example.com", session_id="test")
        assert "browser" in get_session_domains("test")
        tools = get_tools_for_message("now click the submit button", session_id="test")
        assert "browser_click" in _full_names(tools)

    def test_activate_domain_media(self):
        activated = activate_domain("media", session_id="test")
        assert "media" in activated
        tools = get_tools_for_message("ok", session_id="test")
        names = _full_names(tools)
        assert "generate_image" in names
        assert "view_image" in names

    def test_google_email_keywords(self):
        tools = get_tools_for_message("check my email inbox", session_id="test")
        assert "gmail_list" in _full_names(tools)
        # sticky google after email
        tools2 = get_tools_for_message("and what about drive?", session_id="test")
        # drive keyword also matches; gmail should remain via sticky
        assert "gmail_list" in _full_names(tools2) or "drive_list" in _full_names(tools2)

    def test_history_reactivates_used_tools(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "browser_open", "arguments": "{}"},
                    }
                ],
            }
        ]
        tools = get_tools_for_message("continue", session_id="test", messages=messages)
        assert "browser_open" in _full_names(tools)


class TestLazySchemas:
    def test_default_has_domain_controls_not_browser_suite(self):
        tools = get_tools_for_message("hello", session_id="test")
        names = _full_names(tools)
        assert "list_tool_domains" in names
        assert "activate_tool_domain" in names
        assert "browser_open" not in names
        # list_tool_domains description carries names + one-liners
        ltd = next(t for t in tools if t["function"]["name"] == "list_tool_domains")
        desc = ltd["function"]["description"]
        assert "browser" in desc
        assert "media" in desc

    def test_lazy_stubs_opt_in(self, monkeypatch):
        monkeypatch.setenv("MAUDE_LAZY_TOOL_STUBS", "1")
        tools = get_tools_for_message("hello", session_id="test", lazy=True)
        lazy = _lazy_names(tools)
        assert any(n.startswith("domain_") for n in lazy)
        assert "domain_browser" in lazy or "domain_media" in lazy

    def test_lazy_stubs_default_off(self):
        tools = get_tools_for_message("hello", session_id="test")
        lazy = _lazy_names(tools)
        assert not any(n.startswith("domain_") for n in lazy)

    def test_activate_tool_domain_handler(self):
        from maude_core.execute import execute_tool

        clear_session_domains("default")
        result = execute_tool("activate_tool_domain", {"domain": "browser"})
        assert "Activated" in result
        assert "browser" in get_session_domains("default")

    def test_domain_stub_handler(self):
        from maude_core.execute import execute_tool

        clear_session_domains("default")
        result = execute_tool("domain_media", {"activate": True})
        assert "Activated" in result
        assert "media" in get_session_domains("default")

    def test_list_tool_domains(self):
        from maude_core.execute import execute_tool

        result = execute_tool("list_tool_domains", {})
        assert "browser" in result
        assert "media" in result


class TestRareDomains:
    def test_hyperframes_keyword(self):
        tools = get_tools_for_message("create a HyperFrames product video", session_id="test")
        names = _full_names(tools)
        assert "skill_hyperframes" in names or "hyperframes_init" in names

    def test_substack_not_in_default(self):
        tools = get_tools_for_message("hello", session_id="test")
        assert "substack_create_draft" not in _full_names(tools)


class TestCatalogMetadata:
    def test_catalog_exposes_tiers(self):
        catalog = get_catalog()
        assert "session_groups" in catalog
        assert "rare_groups" in catalog
        assert "domains" in catalog
        assert "browser" in catalog["session_groups"] or any(
            g.get("tier") == "session" for g in catalog["groups"].values()
        )
        assert len(catalog["domains"]) >= 5

    def test_filtered_tools_wrapper(self):
        tools = get_filtered_tools("hello world", session_id="test")
        assert "read_file" in _full_names(tools)
        assert "gmail_list" not in _full_names(tools)
