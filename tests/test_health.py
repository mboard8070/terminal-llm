"""
Tests for health.py — HealthChecker.
"""

import pytest
from health import HealthChecker, check_tool_ready, _ALWAYS_READY


class TestCheckDependencies:
    def test_returns_dict(self):
        checker = HealthChecker()
        deps = checker.check_dependencies()
        assert isinstance(deps, dict)
        assert len(deps) > 0

    def test_each_dep_has_available_key(self):
        checker = HealthChecker()
        deps = checker.check_dependencies()
        for pkg, info in deps.items():
            assert "available" in info, f"Dep {pkg} missing 'available' key"

    def test_requests_is_available(self):
        """requests is definitely installed since we use it everywhere."""
        checker = HealthChecker()
        deps = checker.check_dependencies()
        assert deps["requests"]["available"] is True


class TestCheckToolReady:
    def test_always_ready_tools(self):
        for name in _ALWAYS_READY:
            ready, reason = check_tool_ready(name)
            assert ready is True, f"Tool {name} should be ready but: {reason}"
            assert reason == ""

    def test_read_file_ready(self):
        ready, reason = check_tool_ready("read_file")
        assert ready is True

    def test_web_search_depends_on_ddgs(self):
        ready, reason = check_tool_ready("web_search")
        # Either ddgs is installed (True) or it's not (False with reason)
        assert isinstance(ready, bool)
        if not ready:
            assert "ddgs" in reason

    def test_unknown_tool_assumed_ready(self):
        ready, reason = check_tool_ready("totally_made_up_tool")
        assert ready is True

    def test_browser_tools_check_playwright(self):
        ready, reason = check_tool_ready("browser_open")
        assert isinstance(ready, bool)
        if not ready:
            assert "playwright" in reason


class TestCheckServices:
    def test_returns_dict_with_services(self):
        checker = HealthChecker()
        services = checker.check_services()
        assert "llama_server" in services
        assert "personaplex" in services

    def test_each_service_has_status(self):
        checker = HealthChecker()
        services = checker.check_services()
        for name, info in services.items():
            assert "status" in info
            assert info["status"] in ("ok", "down")
            assert "port" in info


class TestCheckAll:
    def test_full_report_structure(self):
        checker = HealthChecker()
        report = checker.check_all()
        assert "status" in report
        assert report["status"] in ("ok", "degraded", "error")
        assert "services" in report
        assert "dependencies" in report
        assert "tools" in report
        assert "uptime" in report

    def test_tools_section(self):
        checker = HealthChecker()
        report = checker.check_all()
        tools = report["tools"]
        assert "total" in tools
        assert "functional" in tools
        assert "degraded" in tools
        assert "reasons" in tools
        assert tools["total"] >= tools["functional"]
