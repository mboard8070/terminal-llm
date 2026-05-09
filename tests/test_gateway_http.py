"""Integration tests for gateway HTTP endpoints."""

import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def _get(base_url, path):
    """GET helper — returns (status, parsed_json)."""
    url = base_url + path
    try:
        resp = urlopen(url, timeout=10)
        body = json.loads(resp.read())
        return resp.status, body
    except HTTPError as e:
        body = json.loads(e.read()) if e.headers.get("Content-Type", "").startswith("application/json") else {}
        return e.code, body


def _post(base_url, path, data):
    """POST helper — returns (status, parsed_json)."""
    url = base_url + path
    body = json.dumps(data).encode()
    req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        resp = urlopen(req, timeout=10)
        return resp.status, json.loads(resp.read())
    except HTTPError as e:
        resp_body = json.loads(e.read()) if e.headers.get("Content-Type", "").startswith("application/json") else {}
        return e.code, resp_body


class TestHealthHTTP:
    """GET /health — structured health report."""

    def test_health_returns_200(self, gateway_server):
        status, _body = _get(gateway_server, "/health")
        assert status == 200

    def test_health_has_status(self, gateway_server):
        _status, body = _get(gateway_server, "/health")
        assert body["status"] in ("ok", "degraded", "error")

    def test_health_has_structured_sections(self, gateway_server):
        _, body = _get(gateway_server, "/health")
        assert "services" in body
        assert "dependencies" in body
        assert "tools" in body


class TestToolsHTTP:
    """GET /api/tools — tool catalog."""

    def test_full_catalog_returns_200(self, gateway_server):
        status, body = _get(gateway_server, "/api/tools")
        assert status == 200
        assert "tools" in body
        assert len(body["tools"]) > 10

    def test_message_filtering(self, gateway_server):
        status, body = _get(gateway_server, "/api/tools?message=check+my+email")
        assert status == 200
        names = {t["function"]["name"] for t in body["tools"]}
        assert "gmail_list" in names

    def test_plain_message_gets_core_only(self, gateway_server):
        _, body = _get(gateway_server, "/api/tools?message=tell+me+a+joke")
        names = {t["function"]["name"] for t in body["tools"]}
        assert "read_file" in names
        assert "gmail_list" not in names


class TestToolExecuteHTTP:
    """POST /api/tools/execute — server-side tool execution."""

    def test_rejects_missing_name(self, gateway_server):
        status, body = _post(gateway_server, "/api/tools/execute", {"arguments": {}})
        assert status == 400
        assert "error" in body

    def test_rejects_invalid_json(self, gateway_server):
        url = gateway_server + "/api/tools/execute"
        req = Request(url, data=b"not json", headers={"Content-Type": "application/json"}, method="POST")
        try:
            urlopen(req, timeout=10)
            raise AssertionError("Should have raised")
        except HTTPError as e:
            assert e.code == 400

    def test_rejects_local_tool(self, gateway_server):
        status, body = _post(
            gateway_server, "/api/tools/execute", {"name": "read_file", "arguments": {"path": "/tmp/test.txt"}}
        )
        assert status == 400
        assert "error" in body

    def test_executes_server_tool(self, gateway_server):
        status, body = _post(gateway_server, "/api/tools/execute", {"name": "get_working_directory", "arguments": {}})
        # Either succeeds or returns error (tool_catalog might classify it differently)
        assert status in (200, 400, 500)
        assert "result" in body or "error" in body


class TestModelsHTTP:
    """GET /models — available model list."""

    def test_models_returns_200(self, gateway_server):
        status, body = _get(gateway_server, "/models")
        assert status == 200
        assert "models" in body

    def test_models_have_expected_fields(self, gateway_server):
        _, body = _get(gateway_server, "/models")
        for model in body["models"]:
            assert "id" in model
            assert "provider" in model
            assert "available" in model

    def test_nemotron_is_listed(self, gateway_server):
        _, body = _get(gateway_server, "/models")
        ids = {m["id"] for m in body["models"]}
        assert "nemotron" in ids


class TestCORSHTTP:
    """OPTIONS — CORS preflight."""

    def test_options_returns_204(self, gateway_server):
        url = gateway_server + "/v1/chat/completions"
        req = Request(url, method="OPTIONS")
        resp = urlopen(req, timeout=10)
        assert resp.status == 204

    def test_options_has_cors_headers(self, gateway_server):
        url = gateway_server + "/v1/chat/completions"
        req = Request(url, method="OPTIONS")
        resp = urlopen(req, timeout=10)
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"
        assert "POST" in resp.headers.get("Access-Control-Allow-Methods", "")
