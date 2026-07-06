"""
Stand-up state machine test for MAUDE gateway readiness.

This is a deterministic CI check that proves the gateway can stand up enough to
serve health, model routing metadata, tool catalog metadata, and a simple server
tool execution path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any
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


class StandupState(Enum):
    BOOTSTRAP = auto()
    HEALTH_OK = auto()
    MODELS_LISTED = auto()
    TOOLS_LISTED = auto()
    TOOL_EXEC_OK = auto()
    READY = auto()


@dataclass
class StandupMachine:
    base_url: str
    state: StandupState = StandupState.BOOTSTRAP
    trace: list[StandupState] = field(default_factory=lambda: [StandupState.BOOTSTRAP])
    observations: dict[str, Any] = field(default_factory=dict)

    def transition(self, expected: StandupState, next_state: StandupState) -> None:
        assert self.state is expected, f"expected {expected.name}, got {self.state.name}"
        self.state = next_state
        self.trace.append(next_state)

    def check_health(self) -> None:
        status, body = _get(self.base_url, "/health")
        assert status == 200
        assert body["status"] in ("ok", "degraded", "error")
        assert "services" in body
        assert "dependencies" in body
        assert "tools" in body
        self.observations["health_status"] = body["status"]
        self.transition(StandupState.BOOTSTRAP, StandupState.HEALTH_OK)

    def check_models(self) -> None:
        status, body = _get(self.base_url, "/models")
        assert status == 200
        ids = {model["id"] for model in body["models"]}
        assert {"nemotron", "nemotron-super", "nemotron-ultra", "nemotron-a3b"}.issubset(ids)
        self.observations["model_count"] = len(ids)
        self.transition(StandupState.HEALTH_OK, StandupState.MODELS_LISTED)

    def check_tools(self) -> None:
        status, body = _get(self.base_url, "/api/tools")
        assert status == 200
        tools = body["tools"]
        names = {tool["function"]["name"] for tool in tools}
        assert "skill_calc" in names
        self.observations["tool_count"] = len(tools)
        self.transition(StandupState.MODELS_LISTED, StandupState.TOOLS_LISTED)

    def check_tool_execution(self) -> None:
        status, body = _post(
            self.base_url,
            "/api/tools/execute",
            {"name": "skill_calc", "arguments": {"expression": "1+1"}},
        )
        assert status == 200
        assert body["error"] is None
        assert "2" in body["result"]
        self.observations["tool_elapsed"] = body["elapsed"]
        self.transition(StandupState.TOOLS_LISTED, StandupState.TOOL_EXEC_OK)

    def mark_ready(self) -> None:
        self.transition(StandupState.TOOL_EXEC_OK, StandupState.READY)

    def run(self) -> None:
        self.check_health()
        self.check_models()
        self.check_tools()
        self.check_tool_execution()
        self.mark_ready()


def test_gateway_standup_state_machine(gateway_server):
    machine = StandupMachine(gateway_server)
    machine.run()

    assert machine.state is StandupState.READY
    assert machine.trace == [
        StandupState.BOOTSTRAP,
        StandupState.HEALTH_OK,
        StandupState.MODELS_LISTED,
        StandupState.TOOLS_LISTED,
        StandupState.TOOL_EXEC_OK,
        StandupState.READY,
    ]
