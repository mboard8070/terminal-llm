"""Tests for Mistral-specific Maude client payload sanitation."""

import sys
from pathlib import Path

CLIENT_DIR = Path(__file__).parent.parent / "maude-client"
if str(CLIENT_DIR) not in sys.path:
    sys.path.insert(0, str(CLIENT_DIR))

from maude_client.cli import _prepare_payload_for_provider, _sanitize_tools_for_mistral


def test_mistral_sanitizer_removes_empty_required_and_unsupported_schema_fields():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "browser.fill form",
                "description": "  Fill   a form.  " * 100,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fields": {
                            "type": "object",
                            "description": "selector/value map",
                            "additionalProperties": {"type": "string"},
                        },
                        "mode": {
                            "type": ["string", "null"],
                            "enum": ["fast", "slow", None],
                            "default": "fast",
                        },
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
        }
    ]

    clean = _sanitize_tools_for_mistral(tools)

    assert clean[0]["function"]["name"] == "browser_fill_form"
    params = clean[0]["function"]["parameters"]
    assert "required" not in params
    assert "additionalProperties" not in params
    assert "default" not in params["properties"]["mode"]
    assert params["properties"]["mode"]["type"] == "string"
    assert params["properties"]["mode"]["enum"] == ["fast", "slow"]
    assert "additionalProperties" not in params["properties"]["fields"]


def test_prepare_payload_only_sanitizes_mistral_tools():
    tool = {
        "type": "function",
        "function": {
            "name": "noop",
            "description": "No-op",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    mistral_payload = {"model": "mistral", "tools": [tool], "tool_choice": "auto"}
    nemotron_payload = {"model": "nemotron-super", "tools": [tool], "tool_choice": "auto"}

    assert "required" not in _prepare_payload_for_provider(mistral_payload)["tools"][0]["function"]["parameters"]
    assert _prepare_payload_for_provider(nemotron_payload)["tools"][0]["function"]["parameters"]["required"] == []
