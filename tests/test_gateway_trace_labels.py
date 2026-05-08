"""Tests for user-facing gateway trace labels."""

from gateway.cloud import CloudMixin


class TestToolTaskLabels:
    def test_run_agent_label_says_spawned_agent(self):
        label = CloudMixin._tool_task_label("run_agent", {"agent": "research"})

        assert label == "Spawned 1 agent: research"

    def test_run_agents_label_includes_count_and_names(self):
        label = CloudMixin._tool_task_label(
            "run_agents",
            {
                "tasks": [
                    {"agent": "research", "task": "Find docs"},
                    {"agent": "code", "task": "Inspect implementation"},
                ]
            },
        )

        assert label == "Spawned 2 agents: research, code"

    def test_execute_plan_label_is_reserved_for_planning(self):
        label = CloudMixin._tool_task_label("execute_plan", {"stages": [[], []]})

        assert label == "Plan mode: executing 2 stages"


class TestModelRouteTrace:
    def test_alias_route_explains_requested_and_resolved_model(self):
        payload = CloudMixin._model_route_trace_payload(
            {"requested_model": "sonnet"},
            {
                "provider": "anthropic",
                "base_url": "https://api.anthropic.com",
                "max_context": 200000,
            },
            "claude-sonnet-4-20250514",
        )

        assert payload["requested_model"] == "sonnet"
        assert payload["resolved_model"] == "claude-sonnet-4-20250514"
        assert payload["provider"] == "anthropic"
        assert payload["endpoint"] == "api.anthropic.com"
        assert payload["route_kind"] == "alias"
        assert payload["summary"] == "sonnet -> claude-sonnet-4-20250514"

    def test_direct_route_uses_model_as_summary(self):
        payload = CloudMixin._model_route_trace_payload(
            {"requested_model": "gemma-4-31b"},
            {
                "provider": "local",
                "base_url": "http://localhost:30013",
                "max_context": 32768,
            },
            "gemma-4-31b",
        )

        assert payload["route_kind"] == "direct"
        assert payload["summary"] == "gemma-4-31b"
        assert payload["endpoint"] == "localhost:30013"
