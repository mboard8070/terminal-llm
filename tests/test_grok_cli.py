"""Tests for Grok CLI history truncation and streaming-json progress traces."""

import json
import os

from gateway.cloud import CloudMixin


class TestPrepareGrokHistory:
    def test_keeps_short_history_intact(self, monkeypatch):
        monkeypatch.setenv("MAUDE_GROK_KEEP_RECENT", "12")
        monkeypatch.setenv("MAUDE_GROK_MAX_PROMPT_CHARS", "48000")
        msgs = [
            {"role": "system", "content": "You are Maude."},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        prepared, meta = CloudMixin._prepare_grok_history(msgs)
        assert meta["removed"] == 0
        assert any(m["role"] == "system" for m in prepared)
        assert prepared[-1]["content"] == "hi"

    def test_summarizes_old_messages_when_over_keep_recent(self, monkeypatch):
        monkeypatch.setenv("MAUDE_GROK_KEEP_RECENT", "4")
        monkeypatch.setenv("MAUDE_GROK_MAX_MSG_CHARS", "4000")
        monkeypatch.setenv("MAUDE_GROK_MAX_PROMPT_CHARS", "48000")
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(10):
            msgs.append({"role": "user", "content": f"user turn {i} " + ("x" * 50)})
            msgs.append({"role": "assistant", "content": f"assistant turn {i}"})

        prepared, meta = CloudMixin._prepare_grok_history(msgs)
        assert meta["removed"] > 0
        summary = next(
            (m for m in prepared if CloudMixin._message_text(m).startswith("[Earlier conversation")),
            None,
        )
        assert summary is not None
        # Latest user/assistant retained
        texts = [CloudMixin._message_text(m) for m in prepared]
        assert any("user turn 9" in t for t in texts)
        assert any("assistant turn 9" in t for t in texts)
        # Early turns only appear in summary, not as full messages
        full_early = [
            m
            for m in prepared
            if CloudMixin._message_text(m) == "user turn 0 " + ("x" * 50)
        ]
        assert full_early == []

    def test_truncates_oversized_message_body(self, monkeypatch):
        monkeypatch.setenv("MAUDE_GROK_MAX_MSG_CHARS", "500")
        monkeypatch.setenv("MAUDE_GROK_KEEP_RECENT", "12")
        msgs = [{"role": "user", "content": "A" * 5000}]
        prepared, meta = CloudMixin._prepare_grok_history(msgs)
        body = CloudMixin._message_text(prepared[-1])
        assert len(body) < 600
        assert "truncated" in body

    def test_messages_to_prompt_flattens_roles(self):
        prompt = CloudMixin._messages_to_grok_prompt(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        )
        assert "USER:\nhi" in prompt
        assert "ASSISTANT:\nhello" in prompt


class TestGrokJsonTraces:
    def test_last_grok_text_accumulates_text_events(self):
        stdout = "\n".join(
            [
                json.dumps({"type": "thought", "data": "hmm"}),
                json.dumps({"type": "text", "data": "Hel"}),
                json.dumps({"type": "text", "data": "lo"}),
                json.dumps({"type": "end", "stopReason": "end_turn"}),
            ]
        )
        assert CloudMixin._last_grok_text(stdout) == "Hello"

    def test_last_grok_text_falls_back_to_result_event(self):
        stdout = json.dumps({"type": "result", "result": "pong", "subtype": "success"})
        assert CloudMixin._last_grok_text(stdout) == "pong"

    def test_emit_tool_call_and_completed_update(self):
        traces = []

        class Fake(CloudMixin):
            def _send_trace_sse(self, trace_type, data):
                traces.append((trace_type, data))

        fake = Fake()
        active = {}
        started = 0.0
        # Shape matches live Grok CLI --output-format streaming-json.
        fake._emit_grok_json_trace(
            json.dumps(
                {
                    "type": "tool_call",
                    "toolCallId": "call-1",
                    "title": "run_terminal_command",
                    "kind": "execute",
                    "toolName": "run_terminal_command",
                    "rawInput": {"command": "ls /tmp", "description": "List tmp"},
                    "status": "pending",
                    "content": [],
                    "locations": [],
                }
            ),
            active,
            started,
        )
        fake._emit_grok_json_trace(
            json.dumps(
                {
                    "type": "tool_call_update",
                    "toolCallId": "call-1",
                    "status": "completed",
                    "rawOutput": {
                        "type": "Bash",
                        "output_for_prompt": "exit: 0\nfile.txt\n",
                        "exit_code": 0,
                    },
                }
            ),
            active,
            started,
        )

        assert traces[0][0] == "tool_call"
        assert traces[0][1]["name"] == "grok_run_terminal_command"
        assert "ls /tmp" in traces[0][1]["args"] or "List" in traces[0][1].get("task", "")
        assert traces[1][0] == "tool_result"
        assert traces[1][1]["name"] == "grok_run_terminal_command"
        assert "file.txt" in traces[1][1]["preview"] or "exit" in traces[1][1]["preview"]
        assert active == {}

    def test_in_progress_updates_do_not_emit_result(self):
        traces = []

        class Fake(CloudMixin):
            def _send_trace_sse(self, trace_type, data):
                traces.append((trace_type, data))

        fake = Fake()
        active = {"call-1": {"name": "grok_run_terminal_command", "started": 0.0}}
        fake._emit_grok_json_trace(
            json.dumps(
                {
                    "type": "tool_call_update",
                    "toolCallId": "call-1",
                    "status": "in_progress",
                    "rawOutput": {"output_for_prompt": "partial"},
                }
            ),
            active,
            0.0,
        )
        assert traces == []
        assert "call-1" in active

    def test_usage_emits_llm_call(self):
        traces = []

        class Fake(CloudMixin):
            def _send_trace_sse(self, trace_type, data):
                traces.append((trace_type, data))

        fake = Fake()
        fake._emit_grok_json_trace(
            json.dumps(
                {
                    "type": "usage",
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "cache_read_input_tokens": 50,
                    },
                }
            ),
            {},
            0.0,
        )
        assert traces[0][0] == "llm_call"
        assert traces[0][1]["prompt_tokens"] == 100
        assert traces[0][1]["completion_tokens"] == 20


class TestGrokToolLabels:
    def test_grok_cli_label(self):
        assert CloudMixin._tool_task_label("grok_cli", {"model": "grok-4.5"}) == "Running Grok agent"

    def test_grok_prefixed_shell_label(self):
        label = CloudMixin._tool_task_label(
            "grok_run_terminal_command",
            {"command": "ls /tmp", "description": "List tmp"},
        )
        assert "local command" in label.lower() or "Inspecting" in label or "List" in label
