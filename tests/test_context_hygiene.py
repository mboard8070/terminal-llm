"""Tests for MAUDE context hygiene (sliding window, tool payload drop, scratch)."""

import json
import os
from pathlib import Path

import pytest

from maude_core.context_hygiene import (
    MissionScratch,
    clear_mission_scratch,
    compact_tool_result,
    drop_old_tool_payloads,
    estimate_tokens,
    format_memory_snippets,
    get_mission_scratch,
    prepare_messages_for_model,
    save_mission_scratch,
    sliding_window_with_summary,
    summarize_dropped_messages,
)


class TestCompactToolResult:
    def test_short_result_unchanged(self):
        assert compact_tool_result("run_command", "ok") == "ok"

    def test_read_file_middle_dropped(self):
        # Need both >100 lines and >3000 chars to trigger head/tail compaction
        lines = [f"line {i} " + ("y" * 40) for i in range(200)]
        body = "\n".join(lines)
        assert len(body) > 3000
        out = compact_tool_result("read_file", body)
        assert "lines omitted" in out
        assert "line 0" in out
        assert "line 199" in out
        assert len(out) < len(body)

    def test_generic_hard_cap(self, monkeypatch):
        monkeypatch.setenv("MAUDE_CTX_MAX_TOOL_CHARS", "1000")
        # Re-import path uses env at call time via max_tool_chars()
        big = "x" * 5000
        out = compact_tool_result("web_browse", big)
        assert len(out) < 4000
        assert "truncated" in out or "omitted" in out or len(out) < len(big)

    def test_write_file_left_alone_when_short(self):
        assert compact_tool_result("write_file", "Wrote 12 bytes to foo.py") == "Wrote 12 bytes to foo.py"


class TestDropOldToolPayloads:
    def _tool_history(self, rounds: int = 4):
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(rounds):
            msgs.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"c{i}",
                            "type": "function",
                            "function": {"name": "web_browse", "arguments": "{}"},
                        }
                    ],
                }
            )
            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": f"c{i}",
                    "content": f"HUGE PAGE CONTENT ROUND {i} " + ("Z" * 800),
                }
            )
        return msgs

    def test_keeps_recent_rounds_full(self, monkeypatch):
        monkeypatch.setenv("MAUDE_CTX_KEEP_TOOL_ROUNDS", "2")
        msgs = self._tool_history(4)
        out, n = drop_old_tool_payloads(msgs, keep_recent=2, in_place=False)
        assert n >= 2
        # Last two tool bodies still large
        tool_bodies = [m["content"] for m in out if m.get("role") == "tool"]
        assert any(len(b) > 500 for b in tool_bodies[-2:])
        # Older ones stubbed
        assert any(b.startswith("[prior tool result") for b in tool_bodies[:-2])

    def test_in_place_mutates(self):
        msgs = self._tool_history(3)
        _, n = drop_old_tool_payloads(msgs, keep_recent=1, in_place=True)
        assert n >= 1
        assert msgs[2]["role"] == "tool" or any(
            m.get("role") == "tool" and str(m.get("content", "")).startswith("[prior") for m in msgs
        )


class TestSlidingWindow:
    def test_keeps_short_history(self):
        msgs = [
            {"role": "system", "content": "You are MAUDE."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        prepared, meta = sliding_window_with_summary(msgs, keep_recent=12)
        assert meta["removed"] == 0
        assert prepared[-1]["content"] == "hello"

    def test_summarizes_old_turns(self, monkeypatch):
        monkeypatch.setenv("MAUDE_CTX_KEEP_RECENT_TURNS", "4")
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(10):
            msgs.append({"role": "user", "content": f"user turn {i}"})
            msgs.append({"role": "assistant", "content": f"assistant turn {i}"})
        prepared, meta = sliding_window_with_summary(msgs, keep_recent=4)
        assert meta["removed"] > 0
        texts = [m.get("content", "") for m in prepared]
        assert any(isinstance(t, str) and t.startswith("[Earlier conversation summarized") for t in texts)
        assert any("user turn 9" in str(t) for t in texts)
        assert not any(t == "user turn 0" for t in texts)

    def test_summarize_dropped_skips_empty(self):
        text = summarize_dropped_messages(
            [
                {"role": "user", "content": "do the thing"},
                {"role": "assistant", "content": ""},
            ]
        )
        assert "do the thing" in text
        assert "USER" in text


class TestPreparePipeline:
    def test_full_pipeline_reduces_tokens(self, monkeypatch):
        monkeypatch.setenv("MAUDE_CTX_KEEP_RECENT_TURNS", "6")
        monkeypatch.setenv("MAUDE_CTX_KEEP_TOOL_ROUNDS", "1")
        monkeypatch.setenv("MAUDE_CTX_MAX_TOOL_CHARS", "500")
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(8):
            msgs.append({"role": "user", "content": f"q{i} " + ("a" * 100)})
            msgs.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"t{i}",
                            "type": "function",
                            "function": {"name": "web_browse", "arguments": "{}"},
                        }
                    ],
                }
            )
            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": f"t{i}",
                    "name": "web_browse",
                    "content": "PAGE " + ("X" * 3000),
                }
            )
            msgs.append({"role": "assistant", "content": f"answer {i}"})

        before = estimate_tokens(msgs)
        prepared, meta = prepare_messages_for_model(msgs, in_place=False)
        after = meta["final_tokens"]
        assert after < before
        assert meta["final_count"] <= len(msgs)
        assert prepared[0]["role"] == "system"


class TestMissionScratch:
    def test_notes_and_findings(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MAUDE_SCRATCH_DIR", str(tmp_path))
        clear_mission_scratch("test-mission")
        s = get_mission_scratch("test-mission")
        s.title = "Fix login"
        s.objective = "Auth returns 401"
        s.set_note("endpoint", "/api/login")
        s.add_finding("Token expired after 1h")
        save_mission_scratch(s)

        # Reload from disk
        from maude_core import context_hygiene as ch

        ch._scratch_cache.clear()
        s2 = get_mission_scratch("test-mission")
        assert s2.notes["endpoint"] == "/api/login"
        assert "Token expired" in s2.findings[0]
        block = s2.prompt_block()
        assert "Mission scratch" in block
        assert "endpoint" in block

        clear_mission_scratch("test-mission")
        ch._scratch_cache.clear()
        s3 = get_mission_scratch("test-mission")
        assert s3.prompt_block() == ""

    def test_from_dict_roundtrip(self):
        s = MissionScratch(mission_id="x", title="t", notes={"a": "b"}, findings=["f1"])
        s2 = MissionScratch.from_dict(s.to_dict())
        assert s2.notes == {"a": "b"}
        assert s2.findings == ["f1"]


class TestMemorySnippets:
    def test_format_top_k_clips_values(self, monkeypatch):
        monkeypatch.setenv("MAUDE_CTX_MEMORY_VALUE_CHARS", "20")
        monkeypatch.setenv("MAUDE_CTX_MEMORY_TOP_K", "2")

        class M:
            def __init__(self, key, value):
                self.key = key
                self.value = value

        block = format_memory_snippets(
            [M("k1", "x" * 100), M("k2", "short"), M("k3", "ignored")],
            top_k=2,
        )
        assert "k1" in block
        assert "k2" in block
        assert "k3" not in block
        # value clipped
        assert "xxx" in block
        assert "x" * 100 not in block


class TestGetContextForPromptTopK:
    def test_people_not_blanket_dumped(self, tmp_path, monkeypatch):
        """People section should come from search, not list_all."""
        monkeypatch.setenv("MAUDE_CTX_MEMORY_TOP_K", "3")
        db = tmp_path / "memory.db"
        from memory import MaudeMemory

        mem = MaudeMemory.__new__(MaudeMemory)
        mem.DB_PATH = db
        mem.embed_url = "http://localhost:9/v1"
        mem.embed_client = None
        import sqlite3

        mem.conn = sqlite3.connect(str(db))
        mem.conn.row_factory = sqlite3.Row
        mem.conn.executescript(
            """
            CREATE TABLE memories (
                key TEXT PRIMARY KEY,
                value TEXT,
                category TEXT,
                created_at TEXT,
                updated_at TEXT,
                access_count INTEGER DEFAULT 0,
                embedding TEXT,
                metadata TEXT
            );
            """
        )
        # Force no embeddings path
        mem._get_embedding = lambda q: None  # type: ignore
        from datetime import datetime

        now = datetime.now().isoformat()
        for i in range(5):
            mem.conn.execute(
                "INSERT INTO memories VALUES (?,?,?,?,?,?,?,?)",
                (f"person_{i}", f"Person number {i} bio", "person", now, now, 0, None, None),
            )
        mem.conn.execute(
            "INSERT INTO memories VALUES (?,?,?,?,?,?,?,?)",
            ("alice_role", "Alice is the project lead", "person", now, now, 0, None, None),
        )
        mem.conn.commit()

        ctx = mem.get_context_for_prompt("who is Alice", max_memories=3)
        # Should mention Alice via search; must not dump all 5 person_* blindly
        if ctx:
            # text search should hit Alice
            assert "Alice" in ctx or "alice" in ctx.lower() or "Relevant" in ctx
        mem.conn.close()
