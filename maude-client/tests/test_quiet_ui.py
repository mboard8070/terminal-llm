import os

import maude_client.cli as cli
from maude_client import client_tools


def test_summarize_search_result_is_compact():
    raw = "a.py:1:foo\nb.py:2:foo\nc.py:3:foo\n... and 10 more matches"
    summary = cli._summarize_tool_result("search_files", raw)
    assert "matches" in summary
    assert "a.py" not in summary


def test_format_tool_status_quiet_hides_payload(monkeypatch):
    monkeypatch.setattr(cli, "_VERBOSE_UI", False)
    result = "line1\nline2\nline3\n" * 50
    out = cli._format_tool_status("read_file", {"path": "/tmp/x.py"}, result)
    assert out.startswith("\n[status] read_file")
    assert "line1" not in out
    assert "lines" in out


def test_format_tool_status_verbose_includes_preview(monkeypatch):
    monkeypatch.setattr(cli, "_VERBOSE_UI", True)
    result = "hello world\nsecond line"
    out = cli._format_tool_status("run_command", {"command": "echo hi"}, result)
    assert "[Tool: run_command]" in out
    assert "hello world" in out


def test_format_trace_quiet_hides_tool_result(monkeypatch):
    monkeypatch.setattr(cli, "_VERBOSE_UI", False)
    assert cli._format_trace({"type": "tool_result", "preview": "huge dump", "elapsed": 1}) == ""
    call = cli._format_trace({"type": "tool_call", "name": "search_files", "args": {"pattern": "x"}})
    assert "search_files" in call
    assert "pattern" not in call


def test_format_trace_quiet_prefers_task_label(monkeypatch):
    monkeypatch.setattr(cli, "_VERBOSE_UI", False)
    call = cli._format_trace(
        {
            "type": "tool_call",
            "name": "grok_cli",
            "task": "Running Grok agent",
            "args": "{}",
        }
    )
    assert "Running Grok agent" in call
    assert "grok_cli" not in call


def test_format_trace_quiet_shows_keepalive(monkeypatch):
    monkeypatch.setattr(cli, "_VERBOSE_UI", False)
    line = cli._format_trace({"type": "keepalive", "name": "grok_cli", "elapsed": 32.5})
    assert "grok_cli" in line
    assert "32.5" in line


def test_search_files_bounds_matches(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    for i in range(40):
        (root / f"f{i}.txt").write_text(f"needle-{i}\n", encoding="utf-8")
    out = client_tools.search_files("needle", str(root))
    assert "needle-0" in out
    assert "more matches" in out
    # Should not return all 40 raw lines unbounded
    assert out.count("\n") < 40


def test_read_file_truncates_large_files(tmp_path):
    path = tmp_path / "big.py"
    path.write_text("\n".join(f"line {i}" for i in range(1, 401)) + "\n", encoding="utf-8")
    out = client_tools.read_file(str(path))
    assert "truncated after 200 lines" in out
    assert "line 1" in out
    assert "line 400" not in out
