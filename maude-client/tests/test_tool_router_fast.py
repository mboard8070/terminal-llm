"""Client tool-router fast paths (list/read/shell/memory/image/URL)."""

from maude_client.tool_router import ToolRouter, _client_shell_cmd, _MULTI_STEP_RE


def test_shell_status_whitelist():
    assert _client_shell_cmd("git status") == "git status"
    assert _client_shell_cmd("please docker ps") == "docker ps"
    assert _client_shell_cmd("run pytest") == "pytest"
    assert _client_shell_cmd("rm -rf /") is None
    assert _client_shell_cmd("git push origin main") is None


def test_multi_step_guard_pattern():
    assert _MULTI_STEP_RE.search("list files and then delete them")
    assert _MULTI_STEP_RE.search("fix the bug")
    assert not _MULTI_STEP_RE.search("git status")


def test_fast_dispatch_list_directory(monkeypatch):
    router = ToolRouter(gateway_url="http://localhost:9")
    called = {}

    def fake_execute(name, args):
        called["name"] = name
        called["args"] = args
        return "file1\nfile2"

    monkeypatch.setattr(router, "execute", fake_execute)
    hit = router.fast_dispatch("list files in /tmp")
    assert hit is not None
    name, args, result = hit
    assert name == "list_directory"
    assert result == "file1\nfile2"
    assert called["name"] == "list_directory"


def test_fast_dispatch_skips_multi_step(monkeypatch):
    router = ToolRouter(gateway_url="http://localhost:9")
    monkeypatch.setattr(router, "execute", lambda *a, **k: "should not run")
    assert router.fast_dispatch("list files and then delete them") is None


def test_fast_dispatch_memory_recall(monkeypatch):
    router = ToolRouter(gateway_url="http://localhost:9")

    def fake_execute(name, args):
        return f"memory about {args.get('query')}"

    monkeypatch.setattr(router, "execute", fake_execute)
    hit = router.fast_dispatch("what do you know about Pixelus")
    assert hit is not None
    name, args, result = hit
    assert name == "recall_memory"
    assert "Pixelus" in args["query"]
    assert "Pixelus" in result
