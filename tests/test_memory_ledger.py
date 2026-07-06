from maude_core.memory_ledger import MemoryLedger


def test_ledger_save_writes_jsonl_and_markdown(tmp_path):
    ledger = MemoryLedger(tmp_path)

    ledger.save("nightly_reports", "Send nightly MAUDE reports to Telegram.", "preference")

    assert (tmp_path / "records.jsonl").exists()
    prefs = (tmp_path / "preferences.md").read_text(encoding="utf-8")
    assert "**nightly_reports**" in prefs
    assert "Telegram" in prefs


def test_ledger_search_dedupes_latest_record(tmp_path):
    ledger = MemoryLedger(tmp_path)

    ledger.save("memory_model", "Use MemPalace as the main abstraction.", "fact")
    ledger.save("memory_model", "Use the ledger as the front-door abstraction.", "fact")

    results = ledger.search("memory ledger")

    assert len(results) == 1
    assert results[0].key == "memory_model"
    assert "front-door" in results[0].value


def test_ledger_forget_removes_record_and_markdown(tmp_path):
    ledger = MemoryLedger(tmp_path)
    ledger.save("old_fact", "Remove me.", "fact")

    assert ledger.forget("old_fact") is True

    assert ledger.records() == []
    assert not (tmp_path / "semantic.md").exists()


def test_ledger_status_counts_categories(tmp_path):
    ledger = MemoryLedger(tmp_path)
    ledger.save("project_brief", "MAUDE memory ledger rollout.", "project")
    ledger.save("artifact_path", "/tmp/report.md", "artifact")

    status = ledger.status()

    assert status["records"] == 2
    assert status["categories"] == {"project": 1, "artifact": 1}


def test_ledger_supports_explicit_memory_types(tmp_path):
    ledger = MemoryLedger(tmp_path)

    ledger.save("release_gate", "Run golden evals before release.", "procedural")
    ledger.save("purdue_call", "Discussed SACC architecture framing.", "episodic")
    ledger.save("current_focus", "Refactor memory into typed stores.", "working")

    assert (tmp_path / "procedures.md").exists()
    assert (tmp_path / "episodes.md").exists()
    assert (tmp_path / "working.md").exists()

    procedural = ledger.search("golden evals", category="procedural")
    assert procedural[0].category == "procedural"
    assert procedural[0].key == "release_gate"


def test_ledger_normalizes_legacy_categories_to_memory_types(tmp_path):
    ledger = MemoryLedger(tmp_path)

    fact = ledger.save("model_gateway", "Provider routing is part of the model layer.", "fact")
    task = ledger.save("next_step", "Add observability events.", "task")

    assert fact.category == "semantic"
    assert task.category == "working"
    assert (tmp_path / "semantic.md").exists()
    assert (tmp_path / "working.md").exists()


def test_ledger_status_includes_new_memory_type_counts(tmp_path):
    ledger = MemoryLedger(tmp_path)
    ledger.save("user_style", "Direct and pragmatic.", "preference")
    ledger.save("tool_contract", "Mutating tools must verify side effects.", "semantic")

    status = ledger.status()

    assert status["records"] == 2
    assert status["categories"] == {"semantic": 1, "preference": 1}
    assert status["verification_gate"] == "jsonl_and_markdown_write_through"


def test_ledger_verifies_saved_record_in_jsonl_and_markdown(tmp_path):
    ledger = MemoryLedger(tmp_path)
    record = ledger.save("verify_memory", "Saved memories must prove persistence.", "semantic")

    verification = ledger.verify_record(record)

    assert verification.verified is True
    assert verification.evidence["jsonl_verified"] is True
    assert verification.evidence["markdown_verified"] is True
