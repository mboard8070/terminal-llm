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
    assert not (tmp_path / "facts.md").exists()


def test_ledger_status_counts_categories(tmp_path):
    ledger = MemoryLedger(tmp_path)
    ledger.save("project_brief", "MAUDE memory ledger rollout.", "project")
    ledger.save("artifact_path", "/tmp/report.md", "artifact")

    status = ledger.status()

    assert status["records"] == 2
    assert status["categories"] == {"project": 1, "artifact": 1}
