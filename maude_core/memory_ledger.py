"""
Human-readable memory ledger for MAUDE.

The ledger is the front-door abstraction for durable context. It stores small,
inspectable files grouped by intent, while the existing SQLite memory remains a
compatibility and search mirror.
"""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_LEDGER_DIR = Path.home() / ".config" / "maude" / "ledger"
LEDGER_CATEGORIES = ("fact", "preference", "person", "task", "project", "mission", "artifact")

_CATEGORY_FILES = {
    "fact": "facts.md",
    "preference": "preferences.md",
    "person": "people.md",
    "task": "tasks.md",
    "project": "projects.md",
    "mission": "missions.md",
    "artifact": "artifacts.md",
}


@dataclass(frozen=True)
class LedgerRecord:
    key: str
    value: str
    category: str
    updated_at: str
    metadata: dict[str, Any] | None = None


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        f.write(content)
        tmp_path = Path(f.name)
    tmp_path.replace(path)


class MemoryLedger:
    """Small file-backed ledger for durable facts, preferences, projects, and artifacts."""

    def __init__(self, root: Path | str = DEFAULT_LEDGER_DIR):
        self.root = Path(root)
        self.records_path = self.root / "records.jsonl"
        self.root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        key: str,
        value: str,
        category: str = "fact",
        metadata: dict[str, Any] | None = None,
    ) -> LedgerRecord:
        key = key.strip()
        value = value.strip()
        category = (category or "fact").strip().lower()
        if not key or not value:
            raise ValueError("key and value are required")
        if category not in LEDGER_CATEGORIES:
            category = "fact"

        record = LedgerRecord(
            key=key,
            value=value,
            category=category,
            updated_at=_now(),
            metadata=metadata or None,
        )
        self._append_record(record)
        self._upsert_markdown(record)
        return record

    def search(self, query: str, category: str | None = None, limit: int = 5) -> list[LedgerRecord]:
        terms = [t for t in re.split(r"\W+", query.lower()) if t]
        if not terms:
            return []

        results: list[tuple[int, LedgerRecord]] = []
        for record in self.records():
            if category and record.category != category:
                continue
            haystack = f"{record.key} {record.value} {record.category}".lower()
            score = sum(3 if record.key.lower() == t else haystack.count(t) for t in terms)
            if score:
                results.append((score, record))

        deduped: dict[tuple[str, str], tuple[int, LedgerRecord]] = {}
        for score, record in results:
            dedupe_key = (record.category, record.key)
            current = deduped.get(dedupe_key)
            if current is None or score >= current[0]:
                deduped[dedupe_key] = (score, record)

        ranked = sorted(deduped.values(), key=lambda item: (item[0], item[1].updated_at), reverse=True)
        return [record for _, record in ranked[:limit]]

    def records(self) -> list[LedgerRecord]:
        if not self.records_path.exists():
            return []
        records: list[LedgerRecord] = []
        for line in self.records_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                records.append(
                    LedgerRecord(
                        key=str(raw["key"]),
                        value=str(raw["value"]),
                        category=str(raw.get("category") or "fact"),
                        updated_at=str(raw.get("updated_at") or ""),
                        metadata=raw.get("metadata"),
                    )
                )
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
        return records

    def latest_records(self, category: str | None = None) -> list[LedgerRecord]:
        latest: dict[tuple[str, str], LedgerRecord] = {}
        for record in self.records():
            if category and record.category != category:
                continue
            latest[(record.category, record.key)] = record
        return sorted(latest.values(), key=lambda record: record.updated_at, reverse=True)

    def forget(self, key: str) -> bool:
        key = key.strip()
        if not key:
            return False

        current = self.records()
        records = [record for record in current if record.key != key]
        removed = len(records) != len(current)
        if removed:
            content = "".join(json.dumps(asdict(record), sort_keys=True) + "\n" for record in records)
            _atomic_write(self.records_path, content)
            self._rewrite_markdown_files(records)
        return removed

    def status(self) -> dict[str, Any]:
        counts = dict.fromkeys(LEDGER_CATEGORIES, 0)
        for record in self.latest_records():
            counts[record.category] = counts.get(record.category, 0) + 1
        return {
            "path": str(self.root),
            "records": sum(counts.values()),
            "categories": {k: v for k, v in counts.items() if v},
            "files": sorted(p.name for p in self.root.glob("*.md")),
        }

    def _append_record(self, record: LedgerRecord) -> None:
        self.records_path.parent.mkdir(parents=True, exist_ok=True)
        with self.records_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    def _upsert_markdown(self, record: LedgerRecord) -> None:
        path = self.root / _CATEGORY_FILES[record.category]
        title = record.category.title() + "s"
        lines = [f"# {title}", ""]
        existing = self._latest_by_key(record.category)
        existing[(record.category, record.key)] = record
        for item in sorted(existing.values(), key=lambda r: r.key.lower()):
            lines.append(f"- **{item.key}**: {item.value}")
        lines.append("")
        _atomic_write(path, "\n".join(lines))

    def _latest_by_key(self, category: str) -> dict[tuple[str, str], LedgerRecord]:
        latest: dict[tuple[str, str], LedgerRecord] = {}
        for record in self.records():
            if record.category == category:
                latest[(record.category, record.key)] = record
        return latest

    def _rewrite_markdown_files(self, records: list[LedgerRecord]) -> None:
        by_category: dict[str, list[LedgerRecord]] = {category: [] for category in LEDGER_CATEGORIES}
        for record in records:
            if record.category in by_category:
                by_category[record.category].append(record)

        for category, items in by_category.items():
            path = self.root / _CATEGORY_FILES[category]
            if not items:
                if path.exists():
                    path.unlink()
                continue
            title = category.title() + "s"
            lines = [f"# {title}", ""]
            latest: dict[str, LedgerRecord] = {}
            for item in items:
                latest[item.key] = item
            for item in sorted(latest.values(), key=lambda r: r.key.lower()):
                lines.append(f"- **{item.key}**: {item.value}")
            lines.append("")
            _atomic_write(path, "\n".join(lines))


_ledger: MemoryLedger | None = None


def get_ledger() -> MemoryLedger:
    global _ledger
    if _ledger is None:
        _ledger = MemoryLedger()
    return _ledger
