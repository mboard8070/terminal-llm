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
from enum import StrEnum
from pathlib import Path
from typing import Any

from maude.config import runtime_paths


class MemoryType(StrEnum):
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    WORKING = "working"
    PREFERENCE = "preference"
    IDENTITY = "identity"
    PERSON = "person"
    PROJECT = "project"
    MISSION = "mission"
    ARTIFACT = "artifact"


DEFAULT_LEDGER_DIR = runtime_paths().config_dir / "ledger"
LEDGER_CATEGORIES = (
    "semantic",
    "episodic",
    "procedural",
    "working",
    "preference",
    "identity",
    "person",
    "project",
    "mission",
    "artifact",
    # Compatibility categories from the original memory model.
    "fact",
    "task",
    "conversation",
)

_CATEGORY_FILES = {
    "semantic": "semantic.md",
    "episodic": "episodes.md",
    "procedural": "procedures.md",
    "working": "working.md",
    "fact": "facts.md",
    "preference": "preferences.md",
    "identity": "identity.md",
    "person": "people.md",
    "task": "tasks.md",
    "project": "projects.md",
    "mission": "missions.md",
    "artifact": "artifacts.md",
    "conversation": "conversations.md",
}

MEMORY_TYPE_DESCRIPTIONS = {
    "semantic": "Durable facts and domain knowledge that should remain true across sessions.",
    "episodic": "Time-bound events, decisions, incidents, meetings, and project history.",
    "procedural": "Reusable procedures, workflows, checklists, and how-to knowledge.",
    "working": "Short-lived task context that is useful now but should not become permanent identity or fact.",
    "preference": "User preferences, style rules, defaults, and standing instructions.",
    "identity": "Stable user, project, product, or system identity information.",
    "person": "Information about people, relationships, collaborators, and contacts.",
    "project": "Project-level goals, constraints, state, and decisions.",
    "mission": "Mission/workflow state and operational progress.",
    "artifact": "Important files, links, outputs, and generated deliverables.",
    "fact": "Legacy compatibility alias for semantic memory.",
    "task": "Legacy compatibility alias for working memory.",
    "conversation": "Legacy compatibility category for durable conversation notes.",
}

MEMORY_CATEGORY_ALIASES = {
    "knowledge": "semantic",
    "fact": "semantic",
    "facts": "semantic",
    "event": "episodic",
    "episode": "episodic",
    "history": "episodic",
    "procedure": "procedural",
    "process": "procedural",
    "workflow": "procedural",
    "howto": "procedural",
    "how-to": "procedural",
    "short_term": "working",
    "short-term": "working",
    "task": "working",
    "scratch": "working",
    "prefs": "preference",
    "preferences": "preference",
    "profile": "identity",
}


def normalize_memory_category(category: str | None) -> str:
    """Normalize user/tool-facing category names into supported memory types."""
    raw = (category or "semantic").strip().lower().replace(" ", "_")
    normalized = MEMORY_CATEGORY_ALIASES.get(raw, raw)
    return normalized if normalized in LEDGER_CATEGORIES else "semantic"


@dataclass(frozen=True)
class LedgerRecord:
    key: str
    value: str
    category: str
    updated_at: str
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class LedgerVerification:
    verified: bool
    evidence: dict[str, Any]
    reason: str


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
        category: str = "semantic",
        metadata: dict[str, Any] | None = None,
    ) -> LedgerRecord:
        key = key.strip()
        value = value.strip()
        category = normalize_memory_category(category)
        if not key or not value:
            raise ValueError("key and value are required")

        record = LedgerRecord(
            key=key,
            value=value,
            category=category,
            updated_at=_now(),
            metadata=metadata or None,
        )
        self._append_record(record)
        self._upsert_markdown(record)
        verification = self.verify_record(record)
        if not verification.verified:
            raise RuntimeError(f"memory ledger verification failed: {verification.reason}")
        return record

    def verify_record(self, record: LedgerRecord) -> LedgerVerification:
        """Verify that a saved record is present in JSONL and markdown projections."""
        records = self.records()
        jsonl_verified = any(
            item.key == record.key
            and item.value == record.value
            and item.category == record.category
            and item.updated_at == record.updated_at
            for item in records
        )

        markdown_path = self.root / _CATEGORY_FILES[record.category]
        markdown_text = markdown_path.read_text(encoding="utf-8") if markdown_path.exists() else ""
        markdown_verified = f"**{record.key}**" in markdown_text and record.value in markdown_text

        evidence = {
            "records_path": str(self.records_path),
            "records_path_exists": self.records_path.exists(),
            "jsonl_verified": jsonl_verified,
            "markdown_path": str(markdown_path),
            "markdown_path_exists": markdown_path.exists(),
            "markdown_verified": markdown_verified,
        }
        verified = jsonl_verified and markdown_verified
        reason = (
            "record verified in jsonl and markdown" if verified else "record missing from jsonl or markdown projection"
        )
        return LedgerVerification(verified=verified, evidence=evidence, reason=reason)

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
            "verification_gate": "jsonl_and_markdown_write_through",
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


def open_ledger(path: str | Path) -> MemoryLedger:
    """Open a typed memory ledger at the provided path."""

    return MemoryLedger(Path(path))
