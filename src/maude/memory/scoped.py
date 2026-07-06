"""Scoped memory access facade."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .ledger import LedgerRecord, MemoryLedger, normalize_memory_category


@dataclass(frozen=True)
class MemoryScope:
    user_id: str | None = None
    project_id: str | None = None
    workspace_id: str | None = None

    def prefix(self) -> str:
        parts = [
            f"user={self.user_id or '*'}",
            f"project={self.project_id or '*'}",
            f"workspace={self.workspace_id or '*'}",
        ]
        return "|".join(parts)

    def scoped_key(self, key: str) -> str:
        return f"{self.prefix()}::{key}"

    def can_access(self, record: LedgerRecord) -> bool:
        prefix, _, _key = record.key.partition("::")
        if not _:
            return True
        values = dict(part.split("=", 1) for part in prefix.split("|") if "=" in part)
        return (
            self._matches(values.get("user"), self.user_id)
            and self._matches(values.get("project"), self.project_id)
            and self._matches(values.get("workspace"), self.workspace_id)
        )

    @staticmethod
    def _matches(record_value: str | None, requested: str | None) -> bool:
        return record_value in {None, "*", requested}


@dataclass(frozen=True)
class MemoryProvenance:
    source: str = "manual"
    actor: str | None = None
    run_id: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).replace(microsecond=0).isoformat())

    def to_prefix(self) -> str:
        parts = [f"source={self.source}", f"created_at={self.created_at}"]
        if self.actor:
            parts.append(f"actor={self.actor}")
        if self.run_id:
            parts.append(f"run_id={self.run_id}")
        return "[provenance " + " ".join(parts) + "]"


@dataclass(frozen=True)
class MemoryExport:
    scope: MemoryScope
    records: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"scope": self.scope.prefix(), "records": self.records}


class ScopedMemoryStore:
    def __init__(self, ledger: MemoryLedger, scope: MemoryScope) -> None:
        self.ledger = ledger
        self.scope = scope

    @classmethod
    def open(cls, root: str | Path, scope: MemoryScope) -> ScopedMemoryStore:
        return cls(MemoryLedger(root), scope)

    def save(
        self,
        key: str,
        value: str,
        category: str = "semantic",
        provenance: MemoryProvenance | None = None,
    ) -> LedgerRecord:
        value_with_provenance = value
        if provenance is not None:
            value_with_provenance = f"{provenance.to_prefix()} {value}"
        return self.ledger.save(self.scope.scoped_key(key), value_with_provenance, normalize_memory_category(category))

    def search(self, query: str, category: str | None = None, limit: int = 5) -> list[LedgerRecord]:
        records = self.ledger.search(query, category=category, limit=limit * 3)
        return [record for record in records if self.scope.can_access(record)][:limit]

    def latest_records(self, category: str | None = None, limit: int = 20) -> list[LedgerRecord]:
        records = self.ledger.latest_records(category=category)
        return [record for record in records if self.scope.can_access(record)][:limit]

    def delete(self, key: str) -> bool:
        return self.ledger.forget(self.scope.scoped_key(key))

    def export(self, category: str | None = None) -> MemoryExport:
        records = [
            {
                "key": record.key,
                "value": record.value,
                "category": record.category,
                "updated_at": record.updated_at,
            }
            for record in self.latest_records(category=category, limit=10_000)
        ]
        return MemoryExport(scope=self.scope, records=records)

    def prune_older_than(self, days: int, category: str | None = None) -> int:
        cutoff = datetime.now(UTC) - timedelta(days=days)
        removed = 0
        for record in self.latest_records(category=category, limit=10_000):
            try:
                updated_at = datetime.fromisoformat(record.updated_at)
            except ValueError:
                continue
            if updated_at < cutoff and self.ledger.forget(record.key):
                removed += 1
        return removed
