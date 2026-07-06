"""Cache facade for orchestration and tools."""

from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Any


@dataclass(frozen=True)
class CacheEntry:
    value: Any
    expires_at: float | None = None

    def expired(self) -> bool:
        return self.expires_at is not None and time() >= self.expires_at


class CacheStore:
    """Small TTL cache contract used before externalizing to Redis or similar."""

    def __init__(self) -> None:
        self._items: dict[str, CacheEntry] = {}

    def get(self, key: str) -> Any | None:
        entry = self._items.get(key)
        if entry is None:
            return None
        if entry.expired():
            self._items.pop(key, None)
            return None
        return entry.value

    def put(self, key: str, value: Any, ttl_seconds: float | None = None) -> None:
        expires_at = time() + ttl_seconds if ttl_seconds is not None else None
        self._items[key] = CacheEntry(value=value, expires_at=expires_at)
