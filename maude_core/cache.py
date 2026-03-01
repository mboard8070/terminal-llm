"""
Tool result cache — TTL-based cache for expensive tool results.
"""

import json
import time
from typing import Dict, Optional


class ToolCache:
    """TTL-based cache for expensive tool results (web, vision)."""

    # Per-tool TTLs in seconds
    TTLS = {
        "web_browse": 1800,   # 30 min
        "web_search": 1800,   # 30 min
        "web_view":   1800,   # 30 min
        "view_image": 300,    # 5 min
    }

    def __init__(self):
        self._store: Dict[tuple, tuple] = {}  # (tool, args_key) -> (result, expiry)

    @staticmethod
    def _make_key(tool_name: str, arguments: dict) -> tuple:
        return (tool_name, json.dumps(arguments, sort_keys=True))

    def get(self, tool_name: str, arguments: dict) -> Optional[str]:
        """Return cached result or None."""
        key = self._make_key(tool_name, arguments)
        entry = self._store.get(key)
        if entry is None:
            return None
        result, expiry = entry
        if time.time() > expiry:
            del self._store[key]
            return None
        return result

    def put(self, tool_name: str, arguments: dict, result: str):
        """Store a result with tool-specific TTL."""
        ttl = self.TTLS.get(tool_name)
        if ttl is None:
            return  # not a cacheable tool
        key = self._make_key(tool_name, arguments)
        self._store[key] = (result, time.time() + ttl)

    def evict_expired(self):
        """Remove all expired entries."""
        now = time.time()
        expired = [k for k, (_, exp) in self._store.items() if now > exp]
        for k in expired:
            del self._store[k]

    def clear(self):
        self._store.clear()


_tool_cache = ToolCache()
