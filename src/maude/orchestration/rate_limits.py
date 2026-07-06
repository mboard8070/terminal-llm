"""Rate limiting contracts."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import time


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    remaining: int
    retry_after_seconds: float = 0.0


class SlidingWindowRateLimiter:
    """In-process sliding-window limiter for clients, tools, and provider calls."""

    def __init__(self, limit: int, window_seconds: float) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = {}

    def check(self, key: str) -> RateLimitDecision:
        now = time()
        hits = self._hits.setdefault(key, deque())
        while hits and now - hits[0] > self.window_seconds:
            hits.popleft()
        if len(hits) >= self.limit:
            retry_after = self.window_seconds - (now - hits[0]) if hits else self.window_seconds
            return RateLimitDecision(False, 0, max(retry_after, 0.0))
        hits.append(now)
        return RateLimitDecision(True, self.limit - len(hits), 0.0)
