"""Retry and backoff policy for orchestration and provider calls."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential backoff configuration."""

    max_attempts: int = 3
    initial_delay_seconds: float = 0.5
    max_delay_seconds: float = 8.0
    multiplier: float = 2.0

    def delay_for_attempt(self, attempt: int) -> float:
        """Return the delay before a retry attempt. Attempt values start at 1."""
        if attempt <= 1:
            return 0.0
        delay = self.initial_delay_seconds * (self.multiplier ** (attempt - 2))
        return min(delay, self.max_delay_seconds)

    def should_retry(self, attempt: int, exc: BaseException | None = None) -> bool:
        """Return whether another attempt is allowed."""
        return attempt < self.max_attempts


def run_with_retries[T](
    operation: Callable[[], T],
    policy: RetryPolicy | None = None,
    sleep: Callable[[float], None] = time.sleep,
    retryable: Callable[[BaseException], bool] | None = None,
) -> T:
    """Run an operation under a bounded retry/backoff policy."""

    active_policy = policy or RetryPolicy()
    attempt = 0
    while True:
        attempt += 1
        if delay := active_policy.delay_for_attempt(attempt):
            sleep(delay)
        try:
            return operation()
        except Exception as exc:
            if retryable is not None and not retryable(exc):
                raise
            if not active_policy.should_retry(attempt, exc):
                raise
