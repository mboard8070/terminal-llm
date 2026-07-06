"""Integration adapter base contracts with retries and normalized errors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any

from maude.orchestration.retries import RetryPolicy, run_with_retries


@dataclass(frozen=True)
class IntegrationError(Exception):
    integration: str
    message: str
    transient: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.integration}: {self.message}"


class IntegrationAdapter:
    """Base class for external integration adapters."""

    name: str = "integration"

    def __init__(self, retry_policy: RetryPolicy | None = None) -> None:
        self.retry_policy = retry_policy or RetryPolicy()

    def run_with_retries[T](self, operation: Callable[[], T]) -> T:
        return run_with_retries(operation, self.retry_policy, retryable=self._is_retryable)

    def audit_event(self, action: str, **metadata: Any) -> dict[str, Any]:
        return {"integration": self.name, "action": action, **metadata}

    def refresh_token(self, force: bool = False) -> str | None:
        return None

    def normalize_error(self, exc: BaseException) -> IntegrationError:
        if isinstance(exc, IntegrationError):
            return exc
        text = str(exc)
        transient = self._looks_transient(text, getattr(exc, "status_code", None))
        return IntegrationError(self.name, text, transient=transient)

    def _is_retryable(self, exc: BaseException) -> bool:
        return self.normalize_error(exc).transient

    @staticmethod
    def _looks_transient(text: str, status_code: int | None = None) -> bool:
        if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
            return True
        lowered = text.lower()
        markers = ("timeout", "connection", "temporarily unavailable", "rate limit", "429", "502", "503", "504")
        return any(marker in lowered for marker in markers)


class TestDoubleAdapter(IntegrationAdapter):
    """Deterministic adapter for integration tests."""

    def __init__(self, responses: Mapping[str, Any] | None = None, retry_policy: RetryPolicy | None = None) -> None:
        super().__init__(retry_policy=retry_policy)
        self.responses = dict(responses or {})
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call(self, operation: str, **kwargs: Any) -> Any:
        self.calls.append((operation, kwargs))
        value = self.responses.get(operation)
        if isinstance(value, BaseException):
            raise value
        return value
