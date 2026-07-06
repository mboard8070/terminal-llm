"""Provider-neutral model contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ProviderCapability:
    """Declared provider capability used by the model router."""

    name: str
    provider: str
    model: str
    supports_tools: bool = False
    supports_vision: bool = False
    local: bool = False
    private: bool = False
    healthy: bool = True
    latency_ms: float | None = None
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0


@dataclass(frozen=True)
class ModelCallMetadata:
    """Auditable metadata captured for a model call."""

    provider: str
    model: str
    model_version: str | None = None
    prompt_version: str | None = None
    routing_decision: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    cost_usd: float = 0.0
    run_id: str | None = None
    trace_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "model_version": self.model_version,
            "prompt_version": self.prompt_version,
            "routing_decision": self.routing_decision,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_seconds": self.latency_seconds,
            "cost_usd": self.cost_usd,
            "run_id": self.run_id,
            "trace_id": self.trace_id,
        }


@dataclass(frozen=True)
class ModelRequest:
    """Provider-neutral model request."""

    messages: list[dict[str, Any]]
    task_type: str = "chat"
    model: str | None = None
    requires_tools: bool = False
    requires_vision: bool = False
    prefer_local: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelResponse:
    """Provider-neutral model response."""

    content: str
    model: str
    provider: str
    usage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelProvider(Protocol):
    """Provider interface implemented by local and cloud model adapters."""

    capability: ProviderCapability

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Execute a completion request."""
