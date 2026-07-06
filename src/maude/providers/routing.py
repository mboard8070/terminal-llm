"""Model routing policies."""

from __future__ import annotations

from dataclasses import dataclass, field

from maude.observability import RunContext, emit_event, record_metric

from .base import ModelRequest, ProviderCapability
from .registry import ProviderRegistry


@dataclass(frozen=True)
class ModelRoute:
    """Selected provider/model route."""

    provider_name: str
    provider: str
    model: str
    reason: str
    fallback_provider_names: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModelRoutingPolicy:
    """Policy controls for model selection."""

    aliases: dict[str, str] = field(default_factory=dict)
    max_cost_per_1k: float | None = None
    max_latency_ms: float | None = None
    require_private: bool = False
    local_only: bool = False
    allow_unhealthy: bool = False
    fallback_enabled: bool = True


class ModelRouter:
    """Selects a provider based on request requirements and declared capabilities."""

    def __init__(self, registry: ProviderRegistry, policy: ModelRoutingPolicy | None = None) -> None:
        self.registry = registry
        self.policy = policy or ModelRoutingPolicy()

    def route(self, request: ModelRequest) -> ModelRoute | None:
        candidates = self._eligible(request)
        if not candidates:
            return None

        requested_model = self._resolve_alias(request.model)
        if requested_model:
            for candidate in candidates:
                if requested_model in (candidate.model, candidate.name):
                    return self._route(candidate, f"matched requested model: {request.model}", candidates)

        if request.prefer_local:
            for candidate in candidates:
                if candidate.local:
                    return self._route(candidate, "matched local preference", candidates)

        candidates.sort(
            key=lambda item: (item.cost_per_1k_input + item.cost_per_1k_output, item.latency_ms or 0, item.name)
        )
        return self._route(candidates[0], "selected lowest declared cost eligible model", candidates)

    def _eligible(self, request: ModelRequest) -> list[ProviderCapability]:
        candidates = []
        for capability in self.registry.list():
            if request.requires_tools and not capability.supports_tools:
                continue
            if request.requires_vision and not capability.supports_vision:
                continue
            if not self.policy.allow_unhealthy and not capability.healthy:
                continue
            if (self.policy.local_only or request.metadata.get("local_only")) and not capability.local:
                continue
            if (self.policy.require_private or request.metadata.get("require_private")) and not capability.private:
                continue
            max_cost = request.metadata.get("max_cost_per_1k", self.policy.max_cost_per_1k)
            if max_cost is not None and capability.cost_per_1k_input + capability.cost_per_1k_output > float(max_cost):
                continue
            max_latency = request.metadata.get("max_latency_ms", self.policy.max_latency_ms)
            if (
                max_latency is not None
                and capability.latency_ms is not None
                and capability.latency_ms > float(max_latency)
            ):
                continue
            candidates.append(capability)
        return candidates

    def _resolve_alias(self, model: str | None) -> str | None:
        if model is None:
            return None
        return self.policy.aliases.get(model, model)

    def _route(self, capability: ProviderCapability, reason: str, candidates: list[ProviderCapability]) -> ModelRoute:
        fallback_provider_names = []
        if self.policy.fallback_enabled:
            fallback_provider_names = [item.name for item in candidates if item.name != capability.name]
        context = RunContext.from_mapping({k: v for k, v in getattr(capability, "metadata", {}).items() if k in {"run_id", "trace_id"}} if hasattr(capability, "metadata") else None)
        record_metric("model.routing_decisions")
        emit_event("model.routed", context, provider_name=capability.name, provider=capability.provider, model=capability.model, reason=reason)
        return ModelRoute(
            provider_name=capability.name,
            provider=capability.provider,
            model=capability.model,
            reason=reason,
            fallback_provider_names=fallback_provider_names,
        )
