"""Gateway governance contracts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RequestContext:
    """Authenticated request metadata attached at the gateway boundary."""

    user_id: str
    client: str
    scopes: set[str] = field(default_factory=set)


class CapabilityPolicy:
    """Scope-based access checks for models, tools, and files."""

    def allows(self, context: RequestContext, capability: str) -> bool:
        return "*" in context.scopes or capability in context.scopes
