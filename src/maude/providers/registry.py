"""Provider registry."""

from __future__ import annotations

from .base import ModelProvider, ProviderCapability


class ProviderRegistry:
    """Tracks available provider adapters and their declared capabilities."""

    def __init__(self) -> None:
        self._providers: dict[str, ModelProvider] = {}

    def register(self, provider: ModelProvider) -> None:
        self._providers[provider.capability.name] = provider

    def get(self, name: str) -> ModelProvider | None:
        return self._providers.get(name)

    def list(self) -> list[ProviderCapability]:
        return [provider.capability for provider in self._providers.values()]
