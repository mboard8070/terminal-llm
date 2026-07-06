"""Adapters over the existing top-level provider configuration."""

from __future__ import annotations

from .base import ProviderCapability


def load_legacy_provider_capabilities() -> list[ProviderCapability]:
    """Expose existing providers.py configuration through the new provider contract."""
    from .config import PROVIDERS, Provider

    capabilities: list[ProviderCapability] = []
    for key, config in PROVIDERS.items():
        capabilities.append(
            ProviderCapability(
                name=key,
                provider=config.provider.value if isinstance(config.provider, Provider) else str(config.provider),
                model=config.default_model,
                supports_tools=config.supports_tools,
                supports_vision=config.supports_vision,
                local=config.provider == Provider.LOCAL,
                cost_per_1k_input=config.cost_per_1k_input,
                cost_per_1k_output=config.cost_per_1k_output,
            )
        )
    return capabilities
