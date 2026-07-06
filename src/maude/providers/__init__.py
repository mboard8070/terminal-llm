"""Provider abstraction and model routing."""

from .base import ModelCallMetadata, ModelProvider, ModelRequest, ModelResponse, ProviderCapability
from .capability_routing import CapabilityRouter, get_router
from .config import PROVIDERS, Provider, ProviderConfig, get_api_key
from .frontier import FrontierResponse, RateLimitError, ask_frontier, get_default_provider, list_available_providers
from .registry import ProviderRegistry
from .routing import ModelRoute, ModelRouter, ModelRoutingPolicy

__all__ = [
    "PROVIDERS",
    "CapabilityRouter",
    "FrontierResponse",
    "ModelCallMetadata",
    "ModelProvider",
    "ModelRequest",
    "ModelResponse",
    "ModelRoute",
    "ModelRouter",
    "ModelRoutingPolicy",
    "Provider",
    "ProviderCapability",
    "ProviderConfig",
    "ProviderRegistry",
    "RateLimitError",
    "ask_frontier",
    "get_api_key",
    "get_default_provider",
    "get_router",
    "list_available_providers",
]
