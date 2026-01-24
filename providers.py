"""
Cloud/Frontier Model Provider Configurations for MAUDE.

Supports: Anthropic (Claude), OpenAI, Google (Gemini), xAI (Grok), Mistral
"""

import os
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional


class Provider(Enum):
    """Supported provider types."""
    LOCAL = "local"        # Ollama / llama.cpp
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    XAI = "xai"
    MISTRAL = "mistral"


@dataclass
class ProviderConfig:
    """Configuration for a model provider."""
    name: str
    provider: Provider
    api_key_env: str          # Environment variable name for API key
    base_url: str
    default_model: str
    supports_vision: bool
    supports_tools: bool
    cost_per_1k_input: float  # USD per 1K input tokens
    cost_per_1k_output: float # USD per 1K output tokens


# Provider configurations
PROVIDERS: Dict[str, ProviderConfig] = {
    # ─────────────────────────────────────────────────────────────────
    # ANTHROPIC (Claude)
    # ─────────────────────────────────────────────────────────────────
    "claude": ProviderConfig(
        name="Claude Sonnet",
        provider=Provider.ANTHROPIC,
        api_key_env="ANTHROPIC_API_KEY",
        base_url="https://api.anthropic.com",
        default_model="claude-sonnet-4-20250514",
        supports_vision=True,
        supports_tools=True,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015
    ),
    "claude-opus": ProviderConfig(
        name="Claude Opus 4.5",
        provider=Provider.ANTHROPIC,
        api_key_env="ANTHROPIC_API_KEY",
        base_url="https://api.anthropic.com",
        default_model="claude-opus-4-5-20251101",
        supports_vision=True,
        supports_tools=True,
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075
    ),

    # ─────────────────────────────────────────────────────────────────
    # OPENAI
    # ─────────────────────────────────────────────────────────────────
    "openai": ProviderConfig(
        name="OpenAI GPT-4o",
        provider=Provider.OPENAI,
        api_key_env="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o",
        supports_vision=True,
        supports_tools=True,
        cost_per_1k_input=0.0025,
        cost_per_1k_output=0.01
    ),
    "openai-o1": ProviderConfig(
        name="OpenAI o1",
        provider=Provider.OPENAI,
        api_key_env="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        default_model="o1",
        supports_vision=True,
        supports_tools=False,  # o1 has limited tool support
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.06
    ),

    # ─────────────────────────────────────────────────────────────────
    # GOOGLE (Gemini)
    # ─────────────────────────────────────────────────────────────────
    "gemini": ProviderConfig(
        name="Google Gemini 2.0 Flash",
        provider=Provider.GOOGLE,
        api_key_env="GOOGLE_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        default_model="gemini-2.0-flash",
        supports_vision=True,
        supports_tools=True,
        cost_per_1k_input=0.0,  # Free tier
        cost_per_1k_output=0.0
    ),
    "gemini-pro": ProviderConfig(
        name="Google Gemini 2.0 Pro",
        provider=Provider.GOOGLE,
        api_key_env="GOOGLE_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        default_model="gemini-2.0-pro-exp",
        supports_vision=True,
        supports_tools=True,
        cost_per_1k_input=0.00125,
        cost_per_1k_output=0.005
    ),

    # ─────────────────────────────────────────────────────────────────
    # XAI (Grok)
    # ─────────────────────────────────────────────────────────────────
    "grok": ProviderConfig(
        name="xAI Grok",
        provider=Provider.XAI,
        api_key_env="XAI_API_KEY",
        base_url="https://api.x.ai/v1",
        default_model="grok-2-latest",
        supports_vision=True,
        supports_tools=True,
        cost_per_1k_input=0.002,
        cost_per_1k_output=0.01
    ),

    # ─────────────────────────────────────────────────────────────────
    # MISTRAL
    # ─────────────────────────────────────────────────────────────────
    "mistral": ProviderConfig(
        name="Mistral Large",
        provider=Provider.MISTRAL,
        api_key_env="MISTRAL_API_KEY",
        base_url="https://api.mistral.ai/v1",
        default_model="mistral-large-latest",
        supports_vision=False,
        supports_tools=True,
        cost_per_1k_input=0.002,
        cost_per_1k_output=0.006
    ),
    "codestral-cloud": ProviderConfig(
        name="Codestral (Cloud)",
        provider=Provider.MISTRAL,
        api_key_env="MISTRAL_API_KEY",
        base_url="https://api.mistral.ai/v1",
        default_model="codestral-latest",
        supports_vision=False,
        supports_tools=True,
        cost_per_1k_input=0.0003,
        cost_per_1k_output=0.0009
    ),
}


def get_available_providers() -> Dict[str, ProviderConfig]:
    """Return providers that have API keys configured."""
    available = {}
    for name, config in PROVIDERS.items():
        if os.environ.get(config.api_key_env):
            available[name] = config
    return available


def get_api_key(provider_name: str) -> Optional[str]:
    """Get API key for a provider, returns None if not set."""
    if provider_name not in PROVIDERS:
        return None
    return os.environ.get(PROVIDERS[provider_name].api_key_env)


def list_providers() -> str:
    """Format provider list for display."""
    lines = ["Available Providers:\n"]
    for name, config in PROVIDERS.items():
        has_key = "✓" if get_api_key(name) else "✗"
        lines.append(f"  [{has_key}] {name:15} - {config.name}")
        lines.append(f"      Model: {config.default_model}")
        if config.cost_per_1k_input > 0:
            lines.append(f"      Cost: ${config.cost_per_1k_input:.4f}/1K in, ${config.cost_per_1k_output:.4f}/1K out")
        else:
            lines.append(f"      Cost: Free tier")
    return "\n".join(lines)
