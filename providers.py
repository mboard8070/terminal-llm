"""
Cloud/Frontier Model Provider Configurations for MAUDE.

Supports: Anthropic (Claude), OpenAI, Google (Gemini), xAI (Grok), Mistral
"""

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Provider(Enum):
    """Supported provider types."""

    LOCAL = "local"  # Ollama / llama.cpp
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
    api_key_env: str  # Environment variable name for API key
    base_url: str
    default_model: str
    supports_vision: bool
    supports_tools: bool
    cost_per_1k_input: float  # USD per 1K input tokens
    cost_per_1k_output: float  # USD per 1K output tokens
    auth_mode: str = "api_key"


# Provider configurations
PROVIDERS: dict[str, ProviderConfig] = {
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
        cost_per_1k_output=0.015,
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
        cost_per_1k_output=0.075,
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
        cost_per_1k_output=0.01,
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
        cost_per_1k_output=0.06,
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
        cost_per_1k_output=0.0,
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
        cost_per_1k_output=0.005,
    ),
    # ─────────────────────────────────────────────────────────────────
    # XAI (Grok)
    # ─────────────────────────────────────────────────────────────────
    "grok": ProviderConfig(
        name="xAI Grok",
        provider=Provider.XAI,
        api_key_env="XAI_API_KEY",
        base_url="https://api.x.ai/v1",
        default_model="grok-4.3",
        supports_vision=True,
        supports_tools=True,
        cost_per_1k_input=0.002,
        cost_per_1k_output=0.01,
    ),
    "grok-oauth": ProviderConfig(
        name="xAI Grok OAuth",
        provider=Provider.XAI,
        api_key_env="",
        base_url="https://api.x.ai/v1",
        default_model="grok-4.3",
        supports_vision=True,
        supports_tools=True,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        auth_mode="xai_oauth",
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
        cost_per_1k_output=0.006,
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
        cost_per_1k_output=0.0009,
    ),
    "devstral": ProviderConfig(
        name="Devstral 2 (123B)",
        provider=Provider.MISTRAL,
        api_key_env="MISTRAL_API_KEY",
        base_url="https://api.mistral.ai/v1",
        default_model="devstral-2512",
        supports_vision=False,
        supports_tools=True,
        cost_per_1k_input=0.0004,
        cost_per_1k_output=0.002,
    ),
    "devstral-small": ProviderConfig(
        name="Devstral Small",
        provider=Provider.MISTRAL,
        api_key_env="MISTRAL_API_KEY",
        base_url="https://api.mistral.ai/v1",
        default_model="devstral-small-latest",
        supports_vision=False,
        supports_tools=True,
        cost_per_1k_input=0.0001,
        cost_per_1k_output=0.0003,
    ),
    "devstral-medium": ProviderConfig(
        name="Devstral Medium",
        provider=Provider.MISTRAL,
        api_key_env="MISTRAL_API_KEY",
        base_url="https://api.mistral.ai/v1",
        default_model="devstral-medium-latest",
        supports_vision=False,
        supports_tools=True,
        cost_per_1k_input=0.0002,
        cost_per_1k_output=0.001,
    ),
}


def get_api_key(provider_name: str) -> str | None:
    """Get API key for a provider, returns None if not set."""
    if provider_name not in PROVIDERS:
        return None
    config = PROVIDERS[provider_name]
    if config.auth_mode == "xai_oauth":
        path = Path.home() / ".config" / "maude" / "xai_oauth.json"
        try:
            state = json.loads(path.read_text())
            token = str((state.get("tokens") or {}).get("access_token") or "").strip()
            return "oauth" if token else None
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
    return os.environ.get(config.api_key_env)
