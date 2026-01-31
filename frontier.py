"""
Frontier Model Client for MAUDE.

Provides unified access to cloud LLMs: Claude, GPT, Gemini, Grok, Mistral.
"""

import os
from typing import Optional, Tuple
from dataclasses import dataclass

from providers import PROVIDERS, Provider, ProviderConfig, get_api_key


@dataclass
class FrontierResponse:
    """Response from a frontier model."""
    content: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


def get_default_provider() -> Optional[str]:
    """Get first available provider based on configured API keys."""
    # Priority order: Claude > GPT > Gemini > Grok > Mistral
    priority = ["claude", "openai", "gemini", "grok", "mistral"]
    for name in priority:
        if name in PROVIDERS and get_api_key(name):
            return name
    return None


def list_available_providers() -> list[str]:
    """List providers that have API keys configured."""
    available = []
    for name in PROVIDERS:
        if get_api_key(name):
            available.append(name)
    return available


def ask_frontier(
    query: str,
    context: str = None,
    provider_name: str = None,
    system_prompt: str = None
) -> FrontierResponse:
    """
    Send a query to a frontier model.

    Args:
        query: The question or task to send
        context: Optional context (e.g., file contents, conversation history)
        provider_name: Specific provider to use (default: first available)
        system_prompt: Optional system prompt override

    Returns:
        FrontierResponse with content and usage info

    Raises:
        ValueError: If no providers are available
        Exception: On API errors
    """
    # Select provider
    if provider_name:
        if provider_name not in PROVIDERS:
            raise ValueError(f"Unknown provider: {provider_name}")
        if not get_api_key(provider_name):
            raise ValueError(f"No API key configured for {provider_name}")
    else:
        provider_name = get_default_provider()
        if not provider_name:
            raise ValueError("No frontier providers configured. Set API keys in .env")

    config = PROVIDERS[provider_name]
    api_key = get_api_key(provider_name)

    # Build messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Add context + query as user message
    user_content = query
    if context:
        user_content = f"Context:\n{context}\n\nQuestion/Task:\n{query}"
    messages.append({"role": "user", "content": user_content})

    # Route to appropriate client
    if config.provider == Provider.ANTHROPIC:
        return _call_anthropic(config, api_key, messages, system_prompt)
    elif config.provider == Provider.GOOGLE:
        return _call_google(config, api_key, messages, system_prompt)
    else:
        # OpenAI-compatible: OpenAI, xAI, Mistral
        return _call_openai_compatible(config, api_key, messages)


def _call_anthropic(
    config: ProviderConfig,
    api_key: str,
    messages: list,
    system_prompt: str = None
) -> FrontierResponse:
    """Call Anthropic Claude API."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    # Anthropic uses separate system parameter
    system = system_prompt or "You are a helpful expert assistant. Be concise and accurate."

    # Filter out system messages (handled separately)
    user_messages = [m for m in messages if m["role"] != "system"]

    response = client.messages.create(
        model=config.default_model,
        max_tokens=4096,
        system=system,
        messages=user_messages
    )

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost = (input_tokens * config.cost_per_1k_input / 1000 +
            output_tokens * config.cost_per_1k_output / 1000)

    return FrontierResponse(
        content=response.content[0].text,
        provider=config.name,
        model=config.default_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost
    )


def _call_google(
    config: ProviderConfig,
    api_key: str,
    messages: list,
    system_prompt: str = None
) -> FrontierResponse:
    """Call Google Gemini API."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        config.default_model,
        system_instruction=system_prompt or "You are a helpful expert assistant."
    )

    # Convert messages to Gemini format
    gemini_messages = []
    for msg in messages:
        if msg["role"] == "system":
            continue  # Handled via system_instruction
        role = "user" if msg["role"] == "user" else "model"
        gemini_messages.append({"role": role, "parts": [msg["content"]]})

    response = model.generate_content(gemini_messages)

    # Gemini token counting
    input_tokens = response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0
    output_tokens = response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0
    cost = (input_tokens * config.cost_per_1k_input / 1000 +
            output_tokens * config.cost_per_1k_output / 1000)

    return FrontierResponse(
        content=response.text,
        provider=config.name,
        model=config.default_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost
    )


def _call_openai_compatible(
    config: ProviderConfig,
    api_key: str,
    messages: list
) -> FrontierResponse:
    """Call OpenAI-compatible API (OpenAI, xAI, Mistral)."""
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=config.base_url
    )

    response = client.chat.completions.create(
        model=config.default_model,
        messages=messages,
        max_tokens=4096,
        temperature=0.3
    )

    input_tokens = response.usage.prompt_tokens if response.usage else 0
    output_tokens = response.usage.completion_tokens if response.usage else 0
    cost = (input_tokens * config.cost_per_1k_input / 1000 +
            output_tokens * config.cost_per_1k_output / 1000)

    return FrontierResponse(
        content=response.choices[0].message.content,
        provider=config.name,
        model=config.default_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost
    )
