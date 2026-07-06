"""
Frontier Model Client for MAUDE.

Provides unified access to cloud LLMs: Claude, GPT, Gemini, Grok, Mistral.
"""

import logging
import time
from dataclasses import dataclass

from maude.observability import RunContext, emit_event, record_metric
from maude.orchestration.retries import RetryPolicy, run_with_retries

from .base import ModelCallMetadata
from .config import PROVIDERS, Provider, ProviderConfig, get_api_key

log = logging.getLogger(__name__)

_TRANSIENT_ERROR_MARKERS = (
    "timeout",
    "timed out",
    "connection",
    "connection reset",
    "connection aborted",
    "broken pipe",
    "ssl",
    "temporarily unavailable",
    "too many requests",
    "rate limit",
    "429",
    "500",
    "502",
    "503",
    "504",
)


class RateLimitError(Exception):
    """Raised when a provider's rate limit (e.g. free tier) is reached."""

    def __init__(self, provider: str, message: str, retry_after: int | None = None):
        self.provider = provider
        self.retry_after = retry_after
        super().__init__(message)


@dataclass
class FrontierResponse:
    """Response from a frontier model."""

    content: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    metadata: dict | None = None


def _build_frontier_response(
    *,
    content: str,
    config: ProviderConfig,
    input_tokens: int,
    output_tokens: int,
    latency_seconds: float,
    prompt_version: str | None = None,
    routing_decision: str | None = None,
    run_context: RunContext | None = None,
) -> FrontierResponse:
    cost = input_tokens * config.cost_per_1k_input / 1000 + output_tokens * config.cost_per_1k_output / 1000
    context = run_context or RunContext()
    metadata = ModelCallMetadata(
        provider=config.name,
        model=config.default_model,
        model_version=config.default_model,
        prompt_version=prompt_version,
        routing_decision=routing_decision,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_seconds=round(latency_seconds, 3),
        cost_usd=cost,
        run_id=context.run_id,
        trace_id=context.trace_id,
    )
    record_metric("model.input_tokens", float(input_tokens))
    record_metric("model.output_tokens", float(output_tokens))
    record_metric("model.cost_usd", float(cost))
    record_metric("model.latency_seconds", float(latency_seconds), kind="timing")
    emit_event("model.completed", context, provider=config.name, model=config.default_model, input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=round(cost, 6), latency_seconds=round(latency_seconds, 3))
    return FrontierResponse(
        content=content,
        provider=config.name,
        model=config.default_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
        metadata=metadata.to_dict(),
    )


def get_default_provider() -> str | None:
    """Get first available provider based on configured API keys."""
    # Priority order: Claude > GPT > Gemini > Grok OAuth > Grok API key > Mistral
    priority = ["claude", "openai", "gemini", "grok-oauth", "grok", "mistral"]
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


def _is_transient_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _TRANSIENT_ERROR_MARKERS)


def _with_retries(label: str, call):
    policy = RetryPolicy(max_attempts=3, initial_delay_seconds=1.0, multiplier=2.0)

    def retryable(exc: BaseException) -> bool:
        if isinstance(exc, RateLimitError):
            return False
        if _is_transient_error(exc):
            log.warning("%s transient error, retrying with backoff: %s", label, exc)
            return True
        return False

    return run_with_retries(call, policy, retryable=retryable)


def ask_frontier(
    query: str,
    context: str = None,
    provider_name: str = None,
    system_prompt: str = None,
    prompt_version: str = None,
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
        RateLimitError: If the provider's rate limit is reached
        Exception: On API errors
    """
    # Select provider
    if provider_name:
        if provider_name not in PROVIDERS:
            raise ValueError(f"Unknown provider: {provider_name}")
        if not get_api_key(provider_name):
            if PROVIDERS[provider_name].auth_mode == "xai_oauth":
                raise ValueError(f"No xAI OAuth login configured for {provider_name}. Run xai_oauth_start first.")
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
        return _call_anthropic(
            config, api_key, messages, system_prompt, prompt_version, f"selected provider: {provider_name}"
        )
    elif config.provider == Provider.GOOGLE:
        return _call_google(
            config, api_key, messages, system_prompt, prompt_version, f"selected provider: {provider_name}"
        )
    else:
        # OpenAI-compatible: OpenAI, xAI, Mistral
        return _call_openai_compatible(config, api_key, messages, prompt_version, f"selected provider: {provider_name}")


def _call_anthropic(
    config: ProviderConfig,
    api_key: str,
    messages: list,
    system_prompt: str = None,
    prompt_version: str = None,
    routing_decision: str = None,
) -> FrontierResponse:
    """Call Anthropic Claude API."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    # Anthropic uses separate system parameter
    system = system_prompt or "You are a helpful expert assistant. Be concise and accurate."

    # Filter out system messages (handled separately)
    user_messages = [m for m in messages if m["role"] != "system"]

    start = time.monotonic()
    response = _with_retries(
        config.name,
        lambda: client.messages.create(
            model=config.default_model, max_tokens=4096, system=system, messages=user_messages
        ),
    )

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    return _build_frontier_response(
        content=response.content[0].text,
        config=config,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_seconds=time.monotonic() - start,
        prompt_version=prompt_version,
        routing_decision=routing_decision,
    )


def _call_google(
    config: ProviderConfig,
    api_key: str,
    messages: list,
    system_prompt: str = None,
    prompt_version: str = None,
    routing_decision: str = None,
) -> FrontierResponse:
    """Call Google Gemini API."""
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted, TooManyRequests

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        config.default_model, system_instruction=system_prompt or "You are a helpful expert assistant."
    )

    # Convert messages to Gemini format
    gemini_messages = []
    for msg in messages:
        if msg["role"] == "system":
            continue  # Handled via system_instruction
        role = "user" if msg["role"] == "user" else "model"
        gemini_messages.append({"role": role, "parts": [msg["content"]]})

    start = time.monotonic()
    try:
        response = _with_retries(config.name, lambda: model.generate_content(gemini_messages))
    except (ResourceExhausted, TooManyRequests) as e:
        error_str = str(e)
        # Parse retry-after if present
        retry_after = None
        if "retry" in error_str.lower():
            import re

            match = re.search(r"(\d+)\s*s(?:ec)?", error_str)
            if match:
                retry_after = int(match.group(1))

        raise RateLimitError(
            provider=config.name,
            message=f"Gemini free tier limit reached. {f'Try again in {retry_after}s.' if retry_after else 'Try again in a minute.'}",
            retry_after=retry_after,
        ) from e

    # Gemini token counting
    input_tokens = response.usage_metadata.prompt_token_count if hasattr(response, "usage_metadata") else 0
    output_tokens = response.usage_metadata.candidates_token_count if hasattr(response, "usage_metadata") else 0
    return _build_frontier_response(
        content=response.text,
        config=config,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_seconds=time.monotonic() - start,
        prompt_version=prompt_version,
        routing_decision=routing_decision,
    )


def _call_openai_compatible(
    config: ProviderConfig,
    api_key: str,
    messages: list,
    prompt_version: str = None,
    routing_decision: str = None,
) -> FrontierResponse:
    """Call OpenAI-compatible API (OpenAI, xAI, Mistral)."""
    from openai import OpenAI

    oauth_token_loader = None
    if config.auth_mode == "xai_oauth":
        from maude_core.tools_xai_oauth import get_oauth_access_token

        oauth_token_loader = get_oauth_access_token
        api_key = oauth_token_loader()

    client = OpenAI(api_key=api_key, base_url=config.base_url)

    start = time.monotonic()
    try:
        response = _with_retries(
            config.name,
            lambda: client.chat.completions.create(
                model=config.default_model, messages=messages, max_tokens=4096, temperature=0.3
            ),
        )
    except Exception as exc:
        status_code = getattr(exc, "status_code", None)
        if config.auth_mode != "xai_oauth" or status_code != 401 or oauth_token_loader is None:
            raise
        api_key = oauth_token_loader(force_refresh=True)
        client = OpenAI(api_key=api_key, base_url=config.base_url)
        response = _with_retries(
            config.name,
            lambda: client.chat.completions.create(
                model=config.default_model, messages=messages, max_tokens=4096, temperature=0.3
            ),
        )

    input_tokens = response.usage.prompt_tokens if response.usage else 0
    output_tokens = response.usage.completion_tokens if response.usage else 0
    return _build_frontier_response(
        content=response.choices[0].message.content,
        config=config,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_seconds=time.monotonic() - start,
        prompt_version=prompt_version,
        routing_decision=routing_decision,
    )
