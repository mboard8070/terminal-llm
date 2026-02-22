"""
Subagent Execution Engine for MAUDE.

Handles routing to local (Ollama), remote (mesh), and cloud (API) providers.
"""

import os
from typing import Optional
from openai import OpenAI
from rich.console import Console

from providers import PROVIDERS, Provider, get_api_key
from subagents import SUBAGENTS, SubAgent, AgentProvider
from cost_tracker import get_tracker

# Lazy imports to avoid circular dependency
_mesh = None
_router = None

def _get_mesh():
    global _mesh
    if _mesh is None:
        try:
            from mesh import get_mesh
            _mesh = get_mesh()
        except ImportError:
            pass
    return _mesh

def _get_router():
    global _router
    if _router is None:
        try:
            from routing import get_router
            _router = get_router()
        except ImportError:
            pass
    return _router

console = Console()


def execute_subagent(
    agent_name: str,
    task: str,
    context: str = None,
    image_base64: str = None,
    prefer_cloud: bool = False
) -> str:
    """
    Execute a task using a specialized subagent.

    Tries providers in order: local first (unless prefer_cloud), then cloud fallback.
    """
    agent = SUBAGENTS.get(agent_name.lower())
    if not agent:
        available = ", ".join(SUBAGENTS.keys())
        return f"Error: Unknown agent '{agent_name}'. Available: {available}"

    # Reorder providers if prefer_cloud
    providers = agent.providers.copy()
    if prefer_cloud:
        # Move cloud providers to front
        cloud = [p for p in providers if p.type == "cloud"]
        local = [p for p in providers if p.type == "local"]
        providers = cloud + local

    # Check cost limit before trying cloud
    tracker = get_tracker()
    cloud_allowed = tracker.check_limit()

    # Try each provider in order
    errors = []
    for provider in providers:
        if provider.type == "cloud" and not cloud_allowed:
            errors.append(f"{provider.provider}: Daily cost limit reached")
            continue

        try:
            if provider.type == "local":
                result = _execute_local(agent, provider, task, context, image_base64)
            else:
                result = _execute_cloud(agent, provider, task, context, image_base64)

            if result and not result.startswith("Error:"):
                return result
            errors.append(result)
        except Exception as e:
            errors.append(f"{provider.model or provider.provider}: {e}")
            continue

    # All providers failed
    error_summary = "\n".join(f"  - {e}" for e in errors)
    return f"Error: All providers for '{agent_name}' failed:\n{error_summary}"


def _execute_local(
    agent: SubAgent,
    provider: AgentProvider,
    task: str,
    context: str = None,
    image_base64: str = None
) -> str:
    """Execute against a local or remote Ollama model."""

    # Get URL from environment or default
    default_url = "http://localhost:11434/v1"
    url = os.environ.get(provider.url_env, default_url) if provider.url_env else default_url
    source = "local"

    # Extract base model name for mesh lookup
    model_base = provider.model.split(":")[0] if ":" in provider.model else provider.model

    # Try local first
    try:
        client = OpenAI(base_url=url, api_key="not-needed")
        # Quick check if model exists locally
        client.models.retrieve(provider.model)
    except Exception:
        # Model not available locally, check mesh for remote nodes via router
        router = _get_router()
        if router:
            remote_url = router.get_ollama_url_for_model(model_base)
            if remote_url:
                url = remote_url
                source = "mesh"
                console.print(f"[dim cyan]  -> Model not local, using mesh node...[/dim cyan]")
            else:
                # Fall back to legacy mesh lookup
                mesh = _get_mesh()
                if mesh:
                    remote_url = mesh.get_remote_ollama_url(model_base)
                    if remote_url:
                        url = remote_url
                        source = "mesh"
                        console.print(f"[dim cyan]  -> Model not local, using mesh node...[/dim cyan]")
                    else:
                        return f"Error: {provider.model}: Not available locally or on mesh"
                else:
                    return f"Error: {provider.model}: Not available locally or on mesh"
        else:
            # No router, fall back to legacy mesh
            mesh = _get_mesh()
            if mesh:
                remote_url = mesh.get_remote_ollama_url(model_base)
                if remote_url:
                    url = remote_url
                    source = "mesh"
                    console.print(f"[dim cyan]  -> Model not local, using mesh node...[/dim cyan]")
                else:
                    return f"Error: {provider.model}: Not available locally or on mesh"
            else:
                return f"Error: {provider.model}: Not available locally"

    console.print(f"[dim cyan]  -> Delegating to {agent.name} ({provider.model})...[/dim cyan]")

    client = OpenAI(base_url=url, api_key="not-needed")

    # Build messages
    messages = [{"role": "system", "content": agent.system_prompt}]

    # Handle vision tasks with images
    if image_base64 and "llava" in provider.model.lower():
        user_content = [
            {"type": "text", "text": _build_user_content(task, context)},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
        ]
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": _build_user_content(task, context)})

    try:
        response = client.chat.completions.create(
            model=provider.model,
            messages=messages,
            temperature=agent.temperature,
            max_tokens=agent.max_tokens
        )
        result = response.choices[0].message.content
        console.print(f"[dim cyan]  -> {agent.name} completed ({source})[/dim cyan]")
        return f"[{agent.name.upper()} - {provider.model}]\n\n{result}"
    except Exception as e:
        return f"Error: {provider.model}: {e}"


def _execute_cloud(
    agent: SubAgent,
    provider: AgentProvider,
    task: str,
    context: str = None,
    image_base64: str = None
) -> str:
    """Execute against a cloud API provider."""

    provider_name = provider.provider
    config = PROVIDERS.get(provider_name)
    if not config:
        return f"Error: Unknown cloud provider '{provider_name}'"

    api_key = get_api_key(provider_name)
    if not api_key:
        return f"Error: No API key for {provider_name}. Use /keys set {provider_name} <key>"

    # Check if this provider supports vision for image tasks
    if image_base64 and not config.supports_vision:
        return f"Error: {provider_name} does not support vision"

    console.print(f"[dim cyan]  -> Delegating to {agent.name} ({config.name})...[/dim cyan]")

    user_content = _build_user_content(task, context)

    try:
        if config.provider == Provider.ANTHROPIC:
            result, usage = _call_anthropic(config, api_key, agent, user_content, image_base64)
        elif config.provider == Provider.OPENAI:
            result, usage = _call_openai(config, api_key, agent, user_content, image_base64)
        elif config.provider == Provider.GOOGLE:
            result, usage = _call_google(config, api_key, agent, user_content, image_base64)
        elif config.provider == Provider.XAI:
            result, usage = _call_xai(config, api_key, agent, user_content, image_base64)
        elif config.provider == Provider.MISTRAL:
            result, usage = _call_mistral(config, api_key, agent, user_content)
        else:
            return f"Error: Provider {config.provider.value} not implemented"

        # Log the API call
        tracker = get_tracker()
        cost = tracker.log_call(
            provider=provider_name,
            model=config.default_model,
            input_tokens=usage.get("input", 0),
            output_tokens=usage.get("output", 0),
            agent_type=agent.name,
            task=task
        )

        console.print(f"[dim cyan]  -> {agent.name} completed (${cost:.4f})[/dim cyan]")
        return f"[{agent.name.upper()} - {config.name}]\n\n{result}"

    except Exception as e:
        return f"Error: {provider_name}: {e}"


def _build_user_content(task: str, context: str = None) -> str:
    """Build the user message content."""
    content = task
    if context:
        content += f"\n\n## Context\n{context}"
    return content


def _call_anthropic(config, api_key, agent, user_content, image_base64=None):
    """Call Anthropic Claude API."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    content = []
    if image_base64:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": image_base64}
        })
    content.append({"type": "text", "text": user_content})

    response = client.messages.create(
        model=config.default_model,
        max_tokens=agent.max_tokens,
        system=agent.system_prompt,
        messages=[{"role": "user", "content": content}]
    )

    usage = {
        "input": response.usage.input_tokens,
        "output": response.usage.output_tokens
    }
    return response.content[0].text, usage


def _call_openai(config, api_key, agent, user_content, image_base64=None):
    """Call OpenAI API."""
    client = OpenAI(api_key=api_key, base_url=config.base_url)

    messages = [{"role": "system", "content": agent.system_prompt}]

    if image_base64:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_content},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
        })
    else:
        messages.append({"role": "user", "content": user_content})

    response = client.chat.completions.create(
        model=config.default_model,
        messages=messages,
        temperature=agent.temperature,
        max_tokens=agent.max_tokens
    )

    usage = {
        "input": response.usage.prompt_tokens,
        "output": response.usage.completion_tokens
    }
    return response.choices[0].message.content, usage


def _call_google(config, api_key, agent, user_content, image_base64=None):
    """Call Google Gemini API."""
    import google.generativeai as genai
    import base64

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        config.default_model,
        system_instruction=agent.system_prompt
    )

    content = [user_content]
    if image_base64:
        image_bytes = base64.b64decode(image_base64)
        content.append({"mime_type": "image/png", "data": image_bytes})

    response = model.generate_content(
        content,
        generation_config={
            "temperature": agent.temperature,
            "max_output_tokens": agent.max_tokens
        }
    )

    # Gemini doesn't always provide token counts
    usage = {"input": 0, "output": 0}
    if hasattr(response, 'usage_metadata'):
        usage["input"] = getattr(response.usage_metadata, 'prompt_token_count', 0)
        usage["output"] = getattr(response.usage_metadata, 'candidates_token_count', 0)

    return response.text, usage


def _call_xai(config, api_key, agent, user_content, image_base64=None):
    """Call xAI Grok API (OpenAI-compatible)."""
    client = OpenAI(api_key=api_key, base_url=config.base_url)

    messages = [{"role": "system", "content": agent.system_prompt}]
    messages.append({"role": "user", "content": user_content})

    response = client.chat.completions.create(
        model=config.default_model,
        messages=messages,
        temperature=agent.temperature,
        max_tokens=agent.max_tokens
    )

    usage = {
        "input": response.usage.prompt_tokens if response.usage else 0,
        "output": response.usage.completion_tokens if response.usage else 0
    }
    return response.choices[0].message.content, usage


def _call_mistral(config, api_key, agent, user_content):
    """Call Mistral API (OpenAI-compatible)."""
    client = OpenAI(api_key=api_key, base_url=config.base_url)

    messages = [{"role": "system", "content": agent.system_prompt}]
    messages.append({"role": "user", "content": user_content})

    response = client.chat.completions.create(
        model=config.default_model,
        messages=messages,
        temperature=agent.temperature,
        max_tokens=agent.max_tokens
    )

    usage = {
        "input": response.usage.prompt_tokens if response.usage else 0,
        "output": response.usage.completion_tokens if response.usage else 0
    }
    return response.choices[0].message.content, usage
