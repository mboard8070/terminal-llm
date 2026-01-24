# MAUDE Subagent Architecture

## Overview

Transform MAUDE from a single-model system into a multi-agent orchestrator where the primary Nemotron model delegates specialized tasks to subagents running locally (Ollama), on remote machines (Tailscale), or via cloud APIs (Claude, OpenAI, Gemini, Grok).

```
                           ┌─────────────────────────────────────────┐
                           │              MAUDE CORE                 │
                           │         (Nemotron Orchestrator)         │
                           └─────────────┬───────────────────────────┘
                                         │
          ┌──────────────────────────────┼──────────────────────────────┐
          │                              │                              │
    ┌─────▼─────┐                 ┌──────▼──────┐                ┌──────▼──────┐
    │   LOCAL   │                 │   REMOTE    │                │    CLOUD    │
    │  (Ollama) │                 │ (Tailscale) │                │   (APIs)    │
    └─────┬─────┘                 └──────┬──────┘                └──────┬──────┘
          │                              │                              │
    ┌─────┴─────┐                 ┌──────┴──────┐                ┌──────┴──────┐
    │ codestral │                 │other MAUDEs │                │   claude    │
    │ llava     │                 │on network   │                │   openai    │
    │ gemma     │                 │             │                │   gemini    │
    │ nomic     │                 │             │                │   grok      │
    └───────────┘                 └─────────────┘                └─────────────┘
```

## Subagent Registry

Define subagents in a configuration that maps capabilities to models:

```python
# subagents.py

SUBAGENTS = {
    "code": {
        "model": "codestral:latest",
        "url": os.environ.get("CODE_AGENT_URL", "http://localhost:11434/v1"),
        "description": "Code generation, completion, refactoring, debugging",
        "triggers": ["write code", "implement", "refactor", "debug", "fix bug", "generate function"],
        "system_prompt": "You are a code generation specialist. Write clean, efficient, well-documented code.",
        "temperature": 0.1,
        "max_tokens": 4096
    },
    "vision": {
        "model": "llava:13b",  # upgrade from 7b
        "url": os.environ.get("VISION_AGENT_URL", "http://localhost:11434/v1"),
        "description": "Image analysis, screenshots, visual understanding",
        "triggers": ["image", "screenshot", "visual", "see", "look at", "diagram"],
        "system_prompt": "You are a vision specialist. Describe images in detail, noting text, UI elements, and visual patterns.",
        "temperature": 0.2,
        "max_tokens": 1024
    },
    "writer": {
        "model": "gemma-writer:latest",
        "url": os.environ.get("WRITER_AGENT_URL", "http://localhost:11434/v1"),
        "description": "Long-form writing, documentation, explanations",
        "triggers": ["write documentation", "explain in detail", "write readme", "draft email", "summarize"],
        "system_prompt": "You are a technical writer. Create clear, well-structured documentation and explanations.",
        "temperature": 0.7,
        "max_tokens": 8192
    },
    "embeddings": {
        "model": "nomic-embed-text:latest",
        "url": os.environ.get("EMBED_AGENT_URL", "http://localhost:11434/v1"),
        "description": "Text embeddings for semantic search and similarity",
        "triggers": [],  # Called programmatically, not by user request
        "is_embedding_model": True
    }
}
```

## Cloud/Frontier Model Providers

Support for commercial APIs when you need maximum capability or specific features:

```python
# providers.py - Cloud API provider configurations

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
import os

class Provider(Enum):
    LOCAL = "local"       # Ollama / llama.cpp
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

PROVIDERS: Dict[str, ProviderConfig] = {
    # ─────────────────────────────────────────────────────────────────
    # ANTHROPIC (Claude)
    # ─────────────────────────────────────────────────────────────────
    "claude": ProviderConfig(
        name="Claude (Anthropic)",
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
        name="Google Gemini 2.0",
        provider=Provider.GOOGLE,
        api_key_env="GOOGLE_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        default_model="gemini-2.0-flash",
        supports_vision=True,
        supports_tools=True,
        cost_per_1k_input=0.00,  # Free tier available
        cost_per_1k_output=0.00
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
    # MISTRAL (Codestral cloud, Le Chat)
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
```

## Extended Subagent Registry (Local + Cloud)

Combine local and cloud agents with preference/fallback:

```python
# subagents.py - Extended with cloud fallback

SUBAGENTS = {
    "code": {
        "providers": [
            # Try in order: local first, then cloud fallback
            {"type": "local", "model": "codestral:latest", "url_env": "CODE_AGENT_URL"},
            {"type": "cloud", "provider": "codestral-cloud"},
            {"type": "cloud", "provider": "claude"},
        ],
        "description": "Code generation, completion, refactoring, debugging",
        "system_prompt": "You are a code generation specialist...",
        "temperature": 0.1,
        "max_tokens": 4096
    },
    "reasoning": {
        "providers": [
            {"type": "cloud", "provider": "claude-opus"},
            {"type": "cloud", "provider": "openai-o1"},
            {"type": "local", "model": "nemotron", "url_env": "LLM_SERVER_URL"},
        ],
        "description": "Complex reasoning, planning, analysis",
        "system_prompt": "Think step by step. Analyze thoroughly before concluding.",
        "temperature": 0.3,
        "max_tokens": 8192
    },
    "vision": {
        "providers": [
            {"type": "local", "model": "llava:13b", "url_env": "VISION_AGENT_URL"},
            {"type": "cloud", "provider": "claude"},
            {"type": "cloud", "provider": "gemini"},
        ],
        "description": "Image analysis, screenshots, visual understanding",
        "system_prompt": "Describe images in detail...",
        "temperature": 0.2,
        "max_tokens": 1024
    },
    "writer": {
        "providers": [
            {"type": "local", "model": "gemma-writer:latest", "url_env": "WRITER_AGENT_URL"},
            {"type": "cloud", "provider": "claude"},
        ],
        "description": "Long-form writing, documentation",
        "system_prompt": "Create clear, well-structured content...",
        "temperature": 0.7,
        "max_tokens": 8192
    },
    "search": {
        "providers": [
            {"type": "cloud", "provider": "grok"},  # Grok has real-time info
            {"type": "cloud", "provider": "gemini"},
        ],
        "description": "Web search, current events, real-time information",
        "system_prompt": "Search for and synthesize current information...",
        "temperature": 0.3,
        "max_tokens": 2048
    }
}
```

## API Key Management

### Secure Key Storage

```python
# keys.py - API key management

import os
import json
from pathlib import Path
from typing import Optional
from cryptography.fernet import Fernet
import keyring  # System keychain integration

class KeyManager:
    """Manages API keys with multiple storage backends."""

    CONFIG_DIR = Path.home() / ".config" / "maude"
    KEYS_FILE = CONFIG_DIR / "keys.json.enc"

    def __init__(self):
        self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self._encryption_key = self._get_or_create_encryption_key()

    def _get_or_create_encryption_key(self) -> bytes:
        """Get encryption key from system keychain or create new one."""
        key = keyring.get_password("maude", "encryption_key")
        if not key:
            key = Fernet.generate_key().decode()
            keyring.set_password("maude", "encryption_key", key)
        return key.encode()

    def set_key(self, provider: str, api_key: str):
        """Store an API key securely."""
        keys = self._load_keys()
        keys[provider] = api_key
        self._save_keys(keys)
        # Also set in environment for current session
        if provider in PROVIDERS:
            os.environ[PROVIDERS[provider].api_key_env] = api_key

    def get_key(self, provider: str) -> Optional[str]:
        """Retrieve an API key."""
        # Check environment first (allows override)
        if provider in PROVIDERS:
            env_key = os.environ.get(PROVIDERS[provider].api_key_env)
            if env_key:
                return env_key
        # Fall back to stored keys
        keys = self._load_keys()
        return keys.get(provider)

    def list_configured(self) -> list:
        """List providers with configured keys."""
        keys = self._load_keys()
        return list(keys.keys())

    def _load_keys(self) -> dict:
        if not self.KEYS_FILE.exists():
            return {}
        fernet = Fernet(self._encryption_key)
        encrypted = self.KEYS_FILE.read_bytes()
        decrypted = fernet.decrypt(encrypted)
        return json.loads(decrypted)

    def _save_keys(self, keys: dict):
        fernet = Fernet(self._encryption_key)
        encrypted = fernet.encrypt(json.dumps(keys).encode())
        self.KEYS_FILE.write_bytes(encrypted)
```

### Key Configuration Commands

Add MAUDE commands for key management:

```python
# In chat_local.py - handle special commands

def handle_command(cmd: str) -> Optional[str]:
    """Handle MAUDE slash commands."""
    parts = cmd.strip().split(maxsplit=2)
    command = parts[0].lower()

    if command == "/keys":
        return handle_keys_command(parts[1:])
    elif command == "/providers":
        return show_providers()
    elif command == "/cost":
        return show_session_cost()
    return None

def handle_keys_command(args: list) -> str:
    """Manage API keys."""
    km = KeyManager()

    if not args:
        # List configured providers
        configured = km.list_configured()
        if not configured:
            return "No API keys configured. Use: /keys set <provider> <key>"
        return "Configured providers:\n" + "\n".join(f"  - {p}" for p in configured)

    action = args[0].lower()

    if action == "set" and len(args) >= 3:
        provider, key = args[1], args[2]
        if provider not in PROVIDERS:
            return f"Unknown provider: {provider}\nAvailable: {', '.join(PROVIDERS.keys())}"
        km.set_key(provider, key)
        return f"API key for {provider} saved securely."

    elif action == "remove" and len(args) >= 2:
        provider = args[1]
        km.remove_key(provider)
        return f"API key for {provider} removed."

    elif action == "test" and len(args) >= 2:
        provider = args[1]
        return test_provider_connection(provider)

    return "Usage: /keys [set <provider> <key> | remove <provider> | test <provider>]"

def show_providers() -> str:
    """Show all available providers and their status."""
    lines = ["Available Providers:\n"]
    for name, config in PROVIDERS.items():
        has_key = "✓" if get_api_key(name) else "✗"
        lines.append(f"  [{has_key}] {name:15} - {config.name}")
        lines.append(f"      Model: {config.default_model}")
        lines.append(f"      Cost: ${config.cost_per_1k_input}/1K in, ${config.cost_per_1k_output}/1K out")
    return "\n".join(lines)
```

## Cloud Provider Execution

```python
# execution.py - Execute against different provider types

import anthropic
from openai import OpenAI
import google.generativeai as genai

def execute_cloud_agent(
    provider_name: str,
    task: str,
    context: str = None,
    image_base64: str = None,
    system_prompt: str = None,
    temperature: float = 0.2,
    max_tokens: int = 2048
) -> str:
    """Execute a task using a cloud provider."""

    config = PROVIDERS.get(provider_name)
    if not config:
        return f"Error: Unknown provider '{provider_name}'"

    api_key = get_api_key(provider_name)
    if not api_key:
        return f"Error: No API key for {provider_name}. Use /keys set {provider_name} <key>"

    user_content = task
    if context:
        user_content += f"\n\n## Context\n{context}"

    try:
        if config.provider == Provider.ANTHROPIC:
            return _call_anthropic(config, api_key, system_prompt, user_content, image_base64, temperature, max_tokens)

        elif config.provider == Provider.OPENAI:
            return _call_openai(config, api_key, system_prompt, user_content, image_base64, temperature, max_tokens)

        elif config.provider == Provider.GOOGLE:
            return _call_google(config, api_key, system_prompt, user_content, image_base64, temperature, max_tokens)

        elif config.provider == Provider.XAI:
            return _call_xai(config, api_key, system_prompt, user_content, image_base64, temperature, max_tokens)

        elif config.provider == Provider.MISTRAL:
            return _call_mistral(config, api_key, system_prompt, user_content, temperature, max_tokens)

    except Exception as e:
        return f"Error calling {provider_name}: {e}"


def _call_anthropic(config, api_key, system, user, image, temp, max_tok) -> str:
    """Call Anthropic Claude API."""
    client = anthropic.Anthropic(api_key=api_key)

    content = []
    if image:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": image}
        })
    content.append({"type": "text", "text": user})

    response = client.messages.create(
        model=config.default_model,
        max_tokens=max_tok,
        system=system or "You are a helpful assistant.",
        messages=[{"role": "user", "content": content}]
    )
    return response.content[0].text


def _call_openai(config, api_key, system, user, image, temp, max_tok) -> str:
    """Call OpenAI API."""
    client = OpenAI(api_key=api_key, base_url=config.base_url)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    if image:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}}
            ]
        })
    else:
        messages.append({"role": "user", "content": user})

    response = client.chat.completions.create(
        model=config.default_model,
        messages=messages,
        temperature=temp,
        max_tokens=max_tok
    )
    return response.choices[0].message.content


def _call_google(config, api_key, system, user, image, temp, max_tok) -> str:
    """Call Google Gemini API."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(config.default_model)

    content = []
    if system:
        content.append(system + "\n\n")
    content.append(user)

    if image:
        import base64
        image_bytes = base64.b64decode(image)
        content.append({"mime_type": "image/png", "data": image_bytes})

    response = model.generate_content(
        content,
        generation_config={"temperature": temp, "max_output_tokens": max_tok}
    )
    return response.text


def _call_xai(config, api_key, system, user, image, temp, max_tok) -> str:
    """Call xAI Grok API (OpenAI-compatible)."""
    client = OpenAI(api_key=api_key, base_url=config.base_url)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    response = client.chat.completions.create(
        model=config.default_model,
        messages=messages,
        temperature=temp,
        max_tokens=max_tok
    )
    return response.choices[0].message.content


def _call_mistral(config, api_key, system, user, temp, max_tok) -> str:
    """Call Mistral API (OpenAI-compatible)."""
    client = OpenAI(api_key=api_key, base_url=config.base_url)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    response = client.chat.completions.create(
        model=config.default_model,
        messages=messages,
        temperature=temp,
        max_tokens=max_tok
    )
    return response.choices[0].message.content
```

## Router Implementation

Add a new tool that Nemotron can use to delegate to subagents:

```python
# Add to TOOLS list
{
    "type": "function",
    "function": {
        "name": "delegate_to_agent",
        "description": """Delegate a specialized task to a subagent. Available agents:

LOCAL (free, private, always available):
- 'code': Code generation, debugging, refactoring (Codestral)
- 'vision': Image/screenshot analysis (LLaVA)
- 'writer': Long documentation or detailed explanations (Gemma)

CLOUD (requires API key, higher capability):
- 'reasoning': Complex analysis, planning (Claude Opus / o1)
- 'search': Real-time web info, current events (Grok)

The router will try local first, then fall back to cloud if configured.""",
        "parameters": {
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "enum": ["code", "vision", "writer", "reasoning", "search"],
                    "description": "Which specialized agent to use"
                },
                "task": {
                    "type": "string",
                    "description": "Clear description of what the agent should do"
                },
                "context": {
                    "type": "string",
                    "description": "Relevant context (code, file contents, requirements)"
                },
                "prefer_cloud": {
                    "type": "boolean",
                    "description": "Set true to prefer cloud models for this task (default: false)"
                }
            },
            "required": ["agent", "task"]
        }
    }
}
```

## Subagent Execution

```python
def execute_subagent(agent_name: str, task: str, context: str = None, image_base64: str = None) -> str:
    """Execute a task using a specialized subagent."""

    if agent_name not in SUBAGENTS:
        return f"Error: Unknown agent '{agent_name}'. Available: {list(SUBAGENTS.keys())}"

    agent = SUBAGENTS[agent_name]

    # Create client for this agent's endpoint
    client = OpenAI(
        base_url=agent["url"],
        api_key="not-needed"
    )

    # Build messages
    messages = [
        {"role": "system", "content": agent["system_prompt"]}
    ]

    # Handle vision tasks with images
    if image_base64 and agent_name == "vision":
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": task + ("\n\nContext: " + context if context else "")},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
        })
    else:
        user_content = task
        if context:
            user_content += f"\n\n## Context\n{context}"
        messages.append({"role": "user", "content": user_content})

    console.print(f"[dim cyan]  -> Delegating to {agent_name} ({agent['model']})...[/dim cyan]")

    try:
        response = client.chat.completions.create(
            model=agent["model"],
            messages=messages,
            temperature=agent.get("temperature", 0.2),
            max_tokens=agent.get("max_tokens", 2048)
        )

        result = response.choices[0].message.content
        console.print(f"[dim cyan]  -> {agent_name} completed[/dim cyan]")
        return f"[{agent_name.upper()} AGENT RESPONSE]\n\n{result}"

    except Exception as e:
        return f"Error from {agent_name} agent: {e}"
```

## Multi-Machine Communication

### Peer Discovery via Environment Variables

```bash
# Machine 1 (spark) - runs nemotron + ollama
export LLM_SERVER_URL="http://localhost:30000/v1"
export OLLAMA_URL="http://localhost:11434/v1"

# Machine 2 - connects to spark
export LLM_SERVER_URL="http://spark:30000/v1"
export OLLAMA_URL="http://spark:11434/v1"

# Or via Tailscale
export LLM_SERVER_URL="http://spark-e26c:30000/v1"
export OLLAMA_URL="http://spark-e26c:11434/v1"
```

### Agent Mesh Configuration

For true peer-to-peer MAUDE instances:

```python
# mesh.py - MAUDE instance discovery and communication

import socket
import json
from dataclasses import dataclass
from typing import Dict, List
import threading
import time

@dataclass
class MaudeNode:
    """Represents a MAUDE instance on the network."""
    hostname: str
    ip: str
    port: int
    capabilities: List[str]  # ["nemotron", "codestral", "llava", ...]
    last_seen: float
    load: float  # 0.0-1.0 current utilization

class MaudeMesh:
    """Mesh network of MAUDE instances."""

    BROADCAST_PORT = 31337
    ANNOUNCE_INTERVAL = 30  # seconds

    def __init__(self, my_capabilities: List[str]):
        self.my_capabilities = my_capabilities
        self.nodes: Dict[str, MaudeNode] = {}
        self.running = False

    def start(self):
        """Start mesh discovery."""
        self.running = True
        threading.Thread(target=self._announce_loop, daemon=True).start()
        threading.Thread(target=self._listen_loop, daemon=True).start()

    def _announce_loop(self):
        """Periodically announce our presence."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        while self.running:
            announcement = {
                "type": "maude_announce",
                "hostname": socket.gethostname(),
                "capabilities": self.my_capabilities,
                "api_port": 30000,
                "ollama_port": 11434
            }
            sock.sendto(json.dumps(announcement).encode(),
                       ('<broadcast>', self.BROADCAST_PORT))
            time.sleep(self.ANNOUNCE_INTERVAL)

    def _listen_loop(self):
        """Listen for other MAUDE instances."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('', self.BROADCAST_PORT))

        while self.running:
            data, addr = sock.recvfrom(1024)
            try:
                msg = json.loads(data.decode())
                if msg.get("type") == "maude_announce":
                    self._register_node(msg, addr[0])
            except:
                pass

    def _register_node(self, msg: dict, ip: str):
        """Register or update a discovered node."""
        hostname = msg["hostname"]
        self.nodes[hostname] = MaudeNode(
            hostname=hostname,
            ip=ip,
            port=msg.get("api_port", 30000),
            capabilities=msg.get("capabilities", []),
            last_seen=time.time(),
            load=msg.get("load", 0.0)
        )

    def find_agent(self, capability: str) -> Optional[MaudeNode]:
        """Find a node with the requested capability, preferring lowest load."""
        candidates = [n for n in self.nodes.values()
                     if capability in n.capabilities
                     and time.time() - n.last_seen < 120]  # 2 min timeout
        if not candidates:
            return None
        return min(candidates, key=lambda n: n.load)
```

## Message Protocol

For inter-MAUDE communication (task delegation between machines):

```python
@dataclass
class AgentRequest:
    """Request to a remote MAUDE agent."""
    request_id: str
    source_host: str
    agent_type: str  # "code", "vision", "writer", etc.
    task: str
    context: Optional[str]
    attachments: List[str]  # base64 encoded files/images
    priority: int  # 0-10, higher = more urgent
    timeout: int  # seconds

@dataclass
class AgentResponse:
    """Response from a MAUDE agent."""
    request_id: str
    agent_host: str
    agent_type: str
    success: bool
    result: str
    error: Optional[str]
    tokens_used: int
    duration_ms: int
```

## Implementation Phases

### Phase 1: Local Subagents
- [ ] Create `subagents.py` with registry
- [ ] Add `delegate_to_agent` tool
- [ ] Implement `execute_subagent()` function
- [ ] Update system prompt to instruct Nemotron when to delegate
- [ ] Test with codestral for code tasks

### Phase 2: Cloud Providers
- [ ] Create `providers.py` with provider configs
- [ ] Create `keys.py` for secure API key storage
- [ ] Create `execution.py` with provider-specific clients
- [ ] Add `/keys`, `/providers`, `/cost` commands
- [ ] Implement provider fallback chain (local → cloud)
- [ ] Add `cost_tracker.py` with daily limits

### Phase 3: Remote Subagents (Tailscale)
- [ ] Add per-agent URL configuration
- [ ] Support mixed local/remote agents
- [ ] Add health checks for remote endpoints
- [ ] Implement retry logic with fallback

### Phase 4: MAUDE Mesh
- [ ] Implement `MaudeMesh` discovery
- [ ] Add capability announcement
- [ ] Route requests to best available node
- [ ] Handle node failures gracefully

### Phase 5: Shared Context
- [ ] Implement conversation sharing between nodes
- [ ] Add distributed memory/context using embeddings
- [ ] Enable collaborative problem-solving across MAUDEs

## Environment Variables (Complete)

```bash
# ─────────────────────────────────────────────────────────────────
# LOCAL MODELS (Ollama / llama.cpp)
# ─────────────────────────────────────────────────────────────────
LLM_SERVER_URL=http://localhost:30000/v1      # Nemotron via llama.cpp
VISION_AGENT_URL=http://localhost:11434/v1    # LLaVA via Ollama
CODE_AGENT_URL=http://localhost:11434/v1      # Codestral via Ollama
WRITER_AGENT_URL=http://localhost:11434/v1    # Gemma-writer via Ollama
EMBED_AGENT_URL=http://localhost:11434/v1     # Nomic embeddings

# ─────────────────────────────────────────────────────────────────
# CLOUD API KEYS (optional - enables frontier models)
# ─────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY=sk-ant-...                  # Claude (Sonnet, Opus)
OPENAI_API_KEY=sk-...                         # GPT-4o, o1
GOOGLE_API_KEY=AI...                          # Gemini 2.0
XAI_API_KEY=xai-...                           # Grok
MISTRAL_API_KEY=...                           # Mistral Large, Codestral cloud

# ─────────────────────────────────────────────────────────────────
# MESH NETWORKING (Tailscale)
# ─────────────────────────────────────────────────────────────────
MAUDE_MESH_ENABLED=true
MAUDE_MESH_PORT=31337
MAUDE_ANNOUNCE_CAPABILITIES=nemotron,codestral,llava,gemma-writer

# Remote MAUDE nodes (if not using mesh auto-discovery)
MAUDE_NODES=spark-e26c:30000,workstation:30000

# ─────────────────────────────────────────────────────────────────
# PREFERENCES
# ─────────────────────────────────────────────────────────────────
MAUDE_PREFER_LOCAL=true                       # Try local before cloud
MAUDE_COST_LIMIT=5.00                         # Daily cloud spend limit (USD)
MAUDE_LOG_COSTS=true                          # Track API costs
```

## Updated System Prompt

```python
system_prompt = """You are MAUDE, an advanced AI orchestrator powered by NVIDIA Nemotron.

You have specialized subagents available via `delegate_to_agent`:

LOCAL AGENTS (always available, free):
- **code** (Codestral): Code generation, debugging, refactoring
- **vision** (LLaVA): Image/screenshot analysis (also used by web_view/view_image)
- **writer** (Gemma): Long documentation, detailed explanations

CLOUD AGENTS (if API keys configured):
- **reasoning** (Claude/o1): Complex analysis, multi-step planning, difficult problems
- **search** (Grok): Real-time information, current events, web search

WHEN TO DELEGATE:
1. Code tasks beyond simple fixes → delegate to 'code'
2. Need to analyze images → delegate to 'vision'
3. Long-form documentation → delegate to 'writer'
4. Complex reasoning or planning → delegate to 'reasoning' (prefer_cloud=true)
5. Need current/real-time info → delegate to 'search'

WHEN TO HANDLE YOURSELF:
- Simple questions and explanations
- Quick file operations
- Conversation and clarification
- Synthesizing results from subagents

You are the orchestrator. Delegate specialist work, synthesize results, and present
coherent answers to the user. Track costs with /cost command."""
```

## File Structure

```
terminal-llm/
├── chat_local.py          # Main MAUDE (add delegate_to_agent tool)
├── maude                   # Launcher script
├── subagents.py           # NEW: Agent registry and execution
├── providers.py           # NEW: Cloud provider configurations
├── keys.py                # NEW: Secure API key management
├── execution.py           # NEW: Local + cloud execution logic
├── mesh.py                # NEW: Multi-machine discovery (Phase 3)
├── protocol.py            # NEW: Message types for inter-MAUDE comms
├── cost_tracker.py        # NEW: Track and limit API spending
└── config/
    └── agents.yaml        # NEW: Agent configuration (optional)
```

## Cost Tracking

```python
# cost_tracker.py - Track API usage and costs

import json
from pathlib import Path
from datetime import datetime, date
from dataclasses import dataclass, asdict
from typing import List

@dataclass
class APICall:
    timestamp: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    agent_type: str

class CostTracker:
    """Track API costs with daily limits."""

    LOG_FILE = Path.home() / ".config" / "maude" / "costs.jsonl"

    def __init__(self, daily_limit: float = 5.00):
        self.daily_limit = daily_limit
        self.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    def log_call(self, provider: str, model: str, input_tok: int, output_tok: int, agent: str):
        """Log an API call and its cost."""
        config = PROVIDERS.get(provider)
        if not config:
            return

        cost = (input_tok / 1000 * config.cost_per_1k_input +
                output_tok / 1000 * config.cost_per_1k_output)

        call = APICall(
            timestamp=datetime.now().isoformat(),
            provider=provider,
            model=model,
            input_tokens=input_tok,
            output_tokens=output_tok,
            cost_usd=cost,
            agent_type=agent
        )

        with open(self.LOG_FILE, "a") as f:
            f.write(json.dumps(asdict(call)) + "\n")

    def get_today_spend(self) -> float:
        """Get total spend for today."""
        today = date.today().isoformat()
        total = 0.0

        if not self.LOG_FILE.exists():
            return 0.0

        with open(self.LOG_FILE) as f:
            for line in f:
                call = json.loads(line)
                if call["timestamp"].startswith(today):
                    total += call["cost_usd"]
        return total

    def check_limit(self) -> bool:
        """Return True if under daily limit."""
        return self.get_today_spend() < self.daily_limit

    def get_summary(self) -> str:
        """Get cost summary for display."""
        today_spend = self.get_today_spend()
        remaining = max(0, self.daily_limit - today_spend)
        return f"Today: ${today_spend:.4f} / ${self.daily_limit:.2f} (${remaining:.2f} remaining)"
```

## User Commands Summary

| Command | Description |
|---------|-------------|
| `/keys` | List configured API keys |
| `/keys set <provider> <key>` | Save an API key |
| `/keys remove <provider>` | Remove an API key |
| `/keys test <provider>` | Test API connection |
| `/providers` | Show all providers with status |
| `/cost` | Show today's API spending |
| `/prefer local` | Prefer local models |
| `/prefer cloud` | Prefer cloud models |
| `/agent <name>` | Force use specific agent |
