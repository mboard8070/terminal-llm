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
| `/memory` | Show memory stats |
| `/remember <key> <value>` | Store a memory |
| `/recall <key>` | Retrieve a memory |
| `/schedule` | List scheduled tasks |
| `/skills` | List installed skills |
| `/channels` | Show connected channels |
| `/voice` | Toggle voice mode |

---

# Extended Features (Clawdbot-Inspired)

## Multi-Channel Messaging

Connect MAUDE to messaging platforms so you can interact from anywhere.

```
                    ┌─────────────────────────────────────┐
                    │           MAUDE GATEWAY             │
                    │        (WebSocket Hub)              │
                    └─────────────┬───────────────────────┘
                                  │
       ┌──────────┬───────────┬───┴───┬───────────┬──────────┐
       │          │           │       │           │          │
   ┌───▼───┐  ┌───▼───┐  ┌────▼──┐ ┌──▼───┐  ┌────▼───┐  ┌───▼───┐
   │Telegram│  │Discord│  │ Slack │ │Matrix│  │Webhook │  │  CLI  │
   └────────┘  └───────┘  └───────┘ └──────┘  └────────┘  └───────┘
```

### Channel Architecture

```python
# channels/__init__.py - Multi-channel gateway

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any
import asyncio

@dataclass
class IncomingMessage:
    """Unified message format from any channel."""
    channel: str           # "telegram", "discord", etc.
    channel_id: str        # Chat/channel ID
    user_id: str           # Sender ID
    username: str          # Display name
    text: str              # Message content
    attachments: list      # Images, files, etc.
    reply_to: Optional[str]  # Message being replied to
    raw: Any               # Original platform message

@dataclass
class OutgoingMessage:
    """Response to send back."""
    text: str
    attachments: list = None
    reply_to: Optional[str] = None
    parse_mode: str = "markdown"


class Channel(ABC):
    """Base class for messaging channels."""

    name: str

    @abstractmethod
    async def connect(self):
        """Connect to the channel."""
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from the channel."""
        pass

    @abstractmethod
    async def send(self, channel_id: str, message: OutgoingMessage):
        """Send a message to a specific chat."""
        pass

    @abstractmethod
    def on_message(self, callback: Callable[[IncomingMessage], None]):
        """Register message handler."""
        pass


class ChannelGateway:
    """Central hub managing all channels."""

    def __init__(self, maude_callback: Callable):
        self.channels: Dict[str, Channel] = {}
        self.maude = maude_callback
        self.running = False

    def register(self, channel: Channel):
        """Register a channel."""
        self.channels[channel.name] = channel
        channel.on_message(self._handle_message)

    async def _handle_message(self, msg: IncomingMessage):
        """Route incoming message to MAUDE and send response."""
        # Check pairing/authorization
        if not self._is_authorized(msg):
            return

        # Process with MAUDE
        response = await self.maude(msg.text, context={
            "channel": msg.channel,
            "user": msg.username,
            "attachments": msg.attachments
        })

        # Send response back
        await self.channels[msg.channel].send(
            msg.channel_id,
            OutgoingMessage(text=response)
        )

    def _is_authorized(self, msg: IncomingMessage) -> bool:
        """Check if user is authorized (pairing system)."""
        # TODO: Implement pairing codes
        return True

    async def start(self):
        """Start all channels."""
        self.running = True
        await asyncio.gather(*[ch.connect() for ch in self.channels.values()])

    async def stop(self):
        """Stop all channels."""
        self.running = False
        await asyncio.gather(*[ch.disconnect() for ch in self.channels.values()])
```

### Telegram Channel

```python
# channels/telegram.py

from telegram import Update
from telegram.ext import Application, MessageHandler, ContextTypes, filters
import asyncio

class TelegramChannel(Channel):
    """Telegram bot channel."""

    name = "telegram"

    def __init__(self, token: str):
        self.token = token
        self.app = None
        self._callback = None

    async def connect(self):
        self.app = Application.builder().token(self.token).build()
        self.app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self._handle_update
        ))
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()

    async def disconnect(self):
        await self.app.updater.stop()
        await self.app.stop()
        await self.app.shutdown()

    async def _handle_update(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = IncomingMessage(
            channel="telegram",
            channel_id=str(update.effective_chat.id),
            user_id=str(update.effective_user.id),
            username=update.effective_user.username or update.effective_user.first_name,
            text=update.message.text,
            attachments=[],  # TODO: Handle photos, documents
            reply_to=None,
            raw=update
        )
        if self._callback:
            await self._callback(msg)

    async def send(self, channel_id: str, message: OutgoingMessage):
        await self.app.bot.send_message(
            chat_id=int(channel_id),
            text=message.text,
            parse_mode="Markdown"
        )

    def on_message(self, callback):
        self._callback = callback
```

### Discord Channel

```python
# channels/discord.py

import discord
from discord.ext import commands

class DiscordChannel(Channel):
    """Discord bot channel."""

    name = "discord"

    def __init__(self, token: str, allowed_channels: list = None):
        self.token = token
        self.allowed_channels = allowed_channels or []
        intents = discord.Intents.default()
        intents.message_content = True
        self.bot = commands.Bot(command_prefix="!", intents=intents)
        self._callback = None
        self._setup_handlers()

    def _setup_handlers(self):
        @self.bot.event
        async def on_message(message):
            if message.author == self.bot.user:
                return
            if self.allowed_channels and message.channel.id not in self.allowed_channels:
                return

            msg = IncomingMessage(
                channel="discord",
                channel_id=str(message.channel.id),
                user_id=str(message.author.id),
                username=message.author.display_name,
                text=message.content,
                attachments=[a.url for a in message.attachments],
                reply_to=str(message.reference.message_id) if message.reference else None,
                raw=message
            )
            if self._callback:
                await self._callback(msg)

    async def connect(self):
        asyncio.create_task(self.bot.start(self.token))

    async def disconnect(self):
        await self.bot.close()

    async def send(self, channel_id: str, message: OutgoingMessage):
        channel = self.bot.get_channel(int(channel_id))
        if channel:
            await channel.send(message.text)

    def on_message(self, callback):
        self._callback = callback
```

### Webhook Channel (for custom integrations)

```python
# channels/webhook.py

from aiohttp import web
import json

class WebhookChannel(Channel):
    """HTTP webhook channel for custom integrations."""

    name = "webhook"

    def __init__(self, host: str = "0.0.0.0", port: int = 8080, secret: str = None):
        self.host = host
        self.port = port
        self.secret = secret
        self._callback = None
        self._app = None
        self._runner = None

    async def connect(self):
        self._app = web.Application()
        self._app.router.add_post("/message", self._handle_webhook)
        self._app.router.add_get("/health", self._health)
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()

    async def disconnect(self):
        if self._runner:
            await self._runner.cleanup()

    async def _handle_webhook(self, request: web.Request):
        # Verify secret if configured
        if self.secret:
            auth = request.headers.get("Authorization")
            if auth != f"Bearer {self.secret}":
                return web.Response(status=401)

        data = await request.json()
        msg = IncomingMessage(
            channel="webhook",
            channel_id=data.get("channel_id", "default"),
            user_id=data.get("user_id", "webhook"),
            username=data.get("username", "Webhook"),
            text=data.get("text", ""),
            attachments=data.get("attachments", []),
            reply_to=data.get("reply_to"),
            raw=data
        )

        if self._callback:
            await self._callback(msg)

        return web.json_response({"status": "ok"})

    async def _health(self, request):
        return web.json_response({"status": "healthy"})

    async def send(self, channel_id: str, message: OutgoingMessage):
        # Webhook is receive-only; responses go back in HTTP response
        pass

    def on_message(self, callback):
        self._callback = callback
```

---

## Persistent Memory

Remember conversations, preferences, facts, and context across sessions.

```python
# memory.py - Persistent memory system with semantic search

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
import sqlite3
from openai import OpenAI

@dataclass
class Memory:
    """A single memory item."""
    key: str
    value: str
    category: str  # "fact", "preference", "conversation", "task"
    created_at: str
    updated_at: str
    access_count: int
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


class MaudeMemory:
    """Persistent memory with semantic search capabilities."""

    DB_PATH = Path.home() / ".config" / "maude" / "memory.db"

    def __init__(self, embed_url: str = "http://localhost:11434/v1"):
        self.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.embed_client = OpenAI(base_url=embed_url, api_key="not-needed")
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database."""
        self.conn = sqlite3.connect(str(self.DB_PATH))
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                category TEXT DEFAULT 'fact',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,
                embedding BLOB,
                metadata TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                channel TEXT DEFAULT 'cli'
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_session
            ON conversations(session_id)
        """)
        self.conn.commit()

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text using nomic-embed-text."""
        try:
            response = self.embed_client.embeddings.create(
                model="nomic-embed-text:latest",
                input=text
            )
            return response.data[0].embedding
        except Exception:
            return []

    def remember(self, key: str, value: str, category: str = "fact",
                 metadata: dict = None) -> bool:
        """Store or update a memory."""
        now = datetime.now().isoformat()
        embedding = self._get_embedding(f"{key}: {value}")

        self.conn.execute("""
            INSERT INTO memories (key, value, category, created_at, updated_at,
                                  access_count, embedding, metadata)
            VALUES (?, ?, ?, ?, ?, 0, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                category = excluded.category,
                updated_at = excluded.updated_at,
                embedding = excluded.embedding,
                metadata = excluded.metadata
        """, (key, value, category, now, now,
              json.dumps(embedding) if embedding else None,
              json.dumps(metadata) if metadata else None))
        self.conn.commit()
        return True

    def recall(self, key: str) -> Optional[str]:
        """Retrieve a memory by key."""
        cursor = self.conn.execute("""
            SELECT value FROM memories WHERE key = ?
        """, (key,))
        row = cursor.fetchone()
        if row:
            # Update access count
            self.conn.execute("""
                UPDATE memories SET access_count = access_count + 1
                WHERE key = ?
            """, (key,))
            self.conn.commit()
            return row[0]
        return None

    def forget(self, key: str) -> bool:
        """Remove a memory."""
        self.conn.execute("DELETE FROM memories WHERE key = ?", (key,))
        self.conn.commit()
        return True

    def search(self, query: str, limit: int = 5, category: str = None) -> List[Memory]:
        """Semantic search across memories."""
        query_embedding = self._get_embedding(query)
        if not query_embedding:
            # Fall back to text search
            return self._text_search(query, limit, category)

        # Get all memories with embeddings
        sql = "SELECT * FROM memories WHERE embedding IS NOT NULL"
        if category:
            sql += f" AND category = '{category}'"
        cursor = self.conn.execute(sql)

        # Calculate cosine similarity
        results = []
        for row in cursor.fetchall():
            mem_embedding = json.loads(row[6]) if row[6] else []
            if mem_embedding:
                similarity = self._cosine_similarity(query_embedding, mem_embedding)
                results.append((similarity, Memory(
                    key=row[0], value=row[1], category=row[2],
                    created_at=row[3], updated_at=row[4], access_count=row[5],
                    metadata=json.loads(row[7]) if row[7] else None
                )))

        # Sort by similarity and return top results
        results.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in results[:limit]]

    def _text_search(self, query: str, limit: int, category: str) -> List[Memory]:
        """Fallback text search."""
        sql = "SELECT * FROM memories WHERE value LIKE ?"
        params = [f"%{query}%"]
        if category:
            sql += " AND category = ?"
            params.append(category)
        sql += f" LIMIT {limit}"

        cursor = self.conn.execute(sql, params)
        return [Memory(
            key=r[0], value=r[1], category=r[2],
            created_at=r[3], updated_at=r[4], access_count=r[5],
            metadata=json.loads(r[7]) if r[7] else None
        ) for r in cursor.fetchall()]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    def save_conversation(self, session_id: str, role: str, content: str,
                          channel: str = "cli"):
        """Save a conversation turn."""
        self.conn.execute("""
            INSERT INTO conversations (session_id, role, content, timestamp, channel)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, role, content, datetime.now().isoformat(), channel))
        self.conn.commit()

    def get_conversation(self, session_id: str, limit: int = 50) -> List[dict]:
        """Retrieve conversation history."""
        cursor = self.conn.execute("""
            SELECT role, content, timestamp FROM conversations
            WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?
        """, (session_id, limit))
        return [{"role": r[0], "content": r[1], "timestamp": r[2]}
                for r in reversed(cursor.fetchall())]

    def summarize_and_archive(self, session_id: str, summary: str):
        """Archive a conversation with its summary."""
        self.remember(
            key=f"conversation:{session_id}",
            value=summary,
            category="conversation",
            metadata={"session_id": session_id, "archived_at": datetime.now().isoformat()}
        )
        # Optionally delete raw conversation
        # self.conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))

    def get_context_for_prompt(self, query: str, max_memories: int = 5) -> str:
        """Get relevant memories to inject into prompt context."""
        memories = self.search(query, limit=max_memories)
        if not memories:
            return ""

        context = "## Relevant Memories\n\n"
        for mem in memories:
            context += f"- **{mem.key}**: {mem.value}\n"
        return context

    def get_stats(self) -> dict:
        """Get memory statistics."""
        cursor = self.conn.execute("""
            SELECT category, COUNT(*) FROM memories GROUP BY category
        """)
        categories = dict(cursor.fetchall())

        cursor = self.conn.execute("SELECT COUNT(*) FROM memories")
        total = cursor.fetchone()[0]

        cursor = self.conn.execute("SELECT COUNT(DISTINCT session_id) FROM conversations")
        sessions = cursor.fetchone()[0]

        return {
            "total_memories": total,
            "by_category": categories,
            "conversation_sessions": sessions
        }
```

### Memory-Enhanced System Prompt

```python
def get_system_prompt_with_memory(memory: MaudeMemory, user_query: str) -> str:
    """Build system prompt with relevant memories injected."""

    base_prompt = """You are MAUDE, an advanced AI orchestrator with persistent memory.

You remember information about the user and past conversations. Use this context
to provide personalized, consistent responses.
"""

    # Get relevant memories
    memory_context = memory.get_context_for_prompt(user_query)

    # Get user preferences
    prefs = memory.search("preference", limit=3, category="preference")
    if prefs:
        pref_text = "\n## User Preferences\n" + "\n".join(f"- {p.value}" for p in prefs)
    else:
        pref_text = ""

    return base_prompt + memory_context + pref_text
```

---

## Skills/Plugins System

Extensible tools that can be shared, installed, and managed.

```python
# skills/__init__.py - Skills framework

import os
import json
import importlib.util
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, List, Any, Optional
from functools import wraps

@dataclass
class SkillMetadata:
    """Metadata for a skill."""
    name: str
    description: str
    version: str
    author: str
    triggers: List[str]  # Keywords that activate this skill
    parameters: Dict[str, Any]  # JSON schema for parameters
    requires: List[str]  # Dependencies (pip packages)


class Skill:
    """A MAUDE skill (plugin)."""

    def __init__(self, metadata: SkillMetadata, execute_fn: Callable):
        self.metadata = metadata
        self.execute = execute_fn
        self.enabled = True


# Global skill registry
_skills: Dict[str, Skill] = {}


def skill(
    name: str,
    description: str = "",
    version: str = "1.0.0",
    author: str = "unknown",
    triggers: List[str] = None,
    parameters: Dict[str, Any] = None,
    requires: List[str] = None
):
    """Decorator to register a function as a skill."""
    def decorator(fn: Callable):
        metadata = SkillMetadata(
            name=name,
            description=description or fn.__doc__ or "",
            version=version,
            author=author,
            triggers=triggers or [name],
            parameters=parameters or {},
            requires=requires or []
        )
        _skills[name] = Skill(metadata, fn)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        return wrapper
    return decorator


class SkillManager:
    """Manages skill discovery, loading, and execution."""

    SKILLS_DIR = Path.home() / ".config" / "maude" / "skills"
    BUILTIN_DIR = Path(__file__).parent / "builtin"

    def __init__(self):
        self.SKILLS_DIR.mkdir(parents=True, exist_ok=True)
        self._load_builtin_skills()
        self._load_user_skills()

    def _load_builtin_skills(self):
        """Load built-in skills."""
        if self.BUILTIN_DIR.exists():
            for skill_file in self.BUILTIN_DIR.glob("*.py"):
                self._load_skill_file(skill_file)

    def _load_user_skills(self):
        """Load user-installed skills."""
        for skill_file in self.SKILLS_DIR.glob("*.py"):
            self._load_skill_file(skill_file)

    def _load_skill_file(self, path: Path):
        """Load a skill from a Python file."""
        try:
            spec = importlib.util.spec_from_file_location(path.stem, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"Failed to load skill {path}: {e}")

    def list_skills(self) -> List[SkillMetadata]:
        """List all registered skills."""
        return [s.metadata for s in _skills.values() if s.enabled]

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        return _skills.get(name)

    def execute_skill(self, name: str, **kwargs) -> Any:
        """Execute a skill."""
        skill = self.get_skill(name)
        if not skill:
            return f"Error: Skill '{name}' not found"
        if not skill.enabled:
            return f"Error: Skill '{name}' is disabled"
        try:
            return skill.execute(**kwargs)
        except Exception as e:
            return f"Error executing skill '{name}': {e}"

    def enable_skill(self, name: str):
        """Enable a skill."""
        if name in _skills:
            _skills[name].enabled = True

    def disable_skill(self, name: str):
        """Disable a skill."""
        if name in _skills:
            _skills[name].enabled = False

    def install_skill(self, url_or_path: str) -> bool:
        """Install a skill from URL or local path."""
        # TODO: Implement skill installation from ClawdHub-like registry
        pass

    def get_tool_definitions(self) -> List[dict]:
        """Get OpenAI-compatible tool definitions for all skills."""
        tools = []
        for skill in _skills.values():
            if not skill.enabled:
                continue
            tools.append({
                "type": "function",
                "function": {
                    "name": f"skill_{skill.metadata.name}",
                    "description": skill.metadata.description,
                    "parameters": skill.metadata.parameters or {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            })
        return tools
```

### Built-in Skills

```python
# skills/builtin/stock_alert.py

from skills import skill
import requests

@skill(
    name="stock_price",
    description="Get current stock price and set alerts",
    triggers=["stock", "price", "market"],
    parameters={
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "Stock ticker symbol"},
            "action": {"type": "string", "enum": ["price", "alert"], "default": "price"},
            "threshold": {"type": "number", "description": "Alert threshold price"}
        },
        "required": ["symbol"]
    }
)
def stock_price(symbol: str, action: str = "price", threshold: float = None) -> str:
    """Get stock price or set price alert."""
    # Using free API (Alpha Vantage, Yahoo Finance, etc.)
    try:
        # Placeholder - implement with real API
        return f"Stock {symbol.upper()}: $XXX.XX"
    except Exception as e:
        return f"Error fetching stock price: {e}"


# skills/builtin/email_triage.py

@skill(
    name="email_triage",
    description="Summarize and prioritize emails from Gmail",
    triggers=["email", "inbox", "mail"],
    parameters={
        "type": "object",
        "properties": {
            "count": {"type": "integer", "default": 10},
            "filter": {"type": "string", "description": "Search filter"}
        }
    },
    requires=["google-api-python-client"]
)
def email_triage(count: int = 10, filter: str = None) -> str:
    """Triage emails from Gmail."""
    # TODO: Implement Gmail API integration
    return "Email triage not yet configured. Set up Gmail API credentials."


# skills/builtin/calendar.py

@skill(
    name="calendar",
    description="Check and manage calendar events",
    triggers=["calendar", "schedule", "meeting", "appointment"],
    parameters={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["today", "week", "add", "find"]},
            "query": {"type": "string"}
        }
    }
)
def calendar(action: str = "today", query: str = None) -> str:
    """Manage calendar events."""
    # TODO: Implement Google Calendar / CalDAV integration
    return "Calendar not yet configured."


# skills/builtin/weather.py

@skill(
    name="weather",
    description="Get weather forecast",
    triggers=["weather", "forecast", "temperature"],
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City or coordinates"}
        },
        "required": ["location"]
    }
)
def weather(location: str) -> str:
    """Get weather for a location."""
    import requests
    try:
        # Using Open-Meteo (free, no API key)
        # First geocode the location
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1"
        geo_resp = requests.get(geo_url).json()
        if not geo_resp.get("results"):
            return f"Location '{location}' not found"

        lat = geo_resp["results"][0]["latitude"]
        lon = geo_resp["results"][0]["longitude"]
        name = geo_resp["results"][0]["name"]

        # Get weather
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather_resp = requests.get(weather_url).json()
        current = weather_resp["current_weather"]

        return f"Weather in {name}: {current['temperature']}°C, wind {current['windspeed']} km/h"
    except Exception as e:
        return f"Error fetching weather: {e}"
```

---

## Proactive Scheduler

MAUDE messages YOU first - alerts, reminders, background tasks.

```python
# scheduler.py - Proactive task scheduler

import asyncio
from datetime import datetime
from dataclasses import dataclass
from typing import Callable, Optional, Dict, List, Any
from croniter import croniter
import json
from pathlib import Path

@dataclass
class ScheduledTask:
    """A scheduled task."""
    id: str
    name: str
    cron: str  # Cron expression
    prompt: str  # What to ask MAUDE
    channel: str  # Where to send result
    channel_id: str  # Specific chat/channel
    enabled: bool = True
    last_run: Optional[str] = None
    next_run: Optional[str] = None


class ProactiveScheduler:
    """Schedule MAUDE to run tasks and message you."""

    CONFIG_FILE = Path.home() / ".config" / "maude" / "schedules.json"

    def __init__(self, maude_callback: Callable, channel_gateway):
        self.maude = maude_callback
        self.gateway = channel_gateway
        self.tasks: Dict[str, ScheduledTask] = {}
        self.running = False
        self._load_tasks()

    def _load_tasks(self):
        """Load scheduled tasks from disk."""
        if self.CONFIG_FILE.exists():
            data = json.loads(self.CONFIG_FILE.read_text())
            for task_data in data:
                task = ScheduledTask(**task_data)
                self.tasks[task.id] = task

    def _save_tasks(self):
        """Save scheduled tasks to disk."""
        data = [
            {
                "id": t.id, "name": t.name, "cron": t.cron,
                "prompt": t.prompt, "channel": t.channel,
                "channel_id": t.channel_id, "enabled": t.enabled,
                "last_run": t.last_run, "next_run": t.next_run
            }
            for t in self.tasks.values()
        ]
        self.CONFIG_FILE.write_text(json.dumps(data, indent=2))

    def schedule(
        self,
        name: str,
        cron: str,
        prompt: str,
        channel: str = "cli",
        channel_id: str = "default"
    ) -> str:
        """Schedule a new task."""
        import uuid
        task_id = str(uuid.uuid4())[:8]

        # Validate cron expression
        try:
            cron_iter = croniter(cron)
            next_run = cron_iter.get_next(datetime).isoformat()
        except Exception as e:
            return f"Invalid cron expression: {e}"

        task = ScheduledTask(
            id=task_id,
            name=name,
            cron=cron,
            prompt=prompt,
            channel=channel,
            channel_id=channel_id,
            next_run=next_run
        )
        self.tasks[task_id] = task
        self._save_tasks()

        return f"Scheduled '{name}' (ID: {task_id})\nNext run: {next_run}"

    def unschedule(self, task_id: str) -> str:
        """Remove a scheduled task."""
        if task_id in self.tasks:
            name = self.tasks[task_id].name
            del self.tasks[task_id]
            self._save_tasks()
            return f"Removed task '{name}'"
        return f"Task {task_id} not found"

    def list_tasks(self) -> str:
        """List all scheduled tasks."""
        if not self.tasks:
            return "No scheduled tasks."

        lines = ["Scheduled Tasks:\n"]
        for task in self.tasks.values():
            status = "✓" if task.enabled else "✗"
            lines.append(f"  [{status}] {task.id}: {task.name}")
            lines.append(f"      Cron: {task.cron}")
            lines.append(f"      Next: {task.next_run}")
            lines.append(f"      Channel: {task.channel}")
        return "\n".join(lines)

    async def start(self):
        """Start the scheduler loop."""
        self.running = True
        while self.running:
            await self._check_and_run_tasks()
            await asyncio.sleep(60)  # Check every minute

    async def stop(self):
        """Stop the scheduler."""
        self.running = False

    async def _check_and_run_tasks(self):
        """Check and run due tasks."""
        now = datetime.now()

        for task in self.tasks.values():
            if not task.enabled:
                continue

            if task.next_run:
                next_run = datetime.fromisoformat(task.next_run)
                if now >= next_run:
                    await self._run_task(task)

    async def _run_task(self, task: ScheduledTask):
        """Execute a scheduled task."""
        try:
            # Run the prompt through MAUDE
            response = await self.maude(task.prompt, context={
                "scheduled_task": task.name,
                "channel": task.channel
            })

            # Send result to channel
            if task.channel != "cli" and self.gateway:
                from channels import OutgoingMessage
                await self.gateway.channels[task.channel].send(
                    task.channel_id,
                    OutgoingMessage(text=f"📅 **{task.name}**\n\n{response}")
                )

            # Update task
            task.last_run = datetime.now().isoformat()
            cron_iter = croniter(task.cron, datetime.now())
            task.next_run = cron_iter.get_next(datetime).isoformat()
            self._save_tasks()

        except Exception as e:
            print(f"Error running scheduled task {task.name}: {e}")


# Example scheduled tasks
EXAMPLE_SCHEDULES = [
    {
        "name": "Morning Briefing",
        "cron": "0 8 * * *",  # 8am daily
        "prompt": "Give me a brief summary of: weather today, my calendar, any urgent emails, and top news headlines."
    },
    {
        "name": "Stock Check",
        "cron": "0 9,12,16 * * 1-5",  # 9am, 12pm, 4pm on weekdays
        "prompt": "Check the prices of NVDA, AAPL, GOOGL and alert me if any moved more than 3% today."
    },
    {
        "name": "Weekly Review",
        "cron": "0 18 * * 5",  # Friday 6pm
        "prompt": "Summarize what I worked on this week based on our conversations and any tasks I completed."
    },
]
```

---

## Voice Mode

Speak and listen - hands-free interaction.

```python
# voice.py - Voice input/output

import os
import asyncio
from typing import Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import tempfile

@dataclass
class VoiceConfig:
    """Voice mode configuration."""
    stt_provider: str = "whisper"  # "whisper" (local), "openai", "google"
    tts_provider: str = "say"  # "say" (macOS), "espeak", "elevenlabs", "openai"
    elevenlabs_voice: str = "Rachel"
    openai_voice: str = "nova"
    wake_word: Optional[str] = None  # e.g., "Hey MAUDE"
    auto_listen: bool = False  # Continuous listening mode


class VoiceMode:
    """Voice input/output for MAUDE."""

    def __init__(self, config: VoiceConfig = None, ollama_url: str = "http://localhost:11434"):
        self.config = config or VoiceConfig()
        self.ollama_url = ollama_url
        self._listening = False

    async def listen(self, timeout: int = 10) -> Optional[str]:
        """Listen for voice input and transcribe."""
        import sounddevice as sd
        import numpy as np
        from scipy.io import wavfile

        sample_rate = 16000

        print("🎤 Listening...")

        # Record audio
        recording = sd.rec(
            int(timeout * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.int16
        )
        sd.wait()

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wavfile.write(f.name, sample_rate, recording)
            temp_path = f.name

        try:
            # Transcribe
            text = await self._transcribe(temp_path)
            return text
        finally:
            os.unlink(temp_path)

    async def _transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text."""
        if self.config.stt_provider == "whisper":
            # Use local Whisper via Ollama or whisper.cpp
            return await self._transcribe_whisper(audio_path)
        elif self.config.stt_provider == "openai":
            return await self._transcribe_openai(audio_path)
        else:
            return ""

    async def _transcribe_whisper(self, audio_path: str) -> str:
        """Transcribe using local Whisper."""
        # Option 1: whisper.cpp
        # Option 2: faster-whisper
        try:
            from faster_whisper import WhisperModel
            model = WhisperModel("base", device="cpu")
            segments, _ = model.transcribe(audio_path)
            return " ".join(s.text for s in segments)
        except ImportError:
            # Fallback to command line whisper
            import subprocess
            result = subprocess.run(
                ["whisper", audio_path, "--model", "base", "--output_format", "txt"],
                capture_output=True, text=True
            )
            return result.stdout.strip()

    async def _transcribe_openai(self, audio_path: str) -> str:
        """Transcribe using OpenAI Whisper API."""
        from openai import OpenAI
        client = OpenAI()
        with open(audio_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        return transcript.text

    async def speak(self, text: str):
        """Convert text to speech and play."""
        if self.config.tts_provider == "say":
            await self._speak_say(text)
        elif self.config.tts_provider == "espeak":
            await self._speak_espeak(text)
        elif self.config.tts_provider == "elevenlabs":
            await self._speak_elevenlabs(text)
        elif self.config.tts_provider == "openai":
            await self._speak_openai(text)

    async def _speak_say(self, text: str):
        """Use macOS 'say' command."""
        import subprocess
        subprocess.run(["say", text])

    async def _speak_espeak(self, text: str):
        """Use espeak for Linux."""
        import subprocess
        subprocess.run(["espeak", text])

    async def _speak_elevenlabs(self, text: str):
        """Use ElevenLabs TTS."""
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            print("ElevenLabs API key not set")
            return

        import requests
        response = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{self.config.elevenlabs_voice}",
            headers={"xi-api-key": api_key},
            json={"text": text, "model_id": "eleven_monolingual_v1"}
        )

        if response.ok:
            # Play audio
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(response.content)
                temp_path = f.name
            import subprocess
            subprocess.run(["ffplay", "-nodisp", "-autoexit", temp_path])
            os.unlink(temp_path)

    async def _speak_openai(self, text: str):
        """Use OpenAI TTS."""
        from openai import OpenAI
        client = OpenAI()

        response = client.audio.speech.create(
            model="tts-1",
            voice=self.config.openai_voice,
            input=text
        )

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(response.content)
            temp_path = f.name

        import subprocess
        subprocess.run(["ffplay", "-nodisp", "-autoexit", temp_path])
        os.unlink(temp_path)

    async def talk_mode(self, maude_callback: Callable):
        """Continuous conversation mode."""
        print("🎙️ Talk mode enabled. Say 'stop' or press Ctrl+C to exit.")
        self._listening = True

        while self._listening:
            try:
                # Listen for input
                text = await self.listen(timeout=10)
                if not text:
                    continue

                print(f"You: {text}")

                if text.lower().strip() in ["stop", "exit", "quit"]:
                    break

                # Get MAUDE response
                response = await maude_callback(text)
                print(f"MAUDE: {response}")

                # Speak response
                await self.speak(response)

            except KeyboardInterrupt:
                break

        self._listening = False
        print("Talk mode ended.")
```

---

## Browser Automation

Enhanced browser control via CDP (Chrome DevTools Protocol).

```python
# browser.py - Enhanced browser automation

from playwright.async_api import async_playwright, Browser, Page
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import asyncio
import base64

@dataclass
class BrowserAction:
    """A browser automation action."""
    action: str  # "goto", "click", "type", "scroll", "screenshot", "extract"
    selector: Optional[str] = None
    value: Optional[str] = None
    options: Optional[Dict[str, Any]] = None


class BrowserAutomation:
    """Enhanced browser control for MAUDE."""

    def __init__(self):
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None

    async def start(self, headless: bool = True):
        """Start browser."""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=headless)
        self.page = await self.browser.new_page()

    async def stop(self):
        """Stop browser."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def goto(self, url: str, wait_until: str = "networkidle") -> str:
        """Navigate to URL."""
        await self.page.goto(url, wait_until=wait_until)
        return f"Navigated to {url}"

    async def click(self, selector: str) -> str:
        """Click an element."""
        await self.page.click(selector)
        return f"Clicked {selector}"

    async def type(self, selector: str, text: str) -> str:
        """Type text into an element."""
        await self.page.fill(selector, text)
        return f"Typed into {selector}"

    async def screenshot(self, full_page: bool = False) -> str:
        """Take screenshot and return base64."""
        screenshot = await self.page.screenshot(full_page=full_page)
        return base64.b64encode(screenshot).decode()

    async def extract_text(self, selector: str = "body") -> str:
        """Extract text from element."""
        element = await self.page.query_selector(selector)
        if element:
            return await element.inner_text()
        return ""

    async def extract_links(self) -> List[Dict[str, str]]:
        """Extract all links from page."""
        links = await self.page.eval_on_selector_all(
            "a[href]",
            "elements => elements.map(e => ({href: e.href, text: e.innerText}))"
        )
        return links

    async def fill_form(self, fields: Dict[str, str]) -> str:
        """Fill a form with multiple fields."""
        for selector, value in fields.items():
            await self.page.fill(selector, value)
        return f"Filled {len(fields)} form fields"

    async def execute_workflow(self, actions: List[BrowserAction]) -> List[str]:
        """Execute a sequence of browser actions."""
        results = []
        for action in actions:
            if action.action == "goto":
                result = await self.goto(action.value)
            elif action.action == "click":
                result = await self.click(action.selector)
            elif action.action == "type":
                result = await self.type(action.selector, action.value)
            elif action.action == "screenshot":
                result = await self.screenshot()
            elif action.action == "extract":
                result = await self.extract_text(action.selector)
            elif action.action == "wait":
                await asyncio.sleep(float(action.value or 1))
                result = f"Waited {action.value}s"
            else:
                result = f"Unknown action: {action.action}"
            results.append(result)
        return results

    async def get_page_summary(self) -> Dict[str, Any]:
        """Get summary of current page."""
        return {
            "url": self.page.url,
            "title": await self.page.title(),
            "links_count": len(await self.extract_links()),
        }
```

---

## Security & Pairing

Approval system for new contacts, sandboxing for untrusted sessions.

```python
# security.py - Security and pairing system

import os
import json
import secrets
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Set, Dict
from dataclasses import dataclass

@dataclass
class PairedUser:
    """A user that has been paired/authorized."""
    user_id: str
    channel: str
    username: str
    paired_at: str
    permissions: Set[str]  # "tools", "files", "shell", "admin"


class SecurityManager:
    """Manage pairing and permissions."""

    CONFIG_DIR = Path.home() / ".config" / "maude"
    PAIRS_FILE = CONFIG_DIR / "paired_users.json"
    PENDING_FILE = CONFIG_DIR / "pending_pairs.json"

    def __init__(self):
        self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self.paired_users: Dict[str, PairedUser] = {}
        self.pending_codes: Dict[str, dict] = {}  # code -> {user_id, channel, expires}
        self._load()

    def _load(self):
        """Load paired users and pending codes."""
        if self.PAIRS_FILE.exists():
            data = json.loads(self.PAIRS_FILE.read_text())
            for item in data:
                user = PairedUser(**{**item, "permissions": set(item["permissions"])})
                key = f"{user.channel}:{user.user_id}"
                self.paired_users[key] = user

        if self.PENDING_FILE.exists():
            self.pending_codes = json.loads(self.PENDING_FILE.read_text())

    def _save(self):
        """Save paired users."""
        data = [
            {**vars(u), "permissions": list(u.permissions)}
            for u in self.paired_users.values()
        ]
        self.PAIRS_FILE.write_text(json.dumps(data, indent=2))
        self.PENDING_FILE.write_text(json.dumps(self.pending_codes, indent=2))

    def is_authorized(self, channel: str, user_id: str) -> bool:
        """Check if user is authorized."""
        key = f"{channel}:{user_id}"
        return key in self.paired_users

    def get_permissions(self, channel: str, user_id: str) -> Set[str]:
        """Get user's permissions."""
        key = f"{channel}:{user_id}"
        user = self.paired_users.get(key)
        return user.permissions if user else set()

    def has_permission(self, channel: str, user_id: str, permission: str) -> bool:
        """Check if user has specific permission."""
        perms = self.get_permissions(channel, user_id)
        return permission in perms or "admin" in perms

    def generate_pairing_code(self, channel: str, user_id: str, username: str) -> str:
        """Generate a pairing code for a new user."""
        code = secrets.token_hex(3).upper()  # e.g., "A1B2C3"
        self.pending_codes[code] = {
            "user_id": user_id,
            "channel": channel,
            "username": username,
            "expires": (datetime.now() + timedelta(minutes=10)).isoformat()
        }
        self._save()
        return code

    def verify_pairing_code(self, code: str) -> Optional[str]:
        """Verify a pairing code and complete pairing."""
        code = code.upper().strip()
        if code not in self.pending_codes:
            return None

        pending = self.pending_codes[code]

        # Check expiration
        expires = datetime.fromisoformat(pending["expires"])
        if datetime.now() > expires:
            del self.pending_codes[code]
            self._save()
            return None

        # Complete pairing
        key = f"{pending['channel']}:{pending['user_id']}"
        self.paired_users[key] = PairedUser(
            user_id=pending["user_id"],
            channel=pending["channel"],
            username=pending["username"],
            paired_at=datetime.now().isoformat(),
            permissions={"tools", "files"}  # Default permissions
        )

        del self.pending_codes[code]
        self._save()

        return pending["username"]

    def revoke(self, channel: str, user_id: str) -> bool:
        """Revoke a user's access."""
        key = f"{channel}:{user_id}"
        if key in self.paired_users:
            del self.paired_users[key]
            self._save()
            return True
        return False

    def grant_permission(self, channel: str, user_id: str, permission: str):
        """Grant a permission to a user."""
        key = f"{channel}:{user_id}"
        if key in self.paired_users:
            self.paired_users[key].permissions.add(permission)
            self._save()

    def list_paired(self) -> str:
        """List all paired users."""
        if not self.paired_users:
            return "No paired users."

        lines = ["Paired Users:\n"]
        for user in self.paired_users.values():
            lines.append(f"  {user.username} ({user.channel})")
            lines.append(f"    Permissions: {', '.join(user.permissions)}")
            lines.append(f"    Paired: {user.paired_at}")
        return "\n".join(lines)
```

---

## Updated Implementation Phases

### Phase 1: Local Subagents ✅
### Phase 2: Cloud Providers ✅
### Phase 3: Remote Subagents (Tailscale)
### Phase 4: MAUDE Mesh

### Phase 5: Persistent Memory
- [ ] Implement `memory.py` with SQLite backend
- [ ] Add semantic search via nomic embeddings
- [ ] Add conversation history storage
- [ ] Inject memories into system prompt
- [ ] Add `/memory`, `/remember`, `/recall` commands

### Phase 6: Skills/Plugins System
- [ ] Implement `skills/__init__.py` framework
- [ ] Add built-in skills (weather, stocks, calendar)
- [ ] Add skill discovery and loading
- [ ] Create skill tool definitions for LLM
- [ ] Add `/skills` command

### Phase 7: Multi-Channel Messaging
- [ ] Implement `channels/__init__.py` gateway
- [ ] Add Telegram channel
- [ ] Add Discord channel
- [ ] Add Webhook channel
- [ ] Add `/channels` command

### Phase 8: Proactive Scheduler
- [ ] Implement `scheduler.py`
- [ ] Add cron-based task scheduling
- [ ] Integrate with channel gateway
- [ ] Add `/schedule` command
- [ ] Add example schedules (morning briefing, etc.)

### Phase 9: Voice Mode
- [ ] Implement `voice.py`
- [ ] Add local Whisper transcription
- [ ] Add TTS options (say, espeak, elevenlabs, openai)
- [ ] Add talk mode (continuous conversation)
- [ ] Add `/voice` command

### Phase 10: Browser Automation
- [ ] Enhance `browser.py` with CDP control
- [ ] Add form filling
- [ ] Add multi-step workflows
- [ ] Add page analysis tools

### Phase 11: Security & Pairing
- [ ] Implement `security.py`
- [ ] Add pairing code system
- [ ] Add per-user permissions
- [ ] Add channel-specific sandboxing

---

## Updated File Structure

```
terminal-llm/
├── chat_local.py          # Main MAUDE CLI
├── maude                   # Launcher script
├── gateway.py             # NEW: Main gateway server
│
├── subagents.py           # Agent registry
├── providers.py           # Cloud providers
├── keys.py                # API key management
├── execution.py           # Agent execution
├── cost_tracker.py        # Cost tracking
│
├── memory.py              # NEW: Persistent memory
├── scheduler.py           # NEW: Proactive scheduler
├── voice.py               # NEW: Voice I/O
├── browser.py             # NEW: Browser automation
├── security.py            # NEW: Pairing & permissions
│
├── channels/              # NEW: Messaging channels
│   ├── __init__.py        # Channel gateway
│   ├── telegram.py
│   ├── discord.py
│   ├── slack.py
│   ├── matrix.py
│   └── webhook.py
│
├── skills/                # NEW: Skills/plugins
│   ├── __init__.py        # Skill framework
│   └── builtin/
│       ├── weather.py
│       ├── stock_alert.py
│       ├── email_triage.py
│       └── calendar.py
│
├── mesh.py                # Multi-machine discovery
├── protocol.py            # Inter-MAUDE protocol
│
└── config/
    └── agents.yaml
```

---

## New Dependencies

```
# requirements.txt additions

# Multi-channel messaging
python-telegram-bot>=21.0
discord.py>=2.0
slack-bolt>=1.18
matrix-nio>=0.24

# Scheduler
croniter>=2.0

# Voice
sounddevice>=0.4
scipy>=1.11
faster-whisper>=1.0  # Optional: local transcription

# Browser automation (already have playwright)
# playwright>=1.40.0

# Memory
# sqlite3 is built-in
```
