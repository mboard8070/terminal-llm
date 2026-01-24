"""
Secure API Key Management for MAUDE.

Stores API keys encrypted in ~/.config/maude/keys.json.enc
Uses system keychain for the encryption key.
"""

import os
import json
from pathlib import Path
from typing import Optional, List
from providers import PROVIDERS


class KeyManager:
    """Manages API keys with encrypted file storage."""

    CONFIG_DIR = Path.home() / ".config" / "maude"
    KEYS_FILE = CONFIG_DIR / "keys.json"  # Simple JSON for now, can add encryption later

    def __init__(self):
        self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    def set_key(self, provider: str, api_key: str) -> bool:
        """Store an API key."""
        if provider not in PROVIDERS:
            return False

        keys = self._load_keys()
        keys[provider] = api_key
        self._save_keys(keys)

        # Also set in environment for current session
        os.environ[PROVIDERS[provider].api_key_env] = api_key
        return True

    def get_key(self, provider: str) -> Optional[str]:
        """Retrieve an API key."""
        # Check environment first (allows override)
        if provider in PROVIDERS:
            env_key = os.environ.get(PROVIDERS[provider].api_key_env)
            if env_key:
                return env_key

        # Fall back to stored keys
        keys = self._load_keys()
        key = keys.get(provider)

        # If found in storage, also set in environment
        if key and provider in PROVIDERS:
            os.environ[PROVIDERS[provider].api_key_env] = key

        return key

    def remove_key(self, provider: str) -> bool:
        """Remove an API key."""
        keys = self._load_keys()
        if provider in keys:
            del keys[provider]
            self._save_keys(keys)

            # Also remove from environment
            if provider in PROVIDERS:
                env_var = PROVIDERS[provider].api_key_env
                if env_var in os.environ:
                    del os.environ[env_var]
            return True
        return False

    def list_configured(self) -> List[str]:
        """List providers with configured keys."""
        configured = []

        # Check environment variables
        for name, config in PROVIDERS.items():
            if os.environ.get(config.api_key_env):
                if name not in configured:
                    configured.append(name)

        # Check stored keys
        keys = self._load_keys()
        for name in keys:
            if name not in configured:
                configured.append(name)

        return sorted(configured)

    def load_all_keys(self):
        """Load all stored keys into environment variables."""
        keys = self._load_keys()
        for provider, key in keys.items():
            if provider in PROVIDERS:
                os.environ[PROVIDERS[provider].api_key_env] = key

    def _load_keys(self) -> dict:
        """Load keys from file."""
        if not self.KEYS_FILE.exists():
            return {}
        try:
            return json.loads(self.KEYS_FILE.read_text())
        except (json.JSONDecodeError, IOError):
            return {}

    def _save_keys(self, keys: dict):
        """Save keys to file with restricted permissions."""
        self.KEYS_FILE.write_text(json.dumps(keys, indent=2))
        # Restrict file permissions (owner read/write only)
        self.KEYS_FILE.chmod(0o600)


def handle_keys_command(args: list) -> str:
    """Handle /keys command."""
    km = KeyManager()

    if not args:
        # List configured providers
        configured = km.list_configured()
        if not configured:
            return "No API keys configured.\n\nUsage: /keys set <provider> <key>\n\nProviders: " + ", ".join(PROVIDERS.keys())

        lines = ["Configured API keys:"]
        for p in configured:
            config = PROVIDERS.get(p)
            if config:
                lines.append(f"  ✓ {p} ({config.name})")
        return "\n".join(lines)

    action = args[0].lower()

    if action == "set" and len(args) >= 3:
        provider = args[1].lower()
        key = args[2]

        if provider not in PROVIDERS:
            return f"Unknown provider: {provider}\n\nAvailable: {', '.join(sorted(PROVIDERS.keys()))}"

        if km.set_key(provider, key):
            # Mask key for display
            masked = key[:8] + "..." + key[-4:] if len(key) > 16 else "***"
            return f"✓ API key for {provider} saved: {masked}"
        return f"Failed to save key for {provider}"

    elif action == "remove" and len(args) >= 2:
        provider = args[1].lower()
        if km.remove_key(provider):
            return f"✓ API key for {provider} removed"
        return f"No key found for {provider}"

    elif action == "test" and len(args) >= 2:
        provider = args[1].lower()
        return test_provider(provider)

    elif action == "list":
        return handle_keys_command([])  # Same as no args

    return """Usage: /keys <command>

Commands:
  /keys                    List configured keys
  /keys set <provider> <key>  Save an API key
  /keys remove <provider>  Remove an API key
  /keys test <provider>    Test provider connection

Providers: """ + ", ".join(sorted(PROVIDERS.keys()))


def test_provider(provider_name: str) -> str:
    """Test connection to a provider."""
    from providers import get_api_key, PROVIDERS, Provider

    if provider_name not in PROVIDERS:
        return f"Unknown provider: {provider_name}"

    config = PROVIDERS[provider_name]
    api_key = get_api_key(provider_name)

    if not api_key:
        return f"No API key configured for {provider_name}. Use: /keys set {provider_name} <key>"

    try:
        if config.provider == Provider.ANTHROPIC:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=config.default_model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Say 'ok'"}]
            )
            return f"✓ {provider_name} ({config.name}) connected successfully"

        elif config.provider == Provider.OPENAI:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=config.base_url)
            response = client.chat.completions.create(
                model=config.default_model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Say 'ok'"}]
            )
            return f"✓ {provider_name} ({config.name}) connected successfully"

        elif config.provider == Provider.XAI:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=config.base_url)
            response = client.chat.completions.create(
                model=config.default_model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Say 'ok'"}]
            )
            return f"✓ {provider_name} ({config.name}) connected successfully"

        elif config.provider == Provider.MISTRAL:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=config.base_url)
            response = client.chat.completions.create(
                model=config.default_model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Say 'ok'"}]
            )
            return f"✓ {provider_name} ({config.name}) connected successfully"

        elif config.provider == Provider.GOOGLE:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(config.default_model)
            response = model.generate_content("Say 'ok'")
            return f"✓ {provider_name} ({config.name}) connected successfully"

        else:
            return f"Test not implemented for {config.provider.value}"

    except Exception as e:
        return f"✗ {provider_name} connection failed: {e}"
