"""
Configuration constants — can be overridden via environment variables.
"""

import os

LOCAL_URL = os.environ.get("LLM_SERVER_URL", "http://localhost:30080/v1")
MODEL = os.environ.get("MAUDE_MODEL", "nemotron-super")
NUM_CTX = int(os.environ.get("MAUDE_NUM_CTX", "32768"))
VISION_URL = os.environ.get("VISION_SERVER_URL", "http://localhost:11434/v1")
VISION_MODEL = os.environ.get("MAUDE_VISION_MODEL", "llava:13b")

# Shared session ID for conversation sync
SESSION_ID = os.environ.get("MAUDE_SESSION_ID", "default")

# Models with native multimodal vision support
VISION_CAPABLE_MODELS = {
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "mistral-large-latest",
}

# Provider routing for vision-capable models (base_url, api_key_env, provider)
VISION_MODEL_ROUTES = {
    "claude-opus-4-20250514": {
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com",
        "api_key_env": "CLAUDE_API_KEY",
    },
    "claude-sonnet-4-20250514": {
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com",
        "api_key_env": "CLAUDE_API_KEY",
    },
    "mistral-large-latest": {
        "provider": "mistral",
        "base_url": "https://api.mistral.ai/v1",
        "api_key_env": "MISTRAL_API_KEY",
    },
}

# Free vision fallback via OpenRouter (replaces local LLaVA)
VISION_FALLBACK_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"
VISION_FALLBACK_URL = "https://openrouter.ai/api/v1"
VISION_FALLBACK_KEY_ENV = "OPEN_ROUTER_API_KEY"

# ── Context hygiene (see maude_core.context_hygiene) ─────────────────
# Recent user/assistant turns kept verbatim; older turns become a rolling summary
CTX_KEEP_RECENT_TURNS = int(os.environ.get("MAUDE_CTX_KEEP_RECENT_TURNS", "12"))
# Tool-result rounds kept in full; older tool payloads are stubbed
CTX_KEEP_TOOL_ROUNDS = int(os.environ.get("MAUDE_CTX_KEEP_TOOL_ROUNDS", "2"))
# Hard cap on a single tool result injected into model context
CTX_MAX_TOOL_CHARS = int(os.environ.get("MAUDE_CTX_MAX_TOOL_CHARS", "4000"))
# Top-k memories injected into the system prompt
CTX_MEMORY_TOP_K = int(os.environ.get("MAUDE_CTX_MEMORY_TOP_K", "5"))
