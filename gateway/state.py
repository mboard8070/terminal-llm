"""
Shared state, configuration, and model routing for the MAUDE Gateway.

All global variables, constants, and model configuration live here so that
every other module in the gateway package can import them from one place.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger("maude.gateway")

# ─────────────────────────────────────────────────────────────────
# Tool execution support for cloud model requests
# ─────────────────────────────────────────────────────────────────

try:
    import maude_core  # noqa: F401
    from maude_core import TOOLS as ALL_TOOLS  # noqa: F401
    from maude_core import execute_tool, get_tools_for_message, reset_rate_limits
    from maude_core.tools_plan import PARALLEL_SAFE

    TOOL_SUPPORT = True
    logger.info("Tool support enabled via maude_core")
except ImportError as e:
    logger.warning("maude_core not available, tool support disabled: %s", e)
    TOOL_SUPPORT = False
    execute_tool = None
    get_tools_for_message = None
    reset_rate_limits = None
    PARALLEL_SAFE = set()

# ─────────────────────────────────────────────────────────────────
# Load environment from variables.env
# ─────────────────────────────────────────────────────────────────

# Resolve relative to the project root (parent of gateway/)
_PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = _PROJECT_ROOT / "variables.env"
if ENV_FILE.exists():
    for _line in ENV_FILE.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _val = _line.split("=", 1)
            os.environ.setdefault(_key.strip(), _val.strip())

# keys.json / common env names vs the names MODEL_ROUTES expects
if not os.environ.get("CLAUDE_API_KEY") and os.environ.get("ANTHROPIC_API_KEY"):
    os.environ["CLAUDE_API_KEY"] = os.environ["ANTHROPIC_API_KEY"]
if not os.environ.get("CODESTRAL_API_KEY") and os.environ.get("MISTRAL_API_KEY"):
    os.environ["CODESTRAL_API_KEY"] = os.environ["MISTRAL_API_KEY"]

# ─────────────────────────────────────────────────────────────────
# Directory and port configuration
# ─────────────────────────────────────────────────────────────────

SHARED_DIR = Path.home() / "nvidia-workbench" / "terminal-llm" / "shared"
TRANSFERS_DIR = Path.home() / "nvidia-workbench" / "terminal-llm" / "transfers"
PWA_DIR = _PROJECT_ROOT / "maude-phone" / "dist"
LLM_PORT = 30010  # llama-server (Nemotron) runs here internally
GATEWAY_PORT = 30000  # clients connect here
VOICE_PORT = 8998  # Voice server

CONVERSATIONS_DIR = _PROJECT_ROOT / "data" / "conversations"

SHARED_DIR.mkdir(parents=True, exist_ok=True)
TRANSFERS_DIR.mkdir(parents=True, exist_ok=True)
CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# Model routing configuration
# ─────────────────────────────────────────────────────────────────

MODEL_ROUTES = {
    # Mistral Large — general conversation (MAUDE default)
    "mistral-large-latest": {
        "provider": "mistral",
        "base_url": "https://api.mistral.ai",
        "api_key_env": "MISTRAL_API_KEY",
        "max_context": 128000,
    },
    # Codestral — code tasks (uses separate endpoint + key)
    "codestral-latest": {
        "provider": "mistral",
        "base_url": "https://codestral.mistral.ai",
        "api_key_env": "CODESTRAL_API_KEY",
        "max_context": 32000,
    },
    # Devstral 2 — frontier code agents (123B, 256K context)
    "devstral-2512": {
        "provider": "mistral",
        "base_url": "https://api.mistral.ai",
        "api_key_env": "MISTRAL_API_KEY",
        "max_context": 256000,
    },
    # Devstral Small — lightweight code agents
    "devstral-small-latest": {
        "provider": "mistral",
        "base_url": "https://api.mistral.ai",
        "api_key_env": "MISTRAL_API_KEY",
        "max_context": 32000,
    },
    # Devstral Medium — mid-tier code agents
    "devstral-medium-latest": {
        "provider": "mistral",
        "base_url": "https://api.mistral.ai",
        "api_key_env": "MISTRAL_API_KEY",
        "max_context": 128000,
    },
    # OpenAI API route. Requires OPENAI_API_KEY.
    os.environ.get("MAUDE_OPENAI_MODEL", "gpt-4o"): {
        "provider": "openai",
        "base_url": "https://api.openai.com",
        "api_key_env": "OPENAI_API_KEY",
        "max_context": int(os.environ.get("MAUDE_OPENAI_MAX_CONTEXT", "128000")),
    },
    # Codex CLI route. Uses the locally authenticated Codex CLI account instead
    # of an OpenAI API key.
    "codex-cli": {
        "provider": "codex-cli",
        "base_url": "",
        "api_key_env": None,
        "max_context": int(os.environ.get("MAUDE_CODEX_MAX_CONTEXT", "128000")),
    },
    # Grok CLI route. Uses the locally authenticated Grok CLI (X Premium /
    # grok.com OAuth session) instead of an XAI_API_KEY, so no API billing.
    "grok-4.6": {
        "provider": "grok-cli",
        "base_url": "",
        "api_key_env": None,
        "max_context": int(os.environ.get("MAUDE_GROK_MAX_CONTEXT", "256000")),
    },
    "grok-4.5": {
        "provider": "grok-cli",
        "base_url": "",
        "api_key_env": None,
        "max_context": int(os.environ.get("MAUDE_GROK_MAX_CONTEXT", "256000")),
    },
    # Nemotron 3 Super — cloud via OpenRouter (free)
    "nvidia/nemotron-3-super-120b-a12b:free": {
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api",
        "api_key_env": "OPEN_ROUTER_API_KEY",
        "max_context": 1000000,
    },
    # Nemotron 3 Nano 30B-A3B — efficient reasoning/agent model via OpenRouter
    "nvidia/nemotron-3-nano-30b-a3b": {
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api",
        "api_key_env": "OPEN_ROUTER_API_KEY",
        "max_context": 262144,
    },
    # Nemotron — local fallback (llama-server)
    "nemotron": {
        "provider": "local",
        "base_url": f"http://localhost:{LLM_PORT}",
        "api_key_env": None,
        "max_context": 8192,
    },
    # LLaVA — vision (local)
    "llava": {
        "provider": "local",
        "base_url": f"http://localhost:{LLM_PORT}",
        "api_key_env": None,
        "max_context": 4096,
    },
    # Hermes 3 70B — local (llama-server on 30011), strong tool use
    "hermes": {
        "provider": "local",
        "base_url": "http://localhost:30011",
        "api_key_env": None,
        "max_context": 8192,
    },
    # Gemma 4 31B — local (llama-server on 30013), multimodal, 256K context
    "gemma-4-31b": {
        "provider": "local",
        "base_url": "http://localhost:30013",
        "api_key_env": None,
        "max_context": 32768,
    },
    # Claude Opus — most capable, deep reasoning
    "claude-opus-4-20250514": {
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com",
        "api_key_env": "CLAUDE_API_KEY",
        "max_context": 200000,
    },
    # Claude Sonnet — fast, capable, cost-effective
    "claude-sonnet-4-20250514": {
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com",
        "api_key_env": "CLAUDE_API_KEY",
        "max_context": 200000,
    },
}

# Aliases for convenience
MODEL_ALIASES = {
    "mistral": "mistral-large-latest",
    "codestral": "codestral-latest",
    "devstral": "devstral-2512",
    "devstral-small": "devstral-small-latest",
    "devstral-medium": "devstral-medium-latest",
    "openai": os.environ.get("MAUDE_OPENAI_MODEL", "gpt-4o"),
    "codex": "codex-cli",
    "nemotron-super": "nvidia/nemotron-3-super-120b-a12b:free",
    "nemotron-a3b": "nvidia/nemotron-3-nano-30b-a3b",
    "a3b": "nvidia/nemotron-3-nano-30b-a3b",
    "local": "nemotron",
    "vision": "llava",
    "hermes": "hermes",
    "hermes3": "hermes",
    "gemma4": "gemma-4-31b",
    "gemma": "gemma-4-31b",
    "claude": "claude-opus-4-20250514",
    "sonnet": "claude-sonnet-4-20250514",
    "grok": "grok-4.6",
    "grok4": "grok-4.6",
    "grok-4.5": "grok-4.6",
}


def get_model_route(model_name):
    """Resolve model name to routing config."""
    name = MODEL_ALIASES.get(model_name, model_name)
    return name, MODEL_ROUTES.get(name)


# ─────────────────────────────────────────────────────────────────
# Mutable shared state (used by server and websocket modules)
# ─────────────────────────────────────────────────────────────────

# HTTP terminal sessions (iOS fallback): session_id -> {master_fd, pid, created}
http_terminal_sessions = {}

# Chat sessions for iOS EventSource fallback: session_id -> {"req": dict, "created": float}
chat_sessions = {}

# Latest phone GPS: {"lat": float, "lng": float, "accuracy": float, "ts": float}
device_location = {}

# ─────────────────────────────────────────────────────────────────
# Tool addendum — injected into system prompts for tool-capable models
# ─────────────────────────────────────────────────────────────────

TOOL_ADDENDUM = (
    "\n\nYou have access to tools for file operations, shell commands, web browsing, and more "
    "on this DGX Spark system. When the user asks about files, directories, code, or system "
    "information, USE THE TOOLS to get real data. Do NOT guess or make up file contents, "
    "directory listings, or system information. The working directory defaults to the user's "
    "home directory. Workbench projects are in ~/nvidia-workbench/. "
    "\n\nEFFICIENCY: When you need multiple pieces of information, call ALL the tools you "
    "need in a single response rather than one at a time. For example, if you need to read "
    "3 files, call read_file 3 times in one response. Similarly batch web_search, gmail_read, "
    "drive_read, github calls, etc. Only chain sequentially when a later call depends on an "
    "earlier result. "
    "\n\nCRITICAL RULES — FOLLOW THESE EXACTLY: "
    "1. DO the work. NEVER tell the user to run commands themselves. You have run_command — use it. "
    "2. VERIFY your work. After writing code: run the build/compile step and check for errors. "
    "After starting a server: curl it and confirm it responds. After creating files: check they exist. "
    "3. FIX errors yourself. If a build fails, read the error, fix the code, and rebuild. "
    "Do NOT just report the error and stop. "
    "4. NEVER give tutorials or step-by-step instructions for things you can do yourself. "
    "If the user asks for a URL, start the server and give them the working URL. "
    "5. NEVER guess system info. Use run_command to check IPs, ports, processes, etc. "
    "6. When serving web apps: bind to 0.0.0.0, check which ports are free first "
    "(run_command: ss -tlnp), and give the Tailscale URL (run_command: tailscale ip -4). "
    "7. Complete the ENTIRE task. Don't stop halfway and describe what remains. "
    "If building a site, it should be built, running, and accessible before you respond. "
    "8. DO NOT ask the user for permission at every step. When given a task, figure it out "
    "and DO IT. Use the tools and information you already have. Only ask when you genuinely "
    "cannot proceed without info you have no way to obtain (e.g. a password). "
    "'Should I proceed?' and 'Do you want me to...?' are almost never appropriate — just do the work. "
    "\n\nIf the user has attached an image, use the view_image tool to analyze it before responding. "
    "The view_image tool uses your own native multimodal vision when available (Claude, Mistral Large), "
    "falling back to local LLaVA otherwise. "
    "IMPORTANT: When a user sends a photo from the phone app, the image is ALREADY uploaded "
    "and saved on this server in /home/mboard76/nvidia-workbench/terminal-llm/shared/. "
    "The image path is provided in the message. Do NOT tell the user to upload it — it is "
    "already here. If the user asks to 'upload to the server' or 'save to the server', "
    "the file is already on the server in the shared/ folder. You can move or copy it "
    "elsewhere if they want, but acknowledge it is already present. "
    "\n\nYou also have full access to Google Workspace: Gmail (read, send), Google Drive "
    "(list, search, read, upload, create docs, create folders), Sheets (read, write, create), "
    "Calendar (list, create, update events), Contacts, YouTube, and Slides. Use these tools "
    "when the user asks about their email, documents, spreadsheets, schedule, or contacts. "
    "CRITICAL: ALWAYS use the provided tool functions for Google operations. NEVER write "
    "Python scripts or code that calls Google APIs directly via run_command. The tools "
    "already handle authentication and API calls. For example, to upload a file to Drive, "
    "use drive_upload — do NOT write a Python script that imports googleapiclient. "
    "To repeat the same operation on multiple files, call the same tool multiple times. "
    "\n\nYou can SEARCH THE WEB using the web_search tool (DuckDuckGo) and READ WEB PAGES "
    "using web_browse. ONLY use web_search when the user explicitly asks you to search or "
    "look something up, or when you genuinely need current external information (news, prices, "
    "docs) to complete the task. Do NOT web search as a first step. For tasks like scheduling, "
    "posting, file operations, coding, or using existing tools — just do the task directly. "
    "Do NOT say you cannot search the web — you CAN when relevant, but don't use it reflexively."
    "\n\nIMAGE DISPLAY: When you use generate_image, share_file, or web_image_search and want "
    "to display images inline, include the markdown ![description](url) in your response. "
    "For generated/shared images use ![description](/download/filename.png). "
    "For web images use the full URL from the search results."
    "\n\nBROWSER: The user has a persistent browser session (Playwright) that stays logged into "
    "social media and other sites. For social media posting, especially X/Twitter posts with "
    "images or videos, ALWAYS use social_post. Do NOT use browser_type/browser_click to compose "
    "or click Post on X; that path cannot reliably attach media. Use browser_navigate and browser "
    "tools for non-posting site actions such as searching LinkedIn or checking Facebook. Use the "
    "EXISTING session — do NOT open a new browser or ask them to log in again. Only use browser_login "
    "when the user explicitly asks to log in or when browser_check_session confirms the session "
    "is expired. browser_login accepts shorthand names (e.g. 'x', 'linkedin', 'instagram'). "
    "\n\nBROWSER WORKFLOWS: workflow_create, workflow_run, workflow_list, workflow_schedule — "
    "create repeatable browser automations with change detection and email notifications. "
    "\n\nPERSISTENT MEMORY: You have save_memory and recall_memory tools "
    "available on EVERY request. Use save_memory PROACTIVELY when the user "
    "shares personal facts, preferences, names, dates, project details, or "
    "anything worth remembering across sessions. Do NOT ask permission to save "
    "— just save it. Use recall_memory when the user references something from "
    "a previous conversation or when context from past sessions would help. "
    "Categories: fact, preference, person, task."
)
