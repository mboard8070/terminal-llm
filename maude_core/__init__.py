"""
MAUDE Core — Shared tools and functionality.

This package provides the core tool implementations used by both
the TUI (chat_local.py) and Telegram (run_telegram.py) interfaces.

All public names are re-exported here for backward compatibility:
    from maude_core import TOOLS, execute_tool, ...
"""

# ── Configuration ──────────────────────────────────────────────
from .config import LOCAL_URL, MODEL, NUM_CTX, VISION_URL, VISION_MODEL, SESSION_ID

# ── Cache ──────────────────────────────────────────────────────
from .cache import ToolCache, _tool_cache

# ── Logging ────────────────────────────────────────────────────
from .log import set_log_callback, log

# ── Paths ──────────────────────────────────────────────────────
from .paths import working_dir, set_working_directory, get_working_directory, resolve_path

# ── Chat Sync ──────────────────────────────────────────────────
from .chat_sync import CHAT_LOG_PATH, append_chat_log, read_chat_log_since

# ── Memory ─────────────────────────────────────────────────────
from .memory_utils import get_memory, save_message, get_conversation_history, build_messages_with_history

# ── Rate Limits ────────────────────────────────────────────────
from .rate_limits import reset_rate_limits, vision_call_count, web_call_count, claude_call_count

# ── Tool Definitions ───────────────────────────────────────────
from .tool_defs import TOOLS

# ── Tool Groups & Filtering ───────────────────────────────────
from .tool_groups import (
    _CORE_TOOL_NAMES, _TOOL_GROUPS, _TOOL_BY_NAME, get_tools_for_message,
)

# ── Fast Dispatch ──────────────────────────────────────────────
from .fast_dispatch import fast_dispatch

# ── Register prefix-based handlers ────────────────────────────
from tool_registry import register_prefix

register_prefix("browser_", "browser", "execute_browser_tool")

def _skill_prefix_handler(tool_name, arguments):
    """Handle skill_* tools by stripping prefix and delegating to skill manager."""
    from skills import get_skill_manager
    skill_name = tool_name[6:]  # Strip "skill_" prefix
    mgr = get_skill_manager()
    return mgr.execute_skill(skill_name, **arguments)

register_prefix("skill_", handler=_skill_prefix_handler)

# ── Import tool modules to trigger @register_tool decorators ──
from . import tools_file      # noqa: F401
from . import tools_web       # noqa: F401
from . import tools_ai        # noqa: F401
from . import tools_shared    # noqa: F401
from . import tools_media     # noqa: F401
from . import tools_schedule  # noqa: F401
from . import tools_google    # noqa: F401
from . import tools_substack  # noqa: F401
from . import tools_collab    # noqa: F401
from . import tools_memory    # noqa: F401
from . import tools_github    # noqa: F401
from . import tools_agents    # noqa: F401

# ── Execution ──────────────────────────────────────────────────
from .execute import execute_tool

# ── Expose tool implementation functions (used by some modules) ─
from .tools_file import (
    tool_read_file, tool_write_file, tool_search_file, tool_search_directory,
    tool_edit_file, tool_list_directory, tool_get_working_directory,
    tool_change_directory, tool_run_command,
)
from .tools_web import tool_web_browse, tool_web_search, tool_web_view, tool_view_image
from .tools_ai import tool_ask_frontier, tool_send_to_claude
from .tools_shared import (
    SHARED_DIR, TRANSFERS_DIR,
    tool_list_shared, tool_share_file, tool_list_transfers, tool_get_transfer,
)
from .tools_media import tool_generate_image
from .tools_schedule import tool_schedule_task
