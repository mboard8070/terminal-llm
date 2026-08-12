"""
MAUDE Core — Shared tools and functionality.

This package provides the core tool implementations used by both
the TUI (chat_local.py) and Telegram (run_telegram.py) interfaces.

All public names are re-exported here for backward compatibility:
    from maude_core import TOOLS, execute_tool, ...
"""

# ── Configuration ──────────────────────────────────────────────
# ── Register prefix-based handlers ────────────────────────────
from tool_registry import register_prefix

# ── Cache ──────────────────────────────────────────────────────
from .cache import ToolCache, _tool_cache

# ── Chat Sync ──────────────────────────────────────────────────
from .chat_sync import CHAT_LOG_PATH, append_chat_log, read_chat_log_since
from .config import LOCAL_URL, MODEL, NUM_CTX, SESSION_ID, VISION_MODEL, VISION_URL

# ── Fast Dispatch ──────────────────────────────────────────────
from .fast_dispatch import (  # noqa: F401
    fast_dispatch,
    get_fast_dispatch_stats,
    match_fast_dispatch,
    reset_fast_dispatch_stats,
)

# ── Logging ────────────────────────────────────────────────────
from .log import log, set_log_callback

# ── Memory ─────────────────────────────────────────────────────
from .memory_utils import build_messages_with_history, get_conversation_history, get_memory, save_message

# ── Context hygiene ────────────────────────────────────────────
from .context_hygiene import (  # noqa: F401
    apply_hygiene_in_place,
    compact_tool_result,
    get_mission_scratch,
    prepare_messages_for_model,
)

# ── Paths ──────────────────────────────────────────────────────
from .paths import get_working_directory, resolve_path, set_working_directory, working_dir

# ── Rate Limits ────────────────────────────────────────────────
from .rate_limits import claude_call_count, reset_rate_limits, vision_call_count, web_call_count

# ── Tool Definitions ───────────────────────────────────────────
from .tool_defs import TOOLS

# ── Tool Groups & Filtering ───────────────────────────────────
from .tool_groups import (
    _CORE_TOOL_NAMES,
    _RARE_TIER_GROUPS,
    _SESSION_TIER_GROUPS,
    _TOOL_BY_NAME,
    _TOOL_GROUPS,
    activate_domain,
    clear_session_domains,
    get_session_domains,
    get_tools_for_message,
    list_domain_catalog,
    payload_stats,
)

register_prefix("browser_", "browser", "execute_browser_tool")
register_prefix("workflow_", "browser_workflows", "execute_workflow_tool")
register_prefix("social_", "social_posting", "execute_social_tool")


def _skill_prefix_handler(tool_name, arguments):
    """Handle skill_* tools by stripping prefix and delegating to skill manager."""
    from skills import get_skill_manager

    skill_name = tool_name[6:]  # Strip "skill_" prefix
    mgr = get_skill_manager()
    return mgr.execute_skill(skill_name, **arguments)


register_prefix("skill_", handler=_skill_prefix_handler)

# ── Import tool modules to trigger @register_tool decorators ──
from . import (
    tools_agents,
    tools_ai,
    tools_collab,
    tools_file,
    tools_github,
    tools_google,
    tools_hyperframes,
    tools_media,
    tools_memory,
    tools_schedule,
    tools_shared,
    tools_substack,
    tools_web,
)

try:
    from . import tools_sandbox
except ImportError:
    pass
try:
    from . import tools_forge
except ImportError:
    pass
try:
    from . import tools_forge3d
except ImportError:
    pass
from . import (
    tools_command_center,
    tools_plan,
)

# ── Execution ──────────────────────────────────────────────────
from .execute import execute_tool
from .tools_ai import tool_ask_frontier, tool_send_to_claude

# ── Expose tool implementation functions (used by some modules) ─
from .tools_file import (
    tool_change_directory,
    tool_edit_file,
    tool_get_working_directory,
    tool_list_directory,
    tool_read_file,
    tool_run_command,
    tool_search_directory,
    tool_search_file,
    tool_write_file,
)
from .tools_media import tool_generate_image
from .tools_schedule import tool_schedule_task
from .tools_shared import (
    SHARED_DIR,
    TRANSFERS_DIR,
    tool_get_transfer,
    tool_list_shared,
    tool_list_transfers,
    tool_share_file,
)
from .tools_web import tool_view_image, tool_web_browse, tool_web_search, tool_web_view
