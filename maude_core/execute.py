"""
Tool execution — registry-based dispatch with pre-flight checks,
rate limiting, and caching.
"""

from tool_registry import get_handler, is_cacheable

from . import rate_limits
from .cache import ToolCache, _tool_cache
from .log import log


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool and return the result."""

    # Pre-flight readiness check
    try:
        from health import check_tool_ready

        ready, reason = check_tool_ready(name)
        if not ready:
            return f"Tool '{name}' unavailable: {reason}. Tell the user this tool isn't currently available and suggest alternatives."
    except ImportError:
        pass  # health module not available — skip check

    # Rate limiting for Claude calls - prevent loops
    if name == "send_to_claude":
        rate_limits.claude_call_count += 1
        if rate_limits.claude_call_count > 2:
            return "STOP: Already contacted Claude twice this turn. Report the results to the user and wait for their next message."

    # Rate limiting
    if name in ("web_view", "view_image"):
        if rate_limits.vision_call_count > 0:
            return "(Vision analysis already completed - see previous result.)"
        rate_limits.vision_call_count += 1

    if name in ("web_browse", "web_search"):
        if rate_limits.web_call_count >= 4:
            return "(Web research limit reached. Use gathered information now.)"
        rate_limits.web_call_count += 1

    # Check tool result cache for expensive operations
    if name in ToolCache.TTLS:
        cached = _tool_cache.get(name, arguments)
        if cached is not None:
            log(f"[cached result] {name}")
            return cached + "\n\n[cached result]"

    # Registry-based dispatch
    handler = get_handler(name)
    if handler is None:
        return f"Error: Unknown tool: {name}"

    result = handler(arguments)

    # Cache if the tool is cacheable
    if is_cacheable(name):
        _tool_cache.put(name, arguments, result)

    return result
