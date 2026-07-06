"""
Tool execution — registry-based dispatch with pre-flight checks,
rate limiting, and caching.
"""

from maude.observability import RunContext, emit_event, observed_span, record_metric
from maude_core.cache import ToolCache, _tool_cache
from maude_core.log import log
from tool_registry import get_handler, is_cacheable

from . import tool_rate_limits as rate_limits
from .retries import RetryPolicy, run_with_retries


def execute_tool(name: str, arguments: dict, retry_policy: RetryPolicy | None = None, run_context: RunContext | None = None) -> str:
    """Execute a tool and return the result."""
    context = run_context or RunContext.from_mapping(arguments.get("_context") if isinstance(arguments, dict) else None)

    # Pre-flight readiness check
    try:
        from health import check_tool_ready

        ready, reason = check_tool_ready(name)
        if not ready:
            emit_event("tool.unavailable", context, tool=name, reason=reason)
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
            record_metric("tool.cache_hits")
            emit_event("tool.cache_hit", context, tool=name)
            return cached + "\n\n[cached result]"

    # Registry-based dispatch
    handler = get_handler(name)
    if handler is None:
        emit_event("tool.unknown", context, tool=name)
        return f"Error: Unknown tool: {name}"

    active_policy = retry_policy or RetryPolicy(max_attempts=1, initial_delay_seconds=0.0)
    with observed_span("tool.execution", context, tool=name):
        result = run_with_retries(lambda: handler(arguments), active_policy)
    record_metric("tools.executed")

    # Cache if the tool is cacheable
    if is_cacheable(name):
        _tool_cache.put(name, arguments, result)

    return result
