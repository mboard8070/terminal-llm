"""
Rate limiting counters — reset per turn by caller.
"""

from .cache import _tool_cache

vision_call_count = 0
web_call_count = 0
claude_call_count = 0


def reset_rate_limits():
    """Reset per-turn rate limits. Call at start of each user turn."""
    global claude_call_count, vision_call_count, web_call_count
    claude_call_count = 0
    vision_call_count = 0
    web_call_count = 0
    _tool_cache.evict_expired()
