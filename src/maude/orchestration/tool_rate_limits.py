"""Per-turn tool loop rate-limit counters."""

vision_call_count = 0
web_call_count = 0
claude_call_count = 0


def reset_rate_limits():
    """Reset per-turn rate limits and expire cached tool results."""

    global claude_call_count, vision_call_count, web_call_count
    claude_call_count = 0
    vision_call_count = 0
    web_call_count = 0
    try:
        from maude_core.cache import _tool_cache

        _tool_cache.evict_expired()
    except Exception:
        pass
