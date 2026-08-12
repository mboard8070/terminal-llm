"""
Memory instance management — lazy-loaded shared memory.
"""

from typing import Any

from .config import SESSION_ID
from .log import log

# Memory instance (lazy loaded)
_memory = None


def get_memory():
    """Get or create the shared memory instance."""
    global _memory
    if _memory is None:
        try:
            from memory import MaudeMemory

            _memory = MaudeMemory()
        except Exception as e:
            log(f"Memory init failed: {e}")
            return None
    return _memory


def save_message(role: str, content: str, channel: str = "cli"):
    """Save a message to shared conversation history."""
    mem = get_memory()
    if mem and content:
        try:
            mem.save_message(SESSION_ID, role, content, channel)
        except Exception as e:
            log(f"Failed to save message: {e}")


def get_conversation_history(limit: int = 10) -> list[dict[str, Any]]:
    """Get recent conversation history from all channels."""
    mem = get_memory()
    if mem:
        try:
            return mem.get_conversation(SESSION_ID, limit)
        except Exception as e:
            log(f"Failed to get history: {e}")
    return []


def build_messages_with_history(system_prompt: str, user_message: str, history_limit: int = 10) -> list[dict[str, str]]:
    """Build messages list with system prompt, recent history, and current message.

    Applies context hygiene (sliding window + body caps) so long cross-channel
    histories do not blow the prompt.
    """
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    # Add recent history (fetch a bit extra; hygiene will compress)
    history = get_conversation_history(max(history_limit, 20))
    for msg in history:
        # Only include user and assistant messages (skip system, tool)
        if msg["role"] in ("user", "assistant") and msg["content"]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current message
    messages.append({"role": "user", "content": user_message})

    try:
        from .context_hygiene import keep_recent_turns, prepare_messages_for_model

        keep = min(history_limit * 2, keep_recent_turns())
        prepared, _meta = prepare_messages_for_model(
            messages,
            keep_recent=keep,
            keep_tool_rounds_n=0,
            in_place=False,
        )
        return prepared
    except Exception:
        # Fall back to a simple tail window
        sys_msgs = [m for m in messages if m.get("role") == "system"]
        rest = [m for m in messages if m.get("role") != "system"]
        return sys_msgs + rest[-history_limit:]
