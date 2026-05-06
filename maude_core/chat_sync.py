"""
Lightweight Chat Sync — file-based, non-blocking message log.
"""

import json
from datetime import datetime
from pathlib import Path

CHAT_LOG_PATH = Path.home() / ".config" / "maude" / "chat_sync.jsonl"


def append_chat_log(channel: str, role: str, content: str):
    """Append a message to the shared chat log. Fast, non-blocking."""
    try:
        CHAT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {"ts": datetime.now().isoformat(), "channel": channel, "role": role, "content": content}
        with open(CHAT_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # Non-critical, don't break anything


def read_chat_log_since(last_position: int = 0) -> tuple:
    """Read new entries from chat log since position. Returns (entries, new_position)."""
    entries = []
    new_position = last_position
    try:
        if not CHAT_LOG_PATH.exists():
            return [], 0
        with open(CHAT_LOG_PATH) as f:
            f.seek(last_position)
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except:
                        pass
            new_position = f.tell()
    except Exception:
        pass
    return entries, new_position
