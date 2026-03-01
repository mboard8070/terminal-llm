"""
MAUDE Core - Shared tools and functionality.

This module provides the core tool implementations used by both
the TUI (chat_local.py) and Telegram (run_telegram.py) interfaces.
"""

import os
import json
import time
import subprocess
from pathlib import Path
from typing import Callable, Optional, List, Dict, Any
from openai import OpenAI

# Configuration - can be overridden via environment variables
LOCAL_URL = os.environ.get("LLM_SERVER_URL", "http://localhost:30080/v1")
MODEL = os.environ.get("MAUDE_MODEL", "nemotron")
NUM_CTX = int(os.environ.get("MAUDE_NUM_CTX", "32768"))
VISION_URL = os.environ.get("VISION_SERVER_URL", "http://localhost:11434/v1")
VISION_MODEL = os.environ.get("MAUDE_VISION_MODEL", "llava:13b")

# Shared session ID for conversation sync
SESSION_ID = os.environ.get("MAUDE_SESSION_ID", "default")

# Working directory for file operations
working_dir = Path.home()

# ─── Tool Result Cache ───────────────────────────────────────────────────────

class ToolCache:
    """TTL-based cache for expensive tool results (web, vision)."""

    # Per-tool TTLs in seconds
    TTLS = {
        "web_browse": 1800,   # 30 min
        "web_search": 1800,   # 30 min
        "web_view":   1800,   # 30 min
        "view_image": 300,    # 5 min
    }

    def __init__(self):
        self._store: Dict[tuple, tuple] = {}  # (tool, args_key) → (result, expiry)

    @staticmethod
    def _make_key(tool_name: str, arguments: dict) -> tuple:
        return (tool_name, json.dumps(arguments, sort_keys=True))

    def get(self, tool_name: str, arguments: dict) -> Optional[str]:
        """Return cached result or None."""
        key = self._make_key(tool_name, arguments)
        entry = self._store.get(key)
        if entry is None:
            return None
        result, expiry = entry
        if time.time() > expiry:
            del self._store[key]
            return None
        return result

    def put(self, tool_name: str, arguments: dict, result: str):
        """Store a result with tool-specific TTL."""
        ttl = self.TTLS.get(tool_name)
        if ttl is None:
            return  # not a cacheable tool
        key = self._make_key(tool_name, arguments)
        self._store[key] = (result, time.time() + ttl)

    def evict_expired(self):
        """Remove all expired entries."""
        now = time.time()
        expired = [k for k, (_, exp) in self._store.items() if now > exp]
        for k in expired:
            del self._store[k]

    def clear(self):
        self._store.clear()


_tool_cache = ToolCache()

# Optional logging callback - set by the interface (TUI or Telegram)
# Signature: log_callback(message: str) -> None
_log_callback: Optional[Callable[[str], None]] = None

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


def get_conversation_history(limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent conversation history from all channels."""
    mem = get_memory()
    if mem:
        try:
            return mem.get_conversation(SESSION_ID, limit)
        except Exception as e:
            log(f"Failed to get history: {e}")
    return []


def build_messages_with_history(system_prompt: str, user_message: str, history_limit: int = 10) -> List[Dict[str, str]]:
    """Build messages list with system prompt, recent history, and current message."""
    messages = [{"role": "system", "content": system_prompt}]

    # Add recent history
    history = get_conversation_history(history_limit)
    for msg in history:
        # Only include user and assistant messages (skip system, tool)
        if msg["role"] in ("user", "assistant") and msg["content"]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current message
    messages.append({"role": "user", "content": user_message})

    return messages


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight Chat Sync (file-based, non-blocking)
# ─────────────────────────────────────────────────────────────────────────────

CHAT_LOG_PATH = Path.home() / ".config" / "maude" / "chat_sync.jsonl"


def append_chat_log(channel: str, role: str, content: str):
    """Append a message to the shared chat log. Fast, non-blocking."""
    import json
    from datetime import datetime
    try:
        CHAT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now().isoformat(),
            "channel": channel,
            "role": role,
            "content": content
        }
        with open(CHAT_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # Non-critical, don't break anything


def read_chat_log_since(last_position: int = 0) -> tuple:
    """Read new entries from chat log since position. Returns (entries, new_position)."""
    import json
    entries = []
    new_position = last_position
    try:
        if not CHAT_LOG_PATH.exists():
            return [], 0
        with open(CHAT_LOG_PATH, "r") as f:
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


def set_log_callback(callback: Callable[[str], None]):
    """Set the logging callback for tool status messages."""
    global _log_callback
    _log_callback = callback


def log(message: str):
    """Log a status message if callback is set."""
    if _log_callback:
        _log_callback(message)


def set_working_directory(path: Path):
    """Set the working directory for file operations."""
    global working_dir
    working_dir = path


def get_working_directory() -> Path:
    """Get the current working directory."""
    return working_dir


def resolve_path(path_str: str) -> Path:
    """Resolve a path relative to working directory."""
    global working_dir
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = working_dir / path
    return path.resolve()


# ─────────────────────────────────────────────────────────────────────────────
# Tool Definitions
# ─────────────────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file with line numbers. Use start_line/end_line for large files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "start_line": {"type": "integer", "description": "First line to read (1-indexed, optional)"},
                    "end_line": {"type": "integer", "description": "Last line to read (1-indexed, optional)"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "content": {"type": "string", "description": "Content to write to the file"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories in a path. Shows file sizes and types.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to list (defaults to working directory)"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_working_directory",
            "description": "Get the current working directory path.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "change_directory",
            "description": "Change the current working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to change to"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command. Use for: pip, python, git, rm, mv, cp, etc. Also use to remove files from the shared/ or transfers/ folder when asked to clean up.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_browse",
            "description": "Fetch and read content from a web URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {"type": "integer", "description": "Number of results (default 5, max 10)"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_view",
            "description": "Screenshot a webpage and analyze it visually using LLaVA.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to screenshot"},
                    "question": {"type": "string", "description": "Optional question about the page"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "view_image",
            "description": "Analyze a local image file using LLaVA vision model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the image file"},
                    "question": {"type": "string", "description": "Optional question about the image"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate an image using Flux AI via ComfyUI. Saves the image to the shared folder and returns a download URL for display. Use markdown ![description](/download/filename.png) to show it in chat.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Text description of the image to generate"},
                    "width": {"type": "integer", "description": "Image width (default 1024)"},
                    "height": {"type": "integer", "description": "Image height (default 1024)"},
                    "seed": {"type": "integer", "description": "Seed for reproducibility (-1 for random)"},
                    "steps": {"type": "integer", "description": "Sampling steps (default 28)"},
                    "lora": {"type": "string", "description": "Optional LoRA name: stillion-style, marker-mech-style"},
                },
                "required": ["prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_file",
            "description": "Search for text/pattern in a single file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "pattern": {"type": "string", "description": "Text to search for"}
                },
                "required": ["path", "pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_directory",
            "description": "Search for text across all files in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Directory to search"},
                    "pattern": {"type": "string", "description": "Text to search for"}
                },
                "required": ["directory", "pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit specific lines in a file. Read the file first to see line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "start_line": {"type": "integer", "description": "First line to replace (1-indexed)"},
                    "end_line": {"type": "integer", "description": "Last line to replace (1-indexed)"},
                    "new_content": {"type": "string", "description": "New content to insert"}
                },
                "required": ["path", "start_line", "end_line", "new_content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ask_frontier",
            "description": "Escalate to a frontier AI model (Claude, GPT, Gemini) for complex questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question requiring expert analysis"},
                    "context": {"type": "string", "description": "Relevant context"},
                    "provider": {"type": "string", "description": "Optional: claude, openai, gemini, grok, mistral"}
                },
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_to_claude",
            "description": "Send a message to Claude Code running in tmux. Use this to delegate complex coding tasks, get expert analysis, or have Claude work on files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "The message/task to send to Claude Code"},
                    "session": {"type": "string", "description": "tmux session name (default: 'claude')"}
                },
                "required": ["message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_task",
            "description": """Schedule automated tasks. Convert natural language schedules to cron expressions:
- "every morning" or "daily at 8am" → 0 8 * * *
- "every hour" → 0 * * * *
- "weekdays at 9am" → 0 9 * * 1-5
- "every evening at 6pm" → 0 18 * * *
- "weekly on Monday" → 0 9 * * 1
- "every 30 minutes" → */30 * * * *

Shortcuts: @hourly, @daily, @morning, @evening, @weekly, @workdays

Actions: add (create task), list (show all), remove (delete), enable, disable, run (execute now)""",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "Action: add, list, remove, enable, disable, run", "enum": ["add", "list", "remove", "enable", "disable", "run"]},
                    "name": {"type": "string", "description": "Descriptive name for the task (for add action)"},
                    "cron": {"type": "string", "description": "Cron expression or shortcut like @daily, @morning (for add action)"},
                    "prompt": {"type": "string", "description": "What MAUDE should do when triggered (for add action)"},
                    "task_id": {"type": "string", "description": "Task ID (for remove/enable/disable/run actions)"}
                },
                "required": ["action"]
            }
        }
    },
    # Google Tools
    {
        "type": "function",
        "function": {
            "name": "gmail_list",
            "description": "List recent emails from Gmail. Use query for searching (same syntax as Gmail search).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (e.g., 'from:someone@example.com', 'subject:invoice', 'is:unread')"},
                    "max_results": {"type": "integer", "description": "Maximum emails to return (default 10)"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "gmail_read",
            "description": "Read a specific email by its message ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message_id": {"type": "string", "description": "The Gmail message ID"}
                },
                "required": ["message_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "gmail_send",
            "description": "Send an email via Gmail.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email address"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body text"},
                    "cc": {"type": "string", "description": "CC recipients (optional)"}
                },
                "required": ["to", "subject", "body"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "drive_list",
            "description": "List files in Google Drive. Use query for filtering.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Drive query (e.g., \"name contains 'report'\")"},
                    "max_results": {"type": "integer", "description": "Maximum files to return (default 20)"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "drive_search",
            "description": "Search Google Drive for files by name or content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search term"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "drive_read",
            "description": "Read the contents of a file from Google Drive (text files, Google Docs, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_id": {"type": "string", "description": "The Google Drive file ID"}
                },
                "required": ["file_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "drive_upload",
            "description": "Upload a local file to Google Drive.",
            "parameters": {
                "type": "object",
                "properties": {
                    "local_path": {"type": "string", "description": "Path to the local file to upload"},
                    "folder_id": {"type": "string", "description": "Optional Drive folder ID to upload into"}
                },
                "required": ["local_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "drive_create_folder",
            "description": "Create a new folder in Google Drive.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name for the new folder"},
                    "parent_id": {"type": "string", "description": "Optional parent folder ID to create inside"}
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "drive_create_doc",
            "description": "Create a new Google Doc in Google Drive.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name for the new document"},
                    "folder_id": {"type": "string", "description": "Optional folder ID to create inside"},
                    "content": {"type": "string", "description": "Optional initial content for the document"}
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "drive_create_sheet",
            "description": "Create a new Google Sheet in Google Drive.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name for the new spreadsheet"},
                    "folder_id": {"type": "string", "description": "Optional folder ID to create inside"}
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "drive_update_doc",
            "description": "Write or append content to an existing Google Doc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "The Google Doc ID"},
                    "content": {"type": "string", "description": "The content to write to the document"},
                    "append": {"type": "boolean", "description": "If true, append to existing content. If false (default), replace all content."}
                },
                "required": ["doc_id", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "drive_delete",
            "description": "Delete a file or folder from Google Drive.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_id": {"type": "string", "description": "The Google Drive file or folder ID to delete"}
                },
                "required": ["file_id"]
            }
        }
    },
    # Google Sheets tools
    {
        "type": "function",
        "function": {
            "name": "sheets_read",
            "description": "Read data from a Google Sheets spreadsheet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "spreadsheet_id": {"type": "string", "description": "The Google Sheets spreadsheet ID"},
                    "range": {"type": "string", "description": "Cell range to read (e.g., 'Sheet1!A1:D10'). Default: 'Sheet1'"}
                },
                "required": ["spreadsheet_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sheets_write",
            "description": "Write data to a Google Sheets spreadsheet (overwrites existing data in range).",
            "parameters": {
                "type": "object",
                "properties": {
                    "spreadsheet_id": {"type": "string", "description": "The Google Sheets spreadsheet ID"},
                    "range": {"type": "string", "description": "Cell range to write to (e.g., 'Sheet1!A1')"},
                    "values": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}, "description": "2D array of values (rows of columns)"}
                },
                "required": ["spreadsheet_id", "range", "values"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sheets_append",
            "description": "Append rows to a Google Sheets spreadsheet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "spreadsheet_id": {"type": "string", "description": "The Google Sheets spreadsheet ID"},
                    "range": {"type": "string", "description": "Range to append after (e.g., 'Sheet1!A1')"},
                    "values": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}, "description": "2D array of rows to append"}
                },
                "required": ["spreadsheet_id", "range", "values"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sheets_create",
            "description": "Create a new Google Sheets spreadsheet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Name for the new spreadsheet"},
                    "folder_id": {"type": "string", "description": "Optional Drive folder ID to create inside"}
                },
                "required": ["title"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sheets_list_sheets",
            "description": "List all sheet tabs in a Google Sheets spreadsheet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "spreadsheet_id": {"type": "string", "description": "The Google Sheets spreadsheet ID"}
                },
                "required": ["spreadsheet_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sheets_clear",
            "description": "Clear a range of cells in a Google Sheets spreadsheet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "spreadsheet_id": {"type": "string", "description": "The Google Sheets spreadsheet ID"},
                    "range": {"type": "string", "description": "Cell range to clear (e.g., 'Sheet1!A1:D10')"}
                },
                "required": ["spreadsheet_id", "range"]
            }
        }
    },
    # Google Calendar tools
    {
        "type": "function",
        "function": {
            "name": "calendar_list_events",
            "description": "List upcoming Google Calendar events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_results": {"type": "integer", "description": "Maximum events to return (default 10)"},
                    "time_min": {"type": "string", "description": "Start time filter (ISO format, e.g., '2025-01-15T00:00:00Z'). Default: now"},
                    "time_max": {"type": "string", "description": "End time filter (ISO format)"},
                    "calendar_id": {"type": "string", "description": "Calendar ID (default: 'primary')"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_create_event",
            "description": "Create a new Google Calendar event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Event title"},
                    "start": {"type": "string", "description": "Start time (ISO format, e.g., '2025-01-15T10:00:00-05:00')"},
                    "end": {"type": "string", "description": "End time (ISO format)"},
                    "description": {"type": "string", "description": "Event description"},
                    "location": {"type": "string", "description": "Event location"},
                    "calendar_id": {"type": "string", "description": "Calendar ID (default: 'primary')"}
                },
                "required": ["summary", "start", "end"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_update_event",
            "description": "Update an existing Google Calendar event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string", "description": "The event ID to update"},
                    "summary": {"type": "string", "description": "New event title"},
                    "start": {"type": "string", "description": "New start time (ISO format)"},
                    "end": {"type": "string", "description": "New end time (ISO format)"},
                    "description": {"type": "string", "description": "New description"},
                    "location": {"type": "string", "description": "New location"},
                    "calendar_id": {"type": "string", "description": "Calendar ID (default: 'primary')"}
                },
                "required": ["event_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_delete_event",
            "description": "Delete a Google Calendar event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string", "description": "The event ID to delete"},
                    "calendar_id": {"type": "string", "description": "Calendar ID (default: 'primary')"}
                },
                "required": ["event_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_search_events",
            "description": "Search Google Calendar events by text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search text"},
                    "max_results": {"type": "integer", "description": "Maximum events to return (default 10)"},
                    "calendar_id": {"type": "string", "description": "Calendar ID (default: 'primary')"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_list_calendars",
            "description": "List all available Google Calendars.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    # Google Slides tools
    {
        "type": "function",
        "function": {
            "name": "slides_get_presentation",
            "description": "Get Google Slides presentation metadata and slide list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "presentation_id": {"type": "string", "description": "The Google Slides presentation ID"}
                },
                "required": ["presentation_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "slides_get_slide",
            "description": "Get text content from a specific slide in a Google Slides presentation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "presentation_id": {"type": "string", "description": "The Google Slides presentation ID"},
                    "slide_index": {"type": "integer", "description": "Slide index (0-based). Default: 0"}
                },
                "required": ["presentation_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "slides_create_presentation",
            "description": "Create a new Google Slides presentation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Title for the new presentation"},
                    "folder_id": {"type": "string", "description": "Optional Drive folder ID to create inside"}
                },
                "required": ["title"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "slides_add_slide",
            "description": "Add a new slide to a Google Slides presentation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "presentation_id": {"type": "string", "description": "The Google Slides presentation ID"},
                    "layout": {"type": "string", "description": "Slide layout: BLANK, TITLE, TITLE_AND_BODY, TITLE_AND_TWO_COLUMNS, TITLE_ONLY, SECTION_HEADER. Default: BLANK"}
                },
                "required": ["presentation_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "slides_add_text",
            "description": "Add a text box to a slide in a Google Slides presentation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "presentation_id": {"type": "string", "description": "The Google Slides presentation ID"},
                    "slide_id": {"type": "string", "description": "The slide object ID to add text to"},
                    "text": {"type": "string", "description": "The text content to add"},
                    "x": {"type": "number", "description": "X position in points (default 100)"},
                    "y": {"type": "number", "description": "Y position in points (default 100)"},
                    "width": {"type": "number", "description": "Text box width in points (default 400)"},
                    "height": {"type": "number", "description": "Text box height in points (default 300)"}
                },
                "required": ["presentation_id", "slide_id", "text"]
            }
        }
    },
    # Google Contacts tools
    {
        "type": "function",
        "function": {
            "name": "contacts_list",
            "description": "List Google Contacts. Optionally search by name or email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_results": {"type": "integer", "description": "Maximum contacts to return (default 20)"},
                    "query": {"type": "string", "description": "Search query to filter contacts"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "contacts_get",
            "description": "Get detailed info for a single Google Contact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "resource_name": {"type": "string", "description": "Contact resource name (e.g., 'people/c1234567890')"}
                },
                "required": ["resource_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "contacts_create",
            "description": "Create a new Google Contact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "given_name": {"type": "string", "description": "First name"},
                    "family_name": {"type": "string", "description": "Last name"},
                    "email": {"type": "string", "description": "Email address"},
                    "phone": {"type": "string", "description": "Phone number"},
                    "organization": {"type": "string", "description": "Company/organization name"}
                },
                "required": ["given_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "contacts_update",
            "description": "Update an existing Google Contact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "resource_name": {"type": "string", "description": "Contact resource name (e.g., 'people/c1234567890')"},
                    "given_name": {"type": "string", "description": "New first name"},
                    "family_name": {"type": "string", "description": "New last name"},
                    "email": {"type": "string", "description": "New email address"},
                    "phone": {"type": "string", "description": "New phone number"}
                },
                "required": ["resource_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "contacts_delete",
            "description": "Delete a Google Contact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "resource_name": {"type": "string", "description": "Contact resource name (e.g., 'people/c1234567890')"}
                },
                "required": ["resource_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "contacts_search",
            "description": "Search Google Contacts by name or email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    # YouTube tools
    {
        "type": "function",
        "function": {
            "name": "youtube_search",
            "description": "Search YouTube for videos, channels, or playlists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Maximum results (default 5)"},
                    "video_type": {"type": "string", "description": "Type: 'video', 'channel', or 'playlist'. Default: 'video'"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "youtube_get_video",
            "description": "Get detailed info about a YouTube video (title, stats, description, duration).",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_id": {"type": "string", "description": "The YouTube video ID"}
                },
                "required": ["video_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "youtube_get_channel",
            "description": "Get YouTube channel info and stats.",
            "parameters": {
                "type": "object",
                "properties": {
                    "channel_id": {"type": "string", "description": "The YouTube channel ID"}
                },
                "required": ["channel_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "youtube_list_playlists",
            "description": "List YouTube playlists. If no channel_id, lists your own playlists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "channel_id": {"type": "string", "description": "Channel ID (omit for your own playlists)"},
                    "max_results": {"type": "integer", "description": "Maximum results (default 10)"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "youtube_get_playlist_items",
            "description": "List videos in a YouTube playlist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "playlist_id": {"type": "string", "description": "The playlist ID"},
                    "max_results": {"type": "integer", "description": "Maximum results (default 20)"}
                },
                "required": ["playlist_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "youtube_create_playlist",
            "description": "Create a new YouTube playlist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Playlist title"},
                    "description": {"type": "string", "description": "Playlist description"},
                    "privacy": {"type": "string", "description": "Privacy: 'public', 'private', or 'unlisted'. Default: 'private'"}
                },
                "required": ["title"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "youtube_add_to_playlist",
            "description": "Add a video to a YouTube playlist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "playlist_id": {"type": "string", "description": "The playlist ID"},
                    "video_id": {"type": "string", "description": "The video ID to add"}
                },
                "required": ["playlist_id", "video_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "youtube_get_comments",
            "description": "Get comments on a YouTube video.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_id": {"type": "string", "description": "The video ID"},
                    "max_results": {"type": "integer", "description": "Maximum comments (default 10)"}
                },
                "required": ["video_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "youtube_post_comment",
            "description": "Post a comment on a YouTube video.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_id": {"type": "string", "description": "The video ID to comment on"},
                    "text": {"type": "string", "description": "Comment text"}
                },
                "required": ["video_id", "text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "youtube_my_channel",
            "description": "Get your own YouTube channel info and stats.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    # Substack tools
    {
        "type": "function",
        "function": {
            "name": "substack_create_draft",
            "description": "Create a draft post on Substack newsletter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Post title"},
                    "body": {"type": "string", "description": "Post body text (plain text, double newlines for paragraphs)"},
                    "subtitle": {"type": "string", "description": "Post subtitle"},
                    "audience": {"type": "string", "description": "Audience: 'everyone' (free) or 'only_paid'. Default: 'everyone'"}
                },
                "required": ["title", "body"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "substack_list_drafts",
            "description": "List draft posts on Substack.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Maximum drafts to return (default 10)"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "substack_list_posts",
            "description": "List published Substack posts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Maximum posts to return (default 10)"},
                    "offset": {"type": "integer", "description": "Offset for pagination (default 0)"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "substack_get_post",
            "description": "Get a specific Substack post or draft by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "post_id": {"type": "string", "description": "The post or draft ID"}
                },
                "required": ["post_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "substack_update_draft",
            "description": "Update an existing Substack draft.",
            "parameters": {
                "type": "object",
                "properties": {
                    "draft_id": {"type": "string", "description": "The draft ID to update"},
                    "title": {"type": "string", "description": "New title"},
                    "body": {"type": "string", "description": "New body text"},
                    "subtitle": {"type": "string", "description": "New subtitle"}
                },
                "required": ["draft_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "substack_delete_draft",
            "description": "Delete a Substack draft.",
            "parameters": {
                "type": "object",
                "properties": {
                    "draft_id": {"type": "string", "description": "The draft ID to delete"}
                },
                "required": ["draft_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "substack_get_stats",
            "description": "Get Substack publication statistics (subscribers, posts, etc.).",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    # ── Shared Folder / File Transfer Tools ──
    {
        "type": "function",
        "function": {
            "name": "list_shared",
            "description": "List files in the shared folder. Files placed here are synced to connected clients automatically. To remove files, use run_command with rm.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "share_file",
            "description": "Copy a file into the shared folder so the client can pull/download it. Use this when the user says 'send this to the client' or 'share this file'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to share"},
                    "filename": {"type": "string", "description": "Optional name for the file in shared folder (defaults to original name)"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_transfers",
            "description": "List files uploaded by the client (in the transfers folder). Use when user asks 'what did the client send' or 'check uploads'.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_transfer",
            "description": "Copy a file from the transfers folder (client uploads) to the working directory. Use when user says 'pull that file' or 'grab the upload'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Name of the file in transfers folder"},
                    "destination": {"type": "string", "description": "Where to copy it (defaults to working directory)"}
                },
                "required": ["filename"]
            }
        }
    },
]

# Add browser automation tools
try:
    from browser import get_browser_tool_definitions
    TOOLS.extend(get_browser_tool_definitions())
except ImportError:
    pass  # Playwright not installed — browser tools unavailable

# Add skill-based tools (social media, etc.)
try:
    from skills import get_skill_manager
    _skill_mgr = get_skill_manager()
    TOOLS.extend(_skill_mgr.get_tool_definitions())
except Exception:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic Tool Selection (keep token usage manageable for local LLM)
# ─────────────────────────────────────────────────────────────────────────────

# Core tools always sent (file ops, web, shell, AI delegation)
_CORE_TOOL_NAMES = {
    "read_file", "write_file", "list_directory",
    "change_directory", "run_command", "web_browse", "web_search",
    "search_file", "search_directory", "edit_file",
    "ask_frontier",
}

# Tool groups activated by keyword detection
_TOOL_GROUPS = {
    "image_gen": {
        "keywords": ["generate image", "create image", "make image", "draw", "flux",
                      "generate a picture", "make a picture", "image of", "picture of",
                      "illustration", "render", "lora", "stillion"],
        "tools": {"generate_image", "share_file"},
    },
    "shared": {
        "keywords": ["shared", "transfer", "client", "pull", "push", "send to client",
                      "grab", "fetch", "upload", "download", "sync"],
        "tools": {"list_shared", "share_file", "list_transfers", "get_transfer"},
    },
    "gmail": {
        "keywords": ["gmail", "email", "inbox", "mail"],
        "tools": {"gmail_list", "gmail_read", "gmail_send"},
    },
    "drive": {
        "keywords": ["drive", "google doc", "google drive", "my documents",
                      "my files on", "cloud files", "gdrive", "folder"],
        "tools": {"drive_list", "drive_search", "drive_read", "drive_upload",
                  "drive_create_doc", "drive_create_folder", "drive_create_sheet",
                  "drive_update_doc", "drive_delete"},
    },
    "sheets": {
        "keywords": ["sheet", "spreadsheet", "csv", "table", "cells", "rows", "columns"],
        "tools": {"sheets_read", "sheets_write", "sheets_append", "sheets_create",
                  "sheets_list_sheets", "sheets_clear"},
    },
    "calendar": {
        "keywords": ["calendar", "event", "meeting", "schedule", "appointment", "reminder"],
        "tools": {"calendar_list_events", "calendar_create_event", "calendar_update_event",
                  "calendar_delete_event", "calendar_search_events", "calendar_list_calendars"},
    },
    "slides": {
        "keywords": ["slide", "presentation", "deck", "powerpoint", "ppt"],
        "tools": {"slides_get_presentation", "slides_get_slide", "slides_create_presentation",
                  "slides_add_slide", "slides_add_text"},
    },
    "contacts": {
        "keywords": ["contact", "phone number", "address book", "people"],
        "tools": {"contacts_list", "contacts_get", "contacts_create", "contacts_update",
                  "contacts_delete", "contacts_search"},
    },
    "youtube": {
        "keywords": ["youtube", "playlist", "channel", "subscribe"],
        "tools": {"youtube_search", "youtube_get_video", "youtube_get_channel",
                  "youtube_list_playlists", "youtube_create_playlist",
                  "youtube_get_comments", "youtube_post_comment", "youtube_my_channel"},
    },
    "substack": {
        "keywords": ["substack", "newsletter", "draft", "publish", "blog post", "article"],
        "tools": {"substack_create_draft", "substack_list_drafts", "substack_list_posts",
                  "substack_get_post", "substack_update_draft", "substack_delete_draft",
                  "substack_get_stats"},
    },
    "browser": {
        "keywords": ["browser", "click", "fill form", "navigate to", "open page", "screenshot", "webpage"],
        "tools": {"browser_open", "browser_click", "browser_type", "browser_navigate",
                  "browser_screenshot", "browser_extract", "browser_fill_form",
                  "browser_select", "browser_close"},
    },
    "social": {
        "keywords": ["tweet", "post to", "social media", "twitter", "linkedin", "bluesky", "share on"],
        "tools": {"skill_post_social", "skill_social_status"},
    },
    "vision": {
        "keywords": ["image", "photo", "picture", "camera", "see", "look at",
                      "what is this", "analyze", "describe", "view_image",
                      "attached", "screenshot", "snap"],
        "tools": {"view_image"},
    },
    "collab": {
        "keywords": ["who's online", "whos online", "mesh status", "devices",
                      "dispatch", "send to spark", "send to mac", "run on",
                      "project", "collaboration", "activity", "task",
                      "what are they doing", "online"],
        "tools": {"mesh_status", "dispatch_task", "create_project",
                  "list_projects", "add_to_project", "list_tasks"},
    },
    "google": {
        "keywords": ["google"],
        "tools": {"gmail_list", "gmail_read", "gmail_send",
                  "drive_list", "drive_search", "drive_read", "drive_upload",
                  "drive_create_doc", "drive_create_folder", "drive_create_sheet",
                  "drive_update_doc", "drive_delete",
                  "sheets_read", "sheets_write", "sheets_create",
                  "calendar_list_events", "calendar_create_event",
                  "contacts_list", "contacts_search"},
    },
}

# Register collaboration tools
try:
    from collab_tools import COLLAB_TOOLS
    TOOLS.extend(COLLAB_TOOLS)
except ImportError:
    pass

# Build lookup: tool name -> tool definition
_TOOL_BY_NAME = {t["function"]["name"]: t for t in TOOLS}


def get_tools_for_message(message: str) -> list:
    """Return a filtered subset of TOOLS relevant to the user's message.

    Always includes core tools. Adds specialized tool groups only when
    keywords in the message suggest they're needed.
    """
    msg_lower = message.lower()

    # Start with core tools
    active_names = set(_CORE_TOOL_NAMES)

    # Add tool groups whose keywords match
    for group in _TOOL_GROUPS.values():
        for kw in group["keywords"]:
            if kw in msg_lower:
                active_names.update(group["tools"])
                break

    # Build filtered list preserving order
    return [t for t in TOOLS if t["function"]["name"] in active_names]


# ─────────────────────────────────────────────────────────────────────────────
# Fast Tool Dispatch — bypass LLM for obvious tool calls
# ─────────────────────────────────────────────────────────────────────────────

import re as _re

# Pattern → (tool_name, argument_builder)
# argument_builder receives the match object and original message, returns dict
_FAST_PATTERNS = [
    # Google Drive
    (_re.compile(r'\b(?:list|show|what.?s (?:on|in))\b.*\b(?:drive|google drive)\b', _re.I),
     "drive_list", lambda m, msg: {"query": "", "max_results": 20}),
    (_re.compile(r'\b(?:search|find|look for)\b.*\b(?:drive|google drive)\b', _re.I),
     "drive_search", lambda m, msg: {"query": _re.sub(r'.*?(?:search|find|look for)\s+', '', msg, flags=_re.I).strip().rstrip('?.')}),
    (_re.compile(r'\b(?:search|find|look for)\b\s+(?:["\'`](.+?)["\'`]|(\S+.+?))\s+(?:on|in|from)\s+(?:drive|google drive)', _re.I),
     "drive_search", lambda m, msg: {"query": m.group(1) or m.group(2)}),

    # Gmail
    (_re.compile(r'\b(?:check|list|show|read|get|any new)\b.*\b(?:email|gmail|inbox|mail)\b', _re.I),
     "gmail_list", lambda m, msg: {"query": "", "max_results": 10}),
    (_re.compile(r'\b(?:search|find)\b.*\b(?:email|gmail|mail)\b.*(?:from|about|subject)\s+(.+)', _re.I),
     "gmail_list", lambda m, msg: {"query": m.group(1).strip().rstrip('?.'), "max_results": 10}),

    # Calendar
    (_re.compile(r'\b(?:what.?s on|check|show|list|any)\b.*\b(?:calendar|schedule|agenda)\b', _re.I),
     "calendar_list_events", lambda m, msg: {"max_results": 10}),
    (_re.compile(r'\b(?:upcoming|next|today.?s?|any)\b.*\b(?:event|meeting|appointment|calendar)\b', _re.I),
     "calendar_list_events", lambda m, msg: {"max_results": 10}),

    # Sheets
    (_re.compile(r'\b(?:read|show|open|get)\b.*\b(?:spreadsheet|sheet)\b', _re.I),
     "sheets_list_sheets", lambda m, msg: {"spreadsheet_id": ""}),

    # Contacts (specific search before generic list)
    (_re.compile(r'\b(?:find|search|look up)\b.*\bcontact.*?(?:for|named?)\s+(.+)', _re.I),
     "contacts_search", lambda m, msg: {"query": m.group(1).strip().rstrip('?.')}),
    (_re.compile(r'\b(?:list|show)\b.*\b(?:contact|contacts|address book)\b', _re.I),
     "contacts_list", lambda m, msg: {"max_results": 20}),

    # YouTube
    (_re.compile(r'\b(?:search|find|look for)\b.*\b(?:youtube|on youtube)\b', _re.I),
     "youtube_search", lambda m, msg: {"query": _re.sub(r'.*?(?:search|find|look for)\s+', '', msg, flags=_re.I).replace('on youtube', '').replace('youtube', '').strip().rstrip('?.'), "max_results": 5}),
    (_re.compile(r'\bmy (?:youtube )?channel\b', _re.I),
     "youtube_my_channel", lambda m, msg: {}),

    # Substack
    (_re.compile(r'\b(?:list|show|check)\b.*\b(?:substack|newsletter)\b.*\b(?:draft|drafts)\b', _re.I),
     "substack_list_drafts", lambda m, msg: {"limit": 10}),
    (_re.compile(r'\b(?:list|show)\b.*\b(?:substack|newsletter)\b.*\b(?:post|posts|articles?)\b', _re.I),
     "substack_list_posts", lambda m, msg: {"limit": 10}),
    (_re.compile(r'\bsubstack\b.*\bstat', _re.I),
     "substack_get_stats", lambda m, msg: {}),

    # Web search (common pattern)
    (_re.compile(r'\b(?:search|google|look up|what is|what are|who is|when is|where is)\b', _re.I),
     "web_search", lambda m, msg: {"query": _re.sub(r'^(?:search\s+(?:for\s+)?|google\s+|look\s+up\s+)', '', msg, flags=_re.I).strip().rstrip('?.'), "num_results": 5}),
]


def fast_dispatch(message: str):
    """
    Try to match the user's message to a direct tool call.

    Returns:
        (tool_name, arguments, result) if matched and executed
        None if no fast path matched
    """
    msg = message.strip()

    for pattern, tool_name, arg_builder in _FAST_PATTERNS:
        match = pattern.search(msg)
        if match:
            try:
                args = arg_builder(match, msg)
                # Skip if args seem empty/broken
                if tool_name in ("drive_search", "web_search", "youtube_search") and not args.get("query"):
                    continue
                result = execute_tool(tool_name, args)
                if result and not result.startswith("Error:"):
                    return tool_name, args, result
            except Exception:
                continue

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Tool Implementations
# ─────────────────────────────────────────────────────────────────────────────

def tool_read_file(path: str, start_line: int = None, end_line: int = None) -> str:
    """Read file contents with line numbers."""
    try:
        file_path = resolve_path(path)
        if not file_path.exists():
            return f"Error: File not found: {file_path}"
        if not file_path.is_file():
            return f"Error: Not a file: {file_path}"
        if file_path.stat().st_size > 1_000_000:
            return f"Error: File too large (>1MB): {file_path}"

        lines = file_path.read_text(errors='replace').splitlines()
        total_lines = len(lines)

        start_idx = (start_line - 1) if start_line else 0
        end_idx = end_line if end_line else min(total_lines, start_idx + 200)
        start_idx = max(0, start_idx)
        end_idx = min(total_lines, end_idx)

        selected = lines[start_idx:end_idx]
        numbered = [f"{start_idx + i + 1:4d} | {line}" for i, line in enumerate(selected)]

        log(f"Read lines {start_idx+1}-{end_idx} of {total_lines} from {file_path}")
        return f"File: {file_path} ({total_lines} lines)\n\n" + '\n'.join(numbered)
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error reading file: {e}"


def tool_write_file(path: str, content: str) -> str:
    """Write content to file."""
    try:
        file_path = resolve_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        log(f"Wrote {len(content)} chars to {file_path}")
        return f"Successfully wrote {len(content)} characters to {file_path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def tool_search_file(path: str, pattern: str) -> str:
    """Search for pattern in file."""
    try:
        file_path = resolve_path(path)
        if not file_path.exists():
            return f"Error: File not found: {file_path}"

        lines = file_path.read_text(errors='replace').splitlines()
        matches = []
        for i, line in enumerate(lines):
            if pattern.lower() in line.lower():
                matches.append(f"{i+1:4d} | {line}")

        if not matches:
            return f"No matches for '{pattern}' in {file_path}"

        log(f"Found {len(matches)} matches for '{pattern}'")
        return f"Matches in {file_path}:\n\n" + '\n'.join(matches[:50])
    except Exception as e:
        return f"Error searching file: {e}"


def tool_search_directory(directory: str, pattern: str) -> str:
    """Search for pattern across all files in directory."""
    EXCLUDE_DIRS = {'.git', 'node_modules', '__pycache__', 'venv', '.venv', 'build', 'dist', '.cache'}
    CODE_EXTENSIONS = {'.py', '.js', '.ts', '.jsx', '.tsx', '.sh', '.yaml', '.yml', '.json', '.md', '.txt', '.html', '.css'}

    try:
        dir_path = resolve_path(directory)
        if not dir_path.exists():
            return f"Error: Directory not found: {dir_path}"
        if not dir_path.is_dir():
            return f"Error: Not a directory: {dir_path}"

        pattern_lower = pattern.lower()
        matches = []
        files_searched = 0

        for root, dirs, files in os.walk(dir_path):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for name in files:
                if not any(name.endswith(ext) for ext in CODE_EXTENSIONS):
                    continue

                filepath = Path(root) / name
                files_searched += 1

                try:
                    content = filepath.read_text(errors='replace')
                    for lineno, line in enumerate(content.splitlines(), start=1):
                        if pattern_lower in line.lower():
                            rel_path = filepath.relative_to(dir_path)
                            matches.append(f"{rel_path}:{lineno}: {line.strip()[:100]}")
                            if len(matches) >= 50:
                                break
                except:
                    continue

                if len(matches) >= 50:
                    break
            if len(matches) >= 50:
                break

        if not matches:
            return f"No matches for '{pattern}' in {dir_path} ({files_searched} files searched)"

        log(f"Found {len(matches)} matches in {files_searched} files")
        result = f"Matches for '{pattern}' in {dir_path}:\n\n" + '\n'.join(matches)
        if len(matches) == 50:
            result += "\n\n(Limited to 50 results)"
        return result
    except Exception as e:
        return f"Error searching directory: {e}"


def tool_edit_file(path: str, start_line: int, end_line: int, new_content: str) -> str:
    """Edit specific lines in a file."""
    try:
        file_path = resolve_path(path)
        if not file_path.exists():
            return f"Error: File not found: {file_path}"

        lines = file_path.read_text(errors='replace').splitlines()
        total = len(lines)

        if start_line < 1 or end_line < start_line:
            return f"Error: Invalid line range {start_line}-{end_line}"
        if start_line > total:
            return f"Error: start_line {start_line} > file length ({total})"

        start_idx = start_line - 1
        end_idx = min(end_line, total)

        lines[start_idx:end_idx] = new_content.splitlines()
        file_path.write_text('\n'.join(lines) + '\n')

        log(f"Edited lines {start_line}-{end_line} in {file_path}")
        return f"Edited lines {start_line}-{end_line} in {file_path}"
    except Exception as e:
        return f"Error editing file: {e}"


def tool_list_directory(path: str = None) -> str:
    """List directory contents."""
    global working_dir
    try:
        dir_path = resolve_path(path) if path else working_dir

        if not dir_path.exists():
            return f"Error: Directory not found: {dir_path}"
        if not dir_path.is_dir():
            return f"Error: Not a directory: {dir_path}"

        entries = []
        for item in sorted(dir_path.iterdir()):
            try:
                if item.is_dir():
                    entries.append(f"[DIR]  {item.name}/")
                else:
                    size = item.stat().st_size
                    if size < 1024:
                        size_str = f"{size}B"
                    elif size < 1024 * 1024:
                        size_str = f"{size // 1024}KB"
                    else:
                        size_str = f"{size // (1024 * 1024)}MB"
                    entries.append(f"[FILE] {item.name} ({size_str})")
            except PermissionError:
                entries.append(f"[????] {item.name} (permission denied)")

        result = f"Directory: {dir_path}\n\n" + "\n".join(entries) if entries else f"Directory: {dir_path}\n\n(empty)"
        log(f"Listed {len(entries)} items in {dir_path}")
        return result
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error listing directory: {e}"


def tool_get_working_directory() -> str:
    """Get current working directory."""
    global working_dir
    return str(working_dir)


def tool_change_directory(path: str) -> str:
    """Change working directory."""
    global working_dir
    try:
        new_path = resolve_path(path)
        if not new_path.exists():
            return f"Error: Directory not found: {new_path}"
        if not new_path.is_dir():
            return f"Error: Not a directory: {new_path}"
        working_dir = new_path
        log(f"Changed directory to {working_dir}")
        return f"Changed working directory to: {working_dir}"
    except Exception as e:
        return f"Error changing directory: {e}"


def tool_run_command(command: str) -> str:
    """Execute a shell command."""
    global working_dir
    try:
        log(f"$ {command}")
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(working_dir),
            capture_output=True,
            text=True,
            timeout=300
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr
        if result.returncode != 0:
            output += f"\n[Exit code: {result.returncode}]"
        if not output.strip():
            output = "(command completed successfully)"
        if len(output) > 10000:
            output = output[:10000] + "\n... (output truncated)"
        return output
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 5 minutes"
    except Exception as e:
        return f"Error executing command: {e}"


def tool_web_browse(url: str) -> str:
    """Fetch and parse web page content."""
    import requests
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return "Error: beautifulsoup4 not installed"

    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        log(f"Fetching {url}")

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'aside', 'header', 'noscript', 'iframe']):
            tag.decompose()

        main_content = soup.find('main') or soup.find('article') or soup.find('div', {'class': ['content', 'post', 'article', 'main']})
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
        else:
            text = soup.get_text(separator='\n', strip=True)

        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)

        if len(text) > 15000:
            text = text[:15000] + "\n\n... (content truncated)"

        log(f"Retrieved {len(text)} chars from {url}")
        return f"Content from {url}:\n\n{text}"

    except requests.exceptions.Timeout:
        return f"Error: Request timed out for {url}"
    except requests.exceptions.RequestException as e:
        return f"Error fetching {url}: {e}"
    except Exception as e:
        return f"Error parsing {url}: {e}"


def tool_web_search(query: str, num_results: int = 5) -> str:
    """Search the web using DuckDuckGo."""
    try:
        from ddgs import DDGS
    except ImportError:
        return "Error: ddgs not installed"

    try:
        num_results = min(max(1, num_results), 10)
        log(f"Searching: {query}")

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))

        if not results:
            return f"No results found for: {query}"

        output = f"Search results for: {query}\n\n"
        for i, r in enumerate(results, 1):
            output += f"{i}. {r.get('title', 'No title')}\n"
            output += f"   URL: {r.get('href', 'No URL')}\n"
            output += f"   {r.get('body', 'No description')}\n\n"

        log(f"Found {len(results)} results")
        return output

    except Exception as e:
        return f"Error searching: {e}"


def tool_web_view(url: str, question: str = None) -> str:
    """Screenshot a webpage and analyze it with LLaVA."""
    import base64

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return "Error: playwright not installed"

    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        log(f"Capturing screenshot of {url}")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={'width': 1024, 'height': 768})
            page.goto(url, wait_until='networkidle', timeout=30000)
            page.wait_for_timeout(1000)
            screenshot_bytes = page.screenshot(full_page=False)
            browser.close()

        base64_image = base64.b64encode(screenshot_bytes).decode('utf-8')
        log(f"Screenshot captured ({len(screenshot_bytes) // 1024}KB)")

        if question:
            prompt = f"This is a screenshot of {url}. {question}"
        else:
            prompt = f"This is a screenshot of {url}. Describe what you see."

        log("Analyzing with LLaVA...")

        vision_client = OpenAI(base_url=VISION_URL, api_key="not-needed")
        response = vision_client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }],
            max_tokens=1024,
            temperature=0.2
        )

        analysis = response.choices[0].message.content
        log("Vision analysis complete")

        result = f"Visual analysis of {url}:\n\n{analysis}"
        return result

    except Exception as e:
        error_msg = str(e)
        if "playwright" in error_msg.lower():
            return f"Error: Playwright issue. Run: playwright install chromium\n{e}"
        elif "connect" in error_msg.lower():
            return f"Error: Cannot connect to vision model at {VISION_URL}\n{e}"
        return f"Error viewing webpage: {e}"


def tool_view_image(path: str, question: str = None) -> str:
    """Analyze a local image file with LLaVA."""
    import base64

    try:
        file_path = resolve_path(path)
        if not file_path.exists():
            return f"Error: Image not found: {file_path}"
        if not file_path.is_file():
            return f"Error: Not a file: {file_path}"

        ext = file_path.suffix.lower()
        if ext not in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
            return f"Error: Unsupported format: {ext}"

        if file_path.stat().st_size > 20_000_000:
            return f"Error: Image too large (>20MB)"

        mime_types = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                      '.gif': 'image/gif', '.webp': 'image/webp'}
        mime_type = mime_types.get(ext, 'image/png')

        log(f"Reading image {file_path}")

        with open(file_path, 'rb') as f:
            image_bytes = f.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        log(f"Image loaded ({len(image_bytes) // 1024}KB)")

        if question:
            prompt = f"This is an image from {file_path.name}. {question}"
        else:
            prompt = f"This is an image from {file_path.name}. Describe what you see."

        log("Analyzing with LLaVA...")

        vision_client = OpenAI(base_url=VISION_URL, api_key="not-needed")
        response = vision_client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                ]
            }],
            max_tokens=1024,
            temperature=0.2
        )

        analysis = response.choices[0].message.content
        log("Vision analysis complete")

        return f"Analysis of {file_path.name}:\n\n{analysis}"

    except Exception as e:
        if "connect" in str(e).lower():
            return f"Error: Cannot connect to vision model at {VISION_URL}"
        return f"Error analyzing image: {e}"


def tool_ask_frontier(question: str, context: str = None, provider: str = None) -> str:
    """Escalate a question to a frontier cloud model."""
    try:
        from frontier import ask_frontier, list_available_providers, RateLimitError

        available = list_available_providers()
        if not available:
            return "Error: No frontier providers configured. Set API keys with /keys set <provider> <key>"

        provider_name = provider if provider in available else None
        log("Escalating to frontier model...")

        response = ask_frontier(
            query=question,
            context=context,
            provider_name=provider_name,
            system_prompt="You are an expert assistant. Be thorough but concise."
        )

        log(f"{response.provider}: {response.input_tokens}+{response.output_tokens} tokens, ${response.cost_usd:.4f}")

        return f"[Expert response from {response.provider}]\n\n{response.content}"

    except RateLimitError as e:
        wait_msg = f" Try again in {e.retry_after}s." if e.retry_after else " Wait a minute and try again."
        return f"[Rate limit] {e.provider} free tier limit reached.{wait_msg} Local models are still available."

    except Exception as e:
        return f"Error calling frontier model: {e}"


def tool_schedule_task(action: str, name: str = None, cron: str = None, prompt: str = None, task_id: str = None) -> str:
    """Manage scheduled tasks."""
    try:
        from scheduler import get_scheduler
        scheduler = get_scheduler()

        if action == "list":
            return scheduler.list_tasks()

        elif action == "add":
            if not name:
                return "Error: 'name' is required for add action"
            if not cron:
                return "Error: 'cron' is required for add action"
            if not prompt:
                return "Error: 'prompt' is required for add action"
            log(f"Scheduling task: {name}")
            return scheduler.schedule(name=name, cron=cron, prompt=prompt)

        elif action == "remove":
            if not task_id:
                return "Error: 'task_id' is required for remove action"
            log(f"Removing task: {task_id}")
            return scheduler.unschedule(task_id)

        elif action == "enable":
            if not task_id:
                return "Error: 'task_id' is required for enable action"
            log(f"Enabling task: {task_id}")
            return scheduler.enable_task(task_id)

        elif action == "disable":
            if not task_id:
                return "Error: 'task_id' is required for disable action"
            log(f"Disabling task: {task_id}")
            return scheduler.disable_task(task_id)

        elif action == "run":
            if not task_id:
                return "Error: 'task_id' is required for run action"
            log(f"Running task: {task_id}")
            # Synchronous wrapper for async run
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(scheduler.run_task_by_id(task_id))
                    return f"Running task {task_id}..."
                else:
                    return loop.run_until_complete(scheduler.run_task_by_id(task_id))
            except:
                return f"Task {task_id} scheduled to run."

        else:
            return f"Unknown action: {action}. Use: add, list, remove, enable, disable, run"

    except Exception as e:
        return f"Error with scheduler: {e}"


def tool_send_to_claude(message: str, session: str = "claude") -> str:
    """Send a message to Claude Code running in tmux and capture the response."""
    import shutil
    import time

    # Check if tmux is available
    if not shutil.which("tmux"):
        return "Error: tmux is not installed"

    # Check if the session exists
    result = subprocess.run(
        ["tmux", "has-session", "-t", session],
        capture_output=True
    )
    if result.returncode != 0:
        return f"Error: tmux session '{session}' not found. Start Claude with: ./start_claude.sh"

    # Capture pane content before sending (to know what's new)
    before = subprocess.run(
        ["tmux", "capture-pane", "-t", session, "-p"],
        capture_output=True,
        text=True
    ).stdout

    # Send the message using -l (literal) flag to avoid interpretation issues
    log(f"Sending to Claude: {message[:50]}...")

    # First send the message text literally
    result = subprocess.run(
        ["tmux", "send-keys", "-t", session, "-l", message],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return f"Error sending to Claude: {result.stderr}"

    # Small delay then send Enter key separately
    time.sleep(0.1)
    result = subprocess.run(
        ["tmux", "send-keys", "-t", session, "Enter"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return f"Error sending Enter: {result.stderr}"

    # Wait for Claude to respond - poll until we see the prompt again
    log("Waiting for Claude's response...")
    max_wait = 120  # seconds
    poll_interval = 1
    waited = 0
    last_content = ""

    while waited < max_wait:
        time.sleep(poll_interval)
        waited += poll_interval

        # Capture current pane content
        result = subprocess.run(
            ["tmux", "capture-pane", "-t", session, "-p", "-S", "-500"],
            capture_output=True,
            text=True
        )
        current = result.stdout

        # Check if Claude is done (prompt visible and content stopped changing)
        if "❯" in current.split("\n")[-5:] or ">" in current.split("\n")[-3:]:
            # Prompt appeared, check if content stabilized
            if current == last_content:
                break
        last_content = current

        # Also break if we've been waiting and content hasn't changed
        if waited > 5 and current == last_content:
            time.sleep(2)  # One more pause to be sure
            final_check = subprocess.run(
                ["tmux", "capture-pane", "-t", session, "-p", "-S", "-500"],
                capture_output=True,
                text=True
            ).stdout
            if final_check == current:
                break

    # Extract new content (what appeared after our message)
    after = subprocess.run(
        ["tmux", "capture-pane", "-t", session, "-p", "-S", "-500"],
        capture_output=True,
        text=True
    ).stdout

    # Find the response - everything after our message
    lines = after.strip().split("\n")

    # Look for our message and get everything after until the prompt
    response_lines = []
    found_message = False
    for line in lines:
        if message[:40] in line:  # Found our message
            found_message = True
            continue
        if found_message:
            # Stop at the prompt
            if line.strip().startswith("❯") or line.strip() == ">":
                break
            if "bypass permissions" in line:
                continue
            response_lines.append(line)

    response = "\n".join(response_lines).strip()

    # Truncate long responses to avoid filling MAUDE's context
    MAX_RESPONSE = 2000
    if len(response) > MAX_RESPONSE:
        response = response[:MAX_RESPONSE] + "\n\n[Response truncated - see Claude's tmux session for full output]"

    # Detect if Claude is asking a follow-up question
    follow_up_indicators = [
        "would you like",
        "do you want",
        "should i",
        "shall i",
        "let me know",
        "what would you",
        "which option",
        "prefer",
    ]
    is_follow_up = any(ind in response.lower() for ind in follow_up_indicators)

    if response:
        log(f"Got response from Claude ({len(response)} chars)")
        if is_follow_up:
            return f"Claude's response:\n\n{response}\n\n[Claude is asking a follow-up question - relay this to the user and wait for their answer before proceeding]"
        return f"Claude completed the task:\n\n{response}"
    else:
        return "Claude received the message but no response was captured. Check tmux session."


# ─────────────────────────────────────────────────────────────────────────────
# Shared Folder / File Transfer Tools
# ─────────────────────────────────────────────────────────────────────────────

SHARED_DIR = Path.home() / "nvidia-workbench" / "terminal-llm" / "shared"
TRANSFERS_DIR = Path.home() / "nvidia-workbench" / "terminal-llm" / "transfers"


def tool_list_shared() -> str:
    """List files in the shared folder."""
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for entry in sorted(SHARED_DIR.iterdir()):
        if entry.is_dir():
            entries.append(f"[DIR]  {entry.name}/")
        else:
            size = entry.stat().st_size
            entries.append(f"[FILE] {entry.name} ({size:,} bytes)")
    if not entries:
        return f"Shared folder is empty: {SHARED_DIR}"
    return f"Shared folder ({SHARED_DIR}):\n" + "\n".join(entries)


def tool_share_file(path: str, filename: str = None) -> str:
    """Copy a file into the shared folder for the client to access."""
    import shutil
    src = resolve_path(path)
    if not src.exists():
        return f"Error: File not found: {src}"
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    dest_name = filename or src.name
    dest = SHARED_DIR / dest_name
    try:
        shutil.copy2(str(src), str(dest))
        return f"Shared '{src.name}' as '{dest_name}' — client can now pull it from the shared folder."
    except Exception as e:
        return f"Error sharing file: {e}"


def tool_list_transfers() -> str:
    """List files uploaded by the client."""
    TRANSFERS_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for entry in sorted(TRANSFERS_DIR.iterdir()):
        if entry.is_dir():
            entries.append(f"[DIR]  {entry.name}/")
        else:
            size = entry.stat().st_size
            entries.append(f"[FILE] {entry.name} ({size:,} bytes)")
    if not entries:
        return f"Transfers folder is empty (no client uploads): {TRANSFERS_DIR}"
    return f"Client uploads ({TRANSFERS_DIR}):\n" + "\n".join(entries)


def tool_get_transfer(filename: str, destination: str = None) -> str:
    """Copy a file from the transfers folder to the working directory."""
    import shutil
    src = TRANSFERS_DIR / filename
    if not src.exists():
        return f"Error: '{filename}' not found in transfers. Use list_transfers to see available files."
    dest_dir = resolve_path(destination) if destination else get_working_directory()
    if dest_dir.is_dir():
        dest = dest_dir / filename
    else:
        dest = dest_dir
    try:
        shutil.copy2(str(src), str(dest))
        return f"Copied '{filename}' to {dest}"
    except Exception as e:
        return f"Error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Tool Execution
# ─────────────────────────────────────────────────────────────────────────────

# Rate limiting counters (reset per turn by caller)
vision_call_count = 0
web_call_count = 0
claude_call_count = 0


def reset_rate_limits():
    """Reset per-turn rate limits. Call at start of each user turn."""
    global claude_call_count
    claude_call_count = 0
    global vision_call_count, web_call_count
    vision_call_count = 0
    web_call_count = 0
    _tool_cache.evict_expired()


def tool_generate_image(prompt: str, width: int = 1024, height: int = 1024,
                        seed: int = -1, steps: int = 28, lora: str = None) -> str:
    """Generate an image using Flux via ComfyUI API."""
    import random
    import shutil

    COMFYUI_HOST = os.environ.get("COMFYUI_HOST", "localhost")
    COMFYUI_PORT = int(os.environ.get("COMFYUI_PORT", "8188"))
    comfyui_url = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"

    # Try mesh router first
    try:
        from routing import get_router
        router = get_router()
        result = router.find_comfyui()
        if result:
            _, cap = result
            comfyui_url = cap.endpoint_url
    except Exception:
        pass

    # Check if ComfyUI is reachable
    import http.client
    from urllib.parse import urlparse
    parsed = urlparse(comfyui_url)
    try:
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 8188, timeout=5)
        conn.request("GET", "/system_stats")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        if resp.status != 200:
            return f"Error: ComfyUI not responding at {comfyui_url}. Start it with: cd ~/nvidia-workbench/ComfyUI && ./start.sh"
    except Exception:
        return f"Error: Cannot connect to ComfyUI at {comfyui_url}. Start it with: cd ~/nvidia-workbench/ComfyUI && ./start.sh"

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    # Build Flux workflow
    filename_prefix = f"maude/gen_{seed}"
    workflow = {
        "3": {"class_type": "KSampler", "inputs": {
            "model": ["10", 0], "positive": ["55", 0], "negative": ["19", 0],
            "latent_image": ["6", 0], "seed": seed, "control_after_generate": "fixed",
            "steps": steps, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0
        }},
        "5": {"class_type": "CLIPTextEncodeFlux", "inputs": {
            "clip": ["4", 0], "clip_l": prompt, "t5xxl": prompt, "guidance": 4
        }},
        "19": {"class_type": "CLIPTextEncodeFlux", "inputs": {
            "clip": ["4", 0], "clip_l": "", "t5xxl": "", "guidance": 4
        }},
        "55": {"class_type": "FluxGuidance", "inputs": {
            "conditioning": ["5", 0], "guidance": 3.5
        }},
        "4": {"class_type": "DualCLIPLoader", "inputs": {
            "clip_name1": "t5xxl_fp16.safetensors", "clip_name2": "clip_l.safetensors", "type": "flux"
        }},
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["8", 0]}},
        "8": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "10": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux1-dev.safetensors", "weight_dtype": "default"}},
        "38": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": filename_prefix}}
    }

    # Add LoRA if requested
    if lora:
        lora_map = {
            "stillion-style": "stillion-style.safetensors",
            "marker-mech-style": "marker-mech-style.safetensors",
        }
        lora_file = lora_map.get(lora, f"{lora}.safetensors")
        workflow["50"] = {"class_type": "LoraLoader", "inputs": {
            "model": ["10", 0], "clip": ["4", 0],
            "lora_name": lora_file, "strength_model": 1.0, "strength_clip": 1.0
        }}
        # Rewire: KSampler uses LoRA output instead of raw model
        workflow["3"]["inputs"]["model"] = ["50", 0]
        # CLIP encoder uses LoRA clip output
        workflow["5"]["inputs"]["clip"] = ["50", 1]
        workflow["19"]["inputs"]["clip"] = ["50", 1]

    # Queue the prompt
    import json
    try:
        body = json.dumps({"prompt": workflow, "client_id": "maude"}).encode()
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 8188, timeout=30)
        conn.request("POST", "/prompt", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        result = json.loads(resp.read())
        conn.close()

        if "error" in result:
            return f"Error queuing prompt: {result['error']}"
        prompt_id = result.get("prompt_id")
        if not prompt_id:
            return f"Failed to queue prompt: {result}"
    except Exception as e:
        return f"Error queuing prompt: {e}"

    log(f"Flux generation queued: {prompt_id} (seed={seed}, steps={steps})")

    # Poll for completion (up to 5 minutes)
    import time
    for _ in range(150):
        time.sleep(2)
        try:
            conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 8188, timeout=10)
            conn.request("GET", f"/history/{prompt_id}")
            resp = conn.getresponse()
            history = json.loads(resp.read())
            conn.close()

            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                if "38" in outputs:
                    images = outputs["38"].get("images", [])
                    if images:
                        img_info = images[0]
                        # Build path to the generated image in ComfyUI output dir
                        comfyui_output = Path.home() / "nvidia-workbench" / "ComfyUI" / "app" / "output"
                        subfolder = img_info.get("subfolder", "")
                        filename = img_info["filename"]
                        src = comfyui_output / subfolder / filename if subfolder else comfyui_output / filename

                        # Copy to shared folder for serving via gateway
                        shared_dir = Path.home() / "nvidia-workbench" / "terminal-llm" / "shared"
                        shared_dir.mkdir(parents=True, exist_ok=True)
                        dest_name = f"flux_{seed}.png"
                        dest = shared_dir / dest_name
                        shutil.copy2(str(src), str(dest))

                        log(f"Image generated: {dest}")
                        return (
                            f"Image generated successfully!\n"
                            f"Seed: {seed}\n"
                            f"File: {dest}\n"
                            f"Display with: ![{prompt[:50]}](/download/{dest_name})"
                        )
        except Exception:
            continue

    return f"Timeout waiting for image generation (prompt_id: {prompt_id}). Check ComfyUI at {comfyui_url}"


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool and return the result."""
    global vision_call_count, web_call_count, claude_call_count

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
        claude_call_count += 1
        if claude_call_count > 2:
            return "STOP: Already contacted Claude twice this turn. Report the results to the user and wait for their next message."

    # Rate limiting
    if name in ("web_view", "view_image"):
        if vision_call_count > 0:
            return "(Vision analysis already completed - see previous result.)"
        vision_call_count += 1

    if name in ("web_browse", "web_search"):
        if web_call_count >= 4:
            return "(Web research limit reached. Use gathered information now.)"
        web_call_count += 1

    # Check tool result cache for expensive operations
    if name in ToolCache.TTLS:
        cached = _tool_cache.get(name, arguments)
        if cached is not None:
            log(f"[cached result] {name}")
            return cached + "\n\n[cached result]"

    # Dispatch to tool
    if name == "read_file":
        return tool_read_file(arguments.get("path", ""), arguments.get("start_line"), arguments.get("end_line"))
    elif name == "write_file":
        return tool_write_file(arguments.get("path", ""), arguments.get("content", ""))
    elif name == "search_file":
        return tool_search_file(arguments.get("path", ""), arguments.get("pattern", ""))
    elif name == "search_directory":
        return tool_search_directory(arguments.get("directory", ""), arguments.get("pattern", ""))
    elif name == "edit_file":
        return tool_edit_file(arguments.get("path", ""), arguments.get("start_line", 1), arguments.get("end_line", 1), arguments.get("new_content", ""))
    elif name == "list_directory":
        return tool_list_directory(arguments.get("path"))
    elif name == "get_working_directory":
        return tool_get_working_directory()
    elif name == "change_directory":
        return tool_change_directory(arguments.get("path", ""))
    elif name == "run_command":
        return tool_run_command(arguments.get("command", ""))
    elif name == "web_browse":
        result = tool_web_browse(arguments.get("url", ""))
        _tool_cache.put(name, arguments, result)
        return result
    elif name == "web_search":
        result = tool_web_search(arguments.get("query", ""), arguments.get("num_results", 5))
        _tool_cache.put(name, arguments, result)
        return result
    elif name == "web_view":
        result = tool_web_view(arguments.get("url", ""), arguments.get("question"))
        _tool_cache.put(name, arguments, result)
        return result
    elif name == "view_image":
        result = tool_view_image(arguments.get("path", ""), arguments.get("question"))
        _tool_cache.put(name, arguments, result)
        return result
    elif name == "generate_image":
        return tool_generate_image(
            arguments.get("prompt", ""),
            arguments.get("width", 1024),
            arguments.get("height", 1024),
            arguments.get("seed", -1),
            arguments.get("steps", 28),
            arguments.get("lora"),
        )
    elif name == "ask_frontier":
        return tool_ask_frontier(arguments.get("question", ""), arguments.get("context"), arguments.get("provider"))
    # Shared folder / file transfer tools
    elif name == "list_shared":
        return tool_list_shared()
    elif name == "share_file":
        return tool_share_file(arguments.get("path", ""), arguments.get("filename"))
    elif name == "list_transfers":
        return tool_list_transfers()
    elif name == "get_transfer":
        return tool_get_transfer(arguments.get("filename", ""), arguments.get("destination"))
    elif name == "schedule_task":
        return tool_schedule_task(
            arguments.get("action", "list"),
            arguments.get("name"),
            arguments.get("cron"),
            arguments.get("prompt"),
            arguments.get("task_id")
        )
    elif name == "send_to_claude":
        return tool_send_to_claude(arguments.get("message", ""), arguments.get("session", "claude"))
    # Google Tools
    elif name == "gmail_list":
        from google_tools import gmail_list_messages
        return gmail_list_messages(arguments.get("query", ""), arguments.get("max_results", 10))
    elif name == "gmail_read":
        from google_tools import gmail_read_message
        return gmail_read_message(arguments.get("message_id", ""))
    elif name == "gmail_send":
        from google_tools import gmail_send_message
        return gmail_send_message(
            arguments.get("to", ""),
            arguments.get("subject", ""),
            arguments.get("body", ""),
            arguments.get("cc")
        )
    elif name == "drive_list":
        from google_tools import drive_list_files
        return drive_list_files(arguments.get("query", ""), arguments.get("max_results", 20))
    elif name == "drive_search":
        from google_tools import drive_search
        return drive_search(arguments.get("query", ""))
    elif name == "drive_read":
        from google_tools import drive_read_file
        return drive_read_file(arguments.get("file_id", ""))
    elif name == "drive_upload":
        from google_tools import drive_upload_file
        return drive_upload_file(arguments.get("local_path", ""), arguments.get("folder_id"))
    elif name == "drive_create_folder":
        from google_tools import drive_create_folder
        return drive_create_folder(arguments.get("name", ""), arguments.get("parent_id"))
    elif name == "drive_create_doc":
        from google_tools import drive_create_doc
        return drive_create_doc(arguments.get("name", ""), arguments.get("folder_id"), arguments.get("content", ""))
    elif name == "drive_create_sheet":
        from google_tools import drive_create_sheet
        return drive_create_sheet(arguments.get("name", ""), arguments.get("folder_id"))
    elif name == "drive_update_doc":
        from google_tools import drive_update_doc
        return drive_update_doc(arguments.get("doc_id", ""), arguments.get("content", ""), arguments.get("append", False))
    elif name == "drive_delete":
        from google_tools import drive_delete_file
        return drive_delete_file(arguments.get("file_id", ""))
    # Google Sheets tools
    elif name == "sheets_read":
        from google_tools import sheets_read
        return sheets_read(arguments.get("spreadsheet_id", ""), arguments.get("range", "Sheet1"))
    elif name == "sheets_write":
        from google_tools import sheets_write
        return sheets_write(arguments.get("spreadsheet_id", ""), arguments.get("range", ""), arguments.get("values", []))
    elif name == "sheets_append":
        from google_tools import sheets_append
        return sheets_append(arguments.get("spreadsheet_id", ""), arguments.get("range", ""), arguments.get("values", []))
    elif name == "sheets_create":
        from google_tools import sheets_create
        return sheets_create(arguments.get("title", ""), arguments.get("folder_id"))
    elif name == "sheets_list_sheets":
        from google_tools import sheets_list_sheets
        return sheets_list_sheets(arguments.get("spreadsheet_id", ""))
    elif name == "sheets_clear":
        from google_tools import sheets_clear
        return sheets_clear(arguments.get("spreadsheet_id", ""), arguments.get("range", ""))
    # Google Calendar tools
    elif name == "calendar_list_events":
        from google_tools import calendar_list_events
        return calendar_list_events(
            arguments.get("max_results", 10),
            arguments.get("time_min"),
            arguments.get("time_max"),
            arguments.get("calendar_id", "primary")
        )
    elif name == "calendar_create_event":
        from google_tools import calendar_create_event
        return calendar_create_event(
            arguments.get("summary", ""),
            arguments.get("start", ""),
            arguments.get("end", ""),
            arguments.get("description"),
            arguments.get("location"),
            arguments.get("calendar_id", "primary")
        )
    elif name == "calendar_update_event":
        from google_tools import calendar_update_event
        return calendar_update_event(
            arguments.get("event_id", ""),
            arguments.get("summary"),
            arguments.get("start"),
            arguments.get("end"),
            arguments.get("description"),
            arguments.get("location"),
            arguments.get("calendar_id", "primary")
        )
    elif name == "calendar_delete_event":
        from google_tools import calendar_delete_event
        return calendar_delete_event(arguments.get("event_id", ""), arguments.get("calendar_id", "primary"))
    elif name == "calendar_search_events":
        from google_tools import calendar_search_events
        return calendar_search_events(arguments.get("query", ""), arguments.get("max_results", 10), arguments.get("calendar_id", "primary"))
    elif name == "calendar_list_calendars":
        from google_tools import calendar_list_calendars
        return calendar_list_calendars()
    # Google Slides tools
    elif name == "slides_get_presentation":
        from google_tools import slides_get_presentation
        return slides_get_presentation(arguments.get("presentation_id", ""))
    elif name == "slides_get_slide":
        from google_tools import slides_get_slide
        return slides_get_slide(arguments.get("presentation_id", ""), arguments.get("slide_index", 0))
    elif name == "slides_create_presentation":
        from google_tools import slides_create_presentation
        return slides_create_presentation(arguments.get("title", ""), arguments.get("folder_id"))
    elif name == "slides_add_slide":
        from google_tools import slides_add_slide
        return slides_add_slide(arguments.get("presentation_id", ""), arguments.get("layout", "BLANK"))
    elif name == "slides_add_text":
        from google_tools import slides_add_text
        return slides_add_text(
            arguments.get("presentation_id", ""),
            arguments.get("slide_id", ""),
            arguments.get("text", ""),
            arguments.get("x", 100),
            arguments.get("y", 100),
            arguments.get("width", 400),
            arguments.get("height", 300)
        )
    # Google Contacts tools
    elif name == "contacts_list":
        from google_tools import contacts_list
        return contacts_list(arguments.get("max_results", 20), arguments.get("query"))
    elif name == "contacts_get":
        from google_tools import contacts_get
        return contacts_get(arguments.get("resource_name", ""))
    elif name == "contacts_create":
        from google_tools import contacts_create
        return contacts_create(
            arguments.get("given_name", ""),
            arguments.get("family_name", ""),
            arguments.get("email"),
            arguments.get("phone"),
            arguments.get("organization")
        )
    elif name == "contacts_update":
        from google_tools import contacts_update
        return contacts_update(
            arguments.get("resource_name", ""),
            arguments.get("given_name"),
            arguments.get("family_name"),
            arguments.get("email"),
            arguments.get("phone")
        )
    elif name == "contacts_delete":
        from google_tools import contacts_delete
        return contacts_delete(arguments.get("resource_name", ""))
    elif name == "contacts_search":
        from google_tools import contacts_search
        return contacts_search(arguments.get("query", ""))
    # YouTube tools
    elif name == "youtube_search":
        from google_tools import youtube_search
        return youtube_search(arguments.get("query", ""), arguments.get("max_results", 5), arguments.get("video_type", "video"))
    elif name == "youtube_get_video":
        from google_tools import youtube_get_video
        return youtube_get_video(arguments.get("video_id", ""))
    elif name == "youtube_get_channel":
        from google_tools import youtube_get_channel
        return youtube_get_channel(arguments.get("channel_id", ""))
    elif name == "youtube_list_playlists":
        from google_tools import youtube_list_playlists
        return youtube_list_playlists(arguments.get("channel_id"), arguments.get("max_results", 10))
    elif name == "youtube_get_playlist_items":
        from google_tools import youtube_get_playlist_items
        return youtube_get_playlist_items(arguments.get("playlist_id", ""), arguments.get("max_results", 20))
    elif name == "youtube_create_playlist":
        from google_tools import youtube_create_playlist
        return youtube_create_playlist(arguments.get("title", ""), arguments.get("description", ""), arguments.get("privacy", "private"))
    elif name == "youtube_add_to_playlist":
        from google_tools import youtube_add_to_playlist
        return youtube_add_to_playlist(arguments.get("playlist_id", ""), arguments.get("video_id", ""))
    elif name == "youtube_get_comments":
        from google_tools import youtube_get_comments
        return youtube_get_comments(arguments.get("video_id", ""), arguments.get("max_results", 10))
    elif name == "youtube_post_comment":
        from google_tools import youtube_post_comment
        return youtube_post_comment(arguments.get("video_id", ""), arguments.get("text", ""))
    elif name == "youtube_my_channel":
        from google_tools import youtube_my_channel
        return youtube_my_channel()
    # Substack tools
    elif name == "substack_create_draft":
        from substack_tools import substack_create_draft
        return substack_create_draft(arguments.get("title", ""), arguments.get("body", ""), arguments.get("subtitle", ""), arguments.get("audience", "everyone"))
    elif name == "substack_list_drafts":
        from substack_tools import substack_list_drafts
        return substack_list_drafts(arguments.get("limit", 10))
    elif name == "substack_list_posts":
        from substack_tools import substack_list_posts
        return substack_list_posts(arguments.get("limit", 10), arguments.get("offset", 0))
    elif name == "substack_get_post":
        from substack_tools import substack_get_post
        return substack_get_post(arguments.get("post_id", ""))
    elif name == "substack_update_draft":
        from substack_tools import substack_update_draft
        return substack_update_draft(arguments.get("draft_id", ""), arguments.get("title"), arguments.get("body"), arguments.get("subtitle"))
    elif name == "substack_delete_draft":
        from substack_tools import substack_delete_draft
        return substack_delete_draft(arguments.get("draft_id", ""))
    elif name == "substack_get_stats":
        from substack_tools import substack_get_stats
        return substack_get_stats()
    # Browser automation tools
    elif name.startswith("browser_"):
        try:
            from browser import execute_browser_tool
            return execute_browser_tool(name, arguments)
        except ImportError:
            return "Error: Browser automation requires Playwright. Install with: pip install playwright && playwright install chromium"
    # Skill-based tools (social media, etc.)
    elif name.startswith("skill_"):
        try:
            from skills import get_skill_manager
            skill_name = name[6:]  # Strip "skill_" prefix
            mgr = get_skill_manager()
            return mgr.execute_skill(skill_name, **arguments)
        except ImportError:
            return f"Error: Skills framework not available"
    # Collaboration tools (mesh status, projects, tasks)
    elif name in ("mesh_status", "dispatch_task", "create_project",
                   "list_projects", "add_to_project", "list_tasks"):
        try:
            from collab_tools import execute_collab_tool
            return execute_collab_tool(name, arguments)
        except ImportError:
            return "Error: Collaboration module not available"
    else:
        return f"Error: Unknown tool: {name}"
