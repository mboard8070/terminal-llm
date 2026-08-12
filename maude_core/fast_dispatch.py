"""
Fast Tool Dispatch — bypass LLM tool-selection for obvious intents.

High-confidence regex patterns map user messages to a single tool call.
Misses fall through to the normal chat / auto-router loop.

Hit-rate stats are tracked so we can see which patterns save round-trips.
"""

from __future__ import annotations

import logging
import re as _re
import threading
from typing import Any, Callable

_log = logging.getLogger("maude.fast_dispatch")

# ---------------------------------------------------------------------------
# Hit-rate stats
# ---------------------------------------------------------------------------

_stats_lock = threading.Lock()
_STATS: dict[str, Any] = {
    "attempts": 0,
    "hits": 0,
    "misses": 0,
    "errors": 0,
    "by_tool": {},  # tool_name -> hit count
}


def get_fast_dispatch_stats() -> dict[str, Any]:
    """Return a snapshot of fast-dispatch hit/miss counters."""
    with _stats_lock:
        return {
            "attempts": _STATS["attempts"],
            "hits": _STATS["hits"],
            "misses": _STATS["misses"],
            "errors": _STATS["errors"],
            "by_tool": dict(_STATS["by_tool"]),
            "hit_rate": (
                round(_STATS["hits"] / _STATS["attempts"], 4) if _STATS["attempts"] else 0.0
            ),
        }


def reset_fast_dispatch_stats() -> None:
    """Reset counters (tests / diagnostics)."""
    with _stats_lock:
        _STATS["attempts"] = 0
        _STATS["hits"] = 0
        _STATS["misses"] = 0
        _STATS["errors"] = 0
        _STATS["by_tool"] = {}


def _record_attempt() -> None:
    with _stats_lock:
        _STATS["attempts"] += 1


def _record_hit(tool_name: str) -> None:
    with _stats_lock:
        _STATS["hits"] += 1
        _STATS["by_tool"][tool_name] = _STATS["by_tool"].get(tool_name, 0) + 1


def _record_miss() -> None:
    with _stats_lock:
        _STATS["misses"] += 1


def _record_error() -> None:
    with _stats_lock:
        _STATS["errors"] += 1


# ---------------------------------------------------------------------------
# Multi-step guard — don't hijack compound requests
# ---------------------------------------------------------------------------

_MULTI_STEP_RE = _re.compile(
    r"\b(?:and then|then |after that|also |as well as|followed by)\b"
    r"|\b(?:fix|debug|implement|refactor|deploy|investigate)\b",
    _re.I,
)

# Path-like token: has a slash, starts with ~, or has a file extension
_PATH_TOKEN_RE = _re.compile(
    r"(?:~|/|\.{1,2}/)[^\s?\"']+|[A-Za-z0-9_.@+-]+\.[A-Za-z0-9]{1,8}\b"
)

# Whitelisted shell status / test commands (never free-form shell)
_SHELL_COMMANDS: list[tuple[_re.Pattern, str]] = [
    (_re.compile(r"^(?:run\s+|please\s+)?git\s+status\s*$", _re.I), "git status"),
    (_re.compile(r"^(?:run\s+|please\s+)?git\s+status\s+--short\s*$", _re.I), "git status --short"),
    (
        _re.compile(r"^(?:run\s+|please\s+)?git\s+log\s+(?:--oneline\s+)?(?:-n?\s*)?(\d{1,2})?\s*$", _re.I),
        "git log --oneline -n 10",
    ),
    (_re.compile(r"^(?:run\s+|please\s+)?docker\s+ps\s*$", _re.I), "docker ps"),
    (_re.compile(r"^(?:run\s+|please\s+)?docker\s+ps\s+-a\s*$", _re.I), "docker ps -a"),
    (_re.compile(r"^(?:run\s+|please\s+)?pytest(?:\s+[\w./-]+)?\s*$", _re.I), None),  # use match text
    (_re.compile(r"^(?:run\s+|please\s+)?npm\s+test\s*$", _re.I), "npm test"),
    (_re.compile(r"^(?:run\s+|please\s+)?cargo\s+test\s*$", _re.I), "cargo test"),
    (_re.compile(r"^(?:run\s+|please\s+)?go\s+test(?:\s+\./\.\.\.)?\s*$", _re.I), "go test ./..."),
    (_re.compile(r"^(?:run\s+)?(?:the\s+)?tests?\s*$", _re.I), "pytest"),  # conservative default
]


def _looks_multi_step(msg: str) -> bool:
    if len(msg) > 240:
        return True
    return bool(_MULTI_STEP_RE.search(msg))


def _extract_path(msg: str) -> str | None:
    """Best-effort path token from a message."""
    m = _PATH_TOKEN_RE.search(msg)
    if not m:
        return None
    path = m.group(0).strip().rstrip(".,;:!?")
    # Reject pure words that look like sentences ending in periods
    if path.lower() in {"i.e", "e.g", "etc", "vs", "mr", "dr", "ms"}:
        return None
    return path


def _strip_query_tail(text: str) -> str:
    return text.strip().strip("\"'`").rstrip("?.,!;:").strip()


def _url_from_message(msg: str) -> str | None:
    m = _re.search(r"(https?://[^\s<>\"']+)", msg, _re.I)
    if m:
        return m.group(1).rstrip(").,;]\"'")
    # bare domain after summarize/browse verbs
    m = _re.search(
        r"\b(?:summarize|summary|tldr|tl;dr|browse|fetch|open|read)\b.{0,40}"
        r"((?:www\.)?[a-z0-9][-a-z0-9.]*\.[a-z]{2,}(?:/[^\s]*)?)",
        msg,
        _re.I,
    )
    if m:
        return m.group(1).rstrip(").,;]\"'")
    return None


def _memory_query(msg: str) -> str | None:
    """Extract memory recall query, or None if not a recall intent."""
    # Save intents must not match
    if _re.search(r"\b(?:remember that|remember to|save (?:this |to )?memory|forget)\b", msg, _re.I):
        return None

    patterns = [
        r"\bwhat do you (?:know|remember) about\s+(.+)",
        r"\b(?:check|search|recall|look\s*up)\s+(?:your\s+)?memory\s+(?:for|about)\s+(.+)",
        r"\b(?:recall|search)\s+memory\s+(?:for|about)?\s*(.+)",
        r"\bmemory\s+(?:for|about|of)\s+(.+)",
        r"\bdo you remember\s+(.+)",
    ]
    for pat in patterns:
        m = _re.search(pat, msg, _re.I)
        if m:
            q = _strip_query_tail(m.group(1))
            if q and len(q) >= 2:
                return q
    return None


def _image_prompt(msg: str) -> str | None:
    patterns = [
        r"\b(?:generate|create|draw|make|paint)\b\s+(?:me\s+)?(?:an?\s+)?"
        r"(?:image|picture|illustration|photo|artwork)\b\s*(?:of\s+|:\s*)(.+)",
        r"\b(?:generate|create|draw)\b\s+(?:me\s+)?(?:an?\s+)?"
        r"(?:image|picture)\b\s+(.+)",
        r"\bimage\s+of\s+(.+)",
    ]
    for pat in patterns:
        m = _re.search(pat, msg, _re.I)
        if m:
            prompt = _strip_query_tail(m.group(1))
            # Skip if looks like a file path or multi-part workflow
            if not prompt or len(prompt) < 3:
                continue
            if prompt.lower().startswith(("file", "from ", "using ")):
                continue
            return prompt
    return None


def _list_dir_path(msg: str) -> str | None:
    """Return path for list_directory, or None if not a list intent.

    Returns '' to mean 'use default working dir' — convert to None/omit at call site.
    """
    # Explicit path forms
    m = _re.search(
        r"\b(?:list|show|ls)\b\s+(?:the\s+)?(?:files?|contents?|dir(?:ectory)?|folder)\s+"
        r"(?:in|at|of|from)\s+([^\s?\"']+)",
        msg,
        _re.I,
    )
    if m:
        return m.group(1).strip().rstrip(".,;:?")

    m = _re.search(r"^(?:ls|dir)\s+([~/.\w][^\s]*)\s*$", msg, _re.I)
    if m:
        return m.group(1).strip()

    m = _re.search(r"^(?:ls|dir)\s*$", msg, _re.I)
    if m:
        return ""

    # Generic "list/show files/directory" without path
    if _re.search(
        r"\b(?:list|show|what.?s in)\b.{0,40}\b(?:files?|dir(?:ectory)?|folder|cwd|working dir)\b",
        msg,
        _re.I,
    ):
        # Prefer an embedded path if present
        p = _extract_path(msg)
        return p if p is not None else ""

    return None


def _read_file_path(msg: str) -> str | None:
    # cat/read with path-like arg
    m = _re.search(
        r"\b(?:cat|type)\s+([~/.\w][^\s?\"']+)",
        msg,
        _re.I,
    )
    if m:
        return m.group(1).strip().rstrip(".,;:?")

    m = _re.search(
        r"\b(?:read|show|open|print|display)\b\s+(?:the\s+)?(?:file\s+|contents?\s+of\s+)?"
        r"([~/.\w][^\s?\"']+\.[A-Za-z0-9]{1,8})\b",
        msg,
        _re.I,
    )
    if m:
        return m.group(1).strip().rstrip(".,;:?")

    m = _re.search(
        r"\b(?:read|show contents of|open file)\s+([~/.\w/][^\s?\"']+)",
        msg,
        _re.I,
    )
    if m:
        path = m.group(1).strip().rstrip(".,;:?")
        # Must look path-like
        if "/" in path or path.startswith("~") or "." in path:
            return path
    return None


def _shell_command(msg: str) -> str | None:
    cleaned = msg.strip().rstrip("?.!")
    for pattern, fixed_cmd in _SHELL_COMMANDS:
        m = pattern.match(cleaned)
        if not m:
            continue
        if fixed_cmd is not None:
            # git log with optional count
            if fixed_cmd.startswith("git log") and m.lastindex:
                n = m.group(1)
                if n:
                    return f"git log --oneline -n {n}"
            return fixed_cmd
        # pytest — use the matched command text (strip leading run/please)
        cmd = _re.sub(r"^(?:run\s+|please\s+)", "", cleaned, flags=_re.I).strip()
        return cmd
    return None


# Pattern -> (tool_name, argument_builder)
# argument_builder receives the match object and original message, returns dict
_FAST_PATTERNS: list[tuple[_re.Pattern, str, Callable]] = [
    # ── Google Drive ──────────────────────────────────────────────
    (
        _re.compile(r"\b(?:list|show|what.?s (?:on|in))\b.*\b(?:drive|google drive)\b", _re.I),
        "drive_list",
        lambda m, msg: {"query": "", "max_results": 20},
    ),
    (
        _re.compile(r"\b(?:search|find|look for)\b.*\b(?:drive|google drive)\b", _re.I),
        "drive_search",
        lambda m, msg: {
            "query": _re.sub(r".*?(?:search|find|look for)\s+", "", msg, flags=_re.I).strip().rstrip("?.")
        },
    ),
    (
        _re.compile(
            r'\b(?:search|find|look for)\b\s+(?:["\'`](.+?)["\'`]|(\S+.+?))\s+(?:on|in|from)\s+(?:drive|google drive)',
            _re.I,
        ),
        "drive_search",
        lambda m, msg: {"query": m.group(1) or m.group(2)},
    ),
    # ── Gmail ─────────────────────────────────────────────────────
    (
        _re.compile(r"\b(?:check|list|show|read|get|any new)\b.*\b(?:emails?|gmail|inbox|mail)\b", _re.I),
        "gmail_list",
        lambda m, msg: {"query": "", "max_results": 10},
    ),
    (
        _re.compile(r"\b(?:search|find)\b.*\b(?:emails?|gmail|mail)\b.*(?:from|about|subject)\s+(.+)", _re.I),
        "gmail_list",
        lambda m, msg: {"query": m.group(1).strip().rstrip("?."), "max_results": 10},
    ),
    # ── Calendar ──────────────────────────────────────────────────
    (
        _re.compile(r"\b(?:what.?s on|check|show|list|any)\b.*\b(?:calendar|schedule|agenda)\b", _re.I),
        "calendar_list_events",
        lambda m, msg: {"max_results": 10},
    ),
    (
        _re.compile(r"\b(?:upcoming|next|today.?s?|any)\b.*\b(?:event|meeting|appointment|calendar)\b", _re.I),
        "calendar_list_events",
        lambda m, msg: {"max_results": 10},
    ),
    # ── Sheets ────────────────────────────────────────────────────
    (
        _re.compile(r"\b(?:read|show|open|get)\b.*\b(?:spreadsheet|sheet)\b", _re.I),
        "sheets_list_sheets",
        lambda m, msg: {"spreadsheet_id": ""},
    ),
    # ── Contacts ──────────────────────────────────────────────────
    (
        _re.compile(r"\b(?:find|search|look up)\b.*\bcontact.*?(?:for|named?)\s+(.+)", _re.I),
        "contacts_search",
        lambda m, msg: {"query": m.group(1).strip().rstrip("?.")},
    ),
    (
        _re.compile(r"\b(?:list|show)\b.*\b(?:contact|contacts|address book)\b", _re.I),
        "contacts_list",
        lambda m, msg: {"max_results": 20},
    ),
    # ── YouTube ───────────────────────────────────────────────────
    (
        _re.compile(r"\b(?:search|find|look for)\b.*\b(?:youtube|on youtube)\b", _re.I),
        "youtube_search",
        lambda m, msg: {
            "query": _re.sub(r".*?(?:search|find|look for)\s+", "", msg, flags=_re.I)
            .replace("on youtube", "")
            .replace("youtube", "")
            .strip()
            .rstrip("?."),
            "num_results": 5,
        },
    ),
    (_re.compile(r"\bmy (?:youtube )?channel\b", _re.I), "youtube_my_channel", lambda m, msg: {}),
    # ── Substack ──────────────────────────────────────────────────
    (
        _re.compile(r"\b(?:list|show|check)\b.*\b(?:substack|newsletter)\b.*\b(?:draft|drafts)\b", _re.I),
        "substack_list_drafts",
        lambda m, msg: {"limit": 10},
    ),
    (
        _re.compile(r"\b(?:list|show)\b.*\b(?:substack|newsletter)\b.*\b(?:post|posts|articles?)\b", _re.I),
        "substack_list_posts",
        lambda m, msg: {"limit": 10},
    ),
    (_re.compile(r"\bsubstack\b.*\bstat", _re.I), "substack_get_stats", lambda m, msg: {}),
    # ── Web search (last among generic intents) ───────────────────
    # NOTE: "google" alone must NOT match when followed by a service name
    (
        _re.compile(
            r"\b(?:search|google|look up|what is|what are|who is|when is|where is)\b"
            r"(?!.*\b(?:doc|drive|sheet|calendar|slide|contact|gmail|emails?|mail|inbox)\b)",
            _re.I,
        ),
        "web_search",
        lambda m, msg: {
            "query": _re.sub(r"^(?:search\s+(?:for\s+)?|google\s+|look\s+up\s+)", "", msg, flags=_re.I)
            .strip()
            .rstrip("?."),
            "num_results": 5,
        },
    ),
]


def match_fast_dispatch(message: str) -> tuple[str, dict] | None:
    """
    Match message to a tool call without executing.

    Returns (tool_name, arguments) or None.
    """
    msg = (message or "").strip()
    if not msg:
        return None

    if _looks_multi_step(msg):
        return None

    # 1) URL summarize / browse  (must beat web_search "what is")
    url = _url_from_message(msg)
    if url and _re.search(r"\b(?:summarize|summary|tldr|tl;dr|browse|fetch|open url|read (?:this )?(?:url|page|link))\b", msg, _re.I):
        return "web_browse", {"url": url}
    if url and _re.match(r"^(?:https?://)\S+$", msg.strip()):
        # Bare URL paste → browse
        return "web_browse", {"url": url}

    # 2) Image generation
    img_prompt = _image_prompt(msg)
    if img_prompt:
        return "generate_image", {"prompt": img_prompt}

    # 3) Shell status / tests (exact-ish whitelist)
    shell_cmd = _shell_command(msg)
    if shell_cmd:
        return "run_command", {"command": shell_cmd}

    # 4) Filesystem read (before list — "read file X" vs "list files")
    read_path = _read_file_path(msg)
    if read_path:
        return "read_file", {"path": read_path}

    # 5) Filesystem list
    list_path = _list_dir_path(msg)
    if list_path is not None:
        args: dict = {}
        if list_path:
            args["path"] = list_path
        return "list_directory", args

    # 6) Memory recall / list
    if _re.search(r"\b(?:list|show)\b.*\bmemories\b", msg, _re.I):
        return "list_memories", {"limit": 20}
    mem_q = _memory_query(msg)
    if mem_q:
        return "recall_memory", {"query": mem_q}

    # 7) Legacy regex table (Google, web search, etc.)
    for pattern, tool_name, arg_builder in _FAST_PATTERNS:
        match = pattern.search(msg)
        if not match:
            continue
        try:
            args = arg_builder(match, msg)
        except Exception:
            continue
        if tool_name in ("drive_search", "web_search", "youtube_search") and not args.get("query"):
            continue
        # Don't let aggressive web_search steal image/memory-ish queries we already skipped
        if tool_name == "web_search":
            q = (args.get("query") or "").lower()
            if any(x in q for x in ("generate image", "draw me", "memory about")):
                continue
        return tool_name, args

    return None


def fast_dispatch(message: str, *, execute: bool = True):
    """
    Try to match the user's message to a direct tool call.

    Returns:
        (tool_name, arguments, result) if matched and executed
        None if no fast path matched

    Args:
        message: user text
        execute: if False, only match (result is None) — used by tests
    """
    _record_attempt()
    matched = match_fast_dispatch(message)
    if not matched:
        _record_miss()
        return None

    tool_name, args = matched
    if not execute:
        _record_hit(tool_name)
        return tool_name, args, None

    from .execute import execute_tool

    try:
        result = execute_tool(tool_name, args)
        if result and not str(result).startswith("Error:"):
            _record_hit(tool_name)
            _log.info("fast_dispatch hit: %s args=%s", tool_name, args)
            return tool_name, args, result
        # Tool returned error — treat as miss so LLM can recover
        _record_error()
        _log.debug("fast_dispatch tool error for %s: %s", tool_name, (result or "")[:200])
        return None
    except Exception as exc:
        _record_error()
        _log.debug("fast_dispatch exception for %s: %s", tool_name, exc)
        return None
