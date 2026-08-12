"""
Context hygiene for MAUDE — keep long sessions fast and under budget.

Pipeline (order matters):
  1. Hard-truncate tool results at injection time
  2. Drop/summarize old tool payloads once newer rounds exist
  3. Sliding window + rolling summary of older turns
  4. Token-budget trim (preserve system + most recent turns)

Env overrides (all optional):
  MAUDE_CTX_KEEP_RECENT_TURNS     default 12  (user/assistant pairs kept verbatim)
  MAUDE_CTX_KEEP_TOOL_ROUNDS      default 2   (recent tool rounds kept full)
  MAUDE_CTX_MAX_TOOL_CHARS        default 4000
  MAUDE_CTX_MAX_MSG_CHARS         default 8000
  MAUDE_CTX_TOKEN_BUDGET          default 0   (0 = derive from MAUDE_NUM_CTX * 0.75)
  MAUDE_CTX_MEMORY_TOP_K          default 5
  MAUDE_CTX_MEMORY_VALUE_CHARS    default 400
  MAUDE_CTX_SUMMARY_ENTRY_CHARS   default 140
"""

from __future__ import annotations

import json
import os
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── Config ────────────────────────────────────────────────────────────


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def keep_recent_turns() -> int:
    return max(2, _env_int("MAUDE_CTX_KEEP_RECENT_TURNS", 12))


def keep_tool_rounds() -> int:
    return max(0, _env_int("MAUDE_CTX_KEEP_TOOL_ROUNDS", 2))


def max_tool_chars() -> int:
    return max(500, _env_int("MAUDE_CTX_MAX_TOOL_CHARS", 4000))


def max_msg_chars() -> int:
    return max(500, _env_int("MAUDE_CTX_MAX_MSG_CHARS", 8000))


def memory_top_k() -> int:
    return max(1, _env_int("MAUDE_CTX_MEMORY_TOP_K", 5))


def memory_value_chars() -> int:
    return max(80, _env_int("MAUDE_CTX_MEMORY_VALUE_CHARS", 400))


def summary_entry_chars() -> int:
    return max(40, _env_int("MAUDE_CTX_SUMMARY_ENTRY_CHARS", 140))


def token_budget(default_ctx: int | None = None) -> int:
    """Max estimated tokens for history sent to the model (excludes tools schema)."""
    explicit = _env_int("MAUDE_CTX_TOKEN_BUDGET", 0)
    if explicit > 0:
        return explicit
    if default_ctx is None:
        default_ctx = _env_int("MAUDE_NUM_CTX", 32768)
    # Leave headroom for system tools + generation
    return max(2048, int(default_ctx * 0.75))


# ── Token / text helpers ──────────────────────────────────────────────


def message_text(msg: dict) -> str:
    """Normalize message content to a plain string."""
    content = msg.get("content", "")
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif block.get("type") == "tool_result":
                    parts.append(str(block.get("content", "")))
                else:
                    parts.append(json.dumps(block, ensure_ascii=False)[:200])
            else:
                parts.append(str(block))
        return "\n".join(parts).strip()
    return str(content or "").strip()


def estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate: chars / 4. Handles string and list content."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total += len(json.dumps(block, ensure_ascii=False))
                elif isinstance(block, str):
                    total += len(block)
        tc = msg.get("tool_calls")
        if tc:
            total += len(json.dumps(tc, ensure_ascii=False))
    return max(0, total // 4)


def _truncate(text: str, limit: int, note: str = "truncated") -> str:
    if not text or len(text) <= limit:
        return text
    keep = max(40, limit - 40)
    return text[:keep] + f"\n... ({note}, {len(text)} chars total)"


# ── 1. Hard-truncate tool results ─────────────────────────────────────


def compact_tool_result(name: str, result: str | None) -> str:
    """Hard-truncate a single tool result before injecting into context.

    Full result may still be shown to the user via UI/traces; this only
    affects what the model sees on subsequent steps.
    """
    if result is None:
        return ""
    if not isinstance(result, str):
        result = str(result)
    if not result:
        return result

    n = len(result)
    hard = max_tool_chars()

    # Short status tools — leave alone
    if name in (
        "write_file",
        "edit_file",
        "change_directory",
        "get_working_directory",
        "save_memory",
        "forget_memory",
    ):
        return result if n <= hard else _truncate(result, hard)

    # read_file: head + tail so structure and end of file stay visible
    if name in ("read_file", "read_server_file") and n > 3000:
        lines = result.split("\n")
        if len(lines) > 100:
            head = "\n".join(lines[:80])
            tail = "\n".join(lines[-20:])
            return f"{head}\n\n... ({len(lines) - 100} lines omitted) ...\n\n{tail}"
        return _truncate(result, 3000)

    # Shell: head + tail (errors often at end)
    if name in ("run_command", "run_server_command") and n > 3000:
        head_n, tail_n = 2000, 800
        return result[:head_n] + f"\n\n... ({n - head_n - tail_n} chars omitted) ...\n\n" + result[-tail_n:]

    # Directory listings
    if name in ("list_directory", "list_server_files", "list_shared", "list_transfers") and n > 2000:
        lines = result.split("\n")
        if len(lines) > 65:
            return "\n".join(lines[:65]) + f"\n... ({len(lines) - 65} more entries)"
        return _truncate(result, 2000)

    # Web / browse dumps
    if name in ("web_browse", "web_search", "web_view", "browse_page") and n > 3500:
        return _truncate(result, 3500)

    if n > hard:
        return _truncate(result, hard)
    return result


# ── 2. Drop old tool payloads ─────────────────────────────────────────


def _tool_round_starts(messages: list[dict], fmt: str = "openai") -> list[int]:
    """Indices where a tool-using assistant turn begins."""
    starts: list[int] = []
    for i, msg in enumerate(messages):
        role = msg.get("role")
        if fmt == "openai":
            if role == "assistant" and msg.get("tool_calls"):
                starts.append(i)
        else:  # claude-ish: assistant with tool_use blocks
            content = msg.get("content")
            if role == "assistant" and isinstance(content, list):
                if any(isinstance(b, dict) and b.get("type") == "tool_use" for b in content):
                    starts.append(i)
    return starts


def _summarize_tool_payload(content: Any, max_len: int = 200) -> str:
    text = content if isinstance(content, str) else message_text({"content": content})
    text = " ".join(text.split())
    if not text:
        return "[tool result dropped]"
    if len(text) <= max_len:
        return f"[prior tool result] {text}"
    return f"[prior tool result summarized] {text[: max_len - 3]}..."


def drop_old_tool_payloads(
    messages: list[dict],
    *,
    keep_recent: int | None = None,
    format: str = "openai",
    in_place: bool = True,
) -> tuple[list[dict], int]:
    """Replace tool result bodies older than the last N tool rounds with stubs.

    Keeps tool_call_id / structure intact so the conversation remains valid.
    Returns (messages, number_of_payloads_compacted).
    """
    if keep_recent is None:
        keep_recent = keep_tool_rounds()

    msgs = messages if in_place else deepcopy(messages)
    starts = _tool_round_starts(msgs, format)
    if len(starts) <= keep_recent:
        return msgs, 0

    cutoff = starts[-keep_recent] if keep_recent > 0 else len(msgs)
    compacted = 0

    if format == "openai":
        for i, msg in enumerate(msgs):
            if i >= cutoff:
                break
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if not content:
                continue
            # Already stubbed?
            if isinstance(content, str) and content.startswith("[prior tool result"):
                continue
            if isinstance(content, str) and len(content) <= 220 and content.startswith("["):
                continue
            msg["content"] = _summarize_tool_payload(content)
            compacted += 1
    else:
        for i, msg in enumerate(msgs):
            if i >= cutoff:
                break
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            changed = False
            new_blocks = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    body = block.get("content", "")
                    body_text = body if isinstance(body, str) else str(body)
                    if body_text and not body_text.startswith("[prior tool result"):
                        new_blocks.append({**block, "content": _summarize_tool_payload(body_text)})
                        compacted += 1
                        changed = True
                    else:
                        new_blocks.append(block)
                else:
                    new_blocks.append(block)
            if changed:
                msg["content"] = new_blocks

    return msgs, compacted


# ── 3. Sliding window + rolling summary ───────────────────────────────


def summarize_dropped_messages(dropped: list[dict], max_entries: int = 12) -> str:
    """Build a short stand-in for omitted conversation turns."""
    if not dropped:
        return ""
    cap = summary_entry_chars()
    lines = [f"[Earlier conversation summarized — {len(dropped)} messages omitted]"]
    entries = 0
    for msg in dropped:
        role = str(msg.get("role", "user")).upper()
        if role == "TOOL":
            continue  # tool bodies already dropped separately
        if role == "ASSISTANT" and msg.get("tool_calls") and not message_text(msg):
            # Tool-call-only assistant turn — one-liner
            names = []
            for tc in msg.get("tool_calls") or []:
                fn = (tc.get("function") or {}).get("name") or tc.get("name")
                if fn:
                    names.append(fn)
            text = f"called {', '.join(names)}" if names else "tool call"
        else:
            text = message_text(msg).replace("\n", " ").strip()
        if not text:
            continue
        if len(text) > cap:
            text = text[: cap - 3] + "..."
        lines.append(f"- {role}: {text}")
        entries += 1
        if entries >= max_entries:
            remaining = len(dropped) - entries
            if remaining > 0:
                lines.append(f"- ... and more earlier turns")
            break
    return "\n".join(lines)


def sliding_window_with_summary(
    messages: list[dict],
    *,
    keep_recent: int | None = None,
    in_place: bool = False,
) -> tuple[list[dict], dict]:
    """Keep recent turns verbatim; compress older non-system turns into one summary.

    Preserves leading system messages. `keep_recent` counts non-system messages
    retained in full (default from env, typically 12).
    """
    if keep_recent is None:
        keep_recent = keep_recent_turns()
    keep_recent = max(2, keep_recent)

    original = messages if in_place else list(messages)
    # Keep only "real" system prompts — drop prior rolling-summary blocks so
    # repeated hygiene passes don't stack summaries forever.
    system_msgs = []
    prior_summaries = []
    for m in original:
        if m.get("role") != "system":
            continue
        text = message_text(m)
        if text.startswith("[Earlier conversation summarized"):
            prior_summaries.append(m)
        else:
            system_msgs.append(m)
    non_system = [m for m in original if m.get("role") != "system"]

    removed = 0
    dropped: list[dict] = []
    if len(non_system) > keep_recent:
        dropped = non_system[: -keep_recent]
        non_system = non_system[-keep_recent:]
        removed = len(dropped)

    prepared: list[dict] = list(system_msgs)
    if dropped:
        # Fold any prior summary text into the new one as a single block
        extra_lines = []
        for s in prior_summaries:
            for line in message_text(s).splitlines()[1:]:
                if line.startswith("- "):
                    extra_lines.append(line)
        summary = summarize_dropped_messages(dropped)
        if extra_lines:
            # Deduplicate bullet lines, keep short
            seen = set()
            merged = []
            for line in extra_lines + summary.splitlines()[1:]:
                if line not in seen:
                    seen.add(line)
                    merged.append(line)
            summary = summary.splitlines()[0] + "\n" + "\n".join(merged[:16])
        prepared.append({"role": "system", "content": summary})
    elif prior_summaries:
        # Nothing new dropped this pass — keep a single prior summary if present
        prepared.append(prior_summaries[-1])
    prepared.extend(non_system)

    meta = {
        "removed": removed,
        "kept": len(prepared),
        "original": len(original),
        "summarized": bool(dropped),
    }
    return prepared, meta


# ── 4. Cap oversized non-tool messages ────────────────────────────────


def cap_message_bodies(messages: list[dict], *, max_chars: int | None = None, in_place: bool = True) -> int:
    """Cap individual message bodies (not tool_calls JSON). Returns count capped."""
    if max_chars is None:
        max_chars = max_msg_chars()
    msgs = messages if in_place else messages
    capped = 0
    for msg in msgs:
        if msg.get("role") == "tool":
            continue  # handled by compact_tool_result
        content = msg.get("content")
        if isinstance(content, str) and len(content) > max_chars:
            # Don't chop system summary markers badly
            if content.startswith("[Earlier conversation summarized"):
                continue
            msg["content"] = _truncate(content, max_chars)
            capped += 1
        elif isinstance(content, list):
            # Cap text blocks only
            new_blocks = []
            changed = False
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = str(block.get("text", ""))
                    if len(text) > max_chars:
                        new_blocks.append({**block, "text": _truncate(text, max_chars)})
                        capped += 1
                        changed = True
                    else:
                        new_blocks.append(block)
                else:
                    new_blocks.append(block)
            if changed:
                msg["content"] = new_blocks
    return capped


# ── 5. Token budget trim ──────────────────────────────────────────────


def trim_to_token_budget(
    messages: list[dict],
    max_tokens: int,
    *,
    format: str = "openai",
    threshold_ratio: float = 0.8,
) -> int:
    """Remove oldest middle messages until under budget. Mutates in place.

    Preserves first system message(s) and the most recent 2 messages.
    Returns number of messages removed.
    """
    if max_tokens <= 0:
        return 0
    threshold = int(max_tokens * threshold_ratio)
    if estimate_tokens(messages) <= threshold:
        return 0

    removed = 0
    # Find first non-system index
    first_keep = 0
    while first_keep < len(messages) and messages[first_keep].get("role") == "system":
        first_keep += 1
    # Keep at least one system + 2 recent
    while estimate_tokens(messages) > threshold and len(messages) > first_keep + 2:
        idx = first_keep
        # Skip summary block if present (prefer dropping real turns first)
        if (
            idx < len(messages) - 2
            and messages[idx].get("role") == "system"
            and message_text(messages[idx]).startswith("[Earlier conversation summarized")
        ):
            idx += 1
        if idx >= len(messages) - 2:
            break

        if format == "openai":
            if messages[idx].get("role") == "assistant":
                messages.pop(idx)
                removed += 1
                while idx < len(messages) - 2 and messages[idx].get("role") == "tool":
                    messages.pop(idx)
                    removed += 1
            else:
                messages.pop(idx)
                removed += 1
        else:
            if messages[idx].get("role") == "assistant":
                messages.pop(idx)
                removed += 1
                if idx < len(messages) - 2 and messages[idx].get("role") == "user":
                    content = messages[idx].get("content", "")
                    is_tool_result = isinstance(content, list) and any(
                        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
                    )
                    if is_tool_result:
                        messages.pop(idx)
                        removed += 1
            else:
                messages.pop(idx)
                removed += 1

    return removed


# ── Full pipeline ─────────────────────────────────────────────────────


def prepare_messages_for_model(
    messages: list[dict],
    *,
    max_tokens: int | None = None,
    keep_recent: int | None = None,
    keep_tool_rounds_n: int | None = None,
    format: str = "openai",
    in_place: bool = False,
) -> tuple[list[dict], dict]:
    """Apply the full hygiene pipeline. Returns (prepared_messages, meta).

    When in_place=False (default), works on a deep copy so UI history is untouched.
    When in_place=True, mutates the list (and nested dicts) for long-session cleanup.
    """
    if in_place:
        msgs = messages
    else:
        msgs = deepcopy(messages)

    meta: dict[str, Any] = {
        "original_count": len(messages),
        "original_tokens": estimate_tokens(messages),
        "tool_payloads_dropped": 0,
        "messages_summarized_away": 0,
        "bodies_capped": 0,
        "token_trim_removed": 0,
        "system_deduped": 0,
        "final_count": 0,
        "final_tokens": 0,
    }

    meta["system_deduped"] = strip_redundant_system_preamble(msgs, in_place=True)

    # Cap any oversized non-tool bodies first
    meta["bodies_capped"] = cap_message_bodies(msgs, in_place=True)

    # Hard-truncate tool results still over budget
    for msg in msgs:
        if msg.get("role") == "tool" and isinstance(msg.get("content"), str):
            # Infer tool name from preceding assistant if possible — use generic
            name = msg.get("name") or "tool"
            msg["content"] = compact_tool_result(name, msg["content"])

    # Drop old tool payloads
    msgs, dropped = drop_old_tool_payloads(
        msgs, keep_recent=keep_tool_rounds_n, format=format, in_place=True
    )
    meta["tool_payloads_dropped"] = dropped

    # Sliding window + rolling summary
    msgs, win_meta = sliding_window_with_summary(msgs, keep_recent=keep_recent, in_place=False)
    # sliding_window always returns a new list; replace contents if in_place requested
    if in_place:
        messages[:] = msgs
        msgs = messages
    meta["messages_summarized_away"] = win_meta.get("removed", 0)

    # Token budget
    budget = max_tokens if max_tokens is not None else token_budget()
    meta["token_trim_removed"] = trim_to_token_budget(msgs, budget, format=format)

    meta["final_count"] = len(msgs)
    meta["final_tokens"] = estimate_tokens(msgs)
    return msgs, meta


def strip_redundant_system_preamble(messages: list[dict], in_place: bool = True) -> int:
    """Drop consecutive duplicate system messages (same content). Returns count removed."""
    msgs = messages if in_place else list(messages)
    removed = 0
    i = 1
    while i < len(msgs):
        if msgs[i].get("role") == "system" and msgs[i - 1].get("role") == "system":
            if message_text(msgs[i]) == message_text(msgs[i - 1]):
                msgs.pop(i)
                removed += 1
                continue
        i += 1
    return removed


def apply_hygiene_in_place(
    messages: list[dict],
    *,
    max_tokens: int | None = None,
    format: str = "openai",
) -> dict:
    """Mutate a live conversation history to stay bounded. Returns meta."""
    strip_redundant_system_preamble(messages, in_place=True)
    _, meta = prepare_messages_for_model(
        messages,
        max_tokens=max_tokens,
        format=format,
        in_place=True,
    )
    return meta


# ── Per-mission scratch state ─────────────────────────────────────────


def _scratch_root() -> Path:
    root = Path(os.environ.get("MAUDE_SCRATCH_DIR", str(Path.home() / ".config" / "maude" / "scratch")))
    root.mkdir(parents=True, exist_ok=True)
    return root


@dataclass
class MissionScratch:
    """Ephemeral per-mission key/value store — findings without chat bloat.

    Stored as JSON under ~/.config/maude/scratch/<mission_id>.json.
    Inject a compact summary into the system prompt via `prompt_block()`.
    """

    mission_id: str
    title: str = ""
    objective: str = ""
    notes: dict[str, str] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
    updated_at: str = ""

    _MAX_FINDINGS: int = 30
    _MAX_NOTE_CHARS: int = 500
    _MAX_FINDING_CHARS: int = 300

    def _touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def set_note(self, key: str, value: str) -> None:
        key = (key or "").strip()[:80]
        if not key:
            return
        self.notes[key] = _truncate(str(value).strip(), self._MAX_NOTE_CHARS)
        self._touch()

    def add_finding(self, text: str) -> None:
        text = _truncate(str(text).strip(), self._MAX_FINDING_CHARS)
        if not text:
            return
        self.findings.append(text)
        if len(self.findings) > self._MAX_FINDINGS:
            self.findings = self.findings[-self._MAX_FINDINGS :]
        self._touch()

    def clear(self) -> None:
        self.notes.clear()
        self.findings.clear()
        self._touch()

    def prompt_block(self, max_findings: int = 8) -> str:
        """Compact block suitable for system-prompt injection."""
        if not self.notes and not self.findings and not self.objective:
            return ""
        lines = [f"## Mission scratch ({self.mission_id})"]
        if self.title:
            lines.append(f"Title: {self.title}")
        if self.objective:
            lines.append(f"Objective: {self.objective}")
        if self.notes:
            lines.append("Notes:")
            for k, v in list(self.notes.items())[:12]:
                lines.append(f"- {k}: {v}")
        if self.findings:
            lines.append("Findings:")
            for f in self.findings[-max_findings:]:
                lines.append(f"- {f}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "mission_id": self.mission_id,
            "title": self.title,
            "objective": self.objective,
            "notes": self.notes,
            "findings": self.findings,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MissionScratch":
        return cls(
            mission_id=str(data.get("mission_id") or "default"),
            title=str(data.get("title") or ""),
            objective=str(data.get("objective") or ""),
            notes=dict(data.get("notes") or {}),
            findings=list(data.get("findings") or []),
            updated_at=str(data.get("updated_at") or ""),
        )


_scratch_lock = threading.Lock()
_scratch_cache: dict[str, MissionScratch] = {}


def get_mission_scratch(mission_id: str | None = None) -> MissionScratch:
    """Load or create a mission scratch pad (process-cached + disk-backed)."""
    mid = (mission_id or os.environ.get("MAUDE_MISSION_ID") or "session").strip() or "session"
    with _scratch_lock:
        if mid in _scratch_cache:
            return _scratch_cache[mid]
        path = _scratch_root() / f"{mid}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                scratch = MissionScratch.from_dict(data)
            except (OSError, json.JSONDecodeError):
                scratch = MissionScratch(mission_id=mid)
        else:
            scratch = MissionScratch(mission_id=mid)
        _scratch_cache[mid] = scratch
        return scratch


def save_mission_scratch(scratch: MissionScratch) -> None:
    """Persist scratch to disk."""
    path = _scratch_root() / f"{scratch.mission_id}.json"
    scratch._touch()
    path.write_text(json.dumps(scratch.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    with _scratch_lock:
        _scratch_cache[scratch.mission_id] = scratch


def clear_mission_scratch(mission_id: str | None = None) -> None:
    scratch = get_mission_scratch(mission_id)
    scratch.clear()
    path = _scratch_root() / f"{scratch.mission_id}.json"
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


# ── Memory injection helpers (top-k) ──────────────────────────────────


def clip_memory_value(value: str, limit: int | None = None) -> str:
    if limit is None:
        limit = memory_value_chars()
    value = str(value or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def format_memory_snippets(
    memories: list[Any],
    *,
    top_k: int | None = None,
    title: str = "Relevant Context",
) -> str:
    """Format memory objects/dicts as a top-k prompt section."""
    if top_k is None:
        top_k = memory_top_k()
    if not memories:
        return ""
    lines = []
    for m in memories[:top_k]:
        if hasattr(m, "key"):
            key, value = m.key, m.value
        elif isinstance(m, dict):
            key, value = m.get("key", "?"), m.get("value", "")
        else:
            key, value = "?", str(m)
        lines.append(f"- **{key}**: {clip_memory_value(value)}")
    if not lines:
        return ""
    return f"## {title}\n" + "\n".join(lines)
