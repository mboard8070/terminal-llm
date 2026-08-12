"""
Client-side context hygiene (steps 1 of MAUDE cost reduction).

Mac/PC clients ship as a standalone package without maude_core. Prefer the
server implementation when available (Spark-local runs), otherwise use this
portable subset so long client sessions stay under budget.

Mirrors maude_core.context_hygiene essentials:
  - compact_tool_result
  - drop_old_tool_payloads
  - sliding_window_with_summary
  - apply_hygiene_in_place
  - clear_mission_scratch (local no-op when no maude_core)
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any


# Prefer shared server package when on Spark / full checkout
try:
    from maude_core.context_hygiene import (  # type: ignore
        apply_hygiene_in_place,
        clear_mission_scratch,
        compact_tool_result,
        drop_old_tool_payloads,
        estimate_tokens,
        sliding_window_with_summary,
    )

    _USING_CORE = True
except Exception:  # noqa: BLE001 — any import/runtime miss falls back
    _USING_CORE = False


if not _USING_CORE:

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

    def summary_entry_chars() -> int:
        return max(40, _env_int("MAUDE_CTX_SUMMARY_ENTRY_CHARS", 140))

    def token_budget(default_ctx: int | None = None) -> int:
        explicit = _env_int("MAUDE_CTX_TOKEN_BUDGET", 0)
        if explicit > 0:
            return explicit
        if default_ctx is None:
            default_ctx = _env_int("MAUDE_NUM_CTX", 32768)
        return max(2048, int(default_ctx * 0.75))

    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 20)] + "\n... (truncated)"

    def message_text(msg: dict) -> str:
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

    def compact_tool_result(name: str, result: str | None) -> str:
        if result is None:
            return ""
        if not isinstance(result, str):
            result = str(result)
        if not result:
            return result

        n = len(result)
        hard = max_tool_chars()

        if name in (
            "write_file",
            "edit_file",
            "change_directory",
            "get_working_directory",
            "save_memory",
            "forget_memory",
        ):
            return result if n <= hard else _truncate(result, hard)

        if name in ("read_file", "read_server_file") and n > 3000:
            lines = result.split("\n")
            if len(lines) > 100:
                head = "\n".join(lines[:80])
                tail = "\n".join(lines[-20:])
                return f"{head}\n\n... ({len(lines) - 100} lines omitted) ...\n\n{tail}"
            return _truncate(result, 3000)

        if name in ("run_command", "run_server_command") and n > 3000:
            head_n, tail_n = 2000, 800
            return (
                result[:head_n]
                + f"\n\n... ({n - head_n - tail_n} chars omitted) ...\n\n"
                + result[-tail_n:]
            )

        if name in ("list_directory", "list_server_files", "list_shared", "list_transfers") and n > 2000:
            lines = result.split("\n")
            if len(lines) > 65:
                return "\n".join(lines[:65]) + f"\n... ({len(lines) - 65} more entries)"
            return _truncate(result, 2000)

        if name in ("web_browse", "web_search", "web_view", "browse_page") and n > 3500:
            return _truncate(result, 3500)

        if n > hard:
            return _truncate(result, hard)
        return result

    def _tool_round_starts(messages: list[dict], fmt: str = "openai") -> list[int]:
        starts: list[int] = []
        for i, msg in enumerate(messages):
            role = msg.get("role")
            if fmt == "openai":
                if role == "assistant" and msg.get("tool_calls"):
                    starts.append(i)
            else:
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
        if keep_recent is None:
            keep_recent = keep_tool_rounds()

        msgs = messages if in_place else deepcopy(messages)
        starts = _tool_round_starts(msgs, format)
        if len(starts) <= keep_recent:
            return msgs, 0

        cutoff = starts[-keep_recent] if keep_recent > 0 else len(msgs)
        compacted = 0

        for i, msg in enumerate(msgs):
            if i >= cutoff:
                break
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if not content:
                continue
            if isinstance(content, str) and content.startswith("[prior tool result"):
                continue
            if isinstance(content, str) and len(content) <= 220 and content.startswith("["):
                continue
            msg["content"] = _summarize_tool_payload(content)
            compacted += 1

        return msgs, compacted

    def summarize_dropped_messages(dropped: list[dict], max_entries: int = 12) -> str:
        if not dropped:
            return ""
        cap = summary_entry_chars()
        lines = [f"[Earlier conversation summarized — {len(dropped)} messages omitted]"]
        entries = 0
        for msg in dropped:
            role = str(msg.get("role", "user")).upper()
            if role == "TOOL":
                continue
            if role == "ASSISTANT" and msg.get("tool_calls") and not message_text(msg):
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
                break
        return "\n".join(lines)

    def sliding_window_with_summary(
        messages: list[dict],
        *,
        keep_recent: int | None = None,
        in_place: bool = False,
    ) -> tuple[list[dict], dict]:
        if keep_recent is None:
            keep_recent = keep_recent_turns()
        keep_recent = max(2, keep_recent)

        original = messages if in_place else list(messages)
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
            dropped = non_system[:-keep_recent]
            non_system = non_system[-keep_recent:]
            removed = len(dropped)

        prepared: list[dict] = list(system_msgs)
        if dropped:
            extra_lines = []
            for s in prior_summaries:
                for line in message_text(s).splitlines()[1:]:
                    if line.startswith("- "):
                        extra_lines.append(line)
            summary = summarize_dropped_messages(dropped)
            if extra_lines:
                seen = set()
                merged = []
                for line in extra_lines + summary.splitlines()[1:]:
                    if line not in seen:
                        seen.add(line)
                        merged.append(line)
                summary = summary.splitlines()[0] + "\n" + "\n".join(merged[:16])
            prepared.append({"role": "system", "content": summary})
        elif prior_summaries:
            prepared.append(prior_summaries[-1])
        prepared.extend(non_system)

        meta = {
            "removed": removed,
            "kept": len(prepared),
            "original": len(original),
            "summarized": bool(dropped),
        }
        return prepared, meta

    def cap_message_bodies(
        messages: list[dict], *, max_chars: int | None = None, in_place: bool = True
    ) -> int:
        if max_chars is None:
            max_chars = max_msg_chars()
        capped = 0
        for msg in messages:
            if msg.get("role") == "tool":
                continue
            content = msg.get("content")
            if isinstance(content, str) and len(content) > max_chars:
                if content.startswith("[Earlier conversation summarized"):
                    continue
                msg["content"] = _truncate(content, max_chars)
                capped += 1
        return capped

    def trim_to_token_budget(
        messages: list[dict], max_tokens: int, format: str = "openai"
    ) -> int:
        removed = 0
        while estimate_tokens(messages) > max_tokens and len(messages) > 4:
            # Drop oldest non-system message
            idx = next((i for i, m in enumerate(messages) if m.get("role") != "system"), None)
            if idx is None or idx >= len(messages) - 2:
                break
            messages.pop(idx)
            removed += 1
        return removed

    def strip_redundant_system_preamble(messages: list[dict], in_place: bool = True) -> int:
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
        strip_redundant_system_preamble(messages, in_place=True)
        cap_message_bodies(messages, in_place=True)

        for msg in messages:
            if msg.get("role") == "tool" and isinstance(msg.get("content"), str):
                name = msg.get("name") or "tool"
                msg["content"] = compact_tool_result(name, msg["content"])

        _, tool_dropped = drop_old_tool_payloads(messages, format=format, in_place=True)
        prepared, win_meta = sliding_window_with_summary(messages, in_place=False)
        messages[:] = prepared
        budget = max_tokens if max_tokens is not None else token_budget()
        token_removed = trim_to_token_budget(messages, budget, format=format)
        return {
            "tool_payloads_dropped": tool_dropped,
            "messages_summarized_away": win_meta.get("removed", 0),
            "token_trim_removed": token_removed,
            "final_count": len(messages),
            "final_tokens": estimate_tokens(messages),
        }

    def clear_mission_scratch(mission_id: str | None = None) -> None:
        """No mission scratch on standalone client — clear is a no-op."""
        return None
