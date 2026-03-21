"""
MAUDE Proactive Heartbeat System.

Runs as a background loop that periodically gathers context -- recent
conversation history, pending scheduled tasks, unread cross-channel
messages, and system status -- then asks the LLM whether any proactive
action is warranted.  When the LLM decides something needs attention it
can autonomously execute tools, send messages to the user via any
connected channel, or update persistent memory.

Architecture
------------
- HeartbeatEngine    : async start()/stop(), configurable interval
- PatternTracker     : observes user activity over time and surfaces trends
- Daily activity log : ~/.config/maude/heartbeat/YYYY-MM-DD.md
- Pattern store      : ~/.config/maude/heartbeat/patterns.json

Configuration (environment variables)
--------------------------------------
MAUDE_HEARTBEAT_INTERVAL   seconds between ticks  (default 900 = 15 min)
MAUDE_HEARTBEAT_ENABLED    "true" / "false"        (default "true")
"""

import asyncio
import json
import logging
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from openai import OpenAI

from frontier import ask_frontier, list_available_providers
from maude_core import (
    LOCAL_URL,
    MODEL,
    append_chat_log,
    get_conversation_history,
    get_memory,
    read_chat_log_since,
)
from scheduler import get_scheduler

log = logging.getLogger("maude.heartbeat")

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

_DEFAULT_INTERVAL = 900  # 15 minutes
_DEFAULT_ENABLED = True

HEARTBEAT_DIR = Path.home() / ".config" / "maude" / "heartbeat"
PATTERNS_FILE = HEARTBEAT_DIR / "patterns.json"


def _cfg_interval() -> int:
    """Return heartbeat interval in seconds from env or default."""
    try:
        return int(os.environ.get("MAUDE_HEARTBEAT_INTERVAL", _DEFAULT_INTERVAL))
    except (ValueError, TypeError):
        return _DEFAULT_INTERVAL


def _cfg_enabled() -> bool:
    """Return whether the heartbeat is enabled."""
    raw = os.environ.get("MAUDE_HEARTBEAT_ENABLED", "true")
    return raw.strip().lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# Heartbeat system prompt
# ---------------------------------------------------------------------------

HEARTBEAT_PROMPT = """\
You are MAUDE's proactive awareness system.  Review the following context
and decide if any action is needed.

RECENT ACTIVITY:
{recent_messages}

PENDING TASKS:
{scheduled_tasks}

OBSERVATIONS:
{patterns}

SYSTEM STATUS:
{system_status}

CURRENT TIME: {now}

If action is needed, describe what you want to do.  Be specific: name the
tool to call, the message to send, or the memory to update.

If nothing needs attention, respond with exactly: ALL_CLEAR

Possible actions:
  - send a message to the user (specify channel and content)
  - run a tool (give the tool name and arguments)
  - update a memory (give the key and value)
  - check on something (describe what to look up)

Respond in this JSON format when action is needed:
{{
  "action": "send_message" | "run_tool" | "update_memory" | "check",
  "details": {{...action-specific fields...}},
  "reason": "brief explanation"
}}

You may return multiple actions as a JSON array.
"""


# ---------------------------------------------------------------------------
# Pattern Tracker
# ---------------------------------------------------------------------------


@dataclass
class ObservedPattern:
    """A single behavioural pattern observed over time."""

    description: str
    frequency: int = 1
    first_seen: str = ""
    last_seen: str = ""
    tags: list[str] = field(default_factory=list)


class PatternTracker:
    """
    Track what the user does frequently and surface trends.

    Patterns are persisted as JSON at ~/.config/maude/heartbeat/patterns.json
    so they survive restarts.
    """

    def __init__(self):
        self.patterns: dict[str, ObservedPattern] = {}
        self._load()

    # -- Persistence --------------------------------------------------------

    def _load(self):
        """Load patterns from disk."""
        if PATTERNS_FILE.exists():
            try:
                raw = json.loads(PATTERNS_FILE.read_text())
                for key, data in raw.items():
                    self.patterns[key] = ObservedPattern(**data)
            except Exception as exc:
                log.warning("Failed to load patterns: %s", exc)

    def _save(self):
        """Persist patterns to disk."""
        HEARTBEAT_DIR.mkdir(parents=True, exist_ok=True)
        data = {k: asdict(v) for k, v in self.patterns.items()}
        PATTERNS_FILE.write_text(json.dumps(data, indent=2))

    # -- Observation API ----------------------------------------------------

    def observe(self, key: str, description: str, tags: list[str] | None = None):
        """
        Record an observation.  If the key already exists, bump its
        frequency and update last_seen.
        """
        now = datetime.now().isoformat()
        if key in self.patterns:
            p = self.patterns[key]
            p.frequency += 1
            p.last_seen = now
            if description and description != p.description:
                # Keep the most recent description
                p.description = description
        else:
            self.patterns[key] = ObservedPattern(
                description=description,
                frequency=1,
                first_seen=now,
                last_seen=now,
                tags=tags or [],
            )
        self._save()

    def get_notable(self, min_frequency: int = 3, limit: int = 10) -> list[ObservedPattern]:
        """Return patterns that have been seen at least *min_frequency* times."""
        notable = [p for p in self.patterns.values() if p.frequency >= min_frequency]
        notable.sort(key=lambda p: p.frequency, reverse=True)
        return notable[:limit]

    def get_recent(self, hours: int = 24, limit: int = 10) -> list[ObservedPattern]:
        """Return patterns observed within the last *hours*."""
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        recent = [p for p in self.patterns.values() if p.last_seen >= cutoff]
        recent.sort(key=lambda p: p.last_seen, reverse=True)
        return recent[:limit]

    def summarise(self) -> str:
        """Human-readable summary suitable for the heartbeat prompt."""
        notable = self.get_notable(min_frequency=2, limit=5)
        recent = self.get_recent(hours=24, limit=5)

        lines: list[str] = []

        if notable:
            lines.append("Recurring patterns:")
            for p in notable:
                lines.append(f"  - {p.description} (seen {p.frequency}x)")

        if recent:
            lines.append("Recent observations:")
            for p in recent:
                lines.append(f"  - {p.description} (last: {p.last_seen[:16]})")

        return "\n".join(lines) if lines else "(no observations yet)"


# ---------------------------------------------------------------------------
# Daily Activity Logger
# ---------------------------------------------------------------------------


class HeartbeatLogger:
    """
    Append structured entries to a daily markdown log at
    ~/.config/maude/heartbeat/YYYY-MM-DD.md
    """

    def __init__(self):
        HEARTBEAT_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _log_path(dt: datetime | None = None) -> Path:
        dt = dt or datetime.now()
        return HEARTBEAT_DIR / f"{dt.strftime('%Y-%m-%d')}.md"

    def write(self, entry: str, level: str = "INFO"):
        """Append a timestamped entry to today's log."""
        path = self._log_path()
        now = datetime.now()
        ts = now.strftime("%H:%M:%S")

        # Create the file with a heading if it is new
        if not path.exists():
            header = f"# MAUDE Heartbeat Log - {now.strftime('%Y-%m-%d')}\n\n"
            path.write_text(header)

        line = f"- **{ts}** [{level}] {entry}\n"
        with open(path, "a") as f:
            f.write(line)

    def write_section(self, title: str, body: str):
        """Write a titled section (e.g. for action reports)."""
        path = self._log_path()
        now = datetime.now()
        ts = now.strftime("%H:%M:%S")

        if not path.exists():
            header = f"# MAUDE Heartbeat Log - {now.strftime('%Y-%m-%d')}\n\n"
            path.write_text(header)

        section = f"\n### {ts} - {title}\n\n{body}\n"
        with open(path, "a") as f:
            f.write(section)


# ---------------------------------------------------------------------------
# Heartbeat Engine
# ---------------------------------------------------------------------------


class HeartbeatEngine:
    """
    Core heartbeat loop.

    Usage::

        engine = HeartbeatEngine()
        engine.set_maude_callback(my_llm_fn)
        engine.set_tool_executor(my_tool_fn)
        engine.set_gateway(my_gateway)
        await engine.start()
        ...
        await engine.stop()
    """

    def __init__(self):
        self.interval: int = _cfg_interval()
        self.enabled: bool = _cfg_enabled()

        # Callbacks -- set externally before start()
        self._maude_callback: Callable | None = None
        self._tool_executor: Callable | None = None
        self._gateway = None  # Channel gateway for proactive messages

        # Internal state
        self._running: bool = False
        self._loop_task: asyncio.Task | None = None
        self._tick_count: int = 0
        self._chat_log_position: int = 0  # file-seek position for chat sync

        # Subsystems
        self.patterns = PatternTracker()
        self.logger = HeartbeatLogger()

        # Local LLM client (lazy)
        self._local_client: OpenAI | None = None

    # -- Dependency injection -----------------------------------------------

    def set_maude_callback(self, callback: Callable):
        """
        Set the callback used to send a prompt to the LLM and receive a
        text response.  Signature: ``async (prompt: str) -> str``
        """
        self._maude_callback = callback

    def set_tool_executor(self, executor: Callable):
        """
        Set the callback for executing MAUDE tools autonomously.
        Signature: ``(tool_name: str, arguments: dict) -> str``
        """
        self._tool_executor = executor

    def set_gateway(self, gateway):
        """
        Set the channel gateway for sending proactive messages to the user
        via Telegram, Discord, etc.
        """
        self._gateway = gateway

    # -- Lifecycle ----------------------------------------------------------

    async def start(self):
        """Start the heartbeat background loop."""
        if not self.enabled:
            log.info("Heartbeat disabled via MAUDE_HEARTBEAT_ENABLED")
            self.logger.write("Heartbeat disabled by configuration", level="WARN")
            return

        self._running = True
        self._loop_task = asyncio.create_task(self._loop())
        self.logger.write(
            f"Heartbeat started (interval={self.interval}s)",
            level="INFO",
        )
        log.info("Heartbeat started, interval=%ds", self.interval)

    async def stop(self):
        """Stop the heartbeat loop gracefully."""
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        self.logger.write("Heartbeat stopped", level="INFO")
        log.info("Heartbeat stopped after %d ticks", self._tick_count)

    # -- Main loop ----------------------------------------------------------

    async def _loop(self):
        """
        Background coroutine that fires a tick on each interval.

        The first tick is slightly delayed (30 s) so the rest of the system
        has time to finish booting.
        """
        log.debug("Heartbeat loop entering initial delay")
        await asyncio.sleep(30)  # Let other services spin up first

        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception("Heartbeat tick failed: %s", exc)
                self.logger.write(f"Tick failed: {exc}", level="ERROR")

            await asyncio.sleep(self.interval)

    async def _tick(self):
        """Execute one heartbeat cycle."""
        self._tick_count += 1
        now = datetime.now()
        log.debug("Heartbeat tick #%d at %s", self._tick_count, now.isoformat())

        # 1. Gather context
        context = await self._gather_context()

        # 2. Check and run due scheduled tasks
        await self._run_due_tasks()

        # 3. Ask the LLM whether action is needed
        decision = await self._evaluate(context)

        # 4. Act on the LLM's decision
        if decision:
            await self._act(decision)

        # 5. Observe patterns from recent activity
        self._observe_patterns(context)

        self.logger.write(
            f"Tick #{self._tick_count} complete (action={'yes' if decision else 'no'})",
        )

    # -- Context gathering --------------------------------------------------

    async def _gather_context(self) -> dict[str, Any]:
        """
        Collect everything the heartbeat LLM needs to make a decision.
        """
        ctx: dict[str, Any] = {}

        # Recent conversation history (last 20 messages from shared memory)
        ctx["recent_messages"] = self._get_recent_messages()

        # Unread cross-channel messages since last check
        ctx["unread"] = self._get_unread_messages()

        # Pending scheduled tasks
        ctx["scheduled_tasks"] = self._get_scheduled_tasks()

        # Pattern observations
        ctx["patterns"] = self.patterns.summarise()

        # Basic system status
        ctx["system_status"] = self._get_system_status()

        ctx["now"] = datetime.now().isoformat()

        return ctx

    def _get_recent_messages(self) -> str:
        """Fetch recent conversation history from persistent memory."""
        try:
            history = get_conversation_history(limit=20)
            if not history:
                return "(no recent conversation)"

            lines = []
            for msg in history[-20:]:
                ts = msg.get("timestamp", "")[:16]
                channel = msg.get("channel", "?")
                role = msg.get("role", "?")
                content = msg.get("content", "")
                # Truncate long messages for the prompt
                if len(content) > 300:
                    content = content[:300] + "..."
                lines.append(f"[{ts}] ({channel}) {role}: {content}")

            return "\n".join(lines)
        except Exception as exc:
            log.warning("Failed to read conversation history: %s", exc)
            return "(error reading history)"

    def _get_unread_messages(self) -> str:
        """Read new entries from the shared chat log since last check."""
        try:
            entries, new_pos = read_chat_log_since(self._chat_log_position)
            self._chat_log_position = new_pos

            if not entries:
                return "(no new cross-channel messages)"

            lines = []
            for entry in entries[-30:]:  # Cap at 30 to keep prompt manageable
                ts = entry.get("ts", "")[:16]
                ch = entry.get("channel", "?")
                role = entry.get("role", "?")
                content = entry.get("content", "")
                if len(content) > 200:
                    content = content[:200] + "..."
                lines.append(f"[{ts}] ({ch}) {role}: {content}")

            return "\n".join(lines)
        except Exception as exc:
            log.warning("Failed to read chat log: %s", exc)
            return "(error reading chat log)"

    def _get_scheduled_tasks(self) -> str:
        """Format pending scheduled tasks for the prompt."""
        try:
            scheduler = get_scheduler()
            tasks = scheduler.tasks

            if not tasks:
                return "(no scheduled tasks)"

            lines = []
            for task in tasks.values():
                status = "enabled" if task.enabled else "disabled"
                next_run = task.next_run or "unknown"
                lines.append(f"- {task.name} [{status}] next: {next_run} | {task.prompt[:80]}")

            return "\n".join(lines)
        except Exception as exc:
            log.warning("Failed to read scheduler: %s", exc)
            return "(error reading scheduler)"

    def _get_system_status(self) -> str:
        """Collect basic system status information."""
        status_lines = []

        # Memory stats
        try:
            mem = get_memory()
            if mem:
                stats = mem.get_stats()
                status_lines.append(f"Memory: {stats['total_memories']} items, {stats['total_messages']} messages")
        except Exception:
            status_lines.append("Memory: unavailable")

        # Available frontier providers
        try:
            providers = list_available_providers()
            if providers:
                status_lines.append(f"Frontier providers: {', '.join(providers)}")
            else:
                status_lines.append("Frontier providers: none configured")
        except Exception:
            status_lines.append("Frontier providers: check failed")

        # Heartbeat meta
        status_lines.append(f"Heartbeat tick: #{self._tick_count}")
        status_lines.append(f"Interval: {self.interval}s")

        return "\n".join(status_lines)

    # -- Scheduler integration ----------------------------------------------

    async def _run_due_tasks(self):
        """Check the scheduler for due tasks and run them."""
        try:
            scheduler = get_scheduler()
            now = datetime.now()

            for task in list(scheduler.tasks.values()):
                if not task.enabled or not task.next_run:
                    continue

                try:
                    next_run = datetime.fromisoformat(task.next_run)
                    if now >= next_run:
                        log.info("Running due scheduled task: %s", task.name)
                        self.logger.write(f"Running scheduled task: {task.name}")
                        _task = asyncio.create_task(scheduler.run_task(task))  # noqa: RUF006
                except (ValueError, TypeError):
                    pass
        except Exception as exc:
            log.warning("Scheduler check failed: %s", exc)

    # -- LLM evaluation -----------------------------------------------------

    async def _evaluate(self, context: dict[str, Any]) -> Any | None:
        """
        Send context to the LLM and ask whether proactive action is needed.

        Tries the maude_callback first (which may route to local or frontier).
        Falls back to a direct local LLM call, then to frontier if local is
        unavailable.

        Returns parsed action dict/list or None if ALL_CLEAR.
        """
        prompt = HEARTBEAT_PROMPT.format(
            recent_messages=context.get("recent_messages", ""),
            scheduled_tasks=context.get("scheduled_tasks", ""),
            patterns=context.get("patterns", ""),
            system_status=context.get("system_status", ""),
            now=context.get("now", datetime.now().isoformat()),
        )

        response_text = await self._query_llm(prompt)

        if not response_text:
            log.debug("No LLM response received")
            return None

        # Check for ALL_CLEAR
        if "ALL_CLEAR" in response_text.upper():
            log.debug("Heartbeat: ALL_CLEAR")
            return None

        # Try to parse structured action(s) from the response
        return self._parse_actions(response_text)

    async def _query_llm(self, prompt: str) -> str | None:
        """
        Send the heartbeat prompt to an LLM and return the raw text.

        Strategy:
          1. Use the maude_callback if set (preferred -- lets the host
             application control routing and token accounting).
          2. Fall back to a direct local LLM call via the OpenAI client.
          3. If local fails, escalate to a frontier provider.
        """
        # Strategy 1: maude_callback
        if self._maude_callback:
            try:
                result = self._maude_callback(prompt)
                # Support both sync and async callbacks
                if asyncio.iscoroutine(result):
                    result = await result
                if result:
                    return result
            except Exception as exc:
                log.warning("Maude callback failed, trying local: %s", exc)

        # Strategy 2: direct local LLM
        try:
            return await self._query_local(prompt)
        except Exception as exc:
            log.warning("Local LLM failed, trying frontier: %s", exc)

        # Strategy 3: frontier escalation
        try:
            return self._query_frontier(prompt)
        except Exception as exc:
            log.error("All LLM strategies failed: %s", exc)
            return None

    async def _query_local(self, prompt: str) -> str:
        """Call the local LLM directly via the OpenAI-compatible endpoint."""
        if self._local_client is None:
            self._local_client = OpenAI(
                base_url=LOCAL_URL,
                api_key="not-needed",
            )

        # Run the blocking call in a thread so we don't stall the event loop
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._local_client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": ("You are MAUDE's proactive heartbeat system. Respond concisely."),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1024,
            ),
        )
        return response.choices[0].message.content

    def _query_frontier(self, prompt: str) -> str:
        """Escalate the heartbeat prompt to a frontier cloud model."""
        providers = list_available_providers()
        if not providers:
            raise RuntimeError("No frontier providers configured")

        resp = ask_frontier(
            query=prompt,
            system_prompt=(
                "You are MAUDE's proactive heartbeat system. Respond concisely in the requested JSON format."
            ),
        )
        return resp.content

    # -- Action parsing & execution -----------------------------------------

    def _parse_actions(self, text: str) -> list[dict[str, Any]] | None:
        """
        Attempt to extract structured action(s) from the LLM response.

        The LLM is instructed to return JSON but may wrap it in markdown
        code fences or return free-form text.  We handle all cases.
        """
        # Strip markdown code fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (possibly ```json)
            first_newline = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
            # Normalise to a list
            if isinstance(parsed, dict):
                return [parsed]
            elif isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

        # If JSON parsing failed, treat the whole response as a free-form
        # action description.
        if cleaned and "ALL_CLEAR" not in cleaned.upper():
            return [{"action": "check", "details": {"description": cleaned}, "reason": "free-form LLM response"}]

        return None

    async def _act(self, actions: list[dict[str, Any]]):
        """
        Execute the action(s) decided by the heartbeat LLM.
        """
        for action_obj in actions:
            action_type = action_obj.get("action", "unknown")
            details = action_obj.get("details", {})
            reason = action_obj.get("reason", "")

            log.info("Heartbeat action: %s -- %s", action_type, reason)
            self.logger.write_section(
                f"Action: {action_type}",
                f"**Reason:** {reason}\n\n**Details:**\n```json\n{json.dumps(details, indent=2)}\n```",
            )

            try:
                if action_type == "send_message":
                    await self._action_send_message(details)

                elif action_type == "run_tool":
                    await self._action_run_tool(details)

                elif action_type == "update_memory":
                    self._action_update_memory(details)

                elif action_type == "check":
                    await self._action_check(details)

                else:
                    log.warning("Unknown heartbeat action type: %s", action_type)
                    self.logger.write(f"Unknown action type: {action_type}", level="WARN")
            except Exception as exc:
                log.exception("Failed to execute heartbeat action %s: %s", action_type, exc)
                self.logger.write(f"Action {action_type} failed: {exc}", level="ERROR")

    async def _action_send_message(self, details: dict[str, Any]):
        """Send a proactive message to the user."""
        content = details.get("content") or details.get("message", "")
        channel = details.get("channel", "cli")
        channel_id = details.get("channel_id", "default")

        if not content:
            log.warning("send_message action with no content")
            return

        # Log the outgoing message to chat sync
        append_chat_log("heartbeat", "assistant", content)

        if self._gateway and channel != "cli":
            try:
                from channels import OutgoingMessage

                await self._gateway.send(
                    channel,
                    channel_id,
                    OutgoingMessage(text=f"[Heartbeat] {content}"),
                )
                self.logger.write(f"Sent proactive message to {channel}/{channel_id}")
            except Exception as exc:
                log.error("Gateway send failed: %s", exc)
                self.logger.write(f"Failed to send via gateway: {exc}", level="ERROR")
        else:
            # Fallback: log to daily file and chat sync so CLI can pick it up
            self.logger.write(f"Proactive message (no gateway): {content}")

    async def _action_run_tool(self, details: dict[str, Any]):
        """Execute a MAUDE tool autonomously."""
        tool_name = details.get("tool") or details.get("name", "")
        arguments = details.get("arguments") or details.get("args", {})

        if not tool_name:
            log.warning("run_tool action with no tool name")
            return

        if self._tool_executor:
            try:
                result = self._tool_executor(tool_name, arguments)
                # Support async executors
                if asyncio.iscoroutine(result):
                    result = await result
                self.logger.write_section(
                    f"Tool: {tool_name}",
                    f"**Arguments:** `{json.dumps(arguments)}`\n\n**Result:**\n```\n{str(result)[:2000]}\n```",
                )
            except Exception as exc:
                log.error("Tool execution failed: %s %s -- %s", tool_name, arguments, exc)
                self.logger.write(f"Tool {tool_name} failed: {exc}", level="ERROR")
        else:
            # No tool executor registered -- fall back to maude_core
            try:
                from maude_core import execute_tool

                result = execute_tool(tool_name, arguments)
                self.logger.write_section(
                    f"Tool (direct): {tool_name}",
                    f"**Arguments:** `{json.dumps(arguments)}`\n\n**Result:**\n```\n{str(result)[:2000]}\n```",
                )
            except Exception as exc:
                log.error("Direct tool execution failed: %s -- %s", tool_name, exc)

    def _action_update_memory(self, details: dict[str, Any]):
        """Update a persistent memory entry."""
        key = details.get("key", "")
        value = details.get("value", "")
        category = details.get("category", "fact")

        if not key or not value:
            log.warning("update_memory action with missing key/value")
            return

        try:
            mem = get_memory()
            if mem:
                mem.remember(key, value, category)
                self.logger.write(f"Memory updated: [{category}] {key}")
            else:
                log.warning("Memory system unavailable")
        except Exception as exc:
            log.error("Memory update failed: %s", exc)

    async def _action_check(self, details: dict[str, Any]):
        """
        Perform a follow-up check.  This sends the check description back
        to the LLM for a deeper investigation, then logs the findings.
        """
        description = details.get("description", "")
        if not description:
            return

        follow_up_prompt = (
            f"MAUDE heartbeat flagged something to check:\n\n"
            f"{description}\n\n"
            f"Investigate and report what you find.  If you need a tool, "
            f"describe which tool and arguments."
        )

        response = await self._query_llm(follow_up_prompt)
        if response:
            self.logger.write_section("Check result", response[:3000])

    # -- Pattern observation ------------------------------------------------

    def _observe_patterns(self, context: dict[str, Any]):
        """
        Analyse recent activity for recurring patterns and record them.
        """
        recent = context.get("recent_messages", "")
        if not recent or recent.startswith("("):
            return

        # Simple heuristic observations -- count message types/channels
        lines = recent.strip().split("\n")
        channels_seen: dict[str, int] = {}
        hour_counts: dict[int, int] = {}

        for line in lines:
            # Extract channel from "(channel)" pattern
            if "(" in line and ")" in line:
                start = line.index("(") + 1
                end = line.index(")")
                ch = line[start:end]
                channels_seen[ch] = channels_seen.get(ch, 0) + 1

            # Extract hour from timestamp "[YYYY-MM-DDTHH:MM]" or "[HH:MM]"
            if "[" in line and "]" in line:
                ts_part = line[line.index("[") + 1 : line.index("]")]
                try:
                    # Handle both "2024-01-15T14:30" and "14:30" formats
                    if "T" in ts_part:
                        hour = int(ts_part.split("T")[1][:2])
                    elif ":" in ts_part:
                        hour = int(ts_part.split(":")[0][-2:])
                    else:
                        continue
                    hour_counts[hour] = hour_counts.get(hour, 0) + 1
                except (ValueError, IndexError):
                    pass

        # Record channel usage patterns
        for ch, count in channels_seen.items():
            if count >= 2:
                self.patterns.observe(
                    key=f"channel_usage:{ch}",
                    description=f"User actively using {ch} channel",
                    tags=["channel", ch],
                )

        # Record active-hour patterns
        for hour, count in hour_counts.items():
            if count >= 3:
                self.patterns.observe(
                    key=f"active_hour:{hour}",
                    description=f"User frequently active around {hour:02d}:00",
                    tags=["time", "activity"],
                )


# ---------------------------------------------------------------------------
# Module-level singleton and convenience accessor
# ---------------------------------------------------------------------------

_engine: HeartbeatEngine | None = None


def get_heartbeat() -> HeartbeatEngine:
    """Get or create the global HeartbeatEngine instance."""
    global _engine
    if _engine is None:
        _engine = HeartbeatEngine()
    return _engine
