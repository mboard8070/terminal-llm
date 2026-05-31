"""
MAUDE Proactive Scheduler.

Schedule MAUDE to run tasks and message you first.
"""

import asyncio
import json
import os
import re
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from rich.console import Console

console = Console()

# Try to import croniter
try:
    from croniter import croniter

    CRONITER_AVAILABLE = True
except ImportError:
    CRONITER_AVAILABLE = False
    console.print("[yellow]croniter not installed. Run: pip install croniter[/yellow]")


@dataclass
class ScheduledTask:
    """A scheduled task."""

    id: str
    name: str
    cron: str  # Cron expression (or special: @hourly, @daily, etc.)
    prompt: str  # What to ask MAUDE
    channel: str = "cli"  # Where to send result
    channel_id: str = "default"  # Specific chat/channel
    enabled: bool = True
    last_run: str | None = None
    next_run: str | None = None
    run_count: int = 0
    last_result: str | None = None
    last_status: str | None = None
    last_duration_seconds: float | None = None
    failure_count: int = 0
    last_error: str | None = None
    task_type: str = "prompt"
    mission_id: str | None = None
    mission_auto_complete: bool | None = None
    mission_max_steps: int = 10


class ProactiveScheduler:
    """Schedule MAUDE to run tasks and message you."""

    CONFIG_FILE = Path.home() / ".config" / "maude" / "schedules.json"

    # Special schedule shortcuts
    SPECIAL_SCHEDULES: ClassVar[dict[str, str]] = {
        "@hourly": "0 * * * *",
        "@daily": "0 9 * * *",  # 9 AM
        "@weekly": "0 9 * * 1",  # Monday 9 AM
        "@monthly": "0 9 1 * *",  # 1st of month 9 AM
        "@morning": "0 8 * * *",  # 8 AM
        "@evening": "0 18 * * *",  # 6 PM
        "@workdays": "0 9 * * 1-5",  # Weekdays 9 AM
    }

    def __init__(self):
        self.tasks: dict[str, ScheduledTask] = {}
        self.maude_callback: Callable | None = None
        self.gateway = None  # Channel gateway for sending messages
        self.running = False
        self._loop_task: asyncio.Task | None = None
        self._running_task_ids: set[str] = set()
        self._load_tasks()

    def _load_tasks(self):
        """Load scheduled tasks from disk."""
        if self.CONFIG_FILE.exists():
            try:
                data = json.loads(self.CONFIG_FILE.read_text())
                field_names = set(ScheduledTask.__dataclass_fields__)
                for task_data in data:
                    task_data = {key: value for key, value in task_data.items() if key in field_names}
                    task = ScheduledTask(**task_data)
                    # Recalculate next_run
                    task.next_run = self._calculate_next_run(task.cron)
                    self.tasks[task.id] = task
            except Exception as e:
                console.print(f"[yellow]Error loading schedules: {e}[/yellow]")

    def _save_tasks(self):
        """Save scheduled tasks to disk."""
        self.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(t) for t in self.tasks.values()]
        self.CONFIG_FILE.write_text(json.dumps(data, indent=2))

    def _resolve_cron(self, cron: str) -> str:
        """Resolve special schedule names to cron expressions."""
        return self.SPECIAL_SCHEDULES.get(cron, cron)

    def _calculate_next_run(self, cron: str) -> str | None:
        """Calculate next run time from cron expression."""
        if not CRONITER_AVAILABLE:
            return None

        try:
            cron_expr = self._resolve_cron(cron)
            cron_iter = croniter(cron_expr, datetime.now())
            next_time = cron_iter.get_next(datetime)
            return next_time.isoformat()
        except:
            return None

    def set_maude_callback(self, callback: Callable):
        """Set the MAUDE processing callback."""
        self.maude_callback = callback

    def set_gateway(self, gateway):
        """Set the channel gateway for sending messages."""
        self.gateway = gateway

    def _mission_report_from_state(self, task: ScheduledTask, result: str) -> str | None:
        """Build a concrete scheduled mission report from persisted mission state."""
        mission_id = self._mission_id_from_task(task)
        if not mission_id:
            return None

        missions_dir = Path(os.environ.get("MAUDE_MISSIONS_DIR", Path(__file__).parent / "data" / "missions"))
        mission_path = missions_dir / f"{mission_id}.json"
        if not mission_path.exists():
            return None

        try:
            mission = json.loads(mission_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        steps = mission.get("steps") or []
        latest = None
        for step in steps:
            if step.get("last_run_at"):
                if latest is None or str(step.get("last_run_at")) > str(latest.get("last_run_at")):
                    latest = step
        if latest is None:
            latest = next((step for step in steps if step.get("status") != "done"), None)
        if latest is None:
            latest = steps[-1] if steps else {}

        done = sum(1 for step in steps if step.get("status") == "done")
        total = len(steps)
        next_step = next((step for step in steps if step.get("status") != "done"), None)
        blockers = mission.get("blockers") or []
        artifacts = mission.get("artifacts") or []
        last_result = str(latest.get("last_result") or result or "")

        evidence = self._extract_report_lines(last_result, limit=4)
        optimization = self._extract_optimization_finding(last_result) or self._extract_optimization_finding(result)
        if not optimization:
            optimization = "No new speed, token, or runtime issue was recorded in this tick."

        changed = "No new artifact paths were added."
        if artifacts:
            changed = "Tracked artifacts: " + ", ".join(str(item) for item in artifacts[-3:])

        lines = [
            "**Done**",
            (
                f"Ran `{latest.get('id', 'unknown')}: {latest.get('title', 'unknown')}` "
                f"with status `{latest.get('status', 'unknown')}`. "
                f"Mission progress is {done}/{total} steps."
            ),
            "",
            "**Evidence**",
            evidence or "No detailed execution evidence was recorded.",
            "",
            "**Files / artifacts**",
            changed,
            "",
            "**Optimization finding**",
            optimization,
            "",
            "**Blockers**",
            "; ".join(str(item) for item in blockers) if blockers else "None.",
            "",
            "**Next**",
            (
                f"Next mission step is `{next_step.get('id')}: {next_step.get('title')}`."
                if next_step
                else "No pending mission steps remain; review whether to reset or extend the mission."
            ),
        ]
        return "\n".join(lines)

    @staticmethod
    def _mission_id_from_task(task: ScheduledTask) -> str:
        """Extract a scheduled mission id from the task prompt."""
        if task.mission_id:
            return Path(str(task.mission_id)).name
        patterns = [
            r"mission_tick tool with id '([^']+)'",
            r"mission_tick[^\n]*\bid['\"]?\s*[:=]\s*['\"]([^'\"]+)",
            r"mission_drain[^\n]*\bid['\"]?\s*[:=]\s*['\"]([^'\"]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, task.prompt)
            if match:
                return Path(match.group(1)).name
        return ""

    @staticmethod
    def _mission_auto_complete_from_task(task: ScheduledTask) -> bool:
        if task.mission_auto_complete is not None:
            return bool(task.mission_auto_complete)
        match = re.search(r"auto_complete\s*=?\s*(true|false)", task.prompt, flags=re.IGNORECASE)
        return match is None or match.group(1).lower() == "true"

    async def _run_mission_task(self, task: ScheduledTask) -> str | None:
        """Run scheduled mission steps directly so cadence does not depend on prompt wording."""
        mission_id = self._mission_id_from_task(task)
        if not mission_id:
            return None

        from maude_core import execute_tool

        return execute_tool(
            "mission_drain",
            {
                "id": mission_id,
                "auto_complete": self._mission_auto_complete_from_task(task),
                "max_steps": task.mission_max_steps,
            },
        )

    @staticmethod
    def _extract_report_lines(text: str, limit: int = 4) -> str:
        preferred = []
        fallback = []
        useful = (
            "Plan executed",
            "Matches for",
            "No matches",
            "issues=",
            "skills_checked",
            "skill_scan_seconds",
            "grep_elapsed",
            "Mission step",
            "Successfully",
            "Logged mission event",
        )
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(("###", "## Stage", "File:", "Directory:")):
                continue
            if re.match(r"^\d+\s+\|", line):
                continue
            if len(line) > 220:
                line = line[:217].rstrip() + "..."
            if any(marker in line for marker in useful):
                preferred.append(line)
            elif not line.startswith(("---", "```", "[DIR]", "[FILE]")):
                fallback.append(line)
        lines = (preferred + fallback)[:limit]
        return "\n".join(f"- {line}" for line in lines)

    @staticmethod
    def _extract_optimization_finding(text: str | None) -> str:
        if not text:
            return ""
        patterns = [
            r"Optimization finding\*\*\s*(.+?)(?:\n\*\*|\Z)",
            r"Optimization finding\s*(.+?)(?:\n[A-Z][A-Za-z ]+:|\Z)",
            r"(?:elapsed|took|latency|token|runtime|slow|grep_elapsed|scan_seconds)[^\n]*(?:\n[^\n]*)?",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if match:
                finding = " ".join(match.group(1 if match.lastindex else 0).split())
                return finding[:500]
        return ""

    def schedule(
        self,
        name: str,
        cron: str,
        prompt: str,
        channel: str = "cli",
        channel_id: str = "default",
        task_type: str = "prompt",
        mission_id: str | None = None,
        mission_auto_complete: bool | None = None,
        mission_max_steps: int = 10,
    ) -> str:
        """Schedule a new task."""

        # Validate cron
        cron_expr = self._resolve_cron(cron)
        next_run = self._calculate_next_run(cron_expr)

        if not next_run and CRONITER_AVAILABLE:
            return f"Invalid cron expression: {cron}\n\nExamples:\n  '0 9 * * *' (daily 9 AM)\n  '@daily', '@hourly', '@morning'"

        task_id = str(uuid.uuid4())[:8]

        task = ScheduledTask(
            id=task_id,
            name=name,
            cron=cron,
            prompt=prompt,
            channel=channel,
            channel_id=channel_id,
            next_run=next_run,
            task_type=task_type,
            mission_id=Path(str(mission_id)).name if mission_id else None,
            mission_auto_complete=mission_auto_complete,
            mission_max_steps=max(1, min(int(mission_max_steps or 10), 25)),
        )
        self.tasks[task_id] = task
        self._save_tasks()

        return f"Scheduled '{name}' (ID: {task_id})\nCron: {cron_expr}\nNext run: {next_run or 'unknown'}"

    def unschedule(self, task_id: str) -> str:
        """Remove a scheduled task."""
        if task_id in self.tasks:
            name = self.tasks[task_id].name
            del self.tasks[task_id]
            self._save_tasks()
            return f"Removed task '{name}'"
        return f"Task {task_id} not found"

    def enable_task(self, task_id: str) -> str:
        """Enable a task."""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = True
            self.tasks[task_id].next_run = self._calculate_next_run(self.tasks[task_id].cron)
            self._save_tasks()
            return f"Enabled task '{self.tasks[task_id].name}'"
        return f"Task {task_id} not found"

    def disable_task(self, task_id: str) -> str:
        """Disable a task."""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = False
            self._save_tasks()
            return f"Disabled task '{self.tasks[task_id].name}'"
        return f"Task {task_id} not found"

    def list_tasks(self) -> str:
        """List all scheduled tasks."""
        if not self.tasks:
            return (
                "No scheduled tasks.\n\n"
                "Schedule a task:\n"
                '  /schedule add "Morning Brief" @morning "Give me a summary of my calendar and weather"\n\n'
                "Cron shortcuts: @hourly, @daily, @morning, @evening, @weekly, @workdays"
            )

        lines = ["Scheduled Tasks:\n"]
        for task in sorted(self.tasks.values(), key=lambda t: t.next_run or ""):
            status = "[green]✓[/green]" if task.enabled else "[red]✗[/red]"
            lines.append(f"  {status} {task.id}: {task.name}")
            lines.append(f"      Schedule: {task.cron}")
            lines.append(f"      Next: {task.next_run or 'disabled'}")
            lines.append(f"      Channel: {task.channel}")
            if task.run_count:
                lines.append(f"      Runs: {task.run_count}")
        return "\n".join(lines)

    async def run_task(self, task: ScheduledTask) -> str:
        """Execute a scheduled task."""
        console.print(f"[cyan]Running scheduled task: {task.name}[/cyan]")
        started_at = datetime.now()

        try:
            result = await self._run_mission_task(task)
            if result is None:
                if not self.maude_callback:
                    return "No MAUDE callback configured"
                # Call MAUDE with the prompt
                result = await self.maude_callback(task.prompt)
            report = self._mission_report_from_state(task, result)
            if report:
                result = report

            # Update task state
            task.last_run = datetime.now().isoformat()
            task.run_count += 1
            task.last_result = result[:2000] if result else None  # Store a compact status snapshot
            task.last_status = "ok"
            task.last_duration_seconds = round((datetime.now() - started_at).total_seconds(), 3)
            task.failure_count = 0
            task.last_error = None
            task.next_run = self._calculate_next_run(task.cron)
            self._save_tasks()

            # Send result to channel
            if self.gateway and task.channel != "cli":
                from channels import OutgoingMessage

                await self.gateway.send(
                    task.channel, task.channel_id, OutgoingMessage(text=f"**{task.name}**\n\n{result}")
                )
            else:
                # CLI output
                console.print(f"\n[bold cyan]Scheduled Task: {task.name}[/bold cyan]")
                console.print(result)
                console.print()

            return result

        except Exception as e:
            error_msg = f"Error running task '{task.name}': {e}"
            console.print(f"[red]{error_msg}[/red]")
            task.last_run = datetime.now().isoformat()
            task.run_count += 1
            task.last_result = error_msg[:2000]
            task.last_status = "error"
            task.last_duration_seconds = round((datetime.now() - started_at).total_seconds(), 3)
            task.failure_count += 1
            task.last_error = str(e)[:1000]
            task.next_run = self._calculate_next_run(task.cron)
            self._save_tasks()
            return error_msg

    async def run_task_by_id(self, task_id: str) -> str:
        """Run a specific task by ID."""
        if task_id not in self.tasks:
            return f"Task {task_id} not found"
        return await self._run_task_once(self.tasks[task_id])

    async def _run_task_once(self, task: ScheduledTask) -> str:
        """Run a scheduled task with duplicate suppression."""
        if task.id in self._running_task_ids:
            return f"Task {task.id} is already running"
        self._running_task_ids.add(task.id)
        try:
            return await self.run_task(task)
        finally:
            self._running_task_ids.discard(task.id)

    async def _scheduler_loop(self):
        """Main scheduler loop - check and run due tasks."""
        console.print("[dim]Scheduler started[/dim]")

        while self.running:
            now = datetime.now()

            for task in list(self.tasks.values()):
                if not task.enabled or not task.next_run:
                    continue

                try:
                    next_run = datetime.fromisoformat(task.next_run)
                    if now >= next_run and task.id not in self._running_task_ids:
                        # Task is due
                        _task = asyncio.create_task(self._run_task_once(task))  # noqa: RUF006
                except:
                    pass

            # Check every 30 seconds
            await asyncio.sleep(30)

    async def start(self):
        """Start the scheduler loop."""
        if not CRONITER_AVAILABLE:
            console.print("[yellow]Scheduler disabled: croniter not installed[/yellow]")
            return

        self.running = True
        self._loop_task = asyncio.create_task(self._scheduler_loop())

    async def stop(self):
        """Stop the scheduler."""
        self.running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        console.print("[dim]Scheduler stopped[/dim]")


# Global scheduler instance
_scheduler: ProactiveScheduler | None = None


def get_scheduler() -> ProactiveScheduler:
    """Get or create the global scheduler."""
    global _scheduler
    if _scheduler is None:
        _scheduler = ProactiveScheduler()
    return _scheduler
