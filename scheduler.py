"""
MAUDE Proactive Scheduler.

Schedule MAUDE to run tasks and message you first.
"""

import os
import json
import asyncio
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Callable, Optional, Dict, List, Any
from pathlib import Path
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
    cron: str                    # Cron expression (or special: @hourly, @daily, etc.)
    prompt: str                  # What to ask MAUDE
    channel: str = "cli"         # Where to send result
    channel_id: str = "default"  # Specific chat/channel
    enabled: bool = True
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    run_count: int = 0
    last_result: Optional[str] = None


class ProactiveScheduler:
    """Schedule MAUDE to run tasks and message you."""

    CONFIG_FILE = Path.home() / ".config" / "maude" / "schedules.json"

    # Special schedule shortcuts
    SPECIAL_SCHEDULES = {
        "@hourly": "0 * * * *",
        "@daily": "0 9 * * *",      # 9 AM
        "@weekly": "0 9 * * 1",     # Monday 9 AM
        "@monthly": "0 9 1 * *",    # 1st of month 9 AM
        "@morning": "0 8 * * *",    # 8 AM
        "@evening": "0 18 * * *",   # 6 PM
        "@workdays": "0 9 * * 1-5", # Weekdays 9 AM
    }

    def __init__(self):
        self.tasks: Dict[str, ScheduledTask] = {}
        self.maude_callback: Optional[Callable] = None
        self.gateway = None  # Channel gateway for sending messages
        self.running = False
        self._loop_task: Optional[asyncio.Task] = None
        self._load_tasks()

    def _load_tasks(self):
        """Load scheduled tasks from disk."""
        if self.CONFIG_FILE.exists():
            try:
                data = json.loads(self.CONFIG_FILE.read_text())
                for task_data in data:
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

    def _calculate_next_run(self, cron: str) -> Optional[str]:
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

    def schedule(
        self,
        name: str,
        cron: str,
        prompt: str,
        channel: str = "cli",
        channel_id: str = "default"
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
            next_run=next_run
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
                "  /schedule add \"Morning Brief\" @morning \"Give me a summary of my calendar and weather\"\n\n"
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
        if not self.maude_callback:
            return "No MAUDE callback configured"

        console.print(f"[cyan]Running scheduled task: {task.name}[/cyan]")

        try:
            # Call MAUDE with the prompt
            result = await self.maude_callback(task.prompt)

            # Update task state
            task.last_run = datetime.now().isoformat()
            task.run_count += 1
            task.last_result = result[:500] if result else None  # Store truncated result
            task.next_run = self._calculate_next_run(task.cron)
            self._save_tasks()

            # Send result to channel
            if self.gateway and task.channel != "cli":
                from channels import OutgoingMessage
                await self.gateway.send(
                    task.channel,
                    task.channel_id,
                    OutgoingMessage(text=f"**{task.name}**\n\n{result}")
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
            return error_msg

    async def run_task_by_id(self, task_id: str) -> str:
        """Run a specific task by ID."""
        if task_id not in self.tasks:
            return f"Task {task_id} not found"
        return await self.run_task(self.tasks[task_id])

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
                    if now >= next_run:
                        # Task is due
                        asyncio.create_task(self.run_task(task))
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
_scheduler: Optional[ProactiveScheduler] = None


def get_scheduler() -> ProactiveScheduler:
    """Get or create the global scheduler."""
    global _scheduler
    if _scheduler is None:
        _scheduler = ProactiveScheduler()
    return _scheduler


def handle_schedule_command(args: list) -> str:
    """Handle /schedule command."""
    scheduler = get_scheduler()

    if not args:
        return scheduler.list_tasks()

    action = args[0].lower()

    if action == "list":
        return scheduler.list_tasks()

    elif action == "add" and len(args) >= 4:
        # /schedule add "name" @daily "prompt"
        name = args[1].strip('"\'')
        cron = args[2]
        prompt = " ".join(args[3:]).strip('"\'')
        return scheduler.schedule(name, cron, prompt)

    elif action == "remove" and len(args) > 1:
        return scheduler.unschedule(args[1])

    elif action == "enable" and len(args) > 1:
        return scheduler.enable_task(args[1])

    elif action == "disable" and len(args) > 1:
        return scheduler.disable_task(args[1])

    elif action == "run" and len(args) > 1:
        # Synchronous wrapper for async run
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule it
                asyncio.create_task(scheduler.run_task_by_id(args[1]))
                return f"Running task {args[1]}..."
            else:
                return loop.run_until_complete(scheduler.run_task_by_id(args[1]))
        except:
            return "Task scheduled to run."

    elif action == "test":
        # Quick test task
        return scheduler.schedule(
            name="Test Task",
            cron="*/5 * * * *",  # Every 5 minutes
            prompt="Say hello and tell me the current time."
        )

    return (
        f"Unknown schedule command: {action}\n\n"
        "Usage:\n"
        "  /schedule                           - List tasks\n"
        "  /schedule add \"name\" @daily \"prompt\" - Add task\n"
        "  /schedule remove <id>               - Remove task\n"
        "  /schedule enable <id>               - Enable task\n"
        "  /schedule disable <id>              - Disable task\n"
        "  /schedule run <id>                  - Run task now\n\n"
        "Cron shortcuts: @hourly, @daily, @morning, @evening, @weekly, @workdays"
    )
