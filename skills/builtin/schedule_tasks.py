"""Schedule skill - natural language task scheduling via MAUDE's ProactiveScheduler."""

import re
from datetime import datetime, timedelta
from skills import skill


# Day name → cron day-of-week number
_DAY_MAP = {
    "monday": "1", "mon": "1",
    "tuesday": "2", "tue": "2",
    "wednesday": "3", "wed": "3",
    "thursday": "4", "thu": "4",
    "friday": "5", "fri": "5",
    "saturday": "6", "sat": "6",
    "sunday": "0", "sun": "0",
}

# Named shortcuts for common schedules
_NAMED_SCHEDULES = {
    "every morning": "0 8 * * *",
    "every evening": "0 18 * * *",
    "every night": "0 21 * * *",
    "every afternoon": "0 14 * * *",
    "every hour": "0 * * * *",
    "hourly": "0 * * * *",
    "daily": "0 9 * * *",
    "weekly": "0 9 * * 1",
    "weekdays": "0 9 * * 1-5",
    "weekends": "0 10 * * 0,6",
    "monthly": "0 9 1 * *",
    "every weekday": "0 9 * * 1-5",
    "every weekend": "0 10 * * 0,6",
}


def _parse_time(time_str: str) -> tuple:
    """Parse a time string like '3pm', '9:30am', '14:00' into (hour, minute)."""
    time_str = time_str.strip().lower().replace(" ", "")

    # 24-hour format: 14:00, 9:30
    m = re.match(r'^(\d{1,2}):(\d{2})$', time_str)
    if m:
        return int(m.group(1)), int(m.group(2))

    # 12-hour format: 3pm, 9:30am, 12am
    m = re.match(r'^(\d{1,2})(?::(\d{2}))?\s*(am|pm)$', time_str)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        if m.group(3) == "pm" and hour != 12:
            hour += 12
        elif m.group(3) == "am" and hour == 12:
            hour = 0
        return hour, minute

    return None, None


def _parse_schedule(when: str) -> str:
    """Parse a natural language schedule into a cron expression.

    Supports:
      - Named shortcuts: "every morning", "weekdays", "hourly"
      - Time patterns: "daily at 3pm", "every day at 9:30am"
      - Day patterns: "every Monday at 10am", "every Friday at 5pm"
      - Weekday-time: "weekdays at 9am", "weekends at 10am"
      - Interval: "every 30 minutes", "every 2 hours", "every 6 hours"
      - One-shot: "in 30 minutes", "in 2 hours"
      - @-shortcuts: "@morning", "@daily", "@hourly", "@workdays"
      - Raw cron: "0 9 * * 1-5"
    """
    when_lower = when.strip().lower()

    # 1. Check named shortcuts first
    if when_lower in _NAMED_SCHEDULES:
        return _NAMED_SCHEDULES[when_lower]

    # 2. @-shortcuts (pass through to scheduler which handles them)
    if when_lower.startswith("@"):
        return when_lower

    # 3. "every N minutes/hours"
    m = re.match(r'every\s+(\d+)\s+(minute|minutes|min|mins|hour|hours|hr|hrs)', when_lower)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit.startswith("min"):
            if n < 1 or n > 59:
                return when  # invalid, pass through
            return f"*/{n} * * * *"
        else:  # hours
            if n < 1 or n > 23:
                return when
            return f"0 */{n} * * *"

    # 4. "every minute" (no number)
    if re.match(r'every\s+minute', when_lower):
        return "* * * * *"

    # 5. "daily at <time>" / "every day at <time>"
    m = re.match(r'(?:daily|every\s+day)\s+at\s+(.+)', when_lower)
    if m:
        hour, minute = _parse_time(m.group(1))
        if hour is not None:
            return f"{minute} {hour} * * *"

    # 6. "every <day> at <time>" — e.g. "every Monday at 10am"
    m = re.match(r'every\s+(\w+)\s+at\s+(.+)', when_lower)
    if m:
        day_name = m.group(1).lower()
        day_num = _DAY_MAP.get(day_name)
        if day_num:
            hour, minute = _parse_time(m.group(2))
            if hour is not None:
                return f"{minute} {hour} * * {day_num}"

    # 7. "weekdays at <time>" / "weekends at <time>"
    m = re.match(r'(weekdays|weekends)\s+at\s+(.+)', when_lower)
    if m:
        days = "1-5" if m.group(1) == "weekdays" else "0,6"
        hour, minute = _parse_time(m.group(2))
        if hour is not None:
            return f"{minute} {hour} * * {days}"

    # 8. "every <day>" without time — default to 9am
    m = re.match(r'every\s+(\w+)$', when_lower)
    if m:
        day_name = m.group(1).lower()
        day_num = _DAY_MAP.get(day_name)
        if day_num:
            return f"0 9 * * {day_num}"

    # 9. "at <time>" — daily at that time
    m = re.match(r'at\s+(.+)', when_lower)
    if m:
        hour, minute = _parse_time(m.group(1))
        if hour is not None:
            return f"{minute} {hour} * * *"

    # 10. "in N minutes/hours" — one-shot (approximate with specific cron)
    m = re.match(r'in\s+(\d+)\s+(minute|minutes|min|mins|hour|hours|hr|hrs)', when_lower)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        now = datetime.now()
        if unit.startswith("min"):
            target = now + timedelta(minutes=n)
        else:
            target = now + timedelta(hours=n)
        return f"{target.minute} {target.hour} {target.day} {target.month} *"

    # 11. Raw cron expression — validate it looks like cron (5 space-separated fields)
    parts = when.strip().split()
    if len(parts) == 5 and all(
        re.match(r'^[\d\*\/\-\,]+$', p) for p in parts
    ):
        return when.strip()

    # Fallback: return as-is and let the scheduler validate
    return when


def _human_relative_time(iso_str: str) -> str:
    """Convert an ISO datetime string to a human-readable relative time."""
    if not iso_str:
        return "unknown"
    try:
        target = datetime.fromisoformat(iso_str)
        now = datetime.now()
        delta = target - now

        if delta.total_seconds() < 0:
            return "overdue"

        total_minutes = int(delta.total_seconds() / 60)
        if total_minutes < 1:
            return "now"
        if total_minutes < 60:
            return f"in {total_minutes} minute{'s' if total_minutes != 1 else ''}"
        total_hours = total_minutes // 60
        remaining_mins = total_minutes % 60
        if total_hours < 24:
            if remaining_mins > 0:
                return f"in {total_hours}h {remaining_mins}m"
            return f"in {total_hours} hour{'s' if total_hours != 1 else ''}"
        days = total_hours // 24
        if days == 1:
            return f"tomorrow at {target.strftime('%I:%M %p').lstrip('0')}"
        return f"in {days} days ({target.strftime('%a %I:%M %p').lstrip('0')})"
    except Exception:
        return iso_str


@skill(
    name="schedule",
    description=(
        "Schedule tasks for MAUDE to run automatically. "
        "Supports natural language schedules like 'every morning', 'weekdays at 9am', "
        "'every 30 minutes', 'in 2 hours'. "
        "Actions: schedule (create), list, cancel, pause, resume, run."
    ),
    version="1.0.0",
    author="MAUDE",
    triggers=["schedule", "remind", "recurring", "cron", "timer", "alarm"],
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["schedule", "list", "cancel", "pause", "resume", "run"],
                "description": "Action to perform",
                "default": "list"
            },
            "task_name": {
                "type": "string",
                "description": "Name/label for the scheduled task (for 'schedule' action)"
            },
            "when": {
                "type": "string",
                "description": "When to run — natural language ('every morning', 'daily at 3pm', 'every 30 minutes', 'in 2 hours') or cron expression"
            },
            "prompt": {
                "type": "string",
                "description": "What MAUDE should do when the task runs"
            },
            "task_id": {
                "type": "string",
                "description": "Task ID (for cancel/pause/resume/run actions)"
            }
        }
    }
)
def schedule(
    action: str = "list",
    task_name: str = "",
    when: str = "",
    prompt: str = "",
    task_id: str = ""
) -> str:
    """Manage scheduled tasks."""
    from scheduler import get_scheduler
    sched = get_scheduler()

    if action == "schedule":
        if not task_name:
            return "Error: please provide a 'task_name' for the scheduled task."
        if not when:
            return "Error: please provide 'when' to schedule (e.g. 'every morning', 'daily at 3pm')."
        if not prompt:
            return "Error: please provide a 'prompt' — what MAUDE should do when the task runs."

        cron_expr = _parse_schedule(when)
        result = sched.schedule(name=task_name, cron=cron_expr, prompt=prompt)

        # Enhance the result with human-readable next run
        if "Invalid cron" in result:
            return (
                f"Could not parse schedule: \"{when}\"\n\n"
                "Examples:\n"
                "  'every morning' — 8 AM daily\n"
                "  'daily at 3pm' — 3 PM daily\n"
                "  'weekdays at 9am' — 9 AM Mon-Fri\n"
                "  'every 30 minutes' — every half hour\n"
                "  'every Monday at 10am' — weekly\n"
                "  'in 2 hours' — one-shot\n"
                "  '0 9 * * 1-5' — raw cron"
            )
        return result

    elif action == "list":
        if not sched.tasks:
            return (
                "No scheduled tasks.\n\n"
                "Create one with:\n"
                "  \"schedule a weather check every morning\"\n"
                "  \"remind me to stand up every 30 minutes\"\n"
                "  \"schedule daily standup notes at 9am\""
            )

        lines = [f"Scheduled Tasks ({len(sched.tasks)}):\n"]
        for task in sorted(sched.tasks.values(), key=lambda t: t.next_run or "z"):
            status = "ON" if task.enabled else "OFF"
            next_human = _human_relative_time(task.next_run) if task.enabled else "paused"
            lines.append(f"  [{status}] {task.id}: {task.name}")
            lines.append(f"       Schedule: {task.cron}")
            lines.append(f"       Next run: {next_human}")
            lines.append(f"       Prompt: {task.prompt[:60]}{'...' if len(task.prompt) > 60 else ''}")
            if task.run_count > 0:
                lines.append(f"       Runs: {task.run_count}")
            lines.append("")
        return "\n".join(lines).rstrip()

    elif action == "cancel":
        if not task_id:
            return "Error: please provide 'task_id' to cancel."
        return sched.unschedule(task_id)

    elif action == "pause":
        if not task_id:
            return "Error: please provide 'task_id' to pause."
        return sched.disable_task(task_id)

    elif action == "resume":
        if not task_id:
            return "Error: please provide 'task_id' to resume."
        return sched.enable_task(task_id)

    elif action == "run":
        if not task_id:
            return "Error: please provide 'task_id' to run."
        # Synchronous wrapper for async run
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(sched.run_task_by_id(task_id))
                return f"Running task {task_id}..."
            else:
                return loop.run_until_complete(sched.run_task_by_id(task_id))
        except Exception:
            return f"Task {task_id} scheduled to run."

    return f"Unknown action: {action}. Use: schedule, list, cancel, pause, resume, run."
