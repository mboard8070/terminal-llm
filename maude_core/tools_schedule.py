"""
Task scheduler tool implementation.
"""

from tool_registry import register_tool
from .log import log


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


# ── Registry wrapper ──────────────────────────────────────────

@register_tool("schedule_task")
def _dispatch_schedule_task(args):
    return tool_schedule_task(
        args.get("action", "list"),
        args.get("name"),
        args.get("cron"),
        args.get("prompt"),
        args.get("task_id")
    )
