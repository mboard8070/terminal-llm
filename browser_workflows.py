"""
Browser Workflow Engine for MAUDE — repeatable, schedulable browser automations.

Workflows are saved as JSON in data/workflows/. Each workflow is a sequence of
browser steps (open, extract, click, type, etc.) that execute deterministically.

Change detection: extract steps are compared to the previous run. If the extracted
text differs, the change is logged and optionally emailed.

Scheduling: integrates with MAUDE's ProactiveScheduler via cron expressions.

Usage (via MAUDE tools):
    workflow_create  — define a new workflow
    workflow_run     — execute a workflow now
    workflow_list    — list saved workflows
    workflow_get     — view a workflow definition
    workflow_delete  — remove a workflow
    workflow_history — view past run results
    workflow_schedule — attach a cron schedule
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"
WORKFLOWS_DIR = DATA_DIR / "workflows"
RUNS_DIR = DATA_DIR / "workflow_runs"


# ─────────────────────────────────────────────────────────────────────────────
# Step Execution
# ─────────────────────────────────────────────────────────────────────────────


def _import_browser():
    """Lazy-import browser tools to avoid circular imports."""
    from browser import (
        browser_click,
        browser_close,
        browser_extract,
        browser_fill_form,
        browser_navigate,
        browser_open,
        browser_screenshot,
        browser_select,
        browser_type,
    )

    return {
        "open": browser_open,
        "click": browser_click,
        "type": browser_type,
        "navigate": browser_navigate,
        "extract": browser_extract,
        "screenshot": browser_screenshot,
        "fill_form": browser_fill_form,
        "select": browser_select,
        "close": browser_close,
    }


def _execute_step(step: dict, browser_fns: dict) -> str:
    """Execute a single workflow step. Returns the result string."""
    action = step.get("action", "")

    if action == "wait":
        seconds = step.get("seconds", 2)
        time.sleep(seconds)
        return f"Waited {seconds}s"

    if action == "open":
        return browser_fns["open"](step["url"])

    if action == "navigate":
        return browser_fns["navigate"](step["url"])

    if action == "click":
        return browser_fns["click"](step["selector"])

    if action == "type":
        return browser_fns["type"](step["selector"], step["text"])

    if action == "extract":
        return browser_fns["extract"](step.get("selector"))

    if action == "screenshot":
        return browser_fns["screenshot"]()

    if action == "fill_form":
        return browser_fns["fill_form"](step["fields"])

    if action == "select":
        return browser_fns["select"](step["selector"], step["value"])

    if action == "close":
        return browser_fns["close"]()

    return f"Unknown action: {action}"


# ─────────────────────────────────────────────────────────────────────────────
# Workflow Engine
# ─────────────────────────────────────────────────────────────────────────────


class WorkflowEngine:
    """Manages and executes browser workflows."""

    def __init__(self):
        WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)
        RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # ── CRUD ──────────────────────────────────────────────────────────────

    def create(self, name: str, steps: list, description: str = "", notify_email: str = None) -> str:
        """Create or update a workflow definition."""
        wf_id = name.lower().replace(" ", "-").replace("_", "-")

        # Validate steps
        valid_actions = {
            "open",
            "navigate",
            "click",
            "type",
            "extract",
            "screenshot",
            "fill_form",
            "select",
            "wait",
            "close",
        }
        for i, step in enumerate(steps):
            action = step.get("action")
            if action not in valid_actions:
                return f"Error: Step {i} has invalid action '{action}'. Valid: {', '.join(sorted(valid_actions))}"

        now = datetime.now(UTC).isoformat()
        path = WORKFLOWS_DIR / f"{wf_id}.json"
        existing = json.loads(path.read_text()) if path.exists() else None

        workflow = {
            "id": wf_id,
            "name": name,
            "description": description,
            "steps": steps,
            "notify_email": notify_email,
            "schedule": existing["schedule"] if existing and "schedule" in existing else None,
            "created": existing["created"] if existing else now,
            "updated": now,
        }
        path.write_text(json.dumps(workflow, indent=2))

        verb = "Updated" if existing else "Created"
        return f"{verb} workflow '{name}' ({len(steps)} steps). ID: {wf_id}"

    def get(self, wf_id: str) -> str:
        """Get a workflow definition as formatted text."""
        path = WORKFLOWS_DIR / f"{wf_id}.json"
        if not path.exists():
            return f"Error: Workflow '{wf_id}' not found."

        wf = json.loads(path.read_text())
        lines = [
            f"Workflow: {wf['name']}",
            f"ID: {wf['id']}",
            f"Description: {wf.get('description', '(none)')}",
            f"Notify: {wf.get('notify_email', '(none)')}",
            f"Schedule: {wf.get('schedule', '(none)')}",
            f"Steps ({len(wf['steps'])}):",
        ]
        for i, step in enumerate(wf["steps"]):
            label = step.get("label", f"step_{i}")
            action = step.get("action", "?")
            detail = ""
            if action == "open" or action == "navigate":
                detail = step.get("url", "")
            elif action in ("click", "extract"):
                detail = step.get("selector", "(full page)")
            elif action == "type":
                detail = f"{step.get('selector', '')} → '{step.get('text', '')[:40]}'"
            elif action == "wait":
                detail = f"{step.get('seconds', 2)}s"
            lines.append(f"  {i}. [{action}] {label}: {detail}")

        return "\n".join(lines)

    def list(self) -> str:
        """List all saved workflows."""
        workflows = []
        for f in sorted(WORKFLOWS_DIR.glob("*.json")):
            wf = json.loads(f.read_text())
            sched = f" (cron: {wf['schedule']})" if wf.get("schedule") else ""
            workflows.append(f"  {wf['id']} — {wf['name']} ({len(wf['steps'])} steps){sched}")

        if not workflows:
            return "No workflows saved yet. Use workflow_create to define one."
        return "Saved workflows:\n" + "\n".join(workflows)

    def delete(self, wf_id: str) -> str:
        """Delete a workflow and its run history."""
        path = WORKFLOWS_DIR / f"{wf_id}.json"
        if not path.exists():
            return f"Error: Workflow '{wf_id}' not found."
        path.unlink()

        # Clean up run history
        for run_file in RUNS_DIR.glob(f"{wf_id}_*.json"):
            run_file.unlink()

        return f"Workflow '{wf_id}' deleted."

    # ── Execution ─────────────────────────────────────────────────────────

    def run(self, wf_id: str) -> str:
        """Execute a workflow: run all steps, detect changes, notify if needed."""
        path = WORKFLOWS_DIR / f"{wf_id}.json"
        if not path.exists():
            return f"Error: Workflow '{wf_id}' not found."

        workflow = json.loads(path.read_text())
        browser_fns = _import_browser()

        # Load previous run for change detection
        prev_run = self._last_run(wf_id)
        prev_extractions = prev_run.get("extractions", {}) if prev_run else {}

        results = []
        extractions = {}
        changes = []
        errors = 0

        for i, step in enumerate(workflow["steps"]):
            label = step.get("label", f"step_{i}")
            action = step.get("action", "?")

            try:
                result = _execute_step(step, browser_fns)
                results.append(
                    {
                        "step": i,
                        "label": label,
                        "action": action,
                        "result": result[:2000] if result else "",
                    }
                )

                # Track extractions for change detection
                if action == "extract":
                    clean = result.strip() if result else ""
                    extractions[label] = clean

                    prev = prev_extractions.get(label)
                    if prev is not None and prev != clean:
                        changes.append(
                            {
                                "label": label,
                                "previous": prev[:500],
                                "current": clean[:500],
                            }
                        )

            except Exception as e:
                errors += 1
                results.append(
                    {
                        "step": i,
                        "label": label,
                        "action": action,
                        "error": str(e),
                    }
                )

        # Close browser when done
        try:
            browser_fns["close"]()
        except Exception:
            pass

        # Save run record
        run_record = {
            "workflow_id": wf_id,
            "workflow_name": workflow["name"],
            "timestamp": datetime.now(UTC).isoformat(),
            "extractions": extractions,
            "changes": changes,
            "results": results,
            "step_count": len(results),
            "error_count": errors,
            "change_count": len(changes),
        }

        run_path = RUNS_DIR / f"{wf_id}_{int(time.time())}.json"
        run_path.write_text(json.dumps(run_record, indent=2))

        # Notify if changes detected
        if changes and workflow.get("notify_email"):
            self._notify(workflow, changes)

        # Format summary
        lines = [f"Workflow '{workflow['name']}' completed."]
        lines.append(f"Steps: {len(results)} | Errors: {errors} | Changes: {len(changes)}")

        if changes:
            lines.append("\n--- CHANGES DETECTED ---")
            for c in changes:
                lines.append(f"\n[{c['label']}]")
                lines.append(f"  Was: {c['previous'][:200]}")
                lines.append(f"  Now: {c['current'][:200]}")
        elif prev_run:
            lines.append("No changes since last run.")
        else:
            lines.append("First run — baseline captured.")

        if errors:
            lines.append(f"\n--- ERRORS ({errors}) ---")
            for r in results:
                if "error" in r:
                    lines.append(f"  Step {r['step']} [{r['action']}] {r['label']}: {r['error']}")

        return "\n".join(lines)

    # ── History ───────────────────────────────────────────────────────────

    def history(self, wf_id: str, limit: int = 5) -> str:
        """Show recent run history for a workflow."""
        runs = sorted(RUNS_DIR.glob(f"{wf_id}_*.json"), reverse=True)[:limit]
        if not runs:
            return f"No runs found for '{wf_id}'."

        lines = [f"Last {len(runs)} run(s) for '{wf_id}':"]
        for r in runs:
            run = json.loads(r.read_text())
            ts = run["timestamp"][:19].replace("T", " ")
            changes = run.get("change_count", 0)
            errors = run.get("error_count", 0)
            status = "CHANGES" if changes else "OK"
            if errors:
                status = "ERRORS" if not changes else "CHANGES+ERRORS"
            lines.append(f"  {ts} UTC — {status} ({changes} change(s), {errors} error(s))")

        return "\n".join(lines)

    # ── Scheduling ────────────────────────────────────────────────────────

    def schedule(self, wf_id: str, cron: str) -> str:
        """Attach a cron schedule to a workflow via MAUDE's ProactiveScheduler."""
        path = WORKFLOWS_DIR / f"{wf_id}.json"
        if not path.exists():
            return f"Error: Workflow '{wf_id}' not found."

        workflow = json.loads(path.read_text())

        # Save cron expression to workflow definition
        workflow["schedule"] = cron
        workflow["updated"] = datetime.now(UTC).isoformat()
        path.write_text(json.dumps(workflow, indent=2))

        # Register with MAUDE's scheduler
        try:
            from scheduler import get_scheduler

            sched = get_scheduler()

            # Remove any existing schedule for this workflow
            for task in sched.list_tasks():
                if hasattr(task, "name") and task.name == f"workflow:{wf_id}":
                    sched.unschedule(task.id)

            task_id = sched.schedule(
                name=f"workflow:{wf_id}",
                cron=cron,
                prompt=f"Run the browser workflow '{wf_id}' using workflow_run. Report any changes detected.",
            )
            return f"Workflow '{workflow['name']}' scheduled.\nCron: {cron}\nScheduler task: {task_id}"
        except Exception as e:
            return (
                f"Workflow cron saved but scheduler registration failed: {e}\n"
                f"You can still run it manually with workflow_run."
            )

    def unschedule(self, wf_id: str) -> str:
        """Remove the cron schedule from a workflow."""
        path = WORKFLOWS_DIR / f"{wf_id}.json"
        if not path.exists():
            return f"Error: Workflow '{wf_id}' not found."

        workflow = json.loads(path.read_text())
        workflow["schedule"] = None
        workflow["updated"] = datetime.now(UTC).isoformat()
        path.write_text(json.dumps(workflow, indent=2))

        # Remove from scheduler
        try:
            from scheduler import get_scheduler

            sched = get_scheduler()
            for task in sched.list_tasks():
                if hasattr(task, "name") and task.name == f"workflow:{wf_id}":
                    sched.unschedule(task.id)
        except Exception:
            pass

        return f"Schedule removed from '{workflow['name']}'."

    # ── Internal ──────────────────────────────────────────────────────────

    def _last_run(self, wf_id: str) -> dict | None:
        """Get the most recent run record for a workflow."""
        runs = sorted(RUNS_DIR.glob(f"{wf_id}_*.json"), reverse=True)
        if not runs:
            return None
        return json.loads(runs[0].read_text())

    def _notify(self, workflow: dict, changes: list):
        """Send email notification about detected changes."""
        try:
            from tool_registry import get_handler

            gmail_send = get_handler("gmail_send")
            if not gmail_send:
                self._log("gmail_send handler not available for notification")
                return

            subject = f"[MAUDE Workflow] Changes detected: {workflow['name']}"
            body_lines = [
                f"Workflow '{workflow['name']}' detected {len(changes)} change(s).\n",
            ]
            for c in changes:
                body_lines.append(f"--- {c['label']} ---")
                body_lines.append(f"Previous:\n{c['previous'][:500]}\n")
                body_lines.append(f"Current:\n{c['current'][:500]}\n")

            body_lines.append(f"Run at: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")

            gmail_send(
                {
                    "to": workflow["notify_email"],
                    "subject": subject,
                    "body": "\n".join(body_lines),
                }
            )
            self._log(f"Change notification sent to {workflow['notify_email']}")
        except Exception as e:
            self._log(f"Notification failed: {e}")

    def _log(self, msg: str):
        try:
            from maude_core import log

            log(f"[workflows] {msg}")
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Module Singleton
# ─────────────────────────────────────────────────────────────────────────────

_engine: WorkflowEngine | None = None


def get_engine() -> WorkflowEngine:
    global _engine
    if _engine is None:
        _engine = WorkflowEngine()
    return _engine


# ─────────────────────────────────────────────────────────────────────────────
# Public Tool Functions
# ─────────────────────────────────────────────────────────────────────────────


def workflow_create(name: str, steps: list, description: str = "", notify_email: str = None) -> str:
    """Create a repeatable browser workflow."""
    return get_engine().create(name, steps, description, notify_email)


def workflow_run(workflow_id: str) -> str:
    """Execute a saved workflow."""
    return get_engine().run(workflow_id)


def workflow_list() -> str:
    """List all saved workflows."""
    return get_engine().list()


def workflow_get(workflow_id: str) -> str:
    """View a workflow definition."""
    return get_engine().get(workflow_id)


def workflow_delete(workflow_id: str) -> str:
    """Delete a workflow."""
    return get_engine().delete(workflow_id)


def workflow_history(workflow_id: str, limit: int = 5) -> str:
    """View run history for a workflow."""
    return get_engine().history(workflow_id, limit)


def workflow_schedule(workflow_id: str, cron: str) -> str:
    """Schedule a workflow to run on a cron expression."""
    return get_engine().schedule(workflow_id, cron)


def workflow_unschedule(workflow_id: str) -> str:
    """Remove the schedule from a workflow."""
    return get_engine().unschedule(workflow_id)


# ─────────────────────────────────────────────────────────────────────────────
# Tool Dispatcher (for prefix registration)
# ─────────────────────────────────────────────────────────────────────────────


def execute_workflow_tool(name: str, args: dict) -> str:
    """Dispatch a workflow_* tool call by name."""
    dispatch = {
        "workflow_create": lambda a: workflow_create(
            name=a.get("name", ""),
            steps=a.get("steps", []),
            description=a.get("description", ""),
            notify_email=a.get("notify_email"),
        ),
        "workflow_run": lambda a: workflow_run(a.get("workflow_id", "")),
        "workflow_list": lambda a: workflow_list(),
        "workflow_get": lambda a: workflow_get(a.get("workflow_id", "")),
        "workflow_delete": lambda a: workflow_delete(a.get("workflow_id", "")),
        "workflow_history": lambda a: workflow_history(
            a.get("workflow_id", ""),
            a.get("limit", 5),
        ),
        "workflow_schedule": lambda a: workflow_schedule(
            a.get("workflow_id", ""),
            a.get("cron", ""),
        ),
        "workflow_unschedule": lambda a: workflow_unschedule(a.get("workflow_id", "")),
    }

    handler = dispatch.get(name)
    if handler is None:
        return f"Error: Unknown workflow tool '{name}'. Available: {', '.join(sorted(dispatch.keys()))}"

    try:
        return handler(args)
    except Exception as e:
        return f"Error executing {name}: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Tool Definitions (OpenAI-compatible)
# ─────────────────────────────────────────────────────────────────────────────

WORKFLOW_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "workflow_create",
            "description": (
                "Create a repeatable browser workflow. Each step has an 'action' "
                "(open, navigate, click, type, extract, screenshot, fill_form, select, wait, close), "
                "a 'label' for identification, and action-specific parameters. "
                "Extract steps are tracked for change detection between runs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Human-readable workflow name (e.g. 'Competitor Price Monitor')",
                    },
                    "steps": {
                        "type": "array",
                        "description": (
                            "List of steps. Each step is an object with 'action', 'label', and "
                            "action-specific fields: open/navigate need 'url', click/extract need "
                            "'selector', type needs 'selector' and 'text', fill_form needs 'fields' "
                            "(object), select needs 'selector' and 'value', wait needs 'seconds'."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": [
                                        "open",
                                        "navigate",
                                        "click",
                                        "type",
                                        "extract",
                                        "screenshot",
                                        "fill_form",
                                        "select",
                                        "wait",
                                        "close",
                                    ],
                                },
                                "label": {"type": "string"},
                                "url": {"type": "string"},
                                "selector": {"type": "string"},
                                "text": {"type": "string"},
                                "seconds": {"type": "number"},
                                "value": {"type": "string"},
                                "fields": {"type": "object", "additionalProperties": {"type": "string"}},
                            },
                            "required": ["action", "label"],
                        },
                    },
                    "description": {"type": "string", "description": "What this workflow does and why"},
                    "notify_email": {
                        "type": "string",
                        "description": "Email address to notify when changes are detected (optional)",
                    },
                },
                "required": ["name", "steps"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "workflow_run",
            "description": (
                "Execute a saved browser workflow. Runs all steps sequentially, "
                "compares extract results to the previous run for change detection, "
                "and sends email notification if changes are found."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "string", "description": "The workflow ID (lowercase-hyphenated name)"}
                },
                "required": ["workflow_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "workflow_list",
            "description": "List all saved browser workflows with their step counts and schedules.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "workflow_get",
            "description": "View the full definition of a saved workflow including all steps.",
            "parameters": {
                "type": "object",
                "properties": {"workflow_id": {"type": "string", "description": "The workflow ID"}},
                "required": ["workflow_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "workflow_delete",
            "description": "Delete a saved workflow and its run history.",
            "parameters": {
                "type": "object",
                "properties": {"workflow_id": {"type": "string", "description": "The workflow ID to delete"}},
                "required": ["workflow_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "workflow_history",
            "description": "View recent run history for a workflow — timestamps, change counts, and errors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "string", "description": "The workflow ID"},
                    "limit": {"type": "integer", "description": "Number of recent runs to show (default 5)"},
                },
                "required": ["workflow_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "workflow_schedule",
            "description": (
                "Schedule a workflow to run automatically on a cron expression. "
                "Examples: '0 8 * * *' (daily 8am), '0 */4 * * *' (every 4 hours), "
                "'0 9 * * 1-5' (weekday mornings)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "string", "description": "The workflow ID to schedule"},
                    "cron": {"type": "string", "description": "Cron expression (e.g. '0 8 * * *' for daily at 8am)"},
                },
                "required": ["workflow_id", "cron"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "workflow_unschedule",
            "description": "Remove the cron schedule from a workflow. The workflow can still be run manually.",
            "parameters": {
                "type": "object",
                "properties": {"workflow_id": {"type": "string", "description": "The workflow ID to unschedule"}},
                "required": ["workflow_id"],
            },
        },
    },
]
