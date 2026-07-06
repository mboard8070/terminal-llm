"""
Mission tools — persistent project-level state for autonomous MAUDE work.

A mission is intentionally small and inspectable: one JSON file with an
objective, steps, status, logs, and artifacts. Higher-level agents can use
these tools as durable state while they research, build, render, publish, or
iterate across turns and channels.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from maude.config import runtime_paths
from tool_registry import register_tool

_DEFAULT_MISSIONS_DIR = runtime_paths().missions_dir
_VALID_STATUSES = {"active", "paused", "complete", "blocked", "archived"}
_VALID_STEP_STATUSES = {"pending", "in_progress", "done", "blocked", "skipped"}
_MISSION_TEMPLATES: dict[str, dict[str, Any]] = {
    "content_engine": {
        "title": "Content Engine",
        "objective": "Research, produce, publish, analyze, and improve recurring useful content.",
        "cadence": "daily or weekly",
        "success_criteria": [
            "Topic selected from current signals or mission memory",
            "Script and visual plan created",
            "Media rendered and reviewed",
            "Published artifact linked in the mission",
            "Performance notes captured for the next run",
        ],
        "steps": [
            {
                "title": "Choose topic and angle",
                "details": "Select the next topic and save the decision as a mission artifact.",
                "plan": [
                    [
                        {
                            "name": "write_file",
                            "args": {
                                "path": "$MISSION_ARTIFACT_DIR/s1-topic-angle.md",
                                "content": "# Topic and Angle\n\nMission: $MISSION_ID\n\n## Selected topic\nTBD from current signals, mission memory, and recent shipped work.\n\n## Angle\nTBD.\n\n## Target viewer takeaway\nTBD.\n",
                            },
                        }
                    ],
                    [
                        {
                            "name": "mission_log",
                            "args": {
                                "id": "$MISSION_ID",
                                "kind": "artifact",
                                "message": "Selected topic and angle placeholder for the content-engine cycle.",
                                "artifacts": ["$MISSION_ARTIFACT_DIR/s1-topic-angle.md"],
                            },
                        }
                    ],
                ],
            },
            {
                "title": "Draft script and asset list",
                "details": "Draft a script, storyboard, and asset checklist for the selected topic.",
                "plan": [
                    [
                        {
                            "name": "write_file",
                            "args": {
                                "path": "$MISSION_ARTIFACT_DIR/s2-script-assets.md",
                                "content": "# Script and Asset List\n\nMission: $MISSION_ID\n\n## Script\nTBD.\n\n## Shot list\n1. Hook\n2. Demonstration\n3. Payoff\n4. CTA\n\n## Assets needed\n- Source footage or screenshots\n- Generated or captured visual assets\n- Publish copy\n",
                            },
                        }
                    ],
                    [
                        {
                            "name": "mission_log",
                            "args": {
                                "id": "$MISSION_ID",
                                "kind": "artifact",
                                "message": "Drafted script and asset-list placeholder for the content-engine cycle.",
                                "artifacts": ["$MISSION_ARTIFACT_DIR/s2-script-assets.md"],
                            },
                        }
                    ],
                ],
            },
            {
                "title": "Generate or collect visuals",
                "details": "Create or gather the visual sources needed for the content artifact.",
                "plan": [
                    [
                        {
                            "name": "write_file",
                            "args": {
                                "path": "$MISSION_ARTIFACT_DIR/s3-visuals.md",
                                "content": "# Visual Sources\n\nMission: $MISSION_ID\n\n## Collected/generated assets\nTBD.\n\n## Notes\nConfirm source footage duration, readability, and permissions before render.\n",
                            },
                        }
                    ],
                    [
                        {
                            "name": "mission_log",
                            "args": {
                                "id": "$MISSION_ID",
                                "kind": "artifact",
                                "message": "Created visual-source checklist for the content-engine cycle.",
                                "artifacts": ["$MISSION_ARTIFACT_DIR/s3-visuals.md"],
                            },
                        }
                    ],
                ],
            },
            {
                "title": "Render final media",
                "details": "Render or stage the media, then record the output path and validation notes.",
                "plan": [
                    [
                        {
                            "name": "write_file",
                            "args": {
                                "path": "$MISSION_ARTIFACT_DIR/s4-render-review.md",
                                "content": "# Render Review\n\nMission: $MISSION_ID\n\n## Render path\nTBD.\n\n## Checks\n- Duration and frame count verified\n- Motion previewed across the timeline\n- Text readable on mobile\n- No raw URLs burned into frame\n",
                            },
                        }
                    ],
                    [
                        {
                            "name": "mission_log",
                            "args": {
                                "id": "$MISSION_ID",
                                "kind": "artifact",
                                "message": "Recorded render-review checklist for the content-engine cycle.",
                                "artifacts": ["$MISSION_ARTIFACT_DIR/s4-render-review.md"],
                            },
                        }
                    ],
                ],
            },
            {
                "title": "Publish and log URL",
                "details": "Create a publish plan, run required pre-publish checks, and log URLs only after publish succeeds.",
                "plan": [
                    [
                        {
                            "name": "write_file",
                            "args": {
                                "path": "$MISSION_ARTIFACT_DIR/s5-publish-plan.md",
                                "content": "# Publish Plan\n\nMission: $MISSION_ID\n\n## Title\nTBD.\n\n## Caption/description\nTBD.\n\n## Clean link placement\nUse description/caption only unless explicitly requested in-frame.\n\n## Tags/hashtags\nTBD.\n\n## Platform/privacy\nTBD.\n",
                            },
                        }
                    ],
                    [
                        {
                            "name": "mission_log",
                            "args": {
                                "id": "$MISSION_ID",
                                "kind": "artifact",
                                "message": "Created publish-plan placeholder for the content-engine cycle.",
                                "artifacts": ["$MISSION_ARTIFACT_DIR/s5-publish-plan.md"],
                            },
                        }
                    ],
                ],
            },
            {
                "title": "Review performance and update next action",
                "details": "Capture feedback or performance notes and set the next cycle's topic candidate.",
                "plan": [
                    [
                        {
                            "name": "write_file",
                            "args": {
                                "path": "$MISSION_ARTIFACT_DIR/s6-review-next.md",
                                "content": "# Review and Next Action\n\nMission: $MISSION_ID\n\n## Performance / feedback\nTBD.\n\n## Lessons\nTBD.\n\n## Next topic candidate\nTBD.\n",
                            },
                        }
                    ],
                    [
                        {
                            "name": "mission_log",
                            "args": {
                                "id": "$MISSION_ID",
                                "kind": "review",
                                "message": "Captured review placeholder and next-action slot for the content-engine cycle.",
                                "artifacts": ["$MISSION_ARTIFACT_DIR/s6-review-next.md"],
                            },
                        }
                    ],
                ],
            },
        ],
    },
    "research_lab": {
        "title": "Research Lab",
        "objective": "Monitor sources, synthesize findings, run experiments, and produce research reports.",
        "cadence": "weekly",
        "success_criteria": [
            "Recent sources reviewed",
            "Key findings summarized with citations",
            "Experiment or analysis run when useful",
            "Report artifact saved and linked",
        ],
        "steps": [
            "Scan current literature and news",
            "Extract open questions and hypotheses",
            "Run experiment or analysis",
            "Write findings report",
            "Log follow-up questions",
        ],
    },
    "codebase_steward": {
        "title": "Codebase Steward",
        "objective": "Continuously improve a codebase through fixes, tests, docs, and PR-ready changes.",
        "cadence": "weekly",
        "success_criteria": [
            "Repo status reviewed",
            "High-value issue selected",
            "Focused patch implemented",
            "Relevant tests pass or gaps are logged",
            "Change summary recorded",
        ],
        "steps": [
            "Inspect repo status and recent TODOs",
            "Select next code improvement",
            "Implement focused change",
            "Run relevant tests",
            "Record summary and follow-up",
        ],
    },
    "personal_ops": {
        "title": "Personal Ops",
        "objective": "Manage recurring personal admin, files, reminders, planning, and purchases.",
        "cadence": "daily",
        "success_criteria": [
            "Open loops captured",
            "Time-sensitive items surfaced",
            "Files or records organized",
            "Completed actions logged",
        ],
        "steps": [
            "Review inboxes and open loops",
            "Prioritize time-sensitive actions",
            "Execute approved admin tasks",
            "Organize related artifacts",
            "Log completions and blockers",
        ],
    },
    "startup_builder": {
        "title": "Startup Builder",
        "objective": "Create prototypes, demos, landing pages, outreach, and investor-ready updates.",
        "cadence": "weekly",
        "success_criteria": [
            "Current bet or experiment selected",
            "Prototype or artifact shipped",
            "Feedback or metric captured",
            "Next business action identified",
        ],
        "steps": [
            "Choose highest-leverage startup task",
            "Build or update artifact",
            "Prepare demo, copy, or outreach",
            "Collect feedback or metric",
            "Set next business action",
        ],
    },
}


def _missions_dir() -> Path:
    path = Path(os.environ.get("MAUDE_MISSIONS_DIR") or _DEFAULT_MISSIONS_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:48] or "mission"


def _mission_path(mission_id: str) -> Path:
    mission_id = Path(str(mission_id)).name
    return _missions_dir() / f"{mission_id}.json"


def _read_mission(mission_id: str) -> dict[str, Any]:
    path = _mission_path(mission_id)
    if not path.exists():
        raise FileNotFoundError(f"Mission '{mission_id}' not found")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_mission(mission: dict[str, Any]) -> None:
    path = _mission_path(mission["id"])
    _refresh_checkpoint(mission)
    payload = json.dumps(mission, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _normalize_steps(raw_steps: Any) -> list[dict[str, Any]]:
    if raw_steps is None:
        return []
    if isinstance(raw_steps, str):
        raw_steps = [line.strip("- ").strip() for line in raw_steps.splitlines() if line.strip()]
    if not isinstance(raw_steps, list):
        raise ValueError("steps must be a list or newline-delimited string")

    steps = []
    for index, item in enumerate(raw_steps, start=1):
        if isinstance(item, str):
            title = item.strip()
            status = "pending"
            details = ""
            plan = None
        elif isinstance(item, dict):
            title = str(item.get("title") or item.get("task") or "").strip()
            status = str(item.get("status") or "pending")
            details = str(item.get("details") or item.get("description") or "")
            plan = item.get("plan")
        else:
            raise ValueError("each step must be a string or object")
        if not title:
            raise ValueError("mission steps cannot have empty titles")
        if status not in _VALID_STEP_STATUSES:
            raise ValueError(f"invalid step status '{status}'")
        step = {"id": f"s{index}", "title": title, "status": status, "details": details}
        if plan is not None:
            if not isinstance(plan, list):
                raise ValueError("step plan must be an execute_plan stages list")
            step["plan"] = plan
        steps.append(step)
    return steps


def _normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [line.strip("- ").strip() for line in value.splitlines() if line.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raise ValueError("value must be a list or newline-delimited string")


def _next_step(mission: dict[str, Any]) -> dict[str, Any] | None:
    return next((step for step in mission.get("steps", []) if step.get("status") != "done"), None)


def _checkpoint(mission: dict[str, Any]) -> dict[str, Any]:
    steps = mission.get("steps", [])
    done = sum(1 for step in steps if step.get("status") == "done")
    next_step = _next_step(mission)
    return {
        "objective": mission.get("objective", ""),
        "status": mission.get("status", ""),
        "progress": {"done": done, "total": len(steps)},
        "next_action": next_step.get("title", "") if next_step else "",
        "recent_logs": mission.get("logs", [])[-5:],
        "artifacts": mission.get("artifacts", [])[-10:],
        "blockers": mission.get("blockers", []),
        "cadence": mission.get("cadence", ""),
        "success_criteria": mission.get("success_criteria", []),
    }


def _refresh_checkpoint(mission: dict[str, Any]) -> None:
    mission["checkpoint"] = _checkpoint(mission)


def _replace_token(value: Any, token: str, replacement: str) -> Any:
    if isinstance(value, str):
        return value.replace(token, replacement)
    if isinstance(value, list):
        return [_replace_token(item, token, replacement) for item in value]
    if isinstance(value, dict):
        return {key: _replace_token(item, token, replacement) for key, item in value.items()}
    return value


def _artifact_dir(mission_id: str) -> str:
    path = _missions_dir() / "artifacts" / Path(str(mission_id)).name
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _prepare_plan(plan: list[Any], mission_id: str) -> list[Any]:
    plan = _replace_token(plan, "$MISSION_ID", mission_id)
    return _replace_token(plan, "$MISSION_ARTIFACT_DIR", _artifact_dir(mission_id))


def _is_recurring_mission(mission: dict[str, Any]) -> bool:
    if mission.get("recurring") is True:
        return True
    schedule = mission.get("schedule") or {}
    cadence = str(mission.get("cadence") or "").strip()
    return bool(schedule.get("task_id") and cadence and mission.get("status") == "complete")


def _reset_for_next_cycle(mission: dict[str, Any], reason: str) -> None:
    now = _now()
    for step in mission.get("steps", []):
        if step.get("status") == "done":
            step["status"] = "pending"
    mission["status"] = "active"
    mission["updated_at"] = now
    mission.setdefault("logs", []).append(
        {
            "time": now,
            "kind": "cycle",
            "message": reason,
            "artifacts": [],
        }
    )


def _cadence_to_cron(cadence: str) -> str:
    normalized = cadence.lower().strip()
    if normalized.startswith("@"):
        return normalized
    if not normalized:
        return ""
    if "hour" in normalized:
        return "@hourly"
    if "morning" in normalized:
        return "@morning"
    if "evening" in normalized:
        return "@evening"
    if "workday" in normalized or "weekday" in normalized:
        return "@workdays"
    if "daily" in normalized or "day" in normalized:
        return "@daily"
    if "weekly" in normalized or "week" in normalized:
        return "@weekly"
    if "monthly" in normalized or "month" in normalized:
        return "@monthly"
    return ""


def _default_schedule_prompt(mission_id: str, auto_complete: bool) -> str:
    auto_complete_text = "true" if auto_complete else "false"
    return (
        "Advance this MAUDE Mission through executable stored-plan steps. "
        f"Mission id: {mission_id}. auto_complete={auto_complete_text}. "
        "The scheduler runs mission metadata directly; this prompt is only fallback context."
    )


def _scheduled_task_id(schedule_result: str) -> str:
    match = re.search(r"\(ID:\s*([^)]+)\)", schedule_result)
    return match.group(1).strip() if match else ""


def _template_names() -> str:
    return ", ".join(sorted(_MISSION_TEMPLATES))


def _format_mission_summary(mission: dict[str, Any]) -> str:
    _refresh_checkpoint(mission)
    steps = mission.get("steps", [])
    done = sum(1 for step in steps if step.get("status") == "done")
    total = len(steps)
    return (
        f"{mission['id']} | {mission['status']} | {mission['title']}\n"
        f"Objective: {mission['objective']}\n"
        f"Progress: {done}/{total} steps complete | Updated: {mission['updated_at']}"
    )


@register_tool("mission_create")
def _dispatch_mission_create(args: dict[str, Any]) -> str:
    template_name = str(args.get("template") or "").strip()
    if template_name:
        if template_name not in _MISSION_TEMPLATES:
            return f"Error: unknown mission template '{template_name}'. Available templates: {_template_names()}"
        template = _MISSION_TEMPLATES[template_name]
    else:
        template = {}

    title = str(args.get("title") or template.get("title") or "").strip()
    objective = str(args.get("objective") or template.get("objective") or "").strip()
    if not title:
        return "Error: title is required"
    if not objective:
        return "Error: objective is required"

    try:
        steps = _normalize_steps(args.get("steps", template.get("steps")))
        success_criteria = _normalize_list(args.get("success_criteria", template.get("success_criteria")))
    except ValueError as e:
        return f"Error: {e}"

    mission_id = f"{_slugify(title)}-{uuid.uuid4().hex[:8]}"
    now = _now()
    mission = {
        "id": mission_id,
        "title": title,
        "objective": objective,
        "status": "active",
        "template": template_name,
        "cadence": str(args.get("cadence") or template.get("cadence") or "").strip(),
        "success_criteria": success_criteria,
        "context": args.get("context") or {},
        "steps": steps,
        "logs": [],
        "artifacts": [],
        "blockers": _normalize_list(args.get("blockers")),
        "schedule": {},
        "created_at": now,
        "updated_at": now,
    }
    _write_mission(mission)
    return "Mission created.\n" + _format_mission_summary(mission)


@register_tool("mission_list")
def _dispatch_mission_list(args: dict[str, Any]) -> str:
    status = str(args.get("status") or "").strip()
    limit = int(args.get("limit") or 20)
    limit = max(1, min(limit, 100))

    missions = []
    for path in _missions_dir().glob("*.json"):
        try:
            mission = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if status and mission.get("status") != status:
            continue
        missions.append(mission)

    missions.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
    if not missions:
        return "No missions found."
    return "\n\n".join(_format_mission_summary(mission) for mission in missions[:limit])


@register_tool("mission_get")
def _dispatch_mission_get(args: dict[str, Any]) -> str:
    mission_id = str(args.get("id") or args.get("mission_id") or "").strip()
    if not mission_id:
        return "Error: id is required"
    try:
        mission = _read_mission(mission_id)
    except FileNotFoundError as e:
        return f"Error: {e}"
    return json.dumps(mission, indent=2, sort_keys=True)


@register_tool("mission_update")
def _dispatch_mission_update(args: dict[str, Any]) -> str:
    mission_id = str(args.get("id") or args.get("mission_id") or "").strip()
    if not mission_id:
        return "Error: id is required"
    try:
        mission = _read_mission(mission_id)
    except FileNotFoundError as e:
        return f"Error: {e}"

    status = args.get("status")
    if status is not None:
        status = str(status)
        if status not in _VALID_STATUSES:
            return f"Error: invalid mission status '{status}'"
        mission["status"] = status

    step_updates = args.get("steps") or []
    if isinstance(step_updates, dict):
        step_updates = [step_updates]
    if not isinstance(step_updates, list):
        return "Error: steps must be a list of updates"

    by_id = {step["id"]: step for step in mission.get("steps", [])}
    for update in step_updates:
        if not isinstance(update, dict):
            return "Error: each step update must be an object"
        step_id = str(update.get("id") or "").strip()
        if step_id not in by_id:
            return f"Error: unknown step id '{step_id}'"
        step = by_id[step_id]
        if "status" in update:
            step_status = str(update["status"])
            if step_status not in _VALID_STEP_STATUSES:
                return f"Error: invalid step status '{step_status}'"
            step["status"] = step_status
        if "title" in update:
            step["title"] = str(update["title"]).strip()
        if "details" in update:
            step["details"] = str(update["details"])
        if "plan" in update:
            plan = update["plan"]
            if plan is not None and not isinstance(plan, list):
                return "Error: step plan must be an execute_plan stages list"
            if plan is None:
                step.pop("plan", None)
            else:
                step["plan"] = plan

    if "notes" in args:
        mission["notes"] = str(args["notes"])
    if "cadence" in args:
        mission["cadence"] = str(args["cadence"])
    if "success_criteria" in args:
        try:
            mission["success_criteria"] = _normalize_list(args["success_criteria"])
        except ValueError as e:
            return f"Error: {e}"
    if "blockers" in args:
        try:
            mission["blockers"] = _normalize_list(args["blockers"])
        except ValueError as e:
            return f"Error: {e}"
    if "artifacts" in args:
        try:
            for artifact in _normalize_list(args["artifacts"]):
                if artifact not in mission.setdefault("artifacts", []):
                    mission["artifacts"].append(artifact)
        except ValueError as e:
            return f"Error: {e}"

    mission["updated_at"] = _now()
    _write_mission(mission)
    return "Mission updated.\n" + _format_mission_summary(mission)


@register_tool("mission_log")
def _dispatch_mission_log(args: dict[str, Any]) -> str:
    mission_id = str(args.get("id") or args.get("mission_id") or "").strip()
    message = str(args.get("message") or "").strip()
    if not mission_id:
        return "Error: id is required"
    if not message:
        return "Error: message is required"
    try:
        mission = _read_mission(mission_id)
    except FileNotFoundError as e:
        return f"Error: {e}"

    artifacts = _normalize_list(args.get("artifacts"))
    entry = {
        "time": _now(),
        "kind": str(args.get("kind") or "note"),
        "message": message,
        "artifacts": artifacts,
    }
    mission.setdefault("logs", []).append(entry)
    for artifact in artifacts:
        if artifact not in mission.setdefault("artifacts", []):
            mission["artifacts"].append(artifact)
    if entry["kind"] == "blocker" and message not in mission.setdefault("blockers", []):
        mission["blockers"].append(message)
    mission["updated_at"] = entry["time"]
    _write_mission(mission)
    return f"Logged mission event for {mission_id}."


@register_tool("mission_run_next")
def _dispatch_mission_run_next(args: dict[str, Any]) -> str:
    mission_id = str(args.get("id") or args.get("mission_id") or "").strip()
    if not mission_id:
        return "Error: id is required"
    try:
        mission = _read_mission(mission_id)
    except FileNotFoundError as e:
        return f"Error: {e}"

    if mission.get("status") not in {"active", "blocked"}:
        return f"Error: mission status is '{mission.get('status')}', not active"

    steps = mission.get("steps", [])
    step = next((item for item in steps if item.get("status") in {"pending", "in_progress", "blocked"}), None)
    if not step:
        return "No runnable mission steps found. Review success criteria and close or extend the mission."

    plan = args.get("plan", step.get("plan"))
    if not isinstance(plan, list) or not plan:
        return (
            f"Error: next step '{step.get('id')}' has no executable plan. "
            "Attach an execute_plan stages list to the step with mission_update, or pass plan to mission_run_next."
        )

    now = _now()
    step["status"] = "in_progress"
    mission["status"] = "active"
    mission["updated_at"] = now
    mission.setdefault("logs", []).append(
        {
            "time": now,
            "kind": "runner",
            "message": f"Started {step['id']}: {step['title']}",
            "artifacts": [],
        }
    )
    _write_mission(mission)

    from maude.tools.handlers.plan import execute_plan

    plan = _prepare_plan(plan, mission_id)
    result = execute_plan(plan)

    try:
        mission = _read_mission(mission_id)
        step = next((item for item in mission.get("steps", []) if item.get("id") == step["id"]), step)
    except FileNotFoundError:
        pass

    failed = result.startswith("Error:") or "\nError:" in result or ": ERROR" in result
    finished_at = _now()
    step["status"] = "blocked" if failed else "done"
    step["last_run_at"] = finished_at
    step["last_result"] = result[:4000]
    if failed:
        blocker = f"{step['id']}: execution failed"
        if blocker not in mission.setdefault("blockers", []):
            mission["blockers"].append(blocker)
    mission["updated_at"] = finished_at
    mission.setdefault("logs", []).append(
        {
            "time": finished_at,
            "kind": "blocker" if failed else "result",
            "message": f"Ran {step['id']}: {step['title']}. Status: {step['status']}.",
            "artifacts": [],
        }
    )
    _write_mission(mission)

    return (
        f"Mission step {step['id']} {step['status']}.\n"
        f"{_format_mission_summary(mission)}\n\n"
        f"Execution result:\n{result}"
    )


@register_tool("mission_tick")
def _dispatch_mission_tick(args: dict[str, Any]) -> str:
    mission_id = str(args.get("id") or args.get("mission_id") or "").strip()
    if not mission_id:
        return "Error: id is required"
    try:
        mission = _read_mission(mission_id)
    except FileNotFoundError as e:
        return f"Error: {e}"

    status = mission.get("status")
    if status == "complete" and _is_recurring_mission(mission):
        _reset_for_next_cycle(mission, "Recurring mission reset for the next scheduled cycle.")
        _write_mission(mission)
        status = mission.get("status")
    if status in {"paused", "complete", "archived"}:
        return f"Mission tick skipped: mission status is '{status}'.\n\n{_dispatch_mission_brief({'id': mission_id})}"
    if status == "blocked" and not args.get("retry_blocked", True):
        return f"Mission tick skipped: mission is blocked.\n\n{_dispatch_mission_brief({'id': mission_id})}"

    step = next(
        (item for item in mission.get("steps", []) if item.get("status") in {"pending", "in_progress", "blocked"}), None
    )
    if not step:
        if args.get("auto_complete"):
            if _is_recurring_mission(mission):
                _reset_for_next_cycle(mission, "Recurring mission reset because all steps were already done.")
                _write_mission(mission)
                return _dispatch_mission_tick(args)
            mission["status"] = "complete"
            mission["updated_at"] = _now()
            mission.setdefault("logs", []).append(
                {
                    "time": mission["updated_at"],
                    "kind": "checkpoint",
                    "message": "Mission auto-completed because all steps are done.",
                    "artifacts": [],
                }
            )
            _write_mission(mission)
            return "Mission tick completed the mission.\n\n" + _dispatch_mission_brief({"id": mission_id})
        return "Mission tick found no runnable steps. Review success criteria and close or extend the mission."

    plan = args.get("plan", step.get("plan"))
    if not isinstance(plan, list) or not plan:
        now = _now()
        mission["status"] = "active"
        mission["updated_at"] = now
        mission.setdefault("logs", []).append(
            {
                "time": now,
                "kind": "checkpoint",
                "message": f"Tick checkpoint: next step '{step['id']}' is manual or needs a plan.",
                "artifacts": [],
            }
        )
        _write_mission(mission)
        return (
            f"Mission tick checkpoint: next step '{step['id']}' needs a plan before it can run.\n\n"
            f"{_dispatch_mission_brief({'id': mission_id})}"
        )

    run_args: dict[str, Any] = {"id": mission_id}
    if "plan" in args:
        run_args["plan"] = plan
    result = _dispatch_mission_run_next(run_args)

    if args.get("auto_complete"):
        try:
            mission = _read_mission(mission_id)
        except FileNotFoundError:
            return result
        if all(step.get("status") == "done" for step in mission.get("steps", [])):
            mission["status"] = "complete"
            mission["updated_at"] = _now()
            mission.setdefault("logs", []).append(
                {
                    "time": mission["updated_at"],
                    "kind": "checkpoint",
                    "message": "Mission auto-completed because all steps are done.",
                    "artifacts": [],
                }
            )
            _write_mission(mission)
            result += "\n\nMission auto-completed."
    return result


@register_tool("mission_drain")
def _dispatch_mission_drain(args: dict[str, Any]) -> str:
    mission_id = str(args.get("id") or args.get("mission_id") or "").strip()
    if not mission_id:
        return "Error: id is required"

    try:
        max_steps = int(args.get("max_steps") or 10)
    except (TypeError, ValueError):
        return "Error: max_steps must be an integer"
    max_steps = max(1, min(max_steps, 25))

    auto_complete = bool(args.get("auto_complete", True))
    retry_blocked = bool(args.get("retry_blocked", True))
    results = []

    for _ in range(max_steps):
        result = _dispatch_mission_tick(
            {
                "id": mission_id,
                "auto_complete": auto_complete,
                "retry_blocked": retry_blocked,
            }
        )
        results.append(result)

        if result.startswith("Error:"):
            break

        try:
            mission = _read_mission(mission_id)
        except FileNotFoundError:
            break

        status = mission.get("status")
        if status in {"paused", "complete", "archived"}:
            break
        if status == "blocked" and not retry_blocked:
            break

        step = next(
            (item for item in mission.get("steps", []) if item.get("status") in {"pending", "in_progress", "blocked"}),
            None,
        )
        if not step:
            break
        if not isinstance(step.get("plan"), list) or not step.get("plan"):
            checkpoint = _dispatch_mission_tick(
                {
                    "id": mission_id,
                    "auto_complete": auto_complete,
                    "retry_blocked": retry_blocked,
                }
            )
            results.append(checkpoint)
            break
        if step.get("status") == "blocked":
            break

    else:
        results.append(f"Mission drain stopped after max_steps={max_steps}.")

    try:
        brief = _dispatch_mission_brief({"id": mission_id})
    except Exception:
        brief = ""

    header = (
        f"Mission drain ran {len([item for item in results if item.startswith('Mission step ')])} executable step(s)."
    )
    return "\n\n---\n\n".join([header, *results, brief] if brief else [header, *results])


@register_tool("mission_schedule")
def _dispatch_mission_schedule(args: dict[str, Any]) -> str:
    mission_id = str(args.get("id") or args.get("mission_id") or "").strip()
    action = str(args.get("action") or "add").strip()
    if not mission_id:
        return "Error: id is required"
    if action not in {"add", "remove", "enable", "disable", "run", "status"}:
        return "Error: action must be one of add, remove, enable, disable, run, status"

    try:
        mission = _read_mission(mission_id)
    except FileNotFoundError as e:
        return f"Error: {e}"

    schedule = mission.get("schedule") or {}
    task_id = str(args.get("task_id") or schedule.get("task_id") or "").strip()

    try:
        from scheduler import get_scheduler

        scheduler = get_scheduler()
    except Exception as e:
        return f"Error with scheduler: {e}"

    if action == "status":
        task = scheduler.tasks.get(task_id) if task_id else None
        payload = {
            "mission_id": mission_id,
            "mission_schedule": schedule,
            "scheduler_task": task.__dict__ if task else None,
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    if action == "add":
        if schedule.get("task_id") and not args.get("replace"):
            return (
                f"Error: mission already has scheduled task {schedule['task_id']}. "
                "Pass replace=true to create a new schedule."
            )
        if schedule.get("task_id") and args.get("replace"):
            scheduler.unschedule(str(schedule["task_id"]))

        cron = str(args.get("cron") or _cadence_to_cron(str(mission.get("cadence") or ""))).strip()
        if not cron:
            return "Error: cron is required when mission cadence cannot be converted to a schedule"
        name = str(args.get("name") or f"Mission tick: {mission['title']}").strip()
        auto_complete = bool(args.get("auto_complete", True))
        prompt = str(args.get("prompt") or _default_schedule_prompt(mission_id, auto_complete)).strip()
        channel = str(args.get("channel") or "cli").strip()
        channel_id = str(args.get("channel_id") or "default").strip()

        result = scheduler.schedule(
            name=name,
            cron=cron,
            prompt=prompt,
            channel=channel,
            channel_id=channel_id,
            task_type="mission",
            mission_id=mission_id,
            mission_auto_complete=auto_complete,
            mission_max_steps=int(args.get("max_steps") or 10),
        )
        if result.startswith("Invalid cron expression"):
            return result
        task_id = _scheduled_task_id(result)
        mission["schedule"] = {
            "task_id": task_id,
            "name": name,
            "cron": cron,
            "prompt": prompt,
            "channel": channel,
            "channel_id": channel_id,
            "auto_complete": auto_complete,
            "enabled": True,
            "created_at": _now(),
        }
        mission["updated_at"] = mission["schedule"]["created_at"]
        mission.setdefault("logs", []).append(
            {
                "time": mission["updated_at"],
                "kind": "schedule",
                "message": f"Scheduled recurring mission tick as task {task_id or 'unknown'}.",
                "artifacts": [],
            }
        )
        _write_mission(mission)
        return f"{result}\n\nMission schedule saved for {mission_id}."

    if not task_id:
        return "Error: no scheduled task id found for mission"

    if action == "remove":
        result = scheduler.unschedule(task_id)
        mission["schedule"] = {}
    elif action == "enable":
        result = scheduler.enable_task(task_id)
        mission.setdefault("schedule", {})["enabled"] = True
    elif action == "disable":
        result = scheduler.disable_task(task_id)
        mission.setdefault("schedule", {})["enabled"] = False
    else:
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                _task = asyncio.create_task(scheduler.run_task_by_id(task_id))  # noqa: RUF006
                result = f"Running scheduled mission task {task_id}..."
            else:
                result = loop.run_until_complete(scheduler.run_task_by_id(task_id))
        except Exception as e:
            result = f"Error running scheduled mission task {task_id}: {e}"

    mission["updated_at"] = _now()
    mission.setdefault("logs", []).append(
        {
            "time": mission["updated_at"],
            "kind": "schedule",
            "message": f"Mission schedule action '{action}' for task {task_id}: {result}",
            "artifacts": [],
        }
    )
    _write_mission(mission)
    return f"{result}\n\nMission schedule updated for {mission_id}."


@register_tool("mission_brief")
def _dispatch_mission_brief(args: dict[str, Any]) -> str:
    mission_id = str(args.get("id") or args.get("mission_id") or "").strip()
    if not mission_id:
        return "Error: id is required"
    try:
        mission = _read_mission(mission_id)
    except FileNotFoundError as e:
        return f"Error: {e}"

    next_step = _next_step(mission)
    recent_logs = mission.get("logs", [])[-5:]
    lines = [
        _format_mission_summary(mission),
        "",
        "Next action:",
        next_step["title"]
        if next_step
        else "No pending steps. Review success criteria and close or extend the mission.",
    ]
    if mission.get("blockers"):
        lines.extend(["", "Blockers:", *[f"- {item}" for item in mission["blockers"][-10:]]])
    if mission.get("success_criteria"):
        lines.extend(["", "Success criteria:", *[f"- {item}" for item in mission["success_criteria"]]])
    if recent_logs:
        lines.extend(["", "Recent log:"])
        lines.extend(f"- {entry['time']} [{entry['kind']}] {entry['message']}" for entry in recent_logs)
    if mission.get("schedule"):
        schedule = mission["schedule"]
        lines.extend(
            [
                "",
                "Schedule:",
                f"- {schedule.get('task_id', 'unknown')} | {schedule.get('cron', '')} | "
                f"{'enabled' if schedule.get('enabled', True) else 'disabled'}",
            ]
        )
    if mission.get("artifacts"):
        lines.extend(["", "Artifacts:", *[f"- {artifact}" for artifact in mission["artifacts"][-10:]]])
    return "\n".join(lines)
