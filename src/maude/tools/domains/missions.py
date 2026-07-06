"""Domain-owned tool schemas."""

TOOL_NAMES = {
    "mission_brief",
    "mission_create",
    "mission_drain",
    "mission_get",
    "mission_list",
    "mission_log",
    "mission_run_next",
    "mission_schedule",
    "mission_status",
    "mission_tick",
    "mission_update",
}

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "mission_status",
            "description": "Show mission dashboard data — active missions, progress, next actions, blockers, "
            "artifacts, and schedules.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "description": "Max missions to return"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mission_create",
            "description": "Create a persistent MAUDE Mission with an objective, success criteria, steps, and "
            "optional cadence/context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "template": {
                        "type": "string",
                        "enum": [
                            "content_engine",
                            "research_lab",
                            "codebase_steward",
                            "personal_ops",
                            "startup_builder",
                        ],
                        "description": "Optional starter template. Explicit "
                        "title/objective/steps override template "
                        "defaults.",
                    },
                    "title": {"type": "string", "description": "Short mission title"},
                    "objective": {"type": "string", "description": "Concrete outcome this mission should achieve"},
                    "success_criteria": {
                        "description": "List or newline-delimited success criteria",
                        "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}],
                    },
                    "steps": {
                        "description": "List of step strings/objects or a newline-delimited checklist",
                        "anyOf": [
                            {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "object"}]}},
                            {"type": "string"},
                        ],
                    },
                    "cadence": {"type": "string", "description": "Optional recurrence or review rhythm"},
                    "context": {"description": "Optional structured context for the mission"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mission_list",
            "description": "List persistent MAUDE Missions, optionally filtered by status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "description": "Optional status filter"},
                    "limit": {"type": "integer", "description": "Max missions to return (default 20, max 100)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mission_get",
            "description": "Read a Mission JSON document by id.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string", "description": "Mission id"}},
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mission_update",
            "description": "Update Mission status, notes, or step statuses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Mission id"},
                    "status": {"type": "string", "enum": ["active", "paused", "complete", "blocked", "archived"]},
                    "steps": {
                        "type": "array",
                        "description": "Step updates, e.g. [{'id':'s1','status':'done'}]",
                        "items": {"type": "object"},
                    },
                    "notes": {"type": "string", "description": "Optional mission notes"},
                    "cadence": {"type": "string", "description": "Optional recurrence or review rhythm"},
                    "success_criteria": {
                        "description": "Replacement list or newline-delimited success criteria",
                        "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}],
                    },
                    "blockers": {
                        "description": "Replacement list or newline-delimited blockers",
                        "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}],
                    },
                    "artifacts": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mission_log",
            "description": "Append a timestamped event to a Mission and optionally attach artifact paths/URLs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Mission id"},
                    "message": {"type": "string", "description": "Log message"},
                    "kind": {"type": "string", "description": "Event type such as note, result, artifact, blocker"},
                    "artifacts": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mission_run_next",
            "description": "Run the next pending Mission step through execute_plan, then log and update the step "
            "status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Mission id"},
                    "plan": {
                        "type": "array",
                        "description": "Optional execute_plan stages list. If omitted, "
                        "uses the next step's stored plan.",
                        "items": {"type": "array"},
                    },
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mission_tick",
            "description": "Advance a Mission by one checkpoint: run the next executable step, or log that the next "
            "manual step needs a plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Mission id"},
                    "plan": {
                        "type": "array",
                        "description": "Optional execute_plan stages list for the next step.",
                        "items": {"type": "array"},
                    },
                    "retry_blocked": {
                        "type": "boolean",
                        "description": "Retry a blocked executable step. Defaults to true.",
                    },
                    "auto_complete": {
                        "type": "boolean",
                        "description": "Mark the mission complete when all steps are done.",
                    },
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mission_drain",
            "description": "Advance a Mission through consecutive executable stored-plan steps until it completes "
            "or reaches a manual/blocking step.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Mission id"},
                    "retry_blocked": {
                        "type": "boolean",
                        "description": "Retry blocked executable steps. Defaults to true.",
                    },
                    "auto_complete": {
                        "type": "boolean",
                        "description": "Mark the mission complete when all steps are done. Defaults to true.",
                    },
                    "max_steps": {
                        "type": "integer",
                        "description": "Safety cap for steps to run in one drain. Defaults to 10, max 25.",
                    },
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mission_schedule",
            "description": "Schedule, inspect, run, disable, enable, or remove recurring scheduler ticks for a "
            "Mission.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Mission id"},
                    "action": {
                        "type": "string",
                        "enum": ["add", "remove", "enable", "disable", "run", "status"],
                        "description": "Schedule action. Defaults to add.",
                    },
                    "cron": {
                        "type": "string",
                        "description": "Cron expression or shortcut such as @daily, "
                        "@weekly, @morning. Defaults from mission "
                        "cadence.",
                    },
                    "name": {"type": "string", "description": "Optional scheduler task name"},
                    "prompt": {
                        "type": "string",
                        "description": "Optional scheduled prompt. Defaults to running "
                        "mission_tick then mission_brief.",
                    },
                    "task_id": {
                        "type": "string",
                        "description": "Optional scheduler task id. Defaults to the task saved on the mission.",
                    },
                    "replace": {"type": "boolean", "description": "Replace an existing mission schedule when adding."},
                    "auto_complete": {
                        "type": "boolean",
                        "description": "Ask scheduled ticks to complete the "
                        "mission once all steps are done. "
                        "Defaults to true.",
                    },
                    "channel": {"type": "string", "description": "Scheduler output channel. Defaults to cli."},
                    "channel_id": {
                        "type": "string",
                        "description": "Scheduler output channel id. Defaults to default.",
                    },
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mission_brief",
            "description": "Produce a compact Mission brief with objective, progress, next action, recent logs, and "
            "artifacts.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string", "description": "Mission id"}},
                "required": ["id"],
            },
        },
    },
]


def schemas() -> list[dict]:
    return list(SCHEMAS)
