"""
MAUDE Collaboration Tools — LLM-callable tool definitions for the collab system.

Lets MAUDE manage presence, projects, tasks, and activity through natural language.
"""

import json
import time
from collab import get_hub

# ── Tool definitions (OpenAI function-calling format) ────────────

COLLAB_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "mesh_status",
            "description": "Show who's online across all devices and what they're doing. Shows presence, recent activity, and task status across the MAUDE mesh network.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "dispatch_task",
            "description": "Dispatch a task to a specific device or platform on the mesh network. Use capability 'SHELL' for shell commands or 'LLM' for AI tasks. Target by client_id (e.g. 'MacBook-Pro-matt') or platform (e.g. 'windows', 'macos'). Client-targeted SHELL tasks are picked up by the client within ~10 seconds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "The task prompt or command to execute"},
                    "target": {"type": "string", "description": "Target hostname, client_id, or platform name. Leave empty for local execution."},
                    "capability": {"type": "string", "description": "Task type: 'SHELL' for commands, 'LLM' for AI tasks", "enum": ["SHELL", "LLM"]},
                    "project_id": {"type": "string", "description": "Optional project ID to associate with this task"},
                    "target_client_id": {"type": "string", "description": "Target a specific client by its client_id (from mesh_status)"},
                    "target_platform": {"type": "string", "description": "Target a platform: 'windows', 'macos', or 'linux'"}
                },
                "required": ["prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_project",
            "description": "Create a new collaboration project to group conversations, files, and tasks together.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Project name"},
                    "description": {"type": "string", "description": "Project description"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorization"
                    }
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_projects",
            "description": "List all collaboration projects across the mesh.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_project",
            "description": "Add a conversation or file to a project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "description": "Project ID"},
                    "conversation_id": {"type": "string", "description": "Conversation ID to link"},
                    "file_path": {"type": "string", "description": "File path to link"}
                },
                "required": ["project_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_tasks",
            "description": "List dispatched tasks and their status (pending, running, completed, failed).",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "description": "Filter by status", "enum": ["pending", "running", "completed", "failed"]}
                },
                "required": []
            }
        }
    },
]

COLLAB_TOOL_NAMES = {t["function"]["name"] for t in COLLAB_TOOLS}


# ── Tool execution ───────────────────────────────────────────────

def execute_collab_tool(name: str, arguments: dict) -> str:
    """Execute a collaboration tool and return the result."""
    hub = get_hub()

    if name == "mesh_status":
        status = hub.get_status()
        lines = ["## MAUDE Mesh Status\n"]

        # Presence
        presence = status.get("presence", [])
        if presence:
            lines.append("### Online Devices")
            for p in presence:
                age = int(status["ts"] - p.get("last_seen", 0))
                activity = p.get("activity", "idle")
                plat = p.get("platform", p.get("client_type", "?"))
                cid = p.get("client_id", "?")
                lines.append(f"- **{p.get('hostname', '?')}** | client_id: `{cid}` | platform: `{plat}` — {activity} ({age}s ago)")
        else:
            lines.append("### No devices currently online")

        # Recent activity
        activity = status.get("activity", [])[:10]
        if activity:
            lines.append("\n### Recent Activity")
            for evt in activity:
                lines.append(f"- [{evt.get('hostname', '?')}] {evt.get('summary', '')}")

        # Tasks
        tasks = status.get("tasks", [])
        active_tasks = [t for t in tasks if t.get("status") in ("pending", "running")]
        if active_tasks:
            lines.append("\n### Active Tasks")
            for t in active_tasks:
                lines.append(f"- **{t['status']}** on {t.get('target', 'local')}: {t.get('prompt', '')[:60]}")

        # Projects
        projects = status.get("projects", [])
        if projects:
            lines.append(f"\n### Projects ({len(projects)})")
            for p in projects[:5]:
                convs = len(p.get("conversations", []))
                files = len(p.get("files", []))
                lines.append(f"- **{p['name']}** — {convs} conversations, {files} files")

        return "\n".join(lines)

    elif name == "dispatch_task":
        task = hub.dispatch_task(
            prompt=arguments.get("prompt", ""),
            target=arguments.get("target", ""),
            capability=arguments.get("capability", "LLM"),
            project_id=arguments.get("project_id", ""),
            target_client_id=arguments.get("target_client_id", ""),
            target_platform=arguments.get("target_platform", ""),
        )
        dest = task.get("target_client_id") or task.get("target_platform") or task.get("target") or "local"
        task_id = task["id"]

        # Already failed (e.g. target not found/offline)
        if task.get("status") == "failed":
            return f"ERROR: {task.get('result', 'Task failed')}"

        # For client-targeted tasks, wait for the result (client polls every 10s)
        if task.get("status") == "queued":
            for _ in range(15):  # wait up to 15s
                time.sleep(1)
                updated = hub.tasks.get(task_id)
                if updated and updated.get("status") in ("completed", "failed"):
                    result = updated.get("result", "(no output)")
                    return f"Task {updated['status']} on {dest}:\n{result}"
            return f"Task {task_id} dispatched to {dest} but no result yet (client may be slow to respond). Use list_tasks to check later."

        return f"Task dispatched: {task_id} → {dest} (status: {task['status']})"

    elif name == "create_project":
        proj = hub.create_project(
            name=arguments.get("name", "Untitled"),
            description=arguments.get("description", ""),
            tags=arguments.get("tags", []),
        )
        return f"Project created: {proj['name']} (id: {proj['id']})"

    elif name == "list_projects":
        projects = hub.list_projects()
        if not projects:
            return "No projects found."
        lines = ["## Projects\n"]
        for p in projects:
            tags = ", ".join(p.get("tags", [])) or "none"
            convs = len(p.get("conversations", []))
            files = len(p.get("files", []))
            lines.append(f"- **{p['name']}** (id: {p['id']})\n  Tags: {tags} | {convs} conversations, {files} files\n  {p.get('description', '')}")
        return "\n".join(lines)

    elif name == "add_to_project":
        ok = hub.add_to_project(
            project_id=arguments.get("project_id", ""),
            conversation_id=arguments.get("conversation_id", ""),
            file_path=arguments.get("file_path", ""),
        )
        return "Added to project." if ok else "Failed — project not found."

    elif name == "list_tasks":
        tasks = hub.tasks.list_all(status=arguments.get("status"))
        if not tasks:
            return "No tasks found."
        lines = ["## Tasks\n"]
        for t in tasks:
            result_preview = ""
            if t.get("result"):
                result_preview = f"\n  Result: {str(t['result'])[:100]}"
            dest = t.get("target_client_id") or t.get("target_platform") or t.get("target") or "local"
            lines.append(f"- **{t['status']}** [{t['id']}] → {dest}: {t.get('prompt', '')[:80]}{result_preview}")
        return "\n".join(lines)

    return f"Unknown collab tool: {name}"
