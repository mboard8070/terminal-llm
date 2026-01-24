"""Quick notes skill - simple key-value note storage."""

import json
from pathlib import Path
from datetime import datetime
from skills import skill


NOTES_FILE = Path.home() / ".config" / "maude" / "notes.json"


def _load_notes() -> dict:
    """Load notes from file."""
    if NOTES_FILE.exists():
        try:
            return json.loads(NOTES_FILE.read_text())
        except:
            pass
    return {}


def _save_notes(notes: dict):
    """Save notes to file."""
    NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    NOTES_FILE.write_text(json.dumps(notes, indent=2))


@skill(
    name="note",
    description="Save and retrieve quick notes",
    version="1.0.0",
    author="MAUDE",
    triggers=["note", "notes", "memo"],
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["save", "get", "list", "delete", "search"],
                "description": "Action to perform",
                "default": "list"
            },
            "key": {
                "type": "string",
                "description": "Note key/title"
            },
            "content": {
                "type": "string",
                "description": "Note content (for save action)"
            },
            "query": {
                "type": "string",
                "description": "Search query (for search action)"
            }
        }
    }
)
def note(
    action: str = "list",
    key: str = None,
    content: str = None,
    query: str = None
) -> str:
    """Manage quick notes."""

    notes = _load_notes()

    if action == "save":
        if not key or not content:
            return "Error: Please provide both 'key' and 'content' parameters"

        notes[key] = {
            "content": content,
            "created": datetime.now().isoformat(),
            "updated": datetime.now().isoformat()
        }
        _save_notes(notes)
        return f"Note '{key}' saved."

    elif action == "get":
        if not key:
            return "Error: Please provide 'key' parameter"

        if key not in notes:
            return f"Note '{key}' not found."

        note_data = notes[key]
        return (
            f"Note: {key}\n"
            f"─" * 30 + "\n"
            f"{note_data['content']}\n"
            f"─" * 30 + "\n"
            f"Created: {note_data.get('created', 'unknown')}"
        )

    elif action == "list":
        if not notes:
            return "No notes saved. Use action='save' with key and content to create one."

        lines = [f"Notes ({len(notes)} total):\n"]
        for k, v in sorted(notes.items()):
            preview = v['content'][:50] + "..." if len(v['content']) > 50 else v['content']
            preview = preview.replace('\n', ' ')
            lines.append(f"  • {k}: {preview}")
        return "\n".join(lines)

    elif action == "delete":
        if not key:
            return "Error: Please provide 'key' parameter"

        if key not in notes:
            return f"Note '{key}' not found."

        del notes[key]
        _save_notes(notes)
        return f"Note '{key}' deleted."

    elif action == "search":
        if not query:
            return "Error: Please provide 'query' parameter"

        query_lower = query.lower()
        matches = []
        for k, v in notes.items():
            if query_lower in k.lower() or query_lower in v['content'].lower():
                preview = v['content'][:50] + "..." if len(v['content']) > 50 else v['content']
                matches.append(f"  • {k}: {preview}")

        if not matches:
            return f"No notes matching '{query}'"

        return f"Notes matching '{query}':\n" + "\n".join(matches)

    return f"Unknown action: {action}"


@skill(
    name="todo",
    description="Simple todo list management",
    version="1.0.0",
    author="MAUDE",
    triggers=["todo", "task", "tasks"],
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "list", "done", "remove", "clear"],
                "description": "Action to perform",
                "default": "list"
            },
            "task": {
                "type": "string",
                "description": "Task description or number"
            }
        }
    }
)
def todo(action: str = "list", task: str = None) -> str:
    """Manage a simple todo list."""

    TODO_FILE = Path.home() / ".config" / "maude" / "todos.json"

    def load_todos():
        if TODO_FILE.exists():
            try:
                return json.loads(TODO_FILE.read_text())
            except:
                pass
        return []

    def save_todos(todos):
        TODO_FILE.parent.mkdir(parents=True, exist_ok=True)
        TODO_FILE.write_text(json.dumps(todos, indent=2))

    todos = load_todos()

    if action == "add":
        if not task:
            return "Error: Please provide 'task' parameter"

        todos.append({
            "task": task,
            "done": False,
            "created": datetime.now().isoformat()
        })
        save_todos(todos)
        return f"Added: {task}"

    elif action == "list":
        if not todos:
            return "No todos. Use action='add' with task='...' to add one."

        lines = ["Todo List:\n"]
        for i, t in enumerate(todos, 1):
            status = "✓" if t.get("done") else " "
            lines.append(f"  [{status}] {i}. {t['task']}")
        return "\n".join(lines)

    elif action == "done":
        if not task:
            return "Error: Please provide task number"

        try:
            idx = int(task) - 1
            if 0 <= idx < len(todos):
                todos[idx]["done"] = True
                save_todos(todos)
                return f"Marked done: {todos[idx]['task']}"
            return f"Invalid task number: {task}"
        except ValueError:
            return "Error: Please provide a task number"

    elif action == "remove":
        if not task:
            return "Error: Please provide task number"

        try:
            idx = int(task) - 1
            if 0 <= idx < len(todos):
                removed = todos.pop(idx)
                save_todos(todos)
                return f"Removed: {removed['task']}"
            return f"Invalid task number: {task}"
        except ValueError:
            return "Error: Please provide a task number"

    elif action == "clear":
        # Clear only completed tasks
        original = len(todos)
        todos = [t for t in todos if not t.get("done")]
        save_todos(todos)
        cleared = original - len(todos)
        return f"Cleared {cleared} completed tasks. {len(todos)} remaining."

    return f"Unknown action: {action}"
