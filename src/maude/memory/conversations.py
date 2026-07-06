"""
Conversation sync — saves/loads chat history through the MAUDE gateway API.

Shared by all interfaces (server TUI, client CLI, phone app) so conversations
carry across devices. The gateway stores JSON files in data/conversations/.
"""

import json
import threading
import time
import uuid
from urllib.error import URLError
from urllib.request import Request, urlopen

GATEWAY_API = "http://localhost:30080"


def _api_get(path: str):
    """GET JSON from gateway API."""
    try:
        r = urlopen(f"{GATEWAY_API}{path}", timeout=3)
        return json.loads(r.read())
    except (URLError, Exception):
        return None


def _api_post(path: str, data):
    """POST JSON to gateway API (fire-and-forget in background thread)."""

    def _do():
        try:
            body = json.dumps(data).encode()
            req = Request(
                f"{GATEWAY_API}{path}", data=body, headers={"Content-Type": "application/json"}, method="POST"
            )
            urlopen(req, timeout=3)
        except (URLError, Exception):
            pass

    threading.Thread(target=_do, daemon=True).start()


def save_conversation(conv_id: str, title: str, model: str, messages: list[dict], project_id: str = ""):
    """Save a conversation's metadata and messages to the server.

    Updates the conversation index and saves messages. Skips system messages.
    Optionally links the conversation to a project.
    """
    # Filter out system messages for storage
    stored_msgs = []
    for m in messages:
        if m.get("role") == "system":
            continue
        stored_msgs.append(
            {
                "id": m.get("id", str(uuid.uuid4())),
                "role": m["role"],
                "content": m["content"],
                "model": m.get("model", model),
                "timestamp": m.get("timestamp", int(time.time() * 1000)),
            }
        )

    # Update index
    now = int(time.time() * 1000)
    index = _api_get("/api/conversations") or []

    existing = next((c for c in index if c["id"] == conv_id), None)
    if existing:
        existing["updatedAt"] = now
        existing["title"] = title
        existing["model"] = model
    else:
        index.insert(
            0,
            {
                "id": conv_id,
                "title": title,
                "createdAt": now,
                "updatedAt": now,
                "model": model,
            },
        )

    # Sort newest first
    index.sort(key=lambda c: c.get("updatedAt", 0), reverse=True)

    _api_post("/api/conversations", index)
    _api_post(f"/api/conversations/{conv_id}/messages", stored_msgs)

    # Link to project if specified
    if project_id:
        try:
            from collab import get_hub

            get_hub().add_to_project(project_id, conversation_id=conv_id)
        except Exception:
            pass


def generate_title(first_message: str) -> str:
    """Generate a short title from the first user message."""
    trimmed = " ".join(first_message.strip().split())
    return trimmed[:37] + "..." if len(trimmed) > 40 else trimmed
