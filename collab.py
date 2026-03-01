"""
MAUDE Collaboration Hub — Presence, activity feed, projects, task dispatch, gossip sync.

Each gateway is authoritative for its own data. Gateways exchange state via
gossip bundles during the mesh health-check cycle (60s). Clients query their
local gateway which returns the merged global view (local + cached peer state).

Storage: JSON files in data/collab/ on each gateway. Thread-safe with locks +
atomic writes (write to .tmp then rename).
"""

import json
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Storage helpers ──────────────────────────────────────────────

COLLAB_DIR = Path(__file__).parent / "data" / "collab"
PROJECTS_DIR = COLLAB_DIR / "projects"
TASKS_DIR = COLLAB_DIR / "tasks"

for _d in (COLLAB_DIR, PROJECTS_DIR, TASKS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MY_HOSTNAME = socket.gethostname().lower()


def _atomic_write(path: Path, data: Any):
    """Write JSON atomically via temp file + rename."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, default=str))
    tmp.rename(path)


def _read_json(path: Path, default=None):
    """Read JSON file, returning default on any error."""
    try:
        return json.loads(path.read_text())
    except Exception:
        return default if default is not None else {}


# ── Presence Manager ─────────────────────────────────────────────

class PresenceManager:
    """Tracks client heartbeats. Prunes stale entries (>90s)."""

    STALE_THRESHOLD = 90  # seconds

    def __init__(self):
        self._lock = threading.Lock()
        self._file = COLLAB_DIR / "presence.json"
        self._clients: Dict[str, dict] = {}
        self._load()

    def _load(self):
        stored = _read_json(self._file, [])
        if isinstance(stored, list):
            for entry in stored:
                cid = entry.get("client_id")
                if cid:
                    self._clients[cid] = entry

    def heartbeat(self, client_id: str, client_type: str = "tui",
                  activity: str = "", conversation_id: str = "",
                  project_id: str = "", hostname: str = "",
                  platform: str = ""):
        with self._lock:
            self._clients[client_id] = {
                "client_id": client_id,
                "client_type": client_type,
                "hostname": hostname or MY_HOSTNAME,
                "platform": platform or client_type,
                "status": "active",
                "activity": activity,
                "active_conversation": conversation_id,
                "active_project": project_id,
                "last_seen": time.time(),
            }
            self._prune()
            self._save()

    def get_all(self) -> List[dict]:
        with self._lock:
            self._prune()
            return list(self._clients.values())

    def _prune(self):
        now = time.time()
        stale = [k for k, v in self._clients.items()
                 if now - v.get("last_seen", 0) > self.STALE_THRESHOLD]
        for k in stale:
            del self._clients[k]

    def _save(self):
        _atomic_write(self._file, list(self._clients.values()))

    def get_bundle(self) -> List[dict]:
        """Return presence data for gossip."""
        return self.get_all()

    def merge_peer(self, entries: List[dict]):
        """Merge peer presence entries (peer is authoritative for its hostname)."""
        with self._lock:
            for entry in entries:
                cid = entry.get("client_id")
                hostname = entry.get("hostname", "")
                if cid and hostname != MY_HOSTNAME:
                    self._clients[cid] = entry
            self._prune()
            self._save()


# ── Activity Feed ────────────────────────────────────────────────

class ActivityFeed:
    """Append-only JSONL activity log. Daily rotation, capped gossip (last 100)."""

    MAX_GOSSIP_EVENTS = 100

    def __init__(self):
        self._lock = threading.Lock()

    def _today_file(self) -> Path:
        return COLLAB_DIR / f"activity-{time.strftime('%Y-%m-%d')}.jsonl"

    def emit(self, event_type: str, summary: str, data: dict = None,
             client_id: str = "", conversation_id: str = "",
             project_id: str = ""):
        event = {
            "id": f"evt-{uuid.uuid4().hex[:12]}",
            "ts": time.time(),
            "hostname": MY_HOSTNAME,
            "client_id": client_id,
            "type": event_type,
            "summary": summary,
            "data": data or {},
        }
        if conversation_id:
            event["data"]["conversation_id"] = conversation_id
        if project_id:
            event["data"]["project_id"] = project_id

        with self._lock:
            with open(self._today_file(), "a") as f:
                f.write(json.dumps(event, default=str) + "\n")

    def get_recent(self, since: float = 0, limit: int = 50) -> List[dict]:
        """Get recent events, optionally since a timestamp."""
        events = []
        # Read today's and yesterday's files
        for days_ago in range(2):
            ts = time.time() - days_ago * 86400
            path = COLLAB_DIR / f"activity-{time.strftime('%Y-%m-%d', time.localtime(ts))}.jsonl"
            if path.exists():
                try:
                    for line in path.read_text().strip().split("\n"):
                        if not line:
                            continue
                        evt = json.loads(line)
                        if evt.get("ts", 0) > since:
                            events.append(evt)
                except Exception:
                    pass
        events.sort(key=lambda e: e.get("ts", 0), reverse=True)
        return events[:limit]

    def get_bundle(self) -> List[dict]:
        """Return last N events for gossip."""
        return self.get_recent(limit=self.MAX_GOSSIP_EVENTS)

    def merge_peer(self, events: List[dict]):
        """Merge peer events — only add events from their hostname we haven't seen."""
        existing_ids = set()
        for evt in self.get_recent(limit=500):
            existing_ids.add(evt.get("id"))

        with self._lock:
            with open(self._today_file(), "a") as f:
                for evt in events:
                    if evt.get("id") not in existing_ids and evt.get("hostname") != MY_HOSTNAME:
                        f.write(json.dumps(evt, default=str) + "\n")


# ── Project Manager ──────────────────────────────────────────────

class ProjectManager:
    """CRUD for projects. Each project links conversations + files."""

    def __init__(self):
        self._lock = threading.Lock()

    def create(self, name: str, description: str = "",
               tags: List[str] = None) -> dict:
        proj = {
            "id": f"proj-{uuid.uuid4().hex[:12]}",
            "name": name,
            "description": description,
            "created_at": time.time(),
            "updated_at": time.time(),
            "hostname": MY_HOSTNAME,
            "conversations": [],
            "files": [],
            "tags": tags or [],
        }
        with self._lock:
            _atomic_write(PROJECTS_DIR / f"{proj['id']}.json", proj)
        return proj

    def get(self, project_id: str) -> Optional[dict]:
        safe_id = project_id.replace("/", "").replace("..", "")
        return _read_json(PROJECTS_DIR / f"{safe_id}.json", None)

    def update(self, project_id: str, **kwargs) -> Optional[dict]:
        proj = self.get(project_id)
        if not proj:
            return None
        for key in ("name", "description", "tags", "conversations", "files"):
            if key in kwargs:
                proj[key] = kwargs[key]
        proj["updated_at"] = time.time()
        with self._lock:
            _atomic_write(PROJECTS_DIR / f"{project_id}.json", proj)
        return proj

    def delete(self, project_id: str) -> bool:
        safe_id = project_id.replace("/", "").replace("..", "")
        path = PROJECTS_DIR / f"{safe_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def list_all(self) -> List[dict]:
        projects = []
        for f in PROJECTS_DIR.glob("proj-*.json"):
            proj = _read_json(f, None)
            if proj:
                projects.append(proj)
        projects.sort(key=lambda p: p.get("updated_at", 0), reverse=True)
        return projects

    def add_conversation(self, project_id: str, conversation_id: str) -> bool:
        proj = self.get(project_id)
        if not proj:
            return False
        if conversation_id not in proj["conversations"]:
            proj["conversations"].append(conversation_id)
            proj["updated_at"] = time.time()
            with self._lock:
                _atomic_write(PROJECTS_DIR / f"{project_id}.json", proj)
        return True

    def add_file(self, project_id: str, file_path: str) -> bool:
        proj = self.get(project_id)
        if not proj:
            return False
        if file_path not in proj["files"]:
            proj["files"].append(file_path)
            proj["updated_at"] = time.time()
            with self._lock:
                _atomic_write(PROJECTS_DIR / f"{project_id}.json", proj)
        return True

    def get_bundle(self) -> List[dict]:
        return self.list_all()

    def merge_peer(self, projects: List[dict]):
        """Merge peer projects — peer is authoritative for its own projects."""
        with self._lock:
            for proj in projects:
                pid = proj.get("id")
                hostname = proj.get("hostname", "")
                if not pid or hostname == MY_HOSTNAME:
                    continue
                existing = _read_json(PROJECTS_DIR / f"{pid}.json", None)
                # Accept peer version if newer or doesn't exist locally
                if not existing or proj.get("updated_at", 0) > existing.get("updated_at", 0):
                    _atomic_write(PROJECTS_DIR / f"{pid}.json", proj)


# ── Task Dispatcher ──────────────────────────────────────────────

class TaskDispatcher:
    """Creates tasks, forwards to remote gateways, tracks state."""

    def __init__(self):
        self._lock = threading.Lock()

    def create(self, prompt: str, target: str = "", capability: str = "LLM",
               project_id: str = "", source: str = "",
               target_client_id: str = "", target_platform: str = "") -> dict:
        # Client-targeted tasks start as "queued" (picked up by polling clients)
        is_client_targeted = bool(target_client_id or target_platform)
        task = {
            "id": f"task-{uuid.uuid4().hex[:12]}",
            "source": source or MY_HOSTNAME,
            "target": target,
            "target_client_id": target_client_id,
            "target_platform": target_platform,
            "capability": capability,
            "status": "queued" if is_client_targeted else "pending",
            "prompt": prompt,
            "result": None,
            "project_id": project_id,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        with self._lock:
            _atomic_write(TASKS_DIR / f"{task['id']}.json", task)
        return task

    def get_queued_for_client(self, client_id: str) -> List[dict]:
        """Return tasks with status='queued' targeting this client_id."""
        tasks = []
        for f in TASKS_DIR.glob("task-*.json"):
            task = _read_json(f, None)
            if task and task.get("status") == "queued":
                if task.get("target_client_id") == client_id:
                    tasks.append(task)
        tasks.sort(key=lambda t: t.get("created_at", 0))
        return tasks

    def resolve_platform_targets(self, presence_clients: List[dict]):
        """For queued tasks with target_platform but no target_client_id,
        assign to the first active client on that platform."""
        # Build platform → client_id map
        platform_map: Dict[str, str] = {}
        for client in presence_clients:
            plat = client.get("platform", "")
            cid = client.get("client_id", "")
            if plat and cid and plat not in platform_map:
                platform_map[plat] = cid

        with self._lock:
            for f in TASKS_DIR.glob("task-*.json"):
                task = _read_json(f, None)
                if (task and task.get("status") == "queued"
                        and task.get("target_platform")
                        and not task.get("target_client_id")):
                    target_plat = task["target_platform"]
                    if target_plat in platform_map:
                        task["target_client_id"] = platform_map[target_plat]
                        task["updated_at"] = time.time()
                        _atomic_write(TASKS_DIR / f"{task['id']}.json", task)

    def get(self, task_id: str) -> Optional[dict]:
        safe_id = task_id.replace("/", "").replace("..", "")
        return _read_json(TASKS_DIR / f"{safe_id}.json", None)

    def update_status(self, task_id: str, status: str, result: str = None) -> Optional[dict]:
        task = self.get(task_id)
        if not task:
            return None
        task["status"] = status
        if result is not None:
            task["result"] = result
        task["updated_at"] = time.time()
        with self._lock:
            _atomic_write(TASKS_DIR / f"{task_id}.json", task)
        return task

    def list_all(self, status: str = None) -> List[dict]:
        tasks = []
        for f in TASKS_DIR.glob("task-*.json"):
            task = _read_json(f, None)
            if task:
                if status and task.get("status") != status:
                    continue
                tasks.append(task)
        tasks.sort(key=lambda t: t.get("created_at", 0), reverse=True)
        return tasks

    def execute(self, task_dict: dict) -> str:
        """Execute an incoming task locally. Returns result string."""
        task_id = task_dict.get("id")
        prompt = task_dict.get("prompt", "")
        capability = task_dict.get("capability", "LLM")

        # Save task locally
        with self._lock:
            _atomic_write(TASKS_DIR / f"{task_id}.json", task_dict)

        self.update_status(task_id, "running")

        try:
            if capability == "SHELL":
                # Execute shell command
                import subprocess
                result = subprocess.run(
                    prompt, shell=True, capture_output=True, text=True, timeout=60
                )
                output = result.stdout or result.stderr or "(no output)"
                self.update_status(task_id, "completed", output)
                return output
            elif capability == "LLM":
                # Use local LLM via maude_core
                try:
                    from maude_core import execute_tool
                    result = execute_tool("run_command", {"command": prompt})
                    self.update_status(task_id, "completed", str(result))
                    return str(result)
                except ImportError:
                    self.update_status(task_id, "failed", "maude_core not available")
                    return "Error: maude_core not available"
            else:
                self.update_status(task_id, "failed", f"Unknown capability: {capability}")
                return f"Unknown capability: {capability}"
        except Exception as e:
            self.update_status(task_id, "failed", str(e))
            return f"Error: {e}"

    def get_bundle(self) -> List[dict]:
        """Return recent tasks for gossip (last 50)."""
        return self.list_all()[:50]

    def merge_peer(self, tasks: List[dict]):
        """Merge peer tasks — update local cache of remote task states."""
        with self._lock:
            for task in tasks:
                tid = task.get("id")
                if not tid:
                    continue
                existing = _read_json(TASKS_DIR / f"{tid}.json", None)
                if not existing or task.get("updated_at", 0) > existing.get("updated_at", 0):
                    _atomic_write(TASKS_DIR / f"{tid}.json", task)


# ── Gossip Sync ──────────────────────────────────────────────────

PEERS_CACHE = COLLAB_DIR / "peers_cache.json"


class GossipSync:
    """Produces and consumes gossip state bundles for peer exchange."""

    def __init__(self, presence: PresenceManager, activity: ActivityFeed,
                 projects: ProjectManager, tasks: TaskDispatcher):
        self.presence = presence
        self.activity = activity
        self.projects = projects
        self.tasks = tasks
        self._lock = threading.Lock()

    def get_bundle(self) -> dict:
        """Produce this node's state bundle for peers to consume."""
        return {
            "hostname": MY_HOSTNAME,
            "ts": time.time(),
            "presence": self.presence.get_bundle(),
            "activity": self.activity.get_bundle(),
            "projects": self.projects.get_bundle(),
            "tasks": self.tasks.get_bundle(),
        }

    def merge(self, hostname: str, bundle: dict):
        """Ingest a peer's gossip bundle."""
        if not bundle or hostname == MY_HOSTNAME:
            return

        self.presence.merge_peer(bundle.get("presence", []))
        self.activity.merge_peer(bundle.get("activity", []))
        self.projects.merge_peer(bundle.get("projects", []))
        self.tasks.merge_peer(bundle.get("tasks", []))

        # Cache the raw bundle for diagnostics
        with self._lock:
            cache = _read_json(PEERS_CACHE, {})
            cache[hostname] = {
                "ts": time.time(),
                "bundle_ts": bundle.get("ts"),
            }
            _atomic_write(PEERS_CACHE, cache)


# ── CollabHub (Singleton) ────────────────────────────────────────

class CollabHub:
    """Main collaboration hub composing all managers."""

    def __init__(self):
        self.presence = PresenceManager()
        self.activity = ActivityFeed()
        self.projects = ProjectManager()
        self.tasks = TaskDispatcher()
        self.gossip = GossipSync(
            self.presence, self.activity, self.projects, self.tasks
        )

    # ── Presence ──

    def heartbeat(self, client_id: str, client_type: str = "tui",
                  activity: str = "", conversation_id: str = "",
                  project_id: str = "", hostname: str = "",
                  platform: str = ""):
        self.presence.heartbeat(client_id, client_type, activity,
                                conversation_id, project_id,
                                hostname, platform)

    # ── Activity ──

    def emit(self, event_type: str, summary: str, data: dict = None,
             client_id: str = "", conversation_id: str = "",
             project_id: str = ""):
        self.activity.emit(event_type, summary, data, client_id,
                           conversation_id, project_id)

    # ── Status (merged view) ──

    def get_status(self) -> dict:
        """Return full merged dashboard view for clients."""
        return {
            "hostname": MY_HOSTNAME,
            "ts": time.time(),
            "presence": self.presence.get_all(),
            "activity": self.activity.get_recent(),
            "projects": self.projects.list_all(),
            "tasks": self.tasks.list_all(),
        }

    # ── Gossip ──

    def get_gossip_bundle(self) -> dict:
        return self.gossip.get_bundle()

    def merge_gossip(self, hostname: str, bundle: dict):
        self.gossip.merge(hostname, bundle)

    # ── Task dispatch ──

    def dispatch_task(self, prompt: str, target: str = "",
                      capability: str = "LLM", project_id: str = "",
                      target_client_id: str = "", target_platform: str = "") -> dict:
        # Resolve target: check if it matches a client_id, hostname, or platform
        if target and not target_client_id and not target_platform:
            presence = self.presence.get_all()
            target_lower = target.lower()
            # 1. Exact client_id match
            for p in presence:
                if p.get("client_id") == target:
                    target_client_id = target
                    break
            # 2. Hostname match (exact or substring, case-insensitive)
            if not target_client_id:
                for p in presence:
                    h = p.get("hostname", "").lower()
                    if h and (h == target_lower or target_lower in h or h in target_lower):
                        target_client_id = p.get("client_id", "")
                        break
            # 3. Platform match
            if not target_client_id:
                for p in presence:
                    if p.get("platform", "").lower() == target_lower:
                        target_platform = target_lower
                        break

        is_client_targeted = bool(target_client_id or target_platform)
        task = self.tasks.create(
            prompt, target, capability, project_id,
            target_client_id=target_client_id,
            target_platform=target_platform,
        )

        dest = target_client_id or target_platform or target or "local"
        self.emit("task_dispatched", f"Dispatched task to {dest}",
                  {"task_id": task["id"], "target": dest})

        if is_client_targeted:
            # Client-targeted: stays queued, client polls for it
            pass
        elif target and target != MY_HOSTNAME:
            # Forward to remote gateway via mesh
            threading.Thread(
                target=self._forward_task, args=(target, task), daemon=True
            ).start()
        else:
            # Execute locally in background
            threading.Thread(
                target=self.tasks.execute, args=(task,), daemon=True
            ).start()

        return task

    def _forward_task(self, target: str, task: dict):
        """Forward task to a remote gateway for execution."""
        import requests
        try:
            from mesh import get_mesh
            mesh = get_mesh()
            node = mesh.nodes.get(target)
            if not node:
                self.tasks.update_status(task["id"], "failed",
                                         f"Node {target} not found in mesh")
                return
            host = node.tailscale_ip or node.ip or node.hostname
            url = f"http://{host}:{node.api_port}/api/collab/tasks/execute"
            resp = requests.post(url, json=task, timeout=120)
            if resp.status_code == 200:
                result = resp.json()
                self.tasks.update_status(
                    task["id"], result.get("status", "completed"),
                    result.get("result", "")
                )
            else:
                self.tasks.update_status(task["id"], "failed",
                                         f"Remote error: {resp.status_code}")
        except Exception as e:
            self.tasks.update_status(task["id"], "failed", str(e))

    def execute_task(self, task_dict: dict) -> dict:
        """Execute an incoming task from a peer. Returns result dict."""
        result = self.tasks.execute(task_dict)
        task = self.tasks.get(task_dict["id"])
        return {
            "task_id": task_dict["id"],
            "status": task.get("status", "completed") if task else "completed",
            "result": result,
        }

    # ── Project CRUD ──

    def create_project(self, name: str, description: str = "",
                       tags: List[str] = None) -> dict:
        proj = self.projects.create(name, description, tags)
        self.emit("project_created", f"Created project: {name}",
                  {"project_id": proj["id"]})
        return proj

    def update_project(self, project_id: str, **kwargs) -> Optional[dict]:
        return self.projects.update(project_id, **kwargs)

    def list_projects(self) -> List[dict]:
        return self.projects.list_all()

    def get_project(self, project_id: str) -> Optional[dict]:
        return self.projects.get(project_id)

    def delete_project(self, project_id: str) -> bool:
        return self.projects.delete(project_id)

    def add_to_project(self, project_id: str, conversation_id: str = "",
                       file_path: str = "") -> bool:
        if conversation_id:
            return self.projects.add_conversation(project_id, conversation_id)
        if file_path:
            return self.projects.add_file(project_id, file_path)
        return False


# ── Module-level singleton ───────────────────────────────────────

_hub: Optional[CollabHub] = None
_hub_lock = threading.Lock()


def get_hub() -> CollabHub:
    """Get or create the global CollabHub instance."""
    global _hub
    if _hub is None:
        with _hub_lock:
            if _hub is None:
                _hub = CollabHub()
    return _hub
