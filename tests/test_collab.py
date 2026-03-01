"""
Tests for collab.py — CollabHub: presence, activity, projects, tasks, gossip.
"""

import json
import os
import time
import pytest
from pathlib import Path
from unittest.mock import patch

# Patch COLLAB_DIR before importing collab to use temp dirs
import collab as collab_module


@pytest.fixture
def hub(tmp_data_dir):
    """Create a CollabHub with temp storage."""
    # Monkey-patch the storage dirs
    orig_collab = collab_module.COLLAB_DIR
    orig_projects = collab_module.PROJECTS_DIR
    orig_tasks = collab_module.TASKS_DIR

    collab_module.COLLAB_DIR = tmp_data_dir
    collab_module.PROJECTS_DIR = tmp_data_dir / "projects"
    collab_module.TASKS_DIR = tmp_data_dir / "tasks"
    collab_module.PROJECTS_DIR.mkdir(exist_ok=True)
    collab_module.TASKS_DIR.mkdir(exist_ok=True)

    # Create fresh instances
    hub = collab_module.CollabHub()
    hub.presence = collab_module.PresenceManager()
    hub.presence._file = tmp_data_dir / "presence.json"
    hub.presence._clients = {}
    hub.activity = collab_module.ActivityFeed()
    hub.projects = collab_module.ProjectManager()
    hub.tasks = collab_module.TaskDispatcher()

    yield hub

    # Restore
    collab_module.COLLAB_DIR = orig_collab
    collab_module.PROJECTS_DIR = orig_projects
    collab_module.TASKS_DIR = orig_tasks


class TestPresence:
    def test_heartbeat_registers(self, hub):
        hub.heartbeat("client-1", "tui", "chatting")
        presence = hub.presence.get_all()
        assert len(presence) >= 1
        assert any(p["client_id"] == "client-1" for p in presence)

    def test_heartbeat_returns_in_status(self, hub):
        hub.heartbeat("client-1", "tui", "chatting")
        status = hub.get_status()
        assert "presence" in status
        assert len(status["presence"]) >= 1

    def test_stale_entries_pruned(self, hub):
        # Register a client with a stale timestamp
        hub.presence._clients["old-client"] = {
            "client_id": "old-client",
            "client_type": "tui",
            "hostname": "test",
            "last_seen": time.time() - 100,  # > 90s threshold
            "activity": "idle",
        }
        # Prune happens on get_all()
        result = hub.presence.get_all()
        assert not any(p["client_id"] == "old-client" for p in result)


class TestActivity:
    def test_emit_and_retrieve(self, hub):
        hub.activity.emit("test", "did a thing", client_id="c1")
        events = hub.activity.get_recent(limit=10)
        assert len(events) >= 1
        assert events[0]["summary"] == "did a thing"


class TestProjects:
    def test_create_and_list(self, hub):
        proj = hub.create_project("Test Project", "A test", ["tag1"])
        assert proj["name"] == "Test Project"
        assert "id" in proj

        projects = hub.list_projects()
        assert len(projects) >= 1
        assert any(p["id"] == proj["id"] for p in projects)

    def test_get_project(self, hub):
        proj = hub.create_project("Fetch Me", "desc")
        fetched = hub.projects.get(proj["id"])
        assert fetched is not None
        assert fetched["name"] == "Fetch Me"

    def test_delete_project(self, hub):
        proj = hub.create_project("To Delete", "bye")
        assert hub.projects.delete(proj["id"]) is True
        assert not hub.projects.get(proj["id"])

    def test_add_to_project(self, hub):
        proj = hub.create_project("Links", "link test")
        hub.add_to_project(proj["id"], conversation_id="conv-123")
        updated = hub.projects.get(proj["id"])
        assert "conv-123" in updated["conversations"]


class TestTasks:
    def test_dispatch_and_status(self, hub):
        task = hub.dispatch_task("say hello", target="local", capability="LLM")
        assert task["status"] == "pending"
        assert "id" in task

        tasks = hub.tasks.list_all()
        assert len(tasks) >= 1
        assert any(t["id"] == task["id"] for t in tasks)

    def test_task_status_filter(self, hub):
        hub.dispatch_task("task1", capability="LLM")
        hub.dispatch_task("task2", capability="SHELL")
        all_tasks = hub.tasks.list_all()
        pending = hub.tasks.list_all(status="pending")
        assert len(pending) <= len(all_tasks)


class TestGossip:
    def test_gossip_bundle_has_sections(self, hub):
        hub.heartbeat("c1", "tui", "working")
        hub.activity.emit("test", "event")
        hub.create_project("GossipProj", "test")
        hub.dispatch_task("gossip task", capability="LLM")

        bundle = hub.get_gossip_bundle()
        assert "presence" in bundle
        assert "activity" in bundle
        assert "projects" in bundle
        assert "tasks" in bundle
