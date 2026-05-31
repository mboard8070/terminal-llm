import json
import re
import asyncio

from maude_core import execute_tool, get_tools_for_message


def _created_id(result: str) -> str:
    match = re.search(r"^([a-z0-9-]+) \| active \|", result, re.MULTILINE)
    assert match, result
    return match.group(1)


def test_mission_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setenv("MAUDE_MISSIONS_DIR", str(tmp_path))

    created = execute_tool(
        "mission_create",
        {
            "title": "Content Engine",
            "objective": "Publish one useful AI video per day.",
            "success_criteria": ["Video rendered", "Video uploaded"],
            "steps": ["Research topic", "Render video", "Publish"],
            "cadence": "daily",
        },
    )
    mission_id = _created_id(created)

    listed = execute_tool("mission_list", {})
    assert mission_id in listed
    assert "Content Engine" in listed

    brief = execute_tool("mission_brief", {"id": mission_id})
    assert "Research topic" in brief
    assert "Video rendered" in brief

    updated = execute_tool("mission_update", {"id": mission_id, "steps": [{"id": "s1", "status": "done"}]})
    assert "1/3 steps complete" in updated

    logged = execute_tool(
        "mission_log",
        {"id": mission_id, "message": "Rendered first draft.", "kind": "result", "artifacts": ["/tmp/video.mp4"]},
    )
    assert f"Logged mission event for {mission_id}" in logged

    raw = execute_tool("mission_get", {"id": mission_id})
    mission = json.loads(raw)
    assert mission["steps"][0]["status"] == "done"
    assert mission["logs"][0]["message"] == "Rendered first draft."
    assert mission["artifacts"] == ["/tmp/video.mp4"]
    assert mission["checkpoint"]["next_action"] == "Render video"
    assert mission["checkpoint"]["artifacts"] == ["/tmp/video.mp4"]


def test_mission_run_next_executes_stored_plan(tmp_path, monkeypatch):
    monkeypatch.setenv("MAUDE_MISSIONS_DIR", str(tmp_path))

    created = execute_tool(
        "mission_create",
        {
            "title": "Runner Mission",
            "objective": "Prove mission steps can execute plans.",
            "steps": [
                {
                    "title": "Read mission brief",
                    "plan": [[{"name": "mission_brief", "args": {"id": "$MISSION_ID"}}]],
                }
            ],
        },
    )
    mission_id = _created_id(created)
    execute_tool(
        "mission_update",
        {
            "id": mission_id,
            "steps": [
                {
                    "id": "s1",
                    "plan": [[{"name": "mission_brief", "args": {"id": mission_id}}]],
                }
            ],
        },
    )

    result = execute_tool("mission_run_next", {"id": mission_id})
    assert "Mission step s1 done." in result
    assert "Plan executed: 1 stages, 1 tools" in result

    raw = execute_tool("mission_get", {"id": mission_id})
    mission = json.loads(raw)
    assert mission["steps"][0]["status"] == "done"
    assert "Plan executed" in mission["steps"][0]["last_result"]
    assert mission["logs"][-1]["kind"] == "result"


def test_mission_run_next_requires_plan(tmp_path, monkeypatch):
    monkeypatch.setenv("MAUDE_MISSIONS_DIR", str(tmp_path))

    created = execute_tool(
        "mission_create",
        {
            "title": "Manual Mission",
            "objective": "Keep manual steps readable.",
            "steps": ["Decide next action"],
        },
    )
    mission_id = _created_id(created)

    result = execute_tool("mission_run_next", {"id": mission_id})
    assert "has no executable plan" in result


def test_mission_tick_checkpoints_manual_step(tmp_path, monkeypatch):
    monkeypatch.setenv("MAUDE_MISSIONS_DIR", str(tmp_path))

    created = execute_tool(
        "mission_create",
        {
            "title": "Tick Manual Mission",
            "objective": "Checkpoint manual steps.",
            "steps": ["Decide next action"],
        },
    )
    mission_id = _created_id(created)

    result = execute_tool("mission_tick", {"id": mission_id})
    assert "needs a plan before it can run" in result
    assert "Decide next action" in result

    raw = execute_tool("mission_get", {"id": mission_id})
    mission = json.loads(raw)
    assert mission["steps"][0]["status"] == "pending"
    assert mission["logs"][-1]["kind"] == "checkpoint"
    assert mission["checkpoint"]["next_action"] == "Decide next action"


def test_mission_tick_runs_executable_step_and_auto_completes(tmp_path, monkeypatch):
    monkeypatch.setenv("MAUDE_MISSIONS_DIR", str(tmp_path))

    created = execute_tool(
        "mission_create",
        {
            "title": "Tick Runner Mission",
            "objective": "Run one executable tick.",
            "steps": [
                {
                    "title": "Read mission brief",
                    "plan": [[{"name": "mission_brief", "args": {"id": "$MISSION_ID"}}]],
                }
            ],
        },
    )
    mission_id = _created_id(created)

    result = execute_tool("mission_tick", {"id": mission_id, "auto_complete": True})
    assert "Mission step s1 done." in result
    assert "Mission auto-completed." in result

    raw = execute_tool("mission_get", {"id": mission_id})
    mission = json.loads(raw)
    assert mission["status"] == "complete"
    assert mission["steps"][0]["status"] == "done"
    assert mission["checkpoint"]["progress"] == {"done": 1, "total": 1}
    assert mission["checkpoint"]["next_action"] == ""


def test_mission_drain_runs_until_manual_step(tmp_path, monkeypatch):
    monkeypatch.setenv("MAUDE_MISSIONS_DIR", str(tmp_path))

    created = execute_tool(
        "mission_create",
        {
            "title": "Drain Mission",
            "objective": "Run planned steps without prompt tweaks.",
            "steps": [
                {
                    "title": "First planned step",
                    "plan": [[{"name": "mission_brief", "args": {"id": "$MISSION_ID"}}]],
                },
                {
                    "title": "Second planned step",
                    "plan": [[{"name": "mission_brief", "args": {"id": "$MISSION_ID"}}]],
                },
                "Manual review",
            ],
        },
    )
    mission_id = _created_id(created)

    result = execute_tool("mission_drain", {"id": mission_id, "auto_complete": True})
    assert "Mission drain ran 2 executable step(s)." in result
    assert "Manual review" in result
    assert "needs a plan before it can run" in result

    raw = execute_tool("mission_get", {"id": mission_id})
    mission = json.loads(raw)
    assert [step["status"] for step in mission["steps"]] == ["done", "done", "pending"]
    assert mission["status"] == "active"


def test_scheduler_directly_drains_mission_tasks(tmp_path, monkeypatch):
    monkeypatch.setenv("MAUDE_MISSIONS_DIR", str(tmp_path / "missions"))

    import scheduler

    monkeypatch.setattr(scheduler.ProactiveScheduler, "CONFIG_FILE", tmp_path / "schedules.json")

    created = execute_tool(
        "mission_create",
        {
            "title": "Scheduled Drain Mission",
            "objective": "Scheduled missions flow through stored plans.",
            "steps": [
                {
                    "title": "First planned step",
                    "plan": [[{"name": "mission_brief", "args": {"id": "$MISSION_ID"}}]],
                },
                {
                    "title": "Second planned step",
                    "plan": [[{"name": "mission_brief", "args": {"id": "$MISSION_ID"}}]],
                },
            ],
        },
    )
    mission_id = _created_id(created)
    task = scheduler.ScheduledTask(
        id="task1",
        name="Mission tick: Scheduled Drain Mission",
        cron="@daily",
        prompt=f"Use the mission_tick tool with id '{mission_id}' and auto_complete=true.",
    )

    runner = scheduler.ProactiveScheduler()
    result = asyncio.run(runner.run_task(task))

    assert "Mission progress is 2/2 steps." in result
    assert task.run_count == 1

    raw = execute_tool("mission_get", {"id": mission_id})
    mission = json.loads(raw)
    assert mission["status"] == "complete"
    assert [step["status"] for step in mission["steps"]] == ["done", "done"]


def test_scheduler_drains_mission_tasks_from_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("MAUDE_MISSIONS_DIR", str(tmp_path / "missions"))

    import scheduler

    monkeypatch.setattr(scheduler.ProactiveScheduler, "CONFIG_FILE", tmp_path / "schedules.json")

    created = execute_tool(
        "mission_create",
        {
            "title": "Metadata Scheduled Mission",
            "objective": "Scheduled missions should not depend on prompt wording.",
            "steps": [
                {
                    "title": "First planned step",
                    "plan": [[{"name": "mission_brief", "args": {"id": "$MISSION_ID"}}]],
                },
                {
                    "title": "Second planned step",
                    "plan": [[{"name": "mission_brief", "args": {"id": "$MISSION_ID"}}]],
                },
            ],
        },
    )
    mission_id = _created_id(created)
    task = scheduler.ScheduledTask(
        id="task1",
        name="Mission tick: Metadata Scheduled Mission",
        cron="@daily",
        prompt="fallback prompt with no tool syntax",
        task_type="mission",
        mission_id=mission_id,
        mission_auto_complete=True,
    )

    runner = scheduler.ProactiveScheduler()
    result = asyncio.run(runner.run_task(task))

    assert "Mission progress is 2/2 steps." in result

    raw = execute_tool("mission_get", {"id": mission_id})
    mission = json.loads(raw)
    assert mission["status"] == "complete"
    assert [step["status"] for step in mission["steps"]] == ["done", "done"]


def test_scheduler_records_and_reschedules_failures(tmp_path, monkeypatch):
    import scheduler

    monkeypatch.setattr(scheduler.ProactiveScheduler, "CONFIG_FILE", tmp_path / "schedules.json")

    task = scheduler.ScheduledTask(
        id="task1",
        name="Failing prompt task",
        cron="@daily",
        prompt="run failing callback",
    )
    runner = scheduler.ProactiveScheduler()
    runner.tasks[task.id] = task

    async def failing_callback(_prompt):
        raise RuntimeError("planned failure")

    runner.set_maude_callback(failing_callback)
    result = asyncio.run(runner.run_task(task))

    assert "planned failure" in result
    assert task.run_count == 1
    assert task.last_status == "error"
    assert task.failure_count == 1
    assert task.last_error == "planned failure"
    assert task.last_duration_seconds is not None
    assert task.last_run is not None
    if scheduler.CRONITER_AVAILABLE:
        assert task.next_run is not None

    saved = json.loads((tmp_path / "schedules.json").read_text())
    assert saved[0]["last_status"] == "error"
    assert saved[0]["failure_count"] == 1
    if scheduler.CRONITER_AVAILABLE:
        assert saved[0]["next_run"]


def test_mission_schedule_add_status_and_remove(tmp_path, monkeypatch):
    monkeypatch.setenv("MAUDE_MISSIONS_DIR", str(tmp_path / "missions"))

    import scheduler

    monkeypatch.setattr(scheduler.ProactiveScheduler, "CONFIG_FILE", tmp_path / "schedules.json")
    monkeypatch.setattr(scheduler, "_scheduler", None)

    created = execute_tool(
        "mission_create",
        {
            "title": "Scheduled Mission",
            "objective": "Run on a cadence.",
            "steps": ["Checkpoint"],
            "cadence": "daily",
        },
    )
    mission_id = _created_id(created)

    scheduled = execute_tool("mission_schedule", {"id": mission_id})
    assert "Scheduled 'Mission tick: Scheduled Mission'" in scheduled
    assert "Mission schedule saved" in scheduled

    raw = execute_tool("mission_get", {"id": mission_id})
    mission = json.loads(raw)
    task_id = mission["schedule"]["task_id"]
    assert mission["schedule"]["cron"] == "@daily"
    assert f"Mission id: {mission_id}" in mission["schedule"]["prompt"]

    status = execute_tool("mission_schedule", {"id": mission_id, "action": "status"})
    payload = json.loads(status)
    assert payload["scheduler_task"]["id"] == task_id
    assert payload["scheduler_task"]["task_type"] == "mission"
    assert payload["scheduler_task"]["mission_id"] == mission_id
    assert payload["scheduler_task"]["mission_auto_complete"] is True

    brief = execute_tool("mission_brief", {"id": mission_id})
    assert f"- {task_id} | @daily | enabled" in brief

    removed = execute_tool("mission_schedule", {"id": mission_id, "action": "remove"})
    assert "Removed task" in removed

    raw = execute_tool("mission_get", {"id": mission_id})
    mission = json.loads(raw)
    assert mission["schedule"] == {}


def test_mission_status_dashboard_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("MAUDE_MISSIONS_DIR", str(tmp_path))

    created = execute_tool(
        "mission_create",
        {
            "title": "Dashboard Mission",
            "objective": "Show up in Command Center.",
            "steps": ["First", "Second"],
            "cadence": "weekly",
            "artifacts": ["/tmp/report.md"],
        },
    )
    mission_id = _created_id(created)
    execute_tool("mission_update", {"id": mission_id, "steps": [{"id": "s1", "status": "done"}]})
    execute_tool("mission_log", {"id": mission_id, "message": "Waiting on approval", "kind": "blocker"})

    raw = execute_tool("mission_status", {})
    payload = json.loads(raw)
    assert payload["stats"]["total"] == 1
    assert payload["stats"]["active"] == 1
    assert payload["stats"]["blocked"] == 1
    mission = payload["missions"][0]
    assert mission["id"] == mission_id
    assert mission["progress"] == {"done": 1, "total": 2}
    assert mission["next_action"] == "Second"
    assert mission["blockers"] == ["Waiting on approval"]


def test_mission_create_from_template(tmp_path, monkeypatch):
    monkeypatch.setenv("MAUDE_MISSIONS_DIR", str(tmp_path))

    created = execute_tool("mission_create", {"template": "content_engine"})
    mission_id = _created_id(created)

    raw = execute_tool("mission_get", {"id": mission_id})
    mission = json.loads(raw)
    assert mission["template"] == "content_engine"
    assert mission["title"] == "Content Engine"
    assert mission["cadence"] == "daily or weekly"
    assert "Published artifact linked in the mission" in mission["success_criteria"]
    assert mission["steps"][0]["title"] == "Choose topic and angle"
    assert mission["steps"][0]["plan"]


def test_content_engine_template_step_plan_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("MAUDE_MISSIONS_DIR", str(tmp_path / "missions"))

    created = execute_tool("mission_create", {"template": "content_engine"})
    mission_id = _created_id(created)

    result = execute_tool("mission_tick", {"id": mission_id})
    assert "Mission step s1 done." in result

    raw = execute_tool("mission_get", {"id": mission_id})
    mission = json.loads(raw)
    assert mission["steps"][0]["status"] == "done"
    artifact = tmp_path / "missions" / "artifacts" / mission_id / "s1-topic-angle.md"
    assert artifact.exists()
    assert str(artifact) in mission["artifacts"]


def test_recurring_complete_mission_resets_on_tick(tmp_path, monkeypatch):
    monkeypatch.setenv("MAUDE_MISSIONS_DIR", str(tmp_path / "missions"))

    created = execute_tool(
        "mission_create",
        {
            "title": "Recurring Mission",
            "objective": "Run again after completion.",
            "steps": [
                {
                    "title": "Read mission brief",
                    "status": "done",
                    "plan": [[{"name": "mission_brief", "args": {"id": "$MISSION_ID"}}]],
                }
            ],
            "cadence": "daily",
        },
    )
    mission_id = _created_id(created)
    execute_tool("mission_update", {"id": mission_id, "status": "complete"})

    raw = execute_tool("mission_get", {"id": mission_id})
    mission = json.loads(raw)
    mission["recurring"] = True
    path = tmp_path / "missions" / f"{mission_id}.json"
    path.write_text(json.dumps(mission), encoding="utf-8")

    result = execute_tool("mission_tick", {"id": mission_id, "auto_complete": True})
    assert "Mission step s1 done." in result

    raw = execute_tool("mission_get", {"id": mission_id})
    mission = json.loads(raw)
    assert any(log["kind"] == "cycle" for log in mission["logs"])
    assert mission["steps"][0]["status"] == "done"


def test_mission_tools_filter_on_mission_keyword():
    names = {tool["function"]["name"] for tool in get_tools_for_message("start a mission for daily videos")}
    assert "mission_create" in names
    assert "mission_run_next" in names
    assert "mission_tick" in names
    assert "mission_drain" in names
    assert "mission_schedule" in names
    assert "mission_brief" in names
    assert "mission_status" in names
