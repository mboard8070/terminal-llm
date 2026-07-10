"""
claude_code_tools.py — MAUDE-side orchestration of Claude Code sessions running
on mesh clients.

Execution model: the dispatched SHELL task only LAUNCHES Claude detached on the
target machine (returns in ~2s), with Claude's stdout redirected to a job file.
MAUDE then polls that file with quick follow-up dispatches. This sidesteps the
client's per-command timeout entirely (a `claude -p` run can take as long as it
needs) and leaves the client free to run other tasks — so several workers can
genuinely run in parallel on one machine.

Each "worker" is a named, resumable Claude Code session pinned to a machine +
project directory. Worker registry persists to ~/.maude/claude_workers.json.
"""

import base64
import json
import os
import re
import threading
import time
import uuid

from collab import get_hub

_WORKERS_PATH = os.path.expanduser("~/.maude/claude_workers.json")
_LOCK = threading.Lock()

DEFAULT_TIMEOUT = 300  # seconds to wait for a Claude reply before going async
MAX_TIMEOUT = 3600
DISPATCH_WAIT = 60  # seconds to wait for one quick dispatched command (launch/poll)
POLL_GAP = 5  # pause between reply polls
DEFAULT_PERMISSION_MODE = "acceptEdits"
_VALID_PERMISSION_MODES = {"acceptEdits", "bypassPermissions", "plan", "default"}

_PENDING = "MAUDE_JOB_PENDING"
_MISSING = "MAUDE_JOB_MISSING"


# ── Registry persistence ─────────────────────────────────────────


def _load() -> dict:
    try:
        with open(_WORKERS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save(data: dict):
    os.makedirs(os.path.dirname(_WORKERS_PATH), exist_ok=True)
    tmp = _WORKERS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, _WORKERS_PATH)


def _update_worker(name: str, **fields):
    with _LOCK:
        workers = _load()
        if name in workers:
            workers[name].update(fields)
            _save(workers)


# ── Target platform resolution ───────────────────────────────────


def _worker_platform(hub, worker: dict) -> str:
    """Best-effort platform of the worker's target machine ('windows' default)."""
    if worker.get("target_platform"):
        return worker["target_platform"]
    want_cid = worker.get("target_client_id", "")
    want_host = (worker.get("target") or "").lower()
    for p in hub.presence.get_all():
        if want_cid and p.get("client_id") == want_cid:
            return p.get("platform", "windows")
        if want_host and want_host in (p.get("hostname") or "").lower():
            return p.get("platform", "windows")
    return "windows"


# ── Job script building ──────────────────────────────────────────
# The inner script is shipped base64-encoded, so prompts can contain any
# quotes/newlines without shell-escaping issues.


def _ps_squote(s: str) -> str:
    """Quote a string as a PowerShell single-quoted literal ('' escapes ')."""
    return "'" + (s or "").replace("'", "''") + "'"


def _sh_squote(s: str) -> str:
    """Quote a string as a bash single-quoted literal."""
    return "'" + (s or "").replace("'", "'\\''") + "'"


def _claude_args(worker: dict, prompt: str, resume: bool, quote) -> str:
    parts = ["claude", "-p", quote(prompt), "--output-format", "json"]
    sid = worker.get("session_id")
    if resume and sid:
        parts += ["--resume", sid]
    elif sid:
        parts += ["--session-id", sid]
    pm = worker.get("permission_mode") or DEFAULT_PERMISSION_MODE
    if pm and pm != "default":
        parts += ["--permission-mode", pm]
    if worker.get("model"):
        parts += ["--model", worker["model"]]
    return " ".join(parts)


def _start_command(worker: dict, prompt: str, resume: bool, job: str, platform: str) -> str:
    """One quick command: write the job script, launch it detached, return."""
    if platform == "windows":
        body_lines = []
        if worker.get("cwd"):
            body_lines.append(f"Set-Location {_ps_squote(worker['cwd'])}")
        body_lines.append(
            _claude_args(worker, prompt, resume, _ps_squote) + f' *> "$env:TEMP\\maude_claude\\{job}.out"'
        )
        body_lines.append(f'Set-Content -Path "$env:TEMP\\maude_claude\\{job}.done" -Value $LASTEXITCODE')
        # UTF-8 BOM so Windows PowerShell 5.1 decodes the script correctly
        b64 = base64.b64encode(b"\xef\xbb\xbf" + "\r\n".join(body_lines).encode("utf-8")).decode()
        return (
            f'$null = New-Item -ItemType Directory -Force -Path "$env:TEMP\\maude_claude"; '
            f'[System.IO.File]::WriteAllBytes("$env:TEMP\\maude_claude\\{job}.ps1",'
            f"[System.Convert]::FromBase64String('{b64}')); "
            f"Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',"
            f'"$env:TEMP\\maude_claude\\{job}.ps1" -WindowStyle Hidden; '
            f'Write-Output "STARTED:{job}"'
        )
    # macOS / linux
    body_lines = []
    if worker.get("cwd"):
        body_lines.append(f"cd {_sh_squote(worker['cwd'])}")
    body_lines.append(_claude_args(worker, prompt, resume, _sh_squote) + f" > /tmp/maude_claude/{job}.out 2>&1")
    body_lines.append(f"echo $? > /tmp/maude_claude/{job}.done")
    b64 = base64.b64encode("\n".join(body_lines).encode("utf-8")).decode()
    return (
        f"mkdir -p /tmp/maude_claude && printf '%s' '{b64}' | base64 -d > /tmp/maude_claude/{job}.sh"
        f" && nohup bash /tmp/maude_claude/{job}.sh >/dev/null 2>&1 & echo STARTED:{job}"
    )


def _poll_command(job: str, platform: str) -> str:
    """One quick command: return the reply if done (and clean up), else PENDING."""
    if platform == "windows":
        return (
            f'$b = "$env:TEMP\\maude_claude\\{job}"; '
            f'if (Test-Path "$b.done") {{ $r = Get-Content "$b.out" -Raw; '
            f'Remove-Item "$b.out","$b.done","$b.ps1" -Force -ErrorAction SilentlyContinue; $r }} '
            f"elseif (Test-Path \"$b.ps1\") {{ '{_PENDING}' }} else {{ '{_MISSING}' }}"
        )
    return (
        f'b=/tmp/maude_claude/{job}; if [ -f "$b.done" ]; then cat "$b.out"; rm -f "$b.out" "$b.done" "$b.sh"; '
        f'elif [ -f "$b.sh" ]; then echo {_PENDING}; else echo {_MISSING}; fi'
    )


# ── Dispatch plumbing ────────────────────────────────────────────


def _dispatch_and_wait(hub, worker: dict, command: str, wait_s: int = DISPATCH_WAIT) -> tuple:
    """Dispatch one quick SHELL command and wait for its result.

    Returns (ok, text): ok=False means dispatch/executor failure.
    """
    kwargs = {"capability": "SHELL"}
    if worker.get("target_client_id"):
        kwargs["target_client_id"] = worker["target_client_id"]
    elif worker.get("target_platform"):
        kwargs["target_platform"] = worker["target_platform"]
    else:
        kwargs["target"] = worker.get("target", "")
    task = hub.dispatch_task(command, **kwargs)
    if task.get("status") == "failed":
        return False, task.get("result", "dispatch failed")

    deadline = time.time() + wait_s
    while time.time() < deadline:
        time.sleep(3)
        cur = hub.tasks.get(task["id"])
        if cur and cur.get("status") in ("completed", "failed"):
            return cur.get("status") == "completed", cur.get("result") or ""
    return False, f"no response from client within {wait_s}s (task {task['id']})"


def _start_job(hub, name: str, worker: dict, prompt: str) -> tuple:
    """Launch a detached Claude job. Returns (job_id or None, err)."""
    job = f"cj{uuid.uuid4().hex[:10]}"
    platform = _worker_platform(hub, worker)
    cmd = _start_command(worker, prompt, resume=bool(worker.get("started")), job=job, platform=platform)
    ok, out = _dispatch_and_wait(hub, worker, cmd)
    if not ok or f"STARTED:{job}" not in (out or ""):
        return None, out or "launch failed"
    _update_worker(name, pending={"job": job, "platform": platform, "prompt": prompt[:120], "ts": time.time()})
    return job, None


def _poll_job(hub, worker: dict, job: str, platform: str) -> tuple:
    """One poll. Returns (state, text) where state ∈ done|pending|missing|error."""
    ok, out = _dispatch_and_wait(hub, worker, _poll_command(job, platform))
    if not ok:
        return "error", out
    stripped = (out or "").strip()
    if stripped == _PENDING:
        return "pending", ""
    if stripped == _MISSING:
        return "missing", ""
    return "done", out


def _parse_reply(raw: str) -> dict:
    """Extract Claude's JSON result envelope from the job file contents."""
    raw = (raw or "").strip().lstrip("﻿")
    m = re.search(r"\{.*\}\s*$", raw, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            return {
                "text": obj.get("result") or obj.get("response") or "",
                "session_id": obj.get("session_id"),
                "cost": obj.get("total_cost_usd"),
                "duration_ms": obj.get("duration_ms"),
                "num_turns": obj.get("num_turns"),
                "is_error": bool(obj.get("is_error", False)),
            }
        except Exception:
            pass
    return {"text": raw, "session_id": None, "cost": None, "duration_ms": None, "num_turns": None, "is_error": True}


def _finish_reply(name: str, raw: str) -> str:
    """Parse a finished job's output, update the worker, format the reply."""
    parsed = _parse_reply(raw)
    fields = {"started": True, "pending": None}
    if parsed["session_id"]:
        fields["session_id"] = parsed["session_id"]
    _update_worker(name, **fields)

    meta = []
    if parsed["cost"] is not None:
        meta.append(f"${parsed['cost']:.4f}")
    if parsed["duration_ms"] is not None:
        meta.append(f"{parsed['duration_ms'] / 1000:.1f}s")
    if parsed["num_turns"] is not None:
        meta.append(f"{parsed['num_turns']} turns")
    tag = "ERROR from" if parsed["is_error"] else "Reply from"
    header = f"[{tag} '{name}'" + (f" — {', '.join(meta)}" if meta else "") + "]"
    return f"{header}\n{parsed['text'] or '(no text returned)'}"


def _await_reply(hub, name: str, worker: dict, job: str, platform: str, deadline: float) -> str:
    """Poll until the job finishes or the deadline passes."""
    while time.time() < deadline:
        state, out = _poll_job(hub, worker, job, platform)
        if state == "done":
            return _finish_reply(name, out)
        if state == "missing":
            _update_worker(name, pending=None)
            return f"ERROR: job for '{name}' disappeared on the target (launch may have failed — is `claude` on PATH?)."
        if state == "error":
            return f"ERROR polling '{name}': {out}"
        time.sleep(POLL_GAP)
    return (
        f"'{name}' is still working (Claude keeps running in the background). "
        f"Use claude_check_reply(worker='{name}') to collect the reply when ready."
    )


# ── Tool implementations ─────────────────────────────────────────


def register_worker(args: dict) -> str:
    name = (args.get("name") or "").strip()
    if not name:
        return "ERROR: 'name' is required."
    if not (args.get("target") or args.get("target_client_id") or args.get("target_platform")):
        return "ERROR: specify a machine via target, target_client_id, or target_platform."
    pm = args.get("permission_mode", DEFAULT_PERMISSION_MODE)
    if pm not in _VALID_PERMISSION_MODES:
        return f"ERROR: permission_mode must be one of {sorted(_VALID_PERMISSION_MODES)}."

    with _LOCK:
        workers = _load()
        existing = workers.get(name, {})
        keep = bool(args.get("keep_session"))
        worker = {
            "name": name,
            "cwd": args.get("cwd", existing.get("cwd", "")),
            "target": args.get("target", existing.get("target", "")),
            "target_client_id": args.get("target_client_id", existing.get("target_client_id", "")),
            "target_platform": args.get("target_platform", existing.get("target_platform", "")),
            "permission_mode": pm,
            "model": args.get("model", existing.get("model", "")),
            "session_id": existing.get("session_id") if keep else str(uuid.uuid4()),
            "started": bool(existing.get("started")) if keep else False,
            "pending": existing.get("pending") if keep else None,
        }
        workers[name] = worker
        _save(workers)
    where = worker["target_client_id"] or worker["target_platform"] or worker["target"]
    return (
        f"Registered Claude worker '{name}' → {where}, dir={worker['cwd'] or '(default)'}, "
        f"permission={worker['permission_mode']}, session={worker['session_id'][:8]}."
    )


def ask_claude_code(args: dict) -> str:
    name = (args.get("worker") or "").strip()
    prompt = args.get("prompt") or ""
    if not name or not prompt:
        return "ERROR: 'worker' and 'prompt' are required."
    timeout = min(int(args.get("timeout", DEFAULT_TIMEOUT)), MAX_TIMEOUT)

    workers = _load()
    worker = workers.get(name)
    if not worker:
        return f"ERROR: no worker '{name}'. Register it with claude_register_worker, or list with claude_list_workers."
    if worker.get("pending"):
        return (
            f"'{name}' already has a job in flight ({worker['pending'].get('prompt', '')!r}...). "
            f"Collect it first with claude_check_reply(worker='{name}')."
        )

    hub = get_hub()
    job, err = _start_job(hub, name, worker, prompt)
    if not job:
        return f"Failed to launch Claude on '{name}': {err}"
    platform = _load().get(name, {}).get("pending", {}).get("platform", "windows")
    return _await_reply(hub, name, worker, job, platform, time.time() + timeout)


def check_reply(args: dict) -> str:
    name = (args.get("worker") or "").strip()
    workers = _load()
    worker = workers.get(name)
    if not worker:
        return f"ERROR: no worker '{name}'."
    pending = worker.get("pending")
    if not pending:
        return f"No job in flight for '{name}'."

    hub = get_hub()
    state, out = _poll_job(hub, worker, pending["job"], pending.get("platform", "windows"))
    if state == "done":
        return _finish_reply(name, out)
    if state == "missing":
        _update_worker(name, pending=None)
        return f"Job for '{name}' is gone on the target (finished and collected already, or launch failed)."
    if state == "error":
        return f"ERROR polling '{name}': {out}"
    age = int(time.time() - pending.get("ts", time.time()))
    return f"'{name}' still working ({age}s elapsed) on: {pending.get('prompt', '')!r}. Check again shortly."


def broadcast(args: dict) -> str:
    """Fan the same prompt out to several workers, then gather their replies.

    Jobs launch detached, so all workers genuinely run concurrently — total
    wall time ≈ the slowest worker.
    """
    prompt = args.get("prompt") or ""
    if not prompt:
        return "ERROR: 'prompt' is required."
    timeout = min(int(args.get("timeout", DEFAULT_TIMEOUT)), MAX_TIMEOUT)
    names = args.get("workers")

    workers = _load()
    if names:
        targets = [(n, workers[n]) for n in names if n in workers]
        missing = [n for n in names if n not in workers]
    else:
        targets = list(workers.items())
        missing = []
    if not targets:
        return "ERROR: no matching workers." + (f" Unknown: {missing}" if missing else "")

    hub = get_hub()
    launched = []  # (name, job, platform_or_err)
    for name, w in targets:
        if w.get("pending"):
            launched.append((name, None, "already has a job in flight"))
            continue
        job, err = _start_job(hub, name, w, prompt)
        if job:
            platform = _load().get(name, {}).get("pending", {}).get("platform", "windows")
            launched.append((name, job, platform))
        else:
            launched.append((name, None, err))

    deadline = time.time() + timeout
    lines = [f"Broadcast to {len(targets)} worker(s):"]
    if missing:
        lines.append(f"  (skipped unknown: {', '.join(missing)})")
    for name, job, extra in launched:
        if not job:
            lines.append(f"\n── {name}: LAUNCH FAILED — {extra}")
            continue
        reply = _await_reply(hub, name, _load().get(name, {}), job, extra, deadline)
        lines.append(f"\n── {name}:\n{reply}")
    return "\n".join(lines)


def list_workers(args: dict) -> str:
    workers = _load()
    if not workers:
        return "No Claude workers registered. Use claude_register_worker."
    lines = ["Registered Claude workers:"]
    for name, w in workers.items():
        where = w.get("target_client_id") or w.get("target_platform") or w.get("target") or "?"
        if w.get("pending"):
            state = f"JOB IN FLIGHT ({int(time.time() - w['pending'].get('ts', 0))}s)"
        elif w.get("started"):
            state = "active thread"
        else:
            state = "not started"
        lines.append(
            f"- {name}: {where}, dir={w.get('cwd') or '(default)'}, "
            f"{w.get('permission_mode')}, {state}, session={str(w.get('session_id'))[:8]}"
        )
    return "\n".join(lines)


def reset_worker(args: dict) -> str:
    name = (args.get("worker") or "").strip()
    with _LOCK:
        workers = _load()
        if name not in workers:
            return f"ERROR: no worker '{name}'."
        workers[name]["session_id"] = str(uuid.uuid4())
        workers[name]["started"] = False
        workers[name]["pending"] = None
        _save(workers)
    return f"Reset '{name}' — next message starts a fresh Claude session."


def remove_worker(args: dict) -> str:
    name = (args.get("worker") or "").strip()
    with _LOCK:
        workers = _load()
        if name not in workers:
            return f"ERROR: no worker '{name}'."
        del workers[name]
        _save(workers)
    return f"Removed worker '{name}'."


def fleet_status(args: dict) -> str:
    """Show which machines are online and, for each registered worker, whether
    it is online and idle or currently working."""
    hub = get_hub()
    status = hub.get_status()
    presence = [p for p in status.get("presence", []) if p.get("platform") != "gateway"]
    tasks = status.get("tasks", [])

    now = time.time()
    running_by_client = {}
    for t in tasks:
        if t.get("status") == "running":
            running_by_client.setdefault(t.get("target_client_id", ""), []).append(t)

    online_ids = {p.get("client_id") for p in presence}
    lines = [f"Mesh: {len(presence)} client(s) online.\n"]
    lines.append("Devices:")
    for p in presence:
        cid = p.get("client_id", "")
        age = int(now - p.get("last_seen", now))
        busy = running_by_client.get(cid)
        state = f"WORKING ({len(busy)} task)" if busy else "idle"
        act = f" — {p.get('activity')}" if p.get("activity") else ""
        lines.append(f"- {p.get('hostname')} [{cid[:8]}] ({p.get('platform')}): {state}, seen {age}s ago{act}")

    workers = _load()
    if workers:
        lines.append("\nClaude workers:")
        for name, w in workers.items():
            cid = w.get("target_client_id", "")
            plat = w.get("target_platform", "")
            if cid:
                online = cid in online_ids
            elif plat:
                online = any(p.get("platform") == plat for p in presence)
            else:
                online = any((w.get("target") or "").lower() in (p.get("hostname") or "").lower() for p in presence)
            if w.get("pending"):
                state = f"WORKING ({int(now - w['pending'].get('ts', now))}s on: {w['pending'].get('prompt', '')!r})"
            elif not online:
                state = "OFFLINE"
            else:
                state = "idle / ready"
            lines.append(f"- {name}: {state} (dir={w.get('cwd') or '(default)'})")
    return "\n".join(lines)


# ── Dispatch table ───────────────────────────────────────────────

_HANDLERS = {
    "claude_register_worker": register_worker,
    "ask_claude_code": ask_claude_code,
    "claude_check_reply": check_reply,
    "claude_broadcast": broadcast,
    "claude_list_workers": list_workers,
    "claude_reset_worker": reset_worker,
    "claude_remove_worker": remove_worker,
    "claude_fleet_status": fleet_status,
}


def execute_claude_tool(name: str, arguments: dict) -> str:
    handler = _HANDLERS.get(name)
    if not handler:
        return f"ERROR: unknown claude tool '{name}'."
    try:
        return handler(arguments or {})
    except Exception as e:
        return f"ERROR in {name}: {e}"
