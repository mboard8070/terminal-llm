"""
Command Center tools — system monitoring, sessions, activity, scheduler, nodes.

Ported from the ai-command-center Next.js app into native MAUDE tools.
"""

import json
import subprocess
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

from tool_registry import register_tool

MAUDE_DATA = Path.home() / ".config" / "maude"


def _run(cmd: list[str], timeout: int = 5) -> str:
    """Run a shell command and return stdout, or empty string on failure."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""


def _get_db():
    """Open memory.db read-only."""
    db_path = MAUDE_DATA / "memory.db"
    if not db_path.exists():
        return None
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


# ── System Stats ─────────────────────────────────────────────

@register_tool("system_stats")
def _dispatch_system_stats(args):
    """CPU, RAM, disk, GPU temperature/utilization/memory."""
    import psutil

    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    result = {
        "cpu_percent": cpu,
        "ram": {
            "used_gb": round(mem.used / (1024**3), 1),
            "total_gb": round(mem.total / (1024**3), 1),
            "percent": mem.percent,
        },
        "disk": {
            "used_gb": round(disk.used / (1024**3), 1),
            "total_gb": round(disk.total / (1024**3), 1),
            "percent": disk.percent,
        },
    }

    # GPU via nvidia-smi (GB10 reports [N/A] for some fields, so parse full output)
    smi = _run([
        "nvidia-smi",
        "--query-gpu=name,temperature.gpu,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ])
    if smi:
        parts = [p.strip() for p in smi.split(",")]
        if len(parts) >= 3:
            result["gpu"] = {
                "name": parts[0],
                "temperature_c": int(parts[1]) if parts[1].isdigit() else parts[1],
                "utilization_percent": int(parts[2]) if parts[2].isdigit() else parts[2],
            }
            if len(parts) >= 4:
                result["gpu"]["power_w"] = parts[3]

    # GPU memory from /proc (more reliable on GB10 unified memory)
    gpu_mem = _run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"])
    if gpu_mem:
        mem_parts = [p.strip() for p in gpu_mem.split(",")]
        if len(mem_parts) >= 2:
            def _parse_mem(s):
                try:
                    return int(s)
                except ValueError:
                    return s
            result.setdefault("gpu", {})["memory_used_mb"] = _parse_mem(mem_parts[0])
            result["gpu"]["memory_total_mb"] = _parse_mem(mem_parts[1])

    return json.dumps(result, indent=2)


# ── GPU Processes ────────────────────────────────────────────

@register_tool("gpu_processes")
def _dispatch_gpu_processes(args):
    """What's currently using the GPU."""
    smi = _run([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ])
    if not smi:
        return "No GPU processes found (or nvidia-smi unavailable)."

    processes = []
    total_used = 0
    for line in smi.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            mem = int(parts[2]) if parts[2].isdigit() else 0
            total_used += mem
            processes.append({
                "pid": parts[0],
                "name": parts[1],
                "memory_mb": mem,
            })

    # Get total GPU memory (GB10 may report [N/A], fall back to 128GB unified)
    total_smi = _run([
        "nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits",
    ])
    try:
        total_mb = int(total_smi.strip())
    except (ValueError, AttributeError):
        total_mb = 131072  # 128GB default for GB10

    return json.dumps({
        "total_mb": total_mb,
        "used_mb": total_used,
        "free_mb": total_mb - total_used,
        "processes": processes,
    }, indent=2)


# ── Memory Browser ───────────────────────────────────────────

@register_tool("memory_browse")
def _dispatch_memory_browse(args):
    """Browse the persistent memory database."""
    conn = _get_db()
    if not conn:
        return "Memory database not found."

    category = args.get("category")
    query = args.get("query")
    limit = min(args.get("limit", 20), 100)

    try:
        if query:
            rows = conn.execute(
                "SELECT key, value, category, updated_at, access_count "
                "FROM memories WHERE key LIKE ? OR value LIKE ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()
        elif category:
            rows = conn.execute(
                "SELECT key, value, category, updated_at, access_count "
                "FROM memories WHERE category = ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (category, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT key, value, category, updated_at, access_count "
                "FROM memories ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

        memories = [dict(r) for r in rows]
        return json.dumps({"count": len(memories), "memories": memories}, indent=2)
    finally:
        conn.close()


# ── Sessions ─────────────────────────────────────────────────

@register_tool("session_list")
def _dispatch_session_list(args):
    """List recent conversation sessions."""
    conn = _get_db()
    if not conn:
        return "Memory database not found."

    limit = min(args.get("limit", 20), 50)

    try:
        rows = conn.execute(
            "SELECT session_id, channel, "
            "MIN(timestamp) as started_at, MAX(timestamp) as last_message_at, "
            "COUNT(*) as message_count "
            "FROM conversations "
            "GROUP BY session_id, channel "
            "ORDER BY last_message_at DESC LIMIT ?",
            (limit,),
        ).fetchall()

        sessions = [dict(r) for r in rows]
        return json.dumps({"count": len(sessions), "sessions": sessions}, indent=2)
    finally:
        conn.close()


# ── Activity Feed ────────────────────────────────────────────

@register_tool("activity_feed")
def _dispatch_activity_feed(args):
    """Recent activity from the chat sync log."""
    log_path = MAUDE_DATA / "chat_sync.jsonl"
    if not log_path.exists():
        return "No activity log found."

    limit = min(args.get("limit", 20), 100)

    try:
        lines = log_path.read_text().strip().split("\n")
        recent = lines[-limit:] if len(lines) > limit else lines
        recent.reverse()  # Most recent first

        activities = []
        for line in recent:
            try:
                entry = json.loads(line)
                activities.append({
                    "channel": entry.get("channel", "?"),
                    "role": entry.get("role", "?"),
                    "content": (entry.get("content", ""))[:200],
                    "timestamp": entry.get("timestamp"),
                })
            except json.JSONDecodeError:
                continue

        return json.dumps({"count": len(activities), "activities": activities}, indent=2)
    except Exception as e:
        return f"Error reading activity log: {e}"


# ── Scheduler Status ─────────────────────────────────────────

@register_tool("scheduler_status")
def _dispatch_scheduler_status(args):
    """Show scheduled tasks and their status."""
    sched_path = MAUDE_DATA / "schedules.json"
    if not sched_path.exists():
        return "No schedules found."

    try:
        tasks = json.loads(sched_path.read_text())

        active = sum(1 for t in tasks if t.get("enabled"))
        total_runs = sum(t.get("run_count", 0) for t in tasks)

        return json.dumps({
            "stats": {"total": len(tasks), "active": active, "total_runs": total_runs},
            "tasks": tasks,
        }, indent=2)
    except Exception as e:
        return f"Error reading schedules: {e}"


# ── Node Status ──────────────────────────────────────────────

@register_tool("node_status")
def _dispatch_node_status(args):
    """Status of connected nodes, services, and channels."""
    nodes = []

    # Check local services via tmux
    tmux_out = _run(["tmux", "list-sessions", "-F", "#{session_name}"])
    tmux_sessions = set(tmux_out.split("\n")) if tmux_out else set()

    maude_running = "maude" in tmux_sessions or bool(
        _run(["pgrep", "-f", "chat_local.py"])
    )
    gateway_running = bool(_run(["pgrep", "-f", "gateway.py"]))
    llama_running = bool(_run(["pgrep", "-f", "llama-server"]))
    telegram_running = bool(_run(["pgrep", "-f", "run_telegram"]))

    nodes.append({
        "name": "spark",
        "type": "hub",
        "status": "online",
        "services": {
            "maude": maude_running,
            "gateway": gateway_running,
            "llama_server": llama_running,
            "telegram": telegram_running,
        },
    })

    # Check Tailscale peers
    ts_json = _run(["tailscale", "status", "--json"])
    if ts_json:
        try:
            ts = json.loads(ts_json)
            for peer_id, peer in ts.get("Peer", {}).items():
                nodes.append({
                    "name": peer.get("HostName", "unknown"),
                    "type": "client",
                    "status": "online" if peer.get("Online") else "offline",
                    "os": peer.get("OS", ""),
                    "ip": peer.get("TailscaleIPs", [None])[0],
                    "last_seen": peer.get("LastSeen", ""),
                })
        except json.JSONDecodeError:
            pass

    # Check heartbeats from remote clients
    hb_path = MAUDE_DATA / "heartbeats.json"
    if hb_path.exists():
        try:
            heartbeats = json.loads(hb_path.read_text())
            for hb in heartbeats if isinstance(heartbeats, list) else heartbeats.values():
                client_id = hb.get("clientId", hb.get("hostname", "unknown"))
                # Skip if already in nodes from tailscale
                if any(n["name"] == client_id for n in nodes):
                    continue
                ts_val = hb.get("timestamp", "")
                nodes.append({
                    "name": client_id,
                    "type": "client",
                    "status": hb.get("status", "unknown"),
                    "platform": hb.get("platform", ""),
                    "version": hb.get("version", ""),
                    "last_seen": ts_val,
                })
        except (json.JSONDecodeError, Exception):
            pass

    return json.dumps({"nodes": nodes}, indent=2)
