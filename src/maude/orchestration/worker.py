"""Process entry point for draining durable orchestration jobs."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .engine import OrchestrationEngine
from .jobs import FileJobStore, JobQueue
from .tool_execution import execute_tool
from .workers import JobHandler, StatelessWorker


def _echo_handler(payload: dict[str, Any]) -> Any:
    return payload.get("message", payload)


def _tool_handler(payload: dict[str, Any]) -> str:
    name = str(payload.get("name") or "")
    arguments = payload.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise ValueError("tool job payload 'arguments' must be an object")
    return execute_tool(name, arguments)


def build_default_worker(extra_handlers: dict[str, JobHandler] | None = None) -> StatelessWorker:
    """Build the default stateless worker used by the process entry point."""

    handlers: dict[str, JobHandler] = {
        "echo": _echo_handler,
        "tool": _tool_handler,
    }
    if extra_handlers:
        handlers.update(extra_handlers)
    return StatelessWorker(handlers)


def run_worker_once(jobs_dir: str | Path | None = None, worker: StatelessWorker | None = None) -> dict[str, Any]:
    """Drain one queued job and return a compact execution summary."""

    queue = JobQueue(FileJobStore(jobs_dir) if jobs_dir is not None else None)
    engine = OrchestrationEngine(worker or build_default_worker(), queue)
    job = engine.run_next()
    if job is None:
        return {"status": "idle", "job_id": None}
    return {
        "status": job.status.value,
        "job_id": job.job_id,
        "kind": job.kind,
        "attempts": job.attempts,
        "error": job.error,
    }


def run_worker_loop(
    jobs_dir: str | Path | None = None,
    worker: StatelessWorker | None = None,
    poll_seconds: float = 2.0,
) -> None:
    """Continuously drain durable jobs without keeping workflow state in memory."""

    active_worker = worker or build_default_worker()
    while True:
        result = run_worker_once(jobs_dir=jobs_dir, worker=active_worker)
        if result["status"] == "idle":
            time.sleep(poll_seconds)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Drain MAUDE durable orchestration jobs.")
    parser.add_argument("--jobs-dir", help="Directory containing jobs.json and dead_letter.json")
    parser.add_argument("--once", action="store_true", help="Process at most one queued job and exit")
    parser.add_argument("--poll-seconds", type=float, default=2.0, help="Polling interval for daemon mode")
    args = parser.parse_args(argv)

    if args.once:
        print(json.dumps(run_worker_once(args.jobs_dir), sort_keys=True))
        return 0

    run_worker_loop(args.jobs_dir, poll_seconds=args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
