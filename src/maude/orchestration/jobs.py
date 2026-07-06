"""Durable queue-based job contracts for stateless worker execution."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from maude.config import runtime_paths
from maude.observability import RunContext, emit_event, record_metric


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


class JobPriority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class Job:
    """A durable unit of background work."""

    kind: str
    payload: dict[str, Any]
    priority: JobPriority = JobPriority.NORMAL
    job_id: str = field(default_factory=lambda: uuid4().hex)
    status: JobStatus = JobStatus.QUEUED
    attempts: int = 0
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    result: Any = None
    error: str | None = None
    run_id: str = field(default_factory=lambda: uuid4().hex)
    trace_id: str = field(default_factory=lambda: uuid4().hex)

    def mark_running(self) -> None:
        self.status = JobStatus.RUNNING
        self.attempts += 1
        self.updated_at = _now()
        emit_event("job.running", RunContext(run_id=self.run_id, trace_id=self.trace_id), job_id=self.job_id, kind=self.kind, attempts=self.attempts)

    def mark_succeeded(self, result: Any) -> None:
        self.status = JobStatus.SUCCEEDED
        self.result = result
        self.updated_at = _now()
        emit_event("job.succeeded", RunContext(run_id=self.run_id, trace_id=self.trace_id), job_id=self.job_id, kind=self.kind, attempts=self.attempts)

    def mark_failed(self, error: str) -> None:
        self.status = JobStatus.FAILED
        self.error = error
        self.updated_at = _now()
        emit_event("job.failed", RunContext(run_id=self.run_id, trace_id=self.trace_id), job_id=self.job_id, kind=self.kind, attempts=self.attempts, error=error)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["priority"] = self.priority.value
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Job:
        payload = dict(data)
        payload["priority"] = JobPriority(payload.get("priority", JobPriority.NORMAL))
        payload["status"] = JobStatus(payload.get("status", JobStatus.QUEUED))
        return cls(**payload)


class FileJobStore:
    """JSON-backed durable job store with dead-letter persistence."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else runtime_paths().jobs_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs_path = self.root / "jobs.json"
        self.dead_letter_path = self.root / "dead_letter.json"

    def load_jobs(self) -> list[Job]:
        return self._load(self.jobs_path)

    def save_jobs(self, jobs: list[Job]) -> None:
        self._save(self.jobs_path, jobs)

    def load_dead_letters(self) -> list[Job]:
        return self._load(self.dead_letter_path)

    def dead_letter(self, job: Job) -> None:
        dead = self.load_dead_letters()
        dead = [item for item in dead if item.job_id != job.job_id]
        dead.append(job)
        self._save(self.dead_letter_path, dead)

    def _load(self, path: Path) -> list[Job]:
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return [Job.from_dict(item) for item in data if isinstance(item, dict)]

    def _save(self, path: Path, jobs: list[Job]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([job.to_dict() for job in jobs], indent=2, sort_keys=True), encoding="utf-8")


class JobQueue:
    """Priority queue backed by durable job storage."""

    def __init__(self, store: FileJobStore | None = None) -> None:
        self.store = store or FileJobStore()
        self._queues: dict[JobPriority, deque[Job]] = {
            JobPriority.HIGH: deque(),
            JobPriority.NORMAL: deque(),
            JobPriority.LOW: deque(),
        }
        self._jobs: dict[str, Job] = {}
        self._load()

    def enqueue(self, job: Job) -> Job:
        job.status = JobStatus.QUEUED
        job.updated_at = _now()
        self._queues[job.priority].append(job)
        self._jobs[job.job_id] = job
        self._persist()
        record_metric("queue.depth", float(len(self._jobs)), kind="gauge")
        emit_event("queue.enqueued", RunContext(run_id=job.run_id, trace_id=job.trace_id), job_id=job.job_id, kind=job.kind, priority=job.priority.value)
        return job

    def dequeue(self) -> Job | None:
        for priority in (JobPriority.HIGH, JobPriority.NORMAL, JobPriority.LOW):
            while self._queues[priority]:
                job = self._queues[priority].popleft()
                if self._jobs.get(job.job_id) is job and job.status == JobStatus.QUEUED:
                    self._persist()
                    record_metric("queue.depth", float(len(self._jobs)), kind="gauge")
                    emit_event("queue.dequeued", RunContext(run_id=job.run_id, trace_id=job.trace_id), job_id=job.job_id, kind=job.kind)
                    return job
        return None

    def update(self, job: Job) -> Job:
        self._jobs[job.job_id] = job
        self._persist()
        return job

    def complete(self, job: Job) -> Job:
        self._jobs.pop(job.job_id, None)
        self._persist()
        return job

    def dead_letter(self, job: Job) -> Job:
        self._jobs.pop(job.job_id, None)
        self.store.dead_letter(job)
        self._persist()
        record_metric("queue.dead_letters")
        record_metric("queue.depth", float(len(self._jobs)), kind="gauge")
        emit_event("queue.dead_lettered", RunContext(run_id=job.run_id, trace_id=job.trace_id), job_id=job.job_id, kind=job.kind, error=job.error)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return list(self._jobs.values())

    def dead_letters(self) -> list[Job]:
        return self.store.load_dead_letters()

    def _load(self) -> None:
        for job in self.store.load_jobs():
            if job.status != JobStatus.QUEUED:
                job.status = JobStatus.QUEUED
            self._jobs[job.job_id] = job
            self._queues[job.priority].append(job)

    def _persist(self) -> None:
        self.store.save_jobs(list(self._jobs.values()))
