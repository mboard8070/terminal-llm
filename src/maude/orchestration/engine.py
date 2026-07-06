"""Workflow orchestration facade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .jobs import Job, JobQueue, JobStatus
from .workers import StatelessWorker


@dataclass(frozen=True)
class WorkflowRequest:
    """Normalized work request crossing into the orchestration layer."""

    kind: str
    payload: dict[str, Any]
    background: bool = False


class OrchestrationEngine:
    """Coordinates synchronous execution and queue-based background work."""

    def __init__(self, worker: StatelessWorker, queue: JobQueue | None = None) -> None:
        self.worker = worker
        self.queue = queue or JobQueue()

    def submit(self, request: WorkflowRequest) -> Job:
        job = Job(kind=request.kind, payload=request.payload)
        if request.background:
            return self.queue.enqueue(job)
        completed = self.worker.execute(job)
        return completed

    def run_next(self) -> Job | None:
        job = self.queue.dequeue()
        if job is None:
            return None
        completed = self.worker.execute(job)
        if completed.status == JobStatus.FAILED:
            self.queue.dead_letter(completed)
        else:
            self.queue.complete(completed)
        return completed
