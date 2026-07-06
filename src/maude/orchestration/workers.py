"""Stateless job workers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from maude.observability import RunContext, emit_event, observed_span, record_metric

from .jobs import Job
from .retries import RetryPolicy, run_with_retries

JobHandler = Callable[[dict[str, Any]], Any]


class StatelessWorker:
    """Executes queued jobs without retaining workflow state between jobs."""

    def __init__(self, handlers: dict[str, JobHandler], retry_policy: RetryPolicy | None = None) -> None:
        self.handlers = handlers
        self.retry_policy = retry_policy or RetryPolicy()

    def execute(self, job: Job) -> Job:
        handler = self.handlers.get(job.kind)
        if handler is None:
            job.mark_failed(f"no handler registered for job kind: {job.kind}")
            return job

        def run_handler() -> Any:
            job.mark_running()
            return handler(job.payload)

        context = RunContext(run_id=job.run_id, trace_id=job.trace_id)
        with observed_span("worker.job", context, job_id=job.job_id, kind=job.kind):
            try:
                result = run_with_retries(run_handler, self.retry_policy)
                job.mark_succeeded(result)
                record_metric("jobs.succeeded")
            except Exception as exc:
                job.mark_failed(str(exc))
                record_metric("jobs.failed")
                emit_event("worker.job.error", context, job_id=job.job_id, kind=job.kind, error=str(exc))
        return job
