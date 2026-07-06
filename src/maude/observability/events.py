"""Structured events, trace contexts, and lightweight metrics."""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from maude.config import runtime_paths


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class RunContext:
    """Correlation context carried across gateway, orchestration, tools, and providers."""

    run_id: str = field(default_factory=lambda: uuid4().hex)
    trace_id: str = field(default_factory=lambda: uuid4().hex)
    span_id: str = field(default_factory=lambda: uuid4().hex[:16])
    parent_span_id: str | None = None
    user_id: str | None = None
    client: str | None = None
    workflow_id: str | None = None

    def child(self) -> RunContext:
        """Create a child span context for nested work."""

        return RunContext(
            run_id=self.run_id,
            trace_id=self.trace_id,
            span_id=uuid4().hex[:16],
            parent_span_id=self.span_id,
            user_id=self.user_id,
            client=self.client,
            workflow_id=self.workflow_id,
        )

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> RunContext:
        data = data or {}
        return cls(
            run_id=str(data.get("run_id") or uuid4().hex),
            trace_id=str(data.get("trace_id") or uuid4().hex),
            span_id=str(data.get("span_id") or uuid4().hex[:16]),
            parent_span_id=data.get("parent_span_id"),
            user_id=data.get("user_id"),
            client=data.get("client"),
            workflow_id=data.get("workflow_id"),
        )

    def to_headers(self) -> dict[str, str]:
        return {
            "x-maude-run-id": self.run_id,
            "x-maude-trace-id": self.trace_id,
            "x-maude-span-id": self.span_id,
        }


@dataclass(frozen=True)
class ObservabilityEvent:
    """Single structured runtime event."""

    name: str
    context: RunContext
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "name": self.name,
            "context": asdict(self.context),
            "payload": self.payload,
        }


class EventSink:
    """In-memory event sink used by tests and local mode before external telemetry is configured."""

    def __init__(self) -> None:
        self.events: list[ObservabilityEvent] = []
        self._lock = Lock()

    def emit(self, event: ObservabilityEvent) -> None:
        with self._lock:
            self.events.append(event)

    def for_run(self, run_id: str) -> list[ObservabilityEvent]:
        with self._lock:
            return [event for event in self.events if event.context.run_id == run_id]


class JsonlEventSink(EventSink):
    """Append structured events to a JSONL file while keeping recent events in memory."""

    def __init__(self, path: str | Path | None = None) -> None:
        super().__init__()
        default_path = runtime_paths().logs_dir / "events.jsonl"
        self.path = Path(path or os.environ.get("MAUDE_EVENTS_FILE", default_path)).expanduser()

    def emit(self, event: ObservabilityEvent) -> None:
        super().emit(event)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")


@dataclass(frozen=True)
class MetricSnapshot:
    """Point-in-time metrics summary."""

    counters: dict[str, float]
    gauges: dict[str, float]
    timings: dict[str, dict[str, float]]


class MetricsRegistry:
    """Small process-local metric registry for readiness and diagnostics."""

    def __init__(self) -> None:
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._timings: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def increment(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] += value

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def timing(self, name: str, seconds: float) -> None:
        with self._lock:
            self._timings[name].append(seconds)

    def snapshot(self) -> MetricSnapshot:
        with self._lock:
            timings = {}
            for name, values in self._timings.items():
                count = len(values)
                timings[name] = {
                    "count": float(count),
                    "avg": sum(values) / count if count else 0.0,
                    "max": max(values) if values else 0.0,
                }
            return MetricSnapshot(dict(self._counters), dict(self._gauges), timings)


_sink: EventSink = JsonlEventSink()
_metrics = MetricsRegistry()


def configure_observability(sink: EventSink | None = None, metrics: MetricsRegistry | None = None) -> None:
    """Override global observability backends, mainly for tests."""

    global _sink, _metrics
    if sink is not None:
        _sink = sink
    if metrics is not None:
        _metrics = metrics


def emit_event(name: str, context: RunContext | None = None, **payload: Any) -> ObservabilityEvent:
    ctx = context or RunContext()
    event = ObservabilityEvent(name=name, context=ctx, payload=payload)
    _sink.emit(event)
    _metrics.increment(f"events.{name}")
    return event


def record_metric(name: str, value: float = 1.0, *, kind: str = "counter") -> None:
    if kind == "counter":
        _metrics.increment(name, value)
    elif kind == "gauge":
        _metrics.gauge(name, value)
    elif kind == "timing":
        _metrics.timing(name, value)
    else:
        raise ValueError(f"unknown metric kind: {kind}")


def metrics_snapshot() -> MetricSnapshot:
    return _metrics.snapshot()


class observed_span:
    """Context manager that emits start/end events and latency metrics."""

    def __init__(self, name: str, context: RunContext | None = None, **payload: Any) -> None:
        self.name = name
        self.context = (context or RunContext()).child()
        self.payload = payload
        self.started = 0.0

    def __enter__(self) -> RunContext:
        self.started = time.monotonic()
        emit_event(f"{self.name}.started", self.context, **self.payload)
        return self.context

    def __exit__(self, exc_type, exc, _tb) -> bool:
        elapsed = time.monotonic() - self.started
        record_metric(f"{self.name}.latency_seconds", elapsed, kind="timing")
        if exc is None:
            emit_event(f"{self.name}.completed", self.context, latency_seconds=round(elapsed, 6), **self.payload)
        else:
            record_metric(f"{self.name}.failures")
            emit_event(
                f"{self.name}.failed",
                self.context,
                latency_seconds=round(elapsed, 6),
                error=str(exc),
                **self.payload,
            )
        return False
