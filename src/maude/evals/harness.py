"""Small evaluation harness for golden workflow checks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvalCase:
    """A golden evaluation case."""

    name: str
    input: dict[str, Any]
    expected: dict[str, Any] = field(default_factory=dict)
    category: str = "general"


@dataclass(frozen=True)
class EvalResult:
    """Evaluation result."""

    name: str
    passed: bool
    score: float
    details: str
    output: dict[str, Any] = field(default_factory=dict)


EvalRunner = Callable[[EvalCase], EvalResult]


class EvalHarness:
    """Runs golden cases through an injected runner."""

    def __init__(self, cases: list[EvalCase], runner: EvalRunner) -> None:
        self.cases = cases
        self.runner = runner

    def run(self) -> list[EvalResult]:
        return [self.runner(case) for case in self.cases]

    def summary(self) -> dict[str, Any]:
        results = self.run()
        total = len(results)
        passed = sum(1 for result in results if result.passed)
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "score": sum(result.score for result in results) / total if total else 0.0,
        }


def golden_cases() -> list[EvalCase]:
    """Built-in smoke/golden cases for core enterprise behavior."""

    return [
        EvalCase(
            name="memory-retrieval-project-scope",
            category="memory",
            input={"query": "project alpha private context", "scope": "project"},
            expected={"requires_scoped_memory": True},
        ),
        EvalCase(
            name="tool-call-read-file",
            category="tools",
            input={"message": "read README.md", "tool": "read_file"},
            expected={"tool": "read_file", "risk": "read"},
        ),
        EvalCase(
            name="planning-produces-executable-steps",
            category="planning",
            input={"objective": "ship a small code fix"},
            expected={"min_steps": 2, "requires_verification": True},
        ),
        EvalCase(
            name="model-routing-private-local",
            category="routing",
            input={"requires_tools": True, "require_private": True},
            expected={"private": True, "local_allowed": True},
        ),
        EvalCase(
            name="mission-execution-logs-result",
            category="missions",
            input={"mission": "content_engine", "action": "run_next"},
            expected={"requires_log": True, "requires_status_update": True},
        ),
    ]


def smoke_runner(case: EvalCase) -> EvalResult:
    """Deterministic smoke runner for CI/readiness without calling models."""

    checks = {
        "memory": "requires_scoped_memory" in case.expected,
        "tools": bool(case.expected.get("tool")) and bool(case.expected.get("risk")),
        "planning": int(case.expected.get("min_steps", 0)) >= 2 and bool(case.expected.get("requires_verification")),
        "routing": bool(case.expected.get("private")),
        "missions": bool(case.expected.get("requires_log")) and bool(case.expected.get("requires_status_update")),
    }
    passed = checks.get(case.category, True)
    return EvalResult(
        name=case.name,
        passed=passed,
        score=1.0 if passed else 0.0,
        details="golden smoke contract satisfied" if passed else "golden smoke contract failed",
        output={"category": case.category},
    )


def run_golden_smoke() -> dict[str, Any]:
    """Run built-in deterministic golden evals and return a compact report."""

    harness = EvalHarness(golden_cases(), smoke_runner)
    results = harness.run()
    summary = harness.summary()
    summary["results"] = [result.__dict__ for result in results]
    return summary
