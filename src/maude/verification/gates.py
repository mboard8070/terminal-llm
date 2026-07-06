"""Verification gates for commits, releases, and runtime side effects."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GateResult:
    """Result from a verification gate."""

    name: str
    passed: bool
    details: str
    evidence: dict[str, Any] = field(default_factory=dict)


VerificationGate = Callable[[], GateResult]


class VerificationSuite:
    """Ordered verification gate runner."""

    def __init__(self, gates: list[VerificationGate] | None = None) -> None:
        self.gates = gates or []

    def add(self, gate: VerificationGate) -> None:
        self.gates.append(gate)

    def run(self) -> list[GateResult]:
        return [gate() for gate in self.gates]

    def passed(self) -> bool:
        return all(result.passed for result in self.run())
