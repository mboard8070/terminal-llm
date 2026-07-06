"""Prompt and model version registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PromptSpec:
    """Versioned prompt/model binding."""

    name: str
    version: str
    template: str
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.name}:{self.version}"


class PromptVersionRegistry:
    """Registry for versioned prompts and their target model bindings."""

    def __init__(self) -> None:
        self._prompts: dict[str, PromptSpec] = {}

    def register(self, spec: PromptSpec) -> None:
        self._prompts[spec.key] = spec

    def get(self, name: str, version: str) -> PromptSpec | None:
        return self._prompts.get(f"{name}:{version}")

    def latest(self, name: str) -> PromptSpec | None:
        matches = [spec for spec in self._prompts.values() if spec.name == name]
        return sorted(matches, key=lambda spec: spec.version)[-1] if matches else None
