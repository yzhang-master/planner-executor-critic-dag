from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def __post_init__(self) -> None:
        if self.input_tokens < 0 or self.output_tokens < 0 or self.cost_usd < 0:
            raise ValueError("usage values must be non-negative")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cost_usd + other.cost_usd,
        )


@dataclass(frozen=True)
class ModelResponse:
    text: str
    usage: Usage


class NodeStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass
class NodeResult:
    task_id: str
    status: NodeStatus
    output: str = ""
    attempts: int = 0
    feedback: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "output": self.output,
            "attempts": self.attempts,
            "feedback": self.feedback,
        }


@dataclass
class RunReport:
    objective: str
    status: str
    nodes: dict[str, NodeResult] = field(default_factory=dict)
    usage: Usage = Usage()
    wall_seconds: float = 0.0
    error: str = ""

    @property
    def outputs(self) -> dict[str, str]:
        return {
            task_id: result.output
            for task_id, result in self.nodes.items()
            if result.status == NodeStatus.SUCCESS
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "status": self.status,
            "nodes": {key: value.to_dict() for key, value in self.nodes.items()},
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "total_tokens": self.usage.total_tokens,
                "cost_usd": self.usage.cost_usd,
            },
            "wall_seconds": self.wall_seconds,
            "error": self.error,
        }
