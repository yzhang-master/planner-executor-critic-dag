from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Protocol

from .types import ModelResponse, Usage


def estimate_text_tokens(text: str) -> int:
    return max(1, (len(text.encode("utf-8")) + 3) // 4)


class Model(Protocol):
    def estimate_usage(self, prompt: str, max_output_tokens: int) -> Usage: ...

    def complete(self, prompt: str, *, max_tokens: int) -> ModelResponse: ...


class ScriptedModel:
    """Deterministic model for tests and local benchmarks; never makes network calls."""

    def __init__(
        self,
        responder: Callable[[str, int], str],
        *,
        input_cost_per_million: float = 0.0,
        output_cost_per_million: float = 0.0,
        latency_seconds: float = 0.0,
    ) -> None:
        self.responder = responder
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        self.latency_seconds = latency_seconds
        self.calls: list[str] = []
        self.active_calls = 0
        self.max_active_calls = 0
        self._lock = threading.Lock()

    def _cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_cost_per_million
            + output_tokens * self.output_cost_per_million
        ) / 1_000_000

    def estimate_usage(self, prompt: str, max_output_tokens: int) -> Usage:
        input_tokens = estimate_text_tokens(prompt)
        return Usage(
            input_tokens,
            max_output_tokens,
            self._cost(input_tokens, max_output_tokens),
        )

    def complete(self, prompt: str, *, max_tokens: int) -> ModelResponse:
        with self._lock:
            self.calls.append(prompt)
            call_index = len(self.calls)
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            if self.latency_seconds:
                time.sleep(self.latency_seconds)
            text = self.responder(prompt, call_index)
            input_tokens = estimate_text_tokens(prompt)
            output_tokens = estimate_text_tokens(text)
            if output_tokens > max_tokens:
                raise ValueError("scripted response exceeds max_tokens")
            return ModelResponse(
                text,
                Usage(input_tokens, output_tokens, self._cost(input_tokens, output_tokens)),
            )
        finally:
            with self._lock:
                self.active_calls -= 1
