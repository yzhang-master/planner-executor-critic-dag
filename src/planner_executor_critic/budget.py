from __future__ import annotations

import threading
from dataclasses import dataclass
from uuid import uuid4

from .models import Model
from .types import ModelResponse, Usage


class BudgetExhausted(RuntimeError):
    pass


@dataclass(frozen=True)
class BudgetLimits:
    max_tokens: int
    max_cost_usd: float

    def __post_init__(self) -> None:
        if self.max_tokens <= 0 or self.max_cost_usd <= 0:
            raise ValueError("budget limits must be positive")


@dataclass
class _Reservation:
    reservation_id: str
    estimated: Usage
    active: bool = True


class Budget:
    """Thread-safe actual usage plus conservative reservations for in-flight calls."""

    def __init__(self, limits: BudgetLimits) -> None:
        self.limits = limits
        self._used = Usage()
        self._reserved_tokens = 0
        self._reserved_cost = 0.0
        self._calls = 0
        self._lock = threading.Lock()

    def _check(self, tokens: int, cost: float) -> None:
        projected_tokens = self._used.total_tokens + self._reserved_tokens + tokens
        projected_cost = self._used.cost_usd + self._reserved_cost + cost
        if projected_tokens > self.limits.max_tokens:
            raise BudgetExhausted(
                f"token budget exceeded: {projected_tokens} > {self.limits.max_tokens}"
            )
        if projected_cost > self.limits.max_cost_usd + 1e-12:
            raise BudgetExhausted(
                f"cost budget exceeded: {projected_cost:.8f} > {self.limits.max_cost_usd:.8f}"
            )

    def reserve(self, estimated: Usage) -> _Reservation:
        with self._lock:
            self._check(estimated.total_tokens, estimated.cost_usd)
            self._reserved_tokens += estimated.total_tokens
            self._reserved_cost += estimated.cost_usd
            return _Reservation(uuid4().hex, estimated)

    def settle(self, reservation: _Reservation, actual: Usage) -> None:
        with self._lock:
            if not reservation.active:
                raise RuntimeError("reservation is already settled")
            self._reserved_tokens -= reservation.estimated.total_tokens
            self._reserved_cost -= reservation.estimated.cost_usd
            reservation.active = False
            try:
                self._check(actual.total_tokens, actual.cost_usd)
            except BudgetExhausted:
                raise
            self._used = self._used + actual
            self._calls += 1

    def release(self, reservation: _Reservation) -> None:
        with self._lock:
            if reservation.active:
                self._reserved_tokens -= reservation.estimated.total_tokens
                self._reserved_cost -= reservation.estimated.cost_usd
                reservation.active = False

    @property
    def used(self) -> Usage:
        with self._lock:
            return self._used

    @property
    def calls(self) -> int:
        with self._lock:
            return self._calls

    @property
    def reserved_tokens(self) -> int:
        with self._lock:
            return self._reserved_tokens


class ModelGateway:
    def __init__(self, model: Model, budget: Budget) -> None:
        self.model = model
        self.budget = budget

    def complete(self, prompt: str, *, max_tokens: int) -> ModelResponse:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        reservation = self.budget.reserve(self.model.estimate_usage(prompt, max_tokens))
        try:
            response = self.model.complete(prompt, max_tokens=max_tokens)
            self.budget.settle(reservation, response.usage)
            return response
        except Exception:
            self.budget.release(reservation)
            raise
