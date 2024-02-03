from concurrent.futures import ThreadPoolExecutor

import pytest

from planner_executor_critic.budget import (
    Budget,
    BudgetExhausted,
    BudgetLimits,
    ModelGateway,
)
from planner_executor_critic.models import ScriptedModel
from planner_executor_critic.types import ModelResponse, Usage


def test_token_budget_blocks_call_before_spending() -> None:
    model = ScriptedModel(lambda prompt, index: "ok")
    gateway = ModelGateway(model, Budget(BudgetLimits(max_tokens=28, max_cost_usd=1.0)))
    gateway.complete("a small prompt", max_tokens=20)
    with pytest.raises(BudgetExhausted, match="token budget"):
        gateway.complete("a small prompt", max_tokens=20)
    assert len(model.calls) == 1
    assert gateway.budget.used.total_tokens <= 28
    assert gateway.budget.reserved_tokens == 0


def test_cost_budget_blocks_first_call_without_key_or_network() -> None:
    model = ScriptedModel(
        lambda prompt, index: "ok",
        input_cost_per_million=10,
        output_cost_per_million=10,
    )
    budget = Budget(BudgetLimits(max_tokens=1000, max_cost_usd=0.00001))
    gateway = ModelGateway(model, budget)
    with pytest.raises(BudgetExhausted, match="cost budget"):
        gateway.complete("this call has a known estimated price", max_tokens=100)
    assert model.calls == []
    assert budget.used == Usage()


def test_parallel_reservations_prevent_oversubscription() -> None:
    model = ScriptedModel(lambda prompt, index: "ok", latency_seconds=0.03)
    budget = Budget(BudgetLimits(max_tokens=60, max_cost_usd=1.0))
    gateway = ModelGateway(model, budget)

    def call() -> str:
        try:
            gateway.complete("parallel", max_tokens=40)
            return "ok"
        except BudgetExhausted:
            return "exhausted"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: call(), range(2)))
    assert sorted(outcomes) == ["exhausted", "ok"]
    assert budget.used.total_tokens <= 60
    assert budget.reserved_tokens == 0


def test_model_overreporting_reservation_cannot_be_settled() -> None:
    class OverreportingModel:
        def estimate_usage(self, prompt: str, max_output_tokens: int) -> Usage:
            return Usage(1, 1)

        def complete(self, prompt: str, *, max_tokens: int) -> ModelResponse:
            return ModelResponse("too much", Usage(1, 100))

    budget = Budget(BudgetLimits(max_tokens=10, max_cost_usd=1.0))
    gateway = ModelGateway(OverreportingModel(), budget)
    with pytest.raises(BudgetExhausted):
        gateway.complete("x", max_tokens=1)
    assert budget.used == Usage()
    assert budget.reserved_tokens == 0
