import json
from types import SimpleNamespace

from planner_executor_critic.adapters import AnthropicModel, OpenAICompatibleModel
from planner_executor_critic.agent import PlannerExecutorCriticAgent
from planner_executor_critic.budget import Budget, BudgetLimits
from planner_executor_critic.models import ScriptedModel
from planner_executor_critic.types import NodeStatus

PLAN = json.dumps(
    {
        "tasks": [
            {"id": "one", "prompt": "first", "expected": "one"},
            {"id": "two", "prompt": "second", "dependencies": ["one"]},
        ]
    }
)


def test_agent_end_to_end_and_budget_report() -> None:
    planner = ScriptedModel(lambda prompt, index: PLAN)
    executor = ScriptedModel(lambda prompt, index: "valid output")
    critic = ScriptedModel(
        lambda prompt, index: '{"accepted":true,"reason":"valid"}'
    )
    budget = Budget(BudgetLimits(max_tokens=20_000, max_cost_usd=1.0))
    report = PlannerExecutorCriticAgent(
        planner, executor, critic, budget, workers=2, max_retries=1
    ).run("local objective")
    assert report.status == "success"
    assert report.outputs == {"one": "valid output", "two": "valid output"}
    assert report.usage.total_tokens > 0
    assert report.usage == budget.used
    assert all(node.status == NodeStatus.SUCCESS for node in report.nodes.values())

    tiny = PlannerExecutorCriticAgent(
        ScriptedModel(lambda prompt, index: PLAN),
        ScriptedModel(lambda prompt, index: "unused"),
        ScriptedModel(lambda prompt, index: "unused"),
        Budget(BudgetLimits(max_tokens=10, max_cost_usd=1.0)),
    ).run("cannot plan")
    assert tiny.status == "budget_exhausted"
    assert tiny.usage.total_tokens <= 10


def test_anthropic_adapter_uses_supplied_client_without_sdk() -> None:
    message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="answer")],
        usage=SimpleNamespace(input_tokens=12, output_tokens=3),
    )
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: message))
    model = AnthropicModel(client=client)
    response = model.complete("prompt", max_tokens=10)
    assert response.text == "answer"
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 3
    assert response.usage.cost_usd == 0.000081


def test_openai_adapter_supports_local_client_and_missing_usage() -> None:
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="local answer"))],
        usage=None,
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: completion)
        )
    )
    model = OpenAICompatibleModel("local-model", client=client)
    response = model.complete("prompt", max_tokens=20)
    assert response.text == "local answer"
    assert response.usage.cost_usd == 0
    assert response.usage.output_tokens > 0
