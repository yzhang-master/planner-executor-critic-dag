import time

from planner_executor_critic.budget import Budget, BudgetLimits, ModelGateway
from planner_executor_critic.critic import VerifyingCritic
from planner_executor_critic.dag import Task, TaskDAG
from planner_executor_critic.executor import ParallelExecutor
from planner_executor_critic.models import ScriptedModel
from planner_executor_critic.types import NodeStatus


def make_executor(
    executor: ScriptedModel,
    critic: ScriptedModel,
    *,
    workers: int = 4,
    retries: int = 2,
) -> ParallelExecutor:
    budget = Budget(BudgetLimits(max_tokens=100_000, max_cost_usd=100.0))
    return ParallelExecutor(
        ModelGateway(executor, budget),
        VerifyingCritic(ModelGateway(critic, budget), max_output_tokens=40),
        workers=workers,
        max_retries=retries,
        max_output_tokens=60,
    )


def accept_critic(prompt: str, index: int) -> str:
    return '{"accepted":true,"reason":"verified"}'


def independent_dag() -> TaskDAG:
    return TaskDAG({f"task-{index}": Task(f"task-{index}", "work") for index in range(4)})


def test_parallel_executor_has_real_speedup_and_dependency_order() -> None:
    serial_model = ScriptedModel(lambda prompt, index: "done", latency_seconds=0.08)
    parallel_model = ScriptedModel(lambda prompt, index: "done", latency_seconds=0.08)
    serial = make_executor(serial_model, ScriptedModel(accept_critic), workers=1)
    parallel = make_executor(parallel_model, ScriptedModel(accept_critic), workers=4)

    started = time.perf_counter()
    serial_results = serial.execute(independent_dag())
    serial_seconds = time.perf_counter() - started
    started = time.perf_counter()
    parallel_results = parallel.execute(independent_dag())
    parallel_seconds = time.perf_counter() - started

    assert all(result.status == NodeStatus.SUCCESS for result in serial_results.values())
    assert all(result.status == NodeStatus.SUCCESS for result in parallel_results.values())
    assert parallel_model.max_active_calls == 4
    assert parallel_seconds < serial_seconds * 0.75

    observed = []

    def ordered_executor(prompt: str, index: int) -> str:
        if "Dependency outputs: {}" not in prompt:
            assert '"source": "source-output"' in prompt
            observed.append("sink-after-source")
        else:
            observed.append("source")
        return "source-output" if "task=source" in prompt else "sink-output"

    dag = TaskDAG(
        {
            "sink": Task("sink", "combine", ("source",)),
            "source": Task("source", "produce"),
        }
    )
    results = make_executor(
        ScriptedModel(ordered_executor), ScriptedModel(accept_critic)
    ).execute(dag)
    assert results["sink"].status == NodeStatus.SUCCESS
    assert observed == ["source", "sink-after-source"]


def test_critic_rejection_retries_with_feedback() -> None:
    def executor(prompt: str, index: int) -> str:
        return "correct answer" if "attempt=2" in prompt else "wrong answer"

    def critic(prompt: str, index: int) -> str:
        if "Candidate output: correct answer" in prompt:
            return '{"accepted":true,"reason":"matches criterion"}'
        return '{"accepted":false,"reason":"use the correct answer"}'

    model = ScriptedModel(executor)
    result = make_executor(model, ScriptedModel(critic)).execute(
        TaskDAG({"only": Task("only", "answer", expected="correct answer")})
    )["only"]
    assert result.status == NodeStatus.SUCCESS
    assert result.output == "correct answer"
    assert result.attempts == 2
    assert result.feedback == ["use the correct answer"]
    assert "Previous critic rejection: use the correct answer" in model.calls[1]


def test_rejection_exhaustion_skips_dependents() -> None:
    executor = ScriptedModel(lambda prompt, index: "bad")
    critic = ScriptedModel(
        lambda prompt, index: '{"accepted":false,"reason":"still wrong"}'
    )
    dag = TaskDAG(
        {
            "first": Task("first", "answer"),
            "dependent": Task("dependent", "use answer", ("first",)),
        }
    )
    results = make_executor(executor, critic, retries=1).execute(dag)
    assert results["first"].status == NodeStatus.FAILED
    assert results["first"].attempts == 2
    assert results["dependent"].status == NodeStatus.SKIPPED
    assert len([call for call in executor.calls if "task=dependent" in call]) == 0


def test_invalid_critic_json_is_a_retryable_rejection() -> None:
    critic = ScriptedModel(lambda prompt, index: "not json")
    result = make_executor(
        ScriptedModel(lambda prompt, index: "candidate"), critic, retries=0
    ).execute(TaskDAG({"task": Task("task", "work")}))["task"]
    assert result.status == NodeStatus.FAILED
    assert "critic returned invalid JSON" in result.feedback[0]
