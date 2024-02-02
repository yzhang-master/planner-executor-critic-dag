from __future__ import annotations

import time

from .budget import Budget, BudgetExhausted, ModelGateway
from .critic import VerifyingCritic
from .executor import ParallelExecutor
from .models import Model
from .planner import DAGPlanner
from .types import NodeStatus, RunReport


class PlannerExecutorCriticAgent:
    def __init__(
        self,
        planner_model: Model,
        executor_model: Model,
        critic_model: Model,
        budget: Budget,
        *,
        workers: int = 4,
        max_retries: int = 2,
    ) -> None:
        self.budget = budget
        self.planner = DAGPlanner(ModelGateway(planner_model, budget))
        self.executor = ParallelExecutor(
            ModelGateway(executor_model, budget),
            VerifyingCritic(ModelGateway(critic_model, budget)),
            workers=workers,
            max_retries=max_retries,
        )

    def run(self, objective: str) -> RunReport:
        started = time.perf_counter()
        try:
            dag = self.planner.decompose(objective)
        except BudgetExhausted as error:
            return RunReport(
                objective,
                "budget_exhausted",
                usage=self.budget.used,
                wall_seconds=time.perf_counter() - started,
                error=str(error),
            )
        nodes = self.executor.execute(dag)
        if all(node.status == NodeStatus.SUCCESS for node in nodes.values()):
            status = "success"
        elif any(node.status == NodeStatus.BUDGET_EXHAUSTED for node in nodes.values()):
            status = "budget_exhausted"
        else:
            status = "failed"
        return RunReport(
            objective,
            status,
            nodes,
            self.budget.used,
            time.perf_counter() - started,
        )
