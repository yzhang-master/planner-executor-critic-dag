from __future__ import annotations

import json
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

from .budget import BudgetExhausted, ModelGateway
from .critic import VerifyingCritic
from .dag import Task, TaskDAG
from .types import NodeResult, NodeStatus


class ParallelExecutor:
    def __init__(
        self,
        gateway: ModelGateway,
        critic: VerifyingCritic,
        *,
        workers: int = 4,
        max_retries: int = 2,
        max_output_tokens: int = 700,
    ) -> None:
        if workers <= 0 or max_retries < 0:
            raise ValueError("workers must be positive and max_retries non-negative")
        self.gateway = gateway
        self.critic = critic
        self.workers = workers
        self.max_retries = max_retries
        self.max_output_tokens = max_output_tokens

    def execute(self, dag: TaskDAG) -> dict[str, NodeResult]:
        dag.validate()
        results: dict[str, NodeResult] = {}
        running: dict[Future[NodeResult], str] = {}
        stop = threading.Event()
        order = dag.topological_order()

        def execute_task(task: Task) -> NodeResult:
            dependencies = {
                dependency: results[dependency].output for dependency in task.dependencies
            }
            feedback: list[str] = []
            for attempt in range(1, self.max_retries + 2):
                prompt = (
                    f"[EXECUTE task={task.task_id} attempt={attempt}]\n"
                    f"Task: {task.prompt}\nCriterion: {task.expected}\n"
                    f"Dependency outputs: {json.dumps(dependencies, sort_keys=True)}\n"
                )
                if feedback:
                    prompt += f"Previous critic rejection: {feedback[-1]}\nCorrect the output."
                try:
                    response = self.gateway.complete(
                        prompt, max_tokens=self.max_output_tokens
                    )
                    verdict = self.critic.verify(task, response.text, dependencies)
                except BudgetExhausted:
                    stop.set()
                    return NodeResult(
                        task.task_id,
                        NodeStatus.BUDGET_EXHAUSTED,
                        attempts=attempt,
                        feedback=feedback,
                    )
                if verdict.accepted:
                    return NodeResult(
                        task.task_id,
                        NodeStatus.SUCCESS,
                        response.text,
                        attempt,
                        feedback,
                    )
                feedback.append(verdict.reason)
            return NodeResult(
                task.task_id, NodeStatus.FAILED, response.text, self.max_retries + 1, feedback
            )

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            while len(results) < len(dag.tasks):
                if not stop.is_set():
                    for task_id in order:
                        if len(running) >= self.workers:
                            break
                        if task_id in results or task_id in running.values():
                            continue
                        task = dag.tasks[task_id]
                        dependency_results = [results.get(dep) for dep in task.dependencies]
                        if any(
                            result is not None and result.status != NodeStatus.SUCCESS
                            for result in dependency_results
                        ):
                            results[task_id] = NodeResult(task_id, NodeStatus.SKIPPED)
                            continue
                        if all(
                            result is not None and result.status == NodeStatus.SUCCESS
                            for result in dependency_results
                        ):
                            running[pool.submit(execute_task, task)] = task_id
                if not running:
                    for task_id in order:
                        if task_id not in results:
                            status = (
                                NodeStatus.BUDGET_EXHAUSTED
                                if stop.is_set()
                                else NodeStatus.SKIPPED
                            )
                            results[task_id] = NodeResult(task_id, status)
                    break
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    task_id = running.pop(future)
                    results[task_id] = future.result()
        return results
