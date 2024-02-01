from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Task:
    task_id: str
    prompt: str
    dependencies: tuple[str, ...] = ()
    expected: str = ""


@dataclass
class TaskDAG:
    tasks: dict[str, Task] = field(default_factory=dict)

    def add(self, task: Task) -> None:
        if task.task_id in self.tasks:
            raise ValueError(f"duplicate task id: {task.task_id}")
        self.tasks[task.task_id] = task
        self.validate()

    def validate(self) -> None:
        for task in self.tasks.values():
            for dependency in task.dependencies:
                if dependency not in self.tasks:
                    raise ValueError(f"unknown dependency: {dependency}")
        self.topological_order()

    def topological_order(self) -> list[str]:
        indegree = {task_id: len(task.dependencies) for task_id, task in self.tasks.items()}
        dependents: dict[str, list[str]] = {task_id: [] for task_id in self.tasks}
        for task in self.tasks.values():
            for dependency in task.dependencies:
                dependents[dependency].append(task.task_id)
        ready = sorted(task_id for task_id, degree in indegree.items() if degree == 0)
        order = []
        while ready:
            task_id = ready.pop(0)
            order.append(task_id)
            for dependent in dependents[task_id]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    ready.append(dependent)
            ready.sort()
        if len(order) != len(self.tasks):
            raise ValueError("task graph contains a cycle")
        return order
