from __future__ import annotations

import json
import re

from .budget import ModelGateway
from .dag import Task, TaskDAG

_TASK_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")


def _json_object(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines[0] not in {"```", "```json"} or lines[-1] != "```":
            raise ValueError("invalid JSON code fence")
        stripped = "\n".join(lines[1:-1])
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("planner response must be a JSON object")
    return value


class DAGPlanner:
    def __init__(self, gateway: ModelGateway, *, max_output_tokens: int = 1200) -> None:
        self.gateway = gateway
        self.max_output_tokens = max_output_tokens

    def decompose(self, objective: str) -> TaskDAG:
        prompt = (
            "[PLAN]\nDecompose the objective into the smallest useful DAG. Return only JSON "
            'with shape {"tasks":[{"id":"task","prompt":"...","dependencies":[],'
            '"expected":"verifiable completion criterion"}]}. Dependencies must refer to '
            f"earlier or later task ids and every non-root objective must have a sink.\n"
            f"Objective: {objective}"
        )
        response = self.gateway.complete(prompt, max_tokens=self.max_output_tokens)
        return self.parse(response.text)

    @staticmethod
    def parse(text: str) -> TaskDAG:
        payload = _json_object(text)
        raw_tasks = payload.get("tasks")
        if not isinstance(raw_tasks, list) or not 1 <= len(raw_tasks) <= 64:
            raise ValueError("planner response must contain 1 to 64 tasks")
        tasks = {}
        for raw in raw_tasks:
            if not isinstance(raw, dict):
                raise ValueError("each task must be an object")
            task_id = raw.get("id")
            prompt = raw.get("prompt")
            dependencies = raw.get("dependencies", [])
            expected = raw.get("expected", "")
            if not isinstance(task_id, str) or not _TASK_ID.fullmatch(task_id):
                raise ValueError(f"invalid task id: {task_id!r}")
            if task_id in tasks:
                raise ValueError(f"duplicate task id: {task_id}")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"task {task_id} has an empty prompt")
            if not isinstance(dependencies, list) or not all(
                isinstance(item, str) for item in dependencies
            ):
                raise ValueError(f"task {task_id} dependencies must be strings")
            if len(set(dependencies)) != len(dependencies) or task_id in dependencies:
                raise ValueError(f"task {task_id} has duplicate or self dependencies")
            if not isinstance(expected, str):
                raise ValueError(f"task {task_id} expected criterion must be a string")
            tasks[task_id] = Task(task_id, prompt, tuple(dependencies), expected)
        dag = TaskDAG(tasks)
        dag.validate()
        return dag
