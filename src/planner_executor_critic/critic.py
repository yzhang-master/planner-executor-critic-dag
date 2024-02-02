from __future__ import annotations

import json
from dataclasses import dataclass

from .budget import ModelGateway
from .dag import Task


@dataclass(frozen=True)
class Verdict:
    accepted: bool
    reason: str


class VerifyingCritic:
    def __init__(self, gateway: ModelGateway, *, max_output_tokens: int = 160) -> None:
        self.gateway = gateway
        self.max_output_tokens = max_output_tokens

    def verify(
        self, task: Task, output: str, dependency_outputs: dict[str, str]
    ) -> Verdict:
        prompt = (
            f"[CRITIC task={task.task_id}]\n"
            "Verify the output against the task and criterion. Check dependency consistency, "
            'not just fluency. Return only {"accepted":true,"reason":"..."} or '
            '{"accepted":false,"reason":"specific correction"}.\n'
            f"Task: {task.prompt}\nCriterion: {task.expected}\n"
            f"Dependency outputs: {json.dumps(dependency_outputs, sort_keys=True)}\n"
            f"Candidate output: {output}"
        )
        response = self.gateway.complete(prompt, max_tokens=self.max_output_tokens)
        try:
            payload = json.loads(response.text)
        except json.JSONDecodeError as error:
            return Verdict(False, f"critic returned invalid JSON: {error.msg}")
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("accepted"), bool)
            or not isinstance(payload.get("reason"), str)
        ):
            return Verdict(False, "critic response must contain boolean accepted and string reason")
        return Verdict(payload["accepted"], payload["reason"])
