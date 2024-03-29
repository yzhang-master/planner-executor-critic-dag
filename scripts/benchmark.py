from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from planner_executor_critic.agent import PlannerExecutorCriticAgent
from planner_executor_critic.budget import Budget, BudgetLimits
from planner_executor_critic.models import ScriptedModel
from planner_executor_critic.types import NodeStatus


@dataclass(frozen=True)
class LocalCase:
    name: str
    objective: str
    tasks: list[dict]
    expected: dict[str, str]
    sink: str
    reject_once: str = ""


def cases() -> list[LocalCase]:
    return [
        LocalCase(
            "arithmetic",
            "Compute two arithmetic results in parallel and add them",
            [
                {"id": "left", "prompt": "Compute 6*7", "expected": "42"},
                {"id": "right", "prompt": "Compute 8+5", "expected": "13"},
                {
                    "id": "sink",
                    "prompt": "Add dependency results",
                    "dependencies": ["left", "right"],
                    "expected": "55",
                },
            ],
            {"left": "42", "right": "13", "sink": "55"},
            "sink",
        ),
        LocalCase(
            "release",
            "Check tests, documentation and package independently, then decide release",
            [
                {"id": f"check_{name}", "prompt": f"Check {name}", "expected": f"{name}:pass"}
                for name in ("tests", "docs", "package")
            ]
            + [
                {
                    "id": "decision",
                    "prompt": "Combine all checks",
                    "dependencies": ["check_tests", "check_docs", "check_package"],
                    "expected": "release:go",
                }
            ],
            {
                "check_tests": "tests:pass",
                "check_docs": "docs:pass",
                "check_package": "package:pass",
                "decision": "release:go",
            },
            "decision",
        ),
        LocalCase(
            "digest",
            "Collect four local facts and produce a digest",
            [
                {"id": f"fact_{index}", "prompt": f"Read fact {index}", "expected": f"fact-{index}"}
                for index in range(4)
            ]
            + [
                {
                    "id": "digest",
                    "prompt": "Combine facts",
                    "dependencies": [f"fact_{index}" for index in range(4)],
                    "expected": "digest:complete",
                }
            ],
            {
                **{f"fact_{index}": f"fact-{index}" for index in range(4)},
                "digest": "digest:complete",
            },
            "digest",
        ),
        LocalCase(
            "critic-retry",
            "Recover from one incorrect local worker output, then combine",
            [
                {"id": "left", "prompt": "Produce left", "expected": "left:good"},
                {"id": "right", "prompt": "Produce right", "expected": "right:good"},
                {
                    "id": "sink",
                    "prompt": "Combine checked outputs",
                    "dependencies": ["left", "right"],
                    "expected": "combined:good",
                },
            ],
            {"left": "left:good", "right": "right:good", "sink": "combined:good"},
            "sink",
            "left",
        ),
    ]


def run_case(case: LocalCase, workers: int) -> dict:
    plan = json.dumps({"tasks": case.tasks})

    def responder(prompt: str, call_index: int) -> str:
        if prompt.startswith("[PLAN]"):
            return plan
        if prompt.startswith("[CRITIC"):
            task_id = prompt.split("task=", 1)[1].split("]", 1)[0]
            expected = case.expected[task_id]
            if f"Candidate output: {expected}" in prompt:
                return '{"accepted":true,"reason":"exact local criterion satisfied"}'
            return '{"accepted":false,"reason":"output differs from local criterion"}'
        task_id = prompt.split("task=", 1)[1].split(" ", 1)[0]
        if case.reject_once == task_id and "attempt=1" in prompt:
            return "incorrect"
        return case.expected[task_id]

    def model(*, latency: float = 0.0) -> ScriptedModel:
        return ScriptedModel(
            responder,
            input_cost_per_million=1.0,
            output_cost_per_million=1.0,
            latency_seconds=latency,
        )

    budget = Budget(BudgetLimits(max_tokens=100_000, max_cost_usd=10.0))
    agent = PlannerExecutorCriticAgent(
        model(), model(latency=0.025), model(), budget, workers=workers, max_retries=1
    )
    report = agent.run(case.objective)
    independent_success = (
        report.status == "success"
        and report.outputs.get(case.sink) == case.expected[case.sink]
        and all(node.status == NodeStatus.SUCCESS for node in report.nodes.values())
    )
    return {
        "case": case.name,
        "workers": workers,
        "success": independent_success,
        "nodes": len(report.nodes),
        "retries": sum(max(0, node.attempts - 1) for node in report.nodes.values()),
        "model_calls": budget.calls,
        "input_tokens": report.usage.input_tokens,
        "output_tokens": report.usage.output_tokens,
        "total_tokens": report.usage.total_tokens,
        "accounted_cost_usd": report.usage.cost_usd,
        "wall_seconds": report.wall_seconds,
    }


def main() -> None:
    runs = [run_case(case, workers) for case in cases() for workers in (1, 4)]
    serial = [run for run in runs if run["workers"] == 1]
    parallel = [run for run in runs if run["workers"] == 4]
    serial_wall = sum(run["wall_seconds"] for run in serial)
    parallel_wall = sum(run["wall_seconds"] for run in parallel)
    result = {
        "measured_at": datetime.now().astimezone().isoformat(),
        "environment": "CPU, deterministic scripted models, no network or provider spend",
        "cost_model": "synthetic accounting at 1 USD per million input and output tokens",
        "runs": runs,
        "summary": {
            "tasks": len(cases()),
            "serial_success": sum(run["success"] for run in serial),
            "parallel_success": sum(run["success"] for run in parallel),
            "serial_cost_usd": sum(run["accounted_cost_usd"] for run in serial),
            "parallel_cost_usd": sum(run["accounted_cost_usd"] for run in parallel),
            "serial_wall_seconds": serial_wall,
            "parallel_wall_seconds": parallel_wall,
            "parallel_speedup": serial_wall / parallel_wall,
        },
    }
    output = Path(__file__).resolve().parents[1] / "results"
    output.mkdir(exist_ok=True)
    (output / "local_benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Local benchmark",
        "",
        f"Measured: `{result['measured_at']}`",
        "",
        "Scripted CPU models; tokens are byte estimates and costs use the declared synthetic ",
        "price. These are not real LLM quality or provider-spend measurements.",
        "",
        "| Case | Workers | Success | Nodes | Retries | Tokens | Accounted USD | Seconds |",
        "|---|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        lines.append(
            f"| {run['case']} | {run['workers']} | {run['success']} | {run['nodes']} | "
            f"{run['retries']} | {run['total_tokens']} | {run['accounted_cost_usd']:.6f} | "
            f"{run['wall_seconds']:.4f} |"
        )
    summary = result["summary"]
    lines += [
        "",
        f"Serial success: **{summary['serial_success']}/{summary['tasks']}**; parallel success: "
        f"**{summary['parallel_success']}/{summary['tasks']}**.",
        f"Measured aggregate parallel speedup: **{summary['parallel_speedup']:.2f}x** "
        f"({serial_wall:.4f}s / {parallel_wall:.4f}s).",
        "",
    ]
    (output / "local_benchmark.md").write_text("\n".join(lines))
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
