import json

import pytest

from planner_executor_critic.dag import Task, TaskDAG
from planner_executor_critic.planner import DAGPlanner


def test_topological_order_is_deterministic() -> None:
    dag = TaskDAG(
        {
            "root": Task("root", "root"),
            "right": Task("right", "right", ("root",)),
            "left": Task("left", "left", ("root",)),
            "sink": Task("sink", "sink", ("right", "left")),
        }
    )
    assert dag.topological_order() == ["root", "left", "right", "sink"]


def test_dag_rejects_cycles_unknown_and_duplicate_dependencies_atomically() -> None:
    with pytest.raises(ValueError, match="cycle"):
        TaskDAG(
            {
                "a": Task("a", "a", ("b",)),
                "b": Task("b", "b", ("a",)),
            }
        ).validate()
    dag = TaskDAG({"a": Task("a", "a")})
    with pytest.raises(ValueError, match="unknown dependency"):
        dag.add(Task("b", "b", ("missing",)))
    assert set(dag.tasks) == {"a"}
    with pytest.raises(ValueError, match="duplicate dependency"):
        dag.add(Task("b", "b", ("a", "a")))


def test_planner_parses_json_dag_and_validates_graph() -> None:
    payload = {
        "tasks": [
            {"id": "sink", "prompt": "combine", "dependencies": ["source"]},
            {"id": "source", "prompt": "find", "expected": "a fact"},
        ]
    }
    dag = DAGPlanner.parse(f"```json\n{json.dumps(payload)}\n```")
    assert dag.topological_order() == ["source", "sink"]
    assert dag.tasks["source"].expected == "a fact"

    payload["tasks"][1]["dependencies"] = ["sink"]
    with pytest.raises(ValueError, match="cycle"):
        DAGPlanner.parse(json.dumps(payload))


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"tasks": []},
        {"tasks": [{"id": "bad id", "prompt": "x"}]},
        {"tasks": [{"id": "a", "prompt": "x", "dependencies": "b"}]},
        {"tasks": [{"id": "a", "prompt": "x"}, {"id": "a", "prompt": "y"}]},
    ],
)
def test_planner_rejects_malformed_decompositions(payload: dict) -> None:
    with pytest.raises(ValueError):
        DAGPlanner.parse(json.dumps(payload))
