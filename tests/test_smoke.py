from planner_executor_critic import TaskDAG
from planner_executor_critic.dag import Task


def test_dag_smoke() -> None:
    dag = TaskDAG()
    dag.add(Task("research", "collect facts"))
    dag.add(Task("write", "write answer", ("research",)))
    assert dag.topological_order() == ["research", "write"]
