# planner-executor-critic-dag

> **Era:** 2023–2024 · Senior AI Engineer focus area (agent systems). Technique from 2023, reimplemented in 2026.

A CPU-only runtime for DAG decomposition, dependency scheduling, parallel execution, critic
verification, and strict token/cost accounting. Implementation and local measurements are
documented below as they are completed.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check .
pytest
```

## License

MIT
