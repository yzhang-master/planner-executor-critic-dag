# planner-executor-critic-dag

> **Era:** 2023–2024 · Senior AI Engineer focus area (agent systems). Technique from 2023, reimplemented in 2026.

A dependency-light planner–executor–critic runtime. A model decomposes an objective into a
validated DAG, independent nodes run concurrently, every output is critic-verified, and all
three phases share one strict token/cost ledger. The default test and benchmark path uses
scripted CPU models and makes no network calls.

## Problem

A linear agent cannot express independent subtasks, and an executor that treats model output as
correct can propagate an early error through the rest of a run. Naive per-call budget checks also
oversubscribe a budget when several workers are in flight. This project connects graph validation,
dependency-aware parallel scheduling, corrective retries, and conservative concurrent accounting.

## Method

```text
objective -> planner -> validated DAG
                           |
                ready independent nodes -> thread pool -> executor
                           |                                  |
                           |                            verifying critic
                           |                                  |
                           +------ accepted dependency outputs <+-- reject -> retry
```

- `DAGPlanner` accepts only a bounded JSON task list, validates IDs and dependency types, rejects
  duplicates, unknown edges, self-edges, and cycles, and computes a deterministic topological order.
- `ParallelExecutor` launches only nodes whose dependencies have succeeded. Failures skip their
  descendants; rejected outputs are retried with the critic's correction, up to `max_retries`.
- `VerifyingCritic` sees the criterion, candidate output, and dependency outputs. Invalid critic
  JSON is a retryable rejection rather than silent acceptance.
- `Budget` is shared by planning, execution, and criticism. Before a call it reserves the estimated
  input tokens, maximum output tokens, and their estimated price. Settlement replaces the
  reservation with actual response usage. With `R` reserved and `U` actual usage, a new estimate
  `E` is admitted only when `U + R + E <= limit` for both tokens and cost. Reservations are locked,
  so concurrent workers cannot each spend the same remaining allowance.
- `AnthropicModel` and `OpenAICompatibleModel` are optional lazy adapters for Claude and local
  vLLM-style servers. Importing and testing them needs neither SDKs nor keys; supplied fake clients
  cover their response handling. `ScriptedModel` is deterministic and offline.

## Results

Measured on 2026-11-14 with four local scripted tasks. Success is independently checked against
the expected sink output, not copied from the critic's verdict. Tokens are byte estimates. Costs
are **synthetic accounting units** at the declared price of 1 USD per million input and output
tokens, not provider charges or an LLM-quality benchmark.

| Case | Workers | Success | Nodes | Retries | Tokens | Accounted USD | Seconds |
|---|---:|:---:|---:|---:|---:|---:|---:|
| arithmetic | 1 | yes | 3 | 0 | 501 | 0.000501 | 0.0809 |
| arithmetic | 4 | yes | 3 | 0 | 501 | 0.000501 | 0.0558 |
| release | 1 | yes | 4 | 0 | 727 | 0.000727 | 0.1110 |
| release | 4 | yes | 4 | 0 | 727 | 0.000727 | 0.0560 |
| digest | 1 | yes | 5 | 0 | 818 | 0.000818 | 0.1338 |
| digest | 4 | yes | 5 | 0 | 818 | 0.000818 | 0.0559 |
| critic-retry | 1 | yes | 3 | 1 | 688 | 0.000688 | 0.1059 |
| critic-retry | 4 | yes | 3 | 1 | 688 | 0.000688 | 0.0823 |

Aggregate success was **4/4 serial and 4/4 parallel**. Total accounted cost was **0.002734 USD**
in each mode, with **1.73×** measured wall-clock speedup (0.4316 s serial / 0.2500 s parallel).
The `critic-retry` case demonstrates one actual rejection and recovery. Raw records and the
generator are in `results/local_benchmark.json` and `scripts/benchmark.py`.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check .
pytest
python scripts/benchmark.py
```

The test suite covers topological behavior, atomic graph validation, dependency propagation,
measured parallel speedup, critic rejection/retry, rejection exhaustion, downstream skipping,
malformed critic output, token and cost exhaustion, concurrent reservations, end-to-end reporting,
and both optional adapter contracts. To use a real adapter, install `.[models]` and supply the
SDK credentials or a local OpenAI-compatible endpoint yourself; CI never does so.

## Limitations

- Reservations are intentionally conservative because output length is unknown before a call;
  a run can stop with unused actual-token headroom. Pre-call input tokens and prices are estimates,
  so production use requires adapter-specific tokenizers and current price configuration.
- A critic is only as reliable as its model and criterion. Scripted local success says nothing
  about open-ended model quality. No real-provider quality or spend results are claimed.
- The planner does not repair malformed decomposition JSON, and the runtime does not dynamically
  rewrite the DAG after an unrecoverable node failure.
- Parallelism uses threads in one process; it suits I/O-bound model calls, not distributed execution
  or CPU-bound Python work.

## References

- Yao et al. *ReAct: Synergizing Reasoning and Acting in Language Models.* 2022. arXiv:2210.03629
- Shinn et al. *Reflexion: Language Agents with Verbal Reinforcement Learning.* 2023. arXiv:2303.11366
- Kim et al. *LLMCompiler: An LLM Compiler for Parallel Function Calling.* 2023. arXiv:2312.04511

## License

MIT
