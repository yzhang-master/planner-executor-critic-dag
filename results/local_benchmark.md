# Local benchmark

Measured: `2026-11-14T15:57:34.475864+01:00`

Scripted CPU models; tokens are byte estimates and costs use the declared synthetic 
price. These are not real LLM quality or provider-spend measurements.

| Case | Workers | Success | Nodes | Retries | Tokens | Accounted USD | Seconds |
|---|---:|:---:|---:|---:|---:|---:|---:|
| arithmetic | 1 | True | 3 | 0 | 501 | 0.000501 | 0.0809 |
| arithmetic | 4 | True | 3 | 0 | 501 | 0.000501 | 0.0558 |
| release | 1 | True | 4 | 0 | 727 | 0.000727 | 0.1110 |
| release | 4 | True | 4 | 0 | 727 | 0.000727 | 0.0560 |
| digest | 1 | True | 5 | 0 | 818 | 0.000818 | 0.1338 |
| digest | 4 | True | 5 | 0 | 818 | 0.000818 | 0.0559 |
| critic-retry | 1 | True | 3 | 1 | 688 | 0.000688 | 0.1059 |
| critic-retry | 4 | True | 3 | 1 | 688 | 0.000688 | 0.0823 |

Serial success: **4/4**; parallel success: **4/4**.
Measured aggregate parallel speedup: **1.73x** (0.4316s / 0.2500s).
