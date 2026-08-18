# Feature A/B: topology_affinity_score (10 seeds, xgboost)

## Workload Specification

| Field | Value |
|---|---|
| name | kermit-pd-topology-h200 |
| trace_type | real |
| model | Qwen/Qwen3-32B |
| gpu | NVIDIA H200 (141GB) |
| gpus_per_replica | 1 |
| platform | CoreWeave CKS (Kermit), Kubernetes |
| stack | llm-d-router EPP (topology-extractor, topology-affinity-filter/scorer, predicted-latency-producer) + vLLM P/D disaggregated (NixlConnector, kv_both) |
| max_model_len | 8192 |
| prefix_caching | enabled |
| generator | inference-perf |
| profile | ShareGPT + long-synthetic (2k-8k tokens) + shared-prefix |
| api | completion (streaming) |
| load_stages | 1 QPS x 300s, 2 QPS x 300s, 4 QPS x 600s |

## Trace Profile

| field | min | max | median | std | unique | % zero |
|---|---|---|---|---|---|---|
| input_token_length | 8.0 | 7542.0 | 1942.0 | 2079.0 | 1082 | 0.0% |
| num_request_running | 0.0 | 124.0 | 11.0 | 34.91 | 119 | 15.5% |
| kv_cache_percentage | 0.0 | 1.0 | 0.01 | 0.35 | 662 | 47.9% |
| prefix_cache_score | 0.0 | 1.0 | 0.0 | 0.05 | 3 | 99.7% |
| actual_ttft_ms | 0.0 | 11060.0 | 0.0 | 2061.82 | 584 | 51.2% |
| actual_tpot_ms | 0.0 | 546.97 | 19.0 | 87.69 | 764 | 48.8% |

> **WARNING:** prefix cache score has only 3 unique values (range 0.0-1.0). Features depending on prefix_cache_score variation cannot show signal.
> **WARNING:** prefix cache score is zero in 99.7% of samples. Features using prefix_cache_score will see near-constant input.

## Contention Gate

**PASSED**: 76.87% of 1500 samples contended (max num_request_running = 124)

## Self-Test: PASS

## Results

| target | metric | without | with | delta | > seed std? |
|---|---|---|---|---|---|
| ttft | quantile_loss | 154.5726 +/- 35.0346 | 155.5027 +/- 31.5653 | 0.6% | NO -- within noise |
| ttft | coverage_pct | 87.3973 +/- 3.8649 | 86.8493 +/- 3.4219 | -0.63% | NO -- within noise |
| ttft | mape_pct | 238.4311 +/- 48.3269 | 236.7062 +/- 51.1324 | -0.72% | NO -- within noise |
| tpot | quantile_loss | 8.5633 +/- 2.4771 | 8.5633 +/- 2.4771 | 0.0% | NO -- within noise |
| tpot | coverage_pct | 86.2338 +/- 4.0739 | 86.2338 +/- 4.0739 | 0.0% | NO -- within noise |
| tpot | mape_pct | 86.2136 +/- 16.0185 | 86.2136 +/- 16.0185 | 0.0% | NO -- within noise |

