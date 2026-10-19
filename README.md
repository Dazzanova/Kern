# Kern

Autonomous ML Model Reliability Platform

Kern is an experimental platform for serving, monitoring,
evaluating, and safely recovering deployed machine-learning models.

The project investigates whether automated drift detection,
model evaluation, canary deployment, and recovery policies can
reduce model incidents while maintaining inference performance.

## Status

🚧 Phase 1 — Baseline model serving

## Batch Inference

Kern supports batch inference to reduce HTTP and Python overhead when processing multiple records.

### API
`POST /predict/batch`
Accepts between 1 and 256 records. Returns predictions in the same order as input.

### How to run
1. Start the API: `uv run uvicorn src.kern.serving.app:app`
2. Run tests: `uv run pytest tests/test_serving.py`
3. Run benchmark: `uv run benchmarks/serving_perf.py`

### Performance Results
Measurements were obtained on Linux 7.0.0-38-generic x86_64 with XGBoost 3.4.1.
Workload: 10,000 records, Concurrency 4, 3 repetitions (median reported).

| Batch Size | P50 Latency (ms) | P95 Latency (ms) | Throughput (Rec/s) | Req/s | Server Time (ms) |
|---|---:|---:|---:|---:|---:|
| 1 | 38.17 | 66.21 | 96.95 | 96.95 | 18.69 |
| 8 | 37.85 | 62.60 | 767.17 | 95.90 | 20.08 |
| 32 | 41.05 | 67.29 | 2,914.63 | 91.08 | 20.67 |
| 64 | 50.40 | 82.25 | 4,981.83 | 77.84 | 24.72 |
| 256 | 55.66 | 81.30 | 17,516.59 | 68.42 | 29.78 |

*Note: Latency refers to the whole HTTP request. Throughput refers to successfully processed prediction records per second. Results are local experimental measurements.*

## Roadmap

- [x] Baseline model and dataset
- [x] Prediction API
- [x] Serving benchmarks
- [ ] Model registry
- [ ] Runtime telemetry
- [ ] Data-quality monitoring
- [ ] Drift detection
- [ ] Automated retraining
- [ ] Shadow evaluation
- [ ] Canary deployment
- [ ] Automated rollback
- [ ] Predictive autoscaling
- [ ] Reliability benchmarks
