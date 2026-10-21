# Kern

Autonomous ML Model Reliability Platform

Kern is an experimental platform for serving, monitoring,
evaluating, and safely recovering deployed machine-learning models.

The project investigates whether automated drift detection,
model evaluation, canary deployment, and recovery policies can
reduce model incidents while maintaining inference performance.

## Status

- ✅ Phase 1 — Baseline model serving
- ✅ Phase 2 — Observability core
- ✅ Phase 3 — Close the loop and drift detection
- 🚧 Phase 4 — From observation to action (detection loop done)

Phase 2 adds a versioned model registry, per-request telemetry, and data-quality
monitoring. Phase 3 makes promotion take effect without a restart, records forecast
outcomes so MAE and RMSE can be reported with explicit coverage, and compares live
feature distributions against a stored reference sample using PSI and KS. Phase 4a
runs those checks on a schedule and records breaches as incidents with a lifecycle.

Kern still observes; it does not act on what it observes.

## Batch Inference

Kern supports batch inference to reduce HTTP and Python overhead when processing multiple records.

### API
`POST /predict/batch`
Accepts between 1 and 256 records. Returns predictions in the same order as input.

### How to run
The API loads its model from the registry, so start the infrastructure first — see
[Observability](#observability).

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

## Observability

Kern versions models in a registry, records per-request telemetry, and compares live
traffic against a reference profile of the training data.

### Infrastructure
```bash
docker-compose up -d
cp .env.example .env
uv run python scripts/register_baseline.py
uv run uvicorn src.kern.serving.app:app
```

### Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | `/models` | Registered versions and stages |
| GET | `/models/{name}/{version}` | Full version metadata |
| POST | `/models/{name}/{version}/promote` | Load, verify, promote, and hot-swap |
| POST | `/outcomes` | Upsert observed load for a target hour |
| GET | `/telemetry/summary?window=1h` | Request counts, latency percentiles, dropped events |
| GET | `/quality/report?window=1h` | Per-feature data-quality checks |
| GET | `/performance/report?window=24h` | MAE, RMSE, samples, coverage for the serving version |
| GET | `/drift/report?window=1h` | PSI/KS per monitored feature, persisted to `drift_checks` |
| GET | `/incidents?status=open` | Incidents produced by the detection loop |
| POST | `/evaluate` | Run one detection cycle immediately |

`KERN_API_TOKEN`, when set, is required as `Authorization: Bearer <token>` on promotion,
`/outcomes`, and `/evaluate`. When unset they are open and startup logs a warning.

The detection loop evaluates quality, drift, and performance every
`KERN_EVALUATION_INTERVAL` seconds (default 300; `0` disables it) and records breaches as
deduplicated incidents. A healthy cycle resolves them; `insufficient_data` does not.
Data older than `KERN_RETENTION_DAYS` (default 30) is pruned, except open incidents.

Batch telemetry carries one feature snapshot for up to 256 rows, so batch predictions do
not record a `target_ts` and are excluded from `/performance/report`. Drift checks skip
calendar features (`hour`, `month`, …) because they are deterministic given the target
timestamp and would alarm on the clock rather than on the input.

### Tests
`uv run pytest` requires the compose stack. `uv run pytest -m "not db"` runs only the
tests that need no infrastructure.

## Roadmap

- [x] Baseline model and dataset
- [x] Prediction API
- [x] Serving benchmarks
- [x] Model registry
- [x] Runtime telemetry
- [x] Data-quality monitoring
- [x] Drift detection
- [ ] Automated retraining
- [ ] Shadow evaluation
- [ ] Canary deployment
- [ ] Automated rollback
- [ ] Predictive autoscaling
- [ ] Reliability benchmarks
