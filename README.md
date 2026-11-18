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
- 🚧 Phase 4 — From observation to action (detection loop, promotion gate, shadow evaluation, canary deployment, automated rollback, automated retraining, and capacity planning done)

Phase 2 adds a versioned model registry, per-request telemetry, and data-quality
monitoring. Phase 3 makes promotion take effect without a restart, records forecast
outcomes so MAE and RMSE can be reported with explicit coverage, and compares live
feature distributions against a stored reference sample using PSI and KS. Phase 4a runs those checks on a schedule and records breaches as incidents with a
lifecycle. Kern now acts on that evidence, but only within bounded policies: promotion is
gated, rollback reverts a recent promotion after a critical performance incident, and
retraining turns the same labeled traffic into a new candidate without touching production.
A capacity planner forecasts serving demand from recent traffic and recommends how many
replicas the next hour needs, but it never scales anything itself.

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

### Reliability Benchmark
`benchmarks/reliability.py` measures how long Kern takes to detect and recover from
three injected faults: out-of-range inputs (quality), a shifted feature distribution
(drift), and forecasts paired with deliberately wrong outcomes (performance). It drives
only the public API, so it measures the running detection loop rather than a mock.

Start the API with a short evaluation cycle and full feature sampling, and with automatic
rollback and retraining disabled so the benchmark controls when production changes:

```bash
KERN_EVALUATION_INTERVAL=5 \
KERN_FEATURE_SAMPLE_RATE=1.0 \
KERN_ROLLBACK_ENABLED=false \
KERN_RETRAIN_ENABLED=false \
uv run uvicorn src.kern.serving.app:app
```

Then run the benchmark with a reset so each scenario starts from an empty window:

```bash
uv run benchmarks/reliability.py \
  --reset \
  --database-url postgresql://kern:kern@localhost:5432/kern
```

Each scenario records detection time (fault start to the first matching incident) and
recovery time (fault removal to the incident resolving), plus the number of fault and
recovery requests. Quality and drift recover by diluting the bad samples with traffic
drawn from the production model's reference distribution; performance recovers by
correcting the outcomes. `--reset` truncates telemetry, checks, incidents, outcomes, and
the other benchmark-owned tables before each scenario; it is for development databases
only. Without `--reset`, the benchmark refuses to start if quality, drift, or performance
incidents are already open, and the measurement depends on how quiet the last hour was.

The script refuses to start when a shadow or canary is attached, because those requests
would be recorded against a non-production version. It writes
`benchmarks/reliability_results.json` and exits non-zero if a scenario times out.

#### Measured Results

Measurement on Linux 7.0.0-42-generic x86_64, Python 3.12.15, evaluation interval 5s,
full feature sampling, 2s polling, and `--reset` before each scenario:

| Scenario | Fault requests | Detect | Recovery requests | Recover |
|---|---:|---:|---:|---:|
| Quality corruption | 60 | 3.2s | 1,750 | 40.4s |
| Feature drift | 150 | 5.9s | 4,500 | 94.8s |
| Performance decay | 30 | 2.6s | 0 (outcome correction) | 4.0s |

Detection is fast because the evaluation loop runs every 5s. Recovery is dominated by
dilution: an incident clears only when its bad samples fall below the quality threshold
within the one-hour window, so drift takes longer to recover than it does to detect.
Performance is different — it recovers as soon as corrected outcomes outweigh the bad
ones, which took one further evaluation cycle.

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
| POST | `/models/{name}/{version}/promote` | Policy-checked promote and hot-swap; `?force=true` overrides |
| POST | `/models/{name}/{version}/shadow` | Attach a candidate to score live traffic without serving it |
| GET | `/shadow` | Current shadow assignment |
| DELETE | `/shadow` | Detach the shadow |
| GET | `/shadow/report?window=24h` | Agreement and outcome MAE for production vs shadow |
| POST | `/models/{name}/{version}/canary?fraction=0.1` | Serve a fraction of traffic from a candidate |
| GET | `/canary` | Current canary assignment and fraction |
| DELETE | `/canary` | Detach the canary |
| GET | `/canary/report?window=24h` | Served split, fallbacks, and outcome MAE for both versions |
| POST | `/rollback` | Manually restore the previous production version |
| GET | `/rollbacks?limit=20` | Recent rollback decisions, including refusals |
| POST | `/retrain` | Train and register a candidate from labeled live traffic |
| GET | `/retrain/runs?limit=20` | Recent retraining decisions, including skips and failures |
| GET | `/scaling/plan` | Latest capacity plan and its recommendation |
| GET | `/scaling/plans?limit=20` | Recent capacity plans, newest first |
| POST | `/outcomes` | Upsert observed load for a target hour |
| GET | `/telemetry/summary?window=1h` | Request counts, latency percentiles, dropped events |
| GET | `/quality/report?window=1h` | Per-feature data-quality checks |
| GET | `/performance/report?window=24h` | MAE, RMSE, samples, coverage for the serving version |
| GET | `/drift/report?window=1h` | PSI/KS per monitored feature, persisted to `drift_checks` |
| GET | `/incidents?status=open` | Incidents produced by the detection loop |
| POST | `/evaluate` | Run one detection cycle immediately |

`KERN_API_TOKEN`, when set, is required as `Authorization: Bearer <token>` on promotion,
shadow/canary attach and detach, `/rollback`, `/retrain`, `/outcomes`, and `/evaluate`. When
unset they are open and startup logs a warning.

The detection loop evaluates quality, drift, and performance every
`KERN_EVALUATION_INTERVAL` seconds (default 300; `0` disables it) and records breaches as
deduplicated incidents. A healthy cycle resolves them; `insufficient_data` does not.
Data older than `KERN_RETENTION_DAYS` (default 30) is pruned, except open incidents.

Promotion is refused with `409` unless the candidate's holdout MAE improves on production
by at least `KERN_PROMOTION_MIN_IMPROVEMENT` (default 2%) and no critical incident is open
for the model. Every attempt, including refusals and `?force=true` overrides, is written to
`promotion_decisions`. Train a candidate with
`uv run python scripts/train_candidate.py --version <name>`; it registers but does not
promote.

A shadow candidate predicts on every served request without changing the response. Its
predictions land in `shadow_predictions`, and `/shadow/report` pairs them with production
by `request_id`: agreement first, then each model's MAE on the same outcomes once they
arrive. A verdict needs at least `KERN_SHADOW_MIN_SAMPLES` (default 20) paired outcomes.
Promoting the shadowed version detaches it automatically.

A canary serves a candidate to a fraction of requests while production answers the rest.
The fraction must be strictly between 0 and 1: serving everything is a promotion, and
pausing is a detach. A canary failure falls back to production, keeps the response `200`,
and is marked `canary_fallback` and counted in `/canary/report`. Because a canary answer
is a served answer, it lives in `prediction_events` under the canary's version, so the
production quality, drift, and performance reports still describe only what production
served. The report compares disjoint samples and needs `KERN_CANARY_MIN_SAMPLES` (default
20) outcomes on each side before issuing a verdict. Shadow and canary are mutually
exclusive: attaching one clears the other and reports which. Promoting the canary detaches
it automatically.

After each detection cycle, an automated rollback check looks for an open critical
**performance** incident first seen after the current version was promoted. If one exists,
production is restored to the most recently promoted archived version that loads, matches
the feature schema, and has never been rolled back from. Quality and drift incidents never
trigger a rollback: they describe inputs that an older model would also face. Automatic
attempts are limited to versions promoted within `KERN_ROLLBACK_MAX_AGE_HOURS` (default 24)
and to one per `KERN_ROLLBACK_COOLDOWN_MINUTES` (default 60), refusals included.
`KERN_ROLLBACK_ENABLED=false` turns the automation off. Every decision is recorded in
`rollback_events` and exposed by `GET /rollbacks`; `POST /rollback` performs the same move
manually.

When a critical performance incident is open, the same post-cycle hook retrains from
labeled live traffic: forecasts whose target hour is in the past, whose prediction was
recorded before that hour, and whose outcome has arrived. Duplicate targets keep the latest
forecast, the newest `KERN_RETRAIN_HOLDOUT_FRACTION` (default 20%) becomes a temporal
holdout, and the model is registered as an `auto-<timestamp>` candidate. Retraining never
attaches, canaries, or promotes it; shadow, canary, and the promotion gate still decide
deployment. Fewer than `KERN_RETRAIN_MIN_ROWS` (default 100) labeled rows in
`KERN_RETRAIN_WINDOW_DAYS` (default 30) records a skip instead of training.
`KERN_RETRAIN_COOLDOWN_HOURS` (default 24) limits automatic decisions per model, and
`KERN_RETRAIN_ENABLED=false` disables automation. Every decision lands in `retraining_runs`
and is exposed by `GET /retrain/runs`; `POST /retrain` runs the same attempt manually.
Feature snapshots are sampled at `KERN_FEATURE_SAMPLE_RATE` (default 5%), so deployments
that retrain regularly should raise it.

The same post-cycle hook produces an advisory capacity plan. Serving requests from
`prediction_events` are counted into completed `KERN_SCALING_STEP_MINUTES` (default 5)
buckets over `KERN_SCALING_LOOKBACK_MINUTES` (default 60); the current partial bucket is
ignored so a half-finished interval cannot look like a traffic drop. The bucket series is
smoothed with Holt's linear trend and projected `KERN_SCALING_HORIZON_MINUTES` (default 60)
ahead. Peak forecast RPS is sized with
`ceil(rps / (KERN_SCALING_RPS_PER_REPLICA × KERN_SCALING_TARGET_UTILIZATION))`, clamped to
`KERN_SCALING_MIN_REPLICAS` / `KERN_SCALING_MAX_REPLICAS` (defaults 1 / 10). Per-replica
capacity defaults to 90 req/s and utilization to 0.7; both are deployment-specific. Fewer
than `KERN_SCALING_MIN_BUCKETS` (default 3) non-empty buckets yields an `insufficient_data`
plan rather than a guess. Plans are stored in `scaling_plans`, exposed by `/scaling/plan`,
`/scaling/plans`, and the `scaling` field of `/evaluate`, and pruned with retention.
`KERN_SCALING_ENABLED=false` disables planning. The recommendation is advisory: Kern does
not call Kubernetes, Compose, or any process manager.

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
- [x] Candidate training and promotion gate
- [x] Automated retraining
- [x] Shadow evaluation
- [x] Canary deployment
- [x] Automated rollback
- [x] Predictive autoscaling
- [x] Reliability benchmarks
