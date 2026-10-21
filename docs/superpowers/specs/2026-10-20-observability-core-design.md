# Kern Phase 2 — Observability Core

**Date:** 2026-10-20
**Status:** Approved design, pending implementation plan

## Context

Kern is an experimental platform for serving, monitoring, evaluating, and recovering
deployed machine-learning models. Phase 1 delivered a synthetic electricity-load
dataset, an XGBoost baseline, a FastAPI prediction API (single and batch), and serving
benchmarks.

Phase 1 has a structural gap: the API reports `model_version="baseline"` as a hardcoded
string, the model artifact path `models/baseline.json` is hardcoded in the app, and
nothing is recorded about requests at runtime. There is no versioned baseline to compare
against and no live signal to compare, which makes the drift-detection roadmap item
impossible to build meaningfully.

Phase 2 closes that gap. It is the substrate Phase 3 (drift detection) measures against.

A preliminary refactor is already complete: training and serving share a single
`build_features` implementation in `kern/data/features.py`, eliminating train/serve skew.

## Goals

1. Version model artifacts and their metadata, with an unambiguous notion of which
   version is live.
2. Record runtime telemetry for every prediction request without adding latency to the
   inference path.
3. Capture a reference profile of the training data and compare live traffic against it.
4. Serve predictions from the registry instead of a hardcoded file path.

## Non-goals

Explicitly deferred to later phases: drift-detection algorithms, automated retraining,
shadow evaluation, canary deployment, automated rollback, predictive autoscaling.
Phase 2 observes. It does not act.

## Runtime shape

Two services via `docker-compose up -d`:

- `postgres:16` on host port 5432
- `minio` on host ports 9000 (S3 API) and 9001 (console)

Both declare healthchecks and use named volumes. The application fails fast at startup if
either is unreachable, printing the remediation command. This keeps failure explicit
rather than silently degrading to a mode where predictions run but nothing is recorded.

New files: `docker-compose.yml`, `.env.example` (committed), `.env` (already gitignored).

## Architecture

| Module | Responsibility |
| --- | --- |
| `kern.config` | Environment configuration with validated defaults |
| `kern.storage.db` | psycopg3 connection pool; migration application |
| `kern.storage.objects` | S3-compatible artifact upload/download, content-addressed |
| `kern.registry` | Register, promote, and load model versions |
| `kern.telemetry` | Queued background writer; aggregation queries |
| `kern.quality` | Reference profiling; windowed comparison against it |
| `migrations/*.sql` | Numbered, idempotent schema files |

**Driver:** psycopg3 (`psycopg[binary,pool]`). Existing endpoints are synchronous `def`
functions already executed in FastAPI's threadpool, so a synchronous driver requires no
endpoint rewrites. asyncpg is faster but async-only; SQLAlchemy adds machinery three
tables do not justify.

**Migrations:** a small runner applies `migrations/00N_*.sql` in filename order inside a
transaction and records each file's name and SHA-256 in `schema_migrations`. Re-running is
a no-op. Alembic is not used.

**Artifacts:** content-addressed in object storage at `models/<sha256>.json`. The registry
row stores the key and the digest, so artifact integrity is verifiable on load.

## Schema

`gen_random_uuid()` is built into PostgreSQL 13+, so no extension is required.

```sql
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename    text PRIMARY KEY,
    sha256      char(64) NOT NULL,
    applied_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_versions (
    id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name               text NOT NULL,
    version            text NOT NULL,
    stage              text NOT NULL DEFAULT 'candidate'
                       CHECK (stage IN ('candidate', 'production', 'archived')),
    artifact_key       text NOT NULL,
    artifact_sha256    char(64) NOT NULL,
    metrics            jsonb NOT NULL DEFAULT '{}'::jsonb,
    feature_schema     jsonb NOT NULL,
    reference_profile  jsonb NOT NULL,
    created_at         timestamptz NOT NULL DEFAULT now(),
    promoted_at        timestamptz,
    UNIQUE (name, version)
);

CREATE UNIQUE INDEX IF NOT EXISTS one_production_per_model
    ON model_versions (name) WHERE stage = 'production';

CREATE TABLE IF NOT EXISTS prediction_events (
    id               bigserial PRIMARY KEY,
    request_id       uuid NOT NULL,
    model_name       text NOT NULL,
    model_version    text NOT NULL,
    endpoint         text NOT NULL,
    batch_size       integer NOT NULL CHECK (batch_size > 0),
    duration_ms      double precision NOT NULL,
    prediction_min   double precision,
    prediction_mean  double precision,
    prediction_max   double precision,
    features         jsonb,
    recorded_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS prediction_events_version_time
    ON prediction_events (model_version, recorded_at DESC);
CREATE INDEX IF NOT EXISTS prediction_events_time
    ON prediction_events (recorded_at DESC);

CREATE TABLE IF NOT EXISTS quality_checks (
    id            bigserial PRIMARY KEY,
    window_start  timestamptz NOT NULL,
    window_end    timestamptz NOT NULL,
    model_version text NOT NULL,
    feature       text NOT NULL,
    check_type    text NOT NULL
                  CHECK (check_type IN ('null_rate', 'range', 'mean_shift')),
    expected      double precision,
    observed      double precision,
    status        text NOT NULL CHECK (status IN ('pass', 'warn', 'fail')),
    recorded_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS quality_checks_window
    ON quality_checks (window_end DESC);
```

The partial unique index `one_production_per_model` makes "which version is live?"
structurally unambiguous: a second production row for the same model name cannot exist.

## Telemetry write path

The path is designed so that inference never waits on Postgres.

1. A request completes prediction, then calls `recorder.record(event)`.
2. `record` performs a non-blocking `put_nowait` on a bounded `queue.Queue`
   (`KERN_TELEMETRY_QUEUE_SIZE`, default 10,000).
3. A daemon writer thread drains the queue and flushes batches via `executemany`, every
   250 ms or every 100 events, whichever comes first.
4. If the queue is full, the event is dropped and an in-process counter increments. The
   request succeeds regardless.
5. `dropped_events` is exposed in `/telemetry/summary` so a silent flood is visible.

Writer exceptions are logged and do not propagate to request threads. A failed flush
increments a separate `failed_events` counter, which `/telemetry/summary` also reports:
a database outage must not look like zero loss. Both counters are process-lifetime, not
windowed. Feature snapshots containing non-finite values are stripped before enqueue,
because `jsonb` rejects `NaN`/`Infinity` and one bad event would otherwise abort the
whole batch. On interpreter shutdown, the writer flushes remaining events with a
bounded timeout.

**Sampling:** request-level metrics are recorded for every request by default
(`KERN_TELEMETRY_SAMPLE_RATE = 1.0`); lowering that value samples request rows instead.
Independently, the raw `features` snapshot is attached to a smaller fraction of requests
(`KERN_FEATURE_SAMPLE_RATE`, default 0.05). Storing 256 feature dictionaries per request
would inflate the table without improving analysis; a 5% sample is sufficient for
distribution comparison.

Measured Phase 1 throughput is ~68 req/s at batch size 256 and ~97 req/s at batch size 1,
so one row per request is a negligible write load.

## Registry behaviour

- `register(name, version, artifact_path, metrics, feature_schema, reference_profile)`
  computes SHA-256, uploads to `models/<sha256>.json`, and inserts a `candidate` row.
  Re-registering an identical `(name, version)` with an identical digest is a no-op;
  a different digest raises rather than silently overwriting history.
- `promote(name, version)` runs in one transaction: it takes a transaction-scoped
  advisory lock on the model name, moves the current production row to `archived`, and
  promotes the target with `promoted_at = now()`. The lock serializes concurrent
  promotions; the partial unique index guarantees the invariant even if the API call
  fails.
- `load(name)` fetches the production row, downloads the artifact, verifies the digest,
  and returns an `XGBRegressor`.
- `feature_schema` records the exact ordered column list so a mismatch between the
  stored model and the current `build_features` output is detectable at load time rather
  than producing silently wrong predictions. An artifact that stores no feature names is
  rejected rather than loaded unverified.

`window` values are parsed as `<number><unit>` where unit is `s`, `m`, `h`, or `d`, and
are capped at 365 days; anything malformed or oversized returns 400 rather than a 500.

`GET /quality/report` only reads feature snapshots for the currently loaded model
version. An empty window returns `sample_count: 0` and no checks rather than reporting
every feature as null — otherwise a freshly promoted model would raise a full set of
false alarms until the first sampled request arrives. A feature that is absent while
other features are present is still reported as a failing null-rate check.

The FastAPI lifespan loads `production` for the `load-forecast` model and stores the
loaded version string on `app.state`, replacing the hardcoded `models/baseline.json`.

## Data quality

At registration, the training CSV is profiled per feature into `min`, `max`, `mean`,
`std`, `null_rate`, and `p01`, `p25`, `p50`, `p75`, `p99`, stored in `reference_profile`.

`GET /quality/report?window=1h` reads the `features` snapshots within the window and
produces, per feature:

| Check | Rule | warn | fail |
| --- | --- | --- | --- |
| `range` | fraction of values outside [p01, p99] | > 5% | > 20% |
| `null_rate` | observed null fraction | > reference + 1% | > reference + 5% |
| `mean_shift` | standard deviations from reference mean | > 3σ | > 5σ |

Thresholds live in one module-level table so they are easy to inspect and tune. Results
are written to `quality_checks` and returned in the response.

These checks are deliberately simple and transparent. They are the fixed baseline that
Phase 3's drift detection is measured against.

## API surface

| Method | Path | Behaviour |
| --- | --- | --- |
| GET | `/models` | Registered versions with name, version, stage, metrics |
| GET | `/models/{name}/{version}` | Full metadata including reference profile summary |
| POST | `/models/{name}/{version}/promote` | Atomic promotion; 404 if unknown |
| GET | `/telemetry/summary?window=1h` | Request count, throughput, p50/p95/p99 latency, drop count |
| GET | `/quality/report?window=1h` | Per-feature check results |
| GET | `/ready` | Extended to report database and object-store connectivity |

`/predict` and `/predict/batch` report the version actually loaded from the registry
instead of a literal string. Their request and response schemas do not change, so
existing clients and tests keep working.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `KERN_DATABASE_URL` | `postgresql://kern:kern@localhost:5432/kern` | Postgres DSN |
| `KERN_S3_ENDPOINT` | `http://localhost:9000` | MinIO / S3 endpoint |
| `KERN_S3_BUCKET` | `kern-models` | Artifact bucket |
| `KERN_S3_ACCESS_KEY` | `kern` | Access key |
| `KERN_S3_SECRET_KEY` | `kern-secret` | Secret key |
| `KERN_TELEMETRY_SAMPLE_RATE` | `1.0` | Fraction of requests recorded |
| `KERN_FEATURE_SAMPLE_RATE` | `0.05` | Fraction of requests storing feature snapshots |
| `KERN_TELEMETRY_QUEUE_SIZE` | `10000` | Bounded queue capacity |
| `KERN_MODEL_NAME` | `load-forecast` | Registry name the API serves |

Development defaults match the compose file; `.env.example` documents all of them.

## Testing

- Existing 19 tests build `TestClient(app)` and therefore require Postgres. A session
  fixture connects to `KERN_TEST_DATABASE_URL`, which defaults to a `kern_test` database
  on the same server as `KERN_DATABASE_URL`; it creates that database if absent, applies
  migrations, and truncates tables between tests.
- New unit tests cover registry promotion invariants (including the second-production
  rejection), digest verification, sampling behaviour, queue-overflow drop counting, and
  each quality check's pass/warn/fail boundaries.
- A regression test asserts `/predict` reports the promoted version, not `"baseline"`.
- Tests that require no infrastructure are marked `db`; `pytest -m "not db"` runs the
  subset that does not need Docker.

## Risks

| Risk | Mitigation |
| --- | --- |
| Test suite now needs Docker | Dedicated marker; session fixture gives a clear error naming the compose command |
| Telemetry thread dies silently | Writer logs every exception; `/telemetry/summary` exposes last-flush time and drop count |
| Feature snapshot storage growth | 5% sampling; windowed queries read only recent ranges of the timestamp index |
| Model/feature schema drift | `feature_schema` compared at load; mismatch fails loudly |
| Artifact corruption | SHA-256 verified on every download |

## Rollout

1. Add compose file, config, migrations, and storage modules; migrations apply on startup.
2. Register the existing `models/baseline.json` as `load-forecast` version `v1` via
   `scripts/register_baseline.py`; promote it to production.
3. Switch the lifespan to load from the registry; verify parity with the current
   predictions for a fixed input vector.
4. Enable telemetry recording; confirm benchmark throughput is unchanged within noise.
5. Add quality reporting.

Phase 2 is complete when: a prediction request is recorded, visible in
`/telemetry/summary`, compared against the training reference profile in
`/quality/report`, and attributable to a versioned artifact that can be promoted or
superseded.
