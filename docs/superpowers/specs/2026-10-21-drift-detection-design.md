# Kern Phase 3 — Drift Detection

## Context

Phase 1 delivered a synthetic electricity-load forecaster behind a FastAPI service and a
batch endpoint. Phase 2 delivered the substrate: a versioned model registry with
content-addressed artifacts, non-blocking request telemetry, and per-feature data-quality
checks measured against a reference profile of the training data.

Phase 2 observes, but it cannot answer the project's central question: *is the model
getting worse?* Two gaps prevent it.

1. **Promotion is not real.** `POST /models/{name}/{version}/promote` updates Postgres,
   but the running process keeps serving the version it loaded at startup. A human has to
   restart the service. Tolerance for this ends the moment a policy acts on a promotion.
2. **There is no ground truth.** `prediction_events` records what was predicted, never
   what actually happened. Feature drift and prediction drift are detectable, but
   performance decay — the only signal that speaks to model incidents — is not.

Phase 3 closes both gaps and adds drift detection on top.

## Goals

1. Make promotion take effect in the running process without a restart.
2. Protect the promotion and ingestion paths with a shared token when one is configured.
3. Ingest observed outcomes that arrive after the fact, and attribute them to the
   prediction they belong to.
4. Report rolling MAE and RMSE per model version, with explicit coverage.
5. Detect feature drift by comparing live traffic against a stored reference sample.
6. Persist drift results so detection has a history, not just a current value.

## Non-goals

Explicitly deferred to Phase 3b: automated retraining, shadow evaluation, canary
deployment, automated rollback, predictive autoscaling. Phase 3a still observes; it does
not act. No policy consumes a drift signal in this phase.

Also out of scope: multi-tenant isolation, per-customer reference profiles, outcome
ingestion from files or object storage, and drift detection on categorical features (all
current features are numeric).

## Runtime shape

```
client ──> POST /predict ──> model (in-memory ModelState)
                 │                    ▲
                 │                    │ atomic swap
                 └─> telemetry queue  │
                                      │
POST /outcomes ──> observed_outcomes  │
                                      │
GET /drift/report ──> samples vs reference_sample ──> drift_checks
GET /performance/report ──> predictions JOIN outcomes
POST /models/{n}/{v}/promote ──> load+verify ──> DB commit ──> swap
```

The inference path does not change. Outcome ingestion is a separate write path, and both
report endpoints are read-only with the same persistence pattern `/quality/report`
already uses.

## Architecture

| Module | Responsibility |
| --- | --- |
| `src/kern/drift.py` | Pure PSI and KS functions. No database, no I/O. |
| `src/kern/quality.py` | Unchanged. Adds `reference_sample`, not a new file, because it owns profiling. |
| `src/kern/registry.py` | Gains `load_version`, `reference_sample` persistence, and backfill support. |
| `src/kern/performance.py` | Pure metrics from paired (prediction, actual) rows. |
| `src/kern/serving/app.py` | New endpoints, `ModelState` swap, token guard. |
| `scripts/backfill_reference_sample.py` | One-off derived-data backfill for already-registered versions. |

Splitting `drift.py` and `performance.py` from `app.py` keeps the statistics testable
without a database, which is how `quality.py` was built.

## Schema

Migration `migrations/002_drift.sql`. It is additive and safe to apply to a populated
database; no existing column changes type and no row is rewritten.

```sql
ALTER TABLE prediction_events
    ADD COLUMN IF NOT EXISTS target_ts timestamptz;

CREATE INDEX IF NOT EXISTS prediction_events_version_target_idx
    ON prediction_events (model_version, target_ts)
    WHERE target_ts IS NOT NULL;

ALTER TABLE model_versions
    ADD COLUMN IF NOT EXISTS reference_sample jsonb;

CREATE TABLE IF NOT EXISTS observed_outcomes (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name   text NOT NULL,
    target_ts    timestamptz NOT NULL,
    actual_mw    double precision NOT NULL,
    source       text NOT NULL DEFAULT 'api',
    observed_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (model_name, target_ts)
);

CREATE TABLE IF NOT EXISTS drift_checks (
    id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    window_start   timestamptz NOT NULL,
    window_end     timestamptz NOT NULL,
    model_version  text NOT NULL,
    feature        text NOT NULL,
    method         text NOT NULL,
    statistic      double precision,
    p_value        double precision,
    sample_size    integer NOT NULL,
    status         text NOT NULL,
    created_at     timestamptz NOT NULL DEFAULT now(),
    CHECK (method IN ('psi', 'ks')),
    CHECK (status IN ('pass', 'warn', 'fail', 'insufficient_data'))
);

CREATE INDEX IF NOT EXISTS drift_checks_lookup_idx
    ON drift_checks (model_version, feature, created_at DESC);
```

`observed_outcomes` is keyed by `(model_name, target_ts)` rather than by request, because
load is a property of an hour: many predictions may target the same timestamp, and the
actual value arrives once. Upsert on conflict overwrites `actual_mw` and refreshes
`observed_at`, which handles late corrections and revised meter readings.

`reference_sample` is a separate column, not a key inside `reference_profile`. The profile
is a flat `dict[str, float]` consumed by `evaluate_feature`; nesting a list inside it would
break that contract for every existing caller.

## Promotion and hot reload

`load_version(pool, store, name, version)` loads a specific version: fetch the row,
download the artifact, verify digest, verify feature schema and ordering, construct the
`XGBRegressor`. `load_production` becomes a lookup of the production version followed by
`load_version`.

Promotion order is deliberate:

1. Load and verify the target version.
2. Promote in Postgres (advisory lock, archive current, set target production).
3. Swap the in-memory state.

If step 1 fails, the database is untouched — a version that cannot serve is never
promoted. If step 3 fails after a successful commit (it should not, since the artifact is
already in memory), the database remains the source of truth and the error is logged with
the version string; the next restart converges.

The swap replaces one object on `app.state`:

```python
@dataclass(frozen=True)
class ModelState:
    name: str
    version: str
    model: XGBRegressor
    reference_profile: dict[str, dict[str, float]]
    reference_sample: dict[str, list[float]]
```

Handlers capture `state = request.app.state.current` once at the top. Because the object is
frozen and replaced wholesale, a request that started under version A finishes under
version A even if a promotion lands mid-request. No lock is needed on the read path.

## Authentication

`KERN_API_TOKEN` is an optional setting, empty by default.

- When set, `POST /models/{name}/{version}/promote` and `POST /outcomes` require
  `Authorization: Bearer <token>`, compared with `secrets.compare_digest`.
- A missing or wrong token returns 401. The comparison is constant-time in the sense that
  it does not short-circuit on content.
- When unset, the endpoints are open and startup logs a single warning naming the
  endpoints that are unauthenticated.

This is deliberately a shared token rather than a user system: there is one operator and
no consumer identity in this phase, and a user table would be security theater.

## Outcomes and performance

`POST /outcomes` always takes one envelope with an `observations` list of 1–1,000 items:

```json
{"observations": [{"target_ts": "2025-04-02T00:00:00Z", "actual_mw": 1180.4}]}
```

`model_name` defaults to the configured `model_name`. `actual_mw` is declared with
`allow_inf_nan=False`, so NaN or infinity is rejected with 422 rather than stored. Each row
is upserted. The response reports how many rows were inserted and how many updated.

`GET /performance/report?window=24h` answers "how is the live model doing".

- Predictions are paired with outcomes on `(model_name, target_ts)` where the prediction
  belongs to the currently loaded version.
- If several predictions share a `(model_version, target_ts)`, the latest one recorded at
  or before `target_ts` wins. Predictions made *after* the target time are excluded — a
  prediction issued after the fact is not a forecast.
- Metrics: `mae`, `rmse`, `samples`, `coverage`, `window`, `model_version`.
- `coverage` is paired predictions divided by distinct predicted target hours in the
  window. It is reported because MAE over three rows is not evidence of health, and a
  silently collapsing coverage would otherwise look like stability.

Outcomes are stored by `model_name`, not `model_version`, because the actual value does
not depend on which version forecast it. Attribution to a version happens at query time
through `prediction_events`.

## Drift detection

### Reference sample

`reference_sample(df, columns, limit=1000)` samples deterministically
(`random_state=42`) up to `limit` non-null values per feature and returns
`dict[str, list[float]]`. It is stored on registration alongside the existing profile.

Determinism matters beyond tidiness: the sample *is* the comparison baseline, so two
registrations of the same training data must produce the same baseline. `register` never
rewrites an existing row, so a nondeterministic draw would silently freeze whichever one
happened first.

### Statistics

`psi(reference, observed, bins=10)`:

1. Derive bin edges from the reference sample using deciles, with the outermost edges
   replaced by `-inf` and `+inf` so out-of-range observations are counted rather than
   dropped.
2. Compute reference and observed bin proportions.
3. Replace zero proportions with `1e-6`.
4. Return `sum((observed_i - reference_i) * ln(observed_i / reference_i))`.

`ks(reference, observed)` wraps `scipy.stats.ks_2samp` and returns the statistic and
p-value. `scipy` becomes an explicit dependency; it is already present transitively via
scikit-learn, but the code imports it directly.

PSI is the primary signal because it is an effect size: it does not become significant
merely because the window is large. KS is reported alongside because PSI is weak at
detecting modest shifts, and a p-value distinguishes "different" from "far away". This
matches published guidance that the two complement each other rather than duplicate one
another.

### Thresholds

| Check | Pass | Warn | Fail |
| --- | --- | --- | --- |
| PSI | `< 0.10` | `0.10 – 0.25` | `> 0.25` |
| KS | `p >= 0.05` | `0.01 <= p < 0.05` | `p < 0.01` |

These are the conventional PSI bands. Smaller samples make PSI noisy in both directions,
so a window with fewer than 100 sampled rows reports `insufficient_data` for every check
instead of a confident `pass`. That status is distinct from `warn` on purpose: "not enough
evidence" and "some evidence" are different operational facts.

### Report

`GET /drift/report?window=1h` reads sampled `prediction_events.features` for the loaded
version, computes PSI and KS per feature against `reference_sample`, persists one
`drift_checks` row per (feature, method), and returns them with an overall status equal to
the worst individual status.

An empty window returns `sample_size: 0`, no checks, and status `insufficient_data` —
never a fabricated failure. This mirrors the Phase 2 decision for `/quality/report`.

A version registered before this migration has no reference sample. The report then
returns `status: "no_reference_sample"` with an actionable message rather than computing
against an empty reference.

Per-check `status` is limited to the four values allowed by the `drift_checks` CHECK
constraint. `no_reference_sample` and an empty-window `insufficient_data` are report-level
statuses; the latter is also what every check would report had there been enough rows.
`no_reference_sample` is never written to the table because no comparison took place.

## API surface

| Method | Path | Behaviour |
| --- | --- | --- |
| POST | `/outcomes` | Upsert 1–1000 observed values; 401 without a token when configured; 422 on non-finite values |
| GET | `/performance/report?window=24h` | MAE, RMSE, samples, coverage for the loaded version |
| GET | `/drift/report?window=1h` | Per-feature PSI/KS, persisted to `drift_checks` |
| POST | `/models/{name}/{version}/promote` | Now loads first, then promotes, then hot-swaps |
| GET | `/ready` | Unchanged shape; reports the version now serving |

Window parsing is unchanged and inherits the 365-day cap and 400-on-malformed behaviour
from Phase 2. `/performance/report` and `/drift/report` default to different windows
(24h and 1h) because outcomes arrive with a delay while drift is meaningful on recent
traffic. Both are overridable.

## Configuration

| Setting | Default | Meaning |
| --- | --- | --- |
| `KERN_API_TOKEN` | `""` | Shared token for promote and outcome ingestion; empty disables auth |
| `KERN_REFERENCE_SAMPLE_SIZE` | `1000` | Values per feature stored at registration |
| `KERN_DRIFT_MIN_SAMPLES` | `100` | Below this, checks report `insufficient_data` |

## Testing

Pure functions, no database:

- PSI of identical distributions is ~0; PSI of shifted distributions crosses the warn and
  fail bands; out-of-range observations are counted, not dropped; empty observed raises.
- KS statistic and p-value ordering on shifted samples; tiny samples return
  `insufficient_data` at the report layer.
- `reference_sample` is deterministic across runs and respects the limit.
- Performance metrics on hand-built pairs, including the "latest prediction at or before
  target" rule and exclusion of post-hoc predictions.
- Coverage arithmetic, including the zero-prediction case.

Database tests (`db` marker):

- Migration applies to a database that already holds Phase 2 rows, and is idempotent.
- Outcome upsert inserts, then updates on conflict.
- Performance report joins correctly across two versions and ignores outcomes with no
  prediction.
- Drift report persists one row per (feature, method) and never fabricates checks for an
  empty window.
- Promote hot-swaps: after promoting v2, a `/predict` in the same process reports v2
  without restart, and the response is finite.
- Promote rejects a bad artifact without changing the production row.
- Token guard: 401 without a header, 401 with a wrong token, 200 with the right one, and
  open when the token is unset.

Existing tests must all continue to pass, including the Phase 2 quality and telemetry
suites.

## Risks

| Risk | Mitigation |
| --- | --- |
| Hot swap races with in-flight requests | Frozen `ModelState` replaced wholesale; handlers capture it once |
| Outcome keyed to the wrong hour | Unique `(model_name, target_ts)`; upsert corrects late data; post-hoc predictions excluded |
| Drift false alarms in small windows | `insufficient_data` below 100 samples; PSI is an effect size, not a significance test |
| PSI unstable at the bin edges | Outermost edges are infinite; empty bins floored at 1e-6 |
| Promotion fails after DB commit | Artifact is loaded and verified before commit; failure is logged and converges on restart |
| Existing artifacts lack a reference sample | Explicit `no_reference_sample` status plus a backfill script |
| Batch predictions cannot be attributed | Documented limitation: performance tracking covers `/predict` only |

## Rollout

1. Apply migration 002.
2. Run `scripts/backfill_reference_sample.py` to populate the column for existing versions.
3. Restart once so the new lifespan and endpoints are live.
4. Ingest a day of outcomes, then read `/performance/report?window=24h` and confirm the
   MAE is in the same range as the offline evaluation (64.49 on the held-out test set).
5. Read `/drift/report?window=1h` under load and confirm checks appear with sane statuses.

## Known limitations

- Performance tracking covers single `/predict` requests only. A batch telemetry event
  carries one feature snapshot for up to 256 rows, so it cannot be attributed to one
  target hour; batch events keep `target_ts` null. Fixing this requires per-row telemetry
  and is deferred.
- Outcomes are attributed to a version at read time, so promoting mid-window changes which
  predictions are counted. Reports are therefore version-scoped and should be read per
  version, not across a promotion boundary.
- Drift is computed on the sampled telemetry stream (`feature_sample_rate`, default 5%),
  so detection lag depends on traffic volume at least as much as on the drift itself.
- The token is a single shared secret with no rotation, revocation, or per-client identity.
