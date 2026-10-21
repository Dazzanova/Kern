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
