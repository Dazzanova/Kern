-- Phase 4c: shadow evaluation.
-- shadow_assignments is the persisted choice of candidate; shadow_predictions holds
-- what that candidate answered, never what was served.

CREATE TABLE IF NOT EXISTS shadow_assignments (
    model_name  text PRIMARY KEY,
    version     text NOT NULL,
    attached_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS shadow_predictions (
    id                 bigserial PRIMARY KEY,
    request_id         uuid NOT NULL,
    model_name         text NOT NULL,
    model_version      text NOT NULL,
    production_version text NOT NULL,
    endpoint           text NOT NULL,
    batch_size         integer NOT NULL CHECK (batch_size > 0),
    duration_ms        double precision NOT NULL,
    prediction_min     double precision,
    prediction_mean    double precision,
    prediction_max     double precision,
    target_ts          timestamptz,
    recorded_at        timestamptz NOT NULL DEFAULT now(),
    UNIQUE (request_id, model_version)
);

CREATE INDEX IF NOT EXISTS shadow_predictions_version_time
    ON shadow_predictions (model_version, recorded_at DESC);
