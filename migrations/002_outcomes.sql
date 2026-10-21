-- Phase 3a: outcome ingestion and forecast attribution.

ALTER TABLE prediction_events
    ADD COLUMN IF NOT EXISTS target_ts timestamptz;

-- Performance queries scope by version and order by target time.
CREATE INDEX IF NOT EXISTS prediction_events_version_target_idx
    ON prediction_events (model_version, target_ts)
    WHERE target_ts IS NOT NULL;

-- One observed value per model and target hour. Upsert on conflict so revised
-- meter readings overwrite rather than accumulate.
CREATE TABLE IF NOT EXISTS observed_outcomes (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name   text NOT NULL,
    target_ts    timestamptz NOT NULL,
    actual_mw    double precision NOT NULL,
    source       text NOT NULL DEFAULT 'api',
    observed_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (model_name, target_ts)
);
