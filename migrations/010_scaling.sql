-- Phase 4g: predictive capacity planning.
-- Plans are advisory. Nothing in this schema is applied to a deployment.

CREATE TABLE IF NOT EXISTS scaling_plans (
    id                   bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_name           text NOT NULL,
    status               text NOT NULL CHECK (status IN ('ok', 'insufficient_data')),
    recommended_replicas integer,
    peak_forecast_rps    double precision,
    observed_rps         double precision,
    plan                 jsonb NOT NULL,
    created_at           timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS scaling_plans_lookup_idx
    ON scaling_plans (model_name, created_at DESC);
