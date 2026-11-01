-- Phase 4f: automated retraining.
-- One row per attempt, including skips and failures. A registered run names the
-- `auto-*` candidate; production is never changed by retraining.

CREATE TABLE IF NOT EXISTS retraining_runs (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_name  text NOT NULL,
    trigger     text NOT NULL CHECK (trigger IN ('automatic', 'manual')),
    decision    text NOT NULL CHECK (decision IN ('registered', 'skipped', 'failed')),
    version     text,
    reasons     jsonb NOT NULL,
    metrics     jsonb,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS retraining_runs_lookup_idx
    ON retraining_runs (model_name, created_at DESC);
