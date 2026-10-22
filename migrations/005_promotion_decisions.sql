-- Phase 4b: audit every promotion attempt, including refusals and overrides.

CREATE TABLE IF NOT EXISTS promotion_decisions (
    id                 bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_name         text NOT NULL,
    version            text NOT NULL,
    production_version text,
    decision           text NOT NULL,
    reasons            jsonb NOT NULL,
    metrics            jsonb,
    created_at         timestamptz NOT NULL DEFAULT now(),
    CHECK (decision IN ('allowed', 'blocked', 'forced'))
);

CREATE INDEX IF NOT EXISTS promotion_decisions_lookup_idx
    ON promotion_decisions (model_name, created_at DESC);
