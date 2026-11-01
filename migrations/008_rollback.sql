-- Phase 4e: automated production rollback.
-- One row per attempt, including refusals. The failed version is excluded from
-- future automatic targets by looking at from_version where decision = 'rolled_back'.

CREATE TABLE IF NOT EXISTS rollback_events (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_name   text NOT NULL,
    from_version text NOT NULL,
    to_version   text,
    trigger      text NOT NULL CHECK (trigger IN ('automatic', 'manual')),
    decision     text NOT NULL CHECK (decision IN ('rolled_back', 'refused')),
    reasons      jsonb NOT NULL,
    metrics      jsonb,
    created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS rollback_events_lookup_idx
    ON rollback_events (model_name, created_at DESC);
