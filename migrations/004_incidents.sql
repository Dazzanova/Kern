-- Phase 4a: incidents produced by the scheduled detection loop.

CREATE TABLE IF NOT EXISTS incidents (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    -- model_name:kind:subject -- one row per ongoing issue.
    dedupe_key    text NOT NULL UNIQUE,
    model_name    text NOT NULL,
    model_version text NOT NULL,
    kind          text NOT NULL,
    subject       text NOT NULL,
    severity      text NOT NULL,
    status        text NOT NULL DEFAULT 'open',
    summary       text NOT NULL,
    details       jsonb,
    first_seen    timestamptz NOT NULL DEFAULT now(),
    last_seen     timestamptz NOT NULL DEFAULT now(),
    resolved_at   timestamptz,
    CHECK (kind IN ('quality', 'drift', 'performance')),
    CHECK (severity IN ('warn', 'critical')),
    CHECK (status IN ('open', 'resolved'))
);

CREATE INDEX IF NOT EXISTS incidents_open_idx
    ON incidents (model_name, status, last_seen DESC);
