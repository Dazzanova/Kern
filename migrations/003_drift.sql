-- Phase 3b: reference samples and drift history.

ALTER TABLE model_versions
    ADD COLUMN IF NOT EXISTS reference_sample jsonb;

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
