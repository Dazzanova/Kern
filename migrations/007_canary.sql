-- Phase 4d: canary deployment.
-- A canary answer is a served answer, so it lives in prediction_events under the
-- canary's version. The flag marks a request that chose the canary but fell back.

ALTER TABLE prediction_events
    ADD COLUMN IF NOT EXISTS canary_fallback boolean NOT NULL DEFAULT false;

CREATE TABLE IF NOT EXISTS canary_assignments (
    model_name  text PRIMARY KEY,
    version     text NOT NULL,
    fraction    double precision NOT NULL
                CHECK (fraction > 0 AND fraction < 1),
    attached_at timestamptz NOT NULL DEFAULT now()
);
