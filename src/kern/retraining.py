"""Automated retraining: learn from labeled live traffic after a failure.

The loop is deliberately narrow. A critical performance incident permits one attempt,
the attempt trains from forecasts whose outcomes have arrived, and success means a new
candidate is registered. Production, shadow, and canary state are never touched here.

`attempt` performs I/O and returns the audit event. The serving app only exposes it.
"""

import logging
import math
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from psycopg.types.json import Jsonb

from kern.registry import get_production, register
from kern.training.candidate import build_candidate

logger = logging.getLogger(__name__)


def open_performance_incident(pool, model_name: str) -> dict | None:
    """The newest open critical performance incident for the model, if any.

    Not scoped to the current version: after a rollback the failed version's incident is
    still open, and that is the episode retraining should learn from.
    """
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT id, model_version, summary, details, first_seen FROM incidents "
            "WHERE model_name = %s AND kind = 'performance' "
            "AND severity = 'critical' AND status = 'open' "
            "ORDER BY first_seen DESC LIMIT 1",
            (model_name,),
        ).fetchone()
    if row is None:
        return None
    return {
        "id": row[0],
        "model_version": row[1],
        "summary": row[2],
        "details": row[3] or {},
        "first_seen": row[4],
    }


def labeled_rows(
    pool, *, model_name: str, window_days: int, now: datetime | None = None
) -> list[dict]:
    """Live forecasts paired with observed outcomes, newest target last.

    One row per target hour: the latest prediction recorded at or before that hour wins.
    Batch events have no target and rows without a usable snapshot are ignored.
    """
    now = now or datetime.now(timezone.utc)
    since = now - timedelta(days=window_days)

    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT ON (p.target_ts)
                   p.target_ts,
                   (p.features ->> 'temperature_c')::double precision,
                   (p.features ->> 'humidity_pct')::double precision,
                   o.actual_mw
            FROM prediction_events p
            JOIN observed_outcomes o
              ON o.model_name = p.model_name AND o.target_ts = p.target_ts
            WHERE p.model_name = %s
              AND p.target_ts IS NOT NULL
              AND p.features IS NOT NULL
              AND p.features ? 'temperature_c'
              AND p.features ? 'humidity_pct'
              AND p.recorded_at <= p.target_ts
              AND p.target_ts >= %s
              AND p.target_ts <= %s
            ORDER BY p.target_ts, p.recorded_at DESC, p.id DESC
            """,
            (model_name, since, now),
        ).fetchall()

    collected: list[dict] = []
    for target_ts, temperature, humidity, actual in rows:
        if temperature is None or humidity is None:
            continue
        values = (float(temperature), float(humidity), float(actual))
        if not all(math.isfinite(value) for value in values):
            continue
        collected.append(
            {
                "timestamp": target_ts,
                "temperature_c": values[0],
                "humidity_pct": values[1],
                "load_mw": values[2],
            }
        )

    collected.sort(key=lambda row: row["timestamp"])
    return collected


def split_rows(
    rows: list[dict], holdout_fraction: float
) -> tuple[list[dict], list[dict]]:
    """Split chronologically, keeping the newest rows as the temporal holdout."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be strictly between 0 and 1")
    if len(rows) < 2:
        raise ValueError("need at least two labeled rows to split")

    ordered = sorted(rows, key=lambda row: row["timestamp"])
    cut = int(len(ordered) * (1.0 - holdout_fraction))
    cut = min(max(cut, 1), len(ordered) - 1)
    return ordered[:cut], ordered[cut:]


def to_frame(rows: list[dict]) -> pd.DataFrame:
    """Reconstruct the raw columns `build_features` expects."""
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [row["timestamp"] for row in rows], utc=True
            ),
            "temperature_c": [float(row["temperature_c"]) for row in rows],
            "humidity_pct": [float(row["humidity_pct"]) for row in rows],
            "load_mw": [float(row["load_mw"]) for row in rows],
        }
    )


def last_run_at(pool, model_name: str) -> datetime | None:
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT max(created_at) FROM retraining_runs WHERE model_name = %s",
            (model_name,),
        ).fetchone()
    return row[0]


def record(
    pool,
    *,
    model_name: str,
    trigger: str,
    decision: str,
    version: str | None,
    reasons: list[str],
    metrics: dict | None,
) -> dict:
    with pool.connection() as conn:
        row = conn.execute(
            "INSERT INTO retraining_runs (model_name, trigger, decision, version, "
            "reasons, metrics) VALUES (%s, %s, %s, %s, %s, %s) "
            "RETURNING id, created_at",
            (
                model_name,
                trigger,
                decision,
                version,
                Jsonb(reasons),
                Jsonb(metrics) if metrics is not None else None,
            ),
        ).fetchone()

    return {
        "id": row[0],
        "model_name": model_name,
        "trigger": trigger,
        "decision": decision,
        "version": version,
        "reasons": reasons,
        "metrics": metrics,
        "created_at": row[1].isoformat(),
    }


def list_runs(pool, model_name: str, *, limit: int = 20) -> list[dict]:
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT id, trigger, decision, version, reasons, metrics, created_at "
            "FROM retraining_runs WHERE model_name = %s "
            "ORDER BY created_at DESC, id DESC LIMIT %s",
            (model_name, limit),
        ).fetchall()

    return [
        {
            "id": row[0],
            "model_name": model_name,
            "trigger": row[1],
            "decision": row[2],
            "version": row[3],
            "reasons": row[4],
            "metrics": row[5],
            "created_at": row[6].isoformat(),
        }
        for row in rows
    ]


def _next_version(now: datetime) -> str:
    return f"auto-{now.strftime('%Y%m%dT%H%M%S%fZ')}"


def _incident_reason(incident: dict | None) -> str:
    if incident is None:
        return "manual retraining requested"
    return (
        f"critical performance incident #{incident['id']}: "
        f"{incident['summary']}"
    )


def attempt(pool, store, *, model_name: str, trigger: str, settings) -> dict | None:
    """Decide and, when allowed, train and register one candidate.

    Returns `None` when there is nothing to do. A skip or failure returns the recorded
    event; so does a successful registration.
    """
    if trigger not in ("automatic", "manual"):
        raise ValueError(f"unknown retraining trigger: {trigger}")

    now = datetime.now(timezone.utc)
    production = get_production(pool, model_name)
    if production is None:
        return None

    incident = None
    if trigger == "automatic":
        if not settings.retrain_enabled:
            return None

        incident = open_performance_incident(pool, model_name)
        if incident is None:
            return None

        if settings.retrain_cooldown_hours > 0:
            last = last_run_at(pool, model_name)
            if last is not None and last > now - timedelta(
                hours=settings.retrain_cooldown_hours
            ):
                logger.debug(
                    "retraining skipped for %s: cooldown active until %s",
                    model_name,
                    (
                        last + timedelta(hours=settings.retrain_cooldown_hours)
                    ).isoformat(),
                )
                return None

    reasons = [_incident_reason(incident)]
    metrics: dict = {
        "incident_id": incident["id"] if incident is not None else None,
        "window_days": settings.retrain_window_days,
    }

    rows = labeled_rows(
        pool,
        model_name=model_name,
        window_days=settings.retrain_window_days,
        now=now,
    )
    metrics["labeled_rows"] = len(rows)

    if len(rows) < settings.retrain_min_rows:
        return record(
            pool,
            model_name=model_name,
            trigger=trigger,
            decision="skipped",
            version=None,
            reasons=reasons
            + [
                f"only {len(rows)} labeled rows in the last "
                f"{settings.retrain_window_days}d; need {settings.retrain_min_rows}"
            ],
            metrics=metrics,
        )

    train_rows, holdout_rows = split_rows(rows, settings.retrain_holdout_fraction)
    metrics["train_rows"] = len(train_rows)
    metrics["holdout_rows"] = len(holdout_rows)

    version = _next_version(now)
    try:
        candidate = build_candidate(
            to_frame(train_rows),
            to_frame(holdout_rows),
            reference_sample_size=settings.reference_sample_size,
        )
        with tempfile.TemporaryDirectory(prefix="kern-retrain-") as directory:
            path = Path(directory) / f"{version}.json"
            candidate.model.save_model(path)
            register(
                pool,
                store,
                name=model_name,
                version=version,
                artifact_path=path,
                metrics=candidate.metrics,
                feature_schema=candidate.feature_schema,
                reference_profile=candidate.reference_profile,
                reference_sample=candidate.reference_sample,
            )
    except Exception as error:
        logger.exception("retraining failed for %s", model_name)
        metrics["error"] = str(error)
        return record(
            pool,
            model_name=model_name,
            trigger=trigger,
            decision="failed",
            version=None,
            reasons=reasons + [f"training failed: {error}"],
            metrics=metrics,
        )

    metrics.update(
        {
            "holdout_mae": candidate.metrics.get("mae"),
            "holdout_rmse": candidate.metrics.get("rmse"),
            "train_start": candidate.metrics.get("train_start"),
            "train_end": candidate.metrics.get("train_end"),
            "test_start": candidate.metrics.get("test_start"),
            "test_end": candidate.metrics.get("test_end"),
            "params": candidate.metrics.get("params"),
        }
    )

    event = record(
        pool,
        model_name=model_name,
        trigger=trigger,
        decision="registered",
        version=version,
        reasons=reasons
        + [
            f"trained on {len(train_rows)} rows and validated on "
            f"{len(holdout_rows)}",
            "registered as a candidate; production unchanged",
        ],
        metrics=metrics,
    )
    logger.warning(
        "registered retrained candidate %s for %s (%s)",
        version,
        model_name,
        trigger,
    )
    return event
