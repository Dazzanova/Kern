"""Predictive capacity planning from recent serving traffic.

The planner is advisory. It buckets request timestamps, forecasts the next horizon with
Holt's linear trend, and converts peak demand into a replica recommendation. Nothing here
touches a deployment; the serving app only exposes the persisted plan.

Pure functions (`bucket_requests`, `forecast_rps`, `replicas_for_rate`, `build_plan`) sit
above the database helpers so the math can be tested without infrastructure.
"""

import logging
import math
from datetime import datetime, timedelta, timezone

from psycopg.types.json import Jsonb

logger = logging.getLogger(__name__)

# Smoothing constants for Holt's linear trend. Fixed rather than configurable: the
# forecast is a short-horizon capacity signal, not a tuned forecasting product.
HOLT_ALPHA = 0.5
HOLT_BETA = 0.3


def _iso_bucket(index: int, step_seconds: int) -> str:
    return datetime.fromtimestamp(
        index * step_seconds, tz=timezone.utc
    ).isoformat()


def bucket_requests(
    rows, *, now: datetime, step_minutes: int, lookback_minutes: int
) -> list[dict]:
    """Count requests into completed step intervals.

    The interval containing `now` is partially elapsed and excluded. Buckets with no
    requests stay in the series as zeros so a quiet period is evidence, not a gap.
    """
    step_seconds = step_minutes * 60
    current_index = int(now.timestamp() // step_seconds)
    bucket_count = max(1, lookback_minutes // step_minutes)
    first_index = current_index - bucket_count

    counts = {index: 0 for index in range(first_index, current_index)}
    for recorded_at in rows:
        index = int(recorded_at.timestamp() // step_seconds)
        if index in counts:
            counts[index] += 1

    return [
        {
            "starts_at": _iso_bucket(index, step_seconds),
            "requests": counts[index],
            "rps": counts[index] / step_seconds,
        }
        for index in range(first_index, current_index)
    ]


def forecast_rps(
    rates: list[float],
    *,
    horizon: int,
    alpha: float = HOLT_ALPHA,
    beta: float = HOLT_BETA,
) -> list[float]:
    """Holt's linear trend, clamped at zero for the next `horizon` steps."""
    if horizon <= 0:
        return []
    if not rates:
        return [0.0] * horizon

    level = float(rates[0])
    trend = float(rates[1] - rates[0]) if len(rates) > 1 else 0.0
    for value in rates[1:]:
        previous_level = level
        level = alpha * float(value) + (1.0 - alpha) * (level + trend)
        trend = beta * (level - previous_level) + (1.0 - beta) * trend

    return [max(0.0, level + trend * (index + 1)) for index in range(horizon)]


def replicas_for_rate(
    rate: float,
    *,
    rps_per_replica: float,
    target_utilization: float,
    min_replicas: int,
    max_replicas: int,
) -> int:
    """Smallest replica count that keeps `rate` under target utilization."""
    lower = max(1, int(min_replicas))
    upper = max(lower, int(max_replicas))

    effective_rps = rps_per_replica * target_utilization
    if effective_rps <= 0:
        return upper
    if rate <= 0:
        return lower

    # Subtract a tiny epsilon so a rate exactly at capacity does not round up to
    # an extra replica because 90 * 0.7 is represented just below 63.
    needed = math.ceil(rate / effective_rps - 1e-9)
    return min(upper, max(lower, needed))


def build_plan(
    buckets: list[dict],
    *,
    step_minutes: int,
    horizon_minutes: int,
    min_buckets: int,
    rps_per_replica: float,
    target_utilization: float,
    min_replicas: int,
    max_replicas: int,
) -> dict:
    """Turn observed buckets into a forecast and a replica recommendation."""
    horizon_steps = max(1, horizon_minutes // step_minutes)
    observed_seconds = len(buckets) * step_minutes * 60
    observed_rps = (
        sum(bucket["requests"] for bucket in buckets) / observed_seconds
        if observed_seconds
        else 0.0
    )
    observed_buckets = sum(1 for bucket in buckets if bucket["requests"] > 0)

    plan = {
        "horizon_minutes": horizon_minutes,
        "step_minutes": step_minutes,
        "lookback_minutes": len(buckets) * step_minutes,
        "observed_rps": observed_rps,
        "observed_buckets": observed_buckets,
        "total_buckets": len(buckets),
        "rps_per_replica": float(rps_per_replica),
        "target_utilization": float(target_utilization),
        "observed": buckets,
        "forecast": [],
        "peak_forecast_rps": None,
        "recommended_replicas": None,
    }

    if observed_buckets < min_buckets:
        return {
            **plan,
            "status": "insufficient_data",
            "reasons": [
                f"only {observed_buckets} of {len(buckets)} buckets have requests; "
                f"need {min_buckets}"
            ],
        }

    rates = forecast_rps(
        [bucket["rps"] for bucket in buckets], horizon=horizon_steps
    )
    last_start = datetime.fromisoformat(buckets[-1]["starts_at"])
    step = timedelta(minutes=step_minutes)
    forecast = [
        {
            "starts_at": (last_start + step * (index + 1)).isoformat(),
            "rps": rate,
            "replicas": replicas_for_rate(
                rate,
                rps_per_replica=rps_per_replica,
                target_utilization=target_utilization,
                min_replicas=min_replicas,
                max_replicas=max_replicas,
            ),
        }
        for index, rate in enumerate(rates)
    ]

    peak_forecast_rps = max(bucket["rps"] for bucket in forecast)
    recommended_replicas = max(bucket["replicas"] for bucket in forecast)
    return {
        **plan,
        "status": "ok",
        "forecast": forecast,
        "peak_forecast_rps": peak_forecast_rps,
        "recommended_replicas": recommended_replicas,
        "reasons": [
            f"peak forecast {peak_forecast_rps:.2f} rps; "
            f"{rps_per_replica:g} rps/replica at "
            f"{target_utilization:.0%} target utilization"
        ],
    }


def request_history(
    pool, *, model_name: str, since: datetime, until: datetime
) -> list[datetime]:
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT recorded_at FROM prediction_events "
            "WHERE model_name = %s AND recorded_at >= %s AND recorded_at < %s",
            (model_name, since, until),
        ).fetchall()
    return [row[0] for row in rows]


def record_plan(pool, *, model_name: str, plan: dict) -> dict:
    with pool.connection() as conn:
        row = conn.execute(
            "INSERT INTO scaling_plans (model_name, status, recommended_replicas, "
            "peak_forecast_rps, observed_rps, plan) "
            "VALUES (%s, %s, %s, %s, %s, %s) "
            "RETURNING id, created_at",
            (
                model_name,
                plan["status"],
                plan["recommended_replicas"],
                plan["peak_forecast_rps"],
                plan["observed_rps"],
                Jsonb(plan),
            ),
        ).fetchone()

    return {
        **plan,
        "id": row[0],
        "model_name": model_name,
        "created_at": row[1].isoformat(),
    }


def latest_plan(pool, model_name: str) -> dict | None:
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT id, plan, created_at FROM scaling_plans "
            "WHERE model_name = %s ORDER BY created_at DESC, id DESC LIMIT 1",
            (model_name,),
        ).fetchone()
    if row is None:
        return None
    return {
        **row[1],
        "id": row[0],
        "model_name": model_name,
        "created_at": row[2].isoformat(),
    }


def list_plans(pool, model_name: str, *, limit: int = 20) -> list[dict]:
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT id, plan, created_at FROM scaling_plans "
            "WHERE model_name = %s ORDER BY created_at DESC, id DESC LIMIT %s",
            (model_name, limit),
        ).fetchall()

    return [
        {
            **row[1],
            "id": row[0],
            "model_name": model_name,
            "created_at": row[2].isoformat(),
        }
        for row in rows
    ]


def plan_and_record(pool, *, model_name: str, settings, now: datetime | None = None):
    """Build and persist one plan. Returns None when planning is disabled."""
    if not settings.scaling_enabled:
        return None

    now = now or datetime.now(timezone.utc)
    since = now - timedelta(
        minutes=settings.scaling_lookback_minutes + settings.scaling_step_minutes
    )
    rows = request_history(
        pool, model_name=model_name, since=since, until=now
    )
    buckets = bucket_requests(
        rows,
        now=now,
        step_minutes=settings.scaling_step_minutes,
        lookback_minutes=settings.scaling_lookback_minutes,
    )
    plan = build_plan(
        buckets,
        step_minutes=settings.scaling_step_minutes,
        horizon_minutes=settings.scaling_horizon_minutes,
        min_buckets=settings.scaling_min_buckets,
        rps_per_replica=settings.scaling_rps_per_replica,
        target_utilization=settings.scaling_target_utilization,
        min_replicas=settings.scaling_min_replicas,
        max_replicas=settings.scaling_max_replicas,
    )
    stored = record_plan(pool, model_name=model_name, plan=plan)
    logger.info(
        "capacity plan for %s: %s, %s replicas (peak %s rps)",
        model_name,
        stored["status"],
        stored["recommended_replicas"],
        stored["peak_forecast_rps"],
    )
    return stored
