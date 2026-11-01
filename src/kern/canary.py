"""Canary deployment: serve a fraction of traffic from a candidate.

The canary's answers are real answers, recorded in `prediction_events` under its own
version. Nothing here promotes, rolls back, or decides anything automatically.
"""

import random
from datetime import datetime
from typing import Callable

from kern.performance import summarize_performance


def should_use_canary(
    fraction: float, rng: Callable[[], float] = random.random
) -> bool:
    """Draw once per request; the candidate answers when the draw is below `fraction`."""
    return rng() < fraction


def attach(pool, *, model_name: str, version: str, fraction: float) -> dict:
    with pool.connection() as conn:
        row = conn.execute(
            "INSERT INTO canary_assignments (model_name, version, fraction) "
            "VALUES (%s, %s, %s) "
            "ON CONFLICT (model_name) DO UPDATE SET "
            "version = EXCLUDED.version, "
            "fraction = EXCLUDED.fraction, "
            "attached_at = now() "
            "RETURNING model_name, version, fraction, attached_at",
            (model_name, version, fraction),
        ).fetchone()
    return {
        "model_name": row[0],
        "version": row[1],
        "fraction": row[2],
        "attached_at": row[3],
    }


def detach(pool, model_name: str) -> bool:
    with pool.connection() as conn:
        row = conn.execute(
            "DELETE FROM canary_assignments WHERE model_name = %s RETURNING 1",
            (model_name,),
        ).fetchone()
    return row is not None


def get_assignment(pool, model_name: str) -> dict | None:
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT version, fraction, attached_at FROM canary_assignments "
            "WHERE model_name = %s",
            (model_name,),
        ).fetchone()
    if row is None:
        return None
    return {"version": row[0], "fraction": row[1], "attached_at": row[2]}


def verdict(production: dict, canary: dict, *, min_samples: int) -> str:
    if production["samples"] < min_samples or canary["samples"] < min_samples:
        return "inconclusive"
    if production["mae"] is None or canary["mae"] is None:
        return "inconclusive"
    if canary["mae"] < production["mae"]:
        return "canary_better"
    if canary["mae"] > production["mae"]:
        return "canary_worse"
    return "tie"


def _version_rows(pool, version: str, since: datetime) -> list:
    with pool.connection() as conn:
        return conn.execute(
            "SELECT recorded_at, target_ts, prediction_mean FROM prediction_events "
            "WHERE model_version = %s AND recorded_at >= %s",
            (version, since),
        ).fetchall()


def _fallback_count(pool, model_name: str, since: datetime) -> int:
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT count(*) FROM prediction_events "
            "WHERE model_name = %s AND recorded_at >= %s AND canary_fallback",
            (model_name, since),
        ).fetchone()
    return int(row[0])


def build_report(
    pool,
    *,
    model_name: str,
    canary_version: str | None,
    production_version: str | None,
    fraction: float | None,
    window: str,
    since: datetime,
    min_samples: int,
) -> dict:
    if canary_version is None:
        return {
            "window": window,
            "model_name": model_name,
            "canary_version": None,
            "production_version": production_version,
            "fraction": None,
            "served": {"production": 0, "canary": 0},
            "observed_fraction": None,
            "fallbacks": 0,
            "production": None,
            "canary": None,
            "verdict": "inconclusive",
        }

    production_rows = (
        _version_rows(pool, production_version, since) if production_version else []
    )
    canary_rows = _version_rows(pool, canary_version, since)

    with pool.connection() as conn:
        outcome_rows = conn.execute(
            "SELECT target_ts, actual_mw FROM observed_outcomes "
            "WHERE model_name = %s AND target_ts >= %s",
            (model_name, since),
        ).fetchall()
    outcomes = {ts: value for ts, value in outcome_rows}

    production = summarize_performance(production_rows, outcomes)
    canary = summarize_performance(canary_rows, outcomes)

    total = len(production_rows) + len(canary_rows)
    return {
        "window": window,
        "model_name": model_name,
        "canary_version": canary_version,
        "production_version": production_version,
        "fraction": fraction,
        "served": {
            "production": len(production_rows),
            "canary": len(canary_rows),
        },
        "observed_fraction": len(canary_rows) / total if total else None,
        "fallbacks": _fallback_count(pool, model_name, since),
        "production": production,
        "canary": canary,
        "verdict": verdict(production, canary, min_samples=min_samples),
    }
