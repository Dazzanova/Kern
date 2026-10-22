"""Promotion policy: decide whether a candidate may replace production.

The rules are pure and separated from the write path so they can be tested without a
database. `record_decision` is the only function here that writes.
"""

from dataclasses import dataclass, field
from typing import Any

from psycopg.types.json import Jsonb


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reasons: list[str]
    details: dict[str, Any] = field(default_factory=dict)


def _metric(metrics: dict[str, Any] | None, key: str):
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(key)
    return value if isinstance(value, (int, float)) else None


def _window(metrics: dict[str, Any] | None) -> tuple[Any, Any] | None:
    if not isinstance(metrics, dict):
        return None
    start, end = metrics.get("test_start"), metrics.get("test_end")
    return (start, end) if start and end else None


def evaluate_promotion(
    *,
    candidate,
    production,
    open_critical: int,
    margin: float,
) -> Decision:
    """Return every reason a promotion should be refused, not just the first.

    `candidate` and `production` are registry records with `.version` and `.metrics`.
    `production` is None for the first version of a model.
    """
    if production is None:
        return Decision(
            allowed=True,
            reasons=["no production version; bootstrap promotion"],
            details={"candidate_version": candidate.version},
        )

    reasons: list[str] = []
    details: dict[str, Any] = {
        "candidate_version": candidate.version,
        "production_version": production.version,
        "margin": margin,
        "open_critical": open_critical,
    }

    candidate_mae = _metric(candidate.metrics, "mae")
    production_mae = _metric(production.metrics, "mae")
    details["candidate_mae"] = candidate_mae
    details["production_mae"] = production_mae

    candidate_window = _window(candidate.metrics)
    production_window = _window(production.metrics)
    if candidate_window and production_window and candidate_window != production_window:
        reasons.append(
            "holdout window differs: candidate "
            f"{candidate_window[0]}..{candidate_window[1]} vs production "
            f"{production_window[0]}..{production_window[1]}"
        )

    if candidate_mae is None:
        reasons.append("candidate has no holdout MAE")
    elif production_mae is None:
        reasons.append("production has no comparable holdout MAE")
    else:
        threshold = production_mae * (1.0 - margin)
        details["required_mae"] = threshold
        # `>=` so a margin of zero still requires a strict improvement.
        if candidate_mae >= threshold:
            reasons.append(
                f"candidate MAE {candidate_mae:.2f} does not improve on production "
                f"{production_mae:.2f} by {margin:.0%} (needs < {threshold:.2f})"
            )

    if open_critical:
        reasons.append(
            f"{open_critical} open critical incident(s) must be resolved first"
        )

    return Decision(allowed=not reasons, reasons=reasons, details=details)


def open_critical_incidents(pool, model_name: str) -> int:
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT count(*) FROM incidents "
            "WHERE model_name = %s AND status = 'open' AND severity = 'critical'",
            (model_name,),
        ).fetchone()
    return int(row[0])


def record_decision(
    pool,
    *,
    model_name: str,
    version: str,
    production_version: str | None,
    decision: str,
    reasons: list[str],
    metrics: dict[str, Any] | None,
) -> None:
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO promotion_decisions "
            "(model_name, version, production_version, decision, reasons, metrics) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (
                model_name,
                version,
                production_version,
                decision,
                Jsonb(reasons),
                Jsonb(metrics) if metrics is not None else None,
            ),
        )
