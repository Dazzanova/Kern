"""Report builders shared by the HTTP endpoints and the evaluation cycle.

The endpoints and the scheduler must compute and persist exactly the same checks, so
the logic lives here rather than in the route handlers. Each builder takes an explicit
`since` so a single cycle can use one clock for all three reports.
"""

from datetime import datetime, timezone
from typing import Protocol

from kern.drift import (
    TIME_DERIVED_FEATURES,
    ks,
    ks_status,
    overall_status,
    psi,
    psi_status,
)
from kern.performance import summarize_performance
from kern.quality import evaluate_features


class ServingModel(Protocol):
    """The subset of ModelState the reports read."""

    name: str
    version: str
    reference_profile: dict[str, dict[str, float]]
    reference_sample: dict[str, list[float]] | None


def quality_report(
    pool, current: ServingModel, *, window: str, since: datetime
) -> dict:
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT model_version, features FROM prediction_events "
            "WHERE recorded_at >= %s AND features IS NOT NULL "
            "AND model_version = %s",
            (since, current.version),
        ).fetchall()

    observations: dict[str, list[float | None]] = {}
    for _version, features in rows:
        for feature, value in features.items():
            observations.setdefault(feature, []).append(value)

    # An empty window means "nothing observed yet", which must not be reported
    # as every feature being 100% null. That would raise a false alarm for the
    # whole sampling interval right after a promotion or restart.
    checks = (
        evaluate_features(current.reference_profile, observations) if rows else []
    )

    window_end = datetime.now(timezone.utc)
    if checks:
        with pool.connection() as conn, conn.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO quality_checks (window_start, window_end, model_version, "
                "feature, check_type, expected, observed, status) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                [
                    (
                        since,
                        window_end,
                        current.version,
                        check["feature"],
                        check["check_type"],
                        check["expected"],
                        check["observed"],
                        check["status"],
                    )
                    for check in checks
                ],
            )

    return {
        "window": window,
        "model_version": current.version,
        "sample_count": len(rows),
        "checks": checks,
    }


def performance_report(
    pool, current: ServingModel, *, window: str, since: datetime
) -> dict:
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT recorded_at, target_ts, prediction_mean "
            "FROM prediction_events "
            "WHERE model_version = %s AND target_ts IS NOT NULL AND target_ts >= %s",
            (current.version, since),
        ).fetchall()
        outcome_rows = conn.execute(
            "SELECT target_ts, actual_mw FROM observed_outcomes "
            "WHERE model_name = %s AND target_ts >= %s",
            (current.name, since),
        ).fetchall()

    report = summarize_performance(rows, {ts: value for ts, value in outcome_rows})
    return {"window": window, "model_version": current.version, **report}


def drift_report(
    pool,
    current: ServingModel,
    *,
    window: str,
    since: datetime,
    min_samples: int,
) -> dict:
    if not current.reference_sample:
        return {
            "window": window,
            "model_version": current.version,
            "sample_size": 0,
            "status": "no_reference_sample",
            "checks": [],
        }

    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT features FROM prediction_events "
            "WHERE recorded_at >= %s AND model_version = %s AND features IS NOT NULL",
            (since, current.version),
        ).fetchall()

    observations: dict[str, list[float]] = {}
    for (features,) in rows:
        for feature, value in features.items():
            if isinstance(value, (int, float)):
                observations.setdefault(feature, []).append(float(value))

    if len(rows) < min_samples:
        # Not enough evidence is a different fact from "no drift"; reporting a
        # pass here would be false confidence.
        return {
            "window": window,
            "model_version": current.version,
            "sample_size": len(rows),
            "status": "insufficient_data",
            "checks": [],
        }

    checks: list[dict] = []
    for feature, reference in current.reference_sample.items():
        if feature in TIME_DERIVED_FEATURES:
            continue
        values = observations.get(feature, [])
        if len(values) < min_samples:
            checks.append(
                {
                    "feature": feature,
                    "method": "psi",
                    "statistic": None,
                    "p_value": None,
                    "sample_size": len(values),
                    "status": "insufficient_data",
                }
            )
            continue

        psi_value = psi(reference, values)
        ks_statistic, p_value = ks(reference, values)
        checks.append(
            {
                "feature": feature,
                "method": "psi",
                "statistic": psi_value,
                "p_value": None,
                "sample_size": len(values),
                "status": psi_status(psi_value),
            }
        )
        checks.append(
            {
                "feature": feature,
                "method": "ks",
                "statistic": ks_statistic,
                "p_value": p_value,
                "sample_size": len(values),
                "status": ks_status(p_value),
            }
        )

    window_end = datetime.now(timezone.utc)
    if checks:
        with pool.connection() as conn, conn.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO drift_checks (window_start, window_end, model_version, "
                "feature, method, statistic, p_value, sample_size, status) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                [
                    (
                        since,
                        window_end,
                        current.version,
                        check["feature"],
                        check["method"],
                        check["statistic"],
                        check["p_value"],
                        check["sample_size"],
                        check["status"],
                    )
                    for check in checks
                ],
            )

    return {
        "window": window,
        "model_version": current.version,
        "sample_size": len(rows),
        "status": overall_status(checks),
        "checks": checks,
    }
