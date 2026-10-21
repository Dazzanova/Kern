"""Scheduled detection: run the reports, sync incidents, prune old data."""

import logging
import threading
from datetime import datetime, timezone

from kern import incidents, reports
from kern.windows import parse_window

logger = logging.getLogger(__name__)

QUALITY_WINDOW = "1h"
DRIFT_WINDOW = "1h"
PERFORMANCE_WINDOW = "24h"

# Arbitrary but stable: two evaluators against the same database contend.
EVALUATION_LOCK_KEY = 4815162342


def run_cycle(pool, current, settings) -> dict:
    """Evaluate every report, then open and resolve incidents.

    The advisory lock is held for the whole cycle on one dedicated connection,
    so two replicas cannot interleave their writes.
    """
    with pool.connection() as lock_conn:
        acquired = lock_conn.execute(
            "SELECT pg_try_advisory_lock(%s)", (EVALUATION_LOCK_KEY,)
        ).fetchone()[0]
        if not acquired:
            logger.debug("evaluation skipped: lock held by another process")
            return {"skipped": "locked"}

        try:
            now = datetime.now(timezone.utc)
            quality = reports.quality_report(
                pool,
                current,
                window=QUALITY_WINDOW,
                since=now - parse_window(QUALITY_WINDOW),
            )
            drift = reports.drift_report(
                pool,
                current,
                window=DRIFT_WINDOW,
                since=now - parse_window(DRIFT_WINDOW),
                min_samples=settings.drift_min_samples,
            )
            performance = reports.performance_report(
                pool,
                current,
                window=PERFORMANCE_WINDOW,
                since=now - parse_window(PERFORMANCE_WINDOW),
            )

            findings, evaluated = incidents.build_findings(
                quality, drift, performance, settings
            )
            synced = incidents.sync_incidents(
                pool,
                model_name=current.name,
                model_version=current.version,
                findings=findings,
                evaluated_kinds=evaluated,
            )
        finally:
            lock_conn.execute(
                "SELECT pg_advisory_unlock(%s)", (EVALUATION_LOCK_KEY,)
            )

    return {
        "quality": quality,
        "drift": drift,
        "performance": performance,
        **synced,
    }


def run_retention(pool, days: int) -> dict:
    """Delete data older than `days`. Open incidents are never deleted."""
    if days <= 0:
        return {}

    statements = {
        "prediction_events": (
            "DELETE FROM prediction_events "
            "WHERE recorded_at < now() - make_interval(days => %s)"
        ),
        "quality_checks": (
            "DELETE FROM quality_checks "
            "WHERE window_end < now() - make_interval(days => %s)"
        ),
        "drift_checks": (
            "DELETE FROM drift_checks "
            "WHERE window_end < now() - make_interval(days => %s)"
        ),
        "incidents": (
            "DELETE FROM incidents WHERE status = 'resolved' "
            "AND resolved_at IS NOT NULL "
            "AND resolved_at < now() - make_interval(days => %s)"
        ),
    }

    deleted: dict[str, int] = {}
    with pool.connection() as conn:
        for table, statement in statements.items():
            deleted[table] = conn.execute(statement, (days,)).rowcount

    return deleted


def run_scheduled(app) -> dict:
    result = run_cycle(app.state.pool, app.state.current, app.state.settings)

    today = datetime.now(timezone.utc).date()
    if app.state.last_retention != today:
        result["retention"] = run_retention(
            app.state.pool, app.state.settings.retention_days
        )
        app.state.last_retention = today

    return result


class EvaluationRunner:
    """Background thread that runs one detection cycle per interval."""

    def __init__(self, app):
        self._app = app
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="kern-evaluation", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout)

    def _run(self) -> None:
        interval = self._app.state.settings.evaluation_interval
        # Sleep first: a short-lived process (tests, CLI, one-off import) never
        # evaluates. The first production cycle lands one interval after startup.
        while not self._stop.wait(interval):
            try:
                run_scheduled(self._app)
            except Exception:
                logger.exception("scheduled evaluation failed")
