"""Shadow evaluation: score a candidate on the requests production serves.

The candidate's prediction never leaves this module. It is recorded asynchronously and
compared with production later, so the serving path and the served answer are unchanged.
"""

import logging
import queue
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean

from kern.performance import summarize_performance

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShadowEvent:
    request_id: str
    model_name: str
    model_version: str
    production_version: str
    endpoint: str
    batch_size: int
    duration_ms: float
    prediction_min: float | None
    prediction_mean: float | None
    prediction_max: float | None
    # Null for batch requests: one shadow row covers up to 256 target hours.
    target_ts: datetime | None = None


class ShadowRecorder:
    """Bounded hand-off; a full queue drops shadow rows, never a prediction."""

    def __init__(self, queue_size: int = 10_000):
        self.queue: queue.Queue[ShadowEvent] = queue.Queue(maxsize=queue_size)
        self._dropped = 0
        self._lock = threading.Lock()

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    def record(self, event: ShadowEvent) -> bool:
        try:
            self.queue.put_nowait(event)
        except queue.Full:
            with self._lock:
                self._dropped += 1
            return False
        return True


_INSERT = """
INSERT INTO shadow_predictions (
    request_id, model_name, model_version, production_version, endpoint,
    batch_size, duration_ms, prediction_min, prediction_mean, prediction_max,
    target_ts
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (request_id, model_version) DO NOTHING
"""


class ShadowWriter:
    """Drains shadow events into Postgres on its own daemon thread.

    Separate from TelemetryWriter so a failure here can never drop served-prediction
    telemetry, and vice versa.
    """

    BATCH_LIMIT = 100

    def __init__(self, pool, recorder: ShadowRecorder, *, flush_interval: float = 0.25):
        self._pool = pool
        self._recorder = recorder
        self._interval = flush_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.persisted = 0
        self.failed = 0
        self.last_flush_at: datetime | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="kern-shadow", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.drain_once()
            self._stop.wait(self._interval)
        self.drain_once()

    def drain_once(self) -> int:
        batch: list[ShadowEvent] = []
        while len(batch) < self.BATCH_LIMIT:
            try:
                batch.append(self._recorder.queue.get_nowait())
            except queue.Empty:
                break

        if not batch:
            return 0

        rows = [
            (
                event.request_id,
                event.model_name,
                event.model_version,
                event.production_version,
                event.endpoint,
                event.batch_size,
                event.duration_ms,
                event.prediction_min,
                event.prediction_mean,
                event.prediction_max,
                event.target_ts,
            )
            for event in batch
        ]

        try:
            with self._pool.connection() as conn, conn.cursor() as cur:
                cur.executemany(_INSERT, rows)
        except Exception:
            self.failed += len(batch)
            logger.exception("shadow flush failed; dropped %d events", len(batch))
            return 0

        self.persisted += len(batch)
        self.last_flush_at = datetime.now(timezone.utc)
        return len(batch)


# --- assignments -----------------------------------------------------------


def attach(pool, *, model_name: str, version: str) -> dict:
    with pool.connection() as conn:
        row = conn.execute(
            "INSERT INTO shadow_assignments (model_name, version) VALUES (%s, %s) "
            "ON CONFLICT (model_name) DO UPDATE SET "
            "version = EXCLUDED.version, attached_at = now() "
            "RETURNING model_name, version, attached_at",
            (model_name, version),
        ).fetchone()
    return {"model_name": row[0], "version": row[1], "attached_at": row[2]}


def detach(pool, model_name: str) -> bool:
    with pool.connection() as conn:
        row = conn.execute(
            "DELETE FROM shadow_assignments WHERE model_name = %s RETURNING 1",
            (model_name,),
        ).fetchone()
    return row is not None


def get_assignment(pool, model_name: str) -> str | None:
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT version FROM shadow_assignments WHERE model_name = %s",
            (model_name,),
        ).fetchone()
    return row[0] if row is not None else None


# --- report ----------------------------------------------------------------


def agreement(pairs) -> dict:
    """Absolute distance between the two models on the same inputs.

    Missing predictions are skipped rather than counted as zero error.
    """
    deltas = [
        abs(shadow - production)
        for shadow, production in pairs
        if shadow is not None and production is not None
    ]
    return {
        "samples": len(deltas),
        "mean_abs_delta": mean(deltas) if deltas else None,
        "max_abs_delta": max(deltas) if deltas else None,
    }


def verdict(production: dict, shadow: dict, *, min_samples: int) -> str:
    if shadow["samples"] < min_samples or production["samples"] < min_samples:
        return "inconclusive"
    if production["mae"] is None or shadow["mae"] is None:
        return "inconclusive"
    if shadow["mae"] < production["mae"]:
        return "shadow_better"
    if shadow["mae"] > production["mae"]:
        return "shadow_worse"
    return "tie"


def build_report(
    pool,
    *,
    model_name: str,
    shadow_version: str | None,
    production_version: str | None,
    window: str,
    since: datetime,
    min_samples: int,
) -> dict:
    if shadow_version is None:
        return {
            "window": window,
            "model_name": model_name,
            "shadow_version": None,
            "production_version": production_version,
            "paired_samples": 0,
            "agreement": agreement([]),
            "production": None,
            "shadow": None,
            "verdict": "inconclusive",
        }

    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT s.request_id, s.prediction_mean, p.prediction_mean, "
            "p.target_ts, s.recorded_at "
            "FROM shadow_predictions s "
            "JOIN prediction_events p ON p.request_id = s.request_id "
            "AND p.model_name = s.model_name "
            "AND p.endpoint IN ('predict', 'predict/batch') "
            "WHERE s.model_name = %s AND s.model_version = %s "
            "AND s.recorded_at >= %s",
            (model_name, shadow_version, since),
        ).fetchall()
        outcome_rows = conn.execute(
            "SELECT target_ts, actual_mw FROM observed_outcomes "
            "WHERE model_name = %s AND target_ts >= %s",
            (model_name, since),
        ).fetchall()

    outcomes = {ts: value for ts, value in outcome_rows}
    production = summarize_performance(
        [
            (recorded_at, target_ts, production_mean)
            for _rid, _shadow, production_mean, target_ts, recorded_at in rows
            if production_mean is not None
        ],
        outcomes,
    )
    shadow = summarize_performance(
        [
            (recorded_at, target_ts, shadow_mean)
            for _rid, shadow_mean, _production, target_ts, recorded_at in rows
            if shadow_mean is not None
        ],
        outcomes,
    )

    return {
        "window": window,
        "model_name": model_name,
        "shadow_version": shadow_version,
        "production_version": production_version,
        "paired_samples": len(rows),
        "agreement": agreement(
            [(shadow_mean, production_mean) for _rid, shadow_mean, production_mean, _ts, _r in rows]
        ),
        "production": production,
        "shadow": shadow,
        "verdict": verdict(production, shadow, min_samples=min_samples),
    }
