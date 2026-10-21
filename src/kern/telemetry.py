import logging
import math
import queue
import random
import threading
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

from psycopg.types.json import Jsonb

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionEvent:
    request_id: str
    model_name: str
    model_version: str
    endpoint: str
    batch_size: int
    duration_ms: float
    prediction_min: float | None
    prediction_mean: float | None
    prediction_max: float | None
    features: dict[str, Any] | None
    # The timestamp the prediction is *for*. Null for batch requests, where one
    # event covers up to 256 different target hours and cannot be attributed.
    target_ts: datetime | None = None


class TelemetryRecorder:
    """Bounded, non-blocking hand-off between request threads and the writer."""

    def __init__(
        self,
        *,
        queue_size: int = 10_000,
        request_sample_rate: float = 1.0,
        feature_sample_rate: float = 0.05,
        rng: random.Random | None = None,
    ):
        self.queue: queue.Queue[PredictionEvent] = queue.Queue(maxsize=queue_size)
        self.request_sample_rate = request_sample_rate
        self.feature_sample_rate = feature_sample_rate
        self._rng = rng or random.Random()
        self._dropped = 0
        self._lock = threading.Lock()

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    def record(self, event: PredictionEvent) -> bool:
        if self.request_sample_rate < 1.0 and self._rng.random() > self.request_sample_rate:
            return False

        if event.features is not None:
            # NaN/Infinity make jsonb reject the whole batch, so drop those keys.
            event = replace(
                event,
                features={
                    key: value
                    for key, value in event.features.items()
                    if isinstance(value, (int, float)) and math.isfinite(value)
                },
            )

        if event.features is not None and self._rng.random() > self.feature_sample_rate:
            event = replace(event, features=None)

        try:
            self.queue.put_nowait(event)
        except queue.Full:
            with self._lock:
                self._dropped += 1
            return False

        return True


_INSERT = """
INSERT INTO prediction_events (
    request_id, model_name, model_version, endpoint, batch_size,
    duration_ms, prediction_min, prediction_mean, prediction_max, features,
    target_ts
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


class TelemetryWriter:
    """Drains the recorder queue into Postgres in batches on a daemon thread."""

    BATCH_LIMIT = 100

    def __init__(
        self,
        pool,
        recorder: TelemetryRecorder,
        *,
        flush_interval: float = 0.25,
    ):
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
            target=self._run, name="kern-telemetry", daemon=True
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
        batch: list[PredictionEvent] = []
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
                event.endpoint,
                event.batch_size,
                event.duration_ms,
                event.prediction_min,
                event.prediction_mean,
                event.prediction_max,
                Jsonb(event.features) if event.features is not None else None,
                event.target_ts,
            )
            for event in batch
        ]

        try:
            with self._pool.connection() as conn, conn.cursor() as cur:
                cur.executemany(_INSERT, rows)
        except Exception:
            self.failed += len(batch)
            logger.exception("telemetry flush failed; dropped %d events", len(batch))
            return 0

        self.persisted += len(batch)
        self.last_flush_at = datetime.now(timezone.utc)
        return len(batch)


def summarize(
    pool,
    *,
    since: datetime,
    dropped_events: int = 0,
    failed_events: int = 0,
    last_flush_at: datetime | None = None,
) -> dict[str, Any]:
    with pool.connection() as conn:
        row = conn.execute(
            """
            SELECT count(*) AS requests,
                   coalesce(sum(batch_size), 0) AS records,
                   coalesce(percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_ms), 0) AS p50,
                   coalesce(percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms), 0) AS p95,
                   coalesce(percentile_cont(0.99) WITHIN GROUP (ORDER BY duration_ms), 0) AS p99,
                   coalesce(avg(duration_ms), 0) AS mean
            FROM prediction_events
            WHERE recorded_at >= %s
            """,
            (since,),
        ).fetchone()

    return {
        "requests": int(row[0]),
        "records": int(row[1]),
        "p50_ms": float(row[2]),
        "p95_ms": float(row[3]),
        "p99_ms": float(row[4]),
        "mean_ms": float(row[5]),
        "dropped_events": dropped_events,
        "failed_events": failed_events,
        "last_flush_at": last_flush_at.isoformat() if last_flush_at else None,
    }
