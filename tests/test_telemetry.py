import threading
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from kern.telemetry import PredictionEvent, TelemetryRecorder, TelemetryWriter, summarize


def make_event(**overrides):
    values = {
        "request_id": str(uuid4()),
        "model_name": "load-forecast",
        "model_version": "v1",
        "endpoint": "predict",
        "batch_size": 1,
        "duration_ms": 12.5,
        "prediction_min": 100.0,
        "prediction_mean": 100.0,
        "prediction_max": 100.0,
        "features": {"temperature_c": 20.0},
    }
    values.update(overrides)
    return PredictionEvent(**values)


def test_record_enqueues_event():
    recorder = TelemetryRecorder(queue_size=10)

    assert recorder.record(make_event()) is True
    assert recorder.queue.qsize() == 1


def test_full_queue_drops_instead_of_blocking():
    recorder = TelemetryRecorder(queue_size=1)
    recorder.record(make_event())

    # A blocking `put` would never return here, so run the call on a worker
    # thread and require it to finish promptly. The thread is a daemon so a
    # regression fails the assertion instead of hanging the test suite.
    result: list[bool] = []
    worker = threading.Thread(
        target=lambda: result.append(recorder.record(make_event())), daemon=True
    )
    worker.start()
    worker.join(timeout=1.0)

    assert not worker.is_alive(), "record() blocked on a full queue"
    assert result == [False]
    assert recorder.dropped == 1


def test_request_sample_rate_zero_records_nothing():
    recorder = TelemetryRecorder(queue_size=10, request_sample_rate=0.0)

    assert recorder.record(make_event()) is False
    assert recorder.queue.qsize() == 0
    assert recorder.dropped == 0


def test_feature_snapshot_respects_feature_sample_rate():
    recorder = TelemetryRecorder(queue_size=10, feature_sample_rate=0.0)

    recorder.record(make_event())
    event = recorder.queue.get_nowait()

    assert event.features is None


def test_non_finite_feature_values_are_stripped():
    # jsonb rejects NaN/Infinity, and one bad row discards the whole batch.
    recorder = TelemetryRecorder(queue_size=10, feature_sample_rate=1.0)

    recorder.record(
        make_event(
            features={"temperature_c": float("nan"), "humidity_pct": 50.0}
        )
    )
    event = recorder.queue.get_nowait()

    assert event.features == {"humidity_pct": 50.0}


def test_writer_counts_failed_events():
    class FailingPool:
        def connection(self):
            raise RuntimeError("db down")

    recorder = TelemetryRecorder(queue_size=10)
    writer = TelemetryWriter(FailingPool(), recorder)
    recorder.record(make_event())
    recorder.record(make_event())

    assert writer.drain_once() == 0
    assert writer.failed == 2


@pytest.mark.db
def test_writer_persists_queue_contents(pg):
    recorder = TelemetryRecorder(queue_size=10)
    writer = TelemetryWriter(pg, recorder)
    event = make_event()
    recorder.record(event)

    persisted = writer.drain_once()

    assert persisted == 1
    with pg.connection() as conn:
        row = conn.execute(
            "SELECT model_version, batch_size FROM prediction_events"
        ).fetchone()
    assert row == ("v1", 1)


@pytest.mark.db
def test_summarize_reports_counts_and_latency(pg):
    recorder = TelemetryRecorder(queue_size=10)
    writer = TelemetryWriter(pg, recorder)
    recorder.record(make_event(duration_ms=10.0, batch_size=2))
    recorder.record(make_event(duration_ms=30.0, batch_size=4))
    writer.drain_once()

    summary = summarize(pg, since=datetime.now(timezone.utc).replace(year=2000))

    assert summary["requests"] == 2
    assert summary["records"] == 6
    assert summary["p50_ms"] == pytest.approx(20.0)
    assert summary["dropped_events"] == 0
