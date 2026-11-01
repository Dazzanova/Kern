import math
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from psycopg.types.json import Jsonb

from kern import retraining
from kern.config import Settings
from kern.registry import get_production, get_version
from kern.serving.app import app

_MISSING = object()
MODEL = "load-forecast"


def settings(**overrides):
    return Settings(_env_file=None, **overrides)


def _insert_prediction(
    pg,
    *,
    target_ts,
    recorded_at,
    temperature=20.0,
    humidity=50.0,
    prediction=100.0,
    features=_MISSING,
    version="test-v1",
    model_name=MODEL,
):
    if features is _MISSING:
        features = {"temperature_c": temperature, "humidity_pct": humidity}
    with pg.connection() as conn:
        conn.execute(
            "INSERT INTO prediction_events (request_id, model_name, model_version, "
            "endpoint, batch_size, duration_ms, prediction_mean, features, "
            "target_ts, recorded_at) "
            "VALUES (gen_random_uuid(), %s, %s, 'predict', 1, 1.0, %s, %s, %s, %s)",
            (
                model_name,
                version,
                prediction,
                Jsonb(features) if features is not None else None,
                target_ts,
                recorded_at,
            ),
        )


def _insert_outcome(pg, target_ts, actual, *, model_name=MODEL):
    with pg.connection() as conn:
        conn.execute(
            "INSERT INTO observed_outcomes (model_name, target_ts, actual_mw) "
            "VALUES (%s, %s, %s) "
            "ON CONFLICT (model_name, target_ts) "
            "DO UPDATE SET actual_mw = EXCLUDED.actual_mw",
            (model_name, target_ts, actual),
        )


def _insert_labeled(
    pg,
    *,
    target_ts,
    actual=100.0,
    recorded_at=None,
    temperature=20.0,
    humidity=50.0,
    **kwargs,
):
    recorded_at = recorded_at or target_ts - timedelta(minutes=5)
    _insert_prediction(
        pg,
        target_ts=target_ts,
        recorded_at=recorded_at,
        temperature=temperature,
        humidity=humidity,
        prediction=actual,
        **kwargs,
    )
    _insert_outcome(pg, target_ts, actual)


def _seed_rows(pg, count=20, *, days_ago=10):
    base = datetime.now(timezone.utc) - timedelta(days=days_ago)
    for index in range(count):
        target = base + timedelta(minutes=index)
        actual = 150.0 + 10.0 * math.sin(index / 3.0) + index
        _insert_labeled(
            pg,
            target_ts=target,
            actual=actual,
            temperature=20.0 + (index % 5),
            humidity=50.0 + (index % 7),
        )


def _open_incident(pg, version, *, kind="performance", severity="critical"):
    with pg.connection() as conn:
        conn.execute(
            "INSERT INTO incidents (dedupe_key, model_name, model_version, kind, "
            "subject, severity, summary, details) "
            "VALUES (%s, %s, %s, %s, 'mae', %s, "
            "'MAE 500.0 mw over 30 forecasts', %s)",
            (
                f"{MODEL}:{kind}:mae",
                MODEL,
                version,
                kind,
                severity,
                Jsonb(
                    {
                        "mae": 500.0,
                        "rmse": 510.0,
                        "samples": 30,
                        "coverage": 1.0,
                        "warn_threshold": 90.0,
                        "fail_threshold": 120.0,
                    }
                ),
            ),
        )


def test_split_rows_keeps_the_newest_rows_as_holdout():
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [{"timestamp": base + timedelta(hours=index)} for index in range(10)]

    train, holdout = retraining.split_rows(rows, 0.2)

    assert [row["timestamp"] for row in train] == [
        base + timedelta(hours=index) for index in range(8)
    ]
    assert [row["timestamp"] for row in holdout] == [
        base + timedelta(hours=8),
        base + timedelta(hours=9),
    ]


def test_split_rows_keeps_both_sides_non_empty():
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [
        {"timestamp": base},
        {"timestamp": base + timedelta(hours=1)},
        {"timestamp": base + timedelta(hours=2)},
    ]

    train, holdout = retraining.split_rows(rows, 0.9)

    assert len(train) == 1
    assert len(holdout) == 2


def test_split_rows_rejects_invalid_input():
    with pytest.raises(ValueError):
        retraining.split_rows([{"timestamp": datetime.now(timezone.utc)}], 0.2)
    with pytest.raises(ValueError):
        retraining.split_rows(
            [
                {"timestamp": datetime.now(timezone.utc)},
                {"timestamp": datetime.now(timezone.utc)},
            ],
            1.0,
        )


@pytest.mark.db
def test_labeled_rows_join_dedupe_and_filter(pg):
    now = datetime.now(timezone.utc)

    target = now - timedelta(days=3)
    _insert_labeled(
        pg,
        target_ts=target,
        actual=100.0,
        temperature=20.0,
        humidity=40.0,
        recorded_at=target - timedelta(hours=2),
    )
    _insert_labeled(
        pg,
        target_ts=target,
        actual=100.0,
        temperature=30.0,
        humidity=45.0,
        recorded_at=target - timedelta(hours=1),
    )

    late = now - timedelta(days=2)
    _insert_labeled(
        pg,
        target_ts=late,
        actual=101.0,
        recorded_at=late + timedelta(minutes=1),
    )

    future = now + timedelta(days=1)
    _insert_labeled(
        pg,
        target_ts=future,
        actual=102.0,
        recorded_at=now,
    )

    missing_outcome = now - timedelta(days=4)
    _insert_prediction(
        pg,
        target_ts=missing_outcome,
        recorded_at=missing_outcome - timedelta(hours=1),
    )

    null_features = now - timedelta(days=5)
    _insert_prediction(
        pg,
        target_ts=null_features,
        recorded_at=null_features - timedelta(hours=1),
        features=None,
    )
    _insert_outcome(pg, null_features, 103.0)

    partial_features = now - timedelta(days=6)
    _insert_prediction(
        pg,
        target_ts=partial_features,
        recorded_at=partial_features - timedelta(hours=1),
        features={"humidity_pct": 55.0},
    )
    _insert_outcome(pg, partial_features, 104.0)

    outside_window = now - timedelta(days=40)
    _insert_labeled(
        pg,
        target_ts=outside_window,
        actual=105.0,
        recorded_at=outside_window - timedelta(hours=1),
    )

    rows = retraining.labeled_rows(
        pg, model_name=MODEL, window_days=30, now=now
    )

    assert len(rows) == 1
    assert rows[0]["temperature_c"] == 30.0
    assert rows[0]["humidity_pct"] == 45.0
    assert rows[0]["load_mw"] == 100.0


@pytest.mark.db
def test_too_few_rows_records_a_skip(pg, store, baseline_model):
    _seed_rows(pg, 3)

    event = retraining.attempt(
        pg,
        store,
        model_name=MODEL,
        trigger="manual",
        settings=settings(retrain_min_rows=10),
    )

    assert event["decision"] == "skipped"
    assert event["version"] is None
    assert "only 3 labeled rows" in event["reasons"][-1]
    assert event["metrics"]["labeled_rows"] == 3
    assert get_production(pg, MODEL).version == baseline_model
    with pg.connection() as conn:
        assert conn.execute(
            "SELECT count(*) FROM retraining_runs WHERE decision = 'skipped'"
        ).fetchone()[0] == 1


@pytest.mark.db
def test_no_incident_makes_no_attempt(pg, store, baseline_model):
    _seed_rows(pg, 5)

    event = retraining.attempt(
        pg,
        store,
        model_name=MODEL,
        trigger="automatic",
        settings=settings(retrain_min_rows=5, retrain_cooldown_hours=0),
    )

    assert event is None
    with pg.connection() as conn:
        assert conn.execute("SELECT count(*) FROM retraining_runs").fetchone()[0] == 0


@pytest.mark.db
def test_only_performance_incidents_trigger_retraining(pg, store, baseline_model):
    _seed_rows(pg, 5)
    _open_incident(pg, baseline_model, kind="quality")

    event = retraining.attempt(
        pg,
        store,
        model_name=MODEL,
        trigger="automatic",
        settings=settings(retrain_min_rows=5, retrain_cooldown_hours=0),
    )

    assert event is None


@pytest.mark.db
def test_disabled_retraining_does_nothing(pg, store, baseline_model):
    _seed_rows(pg, 5)
    _open_incident(pg, baseline_model)

    event = retraining.attempt(
        pg,
        store,
        model_name=MODEL,
        trigger="automatic",
        settings=settings(
            retrain_enabled=False,
            retrain_min_rows=5,
            retrain_cooldown_hours=0,
        ),
    )

    assert event is None
    assert get_production(pg, MODEL).version == baseline_model


@pytest.mark.db
def test_cooldown_suppresses_a_second_automatic_attempt(pg, store, baseline_model):
    _seed_rows(pg, 20)
    _open_incident(pg, baseline_model)

    first = retraining.attempt(
        pg,
        store,
        model_name=MODEL,
        trigger="automatic",
        settings=settings(retrain_min_rows=20, retrain_cooldown_hours=24),
    )
    second = retraining.attempt(
        pg,
        store,
        model_name=MODEL,
        trigger="automatic",
        settings=settings(retrain_min_rows=20, retrain_cooldown_hours=24),
    )

    assert first["decision"] == "registered"
    assert second is None
    with pg.connection() as conn:
        assert conn.execute("SELECT count(*) FROM retraining_runs").fetchone()[0] == 1


@pytest.mark.db
def test_incident_registers_a_candidate_without_changing_production(
    pg, store, baseline_model
):
    _seed_rows(pg, 20)
    _open_incident(pg, baseline_model)

    event = retraining.attempt(
        pg,
        store,
        model_name=MODEL,
        trigger="automatic",
        settings=settings(retrain_min_rows=20, retrain_cooldown_hours=0),
    )

    assert event["decision"] == "registered"
    assert event["trigger"] == "automatic"
    assert event["version"].startswith("auto-")
    assert event["metrics"]["labeled_rows"] == 20
    assert event["metrics"]["train_rows"] == 16
    assert event["metrics"]["holdout_rows"] == 4
    assert event["metrics"]["holdout_mae"] is not None
    assert event["metrics"]["holdout_rmse"] is not None

    registered = get_version(pg, MODEL, event["version"])
    assert registered.stage == "candidate"
    assert get_production(pg, MODEL).version == baseline_model
    assert registered.metrics["test_rows"] == 4


@pytest.mark.db
def test_training_failure_is_recorded(pg, store, baseline_model, monkeypatch):
    _seed_rows(pg, 20)
    _open_incident(pg, baseline_model)

    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(retraining, "build_candidate", explode)

    event = retraining.attempt(
        pg,
        store,
        model_name=MODEL,
        trigger="automatic",
        settings=settings(retrain_min_rows=20, retrain_cooldown_hours=0),
    )

    assert event["decision"] == "failed"
    assert event["version"] is None
    assert event["metrics"]["error"] == "boom"
    assert "training failed: boom" in event["reasons"][-1]
    assert get_production(pg, MODEL).version == baseline_model


@pytest.mark.db
def test_manual_retraining_needs_no_incident(pg, store, baseline_model):
    _seed_rows(pg, 20)

    event = retraining.attempt(
        pg,
        store,
        model_name=MODEL,
        trigger="manual",
        settings=settings(retrain_min_rows=20),
    )

    assert event["decision"] == "registered"
    assert event["trigger"] == "manual"
    assert get_production(pg, MODEL).version == baseline_model


@pytest.mark.db
def test_retrain_endpoint_registers_a_candidate(pg, store, baseline_model):
    _seed_rows(pg, 20)

    with TestClient(app) as client:
        state_settings = client.app.state.settings
        original_enabled = state_settings.retrain_enabled
        original_min_rows = state_settings.retrain_min_rows
        state_settings.retrain_enabled = False
        state_settings.retrain_min_rows = 20
        try:
            response = client.post("/retrain")
            ready = client.get("/ready").json()
        finally:
            state_settings.retrain_enabled = original_enabled
            state_settings.retrain_min_rows = original_min_rows

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "registered"
    assert body["trigger"] == "manual"
    assert body["version"].startswith("auto-")
    assert ready["model_version"] == baseline_model


@pytest.mark.db
def test_retrain_endpoint_reports_a_skip(pg, store, baseline_model):
    with TestClient(app) as client:
        state_settings = client.app.state.settings
        original_min_rows = state_settings.retrain_min_rows
        state_settings.retrain_min_rows = 10
        try:
            response = client.post("/retrain")
        finally:
            state_settings.retrain_min_rows = original_min_rows

    assert response.status_code == 409
    body = response.json()["detail"]
    assert body["message"] == "retraining skipped"
    assert "need 10" in body["reasons"][-1]


@pytest.mark.db
def test_retrain_endpoint_requires_the_token(baseline_model):
    with TestClient(app) as client:
        client.app.state.settings.api_token = "s3cret"
        try:
            assert client.post("/retrain").status_code == 401
        finally:
            client.app.state.settings.api_token = ""


@pytest.mark.db
def test_list_runs_returns_newest_first(pg, baseline_model):
    retraining.record(
        pg,
        model_name=MODEL,
        trigger="manual",
        decision="skipped",
        version=None,
        reasons=["first"],
        metrics={},
    )
    retraining.record(
        pg,
        model_name=MODEL,
        trigger="manual",
        decision="skipped",
        version=None,
        reasons=["second"],
        metrics={},
    )

    with TestClient(app) as client:
        response = client.get("/retrain/runs?limit=10")

    assert response.status_code == 200
    runs = response.json()["runs"]
    assert [run["reasons"][0] for run in runs] == ["second", "first"]


@pytest.mark.db
def test_evaluate_exposes_automatic_retraining(pg, store, baseline_model):
    _seed_rows(pg, 20)
    _open_incident(pg, baseline_model)

    with TestClient(app) as client:
        state_settings = client.app.state.settings
        original = {
            "retrain_min_rows": state_settings.retrain_min_rows,
            "retrain_cooldown_hours": state_settings.retrain_cooldown_hours,
            "rollback_enabled": state_settings.rollback_enabled,
        }
        state_settings.retrain_min_rows = 20
        state_settings.retrain_cooldown_hours = 0
        state_settings.rollback_enabled = False
        try:
            response = client.post("/evaluate")
            ready = client.get("/ready").json()
        finally:
            for key, value in original.items():
                setattr(state_settings, key, value)

    assert response.status_code == 200
    body = response.json()
    assert body["retraining"]["decision"] == "registered"
    assert body["retraining"]["version"].startswith("auto-")
    assert ready["model_version"] == baseline_model
