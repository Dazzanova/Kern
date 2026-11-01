from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from kern import evaluation, scaling
from kern.config import Settings
from kern.serving.app import app

MODEL = "load-forecast"


def settings(**overrides):
    return Settings(_env_file=None, **overrides)


def _seed_requests(
    pg, *, count, recorded_at, model_name=MODEL, version="test-v1"
):
    if count <= 0:
        return
    with pg.connection() as conn, conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO prediction_events (request_id, model_name, model_version, "
            "endpoint, batch_size, duration_ms, prediction_mean, recorded_at) "
            "VALUES (gen_random_uuid(), %s, %s, 'predict', 1, 10.0, 100.0, %s)",
            [(model_name, version, recorded_at)] * count,
        )


def _seed_rising_buckets(pg, *, now, step_minutes=5):
    boundary = now.replace(
        minute=now.minute - now.minute % step_minutes,
        second=0,
        microsecond=0,
    )
    for offset, count in ((3, 10), (2, 30), (1, 60)):
        start = boundary - timedelta(minutes=step_minutes * offset)
        _seed_requests(
            pg, count=count, recorded_at=start + timedelta(minutes=1)
        )


def test_bucket_requests_aligns_and_excludes_the_partial_bucket():
    now = datetime(2026, 1, 1, 12, 0, 30, tzinfo=timezone.utc)
    rows = [
        datetime(2026, 1, 1, 11, 46, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 11, 47, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 11, 48, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 11, 51, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 11, 59, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 11, 59, 30, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 12, 0, 10, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 11, 44, tzinfo=timezone.utc),
    ]

    buckets = scaling.bucket_requests(
        rows, now=now, step_minutes=5, lookback_minutes=15
    )

    assert [bucket["requests"] for bucket in buckets] == [3, 1, 2]
    assert [bucket["rps"] for bucket in buckets] == [3 / 300, 1 / 300, 2 / 300]
    assert buckets[0]["starts_at"] == "2026-01-01T11:45:00+00:00"
    assert buckets[-1]["starts_at"] == "2026-01-01T11:55:00+00:00"


def test_forecast_tracks_a_trend_and_stays_non_negative():
    rising = scaling.forecast_rps([0.1, 0.2, 0.3], horizon=3)
    falling = scaling.forecast_rps([0.3, 0.2, 0.1], horizon=3)
    flat = scaling.forecast_rps([0.5, 0.5, 0.5], horizon=3)

    assert rising == pytest.approx([0.4, 0.5, 0.6])
    assert rising[-1] > 0.3
    assert all(value >= 0 for value in falling)
    assert flat == pytest.approx([0.5, 0.5, 0.5])
    assert scaling.forecast_rps([], horizon=2) == [0.0, 0.0]


def test_replicas_for_rate_uses_utilization_and_clamps():
    common = {
        "rps_per_replica": 90.0,
        "target_utilization": 0.7,
        "min_replicas": 1,
        "max_replicas": 10,
    }

    assert scaling.replicas_for_rate(0.0, **common) == 1
    assert scaling.replicas_for_rate(63.0, **common) == 1
    assert scaling.replicas_for_rate(63.1, **common) == 2
    assert scaling.replicas_for_rate(10_000.0, **common) == 10
    assert (
        scaling.replicas_for_rate(
            100.0,
            rps_per_replica=90.0,
            target_utilization=0.7,
            min_replicas=4,
            max_replicas=2,
        )
        == 4
    )


def test_build_plan_reports_insufficient_data():
    buckets = [
        {
            "starts_at": "2026-01-01T11:45:00+00:00",
            "requests": 1,
            "rps": 1 / 300,
        },
        {
            "starts_at": "2026-01-01T11:50:00+00:00",
            "requests": 0,
            "rps": 0.0,
        },
        {
            "starts_at": "2026-01-01T11:55:00+00:00",
            "requests": 2,
            "rps": 2 / 300,
        },
    ]

    plan = scaling.build_plan(
        buckets,
        step_minutes=5,
        horizon_minutes=30,
        min_buckets=3,
        rps_per_replica=90.0,
        target_utilization=0.7,
        min_replicas=1,
        max_replicas=10,
    )

    assert plan["status"] == "insufficient_data"
    assert plan["recommended_replicas"] is None
    assert plan["forecast"] == []
    assert plan["observed_buckets"] == 2
    assert "need 3" in plan["reasons"][0]


def test_build_plan_sizes_the_peak_across_the_horizon():
    now = datetime(2026, 1, 1, 12, 0, 30, tzinfo=timezone.utc)
    buckets = scaling.bucket_requests(
        [
            datetime(2026, 1, 1, 11, 46, tzinfo=timezone.utc),
        ]
        * 10
        + [datetime(2026, 1, 1, 11, 51, tzinfo=timezone.utc)] * 30
        + [datetime(2026, 1, 1, 11, 56, tzinfo=timezone.utc)] * 60,
        now=now,
        step_minutes=5,
        lookback_minutes=15,
    )

    plan = scaling.build_plan(
        buckets,
        step_minutes=5,
        horizon_minutes=30,
        min_buckets=3,
        rps_per_replica=0.1,
        target_utilization=0.7,
        min_replicas=1,
        max_replicas=10,
    )

    assert plan["status"] == "ok"
    assert plan["observed_buckets"] == 3
    assert len(plan["forecast"]) == 6
    assert plan["peak_forecast_rps"] == max(
        bucket["rps"] for bucket in plan["forecast"]
    )
    assert plan["recommended_replicas"] == max(
        bucket["replicas"] for bucket in plan["forecast"]
    )
    assert plan["recommended_replicas"] > 1


@pytest.mark.db
def test_plan_and_record_persists_a_plan(pg, baseline_model):
    now = datetime(2026, 1, 1, 12, 0, 30, tzinfo=timezone.utc)
    _seed_rising_buckets(pg, now=now)

    plan = scaling.plan_and_record(
        pg,
        model_name=MODEL,
        settings=settings(scaling_rps_per_replica=0.1),
        now=now,
    )

    assert plan["status"] == "ok"
    assert plan["observed_buckets"] == 3
    assert plan["total_buckets"] == 12
    assert len(plan["forecast"]) == 12
    assert plan["recommended_replicas"] > 1
    assert plan["peak_forecast_rps"] > 0

    latest = scaling.latest_plan(pg, MODEL)
    assert latest is not None
    assert latest["id"] == plan["id"]
    assert latest["model_name"] == MODEL
    assert latest["status"] == "ok"
    assert len(scaling.list_plans(pg, MODEL)) == 1


@pytest.mark.db
def test_no_telemetry_records_an_insufficient_plan(pg, baseline_model):
    now = datetime(2026, 1, 1, 12, 0, 30, tzinfo=timezone.utc)

    plan = scaling.plan_and_record(
        pg, model_name=MODEL, settings=settings(), now=now
    )

    assert plan["status"] == "insufficient_data"
    assert plan["recommended_replicas"] is None
    assert plan["observed_buckets"] == 0
    assert scaling.latest_plan(pg, MODEL)["status"] == "insufficient_data"


@pytest.mark.db
def test_disabled_scaling_writes_nothing(pg, baseline_model):
    now = datetime(2026, 1, 1, 12, 0, 30, tzinfo=timezone.utc)
    _seed_rising_buckets(pg, now=now)

    result = scaling.plan_and_record(
        pg,
        model_name=MODEL,
        settings=settings(scaling_enabled=False),
        now=now,
    )

    assert result is None
    with pg.connection() as conn:
        assert conn.execute("SELECT count(*) FROM scaling_plans").fetchone()[0] == 0


@pytest.mark.db
def test_scaling_plan_endpoint_returns_404_without_a_plan(pg, baseline_model):
    with TestClient(app) as client:
        response = client.get("/scaling/plan")

    assert response.status_code == 404


@pytest.mark.db
def test_scaling_endpoints_return_latest_and_history(pg, baseline_model):
    now = datetime(2026, 1, 1, 12, 0, 30, tzinfo=timezone.utc)
    _seed_rising_buckets(pg, now=now)
    first = scaling.plan_and_record(
        pg,
        model_name=MODEL,
        settings=settings(scaling_rps_per_replica=0.1),
        now=now,
    )
    second = scaling.plan_and_record(
        pg,
        model_name=MODEL,
        settings=settings(scaling_rps_per_replica=0.1),
        now=now,
    )

    with TestClient(app) as client:
        latest = client.get("/scaling/plan")
        history = client.get("/scaling/plans?limit=10")

    assert latest.status_code == 200
    assert latest.json()["id"] == second["id"]
    assert history.status_code == 200
    assert [plan["id"] for plan in history.json()["plans"]] == [
        second["id"],
        first["id"],
    ]


@pytest.mark.db
def test_evaluate_exposes_and_persists_scaling(pg, baseline_model):
    now = datetime.now(timezone.utc)
    _seed_rising_buckets(pg, now=now)

    with TestClient(app) as client:
        state_settings = client.app.state.settings
        original = {
            "scaling_rps_per_replica": state_settings.scaling_rps_per_replica,
            "scaling_min_buckets": state_settings.scaling_min_buckets,
        }
        state_settings.scaling_rps_per_replica = 0.1
        state_settings.scaling_min_buckets = 3
        try:
            response = client.post("/evaluate")
            latest = client.get("/scaling/plan")
        finally:
            for key, value in original.items():
                setattr(state_settings, key, value)

    assert response.status_code == 200
    body = response.json()
    assert body["scaling"]["status"] == "ok"
    assert body["scaling"]["recommended_replicas"] > 1
    assert latest.status_code == 200
    assert latest.json()["id"] == body["scaling"]["id"]


@pytest.mark.db
def test_retention_prunes_old_scaling_plans(pg, baseline_model):
    now = datetime(2026, 1, 1, 12, 0, 30, tzinfo=timezone.utc)
    _seed_rising_buckets(pg, now=now)
    scaling.plan_and_record(
        pg,
        model_name=MODEL,
        settings=settings(scaling_rps_per_replica=0.1),
        now=now,
    )
    with pg.connection() as conn:
        conn.execute(
            "UPDATE scaling_plans SET created_at = now() - interval '40 days'"
        )

    deleted = evaluation.run_retention(pg, 30)

    assert deleted["scaling_plans"] == 1
    with pg.connection() as conn:
        assert conn.execute("SELECT count(*) FROM scaling_plans").fetchone()[0] == 0
