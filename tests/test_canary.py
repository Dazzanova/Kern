from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from kern import canary
from kern.canary import build_report, should_use_canary, verdict
from kern.serving.app import ModelState, app

PAYLOAD = {
    "timestamp": "2025-04-01T12:00:00",
    "temperature_c": 25.0,
    "humidity_pct": 55.0,
}


def _register_version(pg, store, tmp_path, version, feature_schema=None, metrics=None):
    """Train a tiny deterministic model and register it under `version`."""
    from xgboost import XGBRegressor

    from kern.data.features import build_features
    from kern.data.generate import generate_dataset
    from kern.quality import profile_reference
    from kern.registry import register

    df = generate_dataset("2025-01-01", 200, 7)
    X = build_features(df)
    model = XGBRegressor(n_estimators=5, max_depth=2, random_state=7)
    model.fit(X, df["load_mw"])

    path = tmp_path / f"{version}.json"
    model.save_model(path)

    return register(
        pg,
        store,
        name="load-forecast",
        version=version,
        artifact_path=path,
        metrics=metrics or {"mae": 1.0},
        feature_schema=feature_schema or list(X.columns),
        reference_profile=profile_reference(X, X.columns),
    )


def _drop_version(pg, version):
    with pg.connection() as conn:
        conn.execute(
            "DELETE FROM model_versions WHERE name = 'load-forecast' AND version = %s",
            (version,),
        )


def test_should_use_canary_is_a_strict_draw():
    assert should_use_canary(0.3, lambda: 0.0) is True
    assert should_use_canary(0.3, lambda: 0.29) is True
    assert should_use_canary(0.3, lambda: 0.3) is False
    assert should_use_canary(0.3, lambda: 1.0) is False


def summary(mae, samples=50):
    return {"mae": mae, "rmse": None, "samples": samples, "coverage": 1.0}


def test_verdict_compares_mae_and_respects_the_sample_gate():
    assert verdict(summary(10.0), summary(9.0), min_samples=20) == "canary_better"
    assert verdict(summary(10.0), summary(11.0), min_samples=20) == "canary_worse"
    assert verdict(summary(10.0), summary(10.0), min_samples=20) == "tie"
    assert verdict(summary(10.0, 5), summary(9.0, 5), min_samples=20) == "inconclusive"


def test_verdict_is_inconclusive_without_an_mae():
    assert verdict(summary(None), summary(9.0), min_samples=20) == "inconclusive"
    assert verdict(summary(10.0), summary(None), min_samples=20) == "inconclusive"


def test_report_without_an_assignment_is_explicit():
    report = build_report(
        None,
        model_name="load-forecast",
        canary_version=None,
        production_version="v1",
        fraction=None,
        window="24h",
        since=datetime.now(timezone.utc),
        min_samples=20,
    )

    assert report["canary_version"] is None
    assert report["fraction"] is None
    assert report["served"] == {"production": 0, "canary": 0}
    assert report["observed_fraction"] is None
    assert report["fallbacks"] == 0
    assert report["production"] is None
    assert report["verdict"] == "inconclusive"


@pytest.mark.db
def test_attach_persists_fraction_and_detach_is_idempotent(pg):
    canary.attach(pg, model_name="load-forecast", version="cand-a", fraction=0.2)
    canary.attach(pg, model_name="load-forecast", version="cand-b", fraction=0.4)

    assignment = canary.get_assignment(pg, "load-forecast")
    assert assignment["version"] == "cand-b"
    assert assignment["fraction"] == 0.4

    with pg.connection() as conn:
        assert conn.execute("SELECT count(*) FROM canary_assignments").fetchone()[0] == 1

    assert canary.detach(pg, "load-forecast") is True
    assert canary.detach(pg, "load-forecast") is False
    assert canary.get_assignment(pg, "load-forecast") is None


@pytest.mark.db
def test_report_separates_versions_and_counts_fallbacks(pg):
    now = datetime.now(timezone.utc)
    t1, t2 = (now + timedelta(hours=n) for n in (1, 2))

    with pg.connection() as conn:
        conn.execute(
            "INSERT INTO prediction_events (request_id, model_name, model_version, "
            "endpoint, batch_size, duration_ms, prediction_mean, target_ts, "
            "canary_fallback) VALUES "
            "(%s, 'load-forecast', 'v1', 'predict', 1, 1.0, 100.0, %s, false), "
            "(%s, 'load-forecast', 'v1', 'predict', 1, 1.0, 300.0, %s, true), "
            "(%s, 'load-forecast', 'cand', 'predict', 1, 1.0, 90.0, %s, false)",
            (uuid4(), t1, uuid4(), t2, uuid4(), t1),
        )
        conn.execute(
            "INSERT INTO observed_outcomes (model_name, target_ts, actual_mw) VALUES "
            "('load-forecast', %s, 105.0), ('load-forecast', %s, 205.0)",
            (t1, t2),
        )

    report = build_report(
        pg,
        model_name="load-forecast",
        canary_version="cand",
        production_version="v1",
        fraction=0.3,
        window="24h",
        since=now - timedelta(minutes=5),
        min_samples=1,
    )

    assert report["served"] == {"production": 2, "canary": 1}
    assert report["observed_fraction"] == 1 / 3
    assert report["fallbacks"] == 1
    assert report["production"]["mae"] == 50.0
    assert report["canary"]["mae"] == 15.0
    assert report["canary"]["samples"] == 1
    assert report["verdict"] == "canary_better"


@pytest.mark.db
def test_report_is_inconclusive_below_the_sample_gate(pg):
    now = datetime.now(timezone.utc)
    target = now + timedelta(hours=1)

    with pg.connection() as conn:
        conn.execute(
            "INSERT INTO prediction_events (request_id, model_name, model_version, "
            "endpoint, batch_size, duration_ms, prediction_mean, target_ts) VALUES "
            "(%s, 'load-forecast', 'v1', 'predict', 1, 1.0, 100.0, %s), "
            "(%s, 'load-forecast', 'cand', 'predict', 1, 1.0, 90.0, %s)",
            (uuid4(), target, uuid4(), target),
        )
        conn.execute(
            "INSERT INTO observed_outcomes (model_name, target_ts, actual_mw) "
            "VALUES ('load-forecast', %s, 95.0)",
            (target,),
        )

    report = build_report(
        pg,
        model_name="load-forecast",
        canary_version="cand",
        production_version="v1",
        fraction=0.5,
        window="24h",
        since=now - timedelta(minutes=5),
        min_samples=5,
    )

    assert report["canary"]["samples"] == 1
    assert report["verdict"] == "inconclusive"


@pytest.mark.db
def test_canary_attach_get_and_detach(pg, store, baseline_model, tmp_path):
    _register_version(pg, store, tmp_path, "test-canary")

    try:
        with TestClient(app) as client:
            attached = client.post(
                "/models/load-forecast/test-canary/canary", params={"fraction": 0.25}
            )
            assert attached.status_code == 200
            assert attached.json()["canary_version"] == "test-canary"
            assert attached.json()["fraction"] == 0.25
            assert attached.json()["shadow_cleared"] is False

            current = client.get("/canary").json()
            assert current["canary_version"] == "test-canary"
            assert current["fraction"] == 0.25

            assert client.delete("/canary").json()["cleared"] is True
            assert client.delete("/canary").json()["cleared"] is False
            assert client.get("/canary").json()["canary_version"] is None
    finally:
        _drop_version(pg, "test-canary")


@pytest.mark.db
def test_canary_rejects_a_fraction_outside_the_open_interval(baseline_model):
    with TestClient(app) as client:
        assert client.post(
            "/models/load-forecast/test-v1/canary", params={"fraction": 0.0}
        ).status_code == 400
        assert client.post(
            "/models/load-forecast/test-v1/canary", params={"fraction": 1.0}
        ).status_code == 400


@pytest.mark.db
def test_canary_rejects_the_production_version(baseline_model):
    with TestClient(app) as client:
        response = client.post(
            "/models/load-forecast/test-v1/canary", params={"fraction": 0.2}
        )

    assert response.status_code == 409
    assert "cannot canary itself" in response.json()["detail"]


@pytest.mark.db
def test_canary_and_shadow_are_mutually_exclusive(
    pg, store, baseline_model, tmp_path
):
    _register_version(pg, store, tmp_path, "test-exclusive")

    try:
        with TestClient(app) as client:
            assert client.post("/models/load-forecast/test-exclusive/shadow").status_code == 200

            attached = client.post(
                "/models/load-forecast/test-exclusive/canary",
                params={"fraction": 0.5},
            )
            assert attached.status_code == 200
            assert attached.json()["shadow_cleared"] is True
            assert client.get("/shadow").json()["shadow_version"] is None

            attached = client.post("/models/load-forecast/test-exclusive/shadow")
            assert attached.status_code == 200
            assert attached.json()["canary_cleared"] is True
            assert client.get("/canary").json()["canary_version"] is None
    finally:
        with pg.connection() as conn:
            conn.execute("DELETE FROM canary_assignments")
            conn.execute("DELETE FROM shadow_assignments")
        _drop_version(pg, "test-exclusive")


@pytest.mark.db
def test_canary_serves_a_selected_request_and_records_its_version(
    pg, store, baseline_model, tmp_path
):
    _register_version(pg, store, tmp_path, "test-canary-route")

    try:
        with TestClient(app) as client:
            client.post(
                "/models/load-forecast/test-canary-route/canary",
                params={"fraction": 0.5},
            )
            client.app.state.canary_rng = lambda: 0.0
            served = client.post("/predict", json=PAYLOAD)
            assert served.status_code == 200
            assert served.json()["model_version"] == "test-canary-route"

            client.app.state.canary_rng = lambda: 0.99
            production = client.post("/predict", json=PAYLOAD)
            assert production.json()["model_version"] == baseline_model

            client.app.state.writer.stop()
            with pg.connection() as conn:
                rows = conn.execute(
                    "SELECT model_version, canary_fallback FROM prediction_events "
                    "ORDER BY id"
                ).fetchall()

        assert rows == [("test-canary-route", False), (baseline_model, False)]
    finally:
        _drop_version(pg, "test-canary-route")


@pytest.mark.db
def test_batch_routing_follows_the_same_draw(pg, store, baseline_model, tmp_path):
    _register_version(pg, store, tmp_path, "test-canary-batch")

    try:
        with TestClient(app) as client:
            client.post(
                "/models/load-forecast/test-canary-batch/canary",
                params={"fraction": 0.5},
            )
            client.app.state.canary_rng = lambda: 0.0
            response = client.post("/predict/batch", json={"records": [PAYLOAD, PAYLOAD]})

            client.app.state.writer.stop()
            with pg.connection() as conn:
                row = conn.execute(
                    "SELECT model_version, endpoint, batch_size FROM prediction_events"
                ).fetchone()

        assert response.status_code == 200
        assert response.json()["model_version"] == "test-canary-batch"
        assert row == ("test-canary-batch", "predict/batch", 2)
    finally:
        _drop_version(pg, "test-canary-batch")


@pytest.mark.db
def test_a_failing_canary_falls_back_to_production(baseline_model):
    class Boom:
        def predict(self, features):
            raise RuntimeError("canary exploded")

    with TestClient(app) as client:
        client.app.state.canary = ModelState(
            name="load-forecast",
            version="broken",
            model=Boom(),
            reference_profile={},
            reference_sample=None,
        )
        client.app.state.canary_fraction = 1.0
        client.app.state.canary_rng = lambda: 0.0

        response = client.post("/predict", json=PAYLOAD)

        client.app.state.writer.stop()
        with client.app.state.pool.connection() as conn:
            row = conn.execute(
                "SELECT model_version, canary_fallback FROM prediction_events"
            ).fetchone()

    assert response.status_code == 200
    assert response.json()["model_version"] == baseline_model
    assert row == (baseline_model, True)


@pytest.mark.db
def test_promoting_the_canary_detaches_it(pg, store, baseline_model, tmp_path):
    _register_version(
        pg, store, tmp_path, "test-canary-promote", metrics={"mae": 1.0}
    )

    try:
        with TestClient(app) as client:
            assert client.post(
                "/models/load-forecast/test-canary-promote/canary",
                params={"fraction": 0.5},
            ).status_code == 200

            assert client.post(
                "/models/load-forecast/test-canary-promote/promote"
            ).status_code == 200
            assert client.get("/canary").json()["canary_version"] is None

        with pg.connection() as conn:
            assert conn.execute("SELECT count(*) FROM canary_assignments").fetchone()[0] == 0
    finally:
        from kern.registry import promote as promote_version

        promote_version(pg, name="load-forecast", version=baseline_model)
        _drop_version(pg, "test-canary-promote")


@pytest.mark.db
def test_canary_report_without_an_assignment(baseline_model):
    with TestClient(app) as client:
        response = client.get("/canary/report")

    assert response.status_code == 200
    body = response.json()
    assert body["canary_version"] is None
    assert body["served"] == {"production": 0, "canary": 0}
    assert body["verdict"] == "inconclusive"


@pytest.mark.db
def test_canary_endpoints_require_the_token(baseline_model):
    with TestClient(app) as client:
        client.app.state.settings.api_token = "s3cret"
        try:
            assert client.post(
                f"/models/load-forecast/{baseline_model}/canary",
                params={"fraction": 0.2},
            ).status_code == 401
            assert client.delete("/canary").status_code == 401
        finally:
            client.app.state.settings.api_token = ""
