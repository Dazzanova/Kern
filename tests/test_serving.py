import math
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from kern.serving.app import app

pytestmark = pytest.mark.db


@pytest.fixture(autouse=True)
def use_baseline(baseline_model):
    """Every test in this module needs a promoted model to start the app."""
    return baseline_model


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction(baseline_model):
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "timestamp": "2025-04-01T12:00:00",
                "temperature_c": 25.0,
                "humidity_pct": 55.0,
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["model_version"] == baseline_model
    assert body["prediction_mw"] > 0
    assert body["inference_latency_ms"] >= 0
    assert body["request_id"]


def test_invalid_humidity():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "timestamp": "2025-04-01T12:00:00",
                "temperature_c": 25.0,
                "humidity_pct": 150.0,
            },
        )

    assert response.status_code == 422


def test_batch_prediction(baseline_model):
    records = [
        {"timestamp": "2025-04-01T12:00:00", "temperature_c": 25.0, "humidity_pct": 55.0},
        {"timestamp": "2025-04-01T13:00:00", "temperature_c": 26.0, "humidity_pct": 50.0},
        {"timestamp": "2025-04-01T14:00:00", "temperature_c": 24.0, "humidity_pct": 60.0},
    ]
    with TestClient(app) as client:
        response = client.post("/predict/batch", json={"records": records})

    assert response.status_code == 200
    body = response.json()
    assert body["batch_size"] == 3
    assert len(body["predictions"]) == 3
    assert body["model_version"] == baseline_model
    assert body["request_id"]
    assert body["total_processing_time_ms"] >= 0


def test_batch_prediction_consistency():
    record = {"timestamp": "2025-04-01T12:00:00", "temperature_c": 25.0, "humidity_pct": 55.0}
    with TestClient(app) as client:
        # Single request
        res_single = client.post("/predict", json=record)
        # Batch request
        res_batch = client.post("/predict/batch", json={"records": [record]})

    assert res_single.status_code == 200
    assert res_batch.status_code == 200

    val_single = res_single.json()["prediction_mw"]
    val_batch = res_batch.json()["predictions"][0]

    assert abs(val_single - val_batch) < 1e-5


def test_batch_single_record():
    record = {"timestamp": "2025-04-01T12:00:00", "temperature_c": 25.0, "humidity_pct": 55.0}
    with TestClient(app) as client:
        response = client.post("/predict/batch", json={"records": [record]})

    assert response.status_code == 200
    assert response.json()["batch_size"] == 1


def test_batch_empty_rejected():
    with TestClient(app) as client:
        response = client.post("/predict/batch", json={"records": []})

    assert response.status_code == 422


def test_batch_too_large_rejected():
    records = [{"timestamp": "2025-04-01T12:00:00", "temperature_c": 25.0, "humidity_pct": 55.0}] * 257
    with TestClient(app) as client:
        response = client.post("/predict/batch", json={"records": records})

    assert response.status_code == 422


def test_batch_invalid_record_rejected():
    records = [
        {"timestamp": "2025-04-01T12:00:00", "temperature_c": 25.0, "humidity_pct": 55.0},
        {"timestamp": "2025-04-01T13:00:00", "temperature_c": 26.0, "humidity_pct": 150.0}, # Invalid
    ]
    with TestClient(app) as client:
        response = client.post("/predict/batch", json={"records": records})

    assert response.status_code == 422


def test_batch_order_preserved():
    records = [
        {"timestamp": "2025-04-01T12:00:00", "temperature_c": 10.0, "humidity_pct": 50.0},
        {"timestamp": "2025-04-01T13:00:00", "temperature_c": 40.0, "humidity_pct": 40.0},
    ]
    with TestClient(app) as client:
        response = client.post("/predict/batch", json={"records": records})

    body = response.json()
    # Since temperatures are very different, predictions should be different
    assert body["predictions"][0] != body["predictions"][1]

    # Verify order by comparing with single requests
    for i, record in enumerate(records):
        res_single = client.post("/predict", json=record)
        assert abs(res_single.json()["prediction_mw"] - body["predictions"][i]) < 1e-5


def test_prediction_reports_promoted_version(baseline_model):
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "timestamp": "2025-04-01T12:00:00",
                "temperature_c": 25.0,
                "humidity_pct": 55.0,
            },
        )

    assert response.status_code == 200
    assert response.json()["model_version"] == baseline_model


def test_models_endpoint_lists_versions(baseline_model):
    with TestClient(app) as client:
        response = client.get("/models")

    assert response.status_code == 200
    versions = response.json()["versions"]
    assert any(v["version"] == baseline_model for v in versions)
    assert any(v["stage"] == "production" for v in versions)


def test_promote_unknown_model_returns_404(baseline_model):
    with TestClient(app) as client:
        response = client.post("/models/load-forecast/does-not-exist/promote")

    assert response.status_code == 404


def test_promote_switches_production(baseline_model):
    with TestClient(app) as client:
        response = client.post(f"/models/load-forecast/{baseline_model}/promote")

    assert response.status_code == 200
    assert response.json()["stage"] == "production"


def test_telemetry_summary_counts_requests(baseline_model):
    with TestClient(app) as client:
        client.post("/predict", json={
            "timestamp": "2025-04-01T12:00:00",
            "temperature_c": 25.0,
            "humidity_pct": 55.0,
        })
        client.app.state.writer.drain_once()

        response = client.get("/telemetry/summary?window=1h")

    assert response.status_code == 200
    body = response.json()
    assert body["requests"] >= 1
    assert body["records"] >= 1
    assert body["dropped_events"] >= 0


def test_telemetry_summary_rejects_bad_window(baseline_model):
    with TestClient(app) as client:
        response = client.get("/telemetry/summary?window=forever")

    assert response.status_code == 400


def test_quality_report_returns_checks(baseline_model):
    with TestClient(app) as client:
        client.post("/predict", json={
            "timestamp": "2025-04-01T12:00:00",
            "temperature_c": 25.0,
            "humidity_pct": 55.0,
        })
        client.app.state.writer.drain_once()

        response = client.get("/quality/report?window=1h")

    assert response.status_code == 200
    body = response.json()
    assert body["checks"]
    assert {c["status"] for c in body["checks"]} <= {"pass", "warn", "fail"}


def test_telemetry_failure_does_not_fail_prediction():
    def boom(event):
        raise RuntimeError("recorder exploded")

    with TestClient(app) as client:
        client.app.state.recorder.record = boom
        response = client.post(
            "/predict",
            json={
                "timestamp": "2025-04-01T12:00:00",
                "temperature_c": 25.0,
                "humidity_pct": 55.0,
            },
        )

    assert response.status_code == 200
    prediction = response.json()["prediction_mw"]
    assert math.isfinite(prediction)


def test_telemetry_summary_reports_failed_events(baseline_model):
    with TestClient(app) as client:
        response = client.get("/telemetry/summary?window=1h")

    assert response.status_code == 200
    assert response.json()["failed_events"] == 0


def test_telemetry_summary_rejects_huge_window(baseline_model):
    with TestClient(app) as client:
        response = client.get("/telemetry/summary?window=999999999999d")

    assert response.status_code == 400


def test_quality_report_rejects_huge_window(baseline_model):
    with TestClient(app) as client:
        response = client.get("/quality/report?window=999999999999d")

    assert response.status_code == 400


def test_window_over_max_is_rejected(baseline_model):
    with TestClient(app) as client:
        response = client.get("/telemetry/summary?window=400d")

    assert response.status_code == 400


def test_window_at_max_is_accepted(baseline_model):
    with TestClient(app) as client:
        response = client.get("/telemetry/summary?window=365d")

    assert response.status_code == 200


def test_ready_reports_dependencies(baseline_model):
    with TestClient(app) as client:
        response = client.get("/ready")

        assert response.status_code == 200
        body = response.json()
        assert body["ready"] is True
        assert body["database"] is True
        assert body["object_store"] is True
        assert body["model_version"] == baseline_model

        def boom():
            raise RuntimeError("object store down")

        client.app.state.store.ensure_bucket = boom
        response = client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["ready"] is False
    assert body["object_store"] is False


def test_model_detail_returns_metadata(baseline_model):
    with TestClient(app) as client:
        response = client.get(f"/models/load-forecast/{baseline_model}")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "load-forecast"
    assert body["version"] == baseline_model
    assert body["stage"] == "production"
    assert body["artifact_sha256"]
    assert body["feature_schema"]


def test_model_detail_unknown_returns_404(baseline_model):
    with TestClient(app) as client:
        response = client.get("/models/load-forecast/does-not-exist")

    assert response.status_code == 404


def test_quality_report_empty_window_does_not_fabricate_failures(baseline_model):
    """No sampled events yet must not be reported as every feature being null.

    This is the state right after a promotion or restart, while the feature
    sample rate has not produced any snapshots for the loaded version.
    """
    with TestClient(app) as client:
        response = client.get("/quality/report?window=1h")

    assert response.status_code == 200
    body = response.json()
    assert body["sample_count"] == 0
    assert body["checks"] == []


def test_quality_report_ignores_other_model_versions(baseline_model):
    from psycopg.types.json import Jsonb
    from datetime import datetime, timezone

    with TestClient(app) as client:
        pool = client.app.state.pool
        with pool.connection() as conn:
            conn.execute(
                "INSERT INTO prediction_events (request_id, model_name, model_version, "
                "endpoint, batch_size, duration_ms, features) "
                "VALUES (gen_random_uuid(), 'load-forecast', 'other-v1', 'predict', 1, "
                "5.0, %s)",
                (Jsonb({"temperature_c": 20.0}),),
            )

        response = client.get("/quality/report?window=1h")

    assert response.status_code == 200
    body = response.json()
    assert body["sample_count"] == 0
    assert body["checks"] == []


def _register_version(pg, store, tmp_path, version, feature_schema=None, metrics=None):
    """Train a tiny deterministic model and register it under `version`."""
    from kern.data.features import build_features
    from kern.data.generate import generate_dataset
    from kern.quality import profile_reference
    from kern.registry import register
    from xgboost import XGBRegressor

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


PAYLOAD = {
    "timestamp": "2025-04-01T12:00:00",
    "temperature_c": 25.0,
    "humidity_pct": 55.0,
}


def test_promote_hot_swaps_the_serving_model(pg, store, baseline_model, tmp_path):
    """Promotion must take effect in this process, without a restart."""
    _register_version(pg, store, tmp_path, "test-v2")

    try:
        with TestClient(app) as client:
            assert client.get("/ready").json()["model_version"] == baseline_model

            response = client.post("/models/load-forecast/test-v2/promote")
            assert response.status_code == 200

            served = client.post("/predict", json=PAYLOAD).json()
            assert served["model_version"] == "test-v2"
            assert math.isfinite(served["prediction_mw"])
    finally:
        from kern.registry import promote as promote_version

        promote_version(pg, name="load-forecast", version=baseline_model)
        _drop_version(pg, "test-v2")


def test_promote_rejects_a_version_that_cannot_be_loaded(pg, store, baseline_model, tmp_path):
    """A bad artifact must not move the production pointer."""
    _register_version(pg, store, tmp_path, "test-bad", feature_schema=["wrong"])

    try:
        with TestClient(app) as client:
            response = client.post("/models/load-forecast/test-bad/promote")
            assert response.status_code == 409

            still_serving = client.get("/ready").json()["model_version"]
            assert still_serving == baseline_model
    finally:
        _drop_version(pg, "test-bad")


def test_outcomes_upsert_and_feed_the_performance_report(baseline_model):
    target = datetime.now(timezone.utc) + timedelta(hours=1)

    with TestClient(app) as client:
        prediction = client.post("/predict", json={
            "timestamp": target.isoformat(),
            "temperature_c": 25.0,
            "humidity_pct": 55.0,
        }).json()
        client.app.state.writer.drain_once()

        first = client.post("/outcomes", json={"observations": [
            {"target_ts": target.isoformat(), "actual_mw": prediction["prediction_mw"] + 10.0}
        ]})
        assert first.status_code == 200
        assert first.json()["inserted"] == 1
        assert first.json()["updated"] == 0

        second = client.post("/outcomes", json={"observations": [
            {"target_ts": target.isoformat(), "actual_mw": prediction["prediction_mw"] + 20.0}
        ]})
        assert second.json()["inserted"] == 0
        assert second.json()["updated"] == 1

        report = client.get("/performance/report?window=24h").json()

    assert report["model_version"] == baseline_model
    assert report["samples"] == 1
    assert report["mae"] == pytest.approx(20.0)
    assert report["coverage"] == 1.0


def test_outcomes_reject_non_finite_values(baseline_model):
    target = datetime.now(timezone.utc) + timedelta(hours=1)
    # Sent as raw text: the default JSON encoder refuses to serialise infinity,
    # but a hostile or buggy client can still put the literal on the wire.
    body = (
        '{"observations": [{"target_ts": "%s", "actual_mw": Infinity}]}'
        % target.isoformat()
    )

    with TestClient(app) as client:
        response = client.post(
            "/outcomes", content=body, headers={"content-type": "application/json"}
        )

    assert response.status_code == 422


def test_single_predictions_record_the_target_timestamp(baseline_model):
    target = datetime.now(timezone.utc) + timedelta(hours=2)

    with TestClient(app) as client:
        client.post("/predict", json={
            "timestamp": target.isoformat(),
            "temperature_c": 25.0,
            "humidity_pct": 55.0,
        })
        client.app.state.writer.drain_once()

        with client.app.state.pool.connection() as conn:
            row = conn.execute(
                "SELECT target_ts FROM prediction_events ORDER BY recorded_at DESC LIMIT 1"
            ).fetchone()

    assert row[0] is not None
    assert abs((row[0] - target).total_seconds()) < 1


def test_batch_predictions_leave_the_target_empty(baseline_model):
    """One batch event covers many hours and cannot be attributed."""
    with TestClient(app) as client:
        client.post("/predict/batch", json={"records": [PAYLOAD, PAYLOAD]})
        client.app.state.writer.drain_once()

        with client.app.state.pool.connection() as conn:
            row = conn.execute(
                "SELECT target_ts FROM prediction_events ORDER BY recorded_at DESC LIMIT 1"
            ).fetchone()

    assert row[0] is None


def test_promote_is_open_when_no_token_is_configured(baseline_model):
    with TestClient(app) as client:
        assert client.app.state.settings.api_token == ""
        response = client.post(f"/models/load-forecast/{baseline_model}/promote")

    assert response.status_code == 200


def test_state_changing_endpoints_require_the_token(baseline_model):
    target = datetime.now(timezone.utc) + timedelta(hours=1)

    with TestClient(app) as client:
        client.app.state.settings.api_token = "s3cret"
        try:
            assert client.post(
                f"/models/load-forecast/{baseline_model}/promote"
            ).status_code == 401
            assert client.post(
                f"/models/load-forecast/{baseline_model}/promote",
                headers={"authorization": "Bearer wrong"},
            ).status_code == 401
            assert client.post(
                "/outcomes",
                json={"observations": [
                    {"target_ts": target.isoformat(), "actual_mw": 1.0}
                ]},
            ).status_code == 401

            ok = client.post(
                f"/models/load-forecast/{baseline_model}/promote",
                headers={"authorization": "Bearer s3cret"},
            )
            assert ok.status_code == 200
        finally:
            client.app.state.settings.api_token = ""


def test_drift_report_reports_insufficient_data_on_an_empty_window(baseline_model):
    with TestClient(app) as client:
        response = client.get("/drift/report?window=1h")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "insufficient_data"
    assert body["sample_size"] == 0
    assert body["checks"] == []


def test_drift_report_requires_a_reference_sample(baseline_model):
    from dataclasses import replace

    with TestClient(app) as client:
        client.app.state.current = replace(
            client.app.state.current, reference_sample=None
        )
        response = client.get("/drift/report?window=1h")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "no_reference_sample"
    assert body["checks"] == []


def test_drift_report_computes_and_persists_psi_and_ks(baseline_model):
    with TestClient(app) as client:
        client.app.state.settings.drift_min_samples = 1
        try:
            client.post("/predict", json=PAYLOAD)
            client.app.state.writer.drain_once()

            response = client.get("/drift/report?window=1h")

            with client.app.state.pool.connection() as conn:
                persisted = conn.execute(
                    "SELECT count(*) FROM drift_checks WHERE model_version = %s",
                    (baseline_model,),
                ).fetchone()[0]
        finally:
            client.app.state.settings.drift_min_samples = 100

    assert response.status_code == 200
    body = response.json()
    assert body["sample_size"] == 1
    assert body["status"] in {"pass", "warn", "fail"}
    assert {check["method"] for check in body["checks"]} == {"psi", "ks"}
    # Time-derived features (hour, month, ...) are excluded: they encode the
    # clock, not the input distribution. Only the 2 exogenous features remain.
    assert {check["feature"] for check in body["checks"]} == {
        "temperature_c",
        "humidity_pct",
    }
    assert len(body["checks"]) == 4  # 2 features x 2 methods
    assert persisted == 4


def test_evaluate_runs_a_cycle_and_records_a_drift_incident(baseline_model):
    with TestClient(app) as client:
        client.app.state.settings.drift_min_samples = 1
        try:
            client.post("/predict", json={
                "timestamp": "2025-04-01T12:00:00",
                "temperature_c": 80.0,
                "humidity_pct": 99.0,
            })
            client.app.state.writer.drain_once()

            response = client.post("/evaluate")

            incidents = client.get("/incidents").json()["incidents"]
        finally:
            client.app.state.settings.drift_min_samples = 100

    assert response.status_code == 200
    body = response.json()
    assert body["findings"] >= 1
    assert body["quality"]["sample_count"] == 1

    kinds = {incident["kind"] for incident in incidents}
    assert "drift" in kinds
    assert all(incident["status"] == "open" for incident in incidents)


def test_evaluate_requires_the_token_when_configured(baseline_model):
    with TestClient(app) as client:
        client.app.state.settings.api_token = "s3cret"
        try:
            assert client.post("/evaluate").status_code == 401
        finally:
            client.app.state.settings.api_token = ""


def test_incidents_endpoint_filters_and_orders_by_last_seen(baseline_model):
    from kern.incidents import Finding, sync_incidents

    with TestClient(app) as client:
        pool = client.app.state.pool
        first = Finding("drift", "temperature_c", "warn", "a", {})
        second = Finding("drift", "humidity_pct", "warn", "b", {})
        sync_incidents(
            pool,
            model_name="load-forecast",
            model_version=baseline_model,
            findings=[first, second],
            evaluated_kinds={"drift"},
        )
        sync_incidents(
            pool,
            model_name="load-forecast",
            model_version=baseline_model,
            findings=[first],
            evaluated_kinds={"drift"},
        )

        open_rows = client.get("/incidents").json()["incidents"]
        all_rows = client.get("/incidents?status=all").json()["incidents"]

    assert [row["subject"] for row in open_rows] == ["temperature_c"]
    assert {row["subject"] for row in all_rows} == {"temperature_c", "humidity_pct"}
    assert all(row["kind"] == "drift" for row in all_rows)


def test_incidents_endpoint_rejects_an_unknown_status(baseline_model):
    with TestClient(app) as client:
        assert client.get("/incidents?status=bogus").status_code == 400


def test_promote_blocks_a_worse_candidate_and_keeps_production(pg, store, baseline_model, tmp_path):
    _register_version(pg, store, tmp_path, "test-worse", metrics={"mae": 500.0})

    try:
        with TestClient(app) as client:
            response = client.post("/models/load-forecast/test-worse/promote")
            served = client.get("/ready").json()["model_version"]

        assert response.status_code == 409
        detail = response.json()["detail"]
        assert detail["production_version"] == baseline_model
        assert any("does not improve" in reason for reason in detail["reasons"])
        assert served == baseline_model

        with pg.connection() as conn:
            row = conn.execute(
                "SELECT decision, reasons FROM promotion_decisions ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert row[0] == "blocked"
        assert any("does not improve" in reason for reason in row[1])
    finally:
        _drop_version(pg, "test-worse")


def test_forced_promotion_overrides_the_gate_and_is_recorded(pg, store, baseline_model, tmp_path):
    _register_version(pg, store, tmp_path, "test-forced", metrics={"mae": 500.0})

    try:
        with TestClient(app) as client:
            response = client.post("/models/load-forecast/test-forced/promote?force=true")
            assert response.status_code == 200
            assert response.json()["decision"] == "forced"
            assert client.get("/ready").json()["model_version"] == "test-forced"

        with pg.connection() as conn:
            row = conn.execute(
                "SELECT decision, reasons FROM promotion_decisions ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert row[0] == "forced"
        assert any("does not improve" in reason for reason in row[1])
    finally:
        from kern.registry import promote as promote_version

        promote_version(pg, name="load-forecast", version=baseline_model)
        _drop_version(pg, "test-forced")


def test_open_critical_incident_blocks_an_otherwise_better_candidate(
    pg, store, baseline_model, tmp_path
):
    from kern.incidents import Finding, sync_incidents

    _register_version(pg, store, tmp_path, "test-gated", metrics={"mae": 1.0})

    try:
        sync_incidents(
            pg,
            model_name="load-forecast",
            model_version=baseline_model,
            findings=[Finding("drift", "temperature_c", "critical", "PSI 0.9", {})],
            evaluated_kinds=set(),
        )

        with TestClient(app) as client:
            blocked = client.post("/models/load-forecast/test-gated/promote")
            assert blocked.status_code == 409
            assert any(
                "open critical incident" in reason
                for reason in blocked.json()["detail"]["reasons"]
            )
            assert client.get("/ready").json()["model_version"] == baseline_model

        sync_incidents(
            pg,
            model_name="load-forecast",
            model_version=baseline_model,
            findings=[],
            evaluated_kinds={"drift"},
        )

        with TestClient(app) as client:
            allowed = client.post("/models/load-forecast/test-gated/promote")
            assert allowed.status_code == 200
            assert allowed.json()["decision"] == "allowed"
    finally:
        from kern.registry import promote as promote_version

        promote_version(pg, name="load-forecast", version=baseline_model)
        _drop_version(pg, "test-gated")
