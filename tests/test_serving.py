import math

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
