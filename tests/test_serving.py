from fastapi.testclient import TestClient

from kern.serving.app import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction():
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

    assert body["model_version"] == "baseline"
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


def test_batch_prediction():
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
    assert body["model_version"] == "baseline"
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
