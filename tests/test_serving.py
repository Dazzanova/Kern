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
