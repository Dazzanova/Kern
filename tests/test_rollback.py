from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from psycopg.types.json import Jsonb

from kern import rollback
from kern.config import Settings
from kern.registry import get_production, promote
from kern.serving.app import app

pytestmark = pytest.mark.db


def settings(**overrides):
    return Settings(_env_file=None, **overrides)


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


def _register_narrow_version(pg, store, tmp_path, version):
    """A self-consistent artifact trained on a different feature set."""
    import pandas as pd
    from xgboost import XGBRegressor

    from kern.data.generate import generate_dataset
    from kern.quality import profile_reference
    from kern.registry import register

    df = generate_dataset("2025-01-01", 100, 3)
    X = pd.DataFrame({"temperature_c": df["temperature_c"]})
    model = XGBRegressor(n_estimators=5, max_depth=2, random_state=3)
    model.fit(X, df["load_mw"])

    path = tmp_path / f"{version}.json"
    model.save_model(path)

    return register(
        pg,
        store,
        name="load-forecast",
        version=version,
        artifact_path=path,
        metrics={"mae": 1.0},
        feature_schema=["temperature_c"],
        reference_profile=profile_reference(X, ["temperature_c"]),
    )


def _drop_version(pg, version):
    with pg.connection() as conn:
        conn.execute(
            "DELETE FROM model_versions WHERE name = 'load-forecast' AND version = %s",
            (version,),
        )


def _make_previous(pg, store, tmp_path, baseline, version="test-previous"):
    """Register a version, promote it, then restore the baseline above it."""
    _register_version(pg, store, tmp_path, version)
    promote(pg, name="load-forecast", version=version)
    promote(pg, name="load-forecast", version=baseline)
    return version


def _cleanup(pg, baseline, *versions):
    promote(pg, name="load-forecast", version=baseline)
    for version in versions:
        _drop_version(pg, version)


def _open_incident(
    pg,
    version,
    *,
    kind="performance",
    severity="critical",
    first_seen=None,
):
    first_seen = first_seen or datetime.now(timezone.utc)
    with pg.connection() as conn:
        conn.execute(
            "INSERT INTO incidents (dedupe_key, model_name, model_version, kind, "
            "subject, severity, summary, details, first_seen) "
            "VALUES (%s, 'load-forecast', %s, %s, 'mae', %s, "
            "'MAE 500.0 mw over 30 forecasts', %s, %s)",
            (
                f"load-forecast:{kind}:mae",
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
                first_seen,
            ),
        )


def test_no_incident_makes_no_attempt(pg, store, baseline_model, tmp_path):
    previous = _make_previous(pg, store, tmp_path, baseline_model)
    try:
        outcome = rollback.attempt(
            pg,
            store,
            model_name="load-forecast",
            trigger="automatic",
            settings=settings(),
        )

        assert outcome is None
        assert get_production(pg, "load-forecast").version == baseline_model
    finally:
        _cleanup(pg, baseline_model, previous)


def test_an_incident_opened_before_promotion_does_not_trigger(
    pg, store, baseline_model, tmp_path
):
    previous = _make_previous(pg, store, tmp_path, baseline_model)
    try:
        _open_incident(
            pg,
            baseline_model,
            first_seen=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        promote(pg, name="load-forecast", version=baseline_model)

        outcome = rollback.attempt(
            pg,
            store,
            model_name="load-forecast",
            trigger="automatic",
            settings=settings(),
        )

        assert outcome is None
        assert get_production(pg, "load-forecast").version == baseline_model
    finally:
        _cleanup(pg, baseline_model, previous)


def test_only_performance_incidents_authorize_a_rollback(
    pg, store, baseline_model, tmp_path
):
    previous = _make_previous(pg, store, tmp_path, baseline_model)
    try:
        _open_incident(pg, baseline_model, kind="quality")

        outcome = rollback.attempt(
            pg,
            store,
            model_name="load-forecast",
            trigger="automatic",
            settings=settings(),
        )

        assert outcome is None
        assert get_production(pg, "load-forecast").version == baseline_model
    finally:
        _cleanup(pg, baseline_model, previous)


def test_eligible_incident_rolls_back_to_the_previous_version(
    pg, store, baseline_model, tmp_path
):
    previous = _make_previous(pg, store, tmp_path, baseline_model)
    try:
        _open_incident(pg, baseline_model)

        outcome = rollback.attempt(
            pg,
            store,
            model_name="load-forecast",
            trigger="automatic",
            settings=settings(),
        )

        assert outcome["event"]["decision"] == "rolled_back"
        assert outcome["event"]["trigger"] == "automatic"
        assert outcome["event"]["from_version"] == baseline_model
        assert outcome["event"]["to_version"] == previous
        assert outcome["event"]["metrics"]["mae"] == 500.0
        assert outcome["record"].version == previous
        assert get_production(pg, "load-forecast").version == previous

        with pg.connection() as conn:
            assert conn.execute(
                "SELECT count(*) FROM rollback_events WHERE decision = 'rolled_back'"
            ).fetchone()[0] == 1
    finally:
        _cleanup(pg, baseline_model, previous)


def test_max_age_refuses_an_old_promotion(pg, store, baseline_model, tmp_path):
    previous = _make_previous(pg, store, tmp_path, baseline_model)
    try:
        with pg.connection() as conn:
            conn.execute(
                "UPDATE model_versions SET promoted_at = now() - interval '48 hours' "
                "WHERE name = 'load-forecast' AND version = %s",
                (baseline_model,),
            )
        _open_incident(
            pg,
            baseline_model,
            first_seen=datetime.now(timezone.utc) - timedelta(hours=47),
        )

        outcome = rollback.attempt(
            pg,
            store,
            model_name="load-forecast",
            trigger="automatic",
            settings=settings(rollback_max_age_hours=24),
        )

        assert outcome["event"]["decision"] == "refused"
        assert outcome["event"]["to_version"] is None
        assert "beyond the 24h rollback window" in outcome["event"]["reasons"][-1]
        assert get_production(pg, "load-forecast").version == baseline_model
    finally:
        _cleanup(pg, baseline_model, previous)


def test_cooldown_suppresses_a_second_attempt(pg, store, baseline_model, tmp_path):
    previous = _make_previous(pg, store, tmp_path, baseline_model)
    try:
        rollback.record(
            pg,
            model_name="load-forecast",
            from_version=previous,
            to_version=baseline_model,
            trigger="automatic",
            decision="rolled_back",
            reasons=["earlier rollback"],
            metrics={},
        )
        _open_incident(pg, baseline_model)

        outcome = rollback.attempt(
            pg,
            store,
            model_name="load-forecast",
            trigger="automatic",
            settings=settings(rollback_cooldown_minutes=60),
        )

        assert outcome is None
        with pg.connection() as conn:
            assert conn.execute("SELECT count(*) FROM rollback_events").fetchone()[0] == 1
    finally:
        _cleanup(pg, baseline_model, previous)


def test_disabled_rollback_does_nothing(pg, store, baseline_model, tmp_path):
    previous = _make_previous(pg, store, tmp_path, baseline_model)
    try:
        _open_incident(pg, baseline_model)

        outcome = rollback.attempt(
            pg,
            store,
            model_name="load-forecast",
            trigger="automatic",
            settings=settings(rollback_enabled=False),
        )

        assert outcome is None
        assert get_production(pg, "load-forecast").version == baseline_model
    finally:
        _cleanup(pg, baseline_model, previous)


def test_no_archived_target_records_a_refusal(pg, store, baseline_model):
    with pg.connection() as conn:
        conn.execute(
            "DELETE FROM model_versions WHERE name = 'load-forecast' "
            "AND version <> %s",
            (baseline_model,),
        )
        conn.execute(
            "UPDATE model_versions SET promoted_at = now() "
            "WHERE name = 'load-forecast' AND version = %s",
            (baseline_model,),
        )
    _open_incident(pg, baseline_model)

    outcome = rollback.attempt(
        pg,
        store,
        model_name="load-forecast",
        trigger="automatic",
        settings=settings(),
    )

    assert outcome["event"]["decision"] == "refused"
    assert outcome["event"]["reasons"][-1] == "no archived version to restore"
    assert get_production(pg, "load-forecast").version == baseline_model


def test_schema_mismatched_target_is_refused(pg, store, baseline_model, tmp_path):
    _register_narrow_version(pg, store, tmp_path, "test-narrow")
    try:
        promote(pg, name="load-forecast", version="test-narrow")
        promote(pg, name="load-forecast", version=baseline_model)
        _open_incident(pg, baseline_model)

        outcome = rollback.attempt(
            pg,
            store,
            model_name="load-forecast",
            trigger="automatic",
            settings=settings(),
        )

        assert outcome["event"]["decision"] == "refused"
        assert "feature schema differs" in outcome["event"]["reasons"][-1]
        assert get_production(pg, "load-forecast").version == baseline_model
    finally:
        _cleanup(pg, baseline_model, "test-narrow")


def test_a_version_that_failed_before_is_never_restored(
    pg, store, baseline_model, tmp_path
):
    _register_version(pg, store, tmp_path, "test-older")
    _register_version(pg, store, tmp_path, "test-newer")
    try:
        promote(pg, name="load-forecast", version="test-older")
        promote(pg, name="load-forecast", version="test-newer")
        promote(pg, name="load-forecast", version=baseline_model)

        rollback.record(
            pg,
            model_name="load-forecast",
            from_version="test-newer",
            to_version="test-older",
            trigger="automatic",
            decision="rolled_back",
            reasons=["earlier rollback"],
            metrics={},
        )
        _open_incident(pg, baseline_model)

        outcome = rollback.attempt(
            pg,
            store,
            model_name="load-forecast",
            trigger="automatic",
            settings=settings(rollback_cooldown_minutes=0),
        )

        assert outcome["event"]["decision"] == "rolled_back"
        assert outcome["event"]["to_version"] == "test-older"
    finally:
        _cleanup(pg, baseline_model, "test-older", "test-newer")


def test_list_events_returns_newest_first(pg, baseline_model):
    rollback.record(
        pg,
        model_name="load-forecast",
        from_version="a",
        to_version="b",
        trigger="automatic",
        decision="rolled_back",
        reasons=["first"],
        metrics={},
    )
    rollback.record(
        pg,
        model_name="load-forecast",
        from_version="c",
        to_version=None,
        trigger="automatic",
        decision="refused",
        reasons=["second"],
        metrics={},
    )

    events = rollback.list_events(pg, "load-forecast", limit=10)

    assert [event["reasons"][0] for event in events] == ["second", "first"]


def test_manual_rollback_needs_no_incident(pg, store, baseline_model, tmp_path):
    previous = _make_previous(pg, store, tmp_path, baseline_model)
    try:
        with TestClient(app) as client:
            response = client.post("/rollback")

            assert response.status_code == 200
            body = response.json()
            assert body["decision"] == "rolled_back"
            assert body["trigger"] == "manual"
            assert body["from_version"] == baseline_model
            assert body["to_version"] == previous
            assert client.get("/ready").json()["model_version"] == previous
    finally:
        _cleanup(pg, baseline_model, previous)


def test_manual_rollback_refuses_without_a_target(baseline_model):
    with TestClient(app) as client:
        with client.app.state.pool.connection() as conn:
            conn.execute(
                "DELETE FROM model_versions WHERE name = 'load-forecast' "
                "AND version <> %s",
                (baseline_model,),
            )

        response = client.post("/rollback")

    assert response.status_code == 409
    assert response.json()["detail"]["message"] == "rollback refused"
    assert "no archived version to restore" in response.json()["detail"]["reasons"]


def test_rollback_endpoint_requires_the_token(baseline_model):
    with TestClient(app) as client:
        client.app.state.settings.api_token = "s3cret"
        try:
            assert client.post("/rollback").status_code == 401
        finally:
            client.app.state.settings.api_token = ""


def test_automatic_rollback_runs_after_evaluate(pg, store, baseline_model, tmp_path):
    previous = _make_previous(pg, store, tmp_path, baseline_model)
    try:
        _open_incident(pg, baseline_model)

        with TestClient(app) as client:
            response = client.post("/evaluate")
            assert response.status_code == 200
            body = response.json()

            assert body["rollback"]["decision"] == "rolled_back"
            assert body["rollback"]["from_version"] == baseline_model
            assert body["rollback"]["to_version"] == previous
            assert client.get("/ready").json()["model_version"] == previous
    finally:
        _cleanup(pg, baseline_model, previous)


def test_rollback_clears_a_canary_on_the_restored_version(
    pg, store, baseline_model, tmp_path
):
    previous = _make_previous(pg, store, tmp_path, baseline_model)
    try:
        with TestClient(app) as client:
            assert client.post(
                f"/models/load-forecast/{previous}/canary",
                params={"fraction": 0.5},
            ).status_code == 200

            assert client.post("/rollback").status_code == 200

            assert client.get("/canary").json()["canary_version"] is None
            assert client.get("/shadow").json()["shadow_version"] is None
        with pg.connection() as conn:
            assert conn.execute("SELECT count(*) FROM canary_assignments").fetchone()[0] == 0
    finally:
        _cleanup(pg, baseline_model, previous)


def test_rollback_clears_a_shadow_on_the_restored_version(
    pg, store, baseline_model, tmp_path
):
    previous = _make_previous(pg, store, tmp_path, baseline_model)
    try:
        with TestClient(app) as client:
            assert client.post(
                f"/models/load-forecast/{previous}/shadow"
            ).status_code == 200

            assert client.post("/rollback").status_code == 200

            assert client.get("/shadow").json()["shadow_version"] is None
        with pg.connection() as conn:
            assert conn.execute("SELECT count(*) FROM shadow_assignments").fetchone()[0] == 0
    finally:
        _cleanup(pg, baseline_model, previous)
