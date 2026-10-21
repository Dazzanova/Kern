import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest
from xgboost import XGBRegressor

from kern.registry import ModelVersion, register
from kern.storage.objects import ObjectStore

FEATURE_COLUMNS = [
    "temperature_c",
    "humidity_pct",
    "hour",
    "day_of_week",
    "is_weekend",
    "day_of_year",
    "month",
    "time_index_hours",
]

pytestmark = pytest.mark.db

FEATURES = {
    "temperature_c": {"min": 0.0, "max": 40.0, "mean": 20.0, "std": 5.0,
                      "null_rate": 0.0, "p01": 5.0, "p25": 15.0, "p50": 20.0,
                      "p75": 25.0, "p99": 35.0},
}


@pytest.fixture(scope="module")
def store():
    from kern.config import Settings

    instance = ObjectStore(Settings(_env_file=None))
    instance.ensure_bucket()
    return instance


@pytest.fixture
def artifact(tmp_path):
    model = XGBRegressor(n_estimators=3, max_depth=2)
    model.fit([[20.0, 50.0, 12, 2, 0, 91, 4, 2160.0]] * 4, [100.0, 110.0, 120.0, 130.0])
    path = tmp_path / "artifact.json"
    model.save_model(path)
    return path


def test_register_stores_artifact_and_metadata(pg, store, artifact):
    record = register(
        pg,
        store,
        name="test-register",
        version="v1",
        artifact_path=artifact,
        metrics={"mae": 1.5},
        feature_schema=["temperature_c"],
        reference_profile=FEATURES,
    )

    assert isinstance(record, ModelVersion)
    assert record.stage == "candidate"
    assert record.version == "v1"
    assert store.get(record.artifact_key) == artifact.read_bytes()
    assert record.artifact_sha256 == hashlib.sha256(artifact.read_bytes()).hexdigest()


def test_reregistering_identical_artifact_is_a_noop(pg, store, artifact):
    first = register(pg, store, name="test-idempotent", version="v1",
                     artifact_path=artifact, metrics={}, feature_schema=["temperature_c"],
                     reference_profile=FEATURES)
    second = register(pg, store, name="test-idempotent", version="v1",
                      artifact_path=artifact, metrics={}, feature_schema=["temperature_c"],
                      reference_profile=FEATURES)

    assert second.id == first.id


def test_reregistering_different_artifact_raises(pg, store, artifact, tmp_path):
    register(pg, store, name="test-conflict", version="v1", artifact_path=artifact,
             metrics={}, feature_schema=["temperature_c"], reference_profile=FEATURES)

    other = XGBRegressor(n_estimators=5)
    other.fit([[20.0, 50.0, 12, 2, 0, 91, 4, 2160.0]] * 4, [1.0, 2.0, 3.0, 4.0])
    other_path = tmp_path / "other.json"
    other.save_model(other_path)

    with pytest.raises(ValueError, match="different artifact"):
        register(pg, store, name="test-conflict", version="v1", artifact_path=other_path,
                 metrics={}, feature_schema=["temperature_c"], reference_profile=FEATURES)


from kern.registry import ModelNotFound, RegistryError, get_version, load_production, promote


def _register(pg, store, artifact, name, version):
    return register(pg, store, name=name, version=version, artifact_path=artifact,
                    metrics={}, feature_schema=["temperature_c"],
                    reference_profile=FEATURES)


def test_promote_makes_exactly_one_production_version(pg, store, artifact):
    _register(pg, store, artifact, "test-promote", "v1")
    _register(pg, store, artifact, "test-promote", "v2")

    promote(pg, name="test-promote", version="v1")
    promote(pg, name="test-promote", version="v2")

    assert get_version(pg, "test-promote", "v1").stage == "archived"
    assert get_version(pg, "test-promote", "v2").stage == "production"
    assert get_version(pg, "test-promote", "v2").promoted_at is not None


def test_promote_unknown_version_raises(pg):
    with pytest.raises(ModelNotFound):
        promote(pg, name="test-promote", version="missing")


def test_promote_does_not_disturb_other_models(pg, store, artifact):
    _register(pg, store, artifact, "test-alpha", "v1")
    _register(pg, store, artifact, "test-beta", "v1")
    promote(pg, name="test-alpha", version="v1")
    promote(pg, name="test-beta", version="v1")

    assert get_version(pg, "test-alpha", "v1").stage == "production"
    assert get_version(pg, "test-beta", "v1").stage == "production"


@pytest.fixture
def named_artifact(tmp_path):
    """Artifact trained on the eight feature columns, so XGBoost records names."""
    frame = pd.DataFrame(
        [[20.0, 50.0, 12, 2, 0, 91, 4, 2160.0]] * 4,
        columns=FEATURE_COLUMNS,
    )
    model = XGBRegressor(n_estimators=3, max_depth=2)
    model.fit(frame, [100.0, 110.0, 120.0, 130.0])
    path = tmp_path / "named-artifact.json"
    model.save_model(path)
    return path


def test_load_production_returns_verified_model(pg, store, named_artifact):
    register(pg, store, name="test-load", version="v1", artifact_path=named_artifact,
             metrics={}, feature_schema=FEATURE_COLUMNS,
             reference_profile={c: FEATURES["temperature_c"] for c in FEATURE_COLUMNS})
    promote(pg, name="test-load", version="v1")

    model, record = load_production(pg, store, "test-load")

    assert record.version == "v1"
    assert record.stage == "production"
    assert list(model.get_booster().feature_names) == FEATURE_COLUMNS
    predictions = model.predict(
        pd.DataFrame([[25.0, 55.0, 12, 1, 0, 91, 4, 2160.0]], columns=FEATURE_COLUMNS)
    )
    assert predictions.shape == (1,)


def test_load_production_rejects_feature_schema_mismatch(pg, store, named_artifact):
    register(pg, store, name="test-drift", version="v1", artifact_path=named_artifact,
             metrics={}, feature_schema=["temperature_c"], reference_profile=FEATURES)
    promote(pg, name="test-drift", version="v1")

    with pytest.raises(RegistryError, match="feature schema mismatch"):
        load_production(pg, store, "test-drift")


def test_load_production_rejects_artifact_without_feature_names(pg, store, artifact):
    # `artifact` is trained on a plain list, so XGBoost stores no names and the
    # schema check would otherwise silently pass.
    _register(pg, store, artifact, "test-no-names", "v1")
    promote(pg, name="test-no-names", version="v1")

    with pytest.raises(RegistryError, match="stores no feature names"):
        load_production(pg, store, "test-no-names")


def test_load_production_rejects_tampered_artifact(pg, store, named_artifact):
    record = register(
        pg, store, name="test-tamper", version="v1", artifact_path=named_artifact,
        metrics={}, feature_schema=FEATURE_COLUMNS,
        reference_profile={c: FEATURES["temperature_c"] for c in FEATURE_COLUMNS},
    )
    promote(pg, name="test-tamper", version="v1")

    original = named_artifact.read_bytes()
    store._client.put_object(
        Bucket=store._bucket, Key=record.artifact_key, Body=b"tampered"
    )
    try:
        with pytest.raises(RegistryError, match="digest mismatch"):
            load_production(pg, store, "test-tamper")
    finally:
        # Artifact keys are content-addressed, so restore the original bytes
        # before another test happens to share this digest.
        store._client.put_object(
            Bucket=store._bucket, Key=record.artifact_key, Body=original
        )


def test_concurrent_promotions_leave_one_production_row(pg, store, artifact):
    _register(pg, store, artifact, "test-race", "v1")
    _register(pg, store, artifact, "test-race", "v2")
    barrier = threading.Barrier(2)

    def promote_version(version: str) -> None:
        barrier.wait(timeout=5)
        promote(pg, name="test-race", version=version)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(promote_version, v) for v in ("v1", "v2")]
        for future in futures:
            future.result()

    with pg.connection() as conn:
        rows = conn.execute(
            "SELECT version FROM model_versions "
            "WHERE name = 'test-race' AND stage = 'production'"
        ).fetchall()

    assert len(rows) == 1
    assert rows[0][0] in {"v1", "v2"}


@pytest.fixture(autouse=True)
def cleanup_registry_rows(pg):
    yield
    with pg.connection() as conn:
        # No parameters are passed, so a literal % is correct here.
        conn.execute("DELETE FROM model_versions WHERE name LIKE 'test-%'")
