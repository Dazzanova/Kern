import os

import pytest

TEST_DSN = os.environ.get(
    "KERN_TEST_DATABASE_URL",
    "postgresql://kern:kern@localhost:5432/kern_test",
)

# Must be set before kern.config is imported anywhere.
os.environ["KERN_DATABASE_URL"] = TEST_DSN
os.environ.setdefault("KERN_TEST_DATABASE_URL", TEST_DSN)
os.environ.setdefault("KERN_FEATURE_SAMPLE_RATE", "1.0")


@pytest.fixture(scope="session")
def pg():
    from kern.storage.db import apply_migrations, create_pool, ensure_database

    ensure_database(TEST_DSN)
    pool = create_pool(TEST_DSN)
    apply_migrations(pool)
    yield pool
    pool.close()


@pytest.fixture(autouse=True)
def clean_telemetry_rows(request):
    """Truncate telemetry between DB tests only.

    This fixture must not depend on `pg` directly, or every test in the suite
    would require the Docker stack and `pytest -m "not db"` could never run.
    `pg` is resolved during setup because dynamic fixture requests from
    teardown are deprecated in pytest 9.
    """
    is_db_test = request.node.get_closest_marker("db") is not None
    pool = request.getfixturevalue("pg") if is_db_test else None

    yield

    if pool is None:
        return
    with pool.connection() as conn:
        conn.execute(
            "TRUNCATE prediction_events, quality_checks, observed_outcomes, "
            "drift_checks, incidents RESTART IDENTITY"
        )


@pytest.fixture(scope="session")
def store():
    from kern.config import Settings
    from kern.storage.objects import ObjectStore

    instance = ObjectStore(Settings(_env_file=None))
    instance.ensure_bucket()
    return instance


@pytest.fixture(scope="session")
def baseline_model(pg, store, tmp_path_factory):
    """Train a small deterministic model and promote it as the test baseline."""
    from xgboost import XGBRegressor

    from kern.data.features import build_features
    from kern.data.generate import generate_dataset
    from kern.quality import profile_reference, reference_sample
    from kern.registry import promote, register

    df = generate_dataset("2025-01-01", 500, 42)
    X = build_features(df)
    y = df["load_mw"]

    model = XGBRegressor(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X, y)

    path = tmp_path_factory.mktemp("artifacts") / "test-baseline.json"
    model.save_model(path)

    # The test database persists between runs; start from a clean slate so a
    # re-trained artifact with a different digest cannot collide with an old row.
    with pg.connection() as conn:
        conn.execute("DELETE FROM model_versions WHERE name = 'load-forecast'")

    register(
        pg,
        store,
        name="load-forecast",
        version="test-v1",
        artifact_path=path,
        metrics={"mae": 0.0},
        feature_schema=list(X.columns),
        reference_profile=profile_reference(X, X.columns),
        reference_sample=reference_sample(X, X.columns),
    )
    promote(pg, name="load-forecast", version="test-v1")
    return "test-v1"
