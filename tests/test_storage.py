import pytest

from kern.storage.db import apply_migrations

pytestmark = pytest.mark.db

EXPECTED_TABLES = {
    "schema_migrations",
    "model_versions",
    "prediction_events",
    "quality_checks",
    "observed_outcomes",
    "drift_checks",
    "incidents",
    "promotion_decisions",
}


def test_migrations_create_all_tables(pg):
    with pg.connection() as conn:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        ).fetchall()

    assert EXPECTED_TABLES <= {row[0] for row in rows}


def test_prediction_events_has_target_timestamp(pg):
    with pg.connection() as conn:
        row = conn.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = 'prediction_events' AND column_name = 'target_ts'"
        ).fetchone()

    assert row is not None
    assert row[0] == "timestamp with time zone"


def test_model_versions_has_reference_sample_column(pg):
    with pg.connection() as conn:
        row = conn.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = 'model_versions' AND column_name = 'reference_sample'"
        ).fetchone()

    assert row is not None
    assert row[0] == "jsonb"


def test_migrations_are_idempotent(pg):
    applied = apply_migrations(pg)

    assert applied == []


def test_changed_migration_is_rejected(pg, tmp_path):
    directory = tmp_path / "migrations"
    directory.mkdir()
    filename = "900_temp_digest_check.sql"
    migration = directory / filename
    migration.write_text("SELECT 1;\n", encoding="utf-8")

    try:
        assert apply_migrations(pg, directory=directory) == [filename]
        assert apply_migrations(pg, directory=directory) == []

        migration.write_text("SELECT 2;\n", encoding="utf-8")

        with pytest.raises(RuntimeError, match="changed after it was applied"):
            apply_migrations(pg, directory=directory)
    finally:
        with pg.connection() as conn:
            conn.execute(
                "DELETE FROM schema_migrations WHERE filename = %s", (filename,)
            )
