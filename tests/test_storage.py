import pytest

from kern.storage.db import apply_migrations

pytestmark = pytest.mark.db

EXPECTED_TABLES = {"schema_migrations", "model_versions", "prediction_events", "quality_checks"}


def test_migrations_create_all_tables(pg):
    with pg.connection() as conn:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        ).fetchall()

    assert EXPECTED_TABLES <= {row[0] for row in rows}


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
