import hashlib
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg_pool import ConnectionPool

MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations"


def create_pool(dsn: str, *, min_size: int = 1, max_size: int = 8) -> ConnectionPool:
    pool = ConnectionPool(dsn, min_size=min_size, max_size=max_size, open=False)
    pool.open()
    pool.wait(timeout=10)
    return pool


def ensure_database(dsn: str) -> None:
    """Create the database named in `dsn` if it does not exist yet."""
    params = conninfo_to_dict(dsn)
    target = params.pop("dbname", "kern")
    admin_dsn = make_conninfo(**params, dbname="postgres")

    with psycopg.connect(admin_dsn, autocommit=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (target,)
        ).fetchone()
        if exists:
            return
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(target)))


def apply_migrations(
    pool: ConnectionPool, directory: Path = MIGRATIONS_DIR
) -> list[str]:
    """Apply pending .sql files in filename order. Returns the files applied."""
    applied: list[str] = []

    with pool.connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                filename   text PRIMARY KEY,
                sha256     char(64) NOT NULL,
                applied_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        known = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT filename, sha256 FROM schema_migrations"
            ).fetchall()
        }

        for path in sorted(directory.glob("*.sql")):
            body = path.read_text(encoding="utf-8")
            digest = hashlib.sha256(body.encode("utf-8")).hexdigest()

            if path.name in known:
                if known[path.name] != digest:
                    raise RuntimeError(f"{path.name} changed after it was applied")
                continue

            conn.execute(body)
            conn.execute(
                "INSERT INTO schema_migrations (filename, sha256) VALUES (%s, %s)",
                (path.name, digest),
            )
            applied.append(path.name)

    return applied
