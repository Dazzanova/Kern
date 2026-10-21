import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import psycopg
from psycopg.types.json import Jsonb
from xgboost import XGBRegressor

from kern.storage.objects import ObjectStore


class RegistryError(Exception):
    """Base error for registry operations."""


class ModelNotFound(RegistryError):
    pass


@dataclass(frozen=True)
class ModelVersion:
    id: str
    name: str
    version: str
    stage: str
    artifact_key: str
    artifact_sha256: str
    metrics: dict[str, Any]
    feature_schema: list[str]
    reference_profile: dict[str, Any]
    created_at: datetime
    promoted_at: datetime | None


_INSERT = """
INSERT INTO model_versions (
    name, version, stage, artifact_key, artifact_sha256,
    metrics, feature_schema, reference_profile
) VALUES (%s, %s, 'candidate', %s, %s, %s, %s, %s)
ON CONFLICT (name, version) DO NOTHING
RETURNING id, name, version, stage, artifact_key, artifact_sha256,
          metrics, feature_schema, reference_profile, created_at, promoted_at
"""


def _row_to_model(row: tuple) -> ModelVersion:
    return ModelVersion(
        id=str(row[0]),
        name=row[1],
        version=row[2],
        stage=row[3],
        artifact_key=row[4],
        artifact_sha256=row[5],
        metrics=row[6],
        feature_schema=row[7],
        reference_profile=row[8],
        created_at=row[9],
        promoted_at=row[10],
    )


def artifact_key(digest: str) -> str:
    return f"models/{digest}.json"


def compute_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def register(
    pool: psycopg.Connection | Any,
    store: ObjectStore,
    *,
    name: str,
    version: str,
    artifact_path: Path,
    metrics: dict[str, Any],
    feature_schema: list[str],
    reference_profile: dict[str, Any],
) -> ModelVersion:
    data = Path(artifact_path).read_bytes()
    digest = compute_digest(data)
    key = artifact_key(digest)
    store.put_if_absent(key, data)

    with pool.connection() as conn:
        row = conn.execute(
            _INSERT,
            (
                name,
                version,
                key,
                digest,
                Jsonb(metrics),
                Jsonb(feature_schema),
                Jsonb(reference_profile),
            ),
        ).fetchone()

        if row is None:
            row = conn.execute(
                _SELECT_COLUMNS + " WHERE name = %s AND version = %s",
                (name, version),
            ).fetchone()
            if row is None:
                raise ModelNotFound(f"{name} {version} not found")
            if row[5] != digest:
                raise ValueError(
                    f"{name} {version} already exists with a different artifact"
                )

    return _row_to_model(row)


_SELECT_COLUMNS = """
SELECT id, name, version, stage, artifact_key, artifact_sha256,
       metrics, feature_schema, reference_profile, created_at, promoted_at
FROM model_versions
"""


def get_version(pool, name: str, version: str) -> ModelVersion:
    with pool.connection() as conn:
        row = conn.execute(
            _SELECT_COLUMNS + " WHERE name = %s AND version = %s", (name, version)
        ).fetchone()
    if row is None:
        raise ModelNotFound(f"{name} {version} not found")
    return _row_to_model(row)


def list_versions(pool, name: str | None = None) -> list[ModelVersion]:
    query = _SELECT_COLUMNS
    params: tuple = ()
    if name is not None:
        query += " WHERE name = %s"
        params = (name,)
    query += " ORDER BY created_at DESC"
    with pool.connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [_row_to_model(row) for row in rows]


def promote(pool, *, name: str, version: str) -> ModelVersion:
    """Atomically make `version` the sole production version for `name`."""
    with pool.connection() as conn, conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (name,))

        target = conn.execute(
            "SELECT stage FROM model_versions WHERE name = %s AND version = %s FOR UPDATE",
            (name, version),
        ).fetchone()
        if target is None:
            raise ModelNotFound(f"{name} {version} not found")

        # Archive first: the partial unique index forbids two production rows.
        conn.execute(
            "UPDATE model_versions SET stage = 'archived' "
            "WHERE name = %s AND stage = 'production'",
            (name,),
        )
        conn.execute(
            "UPDATE model_versions SET stage = 'production', promoted_at = now() "
            "WHERE name = %s AND version = %s",
            (name, version),
        )

    return get_version(pool, name, version)


def _load_record(store: ObjectStore, record: ModelVersion) -> XGBRegressor:
    """Download, verify, and deserialize one registered artifact."""
    data = store.get(record.artifact_key)
    if compute_digest(data) != record.artifact_sha256:
        raise RegistryError(
            f"artifact digest mismatch for {record.name} {record.version}"
        )

    model = XGBRegressor()
    model.load_model(bytearray(data))

    stored = list(model.get_booster().feature_names or [])
    if not stored:
        raise RegistryError(
            f"artifact for {record.name} {record.version} stores no feature names; "
            "cannot verify schema"
        )
    if stored != list(record.feature_schema):
        raise RegistryError(
            f"feature schema mismatch for {record.name} {record.version}: "
            f"artifact expects {stored}, registry says {list(record.feature_schema)}"
        )

    return model


def load_version(
    pool, store: ObjectStore, name: str, version: str
) -> tuple[XGBRegressor, ModelVersion]:
    """Load and verify a specific version, whether or not it is production."""
    record = get_version(pool, name, version)
    return _load_record(store, record), record


def load_production(
    pool, store: ObjectStore, name: str
) -> tuple[XGBRegressor, ModelVersion]:
    with pool.connection() as conn:
        row = conn.execute(
            _SELECT_COLUMNS + " WHERE name = %s AND stage = 'production'", (name,)
        ).fetchone()
    if row is None:
        raise ModelNotFound(f"no production version for {name}")

    record = _row_to_model(row)
    return _load_record(store, record), record
