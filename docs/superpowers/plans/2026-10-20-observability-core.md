# Kern Phase 2 — Observability Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a versioned model registry, non-blocking runtime telemetry, and data-quality reporting to Kern, so predictions are attributable to a versioned artifact and comparable to a fixed training baseline.

**Architecture:** Two Docker services (PostgreSQL 16, MinIO). psycopg3 connection pool with a plain-SQL migration runner. Model artifacts go to S3-compatible object storage under content-addressed keys. Telemetry flows through a bounded in-process queue drained by a daemon thread, so inference never blocks on the database. Quality checks are pure functions over reference statistics and sampled feature values.

**Tech Stack:** Python 3.12, FastAPI, psycopg3 (`psycopg[binary,pool]`), boto3, pydantic-settings, XGBoost, pytest, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-10-20-observability-core-design.md`

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `docker-compose.yml` | Postgres + MinIO services |
| `.env.example` | Documented configuration defaults |
| `migrations/001_init.sql` | Schema for all three tables |
| `src/kern/config.py` | Typed settings loaded from environment |
| `src/kern/storage/__init__.py` | Package marker |
| `src/kern/storage/db.py` | Pool creation, database creation, migration runner |
| `src/kern/storage/objects.py` | S3-compatible artifact store |
| `src/kern/registry.py` | Register / promote / load model versions |
| `src/kern/telemetry.py` | Recorder, writer thread, summary queries |
| `src/kern/quality.py` | Reference profiling and threshold evaluation |
| `src/kern/serving/app.py` | Modify: registry-backed lifespan, version reporting, new endpoints |
| `scripts/register_baseline.py` | Register and promote the Phase 1 artifact |
| `tests/conftest.py` | Test database, object store, and registered-model fixtures |
| `tests/test_storage.py` | Migration and object-store tests |
| `tests/test_registry.py` | Register / promote / load tests |
| `tests/test_telemetry.py` | Recorder, writer, summary tests |
| `tests/test_quality.py` | Profiling and threshold boundary tests |
| `tests/test_serving.py` | Modify: mark as requiring infrastructure, add version test |

---

## Task 1: Infrastructure and configuration

**Files:**
- Create: `docker-compose.yml`
- Create: `.env.example`
- Modify: `pyproject.toml`
- Create: `src/kern/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Add dependencies**

Modify `pyproject.toml` dependencies block:

```toml
dependencies = [
    "boto3>=1.40.0",
    "fastapi>=0.141.1",
    "httpx>=0.28.1",
    "joblib>=1.6.0",
    "numpy>=2.5.3",
    "pandas>=3.0.6",
    "psycopg[binary,pool]>=3.2.0",
    "pydantic>=2.13.5",
    "pydantic-settings>=2.0.0",
    "pytest>=9.1.1",
    "scikit-learn>=1.9.1",
    "uvicorn>=0.54.0",
    "xgboost>=3.4.1",
]
```

Also add below `[build-system]`:

```toml
[tool.pytest.ini_options]
markers = [
    "db: requires the docker-compose stack (PostgreSQL and MinIO)",
]
```

- [ ] **Step 2: Write the compose file**

Create `docker-compose.yml`:

```yaml
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: kern
      POSTGRES_PASSWORD: kern
      POSTGRES_DB: kern
    ports:
      - "5432:5432"
    volumes:
      - kern-pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U kern -d kern"]
      interval: 2s
      timeout: 3s
      retries: 30

  minio:
    # Quay and Docker Hub both deny anonymous pulls of minio/minio in this
    # environment; this is a pullable mirror of the same official image.
    image: pgsty/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: kern
      MINIO_ROOT_PASSWORD: kern-secret
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - kern-minio:/data
    healthcheck:
      test: ["CMD", "mc", "ready", "local"]
      interval: 2s
      timeout: 3s
      retries: 30

volumes:
  kern-pgdata:
  kern-minio:
```

- [ ] **Step 3: Write the example environment file**

Create `.env.example`:

```dotenv
KERN_DATABASE_URL=postgresql://kern:kern@localhost:5432/kern
KERN_TEST_DATABASE_URL=postgresql://kern:kern@localhost:5432/kern_test
KERN_S3_ENDPOINT=http://localhost:9000
KERN_S3_BUCKET=kern-models
KERN_S3_ACCESS_KEY=kern
KERN_S3_SECRET_KEY=kern-secret
KERN_TELEMETRY_SAMPLE_RATE=1.0
KERN_FEATURE_SAMPLE_RATE=0.05
KERN_TELEMETRY_QUEUE_SIZE=10000
KERN_MODEL_NAME=load-forecast
```

- [ ] **Step 4: Write the failing config test**

Create `tests/test_config.py`:

```python
import pytest
from pydantic import ValidationError

from kern.config import Settings


def test_defaults_match_compose_stack(monkeypatch):
    # conftest sets these for the test suite; clear them to observe true defaults.
    for name in ("KERN_DATABASE_URL", "KERN_TEST_DATABASE_URL", "KERN_FEATURE_SAMPLE_RATE"):
        monkeypatch.delenv(name, raising=False)

    settings = Settings(_env_file=None)

    assert settings.database_url == "postgresql://kern:kern@localhost:5432/kern"
    assert settings.s3_bucket == "kern-models"
    assert settings.model_name == "load-forecast"
    assert settings.feature_sample_rate == 0.05


def test_reads_prefixed_environment_variables(monkeypatch):
    monkeypatch.setenv("KERN_MODEL_NAME", "other-model")
    monkeypatch.setenv("KERN_TELEMETRY_QUEUE_SIZE", "5")

    settings = Settings(_env_file=None)

    assert settings.model_name == "other-model"
    assert settings.telemetry_queue_size == 5


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_rejects_out_of_range_sample_rate(monkeypatch, value):
    monkeypatch.setenv("KERN_FEATURE_SAMPLE_RATE", str(value))

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
```

- [ ] **Step 5: Run the test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kern.config'`

- [ ] **Step 6: Implement the settings module**

Create `src/kern/config.py`:

```python
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KERN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql://kern:kern@localhost:5432/kern"
    test_database_url: str | None = None
    s3_endpoint: str = "http://localhost:9000"
    s3_bucket: str = "kern-models"
    s3_access_key: str = "kern"
    s3_secret_key: str = "kern-secret"
    telemetry_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    feature_sample_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    telemetry_queue_size: int = Field(default=10_000, gt=0)
    model_name: str = "load-forecast"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: 4 passed

- [ ] **Step 8: Verify the stack starts**

Run: `docker-compose up -d && docker-compose ps`
Expected: both `postgres` and `minio` report `healthy`

- [ ] **Step 9: Commit**

```bash
git add docker-compose.yml .env.example pyproject.toml src/kern/config.py tests/test_config.py
git commit -m "feat: add observability stack configuration and compose services"
```

---

## Task 2: Database pool, test database, and migrations

**Files:**
- Create: `src/kern/storage/__init__.py`
- Create: `src/kern/storage/db.py`
- Create: `migrations/001_init.sql`
- Create: `tests/conftest.py`
- Test: `tests/test_storage.py`

- [ ] **Step 1: Write the failing migration test**

Create `tests/test_storage.py`:

```python
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
```

Create `tests/conftest.py` (needed by every DB test in this plan):

```python
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
def clean_telemetry_rows(pg):
    yield
    with pg.connection() as conn:
        conn.execute("TRUNCATE prediction_events, quality_checks RESTART IDENTITY")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_storage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kern.storage'`

- [ ] **Step 3: Write the schema migration**

Create `migrations/001_init.sql`:

```sql
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename    text PRIMARY KEY,
    sha256      char(64) NOT NULL,
    applied_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_versions (
    id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name               text NOT NULL,
    version            text NOT NULL,
    stage              text NOT NULL DEFAULT 'candidate'
                       CHECK (stage IN ('candidate', 'production', 'archived')),
    artifact_key       text NOT NULL,
    artifact_sha256    char(64) NOT NULL,
    metrics            jsonb NOT NULL DEFAULT '{}'::jsonb,
    feature_schema     jsonb NOT NULL,
    reference_profile  jsonb NOT NULL,
    created_at         timestamptz NOT NULL DEFAULT now(),
    promoted_at        timestamptz,
    UNIQUE (name, version)
);

CREATE UNIQUE INDEX IF NOT EXISTS one_production_per_model
    ON model_versions (name) WHERE stage = 'production';

CREATE TABLE IF NOT EXISTS prediction_events (
    id               bigserial PRIMARY KEY,
    request_id       uuid NOT NULL,
    model_name       text NOT NULL,
    model_version    text NOT NULL,
    endpoint         text NOT NULL,
    batch_size       integer NOT NULL CHECK (batch_size > 0),
    duration_ms      double precision NOT NULL,
    prediction_min   double precision,
    prediction_mean  double precision,
    prediction_max   double precision,
    features         jsonb,
    recorded_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS prediction_events_version_time
    ON prediction_events (model_version, recorded_at DESC);

CREATE INDEX IF NOT EXISTS prediction_events_time
    ON prediction_events (recorded_at DESC);

CREATE TABLE IF NOT EXISTS quality_checks (
    id            bigserial PRIMARY KEY,
    window_start  timestamptz NOT NULL,
    window_end    timestamptz NOT NULL,
    model_version text NOT NULL,
    feature       text NOT NULL,
    check_type    text NOT NULL
                  CHECK (check_type IN ('null_rate', 'range', 'mean_shift')),
    expected      double precision,
    observed      double precision,
    status        text NOT NULL CHECK (status IN ('pass', 'warn', 'fail')),
    recorded_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS quality_checks_window
    ON quality_checks (window_end DESC);
```

- [ ] **Step 4: Implement the storage module**

Create `src/kern/storage/__init__.py` as an empty file.

Create `src/kern/storage/db.py`:

```python
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
            row[0]
            for row in conn.execute("SELECT filename FROM schema_migrations").fetchall()
        }

        for path in sorted(directory.glob("*.sql")):
            if path.name in known:
                continue
            body = path.read_text(encoding="utf-8")
            digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
            conn.execute(body)
            conn.execute(
                "INSERT INTO schema_migrations (filename, sha256) VALUES (%s, %s)",
                (path.name, digest),
            )
            applied.append(path.name)

    return applied
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_storage.py -v`
Expected: 2 passed

If the database does not exist, this fails with a connection error — run `docker-compose up -d` first.

- [ ] **Step 6: Commit**

```bash
git add src/kern/storage migrations tests/conftest.py tests/test_storage.py
git commit -m "feat: add postgres pool and sql migration runner"
```

---

## Task 3: Object storage

**Files:**
- Create: `src/kern/storage/objects.py`
- Test: `tests/test_objects.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_objects.py`:

```python
import pytest

from kern.config import Settings
from kern.storage.objects import ObjectStore

pytestmark = pytest.mark.db


@pytest.fixture(scope="module")
def store():
    settings = Settings(_env_file=None)
    instance = ObjectStore(settings)
    instance.ensure_bucket()
    return instance


def test_round_trip_of_bytes(store):
    store.put_if_absent("tests/round-trip.bin", b"kern-payload")

    assert store.get("tests/round-trip.bin") == b"kern-payload"


def test_put_if_absent_does_not_overwrite(store):
    store.put_if_absent("tests/immutable.bin", b"first")
    store.put_if_absent("tests/immutable.bin", b"second")

    assert store.get("tests/immutable.bin") == b"first"


def test_missing_key_raises(store):
    with pytest.raises(KeyError):
        store.get("tests/never-written.bin")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_objects.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kern.storage.objects'`

- [ ] **Step 3: Implement the object store**

Create `src/kern/storage/objects.py`:

```python
import logging

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from kern.config import Settings

logger = logging.getLogger(__name__)


class ObjectStore:
    """Minimal S3-compatible store. Keys are opaque; callers choose them."""

    def __init__(self, settings: Settings):
        self._bucket = settings.s3_bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            region_name="us-east-1",
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            ),
        )

    def ensure_bucket(self) -> None:
        try:
            self._client.head_bucket(Bucket=self._bucket)
            return
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code")
            if code not in {"404", "NoSuchBucket"}:
                raise
        self._client.create_bucket(Bucket=self._bucket)
        logger.info("created bucket %s", self._bucket)

    def put_if_absent(self, key: str, data: bytes) -> None:
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return
        except ClientError:
            self._client.put_object(Bucket=self._bucket, Key=key, Body=data)

    def get(self, key: str) -> bytes:
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code")
            if code in {"404", "NoSuchKey"}:
                raise KeyError(key) from error
            raise
        return response["Body"].read()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_objects.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/kern/storage/objects.py tests/test_objects.py
git commit -m "feat: add s3-compatible artifact store"
```

---

## Task 4: Registry registration

**Files:**
- Create: `src/kern/registry.py`
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_registry.py`:

```python
import hashlib

import pytest
from xgboost import XGBRegressor

from kern.registry import ModelVersion, register
from kern.storage.objects import ObjectStore

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


@pytest.fixture(autouse=True)
def cleanup_registry_rows(pg):
    yield
    with pg.connection() as conn:
        # No parameters are passed, so a literal % is correct here.
        conn.execute("DELETE FROM model_versions WHERE name LIKE 'test-%'")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kern.registry'`

- [ ] **Step 3: Implement registration**

Create `src/kern/registry.py`:

```python
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
        existing = conn.execute(
            "SELECT id, artifact_sha256 FROM model_versions WHERE name = %s AND version = %s",
            (name, version),
        ).fetchone()

        if existing is not None:
            if existing[1] == digest:
                return get_version(pool, name, version)
            raise ValueError(
                f"{name} {version} already exists with a different artifact"
            )

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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_registry.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/kern/registry.py tests/test_registry.py
git commit -m "feat: add model registry registration"
```

---

## Task 5: Promotion invariant

**Files:**
- Modify: `src/kern/registry.py`
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_registry.py`:

```python
from kern.registry import ModelNotFound, promote


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_registry.py -v -k promote`
Expected: FAIL with `ImportError: cannot import name 'promote'`

- [ ] **Step 3: Implement promotion**

Append to `src/kern/registry.py`:

```python
def promote(pool, *, name: str, version: str) -> ModelVersion:
    """Atomically make `version` the sole production version for `name`."""
    with pool.connection() as conn, conn.transaction():
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


def load_production(pool, store: ObjectStore, name: str) -> tuple[XGBRegressor, ModelVersion]:
    with pool.connection() as conn:
        row = conn.execute(
            _SELECT_COLUMNS + " WHERE name = %s AND stage = 'production'", (name,)
        ).fetchone()
    if row is None:
        raise ModelNotFound(f"no production version for {name}")

    record = _row_to_model(row)
    data = store.get(record.artifact_key)
    if compute_digest(data) != record.artifact_sha256:
        raise RegistryError(
            f"artifact digest mismatch for {name} {record.version}"
        )

    model = XGBRegressor()
    model.load_model(bytearray(data))

    stored = list(model.get_booster().feature_names or [])
    if stored and stored != list(record.feature_schema):
        raise RegistryError(
            f"feature schema mismatch for {name} {record.version}: "
            f"artifact expects {stored}, registry says {list(record.feature_schema)}"
        )

    return model, record
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_registry.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/kern/registry.py tests/test_registry.py
git commit -m "feat: add atomic model promotion and verified loading"
```

---

## Task 6: Quality profiling and evaluation

Pure functions, so they are tested without a database — do this before wiring endpoints.

**Files:**
- Create: `src/kern/quality.py`
- Test: `tests/test_quality.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_quality.py`:

```python
import numpy as np
import pandas as pd
import pytest

from kern.quality import evaluate_features, profile_reference

REFERENCE = {
    "temperature_c": {
        "min": 0.0, "max": 40.0, "mean": 20.0, "std": 5.0, "null_rate": 0.0,
        "p01": 5.0, "p25": 15.0, "p50": 20.0, "p75": 25.0, "p99": 35.0,
    }
}


def test_profile_reference_computes_expected_statistics():
    df = pd.DataFrame({"temperature_c": [10.0, 20.0, 30.0, 40.0]})

    profile = profile_reference(df, ["temperature_c"])

    stats = profile["temperature_c"]
    assert stats["min"] == 10.0
    assert stats["max"] == 40.0
    assert stats["mean"] == 25.0
    assert stats["null_rate"] == 0.0
    assert stats["p50"] == pytest.approx(25.0)


def test_profile_reference_records_null_rate():
    df = pd.DataFrame({"temperature_c": [10.0, None, 30.0, 40.0]})

    profile = profile_reference(df, ["temperature_c"])

    assert profile["temperature_c"]["null_rate"] == 0.25


def test_in_range_values_pass():
    results = evaluate_features(REFERENCE, {"temperature_c": [10.0, 20.0, 30.0]})

    assert [r["status"] for r in results] == ["pass"] * 3


def test_heavy_out_of_range_fails():
    values = [0.0] * 50 + [20.0] * 50

    results = evaluate_features(REFERENCE, {"temperature_c": values})

    range_check = next(r for r in results if r["check_type"] == "range")
    assert range_check["status"] == "fail"
    assert range_check["observed"] == pytest.approx(0.5)


def test_mild_out_of_range_warns():
    values = [0.0] * 10 + [20.0] * 90

    results = evaluate_features(REFERENCE, {"temperature_c": values})

    range_check = next(r for r in results if r["check_type"] == "range")
    assert range_check["status"] == "warn"


def test_large_mean_shift_fails():
    values = list(np.full(200, 40.0))

    results = evaluate_features(REFERENCE, {"temperature_c": values})

    shift = next(r for r in results if r["check_type"] == "mean_shift")
    assert shift["status"] == "fail"


def test_null_rate_above_reference_fails():
    values = [None] * 20 + [20.0] * 80

    results = evaluate_features(REFERENCE, {"temperature_c": values})

    nulls = next(r for r in results if r["check_type"] == "null_rate")
    assert nulls["status"] == "fail"


def test_unknown_feature_is_skipped():
    results = evaluate_features(REFERENCE, {"unknown_feature": [1.0, 2.0]})

    assert results == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_quality.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kern.quality'`

- [ ] **Step 3: Implement the quality module**

Create `src/kern/quality.py`:

```python
from typing import Any, Iterable

import pandas as pd

RANGE_WARN, RANGE_FAIL = 0.05, 0.20
NULL_WARN_DELTA, NULL_FAIL_DELTA = 0.01, 0.05
Z_WARN, Z_FAIL = 3.0, 5.0

QUANTILES = (0.01, 0.25, 0.50, 0.75, 0.99)


def profile_reference(
    df: pd.DataFrame, feature_columns: Iterable[str]
) -> dict[str, dict[str, float]]:
    profile: dict[str, dict[str, float]] = {}

    for column in feature_columns:
        series = pd.to_numeric(df[column], errors="coerce")
        quantiles = series.quantile(list(QUANTILES))
        profile[column] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "null_rate": float(series.isna().mean()),
            "p01": float(quantiles.loc[0.01]),
            "p25": float(quantiles.loc[0.25]),
            "p50": float(quantiles.loc[0.50]),
            "p75": float(quantiles.loc[0.75]),
            "p99": float(quantiles.loc[0.99]),
        }

    return profile


def _status(value: float, warn: float, fail: float) -> str:
    if value > fail:
        return "fail"
    if value > warn:
        return "warn"
    return "pass"


def evaluate_feature(
    feature: str, reference: dict[str, float], values: list[float | None]
) -> list[dict[str, Any]]:
    observed = pd.Series(values, dtype="float64")
    present = observed.dropna()

    if present.empty:
        return []

    range_fraction = float(
        ((present < reference["p01"]) | (present > reference["p99"])).mean()
    )
    null_rate = float(observed.isna().mean())

    std = reference["std"]
    if std and std > 0:
        z_score = abs(float(present.mean()) - reference["mean"]) / std
    else:
        z_score = 0.0

    return [
        {
            "feature": feature,
            "check_type": "range",
            "expected": 0.0,
            "observed": range_fraction,
            "status": _status(range_fraction, RANGE_WARN, RANGE_FAIL),
        },
        {
            "feature": feature,
            "check_type": "null_rate",
            "expected": reference["null_rate"],
            "observed": null_rate,
            "status": _status(
                null_rate - reference["null_rate"], NULL_WARN_DELTA, NULL_FAIL_DELTA
            ),
        },
        {
            "feature": feature,
            "check_type": "mean_shift",
            "expected": reference["mean"],
            "observed": float(present.mean()),
            "status": _status(z_score, Z_WARN, Z_FAIL),
        },
    ]


def evaluate_features(
    reference_profile: dict[str, dict[str, float]],
    observations: dict[str, list[float | None]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for feature, values in observations.items():
        reference = reference_profile.get(feature)
        if reference is None:
            continue
        results.extend(evaluate_feature(feature, reference, values))

    return results
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_quality.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/kern/quality.py tests/test_quality.py
git commit -m "feat: add reference profiling and quality threshold evaluation"
```

---

## Task 7: Telemetry recorder and writer

**Files:**
- Create: `src/kern/telemetry.py`
- Test: `tests/test_telemetry.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_telemetry.py`:

```python
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from kern.telemetry import PredictionEvent, TelemetryRecorder, TelemetryWriter, summarize

pytestmark = pytest.mark.db


def make_event(**overrides):
    values = {
        "request_id": str(uuid4()),
        "model_name": "load-forecast",
        "model_version": "v1",
        "endpoint": "predict",
        "batch_size": 1,
        "duration_ms": 12.5,
        "prediction_min": 100.0,
        "prediction_mean": 100.0,
        "prediction_max": 100.0,
        "features": {"temperature_c": 20.0},
    }
    values.update(overrides)
    return PredictionEvent(**values)


def test_record_enqueues_event():
    recorder = TelemetryRecorder(queue_size=10)

    assert recorder.record(make_event()) is True
    assert recorder.queue.qsize() == 1


def test_full_queue_drops_instead_of_blocking():
    recorder = TelemetryRecorder(queue_size=1)
    recorder.record(make_event())

    assert recorder.record(make_event()) is False
    assert recorder.dropped == 1


def test_request_sample_rate_zero_records_nothing():
    recorder = TelemetryRecorder(queue_size=10, request_sample_rate=0.0)

    assert recorder.record(make_event()) is False
    assert recorder.queue.qsize() == 0
    assert recorder.dropped == 0


def test_feature_snapshot_respects_feature_sample_rate():
    recorder = TelemetryRecorder(queue_size=10, feature_sample_rate=0.0)

    recorder.record(make_event())
    event = recorder.queue.get_nowait()

    assert event.features is None


def test_writer_persists_queue_contents(pg):
    recorder = TelemetryRecorder(queue_size=10)
    writer = TelemetryWriter(pg, recorder)
    event = make_event()
    recorder.record(event)

    persisted = writer.drain_once()

    assert persisted == 1
    with pg.connection() as conn:
        row = conn.execute(
            "SELECT model_version, batch_size FROM prediction_events"
        ).fetchone()
    assert row == ("v1", 1)


def test_summarize_reports_counts_and_latency(pg):
    recorder = TelemetryRecorder(queue_size=10)
    writer = TelemetryWriter(pg, recorder)
    recorder.record(make_event(duration_ms=10.0, batch_size=2))
    recorder.record(make_event(duration_ms=30.0, batch_size=4))
    writer.drain_once()

    summary = summarize(pg, since=datetime.now(timezone.utc).replace(year=2000))

    assert summary["requests"] == 2
    assert summary["records"] == 6
    assert summary["p50_ms"] == pytest.approx(20.0)
    assert summary["dropped_events"] == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_telemetry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kern.telemetry'`

- [ ] **Step 3: Implement the telemetry module**

Create `src/kern/telemetry.py`:

```python
import logging
import queue
import random
import threading
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import Any

from psycopg.types.json import Jsonb

from kern.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionEvent:
    request_id: str
    model_name: str
    model_version: str
    endpoint: str
    batch_size: int
    duration_ms: float
    prediction_min: float | None
    prediction_mean: float | None
    prediction_max: float | None
    features: dict[str, Any] | None


class TelemetryRecorder:
    """Bounded, non-blocking hand-off between request threads and the writer."""

    def __init__(
        self,
        *,
        queue_size: int = 10_000,
        request_sample_rate: float = 1.0,
        feature_sample_rate: float = 0.05,
        rng: random.Random | None = None,
    ):
        self.queue: queue.Queue[PredictionEvent] = queue.Queue(maxsize=queue_size)
        self.request_sample_rate = request_sample_rate
        self.feature_sample_rate = feature_sample_rate
        self._rng = rng or random.Random()
        self._dropped = 0
        self._lock = threading.Lock()

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    def record(self, event: PredictionEvent) -> bool:
        if self.request_sample_rate < 1.0 and self._rng.random() > self.request_sample_rate:
            return False

        if event.features is not None and self._rng.random() > self.feature_sample_rate:
            event = replace(event, features=None)

        try:
            self.queue.put_nowait(event)
        except queue.Full:
            with self._lock:
                self._dropped += 1
            return False

        return True


_INSERT = """
INSERT INTO prediction_events (
    request_id, model_name, model_version, endpoint, batch_size,
    duration_ms, prediction_min, prediction_mean, prediction_max, features
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


class TelemetryWriter:
    """Drains the recorder queue into Postgres in batches on a daemon thread."""

    BATCH_LIMIT = 100

    def __init__(
        self,
        pool,
        recorder: TelemetryRecorder,
        *,
        flush_interval: float = 0.25,
    ):
        self._pool = pool
        self._recorder = recorder
        self._interval = flush_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.persisted = 0
        self.last_flush_at: datetime | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="kern-telemetry", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.drain_once()
            self._stop.wait(self._interval)
        self.drain_once()

    def drain_once(self) -> int:
        batch: list[PredictionEvent] = []
        while len(batch) < self.BATCH_LIMIT:
            try:
                batch.append(self._recorder.queue.get_nowait())
            except queue.Empty:
                break

        if not batch:
            return 0

        rows = [
            (
                event.request_id,
                event.model_name,
                event.model_version,
                event.endpoint,
                event.batch_size,
                event.duration_ms,
                event.prediction_min,
                event.prediction_mean,
                event.prediction_max,
                Jsonb(event.features) if event.features is not None else None,
            )
            for event in batch
        ]

        try:
            with self._pool.connection() as conn:
                conn.executemany(_INSERT, rows)
        except Exception:
            logger.exception("telemetry flush failed; dropped %d events", len(batch))
            return 0

        self.persisted += len(batch)
        self.last_flush_at = datetime.now(timezone.utc)
        return len(batch)


def summarize(
    pool, *, since: datetime, dropped_events: int = 0, last_flush_at: datetime | None = None
) -> dict[str, Any]:
    with pool.connection() as conn:
        row = conn.execute(
            """
            SELECT count(*) AS requests,
                   coalesce(sum(batch_size), 0) AS records,
                   coalesce(percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_ms), 0) AS p50,
                   coalesce(percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms), 0) AS p95,
                   coalesce(percentile_cont(0.99) WITHIN GROUP (ORDER BY duration_ms), 0) AS p99,
                   coalesce(avg(duration_ms), 0) AS mean
            FROM prediction_events
            WHERE recorded_at >= %s
            """,
            (since,),
        ).fetchone()

    return {
        "requests": int(row[0]),
        "records": int(row[1]),
        "p50_ms": float(row[2]),
        "p95_ms": float(row[3]),
        "p99_ms": float(row[4]),
        "mean_ms": float(row[5]),
        "dropped_events": dropped_events,
        "last_flush_at": last_flush_at.isoformat() if last_flush_at else None,
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_telemetry.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/kern/telemetry.py tests/test_telemetry.py
git commit -m "feat: add non-blocking telemetry recorder and batch writer"
```

---

## Task 8: Wire the application to the registry and telemetry

**Files:**
- Modify: `src/kern/serving/app.py`
- Modify: `tests/test_serving.py`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Extend the test fixtures**

Append to `tests/conftest.py`:

```python
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
    import numpy as np
    from xgboost import XGBRegressor

    from kern.data.features import build_features
    from kern.data.generate import generate_dataset
    from kern.quality import profile_reference
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
        reference_profile=profile_reference(df, X.columns),
    )
    promote(pg, name="load-forecast", version="test-v1")
    return "test-v1"
```

Note: `generate_dataset` expects `seed` as the third positional argument.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_serving.py`:

```python
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
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_serving.py::test_prediction_reports_promoted_version -v`
Expected: FAIL — reported version is `"baseline"`

- [ ] **Step 4: Rewrite the application lifespan and prediction handlers**

Replace the top of `src/kern/serving/app.py` (through the `app = FastAPI(...)` line) with:

```python
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from kern.config import get_settings
from kern.data.features import build_features
from kern.quality import evaluate_features
from kern.registry import (
    ModelNotFound,
    get_version,
    list_versions,
    load_production,
    promote,
)
from kern.storage.db import apply_migrations, create_pool
from kern.storage.objects import ObjectStore
from kern.telemetry import PredictionEvent, TelemetryRecorder, TelemetryWriter, summarize

logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    timestamp: datetime
    temperature_c: float
    humidity_pct: float = Field(ge=0, le=100)


class PredictionResponse(BaseModel):
    request_id: str
    model_version: str
    prediction_mw: float
    inference_latency_ms: float


class BatchPredictionRequest(BaseModel):
    records: list[PredictionRequest] = Field(..., min_length=1, max_length=256)


class BatchPredictionResponse(BaseModel):
    request_id: str
    model_version: str
    batch_size: int
    total_processing_time_ms: float
    predictions: list[float]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    pool = create_pool(settings.database_url)
    apply_migrations(pool)

    store = ObjectStore(settings)
    store.ensure_bucket()

    model, record = load_production(pool, store, settings.model_name)
    recorder = TelemetryRecorder(
        queue_size=settings.telemetry_queue_size,
        request_sample_rate=settings.telemetry_sample_rate,
        feature_sample_rate=settings.feature_sample_rate,
    )
    writer = TelemetryWriter(pool, recorder)

    app.state.settings = settings
    app.state.pool = pool
    app.state.store = store
    app.state.model = model
    app.state.model_version = record.version
    app.state.feature_schema = list(record.feature_schema)
    app.state.reference_profile = record.reference_profile
    app.state.recorder = recorder
    app.state.writer = writer
    writer.start()

    try:
        yield
    finally:
        writer.stop()
        pool.close()


app = FastAPI(
    title="Kern",
    version="0.1.0",
    lifespan=lifespan,
)
```

Replace the `/predict` and `/predict/batch` handlers with:

```python
def _record(app: FastAPI, *, request_id: str, endpoint: str, batch_size: int,
            duration_ms: float, predictions: np.ndarray, feature_frame: pd.DataFrame) -> None:
    """Hand off telemetry without blocking the request.

    `feature_frame` must be the output of `build_features`: its columns are numeric
    and match the reference profile. Values are cast to plain floats because pandas
    returns NumPy scalars, which `json.dumps` (used by `Jsonb`) cannot serialize.
    """
    recorder: TelemetryRecorder = app.state.recorder
    snapshot = {key: float(value) for key, value in feature_frame.iloc[0].items()}
    recorder.record(
        PredictionEvent(
            request_id=request_id,
            model_name=app.state.settings.model_name,
            model_version=app.state.model_version,
            endpoint=endpoint,
            batch_size=batch_size,
            duration_ms=duration_ms,
            prediction_min=float(predictions.min()),
            prediction_mean=float(predictions.mean()),
            prediction_max=float(predictions.max()),
            features=snapshot,
        )
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request):
    request_id = str(uuid4())

    input_df = pd.DataFrame([payload.model_dump()])
    features = build_features(input_df)

    start = perf_counter()
    prediction = request.app.state.model.predict(features)[0]
    latency_ms = (perf_counter() - start) * 1000

    _record(
        request.app,
        request_id=request_id,
        endpoint="predict",
        batch_size=1,
        duration_ms=latency_ms,
        predictions=np.asarray([prediction]),
        feature_frame=features,
    )

    return PredictionResponse(
        request_id=request_id,
        model_version=request.app.state.model_version,
        prediction_mw=float(prediction),
        inference_latency_ms=latency_ms,
    )
```

For `/predict/batch`, replace the body's response construction:

```python
    return BatchPredictionResponse(
        request_id=request_id,
        model_version=request.app.state.model_version,
        batch_size=len(payload.records),
        total_processing_time_ms=processing_time_ms,
        predictions=[float(p) for p in predictions],
    )
```

and add before that return:

```python
    _record(
        request.app,
        request_id=request_id,
        endpoint="predict/batch",
        batch_size=len(payload.records),
        duration_ms=processing_time_ms,
        predictions=predictions,
        feature_frame=features,
    )
```

- [ ] **Step 5: Mark the module as requiring infrastructure and fix the version assertions**

At the top of `tests/test_serving.py`, after the imports, add:

```python
import pytest

pytestmark = pytest.mark.db


@pytest.fixture(autouse=True)
def use_baseline(baseline_model):
    """Every test in this module needs a promoted model to start the app."""
    return baseline_model
```

Two existing assertions hardcode the old literal version. Update them:

```python
def test_prediction(baseline_model):
    # ... unchanged request ...
    assert body["model_version"] == baseline_model
```

```python
def test_batch_prediction(baseline_model):
    # ... unchanged request ...
    assert body["model_version"] == baseline_model
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_serving.py -v`
Expected: 13 passed

- [ ] **Step 7: Commit**

```bash
git add src/kern/serving/app.py tests/test_serving.py tests/conftest.py
git commit -m "feat: serve predictions from the registry and record telemetry"
```

---

## Task 9: Observability endpoints

**Files:**
- Modify: `src/kern/serving/app.py`
- Test: `tests/test_serving.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_serving.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_serving.py -v -k "models_endpoint or promote or telemetry_summary or quality_report"`
Expected: FAIL with 404 responses

- [ ] **Step 3: Add the window parser and endpoints**

Append to `src/kern/serving/app.py`:

```python
_WINDOW_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_window(value: str) -> timedelta:
    unit = value[-1:]
    amount = value[:-1]
    if unit not in _WINDOW_UNITS or not amount.isdigit() or int(amount) == 0:
        raise ValueError(f"invalid window: {value!r}")
    return timedelta(seconds=int(amount) * _WINDOW_UNITS[unit])


@app.get("/models")
def models(request: Request):
    versions = list_versions(request.app.state.pool)
    return {
        "versions": [
            {
                "id": record.id,
                "name": record.name,
                "version": record.version,
                "stage": record.stage,
                "metrics": record.metrics,
                "created_at": record.created_at.isoformat(),
                "promoted_at": record.promoted_at.isoformat() if record.promoted_at else None,
            }
            for record in versions
        ]
    }


@app.get("/models/{name}/{version}")
def model_detail(name: str, version: str, request: Request):
    try:
        record = get_version(request.app.state.pool, name, version)
    except ModelNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    return {
        "id": record.id,
        "name": record.name,
        "version": record.version,
        "stage": record.stage,
        "artifact_key": record.artifact_key,
        "artifact_sha256": record.artifact_sha256,
        "metrics": record.metrics,
        "feature_schema": record.feature_schema,
        "reference_profile": record.reference_profile,
        "created_at": record.created_at.isoformat(),
        "promoted_at": record.promoted_at.isoformat() if record.promoted_at else None,
    }


@app.post("/models/{name}/{version}/promote")
def promote_model(name: str, version: str, request: Request):
    try:
        record = promote(request.app.state.pool, name=name, version=version)
    except ModelNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    return {
        "name": record.name,
        "version": record.version,
        "stage": record.stage,
        "promoted_at": record.promoted_at.isoformat() if record.promoted_at else None,
    }


@app.get("/telemetry/summary")
def telemetry_summary(request: Request, window: str = "1h"):
    try:
        delta = parse_window(window)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    since = datetime.now(timezone.utc) - delta
    state = request.app.state
    return summarize(
        state.pool,
        since=since,
        dropped_events=state.recorder.dropped,
        last_flush_at=state.writer.last_flush_at,
    )


@app.get("/quality/report")
def quality_report(request: Request, window: str = "1h"):
    try:
        delta = parse_window(window)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    since = datetime.now(timezone.utc) - delta
    state = request.app.state

    with state.pool.connection() as conn:
        rows = conn.execute(
            "SELECT model_version, features FROM prediction_events "
            "WHERE recorded_at >= %s AND features IS NOT NULL",
            (since,),
        ).fetchall()

    observations: dict[str, list[float | None]] = {}
    for _version, features in rows:
        for feature, value in features.items():
            observations.setdefault(feature, []).append(value)

    checks = evaluate_features(state.reference_profile, observations)

    window_end = datetime.now(timezone.utc)
    if checks:
        with state.pool.connection() as conn, conn.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO quality_checks (window_start, window_end, model_version, "
                "feature, check_type, expected, observed, status) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                [
                    (since, window_end, state.model_version, check["feature"],
                     check["check_type"], check["expected"], check["observed"],
                     check["status"])
                    for check in checks
                ],
            )

    return {
        "window": window,
        "model_version": state.model_version,
        "checks": checks,
    }
```

Also update the `/ready` handler:

```python
@app.get("/ready")
def ready(request: Request):
    state = request.app.state
    return {
        "ready": hasattr(state, "model"),
        "model_version": getattr(state, "model_version", None),
        "database": hasattr(state, "pool"),
        "object_store": hasattr(state, "store"),
    }
```

Add `timedelta` to the datetime import at the top of the file if it is not already
present (Task 8 included it):

```python
from datetime import datetime, timedelta, timezone
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_serving.py -v`
Expected: 19 passed

- [ ] **Step 5: Commit**

```bash
git add src/kern/serving/app.py tests/test_serving.py
git commit -m "feat: expose registry, telemetry, and quality endpoints"
```

---

## Task 10: Baseline registration script and docs

**Files:**
- Create: `scripts/register_baseline.py`
- Modify: `README.md`

- [ ] **Step 1: Write the registration script**

Create `scripts/register_baseline.py`:

```python
"""Register and promote the Phase 1 baseline artifact.

Requires: docker-compose up -d, and models/baseline.json plus data/train.csv
present locally (both are gitignored; regenerate with the training pipeline).
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from kern.config import get_settings  # noqa: E402
from kern.data.features import build_features  # noqa: E402
from kern.quality import profile_reference  # noqa: E402
from kern.registry import promote, register  # noqa: E402
from kern.storage.db import apply_migrations, create_pool  # noqa: E402
from kern.storage.objects import ObjectStore  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    settings = get_settings()
    artifact = ROOT / "models" / "baseline.json"
    if not artifact.exists():
        raise SystemExit(f"missing artifact: {artifact}")

    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")

    features = build_features(train)
    model = XGBRegressor()
    model.load_model(artifact)
    predictions = model.predict(build_features(test))

    metrics = {
        "mae": float(mean_absolute_error(test["load_mw"], predictions)),
        "rmse": float(np.sqrt(mean_squared_error(test["load_mw"], predictions))),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
    }

    pool = create_pool(settings.database_url)
    apply_migrations(pool)
    store = ObjectStore(settings)
    store.ensure_bucket()

    record = register(
        pool,
        store,
        name=settings.model_name,
        version="v1",
        artifact_path=artifact,
        metrics=metrics,
        feature_schema=list(features.columns),
        reference_profile=profile_reference(train, features.columns),
    )
    promote(pool, name=settings.model_name, version=record.version)

    print(f"registered {record.name} {record.version} ({record.artifact_sha256[:12]})")
    print(f"metrics: {metrics}")

    pool.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

Run: `uv run python scripts/register_baseline.py`
Expected: prints `registered load-forecast v1 (...)` and the metrics

- [ ] **Step 3: Verify the model is live**

Run: `curl -s localhost:8000/models | head -c 400`
Expected: JSON containing `"version": "v1"` and `"stage": "production"`

If port 8000 is served by an older uvicorn process, stop it first — the process running before this plan was `uvicorn` pid 369692.

- [ ] **Step 4: Update the README**

Add after the "Batch Inference" section:

````markdown
## Observability

Kern versions models in a registry, records per-request telemetry, and compares live
traffic against a reference profile of the training data.

### Infrastructure
```bash
docker-compose up -d
cp .env.example .env
uv run python scripts/register_baseline.py
uv run uvicorn src.kern.serving.app:app
```

### Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | `/models` | Registered versions and stages |
| GET | `/models/{name}/{version}` | Full version metadata |
| POST | `/models/{name}/{version}/promote` | Make a version production |
| GET | `/telemetry/summary?window=1h` | Request counts, latency percentiles, dropped events |
| GET | `/quality/report?window=1h` | Per-feature data-quality checks |

### Tests
`uv run pytest` requires the compose stack. `uv run pytest -m "not db"` runs only the
tests that need no infrastructure.
````

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -v`
Expected: all tests pass

- [ ] **Step 6: Verify benchmarks are unaffected**

Stop any stale server, restart with the registry-backed app, then run
`uv run benchmarks/serving_perf.py`. Compare against the README table; record throughput
should stay within ~10% of the Phase 1 figures (17,516 rec/s at batch size 256).
Telemetry writes happen after the timed region, so any larger regression indicates the
writer is blocking the request path.

- [ ] **Step 7: Commit**

```bash
git add scripts/register_baseline.py README.md
git commit -m "feat: add baseline registration script and observability docs"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
| --- | --- |
| Compose stack, fail-fast startup | 1, 8 |
| Config module and `.env.example` | 1 |
| Migration runner | 2 |
| Three tables, partial unique index | 2, 5 |
| Object store, content addressing, digest verification | 3, 5 |
| Registry register / promote / load | 4, 5 |
| Telemetry queue, drop counting, sampling | 7 |
| Telemetry summary | 7, 9 |
| Reference profiling and thresholds | 6 |
| Quality report and persisted checks | 9 |
| API surface | 9 |
| `/ready` extension | 9 |
| Test fixtures and `db` marker | 2, 8 |
| Baseline registration and parity | 10 |
| Documentation | 10 |

**Known deviations from the spec, intentional:**
- The spec's `load(name)` is implemented as `load_production(pool, store, name)`.
- Test fixtures train a small model rather than depending on `models/baseline.json`,
  because `models/` is gitignored and the suite must not depend on local artifacts.
- `pydantic-settings` is used for configuration; the spec only specified a config module.
- The `features` snapshot stores the **built** feature vector (numeric columns), not the
  raw request body. The raw body contains a `datetime` that `Jsonb` cannot serialize, and
  the built vector's keys match `reference_profile` exactly, so the quality report needs
  no key translation.

**Type consistency:** `ModelVersion` fields match the `_SELECT_COLUMNS` order; `register`
and `promote` both return `ModelVersion`; `PredictionEvent` field names match the insert
column order; `evaluate_features` returns dicts with keys `feature`, `check_type`,
`expected`, `observed`, `status`, consumed verbatim by `quality_report`.
