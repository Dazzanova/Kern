import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
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
from kern.telemetry import (
    PredictionEvent,
    TelemetryRecorder,
    TelemetryWriter,
    summarize,
)

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
    pool = None
    try:
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
        app.state.reference_profile = record.reference_profile
        app.state.recorder = recorder
        app.state.writer = writer
        writer.start()
    except Exception as error:
        if pool is not None:
            pool.close()
        raise RuntimeError(
            "Kern failed to start. Run `docker-compose up -d` and "
            "`uv run python scripts/register_baseline.py`, then retry."
        ) from error

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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request):
    state = request.app.state

    database = True
    try:
        with state.pool.connection() as conn:
            conn.execute("SELECT 1")
    except Exception:
        logger.exception("readiness check failed: database unavailable")
        database = False

    object_store = True
    try:
        state.store.ensure_bucket()
    except Exception:
        logger.exception("readiness check failed: object store unavailable")
        object_store = False

    body = {
        "ready": database and object_store,
        "model_version": state.model_version,
        "database": database,
        "object_store": object_store,
    }
    if not body["ready"]:
        return JSONResponse(status_code=503, content=body)
    return body


def _record(
    app: FastAPI,
    *,
    request_id: str,
    endpoint: str,
    batch_size: int,
    duration_ms: float,
    predictions: np.ndarray,
    feature_frame: pd.DataFrame,
) -> None:
    """Hand off telemetry without blocking the request.

    `feature_frame` must be the output of `build_features`: its columns are numeric
    and match the reference profile. Values are cast to plain floats because pandas
    returns NumPy scalars, which `json.dumps` (used by `Jsonb`) cannot serialize.

    Telemetry is best-effort: a recorder failure must never fail a prediction.
    """
    try:
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
    except Exception:
        logger.exception("failed to record telemetry for %s", request_id)


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


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictionRequest, request: Request):
    request_id = str(uuid4())
    start = perf_counter()

    # Convert records to DataFrame efficiently
    input_df = pd.DataFrame([r.model_dump() for r in payload.records])

    # Vectorized feature engineering
    features = build_features(input_df)

    # Single model call for the whole batch
    predictions = request.app.state.model.predict(features)

    # Ensure predictions are finite
    predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)

    processing_time_ms = (perf_counter() - start) * 1000

    _record(
        request.app,
        request_id=request_id,
        endpoint="predict/batch",
        batch_size=len(payload.records),
        duration_ms=processing_time_ms,
        predictions=predictions,
        feature_frame=features,
    )

    return BatchPredictionResponse(
        request_id=request_id,
        model_version=request.app.state.model_version,
        batch_size=len(payload.records),
        total_processing_time_ms=processing_time_ms,
        predictions=[float(p) for p in predictions],
    )


_WINDOW_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
MAX_WINDOW = timedelta(days=365)


def parse_window(value: str) -> timedelta:
    unit = value[-1:]
    amount = value[:-1]
    # Bound the digit count before int()/timedelta arithmetic: an absurdly long
    # amount would otherwise overflow C-level arithmetic and surface as a 500.
    if (
        unit not in _WINDOW_UNITS
        or not amount.isdigit()
        or len(amount) > 6
        or int(amount) == 0
    ):
        raise ValueError(f"invalid window: {value!r}")
    try:
        delta = timedelta(seconds=int(amount) * _WINDOW_UNITS[unit])
    except OverflowError as error:
        raise ValueError(f"invalid window: {value!r}") from error
    if delta > MAX_WINDOW:
        raise ValueError(f"window exceeds {MAX_WINDOW.days} days")
    return delta


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
    except (ValueError, OverflowError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    since = datetime.now(timezone.utc) - delta
    state = request.app.state
    return summarize(
        state.pool,
        since=since,
        dropped_events=state.recorder.dropped,
        failed_events=state.writer.failed,
        last_flush_at=state.writer.last_flush_at,
    )


@app.get("/quality/report")
def quality_report(request: Request, window: str = "1h"):
    try:
        delta = parse_window(window)
    except (ValueError, OverflowError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    since = datetime.now(timezone.utc) - delta
    state = request.app.state

    with state.pool.connection() as conn:
        rows = conn.execute(
            "SELECT model_version, features FROM prediction_events "
            "WHERE recorded_at >= %s AND features IS NOT NULL "
            "AND model_version = %s",
            (since, state.model_version),
        ).fetchall()

    observations: dict[str, list[float | None]] = {}
    for _version, features in rows:
        for feature, value in features.items():
            observations.setdefault(feature, []).append(value)

    # An empty window means "nothing observed yet", which must not be reported
    # as every feature being 100% null. That would raise a false alarm for the
    # whole sampling interval right after a promotion or restart.
    checks = (
        evaluate_features(state.reference_profile, observations) if rows else []
    )

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
        "sample_count": len(rows),
        "checks": checks,
    }
