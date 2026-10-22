import logging
import math
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from xgboost import XGBRegressor

from kern import evaluation, policy, reports
from kern.config import get_settings
from kern.data.features import build_features
from kern.registry import (
    ModelNotFound,
    RegistryError,
    get_production,
    get_version,
    list_versions,
    load_production,
    load_version,
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
from kern.windows import parse_window

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


class OutcomeObservation(BaseModel):
    target_ts: datetime
    actual_mw: float


class OutcomeRequest(BaseModel):
    observations: list[OutcomeObservation] = Field(..., min_length=1, max_length=1000)


@dataclass(frozen=True)
class ModelState:
    """Everything a request needs to serve, replaced wholesale on promotion.

    Handlers capture this once at the top. Because it is frozen and swapped rather
    than mutated, a request that began under one version finishes under it even if
    a promotion lands mid-flight. No lock is needed on the read path.
    """

    name: str
    version: str
    model: XGBRegressor
    reference_profile: dict[str, dict[str, float]]
    reference_sample: dict[str, list[float]] | None


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    pool = None
    runner = None
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
        app.state.current = ModelState(
            name=record.name,
            version=record.version,
            model=model,
            reference_profile=record.reference_profile,
            reference_sample=record.reference_sample,
        )
        app.state.recorder = recorder
        app.state.writer = writer
        app.state.last_retention = None
        writer.start()

        if settings.evaluation_interval > 0:
            runner = evaluation.EvaluationRunner(app)
            runner.start()

        if not settings.api_token:
            logger.warning(
                "KERN_API_TOKEN is unset: /models/*/promote and /outcomes are "
                "unauthenticated; set it before exposing this service"
            )
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
        if runner is not None:
            runner.stop()
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
        "model_version": state.current.version,
        "database": database,
        "object_store": object_store,
    }
    if not body["ready"]:
        return JSONResponse(status_code=503, content=body)
    return body


def _record(
    app: FastAPI,
    current: ModelState,
    *,
    request_id: str,
    endpoint: str,
    batch_size: int,
    duration_ms: float,
    predictions: np.ndarray,
    feature_frame: pd.DataFrame,
    target_ts: datetime | None = None,
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
                model_version=current.version,
                endpoint=endpoint,
                batch_size=batch_size,
                duration_ms=duration_ms,
                prediction_min=float(predictions.min()),
                prediction_mean=float(predictions.mean()),
                prediction_max=float(predictions.max()),
                features=snapshot,
                target_ts=target_ts,
            )
        )
    except Exception:
        logger.exception("failed to record telemetry for %s", request_id)


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request):
    request_id = str(uuid4())
    current = request.app.state.current

    input_df = pd.DataFrame([payload.model_dump()])
    features = build_features(input_df)

    start = perf_counter()
    prediction = current.model.predict(features)[0]
    latency_ms = (perf_counter() - start) * 1000

    # Naive input is read as UTC: the column is timestamptz and the training data
    # is UTC, so letting the session timezone decide would silently shift hours.
    target_ts = payload.timestamp
    if target_ts.tzinfo is None:
        target_ts = target_ts.replace(tzinfo=timezone.utc)

    _record(
        request.app,
        current,
        request_id=request_id,
        endpoint="predict",
        batch_size=1,
        duration_ms=latency_ms,
        predictions=np.asarray([prediction]),
        feature_frame=features,
        target_ts=target_ts,
    )

    return PredictionResponse(
        request_id=request_id,
        model_version=current.version,
        prediction_mw=float(prediction),
        inference_latency_ms=latency_ms,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictionRequest, request: Request):
    request_id = str(uuid4())
    current = request.app.state.current
    start = perf_counter()

    # Convert records to DataFrame efficiently
    input_df = pd.DataFrame([r.model_dump() for r in payload.records])

    # Vectorized feature engineering
    features = build_features(input_df)

    # Single model call for the whole batch
    predictions = current.model.predict(features)

    # Ensure predictions are finite
    predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)

    processing_time_ms = (perf_counter() - start) * 1000

    # No target_ts: one event covers up to 256 distinct target hours, so it cannot
    # be attributed to a single outcome. Batch requests are excluded from
    # performance reporting by design.
    _record(
        request.app,
        current,
        request_id=request_id,
        endpoint="predict/batch",
        batch_size=len(payload.records),
        duration_ms=processing_time_ms,
        predictions=predictions,
        feature_frame=features,
    )

    return BatchPredictionResponse(
        request_id=request_id,
        model_version=current.version,
        batch_size=len(payload.records),
        total_processing_time_ms=processing_time_ms,
        predictions=[float(p) for p in predictions],
    )


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


def _require_token(request: Request) -> None:
    """Bearer-token guard for state-changing endpoints.

    Empty `KERN_API_TOKEN` disables the check; startup warns in that case.
    """
    token = request.app.state.settings.api_token
    if not token:
        return

    scheme, _, provided = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(provided, token):
        raise HTTPException(status_code=401, detail="missing or invalid bearer token")


@app.post("/models/{name}/{version}/promote", dependencies=[Depends(_require_token)])
def promote_model(name: str, version: str, request: Request, force: bool = False):
    state = request.app.state

    # Load first: a version that cannot be verified is never promoted, so the
    # database cannot claim to serve something this process could not run.
    try:
        model, candidate = load_version(state.pool, state.store, name, version)
    except ModelNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RegistryError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error

    production = get_production(state.pool, name)

    if production is not None and production.version == version:
        decision = policy.Decision(
            allowed=True,
            reasons=["already the production version"],
            details={"version": version},
        )
    else:
        decision = policy.evaluate_promotion(
            candidate=candidate,
            production=production,
            open_critical=policy.open_critical_incidents(state.pool, name),
            margin=state.settings.promotion_min_improvement,
        )

        if not decision.allowed and not force:
            policy.record_decision(
                state.pool,
                model_name=name,
                version=version,
                production_version=production.version if production else None,
                decision="blocked",
                reasons=decision.reasons,
                metrics=candidate.metrics,
            )
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "promotion blocked by policy",
                    "reasons": decision.reasons,
                    "metrics": candidate.metrics,
                    "production_version": production.version if production else None,
                },
            )

    try:
        promoted = promote(state.pool, name=name, version=version)
    except ModelNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    try:
        state.current = ModelState(
            name=promoted.name,
            version=promoted.version,
            model=model,
            reference_profile=promoted.reference_profile,
            reference_sample=promoted.reference_sample,
        )
    except Exception:
        # The commit already happened; the database is the source of truth and the
        # next restart converges. Surface the failure rather than pretend success.
        logger.exception("promoted %s %s but failed to swap in-memory state", name, version)
        raise HTTPException(
            status_code=500,
            detail=f"{name} {version} promoted in the registry but not loaded into this process",
        ) from None

    recorded = "forced" if force and not decision.allowed else "allowed"
    policy.record_decision(
        state.pool,
        model_name=name,
        version=version,
        production_version=production.version if production else None,
        decision=recorded,
        reasons=decision.reasons,
        metrics=candidate.metrics,
    )

    return {
        "name": promoted.name,
        "version": promoted.version,
        "stage": promoted.stage,
        "decision": recorded,
        "reasons": decision.reasons,
        "promoted_at": promoted.promoted_at.isoformat() if promoted.promoted_at else None,
    }


@app.post("/outcomes", dependencies=[Depends(_require_token)])
def ingest_outcomes(payload: OutcomeRequest, request: Request):
    state = request.app.state
    model_name = state.settings.model_name
    inserted = 0
    updated = 0

    # Validated here rather than with a field constraint: a NaN/Infinity value
    # makes the 422 body itself unserialisable, so the check must produce a
    # string detail instead of an echo of the offending input.
    if not all(math.isfinite(item.actual_mw) for item in payload.observations):
        raise HTTPException(
            status_code=422, detail="actual_mw must be a finite number"
        )

    with state.pool.connection() as conn, conn.cursor() as cursor:
        for observation in payload.observations:
            target_ts = observation.target_ts
            if target_ts.tzinfo is None:
                target_ts = target_ts.replace(tzinfo=timezone.utc)
            row = cursor.execute(
                """
                INSERT INTO observed_outcomes (model_name, target_ts, actual_mw)
                VALUES (%s, %s, %s)
                ON CONFLICT (model_name, target_ts)
                DO UPDATE SET actual_mw = EXCLUDED.actual_mw, observed_at = now()
                RETURNING (xmax = 0) AS inserted
                """,
                (model_name, target_ts, observation.actual_mw),
            ).fetchone()
            if row[0]:
                inserted += 1
            else:
                updated += 1

    return {
        "model_name": model_name,
        "inserted": inserted,
        "updated": updated,
    }


@app.get("/performance/report")
def performance_report(request: Request, window: str = "24h"):
    try:
        delta = parse_window(window)
    except (ValueError, OverflowError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return reports.performance_report(
        request.app.state.pool,
        request.app.state.current,
        window=window,
        since=datetime.now(timezone.utc) - delta,
    )


@app.get("/drift/report")
def drift_report(request: Request, window: str = "1h"):
    try:
        delta = parse_window(window)
    except (ValueError, OverflowError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    state = request.app.state
    return reports.drift_report(
        state.pool,
        state.current,
        window=window,
        since=datetime.now(timezone.utc) - delta,
        min_samples=state.settings.drift_min_samples,
    )


@app.get("/incidents")
def list_incidents(request: Request, status: str = "open", limit: int = 100):
    if status not in ("open", "resolved", "all"):
        raise HTTPException(
            status_code=400, detail="status must be open, resolved, or all"
        )
    if not 1 <= limit <= 500:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500")

    with request.app.state.pool.connection() as conn:
        rows = conn.execute(
            "SELECT id, dedupe_key, model_name, model_version, kind, subject, "
            "severity, status, summary, details, first_seen, last_seen, resolved_at "
            "FROM incidents WHERE (%s = 'all' OR status = %s) "
            "ORDER BY last_seen DESC LIMIT %s",
            (status, status, limit),
        ).fetchall()

    return {
        "incidents": [
            {
                "id": row[0],
                "dedupe_key": row[1],
                "model_name": row[2],
                "model_version": row[3],
                "kind": row[4],
                "subject": row[5],
                "severity": row[6],
                "status": row[7],
                "summary": row[8],
                "details": row[9],
                "first_seen": row[10].isoformat(),
                "last_seen": row[11].isoformat(),
                "resolved_at": row[12].isoformat() if row[12] else None,
            }
            for row in rows
        ]
    }


@app.post("/evaluate", dependencies=[Depends(_require_token)])
def evaluate_now(request: Request):
    return evaluation.run_scheduled(request.app)


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

    return reports.quality_report(
        request.app.state.pool,
        request.app.state.current,
        window=window,
        since=datetime.now(timezone.utc) - delta,
    )
