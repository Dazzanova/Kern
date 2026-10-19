from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from xgboost import XGBRegressor

from kern.data.features import build_features

ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "models" / "baseline.json"


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


def load_model() -> XGBRegressor:
    model = XGBRegressor()
    model.load_model(MODEL_PATH)
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    yield


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
    return {"ready": hasattr(request.app.state, "model")}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request):
    request_id = str(uuid4())

    input_df = pd.DataFrame([payload.model_dump()])

    features = build_features(input_df)

    start = perf_counter()
    prediction = request.app.state.model.predict(features)[0]
    latency_ms = (perf_counter() - start) * 1000

    return PredictionResponse(
        request_id=request_id,
        model_version="baseline",
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

    return BatchPredictionResponse(
        request_id=request_id,
        model_version="baseline",
        batch_size=len(payload.records),
        total_processing_time_ms=processing_time_ms,
        predictions=[float(p) for p in predictions],
    )
