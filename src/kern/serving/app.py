from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

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

    input_df = pd.DataFrame(
        [
            {
                "timestamp": payload.timestamp,
                "temperature_c": payload.temperature_c,
                "humidity_pct": payload.humidity_pct,
            }
        ]
    )

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
