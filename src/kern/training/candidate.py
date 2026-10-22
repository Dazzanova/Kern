"""Build a candidate model with holdout metrics and a drift baseline.

The candidate is trained on the training split and evaluated on the temporal holdout
(`data/test.csv`). Its reference profile and sample come from the training split only, so
the drift baseline never sees the holdout that decided whether the candidate is good.

This module does no I/O: it takes DataFrames and returns a `Candidate`. The script in
`scripts/train_candidate.py` handles files, the registry, and the object store.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from kern.data.features import FEATURE_COLUMNS, build_features
from kern.quality import profile_reference, reference_sample
from kern.training.train import TARGET, train_model

PARAM_KEYS = (
    "n_estimators",
    "learning_rate",
    "max_depth",
    "subsample",
    "colsample_bytree",
    "reg_lambda",
    "random_state",
)

__all__ = ["Candidate", "build_candidate"]


@dataclass(frozen=True)
class Candidate:
    model: XGBRegressor
    metrics: dict[str, Any]
    feature_schema: list[str]
    reference_profile: dict[str, dict[str, float]]
    reference_sample: dict[str, list[float]]


def _iso(value) -> str:
    return pd.to_datetime(value, utc=True).isoformat()


def build_candidate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    reference_sample_size: int = 1000,
    **overrides: Any,
) -> Candidate:
    X_train = build_features(train_df)
    y_train = train_df[TARGET]
    X_test = build_features(test_df)
    y_test = test_df[TARGET]

    model = train_model(X_train, y_train, **overrides)
    predictions = model.predict(X_test)

    params = model.get_params()
    candidate = Candidate(
        model=model,
        metrics={
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_start": _iso(train_df["timestamp"].min()),
            "train_end": _iso(train_df["timestamp"].max()),
            "test_start": _iso(test_df["timestamp"].min()),
            "test_end": _iso(test_df["timestamp"].max()),
            "params": {key: params[key] for key in PARAM_KEYS if key in params},
        },
        feature_schema=list(FEATURE_COLUMNS),
        reference_profile=profile_reference(X_train, FEATURE_COLUMNS),
        reference_sample=reference_sample(
            X_train, FEATURE_COLUMNS, limit=reference_sample_size
        ),
    )

    return candidate
