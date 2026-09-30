from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from kern.data.features import build_features

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

TARGET = "load_mw"
REFERENCE_TIME = pd.Timestamp("2025-01-01")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    timestamps = pd.to_datetime(df["timestamp"])

    features = df.drop(columns=[TARGET, "timestamp"]).copy()

    features["day_of_year"] = timestamps.dt.dayofyear
    features["month"] = timestamps.dt.month
    features["time_index_hours"] = (
        timestamps - REFERENCE_TIME
    ).dt.total_seconds() / 3600

    return features


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(X_train, y_train)

    return model


def main() -> None:
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    X_train = build_features(train_df)
    y_train = train_df[TARGET]

    X_test = build_features(test_df)
    y_test = test_df[TARGET]

    model = train_model(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    MODEL_DIR.mkdir(exist_ok=True)

    model_path = MODEL_DIR / "baseline.json"
    model.save_model(model_path)

    print(f"Features: {list(X_train.columns)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
