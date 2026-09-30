import numpy as np

from kern.data.generate import generate_dataset
from kern.data.features import build_features
from kern.training.train import train_model


def test_feature_columns():
    df = generate_dataset("2025-01-01", 100, 42)

    features = build_features(df)

    assert list(features.columns) == [
        "temperature_c",
        "humidity_pct",
        "hour",
        "day_of_week",
        "is_weekend",
        "day_of_year",
        "month",
        "time_index_hours",
    ]


def test_feature_count_matches_rows():
    df = generate_dataset("2025-01-01", 100, 42)

    features = build_features(df)

    assert len(features) == len(df)


def test_model_produces_finite_predictions():
    df = generate_dataset("2025-01-01", 200, 42)

    X = build_features(df)
    y = df["load_mw"]

    model = train_model(X, y)

    predictions = model.predict(X)

    assert len(predictions) == len(df)
    assert np.isfinite(predictions).all()
