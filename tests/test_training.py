import numpy as np

from kern.data.features import FEATURE_COLUMNS, build_features
from kern.data.generate import generate_dataset
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


def test_features_match_generated_columns():
    """Guard against train/serve skew: time features rebuilt from the
    timestamp must agree with the columns the generator stored."""
    df = generate_dataset("2025-01-01", 200, 42)

    features = build_features(df)

    for column in ["hour", "day_of_week", "is_weekend"]:
        assert (features[column].to_numpy() == df[column].to_numpy()).all()


def test_training_uses_shared_feature_builder():
    """Training must not define its own copy of the feature logic."""
    import kern.training.train as train_module

    assert train_module.build_features is build_features
    assert list(FEATURE_COLUMNS) == list(build_features(generate_dataset("2025-01-01", 10, 42)).columns)


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
