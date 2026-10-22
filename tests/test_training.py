import numpy as np

from kern.data.features import FEATURE_COLUMNS, build_features
from kern.data.generate import generate_dataset
from kern.training.candidate import build_candidate
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


def split():
    train = generate_dataset("2025-01-01", 240, 1)
    test = generate_dataset("2025-02-01", 120, 2)
    # A sentinel far outside the training range proves the drift baseline is not
    # built from the holdout.
    test = test.assign(temperature_c=1000.0)
    return train, test


def small_candidate(**kwargs):
    train, test = split()
    return build_candidate(train, test, n_estimators=10, max_depth=2, **kwargs)


def test_candidate_records_holdout_metrics_and_their_provenance():
    candidate = small_candidate()

    metrics = candidate.metrics
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= metrics["mae"]
    assert metrics["train_rows"] == 240
    assert metrics["test_rows"] == 120
    assert metrics["train_start"].startswith("2025-01-01")
    assert metrics["train_end"].startswith("2025-01-10")
    assert metrics["test_start"].startswith("2025-02-01")
    assert metrics["test_end"].startswith("2025-02-05")
    assert metrics["params"]["n_estimators"] == 10
    assert metrics["params"]["max_depth"] == 2


def test_candidate_uses_the_declared_feature_schema_in_order():
    candidate = small_candidate()

    assert candidate.feature_schema == FEATURE_COLUMNS


def test_reference_baseline_comes_from_the_training_split_only():
    candidate = small_candidate()

    assert set(candidate.reference_sample) == set(FEATURE_COLUMNS)
    assert 1000.0 not in candidate.reference_sample["temperature_c"]
    assert candidate.reference_profile["temperature_c"]["max"] < 1000


def test_reference_sample_respects_the_limit():
    candidate = small_candidate(reference_sample_size=5)

    assert all(len(values) <= 5 for values in candidate.reference_sample.values())


def test_trained_candidate_predicts_finite_values():
    candidate = small_candidate()
    train, test = split()

    predictions = candidate.model.predict(build_features(test))

    assert len(predictions) == len(test)
    assert np.isfinite(predictions).all()
