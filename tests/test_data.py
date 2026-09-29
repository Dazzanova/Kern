from kern.data.generate import generate_dataset


def test_dataset_schema():
    df = generate_dataset("2025-01-01", 100, 42)

    assert list(df.columns) == [
        "timestamp",
        "temperature_c",
        "humidity_pct",
        "hour",
        "day_of_week",
        "is_weekend",
        "load_mw",
    ]


def test_dataset_has_no_missing_values():
    df = generate_dataset("2025-01-01", 100, 42)

    assert not df.isna().any().any()


def test_dataset_timestamps_are_monotonic():
    df = generate_dataset("2025-01-01", 100, 42)

    assert df["timestamp"].is_monotonic_increasing


def test_load_is_positive():
    df = generate_dataset("2025-01-01", 100, 42)

    assert (df["load_mw"] > 0).all()
