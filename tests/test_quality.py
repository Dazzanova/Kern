import numpy as np
import pandas as pd
import pytest

from kern.quality import evaluate_features, profile_reference, reference_sample

REFERENCE = {
    "temperature_c": {
        "min": 0.0, "max": 40.0, "mean": 20.0, "std": 5.0, "null_rate": 0.0,
        "p01": 5.0, "p25": 15.0, "p50": 20.0, "p75": 25.0, "p99": 35.0,
    }
}


def test_profile_reference_computes_expected_statistics():
    df = pd.DataFrame({"temperature_c": [10.0, 20.0, 30.0, 40.0]})

    profile = profile_reference(df, ["temperature_c"])

    stats = profile["temperature_c"]
    assert stats["min"] == 10.0
    assert stats["max"] == 40.0
    assert stats["mean"] == 25.0
    assert stats["null_rate"] == 0.0
    assert stats["p50"] == pytest.approx(25.0)


def test_profile_reference_records_null_rate():
    df = pd.DataFrame({"temperature_c": [10.0, None, 30.0, 40.0]})

    profile = profile_reference(df, ["temperature_c"])

    assert profile["temperature_c"]["null_rate"] == 0.25


def test_in_range_values_pass():
    results = evaluate_features(REFERENCE, {"temperature_c": [10.0, 20.0, 30.0]})

    assert [r["status"] for r in results] == ["pass"] * 3


def test_heavy_out_of_range_fails():
    values = [0.0] * 50 + [20.0] * 50

    results = evaluate_features(REFERENCE, {"temperature_c": values})

    range_check = next(r for r in results if r["check_type"] == "range")
    assert range_check["status"] == "fail"
    assert range_check["observed"] == pytest.approx(0.5)


def test_mild_out_of_range_warns():
    values = [0.0] * 10 + [20.0] * 90

    results = evaluate_features(REFERENCE, {"temperature_c": values})

    range_check = next(r for r in results if r["check_type"] == "range")
    assert range_check["status"] == "warn"


def test_large_mean_shift_fails():
    # Reference mean 20.0, std 5.0: 50.0 is 6 sigma away, past the >5 sigma fail
    # threshold. (The plan's 40.0 is only 4 sigma, which is a warn by design.)
    values = list(np.full(200, 50.0))

    results = evaluate_features(REFERENCE, {"temperature_c": values})

    shift = next(r for r in results if r["check_type"] == "mean_shift")
    assert shift["status"] == "fail"


def test_null_rate_above_reference_fails():
    values = [None] * 20 + [20.0] * 80

    results = evaluate_features(REFERENCE, {"temperature_c": values})

    nulls = next(r for r in results if r["check_type"] == "null_rate")
    assert nulls["status"] == "fail"


def test_fully_null_feature_fails_null_rate():
    results = evaluate_features(REFERENCE, {"temperature_c": [None] * 100})

    assert len(results) == 1
    assert results[0]["check_type"] == "null_rate"
    assert results[0]["status"] == "fail"


def test_missing_feature_is_reported():
    results = evaluate_features(REFERENCE, {})

    null_check = next(r for r in results if r["check_type"] == "null_rate")
    assert null_check["feature"] == "temperature_c"
    assert null_check["status"] == "fail"
    assert null_check["observed"] == 1.0


def test_unknown_feature_is_skipped():
    results = evaluate_features(REFERENCE, {"unknown_feature": [1.0, 2.0]})

    # Observation keys outside the reference profile are ignored. The profile's
    # own features are still reported (temperature_c has no observations here).
    assert [r for r in results if r["feature"] == "unknown_feature"] == []


def test_reference_sample_is_deterministic():
    df = pd.DataFrame({"temperature_c": np.arange(50, dtype=float)})

    first = reference_sample(df, ["temperature_c"], limit=10)
    second = reference_sample(df, ["temperature_c"], limit=10)

    assert first == second
    assert len(first["temperature_c"]) == 10


def test_reference_sample_respects_the_limit_and_drops_nulls():
    df = pd.DataFrame({"temperature_c": [1.0, None, 3.0, None, 5.0]})

    sample = reference_sample(df, ["temperature_c"], limit=100)

    assert sample["temperature_c"] == [1.0, 3.0, 5.0]


def test_reference_sample_returns_plain_floats():
    df = pd.DataFrame({"temperature_c": np.array([1, 2, 3], dtype="int64")})

    sample = reference_sample(df, ["temperature_c"])

    assert all(type(value) is float for value in sample["temperature_c"])
