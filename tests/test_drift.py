import numpy as np
import pytest

from kern.drift import (
    PSI_FAIL,
    PSI_WARN,
    TIME_DERIVED_FEATURES,
    ks,
    ks_status,
    overall_status,
    psi,
    psi_status,
)

rng = np.random.default_rng(42)
REFERENCE = rng.normal(100.0, 10.0, size=1000).tolist()


def test_psi_of_identical_distributions_is_near_zero():
    assert psi(REFERENCE, REFERENCE) == pytest.approx(0.0, abs=1e-9)


def test_psi_of_the_same_distribution_stays_in_the_pass_band():
    other = rng.normal(100.0, 10.0, size=1000).tolist()

    assert psi(REFERENCE, other) < PSI_WARN


def test_psi_grows_with_shift():
    small = rng.normal(103.0, 10.0, size=1000).tolist()
    large = rng.normal(160.0, 10.0, size=1000).tolist()

    assert psi(REFERENCE, small) < psi(REFERENCE, large)
    assert psi(REFERENCE, large) > PSI_FAIL


def test_psi_counts_observations_outside_the_reference_range():
    """Infinite outer edges: a value far above the training range still counts."""
    inside = psi(REFERENCE, [100.0] * 500)
    outside = psi(REFERENCE, [10_000.0] * 500)

    assert outside > inside


def test_psi_rejects_an_empty_observed_sample():
    with pytest.raises(ValueError, match="empty"):
        psi(REFERENCE, [])


def test_psi_handles_a_degenerate_reference_with_duplicate_quantiles():
    """A constant feature produces duplicate quantiles; bins must still be valid."""
    value = psi([5.0] * 1000, [5.0] * 100)

    assert value == pytest.approx(0.0, abs=1e-9)


def test_ks_statistic_grows_with_shift_and_p_value_falls():
    same = rng.normal(100.0, 10.0, size=500).tolist()
    shifted = rng.normal(140.0, 10.0, size=500).tolist()

    same_stat, same_p = ks(REFERENCE, same)
    shifted_stat, shifted_p = ks(REFERENCE, shifted)

    assert shifted_stat > same_stat
    assert shifted_p < same_p


def test_ks_rejects_an_empty_observed_sample():
    with pytest.raises(ValueError, match="empty"):
        ks(REFERENCE, [])


@pytest.mark.parametrize(
    "value,expected",
    [(0.0, "pass"), (0.09, "pass"), (0.2, "warn"), (0.26, "fail")],
)
def test_psi_status_bands(value, expected):
    assert psi_status(value) == expected


@pytest.mark.parametrize(
    "p_value,expected",
    [(0.5, "pass"), (0.03, "warn"), (0.001, "fail")],
)
def test_ks_status_bands(p_value, expected):
    assert ks_status(p_value) == expected


def test_overall_status_is_driven_by_psi_only():
    checks = [
        {"method": "psi", "status": "pass"},
        # A KS failure alone must not decide the verdict: at high volume it flags
        # almost any difference.
        {"method": "ks", "status": "fail"},
    ]

    assert overall_status(checks) == "pass"


def test_overall_status_takes_the_worst_psi():
    checks = [
        {"method": "psi", "status": "pass"},
        {"method": "psi", "status": "fail"},
        {"method": "psi", "status": "warn"},
    ]

    assert overall_status(checks) == "fail"


def test_overall_status_without_psi_is_insufficient_data():
    assert overall_status([]) == "insufficient_data"


def test_time_derived_features_are_not_drift_signals():
    """A 1h window only ever contains one hour value, so comparing it to the
    training range would report permanent drift."""
    assert TIME_DERIVED_FEATURES == {
        "hour",
        "day_of_week",
        "is_weekend",
        "day_of_year",
        "month",
        "time_index_hours",
    }
    # The exogenous inputs must stay monitored.
    assert "temperature_c" not in TIME_DERIVED_FEATURES
    assert "humidity_pct" not in TIME_DERIVED_FEATURES
