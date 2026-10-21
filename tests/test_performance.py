from datetime import datetime, timedelta, timezone

import pytest

from kern.performance import choose_forecast, summarize_performance

BASE = datetime(2025, 4, 1, tzinfo=timezone.utc)


def ts(hours: float) -> datetime:
    return BASE + timedelta(hours=hours)


def test_choose_forecast_takes_latest_prediction_before_the_target():
    rows = [(ts(0), ts(1), 100.0), (ts(0.5), ts(1), 110.0)]

    assert choose_forecast(rows) == {ts(1): 110.0}


def test_choose_forecast_ignores_predictions_made_after_the_target():
    """A prediction issued after the fact is not a forecast."""
    rows = [(ts(2), ts(1), 999.0)]

    assert choose_forecast(rows) == {}


def test_choose_forecast_ignores_rows_without_a_target():
    rows = [(ts(0), None, 5.0)]

    assert choose_forecast(rows) == {}


def test_metrics_match_hand_computed_values():
    rows = [(ts(0), ts(1), 100.0), (ts(0), ts(2), 200.0)]
    outcomes = {ts(1): 110.0, ts(2): 180.0}

    report = summarize_performance(rows, outcomes)

    assert report["samples"] == 2
    assert report["mae"] == pytest.approx(15.0)
    # sqrt(((10)^2 + (20)^2) / 2) = sqrt(250)
    assert report["rmse"] == pytest.approx(250**0.5)
    assert report["coverage"] == 1.0


def test_coverage_counts_predictions_that_never_got_an_outcome():
    rows = [(ts(0), ts(1), 100.0), (ts(0), ts(2), 200.0)]

    report = summarize_performance(rows, {ts(1): 100.0})

    assert report["samples"] == 1
    assert report["coverage"] == 0.5
    assert report["predicted_targets"] == 2


def test_no_pairs_reports_none_rather_than_a_perfect_score():
    report = summarize_performance([(ts(0), ts(1), 100.0)], {})

    assert report["mae"] is None
    assert report["rmse"] is None
    assert report["samples"] == 0
    assert report["coverage"] == 0.0


def test_empty_input_is_not_a_division_by_zero():
    report = summarize_performance([], {})

    assert report["samples"] == 0
    assert report["coverage"] == 0.0
    assert report["predicted_targets"] == 0


def test_rmse_penalises_large_errors_harder_than_mae():
    rows = [(ts(0), ts(1), 100.0), (ts(0), ts(2), 100.0)]
    outcomes = {ts(1): 100.0, ts(2): 140.0}

    report = summarize_performance(rows, outcomes)

    assert report["mae"] == pytest.approx(20.0)
    assert report["rmse"] > report["mae"]
