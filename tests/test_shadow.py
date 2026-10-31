from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from kern import shadow
from kern.shadow import agreement, build_report, verdict


@pytest.mark.db
def test_attach_upserts_and_detach_is_idempotent(pg):
    shadow.attach(pg, model_name="load-forecast", version="cand-a")
    shadow.attach(pg, model_name="load-forecast", version="cand-b")

    with pg.connection() as conn:
        assert conn.execute("SELECT count(*) FROM shadow_assignments").fetchone()[0] == 1

    assert shadow.get_assignment(pg, "load-forecast") == "cand-b"
    assert shadow.detach(pg, "load-forecast") is True
    assert shadow.detach(pg, "load-forecast") is False
    assert shadow.get_assignment(pg, "load-forecast") is None


@pytest.mark.db
def test_report_pairs_by_request_and_scores_both_models(pg):
    now = datetime.now(timezone.utc)
    t1, t2, t3 = (now + timedelta(hours=n) for n in (1, 2, 3))
    req1, req2, req3 = uuid4(), uuid4(), uuid4()

    with pg.connection() as conn:
        conn.execute(
            "INSERT INTO prediction_events (request_id, model_name, model_version, "
            "endpoint, batch_size, duration_ms, prediction_mean, target_ts) VALUES "
            "(%s, 'load-forecast', 'v1', 'predict', 1, 1.0, 100.0, %s), "
            "(%s, 'load-forecast', 'v1', 'predict', 1, 1.0, 200.0, %s)",
            (req1, t1, req2, t2),
        )
        conn.execute(
            "INSERT INTO shadow_predictions (request_id, model_name, model_version, "
            "production_version, endpoint, batch_size, duration_ms, prediction_mean, "
            "target_ts) VALUES "
            "(%s, 'load-forecast', 'cand', 'v1', 'predict', 1, 1.0, 90.0, %s), "
            "(%s, 'load-forecast', 'cand', 'v1', 'predict', 1, 1.0, 210.0, %s), "
            "(%s, 'load-forecast', 'cand', 'v1', 'predict', 1, 1.0, 5.0, %s)",
            (req1, t1, req2, t2, req3, t3),
        )
        conn.execute(
            "INSERT INTO observed_outcomes (model_name, target_ts, actual_mw) VALUES "
            "('load-forecast', %s, 105.0), ('load-forecast', %s, 205.0)",
            (t1, t2),
        )

    report = build_report(
        pg,
        model_name="load-forecast",
        shadow_version="cand",
        production_version="v1",
        window="24h",
        since=now - timedelta(minutes=5),
        min_samples=2,
    )

    # The shadow-only request has no production pair and must not appear.
    assert report["paired_samples"] == 2
    assert report["agreement"] == {
        "samples": 2,
        "mean_abs_delta": 10.0,
        "max_abs_delta": 10.0,
    }
    assert report["production"]["mae"] == 5.0
    assert report["production"]["coverage"] == 1.0
    assert report["shadow"]["mae"] == 10.0
    assert report["shadow"]["coverage"] == 1.0
    assert report["verdict"] == "shadow_worse"


@pytest.mark.db
def test_report_is_inconclusive_below_the_sample_gate(pg):
    now = datetime.now(timezone.utc)
    target = now + timedelta(hours=1)
    request_id = uuid4()

    with pg.connection() as conn:
        conn.execute(
            "INSERT INTO prediction_events (request_id, model_name, model_version, "
            "endpoint, batch_size, duration_ms, prediction_mean, target_ts) "
            "VALUES (%s, 'load-forecast', 'v1', 'predict', 1, 1.0, 100.0, %s)",
            (request_id, target),
        )
        conn.execute(
            "INSERT INTO shadow_predictions (request_id, model_name, model_version, "
            "production_version, endpoint, batch_size, duration_ms, prediction_mean, "
            "target_ts) VALUES (%s, 'load-forecast', 'cand', 'v1', 'predict', 1, "
            "1.0, 90.0, %s)",
            (request_id, target),
        )
        conn.execute(
            "INSERT INTO observed_outcomes (model_name, target_ts, actual_mw) "
            "VALUES ('load-forecast', %s, 95.0)",
            (target,),
        )

    report = build_report(
        pg,
        model_name="load-forecast",
        shadow_version="cand",
        production_version="v1",
        window="24h",
        since=now - timedelta(minutes=5),
        min_samples=5,
    )

    assert report["shadow"]["samples"] == 1
    assert report["verdict"] == "inconclusive"


@pytest.mark.db
def test_report_without_an_assignment_is_explicit(pg):
    report = build_report(
        pg,
        model_name="load-forecast",
        shadow_version=None,
        production_version="v1",
        window="24h",
        since=datetime.now(timezone.utc),
        min_samples=20,
    )

    assert report["shadow_version"] is None
    assert report["paired_samples"] == 0
    assert report["agreement"] == {
        "samples": 0,
        "mean_abs_delta": None,
        "max_abs_delta": None,
    }
    assert report["production"] is None
    assert report["verdict"] == "inconclusive"


def test_agreement_reports_mean_and_max_delta():
    assert agreement([(10.0, 8.0), (5.0, 9.0), (None, 4.0), (3.0, None)]) == {
        "samples": 2,
        "mean_abs_delta": 3.0,
        "max_abs_delta": 4.0,
    }


def test_agreement_with_no_pairs_is_null_not_zero():
    assert agreement([]) == {
        "samples": 0,
        "mean_abs_delta": None,
        "max_abs_delta": None,
    }


def summary(mae, samples=50):
    return {"mae": mae, "rmse": None, "samples": samples, "coverage": 1.0}


def test_verdict_compares_mae_and_respects_the_sample_gate():
    assert verdict(summary(10.0), summary(9.0), min_samples=20) == "shadow_better"
    assert verdict(summary(10.0), summary(11.0), min_samples=20) == "shadow_worse"
    assert verdict(summary(10.0), summary(10.0), min_samples=20) == "tie"
    assert verdict(summary(10.0, 5), summary(9.0, 5), min_samples=20) == "inconclusive"


def test_verdict_is_inconclusive_without_an_mae():
    assert verdict(summary(None), summary(9.0), min_samples=20) == "inconclusive"
    assert verdict(summary(10.0), summary(None), min_samples=20) == "inconclusive"
