from types import SimpleNamespace

from kern.policy import evaluate_promotion


def record(version, mae=None, **extra):
    metrics = {}
    if mae is not None:
        metrics["mae"] = mae
    metrics.update(extra)
    return SimpleNamespace(version=version, metrics=metrics)


def evaluate(candidate, production, *, open_critical=0, margin=0.02):
    return evaluate_promotion(
        candidate=candidate,
        production=production,
        open_critical=open_critical,
        margin=margin,
    )


def test_first_production_version_is_allowed():
    decision = evaluate(record("v1", 80.0), None)

    assert decision.allowed
    assert "bootstrap" in decision.reasons[0]


def test_a_candidate_past_the_margin_is_allowed():
    decision = evaluate(record("v2", 90.0), record("v1", 100.0))

    assert decision.allowed
    assert decision.reasons == []


def test_a_candidate_inside_the_margin_is_blocked():
    decision = evaluate(record("v2", 99.0), record("v1", 100.0))

    assert not decision.allowed
    assert "does not improve" in decision.reasons[0]


def test_an_equal_candidate_is_blocked_even_with_no_margin():
    blocked = evaluate(record("v2", 100.0), record("v1", 100.0), margin=0.0)
    allowed = evaluate(record("v3", 99.9), record("v1", 100.0), margin=0.0)

    assert not blocked.allowed
    assert allowed.allowed


def test_a_candidate_without_mae_is_blocked():
    decision = evaluate(record("v2"), record("v1", 100.0))

    assert not decision.allowed
    assert "candidate has no holdout MAE" in decision.reasons


def test_production_without_mae_is_blocked():
    decision = evaluate(record("v2", 10.0), record("v1"))

    assert not decision.allowed
    assert "production has no comparable holdout MAE" in decision.reasons


def test_mismatched_holdout_windows_are_blocked():
    decision = evaluate(
        record("v2", 10.0, test_start="2025-04-01", test_end="2025-04-30"),
        record("v1", 100.0, test_start="2025-05-01", test_end="2025-05-31"),
    )

    assert not decision.allowed
    assert "holdout window differs" in decision.reasons[0]


def test_matching_holdout_windows_are_comparable():
    decision = evaluate(
        record("v2", 10.0, test_start="2025-04-01", test_end="2025-04-30"),
        record("v1", 100.0, test_start="2025-04-01", test_end="2025-04-30"),
    )

    assert decision.allowed


def test_open_critical_incidents_block_promotion():
    decision = evaluate(record("v2", 10.0), record("v1", 100.0), open_critical=2)

    assert not decision.allowed
    assert "2 open critical incident(s)" in decision.reasons[0]


def test_every_blocker_is_reported_together():
    decision = evaluate(
        record("v2", 200.0), record("v1", 100.0), open_critical=1
    )

    assert not decision.allowed
    assert len(decision.reasons) == 2
    assert decision.details["candidate_mae"] == 200.0
    assert decision.details["required_mae"] == 98.0
