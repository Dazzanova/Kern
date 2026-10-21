from types import SimpleNamespace

from kern.incidents import (
    build_findings,
    drift_findings,
    performance_findings,
    quality_findings,
)

SETTINGS = SimpleNamespace(
    performance_min_samples=20,
    performance_mae_warn=90.0,
    performance_mae_fail=120.0,
)


def check(
    feature="temperature_c",
    check_type="null_rate",
    status="fail",
    expected=0.0,
    observed=0.5,
):
    return {
        "feature": feature,
        "check_type": check_type,
        "status": status,
        "expected": expected,
        "observed": observed,
    }


def quality_report(*checks, sample_count=10):
    return {"sample_count": sample_count, "checks": list(checks)}


def drift_check(feature="temperature_c", method="psi", status="fail", statistic=0.9):
    return {
        "feature": feature,
        "method": method,
        "status": status,
        "statistic": statistic,
        "p_value": 0.001 if method == "ks" else None,
        "sample_size": 200,
    }


def performance_report(mae=100.0, samples=50, coverage=1.0, rmse=110.0):
    return {"mae": mae, "rmse": rmse, "samples": samples, "coverage": coverage}


def test_quality_fail_becomes_a_critical_finding():
    findings = quality_findings(quality_report(check(status="fail")))

    assert len(findings) == 1
    assert findings[0].kind == "quality"
    assert findings[0].subject == "temperature_c:null_rate"
    assert findings[0].severity == "critical"


def test_quality_warn_is_a_warning_and_pass_is_ignored():
    findings = quality_findings(
        quality_report(
            check(status="warn"),
            check(status="pass"),
            check(status="insufficient_data"),
        )
    )

    assert [finding.severity for finding in findings] == ["warn"]


def test_quality_findings_ignore_calendar_features():
    """`month` leaving its training range is the calendar advancing, not a defect."""
    findings = quality_findings(
        quality_report(
            check(feature="month", check_type="range", status="fail"),
            check(feature="temperature_c", check_type="range", status="fail"),
        )
    )

    assert [finding.subject for finding in findings] == ["temperature_c:range"]


def test_drift_ignores_ks_rows_and_unevaluated_reports():
    report = {
        "status": "fail",
        "checks": [
            drift_check(status="fail"),
            drift_check(method="ks", status="fail"),
        ],
    }

    findings = drift_findings(report)

    assert len(findings) == 1
    assert findings[0].subject == "temperature_c"
    assert drift_findings({"status": "insufficient_data", "checks": []}) == []
    assert drift_findings({"status": "no_reference_sample", "checks": []}) == []


def test_performance_produces_nothing_below_the_sample_gate():
    findings = performance_findings(
        performance_report(mae=500.0, samples=5),
        min_samples=20,
        mae_warn=90.0,
        mae_fail=120.0,
    )

    assert findings == []


def test_performance_bands_map_to_severity():
    def severity(mae):
        findings = performance_findings(
            performance_report(mae=mae),
            min_samples=20,
            mae_warn=90.0,
            mae_fail=120.0,
        )
        return findings[0].severity if findings else None

    assert severity(50.0) is None
    assert severity(100.0) == "warn"
    assert severity(150.0) == "critical"


def test_build_findings_marks_only_reports_with_evidence_as_evaluated():
    findings, evaluated = build_findings(
        quality_report(sample_count=0),
        {"status": "insufficient_data", "checks": []},
        performance_report(mae=None, samples=0, coverage=0.0),
        SETTINGS,
    )

    assert findings == []
    assert evaluated == set()


def test_build_findings_marks_populated_reports_as_evaluated():
    findings, evaluated = build_findings(
        quality_report(check(status="fail")),
        {"status": "fail", "checks": [drift_check()]},
        performance_report(mae=150.0),
        SETTINGS,
    )

    assert {finding.kind for finding in findings} == {
        "quality",
        "drift",
        "performance",
    }
    assert evaluated == {"quality", "drift", "performance"}
