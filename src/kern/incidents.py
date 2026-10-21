"""Turn report output into incidents, and keep their lifecycle.

Pure builders (`*_findings`, `build_findings`) are separated from the one function that
writes (`sync_incidents`) so the mapping rules can be tested without a database.
"""

from dataclasses import dataclass
from typing import Any

from psycopg.types.json import Jsonb

from kern.drift import TIME_DERIVED_FEATURES


@dataclass(frozen=True)
class Finding:
    kind: str
    subject: str
    severity: str
    summary: str
    details: dict[str, Any]


def quality_findings(report: dict) -> list[Finding]:
    findings: list[Finding] = []
    for check in report.get("checks", []):
        status = check["status"]
        if status not in ("warn", "fail"):
            continue
        feature = check["feature"]
        # Calendar features leaving their training range is the date advancing,
        # not a defect. The report still shows the check; it must not become an
        # incident that stays open for a whole month. Same rule as drift.
        if feature in TIME_DERIVED_FEATURES:
            continue
        check_type = check["check_type"]
        findings.append(
            Finding(
                kind="quality",
                subject=f"{feature}:{check_type}",
                severity="critical" if status == "fail" else "warn",
                summary=f"{feature} {check_type} is {status}",
                details={
                    "feature": feature,
                    "check_type": check_type,
                    "status": status,
                    "expected": check.get("expected"),
                    "observed": check.get("observed"),
                },
            )
        )
    return findings


def drift_findings(report: dict) -> list[Finding]:
    """PSI only. KS stays context, as in the report verdict."""
    if report.get("status") in ("insufficient_data", "no_reference_sample"):
        return []

    findings: list[Finding] = []
    for check in report.get("checks", []):
        if check["method"] != "psi" or check["status"] not in ("warn", "fail"):
            continue
        feature = check["feature"]
        findings.append(
            Finding(
                kind="drift",
                subject=feature,
                severity="critical" if check["status"] == "fail" else "warn",
                summary=f"PSI {check['statistic']:.3f} for {feature}",
                details={
                    "feature": feature,
                    "psi": check["statistic"],
                    "status": check["status"],
                    "sample_size": check["sample_size"],
                },
            )
        )
    return findings


def performance_findings(
    report: dict, *, min_samples: int, mae_warn: float, mae_fail: float
) -> list[Finding]:
    mae = report.get("mae")
    if mae is None or report.get("samples", 0) < min_samples:
        return []

    if mae > mae_fail:
        severity = "critical"
    elif mae > mae_warn:
        severity = "warn"
    else:
        return []

    return [
        Finding(
            kind="performance",
            subject="mae",
            severity=severity,
            summary=f"MAE {mae:.1f} mw over {report['samples']} forecasts",
            details={
                "mae": mae,
                "rmse": report.get("rmse"),
                "samples": report["samples"],
                "coverage": report.get("coverage"),
                "warn_threshold": mae_warn,
                "fail_threshold": mae_fail,
            },
        )
    ]


def build_findings(
    quality: dict, drift: dict, performance: dict, settings
) -> tuple[list[Finding], set[str]]:
    """Return current findings and the kinds that had enough evidence to judge.

    Only an evaluated kind may resolve its open incidents. A quiet window, an empty
    reference sample, or too few samples must not clear a real problem.
    """
    findings = [
        *quality_findings(quality),
        *drift_findings(drift),
        *performance_findings(
            performance,
            min_samples=settings.performance_min_samples,
            mae_warn=settings.performance_mae_warn,
            mae_fail=settings.performance_mae_fail,
        ),
    ]

    evaluated: set[str] = set()
    if quality.get("sample_count", 0) > 0:
        evaluated.add("quality")
    if drift.get("status") not in ("insufficient_data", "no_reference_sample"):
        evaluated.add("drift")
    if (
        performance.get("mae") is not None
        and performance.get("samples", 0) >= settings.performance_min_samples
    ):
        evaluated.add("performance")

    return findings, evaluated


def sync_incidents(
    pool,
    *,
    model_name: str,
    model_version: str,
    findings: list[Finding],
    evaluated_kinds: set[str],
) -> dict:
    """Upsert current findings and resolve the rest of each evaluated kind."""
    opened = 0
    updated = 0
    resolved = 0

    with pool.connection() as conn, conn.cursor() as cursor:
        for finding in findings:
            key = f"{model_name}:{finding.kind}:{finding.subject}"
            row = cursor.execute(
                """
                INSERT INTO incidents (
                    dedupe_key, model_name, model_version, kind, subject,
                    severity, summary, details
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (dedupe_key) DO UPDATE SET
                    last_seen = now(),
                    severity = CASE
                        WHEN incidents.severity = 'critical' THEN 'critical'
                        ELSE EXCLUDED.severity
                    END,
                    summary = EXCLUDED.summary,
                    details = EXCLUDED.details,
                    model_version = EXCLUDED.model_version,
                    status = 'open',
                    resolved_at = NULL
                RETURNING (xmax = 0) AS inserted
                """,
                (
                    key,
                    model_name,
                    model_version,
                    finding.kind,
                    finding.subject,
                    finding.severity,
                    finding.summary,
                    Jsonb(finding.details),
                ),
            ).fetchone()
            if row[0]:
                opened += 1
            else:
                updated += 1

        for kind in sorted(evaluated_kinds):
            keys = [
                f"{model_name}:{kind}:{finding.subject}"
                for finding in findings
                if finding.kind == kind
            ]
            cursor.execute(
                "UPDATE incidents SET status = 'resolved', resolved_at = now() "
                "WHERE model_name = %s AND kind = %s AND status = 'open' "
                "AND dedupe_key <> ALL(%s)",
                (model_name, kind, keys),
            )
            resolved += cursor.rowcount

    return {
        "findings": len(findings),
        "opened": opened,
        "updated": updated,
        "resolved": resolved,
    }
