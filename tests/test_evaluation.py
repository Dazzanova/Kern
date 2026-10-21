import pytest

from kern.evaluation import run_retention
from kern.incidents import Finding, sync_incidents

pytestmark = pytest.mark.db

MODEL = "load-forecast"
VERSION = "test-v1"


def finding(subject="temperature_c", severity="warn"):
    return Finding("drift", subject, severity, f"PSI for {subject}", {"psi": 0.3})


def test_sync_incidents_dedupes_and_advances_last_seen(pg):
    first = sync_incidents(
        pg,
        model_name=MODEL,
        model_version=VERSION,
        findings=[finding()],
        evaluated_kinds={"drift"},
    )
    assert first == {"findings": 1, "opened": 1, "updated": 0, "resolved": 0}

    with pg.connection() as conn:
        before = conn.execute(
            "SELECT first_seen, last_seen FROM incidents"
        ).fetchone()

    second = sync_incidents(
        pg,
        model_name=MODEL,
        model_version=VERSION,
        findings=[finding()],
        evaluated_kinds={"drift"},
    )
    assert second["opened"] == 0
    assert second["updated"] == 1

    with pg.connection() as conn:
        after = conn.execute("SELECT first_seen, last_seen FROM incidents").fetchone()

    assert after[0] == before[0]
    assert after[1] > before[1]


def test_sync_incidents_resolves_an_open_incident_when_healthy(pg):
    sync_incidents(
        pg,
        model_name=MODEL,
        model_version=VERSION,
        findings=[finding()],
        evaluated_kinds={"drift"},
    )

    result = sync_incidents(
        pg,
        model_name=MODEL,
        model_version=VERSION,
        findings=[],
        evaluated_kinds={"drift"},
    )

    assert result["resolved"] == 1
    with pg.connection() as conn:
        row = conn.execute("SELECT status, resolved_at FROM incidents").fetchone()
    assert row[0] == "resolved"
    assert row[1] is not None


def test_sync_incidents_leaves_an_incident_open_without_evidence(pg):
    sync_incidents(
        pg,
        model_name=MODEL,
        model_version=VERSION,
        findings=[finding()],
        evaluated_kinds={"drift"},
    )

    result = sync_incidents(
        pg,
        model_name=MODEL,
        model_version=VERSION,
        findings=[],
        evaluated_kinds=set(),
    )

    assert result["resolved"] == 0
    with pg.connection() as conn:
        assert conn.execute("SELECT status FROM incidents").fetchone()[0] == "open"


def test_severity_escalates_and_does_not_de_escalate(pg):
    def sync(severity):
        sync_incidents(
            pg,
            model_name=MODEL,
            model_version=VERSION,
            findings=[finding(severity=severity)],
            evaluated_kinds={"drift"},
        )

    sync("warn")
    sync("critical")
    sync("warn")

    with pg.connection() as conn:
        assert conn.execute("SELECT severity FROM incidents").fetchone()[0] == "critical"


def test_retention_deletes_only_old_rows_and_never_open_incidents(pg):
    with pg.connection() as conn:
        conn.execute(
            """
            INSERT INTO prediction_events (
                request_id, model_name, model_version, endpoint,
                batch_size, duration_ms, recorded_at
            ) VALUES
                (gen_random_uuid(), 'm', 'v1', 'predict', 1, 1.0,
                 now() - interval '40 days'),
                (gen_random_uuid(), 'm', 'v1', 'predict', 1, 1.0, now())
            """
        )
        conn.execute(
            """
            INSERT INTO incidents (
                dedupe_key, model_name, model_version, kind, subject,
                severity, summary, status, resolved_at
            ) VALUES
                ('m:drift:old', 'm', 'v1', 'drift', 'old', 'warn', 'x',
                 'resolved', now() - interval '40 days'),
                ('m:drift:open', 'm', 'v1', 'drift', 'open', 'warn', 'x',
                 'open', NULL)
            """
        )

    deleted = run_retention(pg, 30)

    assert deleted["prediction_events"] == 1
    assert deleted["incidents"] == 1
    with pg.connection() as conn:
        assert conn.execute("SELECT count(*) FROM prediction_events").fetchone()[0] == 1
        remaining = conn.execute("SELECT dedupe_key FROM incidents").fetchall()
    assert [row[0] for row in remaining] == ["m:drift:open"]


def test_retention_is_a_no_op_when_disabled(pg):
    assert run_retention(pg, 0) == {}
