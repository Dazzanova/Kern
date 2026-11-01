"""Automated rollback: restore the last good version when production fails.

The decision is deliberately narrow. Only a critical performance incident first seen
after the current version was promoted may authorize a rollback; quality and drift
describe inputs that an older model would also face.

`attempt` performs the registry change and returns the loaded model. It never touches
the serving process's in-memory state, so the app owns the swap.
"""

import logging
from datetime import datetime, timedelta, timezone

from psycopg.types.json import Jsonb

from kern import canary, shadow
from kern.registry import (
    ModelNotFound,
    ModelVersion,
    RegistryError,
    get_production,
    get_version,
    load_version,
    promote,
)

logger = logging.getLogger(__name__)


def eligible_incident(
    pool, *, model_name: str, version: str, promoted_at: datetime
) -> dict | None:
    """An open critical performance incident opened after this version went live."""
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT id, summary, details, first_seen FROM incidents "
            "WHERE model_name = %s AND model_version = %s AND kind = 'performance' "
            "AND severity = 'critical' AND status = 'open' AND first_seen >= %s "
            "ORDER BY first_seen DESC LIMIT 1",
            (model_name, version, promoted_at),
        ).fetchone()
    if row is None:
        return None
    return {
        "id": row[0],
        "summary": row[1],
        "details": row[2] or {},
        "first_seen": row[3],
    }


def previous_version(
    pool, *, model_name: str, from_version: str
) -> ModelVersion | None:
    """The most recently promoted archived version that has never been rolled back from."""
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT version FROM model_versions v "
            "WHERE v.name = %s AND v.stage = 'archived' "
            "AND v.promoted_at IS NOT NULL AND v.version <> %s "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM rollback_events r "
            "  WHERE r.model_name = v.name AND r.from_version = v.version "
            "  AND r.decision = 'rolled_back'"
            ") "
            "ORDER BY v.promoted_at DESC LIMIT 1",
            (model_name, from_version),
        ).fetchone()
    if row is None:
        return None
    return get_version(pool, model_name, row[0])


def last_attempt_at(pool, model_name: str) -> datetime | None:
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT max(created_at) FROM rollback_events WHERE model_name = %s",
            (model_name,),
        ).fetchone()
    return row[0]


def record(
    pool,
    *,
    model_name: str,
    from_version: str,
    to_version: str | None,
    trigger: str,
    decision: str,
    reasons: list[str],
    metrics: dict | None,
) -> dict:
    with pool.connection() as conn:
        row = conn.execute(
            "INSERT INTO rollback_events (model_name, from_version, to_version, "
            "trigger, decision, reasons, metrics) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) "
            "RETURNING id, created_at",
            (
                model_name,
                from_version,
                to_version,
                trigger,
                decision,
                Jsonb(reasons),
                Jsonb(metrics) if metrics is not None else None,
            ),
        ).fetchone()

    return {
        "id": row[0],
        "model_name": model_name,
        "from_version": from_version,
        "to_version": to_version,
        "trigger": trigger,
        "decision": decision,
        "reasons": reasons,
        "metrics": metrics,
        "created_at": row[1].isoformat(),
    }


def list_events(pool, model_name: str, *, limit: int = 20) -> list[dict]:
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT id, from_version, to_version, trigger, decision, reasons, "
            "metrics, created_at FROM rollback_events "
            "WHERE model_name = %s ORDER BY created_at DESC, id DESC LIMIT %s",
            (model_name, limit),
        ).fetchall()

    return [
        {
            "id": row[0],
            "model_name": model_name,
            "from_version": row[1],
            "to_version": row[2],
            "trigger": row[3],
            "decision": row[4],
            "reasons": row[5],
            "metrics": row[6],
            "created_at": row[7].isoformat(),
        }
        for row in rows
    ]


def _incident_metrics(incident: dict) -> dict:
    details = incident.get("details") or {}
    return {
        "incident_id": incident["id"],
        "mae": details.get("mae"),
        "rmse": details.get("rmse"),
        "samples": details.get("samples"),
        "coverage": details.get("coverage"),
        "warn_threshold": details.get("warn_threshold"),
        "fail_threshold": details.get("fail_threshold"),
    }


def _refusal(
    pool,
    *,
    model_name: str,
    from_version: str,
    trigger: str,
    reasons: list[str],
    metrics: dict,
) -> dict:
    event = record(
        pool,
        model_name=model_name,
        from_version=from_version,
        to_version=None,
        trigger=trigger,
        decision="refused",
        reasons=reasons,
        metrics=metrics,
    )
    logger.warning(
        "rollback refused for %s %s: %s", model_name, from_version, "; ".join(reasons)
    )
    return {
        "event": event,
        "model": None,
        "record": None,
        "canary_cleared": False,
        "shadow_cleared": False,
    }


def _detach_assignments(
    pool, *, model_name: str, from_version: str, to_version: str
) -> tuple[bool, bool]:
    canary_cleared = False
    assignment = canary.get_assignment(pool, model_name)
    if assignment is not None and assignment["version"] in (from_version, to_version):
        try:
            canary_cleared = canary.detach(pool, model_name)
        except Exception:
            logger.exception("failed to detach canary after rollback")

    shadow_cleared = False
    if shadow.get_assignment(pool, model_name) == to_version:
        try:
            shadow_cleared = shadow.detach(pool, model_name)
        except Exception:
            logger.exception("failed to detach shadow after rollback")

    return canary_cleared, shadow_cleared


def attempt(pool, store, *, model_name: str, trigger: str, settings) -> dict | None:
    """Decide and, when allowed, perform one rollback.

    Returns `None` when there is nothing to do. A refusal returns the recorded event
    with `model`/`record` null.
    """
    if trigger not in ("automatic", "manual"):
        raise ValueError(f"unknown rollback trigger: {trigger}")

    now = datetime.now(timezone.utc)
    current = get_production(pool, model_name)
    if current is None:
        return None

    incident = None
    if trigger == "automatic":
        if not settings.rollback_enabled:
            return None
        if current.promoted_at is None:
            return None

        incident = eligible_incident(
            pool,
            model_name=model_name,
            version=current.version,
            promoted_at=current.promoted_at,
        )
        if incident is None:
            return None

        if settings.rollback_cooldown_minutes > 0:
            last = last_attempt_at(pool, model_name)
            if last is not None and last > now - timedelta(
                minutes=settings.rollback_cooldown_minutes
            ):
                logger.debug(
                    "rollback skipped for %s: cooldown active until %s",
                    model_name,
                    (last + timedelta(minutes=settings.rollback_cooldown_minutes)).isoformat(),
                )
                return None

    age_hours = (
        (now - current.promoted_at).total_seconds() / 3600
        if current.promoted_at is not None
        else None
    )
    metrics: dict = _incident_metrics(incident) if incident is not None else {}
    metrics["from_promoted_at"] = (
        current.promoted_at.isoformat() if current.promoted_at else None
    )
    metrics["age_hours"] = age_hours

    reasons: list[str] = []
    if incident is not None:
        reasons.append(
            f"critical performance incident #{incident['id']}: {incident['summary']}"
        )
    else:
        reasons.append("manual rollback requested")

    if (
        trigger == "automatic"
        and settings.rollback_max_age_hours > 0
        and age_hours is not None
        and age_hours > settings.rollback_max_age_hours
    ):
        return _refusal(
            pool,
            model_name=model_name,
            from_version=current.version,
            trigger=trigger,
            reasons=reasons
            + [
                f"production {current.version} was promoted {age_hours:.1f}h ago, "
                f"beyond the {settings.rollback_max_age_hours:g}h rollback window"
            ],
            metrics=metrics,
        )

    target = previous_version(
        pool, model_name=model_name, from_version=current.version
    )
    if target is None:
        return _refusal(
            pool,
            model_name=model_name,
            from_version=current.version,
            trigger=trigger,
            reasons=reasons + ["no archived version to restore"],
            metrics=metrics,
        )

    metrics["to_promoted_at"] = (
        target.promoted_at.isoformat() if target.promoted_at else None
    )

    try:
        model, target_record = load_version(
            pool, store, model_name, target.version
        )
    except (ModelNotFound, RegistryError) as error:
        return _refusal(
            pool,
            model_name=model_name,
            from_version=current.version,
            trigger=trigger,
            reasons=reasons
            + [f"target {target.version} cannot be loaded: {error}"],
            metrics=metrics,
        )

    if target_record.feature_schema != current.feature_schema:
        return _refusal(
            pool,
            model_name=model_name,
            from_version=current.version,
            trigger=trigger,
            reasons=reasons
            + [
                f"target {target.version} feature schema differs from production"
            ],
            metrics=metrics,
        )

    promoted = promote(pool, name=model_name, version=target.version)
    canary_cleared, shadow_cleared = _detach_assignments(
        pool,
        model_name=model_name,
        from_version=current.version,
        to_version=promoted.version,
    )

    event = record(
        pool,
        model_name=model_name,
        from_version=current.version,
        to_version=promoted.version,
        trigger=trigger,
        decision="rolled_back",
        reasons=reasons,
        metrics=metrics,
    )
    logger.warning(
        "rolled back %s from %s to %s (%s)",
        model_name,
        current.version,
        promoted.version,
        trigger,
    )
    return {
        "event": event,
        "model": model,
        "record": promoted,
        "canary_cleared": canary_cleared,
        "shadow_cleared": shadow_cleared,
    }
