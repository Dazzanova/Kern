"""Reliability benchmark: inject faults and measure detect-to-recover time.

This is an operational benchmark, not a unit test. It drives a running Kern API
through its public endpoints:

1. Quality corruption: extreme but valid feature values open quality incidents.
2. Drift: a shifted feature distribution opens a drift incident.
3. Performance decay: forecasts paired with deliberately wrong outcomes open a
   performance incident.
4. Recovery: normal traffic (or corrected outcomes) is sent and the benchmark
   waits until the incidents it opened are resolved.

The server must run a short detection interval so MTTD is observable, and must
record every feature snapshot so drift has enough samples. The recommended
launch command is in the README. State-changing automation is disabled there on
purpose: a performance incident would otherwise roll production back and retrain
mid-benchmark.

The benchmark is additive by default: it does not truncate telemetry or delete
incidents; it only sends traffic and corrects the outcomes it created. `--reset`
truncates the benchmark-owned tables before each scenario so the window starts
empty. Either way, run it against a development database, not a production one.
Ensure the API and the benchmark point at the same database when using `--reset`.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import psycopg

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_OUTPUT = Path("benchmarks/reliability_results.json")
NORMAL_TEMPERATURE = 24.5
NORMAL_HUMIDITY = 59.0
TARGET_KINDS = ("quality", "drift", "performance")

# Benchmark-owned state. Registry tables and promotion history are not touched.
BENCHMARK_TABLES = (
    "prediction_events",
    "quality_checks",
    "drift_checks",
    "incidents",
    "observed_outcomes",
    "shadow_predictions",
    "shadow_assignments",
    "canary_assignments",
    "rollback_events",
    "retraining_runs",
    "scaling_plans",
)


class BenchmarkError(RuntimeError):
    """A precondition or API call failed; the benchmark cannot continue."""


@dataclass
class ScenarioResult:
    name: str
    trigger_kind: str
    cleanup_kinds: list[str]
    status: str = "ok"
    detection_seconds: float | None = None
    recovery_seconds: float | None = None
    fault_requests: int = 0
    recovery_requests: int = 0
    incidents: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class KernClient:
    def __init__(
        self,
        base_url: str,
        token: str | None,
        timeout: float,
        *,
        seed: int = 42,
    ):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
        )
        self._rng = random.Random(seed)
        self.reference_profile: dict[str, dict[str, float]] = {}
        self.reference_sample: dict[str, list[float]] = {}

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as error:
            raise BenchmarkError(f"{method} {path} failed: {error}") from error

        if response.status_code >= 400:
            raise BenchmarkError(
                f"{method} {path} returned {response.status_code}: "
                f"{response.text[:300]}"
            )
        return response.json()

    def ready(self) -> dict[str, Any]:
        return self._request("GET", "/ready")

    def production_version(self) -> str | None:
        versions = self._request("GET", "/models").get("versions", [])
        for version in versions:
            if version.get("stage") == "production":
                return version.get("version")
        return None

    def production_model(self) -> tuple[str, str] | None:
        versions = self._request("GET", "/models").get("versions", [])
        for version in versions:
            if version.get("stage") == "production":
                return version.get("name"), version.get("version")
        return None

    def model_detail(self, name: str, version: str) -> dict[str, Any]:
        return self._request("GET", f"/models/{name}/{version}")

    def sample_normal(self) -> tuple[float, float]:
        """Draw an input from the production model's reference distribution.

        Constant "normal" values would look like a point mass to PSI even after
        the fault is diluted, and a Gaussian fitted to the profile mean/std puts
        more mass outside p01/p99 than the real training data. Prefer the stored
        empirical sample; fall back to the profile quantiles when it is absent.
        """
        temperatures = self.reference_sample.get("temperature_c") or []
        humidities = self.reference_sample.get("humidity_pct") or []
        if temperatures and humidities:
            index = self._rng.randrange(len(temperatures))
            return temperatures[index], humidities[index % len(humidities)]

        return (
            self._sample_quantile("temperature_c"),
            self._sample_quantile("humidity_pct"),
        )

    def _sample_quantile(self, feature: str) -> float:
        profile = self.reference_profile.get(feature, {})
        points = [
            (0.01, profile.get("p01")),
            (0.25, profile.get("p25")),
            (0.50, profile.get("p50")),
            (0.75, profile.get("p75")),
            (0.99, profile.get("p99")),
        ]
        points = [(quantile, value) for quantile, value in points if value is not None]
        if not points:
            return NORMAL_TEMPERATURE if feature == "temperature_c" else NORMAL_HUMIDITY

        draw = self._rng.random()
        if draw <= points[0][0]:
            return float(points[0][1])
        for (low_q, low_v), (high_q, high_v) in zip(points, points[1:]):
            if draw <= high_q:
                span = high_q - low_q
                if span <= 0:
                    return float(high_v)
                weight = (draw - low_q) / span
                return float(low_v + weight * (high_v - low_v))
        return float(points[-1][1])

    def shadow_version(self) -> str | None:
        return self._request("GET", "/shadow").get("shadow_version")

    def canary_version(self) -> str | None:
        return self._request("GET", "/canary").get("canary_version")

    def open_incidents(self) -> list[dict[str, Any]]:
        return self._request(
            "GET", "/incidents", params={"status": "open", "limit": 500}
        ).get("incidents", [])

    def quality_sample_count(self) -> int:
        return int(
            self._request("GET", "/quality/report", params={"window": "1h"}).get(
                "sample_count", 0
            )
        )

    def predict(
        self,
        *,
        temperature_c: float,
        humidity_pct: float,
        timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        target = timestamp or datetime.now(timezone.utc)
        return self._request(
            "POST",
            "/predict",
            json={
                "timestamp": target.isoformat(),
                "temperature_c": temperature_c,
                "humidity_pct": humidity_pct,
            },
        )

    def record_outcomes(self, observations: list[tuple[datetime, float]]) -> None:
        if not observations:
            return
        self._request(
            "POST",
            "/outcomes",
            json={
                "observations": [
                    {"target_ts": target.isoformat(), "actual_mw": actual}
                    for target, actual in observations
                ]
            },
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure Kern's detect-to-recover reliability against injected faults."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--token", default=os.environ.get("KERN_API_TOKEN") or None)
    parser.add_argument(
        "--database-url",
        default=os.environ.get("KERN_DATABASE_URL") or None,
        help="Database used by the API; required with --reset",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Truncate benchmark tables before each scenario for an isolated "
            "measurement (development databases only)"
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=("all", "quality", "drift", "performance"),
        default="all",
        help="Run one scenario, or all of them in sequence (default: all)",
    )
    parser.add_argument("--detection-timeout", type=float, default=180.0)
    parser.add_argument("--recovery-timeout", type=float, default=180.0)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--baseline-requests",
        type=int,
        default=20,
        help="Normal requests sent before the scenarios to verify feature sampling",
    )
    parser.add_argument(
        "--quality-count",
        type=int,
        default=60,
        help="Extreme requests used to open quality incidents",
    )
    parser.add_argument(
        "--drift-count",
        type=int,
        default=150,
        help="Shifted requests used to open a drift incident (must exceed drift_min_samples)",
    )
    parser.add_argument(
        "--performance-count",
        type=int,
        default=30,
        help="Forecasts with bad outcomes used to open a performance incident",
    )
    parser.add_argument(
        "--performance-offset",
        type=float,
        default=500.0,
        help="MW error added to each performance-scenario outcome",
    )
    parser.add_argument(
        "--recovery-count",
        type=int,
        default=8000,
        help=(
            "Normal requests sent to dilute a quality or drift incident; the "
            "quality range check warns above 5% out-of-range samples, so a "
            "150-sample drift fault needs roughly 5,000 normal samples to clear"
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def reset_benchmark_tables(database_url: str) -> None:
    """Clear benchmark-owned rows so each scenario starts from an empty window.

    Without this, prior telemetry dilutes a new fault: a 150-sample drift
    injection into a 3,000-sample hour never crosses the PSI threshold, and
    normal traffic cannot recover a window that already contains a point mass
    from an earlier scenario.
    """
    tables = ", ".join(BENCHMARK_TABLES)
    with psycopg.connect(database_url) as conn:
        conn.execute(f"TRUNCATE {tables} RESTART IDENTITY")
    # The telemetry writer batches at 0.25s. Give it time to drain any events
    # that were queued under the old window before the fault starts.
    time.sleep(1.0)


def load_reference_sample(
    database_url: str, model_name: str, version: str
) -> dict[str, list[float]]:
    """Read the stored empirical sample so recovery traffic matches training."""
    with psycopg.connect(database_url) as conn:
        row = conn.execute(
            "SELECT reference_sample FROM model_versions WHERE name = %s AND version = %s",
            (model_name, version),
        ).fetchone()
    if row is None or not row[0]:
        return {}
    return {
        feature: [float(value) for value in values]
        for feature, values in row[0].items()
        if values
    }


def open_kinds(client: KernClient, kinds: list[str]) -> list[dict[str, Any]]:
    wanted = set(kinds)
    return [
        incident
        for incident in client.open_incidents()
        if incident.get("kind") in wanted
    ]


def wait_for_kind(
    client: KernClient,
    kind: str,
    started: float,
    *,
    timeout: float,
    poll: float,
) -> tuple[float | None, list[dict[str, Any]]]:
    """Wait until an open incident of `kind` exists, returning elapsed seconds."""
    deadline = started + timeout
    while True:
        matches = open_kinds(client, [kind])
        if matches:
            return time.monotonic() - started, matches
        if time.monotonic() >= deadline:
            return None, []
        time.sleep(poll)


def wait_until_resolved(
    client: KernClient,
    kinds: list[str],
    started: float,
    *,
    timeout: float,
    poll: float,
) -> tuple[float | None, list[dict[str, Any]]]:
    """Wait until no open incident of any `kind` remains."""
    deadline = started + timeout
    while True:
        remaining = open_kinds(client, kinds)
        if not remaining:
            return time.monotonic() - started, []
        if time.monotonic() >= deadline:
            return None, remaining
        time.sleep(poll)


def send_normal(
    client: KernClient, count: int, *, timestamp: datetime | None = None
) -> None:
    for _ in range(count):
        temperature, humidity = client.sample_normal()
        client.predict(
            temperature_c=temperature,
            humidity_pct=humidity,
            timestamp=timestamp,
        )


def recover_with_normal_traffic(
    client: KernClient,
    kinds: list[str],
    *,
    count: int,
    chunk: int = 250,
    poll: float,
) -> tuple[int, float | None, list[dict[str, Any]]]:
    """Send normal traffic until the incidents resolve or the budget is spent.

    Dilution is how a quality or drift incident recovers: the bad samples stay in
    the one-hour window, so enough normal samples must outweigh them. Checking
    after each chunk stops early instead of always sending the full budget.
    """
    started = time.monotonic()
    sent = 0
    while sent < count:
        batch = min(chunk, count - sent)
        send_normal(client, batch)
        sent += batch
        if not open_kinds(client, kinds):
            return sent, time.monotonic() - started, []
        time.sleep(poll)
    return sent, None, open_kinds(client, kinds)


def run_quality_scenario(
    client: KernClient, args: argparse.Namespace
) -> ScenarioResult:
    result = ScenarioResult(
        name="quality-corruption",
        trigger_kind="quality",
        cleanup_kinds=["quality", "drift"],
    )
    started = time.monotonic()
    for _ in range(args.quality_count):
        # Valid per the request schema, but outside the training range.
        client.predict(temperature_c=70.0, humidity_pct=95.0)
    result.fault_requests = args.quality_count

    result.detection_seconds, hits = wait_for_kind(
        client,
        "quality",
        started,
        timeout=args.detection_timeout,
        poll=args.poll_interval,
    )
    result.incidents = hits
    if hits:
        result.notes.append(f"detected {len(hits)} quality incident(s)")
    else:
        result.status = "detection_timeout"
        result.notes.append(
            "no quality incident appeared; check KERN_EVALUATION_INTERVAL and "
            "KERN_FEATURE_SAMPLE_RATE"
        )

    sent, elapsed, remaining = recover_with_normal_traffic(
        client,
        result.cleanup_kinds,
        count=args.recovery_count,
        poll=args.poll_interval,
    )
    result.recovery_requests = sent
    result.recovery_seconds = elapsed
    if elapsed is None:
        result.status = "recovery_timeout"
        result.notes.append(
            f"{len(remaining)} incident(s) still open after {sent} normal requests"
        )
    return result


def run_drift_scenario(client: KernClient, args: argparse.Namespace) -> ScenarioResult:
    result = ScenarioResult(
        name="feature-drift",
        trigger_kind="drift",
        cleanup_kinds=["quality", "drift"],
    )
    started = time.monotonic()
    for _ in range(args.drift_count):
        # Shift both monitored inputs far outside the reference distribution.
        client.predict(temperature_c=70.0, humidity_pct=70.0)
    result.fault_requests = args.drift_count

    result.detection_seconds, hits = wait_for_kind(
        client,
        "drift",
        started,
        timeout=args.detection_timeout,
        poll=args.poll_interval,
    )
    result.incidents = hits
    if hits:
        result.notes.append(f"detected {len(hits)} drift incident(s)")
    else:
        result.status = "detection_timeout"
        result.notes.append(
            "no drift incident appeared; drift needs at least "
            "KERN_DRIFT_MIN_SAMPLES recorded feature snapshots"
        )

    sent, elapsed, remaining = recover_with_normal_traffic(
        client,
        result.cleanup_kinds,
        count=args.recovery_count,
        poll=args.poll_interval,
    )
    result.recovery_requests = sent
    result.recovery_seconds = elapsed
    if elapsed is None:
        result.status = "recovery_timeout"
        result.notes.append(
            f"{len(remaining)} incident(s) still open after {sent} normal requests"
        )
    return result


def run_performance_scenario(
    client: KernClient, args: argparse.Namespace
) -> ScenarioResult:
    result = ScenarioResult(
        name="performance-decay",
        trigger_kind="performance",
        cleanup_kinds=["performance"],
    )
    started = time.monotonic()
    now = datetime.now(timezone.utc)
    forecasts: list[tuple[datetime, float]] = []
    for index in range(args.performance_count):
        target = now + timedelta(hours=index + 1)
        temperature, humidity = client.sample_normal()
        body = client.predict(
            temperature_c=temperature,
            humidity_pct=humidity,
            timestamp=target,
        )
        forecasts.append((target, float(body["prediction_mw"])))
    client.record_outcomes(
        [
            (target, prediction + args.performance_offset)
            for target, prediction in forecasts
        ]
    )
    result.fault_requests = len(forecasts)

    result.detection_seconds, hits = wait_for_kind(
        client,
        "performance",
        started,
        timeout=args.detection_timeout,
        poll=args.poll_interval,
    )
    result.incidents = hits
    if hits:
        result.notes.append(f"detected {len(hits)} performance incident(s)")
    else:
        result.status = "detection_timeout"
        result.notes.append(
            "no performance incident appeared; the forecast/outcome pairing "
            "needs at least KERN_PERFORMANCE_MIN_SAMPLES pairs"
        )

    # Correcting the outcomes makes MAE fall without deleting the forecasts.
    recovery_start = time.monotonic()
    client.record_outcomes(forecasts)
    elapsed, remaining = wait_until_resolved(
        client,
        result.cleanup_kinds,
        recovery_start,
        timeout=args.recovery_timeout,
        poll=args.poll_interval,
    )
    result.recovery_seconds = elapsed
    if elapsed is None:
        result.status = "recovery_timeout"
        result.notes.append(f"{len(remaining)} performance incident(s) still open")
    return result


def check_preconditions(
    client: KernClient, args: argparse.Namespace
) -> tuple[str, str, dict[str, Any]]:
    ready = client.ready()
    if not ready.get("ready"):
        raise BenchmarkError(f"API is not ready: {ready}")

    version = client.production_version()
    production = client.production_model()
    if version is None or production is None:
        raise BenchmarkError("no production version is registered")

    model_name, version = production
    detail = client.model_detail(model_name, version)
    profile = detail.get("reference_profile") or {}
    if not profile:
        raise BenchmarkError(
            f"production version {version} has no reference profile; drift "
            "cannot be measured"
        )

    shadow = client.shadow_version()
    canary = client.canary_version()
    if shadow or canary:
        raise BenchmarkError(
            "shadow/canary traffic is attached; detach both so every prediction "
            "is recorded against production"
        )

    existing = [
        incident
        for incident in client.open_incidents()
        if incident.get("kind") in TARGET_KINDS
    ]
    if existing and not args.reset:
        summary = ", ".join(
            f"{incident['kind']} (id {incident['id']})" for incident in existing
        )
        raise BenchmarkError(
            "open incidents already exist; resolve them before measuring detection: "
            + summary
        )
    return model_name, version, profile


def check_feature_sampling(client: KernClient, requests: int) -> None:
    """Fail early when feature snapshots are sampled too sparsely for drift."""
    before = client.quality_sample_count()
    send_normal(client, requests)
    time.sleep(1.0)
    observed = client.quality_sample_count() - before
    minimum = max(1, requests // 2)
    if observed < minimum:
        raise BenchmarkError(
            f"only {observed} of {requests} requests stored features; start the API "
            "with KERN_FEATURE_SAMPLE_RATE=1.0 so quality and drift have evidence"
        )


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    client = KernClient(args.base_url, args.token, args.request_timeout, seed=args.seed)
    started_at = datetime.now(timezone.utc)
    try:
        model_name, version, profile = check_preconditions(client, args)
        client.reference_profile = profile
        if args.database_url:
            client.reference_sample = load_reference_sample(
                args.database_url, model_name, version
            )
        print(f"reliability benchmark against {args.base_url}")
        print(f"production version: {version}")
        if args.reset:
            if not args.database_url:
                raise BenchmarkError(
                    "--reset requires --database-url or KERN_DATABASE_URL"
                )
            print("reset: benchmark tables will be truncated before each scenario")
        check_feature_sampling(client, args.baseline_requests)
        print(f"feature sampling: ok ({args.baseline_requests} normal requests)")

        runners = {
            "quality": run_quality_scenario,
            "drift": run_drift_scenario,
            "performance": run_performance_scenario,
        }
        selected = list(runners) if args.scenario == "all" else [args.scenario]

        results: list[ScenarioResult] = []
        blocked = False
        for name in selected:
            if blocked:
                results.append(
                    ScenarioResult(
                        name=name,
                        trigger_kind=name,
                        cleanup_kinds=[],
                        status="skipped",
                        notes=["an earlier scenario left incidents open"],
                    )
                )
                continue

            if args.reset:
                reset_benchmark_tables(args.database_url)
            result = runners[name](client, args)
            results.append(result)
            current_version = client.production_version()
            if current_version != version:
                result.status = "production_changed"
                result.notes.append(
                    f"production moved from {version} to {current_version}; "
                    "disable rollback/retraining during the benchmark"
                )
                blocked = True
            elif result.status != "ok" and not args.reset:
                # Without a reset, even a detection timeout leaves fault traffic
                # in the window, so the next scenario starts contaminated.
                blocked = True
    finally:
        client.close()

    finished_at = datetime.now(timezone.utc)
    payload = {
        "metadata": {
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "base_url": args.base_url,
            "production_version": version if "version" in locals() else None,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "config": {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in vars(args).items()
                if key not in {"token", "database_url"}
            },
        },
        "scenarios": [asdict(result) for result in results],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    print()
    print(
        f"{'scenario':<22} {'detect':>8} {'recover':>8} "
        f"{'fault':>7} {'normal':>7}  status"
    )
    for result in results:
        detect = (
            f"{result.detection_seconds:.1f}s"
            if result.detection_seconds is not None
            else "-"
        )
        recover = (
            f"{result.recovery_seconds:.1f}s"
            if result.recovery_seconds is not None
            else "-"
        )
        print(
            f"{result.name:<22} {detect:>8} {recover:>8} "
            f"{result.fault_requests:>7} {result.recovery_requests:>7}  "
            f"{result.status}"
        )
        for note in result.notes:
            print(f"  note: {note}")

    print(f"\nwrote {args.output}")
    return payload, 0 if all(result.status == "ok" for result in results) else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _, code = run(args)
    except BenchmarkError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
