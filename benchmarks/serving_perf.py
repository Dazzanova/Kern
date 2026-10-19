import asyncio
import json
import platform
import statistics
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import xgboost as xgb

@dataclass
class BenchmarkResult:
    batch_size: int
    concurrency: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    req_per_sec: float
    rec_per_sec: float
    error_rate: float
    avg_server_time_ms: float

@dataclass
class EnvMetadata:
    python_version: str
    platform: str
    xgboost_version: str
    total_records: int
    warmup_count: int
    repetitions: int

async def run_benchmark(
    client: httpx.AsyncClient,
    url: str,
    records: list[dict],
    batch_size: int,
    concurrency: int,
    warmup_count: int,
) -> tuple[list[float], list[float], float, int]:
    """
    Runs a benchmark for a specific configuration.
    Returns (request_latencies, server_latencies, total_time, success_count).
    """
    # Prepare batches
    batches = [records[i : i + batch_size] for i in range(0, len(records), batch_size)]
    total_requests = len(batches)

    # Warm-up
    for i in range(min(warmup_count, total_requests)):
        try:
            await client.post(url, json={"records": batches[i]})
        except Exception:
            pass

    semaphore = asyncio.Semaphore(concurrency)

    async def send_request(batch):
        async with semaphore:
            start = time.perf_counter()
            try:
                resp = await client.post(url, json={"records": batch})
                elapsed = (time.perf_counter() - start) * 1000
                if resp.status_code == 200:
                    return elapsed, resp.json().get("total_processing_time_ms", 0.0), True
                return elapsed, 0.0, False
            except Exception:
                return (time.perf_counter() - start) * 1000, 0.0, False

    start_wall = time.perf_counter()
    tasks = [send_request(b) for b in batches]
    results = await asyncio.gather(*tasks)
    end_wall = time.perf_counter()

    total_wall_time = end_wall - start_wall

    req_latencies = [r[0] for r in results]
    srv_latencies = [r[1] for r in results if r[2]]
    successes = sum(1 for r in results if r[2])

    return req_latencies, srv_latencies, total_wall_time, successes

async def main():
    # Configuration
    BASE_URL = "http://127.0.0.1:8000/predict/batch"
    TOTAL_RECORDS = 10_000
    CONCURRENCY = 4
    WARMUP_COUNT = 50
    REPETITIONS = 3
    BATCH_SIZES = [1, 8, 32, 64, 256]

    # Deterministic input data
    records = [
        {
            "timestamp": "2025-04-01T12:00:00",
            "temperature_c": 20.0 + (i % 10),
            "humidity_pct": 50.0 + (i % 20),
        }
        for i in range(TOTAL_RECORDS)
    ]

    metadata = EnvMetadata(
        python_version=platform.python_version(),
        platform=platform.platform(),
        xgboost_version=xgb.__version__,
        total_records=TOTAL_RECORDS,
        warmup_count=WARMUP_COUNT,
        repetitions=REPETITIONS,
    )

    all_results = []

    async with httpx.AsyncClient(
        timeout=30.0,
        limits=httpx.Limits(max_connections=CONCURRENCY, max_keepalive_connections=CONCURRENCY)
    ) as client:
        # Health check
        try:
            await client.get("http://127.0.0.1:8000/health")
        except Exception as e:
            print(f"Error: API not reachable: {e}")
            return

        for batch_size in BATCH_SIZES:
            print(f"Benchmarking batch_size={batch_size}, concurrency={CONCURRENCY}...", end=" ", flush=True)

            rep_results = []
            for r in range(REPETITIONS):
                req_lats, srv_lats, wall_time, successes = await run_benchmark(
                    client, BASE_URL, records, batch_size, CONCURRENCY, WARMUP_COUNT
                )

                # Calculate metrics for this repetition
                req_lats.sort()
                p50 = statistics.median(req_lats)
                p95 = req_lats[int(len(req_lats) * 0.95)]
                p99 = req_lats[int(len(req_lats) * 0.99)]

                num_reqs = len(req_lats)
                req_per_sec = num_reqs / wall_time
                rec_per_sec = (successes * batch_size) / wall_time
                error_rate = (1 - successes / num_reqs) * 100
                avg_srv = statistics.mean(srv_lats) if srv_lats else 0.0

                rep_results.append(BenchmarkResult(
                    batch_size=batch_size,
                    concurrency=CONCURRENCY,
                    p50_ms=p50,
                    p95_ms=p95,
                    p99_ms=p99,
                    req_per_sec=req_per_sec,
                    rec_per_sec=rec_per_sec,
                    error_rate=error_rate,
                    avg_server_time_ms=avg_srv
                ))

            # Median across repetitions
            final_res = BenchmarkResult(
                batch_size=batch_size,
                concurrency=CONCURRENCY,
                p50_ms=statistics.median([r.p50_ms for r in rep_results]),
                p95_ms=statistics.median([r.p95_ms for r in rep_results]),
                p99_ms=statistics.median([r.p99_ms for r in rep_results]),
                req_per_sec=statistics.median([r.req_per_sec for r in rep_results]),
                rec_per_sec=statistics.median([r.rec_per_sec for r in rep_results]),
                error_rate=statistics.median([r.error_rate for r in rep_results]),
                avg_server_time_ms=statistics.median([r.avg_server_time_ms for r in rep_results]),
            )
            all_results.append(final_res)
            print(f"Done. Rec/s: {final_res.rec_per_sec:.2f}")

    # Summary Table
    print("\n" + "="*80)
    print(f"{'Batch':<10} {'P50 (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12} {'Req/s':<12} {'Rec/s':<12} {'Err%':<8}")
    print("-" * 80)
    for res in all_results:
        print(f"{res.batch_size:<10} {res.p50_ms:<12.2f} {res.p95_ms:<12.2f} {res.p99_ms:<12.2f} {res.req_per_sec:<12.2f} {res.rec_per_sec:<12.2f} {res.error_rate:<8.2f}")
    print("="*80)

    # Save to JSON
    output = {
        "metadata": asdict(metadata),
        "results": [asdict(r) for r in all_results]
    }
    with open("benchmarks/results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to benchmarks/results.json")

if __name__ == "__main__":
    asyncio.run(main())
