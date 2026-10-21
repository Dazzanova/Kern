import asyncio
import statistics
import time

import httpx

URL = "http://127.0.0.1:8000/predict"

TOTAL_REQUESTS = 10_000
CONCURRENCY = 1


PAYLOAD = {
    "timestamp": "2025-04-01T12:00:00",
    "temperature_c": 25.0,
    "humidity_pct": 55.0,
}


async def send_request(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> tuple[float, bool]:
    async with semaphore:
        start = time.perf_counter()

        try:
            response = await client.post(URL, json=PAYLOAD)

            elapsed = (time.perf_counter() - start) * 1000

            return elapsed, response.status_code == 200

        except httpx.HTTPError:
            elapsed = (time.perf_counter() - start) * 1000

            return elapsed, False


async def main() -> None:
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async with httpx.AsyncClient(
        timeout=30.0,
        limits=httpx.Limits(
            max_connections=CONCURRENCY,
            max_keepalive_connections=CONCURRENCY,
        ),
    ) as client:
        await client.get("http://127.0.0.1:8000/health")

        tasks = [
            send_request(client, semaphore)
            for _ in range(TOTAL_REQUESTS)
        ]

        start = time.perf_counter()

        results = await asyncio.gather(*tasks)

        total_time = time.perf_counter() - start

    latencies = [latency for latency, success in results]
    successes = sum(success for _, success in results)

    latencies.sort()

    p50 = statistics.median(latencies)
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]

    throughput = TOTAL_REQUESTS / total_time
    error_rate = (TOTAL_REQUESTS - successes) / TOTAL_REQUESTS * 100

    print(f"Requests:       {TOTAL_REQUESTS}")
    print(f"Concurrency:    {CONCURRENCY}")
    print(f"Successful:     {successes}")
    print(f"Error rate:     {error_rate:.2f}%")
    print(f"P50 latency:    {p50:.2f} ms")
    print(f"P95 latency:    {p95:.2f} ms")
    print(f"P99 latency:    {p99:.2f} ms")
    print(f"Throughput:     {throughput:.2f} req/s")


if __name__ == "__main__":
    asyncio.run(main())
