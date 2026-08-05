"""
Latency/throughput benchmark for the /score API endpoint. Uses
TestClient (in-process, not real network) so results reflect app
processing time, not network overhead.

Run as: python -m src.benchmark --requests 500
"""
import argparse
import time
import numpy as np
from fastapi.testclient import TestClient
from src.api import app

def _base_payload(amount=50.0, time_val=5000.0):
    payload = {f"V{i}": 0.1 for i in range(1, 29)}
    payload["Time"] = time_val
    payload["Amount"] = amount
    return payload


def run_benchmark(n_requests: int, flagged_ratio: float = 0.1):
    # Mixes mostly "allow" with some "review"/"block" (SHAP-triggering) for a realistic blend.
    latencies_ms = []
    with TestClient(app) as client:
        for i in range(n_requests):
            is_flagged_case = (i % int(1 / flagged_ratio)) == 0
            amount = 5000.0 if is_flagged_case else 50.0
            payload = _base_payload(amount=amount)
            start = time.perf_counter()
            response = client.post("/score", json=payload)
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert response.status_code == 200
            latencies_ms.append(elapsed_ms)
    return latencies_ms

def report(latencies_ms: list, n_requests: int, total_wall_time_s: float):
    arr = np.array(latencies_ms)
    print(f"\n=== Benchmark results ({n_requests} requests) ===")
    print(f"Total wall time: {total_wall_time_s:.2f}s")
    print(f"Throughput: {n_requests / total_wall_time_s:.1f} req/sec")
    print(f"\nLatency (ms):")
    print(f"  p50: {np.percentile(arr, 50):.2f}")
    print(f"  p95: {np.percentile(arr, 95):.2f}")
    print(f"  p99: {np.percentile(arr, 99):.2f}")
    print(f"  min: {arr.min():.2f}")
    print(f"  max: {arr.max():.2f}")
    print(f"  mean: {arr.mean():.2f}")

def main():
    parser = argparse.ArgumentParser(description="Benchmark the /score API endpoint.")
    parser.add_argument("--requests", type=int, default=500,help="Number of requests to send (default: 500)")
    parser.add_argument("--flagged-ratio", type=float, default=0.1,help="Approximate fraction triggering review/block + SHAP (default: 0.1)")
    args = parser.parse_args()
    print(f"Running benchmark: {args.requests} requests, "f"~{args.flagged_ratio:.0%} flagged (SHAP-triggering)...")
    start = time.perf_counter()
    latencies = run_benchmark(args.requests, args.flagged_ratio)
    total_time = time.perf_counter() - start
    report(latencies, args.requests, total_time)

if __name__ == "__main__":
    main()