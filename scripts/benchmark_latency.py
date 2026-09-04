"""
benchmark_latency.py — Decision engine latency benchmark.

Measures end-to-end scoring latency: feature dict → model score → argmin action.
Prints a latency table (p50 / p95 / p99 / mean) to stdout.

Usage:
    python scripts/benchmark_latency.py
    python scripts/benchmark_latency.py --n 5000 --warmup 200
"""

import argparse
import time
import statistics
import sys
import os

# ── path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.serve.scorer import score_order  # noqa: E402


# ── representative synthetic order (median COD order profile) ────────────────
SAMPLE_ORDER = {
    "order_value": 826.89,
    "category": "Apparel",
    "payment_method": "COD",
    "quantity": 2,
    "discount_pct": 5.0,
    "cod_charge": 49.0,
    "account_age_days": 180,
    "prior_orders": 3,
    "prior_rto_count": 0,
    "orders_last_24h": 1,
    "device_cluster": 1,
    "pincode": "400001",
    "courier_id": "Courier_A",
}


def run_benchmark(n: int = 10_000, warmup: int = 1_000) -> None:
    latencies_ms = []

    # warmup — allow JIT / import caches to stabilise
    for _ in range(warmup):
        score_order(SAMPLE_ORDER)

    # timed runs
    for _ in range(n):
        t0 = time.perf_counter()
        score_order(SAMPLE_ORDER)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)

    latencies_ms.sort()
    p50  = latencies_ms[int(n * 0.50)]
    p95  = latencies_ms[int(n * 0.95)]
    p99  = latencies_ms[int(n * 0.99)]
    mean = statistics.mean(latencies_ms)

    print(f"\n{'─' * 42}")
    print(f"  RTO Shield — Decision Engine Latency")
    print(f"  n={n:,} iterations  |  warmup={warmup:,}")
    print(f"{'─' * 42}")
    print(f"  {'Percentile':<12} {'Latency':>10}")
    print(f"  {'─'*12} {'─'*10}")
    print(f"  {'p50':<12} {p50:>9.2f}ms")
    print(f"  {'p95':<12} {p95:>9.2f}ms")
    print(f"  {'p99':<12} {p99:>9.2f}ms")
    print(f"  {'mean':<12} {mean:>9.2f}ms")
    print(f"{'─' * 42}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTO Shield latency benchmark")
    parser.add_argument("--n", type=int, default=10_000, help="Number of timed iterations")
    parser.add_argument("--warmup", type=int, default=1_000, help="Warmup iterations")
    args = parser.parse_args()
    run_benchmark(n=args.n, warmup=args.warmup)
