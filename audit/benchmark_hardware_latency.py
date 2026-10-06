"""
audit/benchmark_hardware_latency.py — Task 7 Latency Benchmark.
Runs 5 repeated runs of 1,000 requests each (cold and warm).
Records CPU model, core count, Python version, OS, and latency ranges.
"""
import os
import sys
import time
import platform
import subprocess
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.serve.scorer import score_order, validate_and_clean, resolve_pincode_info, _build_model_row, tree_cal, engine as serving_engine
from scripts.benchmark_latency import SAMPLE_ORDER

def get_cpu_info():
    try:
        if platform.system() == "Windows":
            res = subprocess.run("wmic cpu get Name,NumberOfCores,NumberOfLogicalProcessors /format:csv",
                                 capture_output=True, text=True, shell=True)
            lines = [line.strip() for line in res.stdout.splitlines() if line.strip() and not line.startswith("Node")]
            if len(lines) >= 2:
                parts = lines[1].split(",")
                if len(parts) >= 4:
                    return f"{parts[1]} ({parts[2]} cores, {parts[3]} logical)"
            return platform.processor()
        else:
            return platform.processor()
    except Exception as e:
        return f"{platform.processor()} ({e})"

def benchmark_run(n_requests=1000):
    clean, _ = validate_and_clean(SAMPLE_ORDER)
    pin_info, _ = resolve_pincode_info(clean['pincode'])
    df_row = _build_model_row(clean, pin_info)
    order_val = clean['order_value']

    # Cold run (first request)
    t0 = time.perf_counter()
    score_order(SAMPLE_ORDER)
    cold_shap = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    p = float(tree_cal.predict_proba(df_row)[0, 1])
    el = serving_engine.evaluate_interventions(order_val, p)
    min(el, key=el.get)
    cold_no_shap = (time.perf_counter() - t0) * 1000.0

    # Warm runs
    lat_shap = []
    lat_no_shap = []

    for _ in range(n_requests):
        t0 = time.perf_counter()
        score_order(SAMPLE_ORDER)
        lat_shap.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        p = float(tree_cal.predict_proba(df_row)[0, 1])
        el = serving_engine.evaluate_interventions(order_val, p)
        min(el, key=el.get)
        lat_no_shap.append((time.perf_counter() - t0) * 1000.0)

    return {
        "cold_shap_ms": cold_shap,
        "cold_no_shap_ms": cold_no_shap,
        "shap": {
            "p50": float(np.percentile(lat_shap, 50)),
            "p95": float(np.percentile(lat_shap, 95)),
            "p99": float(np.percentile(lat_shap, 99)),
            "mean": float(np.mean(lat_shap))
        },
        "no_shap": {
            "p50": float(np.percentile(lat_no_shap, 50)),
            "p95": float(np.percentile(lat_no_shap, 95)),
            "p99": float(np.percentile(lat_no_shap, 99)),
            "mean": float(np.mean(lat_no_shap))
        }
    }

def main():
    cpu_info = get_cpu_info()
    os_info = f"{platform.system()} {platform.release()} ({platform.version()})"
    py_ver = platform.python_version()

    print("=" * 80)
    print("TASK 7: HARDWARE AND LATENCY BENCHMARK (5 RUNS x 1,000 REQUESTS)")
    print("=" * 80)
    print(f"CPU:            {cpu_info}")
    print(f"OS:             {os_info}")
    print(f"Python:         {py_ver}")
    print(f"Architecture:   {platform.machine()}")
    print("-" * 80)

    runs = []
    for r in range(5):
        res = benchmark_run(1000)
        runs.append(res)
        print(f"Run {r+1}:")
        print(f"  With SHAP    : p50={res['shap']['p50']:.2f}ms, p95={res['shap']['p95']:.2f}ms, p99={res['shap']['p99']:.2f}ms (cold: {res['cold_shap_ms']:.2f}ms)")
        print(f"  Without SHAP : p50={res['no_shap']['p50']:.2f}ms, p95={res['no_shap']['p95']:.2f}ms, p99={res['no_shap']['p99']:.2f}ms (cold: {res['cold_no_shap_ms']:.2f}ms)")

    # Aggregate ranges across 5 runs
    p50_s = [r['shap']['p50'] for r in runs]
    p95_s = [r['shap']['p95'] for r in runs]
    p99_s = [r['shap']['p99'] for r in runs]

    p50_ns = [r['no_shap']['p50'] for r in runs]
    p95_ns = [r['no_shap']['p95'] for r in runs]
    p99_ns = [r['no_shap']['p99'] for r in runs]

    print("\n" + "=" * 80)
    print("LATENCY SUMMARY RANGES ACROSS 5 RUNS:")
    print("=" * 80)
    print(f"With SHAP    : p50 = [{min(p50_s):.2f} - {max(p50_s):.2f}] ms | p95 = [{min(p95_s):.2f} - {max(p95_s):.2f}] ms | p99 = [{min(p99_s):.2f} - {max(p99_s):.2f}] ms")
    print(f"Without SHAP : p50 = [{min(p50_ns):.2f} - {max(p50_ns):.2f}] ms | p95 = [{min(p95_ns):.2f} - {max(p95_ns):.2f}] ms | p99 = [{min(p99_ns):.2f} - {max(p99_ns):.2f}] ms")
    print("=" * 80)

if __name__ == "__main__":
    main()
