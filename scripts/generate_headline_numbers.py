"""
scripts/generate_headline_numbers.py — One source of truth for all headline numbers.

Computes headline metrics directly from frozen artifacts and held-out test data,
then writes reports/headline_numbers.json.
"""

import json
import os
import sys
import time
import subprocess
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("RTO_SHIELD_ENV", "development")

from src.policy.cost_engine import CostEngine
from src.eval.stage4_evaluate import get_cod_subset, eval_multi_action, calc_operational_metrics
from src.serve.scorer import score_order, validate_and_clean, resolve_pincode_info, _build_model_row, tree_cal, engine as serving_engine
from scripts.benchmark_latency import SAMPLE_ORDER


def compute_ece(y_true, y_prob, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, edges) - 1, 0, n_bins - 1)
    error = 0.0
    for b in range(n_bins):
        mask = (bin_idx == b)
        if mask.any():
            error += abs(y_true[mask].mean() - y_prob[mask].mean()) * mask.mean()
    return float(error)


def canonical_realized_pl(actions, y, V, cfg):
    rto_cost = cfg["rto_logistics_cost"]
    mpct = cfg["average_margin_pct"]
    iv = cfg["interventions"]
    m = V * mpct
    base = np.where(y == 1, rto_cost, -m).sum()
    fr = np.array([iv[a]["friction_cost"] for a in actions], float)
    dr = np.array([iv[a]["success_drop_pct"] for a in actions], float)
    rr = np.array([iv[a]["rto_reduction_pct"] for a in actions], float)
    pol = np.where(y == 1, fr + (1 - rr) * rto_cost, fr - (1 - dr) * m).sum()
    return float(base - pol)


def benchmark_latency(n_iter=500, warmup=50):
    # With SHAP
    lat_shap = []
    for _ in range(warmup):
        score_order(SAMPLE_ORDER)
    for _ in range(n_iter):
        t0 = time.perf_counter()
        score_order(SAMPLE_ORDER)
        t1 = time.perf_counter()
        lat_shap.append((t1 - t0) * 1000.0)

    # Without SHAP (pure scoring + cost matrix)
    clean, _ = validate_and_clean(SAMPLE_ORDER)
    pin_info, _ = resolve_pincode_info(clean['pincode'])
    df_row = _build_model_row(clean, pin_info)
    order_val = clean['order_value']

    lat_no_shap = []
    for _ in range(warmup):
        p = float(tree_cal.predict_proba(df_row)[0, 1])
        el = serving_engine.evaluate_interventions(order_val, p)
        min(el, key=el.get)
    for _ in range(n_iter):
        t0 = time.perf_counter()
        p = float(tree_cal.predict_proba(df_row)[0, 1])
        el = serving_engine.evaluate_interventions(order_val, p)
        min(el, key=el.get)
        t1 = time.perf_counter()
        lat_no_shap.append((t1 - t0) * 1000.0)

    p50_s, p95_s, p99_s = np.percentile(lat_shap, [50, 95, 99])
    p50_ns, p95_ns, p99_ns = np.percentile(lat_no_shap, [50, 95, 99])

    return {
        "with_shap": {
            "p50_ms": round(float(p50_s), 2),
            "p95_ms": round(float(p95_s), 2),
            "p99_ms": round(float(p99_s), 2),
            "mean_ms": round(float(np.mean(lat_shap)), 2)
        },
        "without_shap": {
            "p50_ms": round(float(p50_ns), 2),
            "p95_ms": round(float(p95_ns), 2),
            "p99_ms": round(float(p99_ns), 2),
            "mean_ms": round(float(np.mean(lat_no_shap)), 2)
        }
    }


def get_test_count():
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q"]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = res.stdout + res.stderr
    # Look for "X tests collected"
    import re
    match = re.search(r"(\d+)\s+tests?\s+collected", out)
    if match:
        return int(match.group(1))
    lines = [line.strip() for line in out.splitlines() if line.strip() and not line.startswith("=")]
    return len(lines)


def generate_numbers():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs/cost_config.yaml")))
    test = pd.read_csv(os.path.join(ROOT, "data/processed/test.csv"), dtype={'pincode': str})
    val_cal = pd.read_csv(os.path.join(ROOT, "data/processed/val_cal.csv"), dtype={'pincode': str})

    drop_cols = ['rto_label', 'timestamp', 'order_id']
    X_test = test.drop(columns=drop_cols, errors='ignore')
    y_test = test['rto_label'].values

    X_val_cal = val_cal.drop(columns=drop_cols, errors='ignore')
    y_val_cal = val_cal['rto_label'].values

    # Load frozen models
    lr = joblib.load(os.path.join(ROOT, "models/logistic_baseline.pkl"))
    lgbm_uncal = joblib.load(os.path.join(ROOT, "models/tree_model.pkl"))
    lgbm_iso = joblib.load(os.path.join(ROOT, "models/tree_model_calibrated.pkl"))

    # Standard Platt fitting on logits via LogisticRegressionCV (Task 0.3)
    eps = 1e-12
    p_val_cal_uncal = np.clip(lgbm_uncal.predict_proba(X_val_cal)[:, 1], eps, 1.0 - eps)
    logits_val = np.log(p_val_cal_uncal / (1.0 - p_val_cal_uncal)).reshape(-1, 1)

    platt = LogisticRegressionCV(Cs=10, cv=5, scoring="neg_log_loss", random_state=42)
    platt.fit(logits_val, y_val_cal)

    p_test_uncal = np.clip(lgbm_uncal.predict_proba(X_test)[:, 1], eps, 1.0 - eps)
    logits_test = np.log(p_test_uncal / (1.0 - p_test_uncal)).reshape(-1, 1)

    # Isotonic acceptance gate on held-out validation data (Task 0.3)
    p_val_iso = lgbm_iso.predict_proba(X_val_cal)[:, 1]
    brier_val_uncal = brier_score_loss(y_val_cal, p_val_cal_uncal)
    brier_val_iso = brier_score_loss(y_val_cal, p_val_iso)
    ece_val_uncal = compute_ece(y_val_cal, p_val_cal_uncal, n_bins=10)
    ece_val_iso = compute_ece(y_val_cal, p_val_iso, n_bins=10)

    top_decile_thresh = np.percentile(p_val_iso, 90)
    tail_mask = p_val_iso >= top_decile_thresh
    tail_pred_mean = float(p_val_iso[tail_mask].mean()) if np.any(tail_mask) else 1.0
    tail_emp_mean = float(y_val_cal[tail_mask].mean()) if np.any(tail_mask) else 1.0
    tail_bias = abs(tail_pred_mean - tail_emp_mean)

    isotonic_accepted = bool((ece_val_iso < ece_val_uncal) and (brier_val_iso <= brier_val_uncal) and (tail_bias < 0.25))
    calibration_gate_report = {
        "isotonic_accepted": isotonic_accepted,
        "ece_val_uncal": round(float(ece_val_uncal), 6),
        "ece_val_iso": round(float(ece_val_iso), 6),
        "brier_val_uncal": round(float(brier_val_uncal), 6),
        "brier_val_iso": round(float(brier_val_iso), 6),
        "tail_pred_mean": round(tail_pred_mean, 4),
        "tail_emp_mean": round(tail_emp_mean, 4),
        "tail_bias": round(tail_bias, 4),
    }

    # Test predictions
    preds = {
        "logistic_regression": lr.predict_proba(X_test)[:, 1],
        "lgbm_uncalibrated": lgbm_uncal.predict_proba(X_test)[:, 1],
        "lgbm_isotonic": lgbm_iso.predict_proba(X_test)[:, 1],
        "lgbm_platt": platt.predict_proba(logits_test)[:, 1],
    }

    model_metrics = {}
    for name, p in preds.items():
        p_safe = np.clip(p, 1e-15, 1.0 - 1e-15)
        model_metrics[name] = {
            "pr_auc": round(float(average_precision_score(y_test, p)), 6),
            "roc_auc": round(float(roc_auc_score(y_test, p)), 6),
            "brier": round(float(brier_score_loss(y_test, p)), 6),
            "log_loss": round(float(log_loss(y_test, p_safe)), 6),
            "ece": round(compute_ece(y_test, p, n_bins=10), 6),
        }

    # Primary model selection per config / D3 rule (Task 0.2)
    primary_model_name = cfg.get("primary_model", "logistic_regression")

    # COD Subset evaluation
    cod_mask = (test['payment_method'] == 'COD').values
    test_cod = test[cod_mask].reset_index(drop=True)
    y_cod = test_cod['rto_label'].values
    V_cod = test_cod['order_value'].values
    p_cod = preds[primary_model_name][cod_mask]

    cod_base_rto = float(y_cod.mean())
    baseline_cod_rtos = int(y_cod.sum())
    n_cod = len(test_cod)

    engine = CostEngine(os.path.join(ROOT, "configs/cost_config.yaml"))
    actions, policy_losses = engine.get_optimal_policy(test_cod.assign(payment_method="COD"), p_cod)
    actions = np.asarray(actions)

    # Baseline expected loss: ALLOW_COD on everyone
    baseline_losses = p_cod * 150.0 - (1.0 - p_cod) * (0.20 * V_cod)
    baseline_expected_total = float(np.sum(baseline_losses))
    policy_expected_total = float(np.sum(policy_losses))
    expected_savings = round(baseline_expected_total - policy_expected_total, 2)

    # Realized savings
    realized_savings = round(canonical_realized_pl(actions, y_cod, V_cod, cfg), 2)

    # Operational metrics
    orders_touched, rtos_prev_exp, drops_exp, friction_spend = calc_operational_metrics(
        test_cod, p_cod, actions, engine
    )

    # Action mix
    unique_actions, counts = np.unique(actions, return_counts=True)
    action_counts = dict(zip(unique_actions, [int(c) for c in counts]))
    for a in ["ALLOW_COD", "VERIFY_ADDRESS", "REQUIRE_DEPOSIT", "PREPAID_ONLY"]:
        if a not in action_counts:
            action_counts[a] = 0

    action_mix_pct = {
        k: round(v / n_cod * 100.0, 2) for k, v in action_counts.items()
    }

    # Effective post policy RTO
    effective_post_policy_rto = round(
        (baseline_cod_rtos - rtos_prev_exp) / (n_cod - rtos_prev_exp - drops_exp) * 100.0, 2
    )

    # Uplift %
    uplift_pct = round((expected_savings / abs(baseline_expected_total)) * 100.0, 2)

    # Binary verify policy (threshold 0.20 tuned on val split)
    bin_actions = np.where(p_cod > 0.20, "VERIFY_ADDRESS", "ALLOW_COD")
    bin_losses = []
    for i in range(n_cod):
        act = bin_actions[i]
        el = engine.evaluate_interventions(V_cod[i], p_cod[i])
        bin_losses.append(el[act])
    binary_verify_expected_savings = round(baseline_expected_total - float(np.sum(bin_losses)), 2)
    ratio_2_0x = round(expected_savings / binary_verify_expected_savings, 4)

    # Latency benchmark
    print("Benchmarking latency...")
    latency = benchmark_latency(n_iter=500, warmup=50)

    # Test count
    print("Collecting test count...")
    test_count = get_test_count()

    # 5-seed sweep statistics for the shipped model (Task 3a)
    sweep_csv_path = os.path.join(ROOT, "audit/seed_sweep.csv")

    if os.path.exists(sweep_csv_path):
        sweep_df = pd.read_csv(sweep_csv_path)
        if primary_model_name == "logistic_regression":
            m_realized = sweep_df["LR_realized_Rs"].values
            m_expected = sweep_df["LR_expected_Rs"].values
            m_gap = sweep_df["LR_gap_pct"].values
        else:
            m_realized = sweep_df["LGBM_iso_realized_Rs"].values
            m_expected = sweep_df["LGBM_iso_expected_Rs"].values
            m_gap = sweep_df["LGBM_iso_gap_pct"].values
        five_seed_stats = {
            "description": "5-seed sweep over seeds [42, 101, 2024, 777, 999]",
            "expected_savings_mean_inr": round(float(np.mean(m_expected)), 2),
            "expected_savings_std_inr": round(float(np.std(m_expected, ddof=1)), 2),
            "realized_savings_mean_inr": round(float(np.mean(m_realized)), 2),
            "realized_savings_std_inr": round(float(np.std(m_realized, ddof=1)), 2),
            "gap_pct_mean": round(float(np.mean(m_gap)), 2),
            "gap_pct_std": round(float(np.std(m_gap, ddof=1)), 2),
        }
    else:
        five_seed_stats = {}

    # Config hash and git commit
    import hashlib
    with open(os.path.join(ROOT, "configs/cost_config.yaml"), "rb") as f:
        cfg_sha256 = hashlib.sha256(f.read()).hexdigest()
    try:
        git_hash = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        git_hash = "UNKNOWN"

    oracle_ceiling_inr = round(float(113.0 * y_cod.sum()), 2)
    pct_oracle_ceiling = round(realized_savings / oracle_ceiling_inr * 100.0, 2)
    savings_per_1k = round(realized_savings / n_cod * 1000.0, 2)
    expected_savings_per_1k = round(expected_savings / n_cod * 1000.0, 2)

    primary_note = (
        f"Seed 42 is a single draw; 5-seed mean realized savings is Rs "
        f"{five_seed_stats.get('realized_savings_mean_inr', realized_savings):,.2f} +/- "
        f"{five_seed_stats.get('realized_savings_std_inr', 0.0):,.2f}"
    )

    results = {
        "config_sha256": cfg_sha256,
        "git_commit_hash": git_hash,
        "cod_subset_size": n_cod,
        "cod_base_rto_rate": round(cod_base_rto, 4),
        "cod_base_rto_pct": round(cod_base_rto * 100.0, 2),
        "baseline_cod_rtos": baseline_cod_rtos,
        "models": model_metrics,
        "calibration_gate": calibration_gate_report,
        "primary_model": primary_model_name,
        "primary_model_note": primary_note,

        "single_draw_label": "seed_42_single_draw",
        "expected_savings_inr": expected_savings,
        "expected_savings_per_1k_cod_inr": expected_savings_per_1k,
        "realized_savings_inr": realized_savings,
        "savings_per_1k_cod_inr": savings_per_1k,
        "oracle_ceiling_inr": oracle_ceiling_inr,
        "pct_of_oracle_ceiling": pct_oracle_ceiling,
        "seed_42_single_draw_realized_savings_inr": realized_savings,
        "five_seed_sweep_shipped_model": five_seed_stats,
        "expected_rtos_prevented": round(float(rtos_prev_exp), 2),
        "rtos_prevented_pct": round(float(rtos_prev_exp) / baseline_cod_rtos * 100.0, 2),
        "good_customer_drops": round(float(drops_exp), 2),
        "friction_spend_inr": round(float(friction_spend), 2),
        "action_mix_counts": action_counts,
        "action_mix_pct": action_mix_pct,
        "effective_post_policy_rto_pct": effective_post_policy_rto,
        "uplift_pct": uplift_pct,
        "binary_verify_expected_savings_inr": round(binary_verify_expected_savings, 2),
        "ratio_vs_single_threshold": ratio_2_0x,
        "latency": latency,
        "test_count": test_count,
    }


    out_path = os.path.join(ROOT, "reports/headline_numbers.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Successfully generated headline numbers at: {out_path}")
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    generate_numbers()
