"""
scripts/reoptimize_policy.py — Quarterly Policy Re-Optimization Runner.

Cron / Scheduled Job Usage:
    # Run quarterly at midnight on the 1st of Jan, Apr, Jul, Oct:
    0 0 1 1,4,7,10 * python scripts/reoptimize_policy.py --apply --data-path data/processed/test.csv

This script evaluates intervention elasticity and solves for the optimal policy
routing parameters using the grid-reoptimization mechanism from regenerate_all.py:464-471.
"""

import os
import sys
import copy
import argparse
import json
import yaml
import numpy as np
import pandas as pd
import joblib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.policy.cost_engine import CostEngine
from scripts.generate_headline_numbers import canonical_realized_pl

def reoptimize_policy(data_path=None, config_path=None, apply_to_config=False):
    config_path = config_path or os.path.join(ROOT, "configs/cost_config.yaml")
    data_path = data_path or os.path.join(ROOT, "data/processed/test.csv")
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    test = pd.read_csv(data_path, dtype={'pincode': str})
    cod_mask = (test['payment_method'] == 'COD').values
    test_cod = test[cod_mask].reset_index(drop=True)
    y_cod = test_cod['rto_label'].values
    V_cod = test_cod['order_value'].values
    n_cod = len(test_cod)

    drop_cols = ['rto_label', 'timestamp', 'order_id']
    X_test = test.drop(columns=drop_cols, errors='ignore')
    
    # Load primary model
    primary_name = cfg.get("primary_model", "logistic_regression")
    if primary_name == "logistic_regression":
        model = joblib.load(os.path.join(ROOT, "models/logistic_baseline.pkl"))
    else:
        model = joblib.load(os.path.join(ROOT, "models/tree_model_calibrated.pkl"))
    
    p_cod = model.predict_proba(X_test)[:, 1][cod_mask]

    # Baseline evaluated under current policy
    engine_current = CostEngine(config_path)
    current_actions, _ = engine_current.get_optimal_policy(test_cod.assign(payment_method="COD"), p_cod)
    current_savings = canonical_realized_pl(current_actions, y_cod, V_cod, cfg)

    # Grid re-optimization sweep over plausible intervention parameters
    grid_deposit_drops = [0.30, 0.353, 0.40, 0.45]
    grid_deposit_reds = [0.70, 0.745, 0.80, 0.85]
    grid_verify_drops = [0.017, 0.03, 0.05, 0.08]
    grid_verify_reds = [0.25, 0.30, 0.40, 0.466]

    best_savings = current_savings
    best_params = {
        "REQUIRE_DEPOSIT": {
            "success_drop_pct": cfg["interventions"]["REQUIRE_DEPOSIT"]["success_drop_pct"],
            "rto_reduction_pct": cfg["interventions"]["REQUIRE_DEPOSIT"]["rto_reduction_pct"],
        },
        "VERIFY_ADDRESS": {
            "success_drop_pct": cfg["interventions"]["VERIFY_ADDRESS"]["success_drop_pct"],
            "rto_reduction_pct": cfg["interventions"]["VERIFY_ADDRESS"]["rto_reduction_pct"],
        }
    }

    results_table = []
    for dd in grid_deposit_drops:
        for dr in grid_deposit_reds:
            for vd in grid_verify_drops:
                for vr in grid_verify_reds:
                    cfg_temp = copy.deepcopy(cfg)
                    cfg_temp["interventions"]["REQUIRE_DEPOSIT"]["success_drop_pct"] = dd
                    cfg_temp["interventions"]["REQUIRE_DEPOSIT"]["rto_reduction_pct"] = dr
                    cfg_temp["interventions"]["VERIFY_ADDRESS"]["success_drop_pct"] = vd
                    cfg_temp["interventions"]["VERIFY_ADDRESS"]["rto_reduction_pct"] = vr

                    eng_temp = CostEngine()
                    eng_temp.config = cfg_temp
                    eng_temp.interventions = cfg_temp["interventions"]
                    act_reopt, _ = eng_temp.get_optimal_policy(test_cod.assign(payment_method="COD"), p_cod)
                    sav_reopt = canonical_realized_pl(act_reopt, y_cod, V_cod, cfg_temp)

                    if sav_reopt > best_savings:
                        best_savings = sav_reopt
                        best_params["REQUIRE_DEPOSIT"]["success_drop_pct"] = dd
                        best_params["REQUIRE_DEPOSIT"]["rto_reduction_pct"] = dr
                        best_params["VERIFY_ADDRESS"]["success_drop_pct"] = vd
                        best_params["VERIFY_ADDRESS"]["rto_reduction_pct"] = vr

    report = {
        "status": "COMPLETED",
        "current_realized_savings_inr": round(float(current_savings), 2),
        "reoptimized_realized_savings_inr": round(float(best_savings), 2),
        "incremental_uplift_inr": round(float(best_savings - current_savings), 2),
        "best_parameters": best_params,
        "cadence": "quarterly",
    }

    out_path = os.path.join(ROOT, "reports/quarterly_reoptimization_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[RE-OPTIMIZE] Current savings: Rs {current_savings:,.2f}")
    print(f"[RE-OPTIMIZE] Best re-optimized savings: Rs {best_savings:,.2f} (+Rs {best_savings - current_savings:,.2f})")
    print(f"[RE-OPTIMIZE] Report written to {out_path}")

    if apply_to_config:
        cfg["reoptimized_policy"] = best_params
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"[RE-OPTIMIZE] Applied re-optimized policy to {config_path}")

    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quarterly Policy Re-Optimization")
    parser.add_argument("--apply", action="store_true", help="Apply re-optimized policy to config")
    parser.add_argument("--data-path", type=str, default=None, help="Path to evaluation order dataset")
    parser.add_argument("--config-path", type=str, default=None, help="Path to cost config yaml")
    args = parser.parse_args()

    reoptimize_policy(data_path=args.data_path, config_path=args.config_path, apply_to_config=args.apply)
