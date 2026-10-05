"""
test_sensitivity.py — Executable claims for the deposit-effectiveness
sensitivity analysis (src/eval/deposit_effectiveness_sensitivity.py).

The frozen headline ("2.0x savings") is linear in the unvalidated DEPOSIT
constant rto_reduction_pct=0.80. These tests pin the honest disclosure:
the anchor reproduces the frozen numbers, and the assumption-bound range
is what the README now quotes alongside the point estimate.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

SUMMARY_PATH = "reports/deposit_effectiveness_summary.json"
REPORT_PATH = "reports/deposit_effectiveness_sensitivity.md"
PLOT_PATH = "reports/stage4/deposit_effectiveness_heatmap.png"


@pytest.fixture(scope="module")
def summary():
    if not os.path.exists(SUMMARY_PATH):
        pytest.skip("run src/eval/deposit_effectiveness_sensitivity.py to generate artifacts")
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_artifacts_exist():
    assert os.path.exists(REPORT_PATH), "sensitivity report missing"
    assert os.path.exists(PLOT_PATH), "heatmap missing"


def test_anchor_reproduces_frozen_claim(summary):
    # At the claimed constants the script must reproduce the frozen Stage-5
    # numbers exactly: primary 71741, best-binary 35919, ratio ~1.997.
    assert summary["anchor_primary_savings"] == pytest.approx(71741, abs=2)
    assert summary["best_binary_savings"] == pytest.approx(35919, abs=2)
    assert summary["anchor_ratio"] == pytest.approx(1.997, abs=0.01)


def test_assumption_bound_range_is_disclosed(summary):
    lo, hi = summary["ratio_range_at_claimed_d"]
    # The ratio collapses toward ~1.2x at low deposit effectiveness and
    # reaches ~2.4x at high effectiveness. The README quotes this range.
    assert 1.15 <= lo <= 1.35, f"lower bound {lo} outside expected band"
    assert 2.25 <= hi <= 2.55, f"upper bound {hi} outside expected band"
    assert summary["ratio_at_r_0.50"] == pytest.approx(1.28, abs=0.05)


def test_fee_correction_reduces_savings(summary):
    # Including COD-fee revenue on delivered orders must REDUCE savings
    # (interventions become relatively less attractive).
    assert summary["fee_corrected_primary_savings"] < 0.65 * summary["anchor_primary_savings"]


def test_report_states_the_caveat():
    with open(REPORT_PATH, encoding="utf-8") as f:
        report = f.read()
    assert "unvalidated" in report.lower()
    assert "Menu degeneration" in report or "degeneration" in report
    assert "linear" in report.lower()


def test_frozen_policy_deposit_reduction_misspecification():
    # Verify policy behavior under deposit RTO reduction misspecification
    import pandas as pd
    import numpy as np
    import joblib
    import yaml
    from src.policy.cost_engine import CostEngine
    from src.eval.stage4_evaluate import get_cod_subset, eval_multi_action

    cfg = yaml.safe_load(open("configs/cost_config.yaml"))
    test = pd.read_csv("data/processed/test.csv", dtype={'pincode': str})
    model = joblib.load("models/tree_model_calibrated.pkl")
    proba = model.predict_proba(test.drop(columns=['rto_label', 'timestamp', 'order_id'], errors='ignore'))[:, 1]
    
    test_cod, proba_cod = get_cod_subset(test, proba)
    y_cod = test_cod['rto_label'].values
    V_cod = test_cod['order_value'].values
    margin = 0.20 * V_cod
    rto_cost = 150.0

    engine = CostEngine()
    frozen_actions, _ = eval_multi_action(test_cod, proba_cod, engine)

    # At 50% deposit RTO reduction (assumed 80%), policy must remain profitable
    tot_base = np.sum(np.where(y_cod == 1, rto_cost, -margin))
    
    # Eval under 50% true reduction
    losses_50 = []
    for i, a in enumerate(frozen_actions):
        yc = y_cod[i]
        m = margin[i]
        if a == "ALLOW_COD":
            losses_50.append(rto_cost if yc == 1 else -m)
        elif a == "VERIFY_ADDRESS":
            losses_50.append(2.0 + (0.70 * rto_cost if yc == 1 else -0.95 * m))
        elif a == "REQUIRE_DEPOSIT":
            losses_50.append((1.0 - 0.50) * rto_cost if yc == 1 else -0.60 * m)
        elif a == "PREPAID_ONLY":
            losses_50.append(0.45 * rto_cost if yc == 1 else -0.30 * m)
    
    savings_50 = tot_base - np.sum(losses_50)
    assert savings_50 > 30000.0, f"Expected > Rs 30,000 savings at 50% deposit reduction, got {savings_50}"


def test_frozen_policy_joint_pessimistic_misspecification():
    # Proves that joint deterioration is detected and causes loss (bounding fragility)
    import pandas as pd
    import numpy as np
    import joblib
    import yaml
    from src.policy.cost_engine import CostEngine
    from src.eval.stage4_evaluate import get_cod_subset, eval_multi_action

    test = pd.read_csv("data/processed/test.csv", dtype={'pincode': str})
    model = joblib.load("models/tree_model_calibrated.pkl")
    proba = model.predict_proba(test.drop(columns=['rto_label', 'timestamp', 'order_id'], errors='ignore'))[:, 1]
    
    test_cod, proba_cod = get_cod_subset(test, proba)
    y_cod = test_cod['rto_label'].values
    V_cod = test_cod['order_value'].values
    margin = 0.20 * V_cod
    rto_cost = 150.0

    engine = CostEngine()
    frozen_actions, _ = eval_multi_action(test_cod, proba_cod, engine)
    tot_base = np.sum(np.where(y_cod == 1, rto_cost, -margin))

    # Joint pessimistic: deposit red 60% / drop 50%, verify red 15% / drop 10%
    losses_pess = []
    for i, a in enumerate(frozen_actions):
        yc = y_cod[i]
        m = margin[i]
        if a == "ALLOW_COD":
            losses_pess.append(rto_cost if yc == 1 else -m)
        elif a == "VERIFY_ADDRESS":
            losses_pess.append(2.0 + (0.85 * rto_cost if yc == 1 else -0.90 * m))
        elif a == "REQUIRE_DEPOSIT":
            losses_pess.append((1.0 - 0.60) * rto_cost if yc == 1 else -0.50 * m)
        elif a == "PREPAID_ONLY":
            losses_pess.append(0.45 * rto_cost if yc == 1 else -0.30 * m)

    savings_pess = tot_base - np.sum(losses_pess)
    assert savings_pess < 0.0, f"Joint pessimistic scenario must produce loss to reveal fragility: {savings_pess}"


def test_frozen_policy_single_parameter_misspecifications():
    # Evaluates the frozen policy against single-parameter misspecification sweeps:
    # (a) Deposit RTO reduction 80% -> 60/40/20%
    # (b) Verify RTO reduction 30% -> 15/0%
    # (c) Verify drop-off 5% -> 10/20%
    import pandas as pd
    import numpy as np
    import joblib
    from src.policy.cost_engine import CostEngine
    from src.eval.stage4_evaluate import get_cod_subset, eval_multi_action

    test = pd.read_csv("data/processed/test.csv", dtype={'pincode': str})
    model = joblib.load("models/tree_model_calibrated.pkl")
    proba = model.predict_proba(test.drop(columns=['rto_label', 'timestamp', 'order_id'], errors='ignore'))[:, 1]
    
    test_cod, proba_cod = get_cod_subset(test, proba)
    y_cod = test_cod['rto_label'].values
    V_cod = test_cod['order_value'].values
    margin = 0.20 * V_cod
    rto_cost = 150.0

    engine = CostEngine()
    frozen_actions, _ = eval_multi_action(test_cod, proba_cod, engine)
    tot_base = np.sum(np.where(y_cod == 1, rto_cost, -margin))

    def eval_scenario(dep_red=0.80, dep_drop=0.40, ver_red=0.30, ver_drop=0.05, ver_cost=2.0):
        losses = []
        for i, a in enumerate(frozen_actions):
            yc = y_cod[i]
            m = margin[i]
            if a == "ALLOW_COD":
                losses.append(rto_cost if yc == 1 else -m)
            elif a == "VERIFY_ADDRESS":
                losses.append(ver_cost + ((1.0 - ver_red) * rto_cost if yc == 1 else -(1.0 - ver_drop) * m))
            elif a == "REQUIRE_DEPOSIT":
                losses.append((1.0 - dep_red) * rto_cost if yc == 1 else -(1.0 - dep_drop) * m)
            elif a == "PREPAID_ONLY":
                losses.append(0.45 * rto_cost if yc == 1 else -0.30 * m)
        return tot_base - np.sum(losses)

    # (a) Deposit RTO reduction: 60% and 40% still beat baseline; 20% drops below baseline
    s_dep_60 = eval_scenario(dep_red=0.60)
    s_dep_40 = eval_scenario(dep_red=0.40)
    s_dep_20 = eval_scenario(dep_red=0.20)
    assert s_dep_60 > 0, f"Deposit red 60% should remain profitable: {s_dep_60}"
    assert s_dep_40 > 0, f"Deposit red 40% should remain profitable: {s_dep_40}"
    assert s_dep_20 < 0, f"Deposit red 20% should drop below baseline: {s_dep_20}"

    # (b) Verify RTO reduction: 15% and 0% both remain profitable (deposit carries savings)
    s_ver_15 = eval_scenario(ver_red=0.15)
    s_ver_00 = eval_scenario(ver_red=0.00)
    assert s_ver_15 > 0, f"Verify red 15% should remain profitable: {s_ver_15}"
    assert s_ver_00 > 0, f"Verify red 0% should remain profitable: {s_ver_00}"

    # (c) Verify drop-off: 10% and 20% both remain profitable
    s_vdrop_10 = eval_scenario(ver_drop=0.10)
    s_vdrop_20 = eval_scenario(ver_drop=0.20)
    assert s_vdrop_10 > 0, f"Verify drop 10% should remain profitable: {s_vdrop_10}"
    assert s_vdrop_20 > 0, f"Verify drop 20% should remain profitable: {s_vdrop_20}"


